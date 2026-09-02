---
layout: post
title: "推测解码：从拒绝采样到工程实践"
subtitle: "用并行验证减少大模型的串行解码轮次"
date: 2026-05-12
last_modified_at: 2026-09-03
author: iStar
catalog: true
series: speculative-decoding
series_order: 10
technology_year: 2022
mathjax: true
tags: [推测解码, LLM推理]
---

自回归模型生成第 $t+1$ 个 token 之前，必须先知道第 $t$ 个 token。生成 100 个 token，目标模型通常要串行执行约 100 轮；每轮都读取大量权重，却只为很少的新位置计算，尤其在低并发 decode 中容易受显存带宽和 kernel 启动开销限制。

推测解码（Speculative Decoding）并没有打破自回归依赖，而是把工作分成“低成本串行提出候选”和“高成本并行验证候选”：一个更快的 proposer 先猜若干 token，目标模型再用一次 forward 同时评估这些位置。若前几个候选被接受，一轮目标模型调用就能正式推进多个 token。

常见比喻是助理先写草稿、专家批量审稿，但还要补上一点：随机采样下，专家不能只把错误词改掉；必须使用修正分布，才能保证最终样本仍来自目标模型。本文从这一步开始推导。

术语上，这套机制通常称为 speculative sampling；Chen 等人称其为 modified rejection sampling。它虽然包含接受/拒绝测试，却不同于带包络常数 $M$、拒绝后反复从 proposal 重试的经典 rejection sampling：候选一旦被拒绝，这里会直接从归一化的正残差 $\max(0,p-q)$ 采样。因此，不能把经典拒绝采样的接受率与重试步骤直接套到这里。

## 为什么目标模型能一次验证多个位置

标准 decode 在上下文 $x_{<t}$ 上计算目标分布：

$$
p(x_t\mid x_{<t})
$$

采样出 $x_t$ 后，下一轮才能得到 $p(x_{t+1}\mid x_{\le t})$。串行性来自“下一位置的上下文必须包含刚才生成的 token”。

现在让草稿模型 $q$ 先按自回归方式生成 $k$ 个候选：

$$
\tilde{x}_1,\tilde{x}_2,\ldots,\tilde{x}_k
$$

这些候选已经组成一条完整假设路径。将整条路径接到原上下文后输入目标模型，causal attention 会在一次 forward 中同时算出：

```text
p(· | context)
p(· | context, x̃1)
p(· | context, x̃1, x̃2)
...
p(· | context, x̃1, ..., x̃k)
```

矩阵计算可以并行处理这些位置。目标模型并不是“相信”候选，而是暂时把它们当作已知输入来评分；一旦某个位置被拒绝，其后的分布都建立在错误前缀上，必须丢弃。

## 一轮推测解码

设目标模型为 $p$，草稿模型为 $q$，一轮最多提出 $k$ 个 token。本文沿用 Leviathan 等人的记号；Chen 等人的论文恰好用 $q$ 表示目标、$p$ 表示草稿，阅读公式时需要交换两者。

下面的 $p_i$、$q_i$ 都指真正参与采样的归一化分布：$p_i$ 必须与不开推测解码时的目标采样分布一致，$q_i$ 必须与生成候选时的 proposal 分布一致，并定义在同一个离散 token 样本空间。经典公式通常要求目标与草稿使用兼容的 tokenizer 和词表；如果 tokenizer 不同，就必须另行定义共同样本空间或对齐过程，并重新证明分布无损，不能只把 token ID 做普通映射。若系统使用 temperature、top-k/top-p、grammar 或其他 logits processor，证明中使用的是相应处理后的分布，而不是未经处理的原始 logits；有状态的 processor 必须处于与对应条件前缀一致的状态，其暂态更新也只能随接受前缀正式提交。

### 1. 草稿阶段

$q$ 串行采样候选，并保留每一步的草稿分布 $q_i$，或保留足以在验证时无损恢复它的信息。接受测试只查询候选点的概率，但首次拒绝后的残差修正需要全词表上的 $q_i$：

$$
\tilde{x}_i\sim q_i,\qquad q_i=q(\cdot\mid x_{<t},\tilde{x}_{<i})
$$

### 2. 验证阶段

目标模型并行计算对应条件下的分布：

$$
p_i=p(\cdot\mid x_{<t},\tilde{x}_{<i})
$$

从算法视角看，一轮验证取得 $p_1,\ldots,p_{k+1}$。在复用 KV Cache 的实现中，$p_1$ 可能是上一轮已经保存的边界 logits，本轮 forward 主要为草稿输入产生 $p_2,\ldots,p_{k+1}$；这只是计算来源不同，不改变各分布的条件前缀。$p_{k+1}$ 可在候选全部接受后用于采样下一个位置。

### 3. 逐位置接受

对候选 $\tilde{x}_i$，生成一个与草稿采样以及其他接受测试相互独立的均匀随机数 $u_i\sim U(0,1)$，当

$$
u_i<\min\left(1,
\frac{p_i(\tilde{x}_i)}{q_i(\tilde{x}_i)}
\right)
$$

时接受。若被接受，就继续检查下一位置。

### 4. 首次拒绝后的修正

若第 $i$ 个候选首次被拒绝，使用新的独立随机数从正残差分布采样一个修正 token：

$$
p'_i(x)=
\frac{\max(0,p_i(x)-q_i(x))}
{\sum_y\max(0,p_i(y)-q_i(y))}
$$

本轮提交前 $i-1$ 个已接受候选和这个修正 token，候选 $i+1\ldots k$ 全部作废。

### 5. 全部接受

若 $k$ 个候选全部通过，并且尚未触发 EOS、stop 条件或最大长度，再从目标模型已经算出的 $p_{k+1}$ 采样一个额外 token。若已满足停止条件，就按正式输出边界结束，不再追加 token。于是一次目标验证最多可以推进 $k+1$ 个 token。

## 用一个三词词表手算接受过程

假设当前位置只有 `{A, B, C}` 三个 token：

| token | 草稿分布 $q$ | 目标分布 $p$ |
| --- | ---: | ---: |
| A | 0.50 | 0.40 |
| B | 0.30 | 0.50 |
| C | 0.20 | 0.10 |

如果草稿采到了 A，接受概率为：

$$
\min(1,0.40/0.50)=0.8
$$

如果草稿采到 B，$p(B)>q(B)$，接受概率为 1。采到 C 的接受概率是 $0.10/0.20=0.5$。

拒绝发生时，正残差为：

```text
max(0, p - q)
A: max(0, 0.40 - 0.50) = 0
B: max(0, 0.50 - 0.30) = 0.20
C: max(0, 0.10 - 0.20) = 0
```

归一化后只能选择 B。从联合概率看，A 被拒绝的概率质量是 $0.50\times(1-0.8)=0.10$，C 被拒绝的概率质量是 $0.20\times(1-0.5)=0.10$；两者合计 0.20，拒绝后都修正为 B，恰好补上草稿对 B 少分配的 0.20。注意这条路径不是“草稿提出 B、拒绝 B、再采到 B”：草稿提出 B 时接受概率为 1。

## 为什么最终分布仍等于目标模型

固定一个待验证位置、此前已提交的前缀及其采样状态，令 $Y\sim q$ 表示该位置的草稿 token，$A$ 表示它被接受，$X$ 表示该位置最终提交的 token。对于 $q(x)>0$ 的候选，接受概率是 $\min(1,p(x)/q(x))$；$q(x)=0$ 的 token 不会被草稿采到，无需计算这个比值。因此，对任意 token $x$，接受路径贡献的概率质量为：

$$
P(A,Y=x)=
\begin{cases}
q(x)\min\left(1,\dfrac{p(x)}{q(x)}\right), & q(x)>0,\\
0, & q(x)=0
\end{cases}
=\min(q(x),p(x))
$$

这一步的总接受概率于是为：

$$
\beta=P(A)=\sum_y\min(q(y),p(y))
$$

再记正残差的归一化常数为：

$$
Z=\sum_y\max(0,p(y)-q(y))
$$

由于 $p$、$q$ 都是概率分布，$\sum_y(p(y)-q(y))=0$，正负两侧的差值总量相等。因此：

$$
Z
=\sum_y\max(0,q(y)-p(y))
=1-\sum_y\min(q(y),p(y))
=1-\beta
$$

这正是“当前草稿 token 被拒绝”的总概率。当 $Z>0$ 时，修正路径要先对这个事件的所有可能草稿取值求和，再从 $p'$ 采样。它对任意最终输出 token $x$ 贡献的概率质量为：

$$
\begin{aligned}
P(A^c, X=x)
&=P(A^c)P(X=x\mid A^c)\\
&=(1-\beta)\frac{\max(0,p(x)-q(x))}{Z}\\
&=\max(0,p(x)-q(x))
\end{aligned}
$$

所以：

$$
\begin{aligned}
P(X=x)
&=P(A,Y=x)+P(A^c,X=x)\\
&=\min(q(x),p(x))+\max(0,p(x)-q(x))\\
&=p(x)
\end{aligned}
$$

这里不能把修正路径理解成“草稿提出 $x$、拒绝 $x$，然后残差分布又采到 $x$”。当 $q(x)>0$ 且 $p(x)>q(x)$ 时，草稿提出 $x$ 会以概率 1 被接受；在 $Z>0$ 的拒绝分支中，当 $p(x)\le q(x)$ 时，$p'(x)=0$。因此，若修正采样最终得到 $x$，触发拒绝的一定是其他 token $y\ne x$；上式中的 $P(A^c)$ 已经把所有这样的 $y$ 汇总起来。若 $Z=0$，则 $p=q$、拒绝概率为 0，修正分布根本不会被调用。

这个等式是在任意已提交前缀及其正式采样状态所确定的条件分布上成立的。记最终生成序列为 $X_{1:n}$，对任意固定序列 $x_{1:n}$，概率的链式分解给出：

$$
\begin{aligned}
P(X_{1:n}=x_{1:n})
&=\prod_{t=1}^{n}P(X_t=x_t\mid X_{<t}=x_{<t})\\
&=\prod_{t=1}^{n}p(x_t\mid x_{<t})
\end{aligned}
$$

若某轮走修正分支，系统会基于修正后已提交的前缀重新起草；若候选被接受，后续候选原本就是在这个已接受前缀下从对应 $q_i$ 采出的；若全部候选通过，额外 token 直接从 $p_{k+1}$ 采样。所以上面的单步结论可以沿位置连续应用，保证的是整条序列的联合分布，而不只是某个位置的边缘分布。

这也解释了两个看似合理但错误的简化：

- 拒绝后直接从完整 $p$ 采样，会重复分配接受路径已经覆盖的概率质量；
- 只比较草稿与目标的 argmax，不能保持有温度采样时的目标分布。

论文中的“无损”指分布在数学上保持一致。有限精度、batch shape 与非确定性 kernel 可能造成数值差异，因此工程上不保证相同随机种子必然逐 token 复现同一条采样序列。

## Greedy 模式更简单

当 temperature 为 0 时，基线目标模型每步选择：

$$
x_i^*=\arg\max_x p_i(x)
$$

验证可以直接比较候选是否等于目标 argmax：

```text
draft:  A  B  C  D
target: A  B  X  ?
        ✓  ✓  ✗
```

提交 A、B 和目标 token X，D 作废。若全部候选相同，再提交目标模型额外算出的下一个 argmax。正确实现应与不开推测解码的 greedy 序列一致，数值 tie 和底层非确定性需要单独处理。

## 加速来自减少串行轮数

推测解码没有减少目标模型参数，也没有跳过目标验证。它利用一个实际硬件现象：在低 batch 下，目标模型一次验证几个连续位置的耗时，可能没有比验证一个位置高出同样倍数。

设：

- 一轮草稿耗时为 $T_d(k)$；
- 目标批量验证耗时为 $T_v(k)$；
- 本轮正式推进 token 数的期望为 $E[L]$；
- 缓存提交、采样等开销为 $T_o$。

平均每个输出 token 的时间近似：

$$
\operatorname{TPOT}_{spec}
\approx\frac{T_d(k)+T_v(k)+T_o}{E[L]}
$$

基线 TPOT 为 $T_p(1)$。只有上式明显更小，推测解码才有实际收益。

`acceptance rate` 不能单独代表加速。假设接受率很高，但草稿模型几乎和目标模型一样慢，总耗时仍可能增加；反之，一个几乎零成本的 n-gram proposer 接受率不算高，也可能在重复文本中划算。

## 草稿长度并非越大越好

若每个位置独立接受概率粗略为 $\alpha$，连续接受至少 $j$ 个候选的概率约为 $\alpha^j$。候选越往后，能被走到的概率越低；与此同时，草稿生成、目标验证宽度和临时 KV 都在增加。

例如 $\alpha=0.7$：

```text
第 1 个候选被接受的概率约 0.70
前 2 个都被接受的概率约 0.49
前 4 个都被接受的概率约 0.24
前 8 个都被接受的概率约 0.06
```

真实接受事件并不独立，这个例子只说明边际收益会下降。最佳 $k$ 取决于 prompt、领域、采样温度、batch 和硬件。动态 speculative decoding 会根据近期接受情况或系统负载调整候选长度，而不是为所有请求固定一个值。

## Proposer 不一定是另一台小模型

### 独立 draft model

使用与目标模型词表兼容的小模型，概念最接近原始算法。优点是通用，缺点是额外权重、KV Cache 和调度；草稿模型过小又可能接受率不足。

### EAGLE 与特征级草稿

EAGLE 系列利用目标模型的特征来预测后续 token/特征，试图用较小的草稿头获得更贴近目标的候选。它需要配套训练与 checkpoint，不能把任意模型名称填入配置就启用。

### Multi-Token Prediction

部分目标模型在训练时就包含 MTP 模块，可从当前状态预测多个未来 token。部署引擎需要理解对应权重结构和验证路径。

### Medusa/多头预测

在目标模型上增加若干预测头并构造候选树。目标验证可能不只是一条线性候选，而要处理树形 attention 与接受规则。

### N-gram、prompt lookup 与 suffix

从当前 prompt、已生成文本或全局后缀结构中查找重复模式，不需要额外神经网络。代码补全、模板文本和重复短语更容易命中，开放式创作中的覆盖会低一些。

这些方法不能只按论文中的最大 speedup 排序。需要同时考虑 checkpoint 可得性、额外显存、候选质量、运行时支持和目标流量。

## KV Cache 是实现正确性的核心

目标模型批量验证 $k$ 个候选时，会为这些临时位置产生 K/V。验证完成后，cache manager 必须像事务一样处理：

```text
reserve slots for draft inputs
        │
        ▼
target forward writes tentative KV for verified draft positions
        │
        ▼
verify candidates
        │
        ├─ accepted prefix ─► commit corresponding slots
        └─ rejected suffix ─► free / overwrite slots
```

修正 token 或全部接受后的额外 token 是从目标 logits 采出的，这不等于本轮 target forward 已经产生了它的 K/V；通常要等它在后续 forward 中作为输入时才物化，除非引擎显式融合了额外计算。

草稿模型也必须回到相同的正式位置。若目标接受了 2 个候选并生成 1 个修正 token，下一轮的逻辑上下文只能包含这 3 个正式 token；被拒候选之后的草稿 K/V 不能继续使用，而修正 token 对应的目标 K/V 也不能误用被拒 token 的缓存槽。

下列状态也要按正式提交长度推进：

- EOS、stop token 与最大长度；
- grammar/FSM 状态；
- repetition penalty 和 logits processor 历史；
- beam/sequence 状态；
- 流式输出缓冲区。

如果先把候选流式返回给用户，之后才发现拒绝，协议就需要“撤回 token”；多数文本 API 没有这种语义。因此系统通常只流式发送已经验证并提交的 token。

## 何时容易加速，何时可能变慢

推测解码更适合：

- 中低 QPS，目标 decode 明显受带宽限制；
- 单请求或小 batch 的交互式延迟优化；
- 草稿便宜且与目标分布接近；
- 文本有较强可预测性或重复结构；
- 目标验证多个 token 能充分利用 GPU。

收益可能很小甚至为负的情况包括：

- 高 QPS 下基线 continuous batch 已把 GPU 填满；
- 草稿模型占用额外 GPU 或通信成为瓶颈；
- 高温度、跨领域 prompt 导致接受率下降；
- 上下文极长，草稿和目标都要读取昂贵 KV；
- 与 pipeline parallel、grammar 或某量化 backend 的组合尚不成熟。

因此，官方文档将其主要定位为降低中低 QPS、memory-bound workload 的 inter-token latency，而不是保证所有服务提升总吞吐。

## 在 vLLM 中配置时应注意什么

配置 schema 会随版本演进，下面只展示当前文档中的结构，不应复制后长期不核对：

```bash
vllm serve <target-model> \
  --speculative-config '{
    "method": "draft_model",
    "model": "<draft-model>",
    "num_speculative_tokens": 5
  }'
```

上线前先从引擎日志确认 proposer 类型、草稿 checkpoint、候选数和并行配置确实生效。还要查目标版本的 known incompatibility：引擎支持“speculative decoding”不等于与任意 PP、量化、prefix cache、structured output 和模型架构组合都经过验证。

## 一组能解释收益的实验

准备对话、代码、摘要和推理四类 prompt，在完全相同的目标模型与 sampling 参数下比较基线。分别测试 greedy 和 temperature sampling：

### 正确性

- Greedy：比较输出 token 序列与无推测基线；
- Sampling：在小词表/可控 logits 上做拒绝采样统计检验，再做任务质量分布比较；
- 边界：EOS、stop string、最大长度、grammar 和取消；
- 状态：人工制造首个、中间和全部候选通过的情况，检查 KV 长度。

### 性能

扫描不同 `num_speculative_tokens` 和请求率，记录：

- 每轮 draft token 数、accepted token 数和平均推进长度；
- proposer、target verify、sampler 各自耗时；
- TTFT、TPOT/ITL、E2E；
- output tokens/s 与 SLO goodput；
- 目标/草稿 KV Cache 和峰值显存；
- 高并发下与基线的交叉点。

最终结论应类似“在这类流量和低于某请求率时降低 TPOT”，而不是脱离条件写成固定倍数。原始论文报告的 2—3 倍或 2—2.5 倍是其模型、硬件和实现上的实验结果，不是所有现代 serving 系统的承诺。

## 小结

推测解码把昂贵目标模型的串行步骤换成便宜 proposer 的串行草稿和目标模型的并行验证。它能够保持 greedy 结果或目标采样分布，关键在于逐位置接受、首次拒绝后的正残差修正，以及对临时 KV 状态的正确提交与回滚。

它是否更快则是另一件事。平均接受长度必须足以覆盖草稿、宽验证、额外缓存和采样开销；高并发时，基线 batch 本来已经充分利用 GPU，收益还可能缩小。正确的评估顺序始终是：先验证分布与状态，再分解每个阶段耗时，最后在真实 QPS 下比较 SLO goodput。

## 参考资料

- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
- [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318)
- [EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077)
- [vLLM Speculative Decoding 文档](https://docs.vllm.ai/en/latest/features/spec_decode/)
