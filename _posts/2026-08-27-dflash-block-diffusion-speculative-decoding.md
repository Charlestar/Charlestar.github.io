---
layout: post
title: "DFlash：Block Diffusion 怎样一次生成一整段 Draft"
subtitle: "从目标隐藏状态、并行块预测到前缀验证，理解扩散式推测解码的收益边界"
date: 2026-08-27 09:00:00 +0800
last_modified_at: 2026-09-03
author: iStar
catalog: true
series: speculative-decoding
series_order: 70
technology_year: 2026
mathjax: true
tags: [推测解码, LLM推理, GPU优化]
---

推测解码通常把大模型的一次串行生成拆成两件事：便宜的草稿模型先猜一段，目标模型再用一次前向并行检查这些位置。它并不要求草稿模型每次都猜对；只要正确前缀足够长，而生成草稿与验证的总成本又低于逐 token 调用目标模型，就能获得端到端收益。

EAGLE-3 一类方法已经证明，使用目标模型隐藏状态训练轻量 drafter，可以比单纯蒸馏下一个 token 获得更高的接受率。但它的草稿仍然沿着时间维自回归展开：要提出 8 个位置，关键路径上仍有多轮小模型前向。草稿模型可以很小，却不能消除这条串行链。

DFlash 改变的是草稿的执行方式。它把一个确定的 anchor token 与若干 mask 位置组成一个 block，让轻量 block diffusion drafter 在一次前向中同时预测所有 mask。目标模型的多层隐藏状态以 Key/Value 的形式注入 drafter 的每一层，弥补小模型独立预测远端位置时的信息不足。最终结果仍由目标模型验证，因此扩散模型只影响性能，不独自决定输出。

这篇文章沿一次完整 decoding cycle 展开 DFlash：它怎样并行预测、为什么仍只能接受最长前缀、训练目标如何匹配推理，以及把论文里的单卡加速变成在线服务收益还缺哪些工程条件。

## 推测解码优化的不是 FLOPs，而是串行轮数

设目标模型为 $M_t$，草稿模型为 $M_d$，每轮最多提出 $\gamma$ 个候选 token。一次推测循环可以抽象成：

```text
target context ──> drafter 提出 γ 个候选 ──> target 并行验证
                                                  │
                                                  └─> 接受连续正确前缀并继续
```

若一轮包含草稿时间 $T_{draft}$、验证时间 $T_{verify}$，平均能提交 $\tau$ 个 token，那么平均每个已提交 token 的时间近似为：

$$
L_{spec}=\frac{T_{draft}+T_{verify}}{\tau}
$$

相对普通自回归单步延迟 $L_{target}$，加速比为：

$$
S=\frac{L_{target}}{L_{spec}}
=\frac{L_{target}\cdot\tau}{T_{draft}+T_{verify}}
$$

这个式子同时解释了两类路线：

- EAGLE 类方法重点提高草稿与目标模型的一致性，从而增大 $\tau$；
- DFlash 除了追求接受长度，还试图把 $T_{draft}$ 从“多轮小前向”压缩为“一轮块前向”。

不能只比较接受率。一个更准但需要 8 次串行前向的 drafter，可能输给一个接受长度稍短、却只需要一次前向的 drafter；反过来，若块预测质量太低，一次生成再便宜也会被频繁拒绝抵消。

## 自回归草稿为什么仍有串行瓶颈

自回归 drafter 生成 $\gamma$ 个候选时，第 $i$ 个 token 依赖前 $i-1$ 个候选：

$$
q(y_{1:\gamma}\mid x)
=\prod_{i=1}^{\gamma}q(y_i\mid x,y_{<i})
$$

即使每一步只运行一层 Transformer，草稿延迟仍近似为：

$$
T_{draft}^{AR}\approx \gamma\cdot t_{step}
$$

GPU 可以在每一步内部并行计算矩阵，但不能让第 8 步越过第 1 步先执行。增加草稿长度会线性拉长关键路径；增加 drafter 深度又会放大每一步成本。工程上常见的浅层、树状候选，本质是在模型容量、候选宽度与串行深度之间折中。

DFlash 则把待预测位置同时放入一次前向：

$$
T_{draft}^{DFlash}\approx t_{block}(\gamma)
$$

$t_{block}$ 并非与块长完全无关，attention、输出投影和采样都会随位置数增长；但在适中的块长内，一次较宽的 GPU 运算通常比 $\gamma$ 次窄而串行的运算更容易吃满硬件。这是“并行草稿”成立的系统基础。

## 一个 block 里究竟放了什么

假设上一轮目标验证已经提交了一个可靠 token $a$。先固定一个容易混淆的计数约定：DFlash 配置中的 `block_size=b` 包含 anchor 槽位；默认 `sample_from_anchor=false` 时，anchor 不作为预测位，后面只有 $b-1$ 个 mask 候选。若用 $\gamma$ 表示候选数，则 $\gamma=b-1$：

```text
输入位置:   [历史上下文 ...] [anchor] [MASK] [MASK] [MASK] [MASK]
槽位计数:                         1      2      3      4      5   (b=5)
预测目标:                                 y1     y2     y3     y4   (γ=4)
```

这些 mask 位置使用非因果的 block 内 attention，可以同时看到 anchor、目标模型注入的上下文特征以及同一 block 的 mask 表示。一次 drafter 前向直接得到各位置的 vocabulary logits：

$$
(z_1,z_2,\ldots,z_\gamma)=M_d(h_t,a,m_{1:\gamma})
$$

再从每个 $z_i$ 采样或取最大概率 token，形成候选块。这里的“diffusion”容易造成两个误解：

1. **它不是让目标模型改成扩散生成。** 最终模型仍是原来的 autoregressive LLM。
2. **它不是在生产关键路径上做很多轮去噪。** DFlash 为降低 drafting latency，直接用一次块预测提出候选；目标模型验证承担质量兜底。

因此它更像一个针对 future block 训练的轻量扩散适配器，而不是独立完成整段文本生成的通用 diffusion LLM。

## 并行预测不等于各位置互相独立

若简单地让所有位置只看相同历史上下文，然后分别预测下一个、第 2 个、第 3 个 token，远端位置会缺少中间 token 信息：

$$
q(y_i\mid x)\quad\text{而目标分布是}\quad p(y_i\mid x,y_{<i})
$$

位置越靠后，这个条件差异通常越大。DFlash 用两种信息缓解它：

- block 内使用双向 attention，使各 mask 位置能够联合建模块的结构；
- 从目标模型多个深度抽取隐藏状态，融合后持续注入 drafter，提供比单个 token embedding 更丰富的语义上下文。

但注意，mask 在推理时还不是正确 token。双向 attention 能建立位置之间的联合表示，却不会凭空提供真实的 $y_{<i}$。所以 DFlash 仍会出现“前面准、后面逐渐偏离”的接受分布，块长不是越大越好。

## 目标隐藏状态为何要注入每一层 KV

目标模型在 prefill 或上一轮验证时已经计算了隐藏状态。DFlash 从若干由浅到深的目标层取特征，拼接后通过轻量 projection 得到紧凑上下文 $h_t$：

$$
h_t=W_p[h_t^{(l_1)};h_t^{(l_2)};\ldots;h_t^{(l_k)}]
$$

一种直接做法，是把 $h_t$ 与 token embedding 相加或拼接，只送入 drafter 的输入层。问题是 drafter 变深后，目标信息可能逐层稀释。DFlash 把融合后的特征映射为每个 draft layer 的 Key 和 Value：

$$
K_h^{(l)}=W_K^{(l)}h_t,\qquad
V_h^{(l)}=W_V^{(l)}h_t
$$

mask query 在每一层都能重新访问这些特征。对应的 KV 可以跨 drafting cycle 复用，避免反复从头构造目标上下文。这让 drafter 有空间增加到多层，同时仍与目标模型的表示对齐。

代价也很明确：

- serving engine 必须暴露并保存选定目标层的 hidden states；
- target 与 drafter 的层选择、投影、embedding、LM head 必须严格匹配 checkpoint；
- 多一份 draft KV 与中间 buffer 会占用显存；
- tensor parallel 或 pipeline parallel 下，隐藏状态布局与通信也要纳入关键路径。

所以 DFlash 不是一个只靠切换命令行参数就能套在任意模型上的无状态插件。

## 为什么验证仍然只接受最长前缀

目标模型一次可以计算候选块各位置的条件分布：

$$
p_i=p(y_i\mid x,y_{<i})
$$

但生成的语义仍是自回归的。若第 3 个候选被拒绝，第 4 个候选是在错误的第 3 个 token 条件下构造或验证的，不能跳过第 3 个继续接受。因此提交结果必然是：

```text
draft:   y1 ✓   y2 ✓   y3 ✗   y4 ?   y5 ?
commit:  [y1, y2] + 目标分布给出的校正 token
```

greedy decoding 可以直接比较 candidate 与目标模型 argmax。带温度采样时则必须使用与 speculative sampling 一致的接受—拒绝与残差分布，才能保持目标模型的采样分布。不能把“目标模型看过一遍”误写成“无条件无损”；无损来自正确的验证算法，而不是 DFlash 名字本身。

若第 $i$ 个位置在此前缀已经通过的条件下被接受的概率为 $c_i$，至少接受到第 $k$ 个候选的概率为：

$$
P(A\ge k)=\prod_{i=1}^{k}c_i
$$

期望接受长度可写成这些前缀存活概率之和。早期位置的一次错误会让整个后缀失效，所以训练和评测都不能只看逐 token accuracy。

## 训练目标怎样贴近真实推测循环

DFlash 训练时冻结目标模型，并让训练样本先经过目标模型以提取 clean target features。对 response 随机选择 anchor，把 anchor 之后的 block 位置遮住，要求 drafter 并行恢复后续 token。

随机 anchor 有两个作用：

- 训练输入与线上状态一致——每轮都从目标模型已经确认的 token 开始；
- 同一条长 response 能在不同 epoch 暴露不同位置，而不必把所有块都展开。

多个训练 block 可以拼到同一 sequence，通过稀疏 attention mask 隔离：block 内允许双向 attention，可以访问对应目标特征；block 之间不能泄漏答案。这样一次 forward/backward 能训练多个 anchor，而显存成本不必随完整长上下文中的所有候选块增长。

## 前面的位置为何权重更高

普通交叉熵会把块内各位置视作同等重要，但推测解码的收益不是对称的。第 1 个位置错误时，后面即使都猜对也无法提交；第 12 个位置错误，前 11 个仍可能带来收益。

DFlash 使用随位置衰减的 loss weight：

$$
w_k=\exp\left(-\frac{k-1}{\gamma_w}\right)
$$

训练损失为：

$$
\mathcal{L}=\sum_{k=1}^{\gamma}w_k\,
\mathrm{CE}(q_k,y_k)
$$

它不是说后缀质量无关，而是让优化目标更贴近 expected accepted prefix。实际训练仍需同时观察各深度 accuracy、完整块接受率和平均接受长度，否则可能得到一个只擅长前几个位置的过度保守 drafter。

论文还让 drafter 与 target 共享并冻结 token embedding 和 LM head，只训练 draft Transformer 与相关投影。这样既减少训练参数，也避免另建词表映射破坏 logits 对齐。后续 DSpark 使用另一种 `sample_from_anchor=true` 约定：anchor 槽位本身也产出第一个 future-token logit，因此相同 `block_size` 会得到 $b$ 个候选。checkpoint 的训练约定与 serving 配置必须一致，不能只看张量 shape 互换两种模式。

## DFlash、EAGLE-3 与 DSpark 的区别

三者都服务于“少调用几次大模型”，但解决的是不同瓶颈：

| 方法 | 草稿形成方式 | 主要优势 | 主要约束 |
| --- | --- | --- | --- |
| EAGLE-3 | 基于目标 hidden features 自回归展开候选树 | 候选质量高，生态成熟 | drafting 仍有串行深度，树验证会放大 token 数 |
| DFlash | 一次 block diffusion 前向并行预测固定块 | 显著缩短草稿关键路径，可使用更深 drafter | 后部位置缺少真实前缀，模型与引擎耦合较深 |
| DSpark | 半自回归并行草稿加置信度调度 | 按请求选择验证长度，减少低价值验证 | 需要可靠的 confidence calibration 与 ragged execution |

从技术演进看，DFlash 先回答“能否快速并行提出一整块”，DSpark 进一步回答“这一块究竟值得验证多远”。二者不是简单的替代关系；在工程系统中，块预测、置信度和变长验证可以成为相互配合的层次。

## block size 是训练参数，也是运行时资源参数

增大 block size $\gamma$ 会带来三组相反变化：

1. 可接受的上限提高，理论上每轮可以推进更多 token；
2. 后部位置更难预测，完整前缀存活概率下降；
3. target verification 的 token 数、attention workspace 与采样成本增加。

因此应优化实际每 token 成本，而非最大接受长度：

$$
\gamma^*=\arg\min_{\gamma}
\frac{T_{draft}(\gamma)+T_{verify}(\gamma,B,L)}{E[A_\gamma]}
$$

其中 $B$ 是并发 batch，$L$ 是上下文长度。低并发下，verification 的宽度可能几乎免费，较大 block 更有利；高并发下，额外候选会挤占其他请求的计算，较小 block 反而有更好的集群吞吐和尾延迟。

论文观察到：用大 block 训练的 drafter 通常能向下兼容较小的推理 block，但反向泛化较差。这为动态 block scheduling 留出空间，不过上线前仍要逐个 checkpoint 与 workload 验证，不能把跨长度泛化当成协议保证。

## 在线调度不能把 draft 与 verify 当成一台普通模型

一个 serving iteration 现在至少有两类工作：

```text
Draft queue  ──> block drafter ──┐
                                 ├─> Verify queue ──> target model
Normal decode ───────────────────┘
```

如果调度器先积累很大的 draft batch，再把所有候选一口气送去验证，可能制造 verification burst；如果 target 一直优先处理普通 decode，draft 请求会老化；如果 draft 与 verify 串行运行在同一 stream，论文中的 overlap 收益也不会自动出现。

调度器至少要显式记录：

- `draft_block_size`：本轮实际提出多少位置；
- `draft_tokens` 与 `verified_tokens`：两类计算量不能混为 output tokens；
- `accepted_prefix`：每轮真实推进长度；
- `draft_wait_ms`、`verify_wait_ms`：定位排队来自哪一侧；
- `draft_model_id`、`target_model_id` 与 checkpoint hash：防止错误配对；
- sampling 参数与 verifier mode：保证请求语义一致。

并发升高后，“单请求加速比”通常会下降，因为 baseline 本身已通过 batching 提高 GPU 利用率，而 DFlash 仍要支付 draft、验证和调度开销。论文在 SGLang 中也报告了加速比随并发变化；容量规划应使用目标生产并发，而不是只引用 concurrency=1 的最佳数字。

## KV Cache 要分清三份状态

DFlash 系统中容易把不同 KV 混在一起：

1. **目标模型 KV Cache**：保存已提交历史，决定正常 autoregressive 语义；
2. **目标上下文特征**：从若干目标层抽取、融合，作为 drafter 条件；
3. **draft KV Cache**：保存注入特征与 drafter 自身需要复用的状态。

候选被拒绝时，只能把目标 KV 提交到已接受边界；未接受候选对应的临时 target KV 必须回滚、截断或留在可覆盖的 scratch page。draft 状态也要以同一 commit point 为界。这里还要把“正式 token 已提交到哪里”和“target KV 已物化到哪里”分成两个游标：

```text
committed_token_end 已验证并进入正式序列的位置（可含本轮修正/bonus token）
target_kv_end       目标模型已经把 token 当作输入并写出 KV 的位置
speculated_end      本轮候选末尾
accepted_draft_end  本轮已接受草稿前缀的末尾
draft_epoch         draft 状态所属轮次
```

只有验证成功后才能推进 `committed_token_end`；本轮由 target logits 产生的修正/bonus token 尚未作为输入计算，因此 `target_kv_end` 通常暂时比它落后一个位置，并在下一轮补齐。请求取消、超时、迁移或 worker 重启时，应按 KV 的实际物化边界回收 speculative pages，避免把拒绝候选页误当成修正 token 的状态，也避免幽灵引用与显存泄漏。

## CUDA Graph 与 ragged batch 的矛盾

固定 block size 便于捕获 CUDA Graph，但真实请求会在不同位置结束，且动态调度可能为不同请求选择不同长度。若把所有请求 padding 到最大块：

$$
N_{verify}^{padded}=B\cdot\max_i\gamma_i
$$

而真正需要的验证 token 数只是：

$$
N_{verify}^{ragged}=\sum_i\gamma_i
$$

要避免省下的候选又被 padding 补回去，可以准备少量分级 graph，例如 4、8、12、16，或使用支持 ragged verification 的 kernel。graph 数量过多会增加捕获时间、显存常驻和调度复杂度；分级太粗则浪费验证计算。最优 bucket 应来自真实 `gamma_i` 分布，而不是拍脑袋选择。

## 质量不下降需要哪些前提

“lossless acceleration”有严格前提：

- target checkpoint、tokenizer、chat template 与采样实现保持一致；
- greedy 或 stochastic verification 实现符合对应算法；
- rejected suffix 不会污染 target KV、random state 或 structured-output state；
- 停止词、EOS、logit processor、grammar constraint 在 draft 与 verify 边界上语义一致；
- 数值精度、并行归约顺序带来的差异在允许范围内。

结构化输出尤其容易踩坑。若 grammar automaton 在“草稿生成时”就永久前进，而候选随后被拒绝，下一轮约束状态便会错位。正确做法是让 parser、tool-call detector 和 stop matcher 都具有 speculative snapshot，验证后只提交接受前缀。

## 评测要拆成算法层与服务层

### 算法层

- 各任务、各温度下的平均接受长度 $\tau$；
- $P(A\ge k)$ 前缀存活曲线，而非只有均值；
- 每个 block 位置的 token accuracy；
- draft layer 数、block size、特征层选择的消融；
- 短上下文与长上下文的接受率漂移；
- greedy 输出逐 token 一致性，采样输出的分布检验。

### 引擎层

- $T_{draft}$、$T_{verify}$、采样和状态提交的分项延迟；
- concurrency 1 到饱和点的 TPS、TTFT、TPOT、P95/P99；
- CUDA Graph 命中率与 eager fallback；
- target/draft KV 占用、临时 verification pages 和 OOM recovery；
- 不同 tensor parallel、attention backend、量化配置的正确性。

### 集群层

- 每 GPU 输出 token/s 与每百万 token 成本；
- 混合长短请求时的公平性；
- draft/verify queue 深度和 backpressure；
- worker 滚动升级期间的 checkpoint 配对错误；
- 取消、超时、迁移与故障恢复后的状态泄漏。

只有三层都成立，才能把实验室里的 decoding speedup 换算成稳定的容量收益。

## 什么时候不应启用 DFlash

DFlash 并非所有流量的默认最优解。以下场景需要谨慎：

- 目标模型很小，普通 decode 已经足够快，draft overhead 占比反而过高；
- 高并发下 target 已接近计算饱和，宽验证抢占正常 batch；
- workload 与 drafter 训练分布差异大，接受长度长期接近 1；
- 超长上下文未做适配，目标特征分布漂移；
- 量化、LoRA、grammar、logprobs 或特殊 sampler 尚未通过兼容性验证；
- 显存紧张，新增 draft weights、KV 与 scratch pages 会降低可服务 batch。

一个实用的启用条件是估计滑动窗口收益：

$$
\widehat{G}=
\frac{\widehat{\tau}\,L_{target}}
{\widehat{T}_{draft}+\widehat{T}_{verify}}
$$

当 $\widehat{G}$ 连续低于包含安全余量的阈值时，回退普通 decode；恢复则使用滞回阈值，避免在临界点频繁切换。控制器应按 model、task class、context bucket、sampling mode 和并发分桶，而不是只维护一个全局平均值。

## 从实验到上线的一条路径

1. 固定 target checkpoint、tokenizer 与采样语义，选择严格配对的 DFlash checkpoint。
2. 离线测量位置 accuracy、接受长度和长上下文退化，先验证正确性。
3. 在目标 engine/backend 上分别 profile draft、verify、commit，确认一次块预测没有被隐藏的同步抵消。
4. 以真实并发和输入/输出长度回放，选择 block bucket 与 graph 集合。
5. 小流量灰度，按请求记录 baseline shadow estimate 与真实 accepted prefix。
6. 建立自动回退：低收益、内存压力、checkpoint 不匹配或 unsupported feature 均切回普通 decode。
7. 最后才把 GPU 小时、输出 token 与 SLO 违约一起纳入成本结论。

## 总结

DFlash 的关键不是“扩散模型比自回归模型生成得更好”，而是重新分工：目标模型负责语义与最终分布，轻量 block diffusion 模型只负责用一次 GPU 前向尽可能多地提出候选。目标隐藏状态的逐层 KV injection 提高草稿质量，anchor-aligned training 和前缀加权 loss 让训练目标贴近真实接受收益，target verification 则把错误候选限制在尚未提交的 speculative state 中。

它带来的新问题同样具体：block size 要随并发变化，draft 与 target 状态必须事务化提交，动态长度要避免 padding 回填，长上下文与特殊采样需要单独校验。理解这些边界后，DFlash 才不只是一个漂亮的加速比，而是一套可以被测量、灰度和回退的推理路径。

## 参考资料

- [DFlash: Block Diffusion for Flash Speculative Decoding](https://arxiv.org/abs/2602.06036)
- [vLLM Speculators: DFlash](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/dflash/)
- [EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test](https://arxiv.org/abs/2503.01840)
- [Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models](https://arxiv.org/abs/2503.09573)
- [快手万擎大模型推理成本和性能优化实践](https://zhuanlan.zhihu.com/p/2067652898524345525)
