---
layout: post
title: "MoE 与推测解码：计算、通信和接受率的联合优化"
subtitle: "从专家散射看清为什么验证更多 token 可能反而更慢"
date: 2026-05-29 12:00:00 +0800
last_modified_at: 2026-08-09
author: iStar
catalog: true
series: speculative-decoding
series_order: 60
technology_year: 2026
mathjax: true
tags: [推测解码, MoE, 专家并行]
---

推测解码把多个候选 token 交给目标模型并行验证。对于 dense 模型，这通常是在一次权重读取中做更多计算：矩阵从一个向量乘法变成较小的矩阵乘法，硬件利用率提高，而模型权重集合没有变化。

把目标模型换成 sparse MoE 后，直觉容易失效。每个 token 只访问少数专家，但不同候选可能访问不同专家。原来自回归一步只需读取一小组专家权重，一次验证整条候选链或整棵树时，却可能触及更大的专家并集。并行验证减少了串行轮数，也扩大了每轮的稀疏工作集。

这就是 MoE 推测解码的核心矛盾：

> 候选越多，命中更长前缀的机会越大；候选越分散，目标模型为验证它们付出的专家访存与通信成本也越高。

因此，问题不再只是“草稿模型能猜中多少 token”，而是“每多接受一个 token，需要额外触发多少专家计算和数据移动”。

## 从一层 MoE 的执行过程开始

一个典型 sparse MoE 层包含 router、若干 routed experts，模型也可能同时包含 shared expert。对 token 表示 \(x\)，router 给出各专家分数，并选择 top-\(r\) 个专家：

\[
\mathcal{E}(x)=\operatorname{TopR}(\operatorname{softmax}(W_r x))
\]

该层输出可以抽象为：

\[
y=\sum_{e\in\mathcal{E}(x)}g_e(x)E_e(x)+E_{shared}(x)
\]

参数总量可能很大，但单个 token 只经过少数 routed experts，这就是条件计算带来的优势。

在单 GPU 上，一批 token 通常经历：

```text
hidden states
    │
    ▼
top-k router
    │  生成 token -> expert assignments
    ▼
permute / group by expert
    │
    ▼
grouped GEMM for active experts
    │
    ▼
unpermute + weighted combine
```

当专家被分布到多张 GPU 上，即 Expert Parallelism（EP），中间还会加入 dispatch/combine 通信：

```text
本地 token
  -> router
  -> all-to-all dispatch 到专家所在 GPU
  -> grouped expert GEMM
  -> all-to-all combine 回原 rank
```

这里需要区分两类数据移动：

- 专家常驻各 GPU 时，EP 主要移动 token hidden states 和路由元数据；
- 专家被卸载到 CPU 或分层缓存时，还可能把专家权重迁移到 GPU。

两种系统都受“激活了多少不同专家”影响，但瓶颈不同。前者更关注 HBM 权重读取、token dispatch 和负载不均，后者还要承担 PCIe/NVLink 上的权重迁移。

## dense 验证与 MoE 验证的差别

假设目标模型验证 \(m\) 个 token。dense FFN 无论 \(m=1\) 还是 \(m=8\)，每层都会访问同一组权重：

\[
\mathcal{W}_{dense}(m)=\mathcal{W}_{dense}
\]

当 batch 很小时，权重读取通常占主要成本；一次处理更多 token 能提高算术强度，所以 \(T_{dense}(8)\) 往往远小于 \(8T_{dense}(1)\)。这为推测解码创造了空间。

MoE 的活跃权重集合则取决于 token 路由：

\[
\mathcal{W}_{moe}(m)=\bigcup_{i=1}^{m}\mathcal{E}(x_i)
\]

若 8 个候选都落到相近专家，专家权重可以被同一 grouped GEMM 中的多个 token 复用；若它们路由到互不重叠的专家，活跃专家并集会迅速扩大。后一种现象通常称为 **expert scattering**。

考虑一层有 16 个 routed experts、每 token 选择 2 个专家：

```text
候选 1 -> {E2, E7}
候选 2 -> {E2, E7}
候选 3 -> {E2, E9}
候选 4 -> {E4, E11}
```

四个 token 共有 8 次 expert assignment，但只触及 5 个不同专家。如果四个候选分别落入完全不同的组合，最多会触及 8 个专家。计算量都按 8 次 assignment 计，权重工作集、kernel 形状和通信分布却不同。

所以“验证 token 数”不是 MoE 验证成本的充分代理。至少还要知道：

- 每层 unique active experts；
- 每个专家接收的 token 数；
- 专家集合在相邻验证轮之间的复用；
- assignment 跨 GPU、跨节点的比例；
- 最慢 rank 的 token 数与通信完成时间。

## 推测采样如何保持目标分布

先把系统成本放在一旁，经典推测采样的正确性来自严格的接受与校正规则。

草稿模型按 \(q\) 生成候选 \(y\)，目标模型计算对应概率 \(p(y)\)。候选以如下概率被接受：

\[
P(accept\ y)=\min\left(1,\frac{p(y)}{q(y)}\right)
\]

在第一次拒绝处，不能简单地丢掉候选后直接从 \(p\) 重采样；需要从校正分布中采样：

\[
p'(x)=\operatorname{Normalize}\left(\max(p(x)-q(x),0)\right)
\]

随后停止接受该候选链后面的 token。tree-based 方法的记账更复杂，但原则相同：候选生成可以任意加速，最终目标分布必须通过验证与校正恢复。

这个边界对 MoE 优化非常重要：

- 改变草稿树选择策略，不必改变目标分布，只要后续验证仍严格执行；
- 更换 target MoE kernel 或通信后端，不应改变数学结果，只需控制数值误差；
- 在目标模型验证时直接跳过 router 选中的专家，则改变了目标 logits，经典 lossless 证明不再成立。

后者可以成为有用的近似解码方法，但应明确报告质量—性能权衡，不能与严格推测采样混为同一种结果。

## 接受率为什么仍然不等于加速比

设一次推测轮：

- 草稿阶段提出 \(k\) 个候选；
- 目标模型验证 \(n\) 个节点，链式草稿时常有 \(n\approx k+1\)；
- 本轮最终提交 \(R\) 个 token；
- 记账、采样与 KV 处理成本为 \(T_{book}\)。

推测轮成本为：

\[
T_{spec}=T_{draft}(k)+T_{verify}^{moe}(n,\mathcal{U})+T_{book}
\]

其中 \(\mathcal{U}\) 表示各层活跃专家并集及其分布。若普通自回归生成一个 token 的成本为 \(T_{AR}\)，一个直观的相对效用可以写成：

\[
Utility=\frac{R\cdot T_{AR}}{T_{spec}}
\]

只有 \(Utility>1\) 时，这一轮才真正节省时间。

平均接受长度变大通常提高 \(R\)，但以下情况仍可能让效用下降：

1. drafter 自己很慢，尤其 drafter 也包含 MoE 层；
2. 更宽的候选树触发大量 unique experts；
3. 验证 token 在专家间过于分散，grouped GEMM 都是很小的 shape；
4. 某个 rank 收到远多于其他 rank 的 assignment，所有 rank 等待 straggler；
5. dispatch/combine 的固定启动成本和跨节点流量上升；
6. 大量已验证分支最终被拒绝，计算没有转化为提交 token。

因此，比 acceptance rate 更接近系统收益的指标是：

\[
\text{target efficiency}
=\frac{\text{committed tokens}}
{\text{target expert executions or verification time}}
\]

它迫使评测同时考虑“得到了多少”和“验证花了多少”。

## 候选链与候选树怎样改变专家工作集

链式草稿在每个深度只有一个候选。它验证的 token 少，但一旦某一步不被接受，后续候选全部失效。

树式草稿在每层保留多个分支，能够覆盖更多可能的目标路径。代价是目标模型要为最终只会选择一条的分支都进行 router 和 expert 计算：

```text
root
├── token A -> experts {1, 3}
│   ├── token C -> experts {1, 8}
│   └── token D -> experts {4, 9}
└── token B -> experts {6, 7}
    └── token E -> experts {2, 10}
```

如果只按草稿概率选择节点，A、B、C、D、E 都可能是很好的候选；从系统视角看，这棵树却触及 9 个专家。另一个概率质量略低、但复用 `{1,3,8}` 的候选组合，可能以更低验证成本取得更高端到端效用。

这引出两个不同控制量：

- **speculation depth/width**：控制验证多少节点；
- **expert footprint**：控制这些节点共同触及多少专家。

在 dense 模型中，前者几乎足以描述验证宽度；在 MoE 中，二者不能合并成一个数字。

## 三条优化路径解决的不是同一个问题

现有研究可以按是否改变 target verification 来区分。

### 根据实时效用调整是否推测以及 draft length

Cascade 的思路是先短暂测试当前请求的推测效用，再在一段稳定区间中选择是否启用 speculation 以及使用多大的 \(K\)。它利用相邻 iteration 行为具有一定局部性的假设，避免每一步都进行昂贵搜索。

这类方法不需要预测具体专家，解决的是“当前请求现在值不值得推测”：

```text
test phase: 试运行若干候选 K，测 token gain / verification cost
     │
     ├── utility <= 1 -> 暂时关闭 speculation
     └── utility > 1  -> 选择效用最高的 K
                              │
                              ▼
                          set phase
```

它适合接受行为随任务和生成阶段变化的场景。例如代码缩进和固定格式区域可能易于预测，自由文本或复杂推理转折处则未必。

### 在不改变目标验证的前提下选择更便宜的候选

EcoSpec 把 predicted marginal expert activation cost 纳入候选树选择。其目标不是盲目选择触发专家最少的 token，而是在候选概率与新增专家成本之间做权衡：优先选择既可能被接受、又能复用当前验证集合已覆盖专家的路径。

可以把候选节点 \(v\) 的启发式评分写成：

\[
Score(v)=\log P_q(v)-\lambda\cdot \Delta C_{expert}(v)
\]

其中 \(\Delta C_{expert}(v)\) 是把节点加入树后预计新增的专家成本，\(\lambda\) 控制概率质量与成本之间的权衡。这只是帮助选择“验证谁”；最终入选节点仍由完整目标模型验证，所以不会因为候选排序本身改变目标分布。

难点是专家路由依赖目标模型各层隐藏状态，在真正运行 target 前并不完全已知。因此需要轻量 expert predictor，并且预测误差也要进入评估：预测器太重会吞掉收益，预测不准则无法降低真实专家并集。

### 为验证阶段设置专家预算

MoE-Spec 从另一侧入手：限制每层验证时实际加载或执行的专家数量，丢弃长尾专家，以固定 expert capacity 控制内存成本。这样可以在不缩短草稿深度的情况下限制工作集。

这条路径更直接，但会改变目标模型原本的专家计算，因此属于质量可调的近似方法。预算越紧，性能潜力越大，目标 logits 偏差也可能越大。评测必须同时给出任务质量、分布差异或回归率，而不能只报告吞吐。

三类方法的边界可以归纳为：

| 路径 | 控制对象 | 是否完整执行目标模型 | 主要风险 |
| --- | --- | --- | --- |
| 动态 \(K\) / 开关 | 候选数量 | 是 | 控制滞后、频繁切换 |
| 成本感知树选择 | 验证哪些候选 | 是 | 专家预测器成本与误差 |
| 专家预算 | target 执行哪些专家 | 否 | 输出分布或任务质量变化 |

把这三种结果放进同一张性能表时，应单独标记 exact 与 approximate，否则读者无法判断加速来自系统优化还是少做了目标模型计算。

## MTP、EAGLE 与独立 drafter 在 MoE 上的成本不同

候选从哪里来，也会改变总成本。

### 独立小模型

它与目标模型解耦，可以是 dense 模型，不需要运行 target experts。缺点是与目标分布可能不够一致，并占用额外显存。若放在另一组 GPU，还要考虑跨设备传输和调度延迟。

### EAGLE 类特征 drafter

它复用目标模型隐藏状态，通常结构较浅，接受率可能更高。目标模型需要暴露特征，runtime 要维护草稿树、tree attention 与额外 KV 状态。若 drafter 内部含 MoE 层，它的 runner 和通信精度可能与 target 不同。

### Multi-Token Prediction（MTP）模块

MTP 与目标模型一同训练或随模型发布，词表和表征天然对齐，免去外部 drafter。它并不意味着“零成本”：生成多步候选仍有前向开销，MTP 层本身也可能复用 MoE 结构。

当前 SGLang 为 speculative MoE runner 和 A2A backend 提供独立配置入口。例如 target experts 使用量化 kernel，而 MTP drafter 仍为 BF16 时，两者可能需要不同 runner。直接沿用 target backend，未必支持 drafter 的 dtype 或权重布局。

选择 drafter 时应比较完整路径：

```text
drafter latency
+ drafter memory
+ feature / token transfer
+ target verification cost
+ acceptance gain
```

只比较模型参数量会漏掉通信与集成开销。

## Expert Parallelism 让验证成本带上拓扑

在 EP 中，router 决定 token 发往哪个 rank。即使所有 rank 总 assignment 数相同，跨节点比例和最大 rank 负载也会不同。

一次 MoE 验证可以粗略拆成：

\[
T_{verify}^{moe}=
T_{attn}+T_{route}+T_{dispatch}+T_{grouped\ GEMM}+T_{combine}+T_{sample}
\]

其中：

- \(T_{route}\) 受验证 token 数影响；
- \(T_{dispatch}\) 和 \(T_{combine}\) 受 assignment 数、消息大小、跨节点路径和 collective 实现影响；
- \(T_{grouped\ GEMM}\) 受 unique experts、每专家 token 数和负载倾斜影响；
- 整层延迟往往由最慢 rank 决定。

候选树可能让某些专家形成更大的局部 batch，提高 GEMM 效率；也可能把 token 稀疏地摊到许多专家，使每组矩阵都太小。这说明“更多并行 token”既可能改善 kernel shape，也可能破坏权重复用，必须 profile 才能判断净效应。

常见可优化点包括：

- 选择适合硬件拓扑的 A2A backend；
- 使用面向 decode 小 token batch 的 grouped GEMM kernel；
- 将 shared expert 计算与 routed expert 通信重叠；
- 根据真实路由统计做 expert placement 或复制热点专家；
- 避免不同 rank 以不同顺序发起 collective；
- 将候选树节点打包成后端期望的连续或 masked layout。

这些优化不应在一次实验中同时全部打开。否则端到端变快后，也无法确认收益来自候选质量、expert placement 还是通信后端。

## CPU offload 场景为什么又是另一种问题

若所有专家无法常驻 GPU，每个 token 可能触发 CPU 到 GPU 的专家迁移。普通自回归逐 token 执行，迁移高度串行；推测验证能提前看到多个 token 需要的专家，并合并传输，但候选也可能扩大所需专家集合。

SpecMoE 针对的是这类 memory-constrained system。其 self-assisted drafter 使用目标 MoE 的一部分常驻专家构成便宜草稿路径，验证阶段再集中处理目标模型真正需要的专家迁移。收益来自减少或合并迁移，而不是说明所有 GPU-resident MoE 都会得到相同倍率。

分析 offload 系统时，应额外记录：

- 每轮迁移的 unique expert weights；
- host-to-device 字节数和带宽利用率；
- expert cache hit rate；
- 预取命中率与错误预取字节；
- 迁移与计算的重叠比例。

如果这些指标不变，声称速度来自“专家预取”就缺少证据。

## KV Cache 与回滚仍要精确

MoE 的 expert routing 通常不产生像 attention KV 那样跨 token 长期保留的缓存，但推测解码仍要维护多种临时状态：

- target KV Cache 中候选节点占用的页；
- drafter 自己的 KV 或特征状态；
- tree parent、position 和 attention mask；
- token-to-expert assignment 与 permute metadata；
- sampling RNG、接受/拒绝和校正分布状态。

验证后只有被提交路径对应的 target KV 可以保留。被拒绝分支必须释放或回收到 allocator；下一轮不能误用其位置和路由 metadata。

确定性 greedy 测试常常发现不了 RNG 和校正采样错误。正确性测试应同时覆盖：

1. 贪心生成与无推测基线逐 token 一致；
2. 温度采样在大量重复实验中与 target-only 分布一致；
3. 不同接受长度下 KV 页计数能回到预期值；
4. batch 内某些请求提前结束时，不污染其他请求；
5. EP rank 在空 token、极端倾斜和跨节点路由下仍按同一 collective 顺序执行。

## 怎样设计一组能解释结果的实验

直接比较“关闭推测”和“打开所有优化”只能得到一个数字，无法知道该如何继续调优。更有效的实验按层次推进。

### 先验证算法正确性

在单 GPU 或最小 EP 拓扑上，用很小的请求集对比 target-only。固定 seed，覆盖 greedy、temperature、top-p、EOS 和不同 draft length。若方法使用专家预算，明确把它归入 approximate 组，并单独跑质量回归。

### 再测候选行为

记录每轮：

```text
proposed nodes
accepted prefix length
committed tokens
acceptance by depth
draft confidence
rejection position
```

这样能区分是 drafter 不准，还是候选树把预算分配到了错误分支。

### 然后加入 MoE 路由指标

对每个 MoE 层采集：

```text
unique experts per verification
token assignments per expert
expert overlap across candidates
expert overlap across iterations
max / mean tokens per EP rank
cross-node assignment ratio
```

可以定义专家复用率：

\[
Reuse=1-\frac{|\bigcup_i\mathcal{E}(x_i)|}
{\sum_i|\mathcal{E}(x_i)|}
\]

它不是跨模型通用的性能指标，但在 top-\(r\) 固定时，能帮助比较两种候选树是否把 assignment 集中在更小的专家集合中。

### 最后拆解系统时间

使用 GPU trace 和通信 profiler 分离：

```text
draft
target attention
router + permutation
dispatch
expert GEMM
combine
sampling + bookkeeping
allocator / host scheduling gaps
```

如果 target verify 变贵来自 grouped GEMM，而不是 A2A，继续更换通信库不会解决问题；如果 GPU kernel 之间有大量 host gap，则应先检查调度与同步。

## 负载维度决定结论能否外推

MoE 推测解码对工作负载尤其敏感。至少要覆盖：

| 维度 | 为什么会改变结果 |
| --- | --- |
| batch / concurrency | 决定权重是否已有复用、GEMM 是否饱和 |
| prompt 与 output length | 区分 prefill 占比、长 decode 与 KV 压力 |
| 任务类型 | 代码、数学、对话的可预测性和路由模式不同 |
| reasoning phase | 固定格式、反思和答案阶段接受行为不同 |
| draft tree 参数 | 同时改变接受机会与 expert footprint |
| EP size 与节点数 | 改变跨卡、跨节点 dispatch 成本 |
| expert placement | 改变热点与 straggler 所在位置 |
| 精度与 kernel | 改变 target 和 drafter 的算力/带宽平衡 |

报告均值之外，还应提供 P50/P95/P99 TPOT 和每请求效用分布。静态 \(K\) 可能让一部分请求大幅加速、另一部分请求明显减速，均值无法表达这种风险。

## 如何阅读论文中的性能数字

这一方向的研究设置差异很大：

- Cascade 在其 vLLM 原型与五个 MoE 工作负载中，重点展示动态选择可以避免静态 \(K\) 的减速，并报告相对静态策略的吞吐改善；
- MoE-Spec 报告的是在可调质量前提下进行 verification-time expert budgeting；
- SpecMoE 的高倍率来自 CPU-offloaded、内存受限系统中的专家迁移优化；
- EcoSpec 在 DeepSeek-V3.1、Qwen3-235B-A22B 与 GPT-OSS-120B 等实验上，将候选接受概率与边际专家成本共同纳入选择。

这些结果不能横向拼成“MoE 推测解码通常能加速多少”。复现数字前必须先确认：

- experts 是 GPU-resident 还是 CPU-offloaded；
- 方法是否严格保持 target 分布；
- baseline 使用固定还是调优后的 draft length；
- drafter、target、通信和采样成本是否全部计入；
- throughput 是 token/s、request/s 还是 goodput；
- 硬件互联、EP 拓扑和 batch 是否一致。

只有实验边界相同，倍率才有可比性。

## 从现象回到可操作的诊断

| 现象 | 需要对照的指标 | 可能的下一步 |
| --- | --- | --- |
| 接受率高但速度下降 | unique experts、verify latency、draft latency | 减小树宽，启用效用控制，检查专家散射 |
| 单卡加速、EP 变慢 | dispatch/combine、跨节点 assignment、rank imbalance | 调 A2A 后端和 expert placement |
| 低并发变慢、高并发正常 | 每专家 token 数、权重复用、kernel shape | 按并发动态开关 speculation |
| 第一层候选好、深层差 | acceptance by depth、drafter 自回归误差 | 缩短 depth 或改进多步 drafter |
| candidate 数增加但专家并集不变 | grouped GEMM 与 attention 时间 | 可能还有扩大 tree 的空间 |
| 专家预算很快但质量波动 | logits divergence、任务分桶准确率 | 放宽预算或只在容许近似的流量启用 |
| offload 传输下降但端到端不变 | host gap、expert GEMM、cache miss | 新瓶颈已转移，重新 profile |
| 尾延迟恶化 | 每请求 utility、最慢 EP rank、队列等待 | 隔离低效请求并限制最大验证工作集 |

诊断过程应始终保留 target-only baseline。否则系统负载变化、batch 合并或 expert rebalance 也可能被误认为推测解码收益。

## 最终要优化的是“每个已提交 token 的目标成本”

在 dense 模型中，推测解码常被理解为用廉价计算换取更少的目标模型串行轮次。到了 MoE，这句话还要补上一半：每次目标验证的成本本身不是常数，它由候选触发的专家集合、token 分布和硬件拓扑共同决定。

因此，一个成熟的控制器需要同时观察两种信号：

- 模型信号：候选概率、按深度接受率、预测不确定性；
- 系统信号：边际 expert footprint、rank imbalance、通信和验证时间。

只优化前者会产生概率很高但昂贵的候选树，只优化后者又可能为了复用专家而选择几乎不会被接受的路径。真正有效的策略，是让每个新增候选都以足够高的接受收益支付它带来的边际专家成本。

这也给出了最直接的上线标准：在目标分布或明确约定的质量边界内，真实请求上的 committed tokens / target cost 是否提高，P99 是否仍可控。满足这个标准，MoE 与推测解码的组合才从算法可能性变成系统收益。

## 参考资料

- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
- [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318)
- [EAGLE-3](https://arxiv.org/abs/2503.01840)
- [Utility-Driven Speculative Decoding for Mixture-of-Experts（Cascade）](https://arxiv.org/abs/2506.20675)
- [MoE-Spec: Expert Budgeting for Efficient Speculative Decoding](https://arxiv.org/abs/2602.16052)
- [SpecMoE: Self-Assisted Speculative Decoding](https://arxiv.org/abs/2604.10152)
- [Less Experts, Faster Decoding（EcoSpec）](https://arxiv.org/abs/2607.12696)
- [SGLang Expert Parallelism 文档](https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/expert_parallelism.md)
- [SGLang Speculative Decoding 文档](https://github.com/sgl-project/sglang/blob/main/docs_new/docs/advanced_features/speculative_decoding.mdx)
