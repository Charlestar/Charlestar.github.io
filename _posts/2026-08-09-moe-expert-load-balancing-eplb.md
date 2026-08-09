---
layout: post
title: "MoE 负载均衡：从 Router 偏置到 Expert 副本与 EPLB"
subtitle: "分清训练时均衡、推理时重排，以及不改变 Top-k 语义的热点分流"
date: 2026-08-09 16:53:00 +0800
last_modified_at: 2026-08-09
author: iStar
catalog: true
series: moe-communication
series_order: 30
technology_year: 2024
mathjax: true
tags: [AI Infra, MoE, EPLB, 负载均衡, Expert Parallel]
---

MoE 的 router 为每个 token 选择少数 experts。即使长期看每个 expert 获得的 token 数相近，一次在线 batch 仍可能很倾斜；如果某些主题、语言或请求类型反复命中同一组 experts，热点还会持续存在。

在 Expert Parallel 中，这种倾斜会同时影响三件事：热点 rank 接收更多 dispatch 数据、执行更大的 expert GEMM、最后完成 combine。其他 ranks 即使早已完成，也要等待最慢者。因此整层延迟通常更接近：

$$
T_{MoE}\approx\max_r
\left(T_{dispatch,r}+T_{expert,r}+T_{combine,r}\right)
$$

而不是所有 rank 平均耗时。

“MoE 负载均衡”常被当作一个概念，实际至少包含两类完全不同的机制：

1. **训练时路由均衡**：改变 router 的学习信号或选择偏置，让逻辑 experts 获得更均匀的 token；
2. **部署时物理均衡**：不改变模型选中的逻辑 expert，通过复制热点 expert、调整 placement、在等价副本间分流，使 GPU 负载更均匀。

前者会影响模型学到的专家分工，后者是 serving 系统的执行映射。把二者混在一起，容易让 runtime 为了性能擅自改掉 checkpoint 语义，或者误以为训练均衡后线上就不再需要调度。

## 先区分逻辑 expert 与物理 expert

模型 checkpoint 定义 $E$ 个逻辑 experts：

```text
logical experts: e0, e1, e2, ..., e(E-1)
```

Router 输出的是逻辑 expert id。若没有副本，每个逻辑 expert 对应一个物理权重实例：

```text
e0 → GPU0 slot0
e1 → GPU0 slot1
e2 → GPU1 slot0
...
```

EPLB 可以为热点逻辑 expert 建立多个完全相同的物理副本：

```text
logical e5
  ├── physical replica e5#0 on GPU1
  ├── physical replica e5#1 on GPU4
  └── physical replica e5#2 on GPU7
```

Router 仍然选择 `e5`，runtime 再从 `e5#0/#1/#2` 中选择一个执行。只要权重、数值路径和 router weight 一致，这些副本在模型语义上等价；改变的是 token 去哪张 GPU，而不是 token 由哪个逻辑 expert 处理。

因此需要两张映射表：

```text
physical_to_logical[physical_slot] → logical_expert
logical_to_physical[logical_expert] → [replica slots...]
```

DeepSeek 开源 EPLB 的输出正是这两类映射，以及每个逻辑 expert 的副本数量。通信 dispatcher 要消费的是最终 physical destination，但日志和模型语义仍应保留 logical id。

## 三个层级的“不均衡”

只看每个 expert 的平均 token 数不够。至少要在 expert、rank 和 node 三层测量。

### Expert load

设一层本轮共有 $N_a$ 个 assignments，逻辑 expert $e$ 收到 $n_e$：

$$
\bar n_e=\frac{N_a}{E}
$$

$$
I_{expert}=\frac{\max_e n_e}{\bar n_e}
$$

它回答“最热逻辑 expert 是平均值的多少倍”。

### Rank load

rank $r$ 上物理 experts 的总负载为：

$$
L_r=\sum_{p\in\mathcal P_r}\ell_p
$$

其中物理副本 $p$ 的负载为 $\ell_p$。Rank imbalance：

$$
I_{rank}
=\frac{\max_r L_r}{N_a/P_e}
$$

两个中等热点 experts 若放在同一 GPU，expert 指标可能不夸张，rank 却成为明显 straggler。

### Node load 与跨节点 traffic

节点 $m$ 的总负载：

$$
L_m=\sum_{r\in m}L_r
$$

节点间均衡不只影响计算，还影响 RDMA 入流量。如果一个 expert group 的所有副本都在同一节点，选择该组的 token 会集中穿过网络。

三个指标需要逐层统计，因为不同 MoE layers 的专家分工不同。把所有层合成一个平均值，会掩盖少数关键层的热点。

## 训练时辅助损失在做什么

经典 Top-$k$ routing 只优化主任务，router 可能把大量 token 集中到少数 experts。训练系统常增加 auxiliary load-balancing loss，让“路由概率”和“实际选择频率”更均匀。

一种抽象写法是：对 expert $e$，设 $f_e$ 为被选择的 token 比例，$q_e$ 为平均 routing probability，则：

$$
L_{balance}
=\alpha E\sum_{e=1}^{E}f_eq_e
$$

具体公式因 Switch Transformer、GShard、模型实现而不同，但目的相似：给热点路由增加代价。

问题在于 auxiliary loss 的梯度会与语言建模主目标共同更新 router。系数太小，均衡效果不够；系数太大，模型可能为了均匀而牺牲最合适的专家选择，形成所谓 interference gradient。

这属于训练时模型设计。Serving runtime 只能执行 checkpoint 已经学到的 router，不能在推理阶段临时增加 loss 修复热点。

## Auxiliary-loss-free balancing 的思路

DeepSeek-V3 使用了一种不把全局均衡项直接加入主训练损失的策略。核心是在 Top-$k$ 选择前，为每个 expert 的 affinity score 加一个动态 bias：

$$
s'_e(x)=s_e(x)+b_e
$$

Top-$k$ 使用 $s'_e$ 决定选谁；用于合并 expert output 的 gating weight 仍基于原始 affinity score 的模型定义。训练过程中根据近期 expert load 更新 $b_e$：

- expert 过载，降低其 bias；
- expert 欠载，提高其 bias。

可以把一个简化更新写成：

$$
b_e^{(t+1)}
=b_e^{(t)}+\eta\,\operatorname{sign}(\bar n-n_e)
$$

实际实现的统计窗口、更新规则和 group 限制应以模型论文/代码为准。关键区别是：均衡信号通过选择偏置调节，而不是让辅助损失梯度直接干扰主任务表示学习。

“loss-free”不等于完全没有其他辅助约束。DeepSeek-V3 技术报告还保留了较小的 sequence-wise balance loss，用来避免单个 sequence 内的极端不均衡。名称描述的是核心全局策略，而不是所有均衡项都消失。

训练完成后，bias 已是 checkpoint/router 语义的一部分。推理不能为了当前 GPU 负载随意修改它，否则逻辑 expert 选择会改变，输出质量也可能变化。

## 为什么训练均衡仍解决不了线上倾斜

训练数据分布与线上请求不完全相同。即使训练全局统计均匀，serving 仍会出现：

- 某个租户集中发送一种语言或领域；
- 白天和夜间的 request mix 不同；
- Prefill 大 batch 与 Decode 小 batch 方差不同；
- continuous batching 把相关请求聚到同一轮；
- 单个长 prompt 在一层产生大量相似 routing；
- 小 batch 的随机波动让某 rank 瞬时过热；
- expert placement 把多个次热点放在同一 GPU。

训练均衡优化的是模型分工的长期统计，EPLB 优化的是特定部署和当前 workload 下的物理执行。两者互补，不是替代关系。

## EPLB 的输入不是 Router logits，而是负载估计

DeepSeek EPLB 的核心函数接收形如：

```text
weight[layer, logical_expert]
```

的负载矩阵。`weight[l,e]` 表示第 $l$ 层逻辑 expert $e$ 的预计工作量。仓库明确说明“如何预测 expert load”不在其范围内，常见输入是历史统计的 moving average。

一个基础 EMA 为：

$$
\hat n_e^{(t)}
=\beta\hat n_e^{(t-1)}+(1-\beta)n_e^{(t)}
$$

$\beta$ 大，placement 稳定但追踪热点慢；$\beta$ 小，响应快却容易随 batch 噪声反复迁移。

负载不一定只用 token count。若不同 experts 的 GEMM kernel、量化格式或命中行为不同，可以估算实际 service time：

$$
w_e
=a\cdot n_e+b\cdot \text{bytes}_e+c\cdot T_{gemm,e}
$$

但官方 EPLB 简化算法以给定 weight 为基础，并不自动学习这个成本模型。接入系统需要先证明 token count 与 layer/rank time 的相关性。

## 怎样给热点 expert 分配副本

设逻辑 experts 数为 $E$，可用物理 slots 数为 $E+R$，其中 $R$ 是 redundant experts 数。每个逻辑 expert 至少有一个副本，额外 $R$ 个 slots 分给热点。

若 expert $e$ 有 $c_e$ 个副本，并能理想均匀分流，它的单副本负载近似：

$$
\ell_e=\frac{w_e}{c_e}
$$

目标是选择整数 $c_e\ge1$，满足：

$$
\sum_e c_e=E+R
$$

并尽量降低最大单副本负载：

$$
\min \max_e\frac{w_e}{c_e}
$$

EPLB 使用一个直观的贪心过程：每次把新副本分给当前 `weight / replica_count` 最大的逻辑 expert。伪代码为：

```text
replica_count[e] = 1 for every logical expert

repeat R times:
    e = argmax(weight[e] / replica_count[e])
    replica_count[e] += 1
```

### 一个例子

4 个逻辑 experts 的历史负载为：

```text
e0=100, e1=55, e2=25, e3=20
```

有两个冗余 slots：

1. 第一个副本给 e0，估算单副本负载从 100 降到 50；
2. 此时 e1=55 最大，第二个副本给 e1，降到 27.5。

最终：

```text
e0 replicas=2 → ~50 each
e1 replicas=2 → ~27.5 each
e2 replicas=1 → 25
e3 replicas=1 → 20
```

若第二个副本继续给 e0，其单副本约 33.3，但 e1 仍为 55，最大负载更差。这个例子说明“只复制最热 expert”不一定使全局最大值最小。

## 有副本后，Token 去哪个物理 slot

复制权重只是第一步。Router 仍输出 logical expert，runtime 必须在其合法副本中选一个 physical destination。

最简单的静态均匀分流可以使用 round-robin 或 hash：

```text
replicas = logical_to_physical[e]
p = replicas[hash(request_id, token_position, layer_id) % len(replicas)]
```

这种方法无需跨 rank 实时交换队列长度，但只能实现统计均匀。动态选择可以考虑：

- 当前 batch 已分给每个副本的 assignment 数；
- GPU queue、expected GEMM time；
- source 与 replica 是否同节点；
- 通信 buffer 和 NIC 拥塞；
- 副本是否 ready。

选择过程本身不能比要节省的 latency 更贵。Decode 小 batch 若为了每个 token 运行一次全局调度或同步所有 rank load，可能得不偿失。

还要保持确定性映射的边界：同一 assignment 只能发给一个副本；combine 仍把结果映射回原 logical expert/Top-$k$ slot。副本不是让一个 token 重复算多次再投票。

## 把物理 experts 装箱到 GPUs

得到副本集合后，还要把 physical experts 放入固定容量的 GPU slots。设每张 GPU 必须保存 $S$ 个 experts，EPLB 的 `balanced_packing` 按负载从大到小处理，并把每个 expert 放入当前总负载最小、且仍有 slot 的 GPU。

可以理解为带等数量约束的 greedy bin packing：

```text
sort physical experts by estimated load descending

for expert in sorted list:
    choose GPU with:
      - available expert slot
      - minimum current total load
    place expert
    add its estimated load to GPU total
```

严格最优的装箱是组合优化问题，贪心不保证全局最优，但计算快、结果可解释，适合周期性重平衡。评估时应比较重排前后的预测最大 rank load，并用真实 trace replay 验证，而不是因为算法名含 “balancer” 就默认一定改善。

## Hierarchical policy：先平衡节点，再平衡 GPU

DeepSeek-V3 使用 group-limited routing：experts 被划为 groups，一个 token 只从有限 groups 中选择 experts。这给 topology-aware placement 一个机会。

当 expert group 数能被节点数整除时，EPLB 的 hierarchical policy 按三步处理：

1. 按 group 总负载，把 expert groups 均衡装到 nodes；
2. 在每个 node 内，为热点 experts 建立冗余副本；
3. 把 node 内的物理 experts 均衡装到各 GPUs。

```text
logical expert groups
    │ balance group load
    ▼
nodes
    │ replicate hot experts inside node
    ▼
physical expert replicas
    │ balance replica load
    ▼
GPUs inside each node
```

好处是同组 experts 尽量留在同一节点，配合 group-limited routing 减少跨节点 fan-out。代价是副本只能在较小的节点域内分配，某节点特别热时不能充分借用其他节点的空闲 GPU。

官方 EPLB 说明该策略可用于 EP size 较小的 Prefill 阶段。

## Global policy：忽略 group 边界，全局平衡副本

当 group/node 条件不适用时，EPLB 使用 global policy：把所有 experts 视为一个全局集合，跨全部 GPUs 复制与装箱。

它有更大的均衡自由度，适合 EP size 较大的 Decode 部署，但可能增加跨节点 traffic。是否更快取决于：

$$
\text{straggler reduction}
\quad\text{vs}\quad
\text{additional network cost}
$$

不能只比较 rank token count。一个 plan 把最大 rank load 降低 20%，若同时让大量 assignment 从 NVLink 变成 RDMA，端到端 TPOT 可能恶化。

因此候选 placement 的评分应包含拓扑成本，例如：

$$
J
=\lambda_1\max_r L_r
+\lambda_2 V_{inter-node}
+\lambda_3 M_{migration}
$$

其中 $M_{migration}$ 是从当前 plan 切换所需移动/加载的 expert 权重。权重系数来自实际硬件 profile，而不是固定经验值。

## Prefill 与 Decode 为什么可能用不同 Plan

DeepSeek 官方 EPLB 把 hierarchical policy 与 Prefill、小 EP，把 global policy 与 Decode、大 EP 联系起来。原因可以从 workload 推导。

### Prefill

- 每轮 token 多，专家负载统计相对稳定；
- expert GEMM 较大；
- 大 payload 的跨节点带宽成本明显；
- 较小 EP domain 可以在节点组内形成高吞吐。

Hierarchical placement 能限制跨节点流量，同时在节点内平衡大 batch。

### Decode

- 每轮 token 少，瞬时方差大；
- expert GEMM 小，straggler 对 TPOT 敏感；
- 模型总权重可能需要更大的 EP domain；
- 更全局的副本池提供更高分流自由度。

P/D 解耦时，两边可以使用不同 physical mapping；但权重管理、路由 metadata、KV 交接与监控必须明确当前 pool 的 plan version，不能把 Prefill 的 physical id 直接当成 Decode 的 physical id。跨阶段传递 logical expert 语义即可，KV 本身也不应依赖某个 expert physical slot。

## No Token Dropping 与负载均衡的关系

一些 MoE 训练实现为每个 expert 设置容量，超出 capacity 的 assignments 被丢弃或重新路由。推理时 dropping 会改变模型计算，可能影响输出质量。

EPLB 通过副本和 placement 分摊热点，但不能保证任何突发 batch 都不会超过 buffer/compute 上界。若目标是 dropless inference，还需要：

- 动态或足够大的 dispatch buffer；
- 对极端热点进行 chunked expert execution；
- 不把 padding capacity 当作真实丢弃策略；
- 在 admission control 层限制无法承载的总 token；
- overflow 时明确 fallback/recompute，而不是静默删除 assignment。

“有 EPLB”不自动等于“永不丢 token”。正确性承诺来自整条 runtime 数据路径。

## 负载统计窗口怎样选

EPLB 仓库不负责预测 load，实际系统需要决定采样、聚合与更新时间。

### 过短窗口

直接用最近一个 batch：

- 对热点响应快；
- 容易追逐随机噪声；
- plan 频繁变化；
- 权重迁移与缓存失效可能大于收益。

### 过长窗口

使用数小时平均：

- placement 稳定；
- 无法追踪租户/时段变化；
- 切流后旧统计污染新 workload；
- 热点已造成 SLO 下降，plan 仍迟迟不变。

可以使用多时间尺度：

```text
fast EMA   → detect sudden skew / alert
slow EMA   → generate stable placement candidate
trace replay → validate candidate before rollout
```

再加入 hysteresis：只有候选 plan 的预测收益超过阈值并持续若干窗口，才触发重排。切换后设置 cooldown，避免在两个近似 plan 间振荡。

负载还应按 Prefill/Decode、模型 revision、Adapter、租户或 request class 分层。混在一起的平均值可能不代表任何实际 worker pool。

## 重排不是更新一张表这么简单

新的 `physical_to_logical` plan 意味着某些 GPU slot 要换成另一份 expert 权重。安全 rollout 可以分为：

```text
1. generate candidate plan from stable load statistics
2. replay trace / estimate rank and network cost
3. load new or changed expert replicas into spare/staging memory
4. verify checksum, dtype and model revision
5. publish plan generation N+1 at a batch/layer-safe boundary
6. route new assignments using N+1
7. wait for plan N in-flight operations to finish
8. release old replicas and buffers
```

若显存没有 staging 空间，可以 drain 整个 worker group 后重载，但切换成本更高。不能在某些 ranks 已使用新 mapping、另一些仍使用旧 mapping 时开始 All-to-All；同一个 logical id 会被解释为不同 physical destination。

所有 dispatch handle、日志与 profile 应携带：

```text
model_revision
layer_id
placement_generation
physical_to_logical checksum
```

失败回滚也以完整 EP group 为单位。旧 plan 至少保留到所有使用它的 in-flight MoE operations 完成。

## 副本的显存成本

冗余 expert 是用显存换负载均衡。一个 gated FFN expert 常包含 gate/up/down matrices。粗略权重大小为：

$$
M_{expert}
\approx (2H I + I H)\times b_w
=3HIb_w
$$

其中 $H$ 是 hidden size，$I$ 是 expert intermediate size，$b_w$ 是每权重字节数；量化 scale、alignment 和 runtime layout 还会增加开销。

$R$ 个冗余副本约增加：

$$
M_{redundant}\approx R\times M_{expert}
$$

这些显存原本可能用于 KV Cache、larger batch 或通信 buffer。vLLM 的 Expert Parallel 部署文档也提醒，开启 redundant experts 会有明确的额外显存成本；具体数值与模型和精度绑定，不能跨模型照搬。

因此要用 goodput 评估：副本减少的 straggler 是否足以抵消 KV 容量下降或可并发请求数减少。

## 实时动态均衡的边界

静态/周期 EPLB 处理的是相对稳定的专家热点。小 Decode batch 还有很强的瞬时波动：即使历史完全均衡，本轮也可能随机把很多 token 分到同一 rank。

更动态的方法可以在有多个合法副本时，根据本轮 load 实时选择；DeepSeek 的 LPLB 仓库还探索使用线性规划在既定副本拓扑上重分配 token flow。但官方明确将其标为早期研究阶段，并列出：

- solver 延迟对小 batch 可能不可忽略；
- 当前目标主要按 token count，不完全代表 grouped GEMM 非线性耗时；
- 极端全局倾斜下不一定优于 EPLB；
- 需要额外的实时负载同步与通信。

所以动态优化不应替换掉稳定基线。更合理的层级是：

```text
training router balance
  → periodic expert replication/placement (EPLB)
  → cheap per-batch replica selection
  → experimental optimizer for residual skew
```

每层只解决前一层剩余的问题，并接受严格的调度开销预算。

## 正确性验证

EPLB 改变执行位置，不应改变逻辑输出。验证需要覆盖：

### 权重一致性

同一 logical expert 的所有 replicas 必须来自相同 model revision、dtype 和量化参数，checksum 一致。Adapter 或 expert-specific delta 也必须一同复制。

### 映射互逆

对每个 physical slot：

```text
logical = physical_to_logical[p]
assert p in logical_to_physical[logical]
```

每个 logical expert 至少有一个 ready replica。

### Assignment 守恒

切换 plan 前后，logical Top-$k$ ids 与 router weights 完全相同；每个 assignment 恰好选择一个 physical replica。

### 数值对照

固定输入分别在无副本 baseline、旧 plan、新 plan 上执行，逐层比较 MoE output。若 kernel 的浮点归约顺序不同，应使用有依据的误差阈值，并另行评估最终生成差异。

### 极端与故障场景

- 所有 token 选择同一个逻辑 expert；
- 某 replica 在切换期间不 ready；
- plan event 延迟到达；
- 一个 rank 加载失败；
- plan generation 回滚；
- buffer capacity 小于瞬时热点；
- worker group 在 in-flight combine 时重启。

任何 rank 无法采用新 plan 时，整组都不能发布该 generation。

## 应该监控哪些指标

### Router 层

```text
logical assignments per expert/layer
expert max-to-mean and entropy
Top-k group distribution
Prefill vs Decode routing histogram
```

### Physical placement 层

```text
physical assignments per replica
rank/node max-to-mean
replica split ratio
local vs cross-node assignments
predicted vs observed rank time
```

### Rebalance 生命周期

```text
plan generation and checksum
candidate predicted gain
expert bytes moved
load/warmup duration
rollout success/failure
time since last rebalance
```

### Serving 结果

```text
dispatch / expert GEMM / combine P50-P99
TPOT and TTFT SLO attainment
KV capacity lost to replicas
goodput before/after
```

特别值得看“预测最大 rank load”与“实际最慢 rank time”的偏差。如果长期不相关，说明 token count 不是足够好的成本模型，或瓶颈其实在网络、kernel shape、NUMA/NIC 拓扑。

## 评估一个 EPLB Plan

可以使用最近一段真实 routing trace 做离线 replay：

1. 按 layer 统计 logical expert load；
2. 用训练窗口生成 candidate replica/placement；
3. 在独立验证窗口上把 assignments 映射到 physical replicas；
4. 计算每轮 rank/node load 与跨节点 bytes；
5. 用 profile 得到的 GEMM/communication cost 估算 layer time；
6. 与当前 plan 比较 P50、P95、P99，而非只看平均；
7. 扣除副本显存、权重加载和切换成本；
8. 通过后再小流量 rollout，观察真实 SLO。

避免用同一段 trace 同时拟合和验证，否则 candidate 会过度适配历史热点。还要在 request mix 改变、流量突发和均匀路由三种场景下验证，确保 plan 不只在单一窗口有效。

一个可执行的接受条件可以是：

```text
P99 rank-load imbalance decreases by target
P99 MoE layer time does not regress
cross-node bytes remain within budget
KV capacity reduction is acceptable
no output correctness regression
rebalance completes within operational window
```

## 实施顺序

### 第一阶段：测清楚倾斜

- 记录逐层 logical expert counts；
- 聚合到 rank/node 并关联 layer time；
- 分开 Prefill、Decode 与主要 workload；
- 验证热点是持续的还是 batch 噪声。

### 第二阶段：只重排，不增加副本

- 保持每个 logical expert 一个 physical instance；
- 把多个热点拆到不同 GPUs；
- 验证 mapping generation 与安全 rollout；
- 量化节点间 traffic 变化。

### 第三阶段：加入少量冗余 experts

- 从最热 logical experts 开始；
- 用真实显存 layout 计算副本成本；
- 建立副本选择和 assignment 守恒检查；
- 比较 goodput，而不是只看均衡比。

### 第四阶段：周期与拓扑优化

- EMA、hysteresis 与 cooldown；
- hierarchical/global plan 对比；
- Prefill/Decode 独立 plan；
- trace replay 和自动 rollback。

### 第五阶段：评估动态算法

- 先衡量剩余 per-batch skew；
- 给 planner 设置严格微秒级预算；
- 计入实时同步开销；
- 保留静态 EPLB fallback。

## 小结

MoE 负载均衡不是一个开关，而是从模型训练到物理部署的多层控制系统。

可以用五条边界把它理清：

1. 训练 auxiliary loss 或 dynamic bias 决定逻辑 experts 怎样被选中；
2. EPLB 不改变 Top-$k$ 逻辑选择，只复制热点 experts 并调整物理 placement；
3. Expert 均衡不等于 rank/node 均衡，必须把多个 experts 的负载与网络拓扑合并观察；
4. 副本用权重显存换 straggler 改善，最终应以 TTFT/TPOT goodput 衡量；
5. Rebalance 是带 generation 的分布式 rollout，不是原地改一张索引表。

当 logical/physical 两层被清楚分开，训练策略、EPLB、DeepEP 就能各司其职：模型决定哪些 experts 合适，EPLB 决定等价副本放在哪里，DeepEP 负责 token 怎样高效到达那里。再向更大集群扩展时，问题就变成 Wide EP：当一个 EP group 横跨数十乃至数百 GPU，怎样控制 fan-out、层次化通信、并行分组和全局负载波动。

## 参考资料

- [DeepSeek EPLB 官方仓库](https://github.com/deepseek-ai/EPLB)
- [EPLB 核心算法实现](https://github.com/deepseek-ai/EPLB/blob/main/eplb.py)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- [Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts](https://arxiv.org/abs/2408.15664)
- [DeepSeek LPLB 官方仓库](https://github.com/deepseek-ai/LPLB)
- [vLLM: Expert Parallel Deployment](https://docs.vllm.ai/en/stable/serving/expert_parallel_deployment/)
