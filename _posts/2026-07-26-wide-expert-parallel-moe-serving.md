---
layout: post
title: "Wide EP：当 MoE Expert Parallel 横跨几十张 GPU"
subtitle: "从 Attention DP、跨节点 All-to-All 到 P/D 两套并行布局"
date: 2026-07-26 09:00:00 +0800
last_modified_at: 2026-09-03
author: iStar
catalog: true
series: moe-communication
series_order: 50
technology_year: 2025
mathjax: true
tags: [MoE, 专家并行, 分布式推理]
---

常规 Expert Parallel 可能只在单台 8-GPU 节点内分配 experts。随着 MoE 总参数、expert 数和在线吞吐增长，EP group 会扩展到多个节点、几十甚至上百张 GPU。这里把这种大规模、跨节点的 Expert Parallel 部署称为 **Wide EP**。

Wide EP 不是简单把 `expert_parallel_size` 从 8 改成 64。EP domain 变宽后，一个模型实例内部同时出现两种看似矛盾的执行方式：

- Attention、shared expert 和 KV Cache 可以按请求做 Data Parallel，每张 GPU 只处理自己的 sequences；
- Routed experts 却分散在整个 EP group，任意 GPU 上的 token 都可能跨节点访问远端 expert。

于是每个 MoE layer 都经历一次“从 Attention DP 布局展开到全局 EP，再折回原 DP 布局”的过程。网络 fan-out、每 rank token 数、expert 副本、P/D 两阶段并行度和故障域都要一起设计。

## 为什么要把 EP 做宽

先从显存看。模型有 $E$ 个逻辑 routed experts，额外复制 $R$ 个冗余 experts，EP size 为 $P_e$。均匀放置时每 rank 保存：

$$
E_{local}=\frac{E+R}{P_e}
$$

个物理 experts。扩大 $P_e$ 可以让每张 GPU 只保存少量 expert 权重，为 KV Cache、通信 buffer 和更大 batch 留出空间。

但 Wide EP 的价值不只在容量。每个 logical expert 可以从整个 EP group 聚合 token。假设每 rank 进入一层的 token 数为 $N_r$，Top-$k$ 为 $k$，全局 assignments 为：

$$
N_a=k\sum_{r=1}^{P_e}N_r
$$

路由近似均匀时，单 logical expert 的平均 batch 为：

$$
\bar n_e=\frac{N_a}{E}
$$

EP group 扩大、同时有足够请求填充各 ranks 时，$N_a$ 增长，单 expert 的 GEMM $m$ 维也能增大。对于 fine-grained MoE，单 expert 矩阵较窄，只有聚合足够多 token 才能充分利用 GPU。

DeepSeek 官方推理系统说明把这点概括为：大规模 EP 扩大整体 batch，使每个 expert 获得更有效的矩阵计算形状，同时每张 GPU 只需访问少量 expert weights。

这里有一个必要前提：**流量必须足以填满更宽的 group。** 若总 token 数不变，仅把它摊到更多 ranks，每 rank 的 Attention 和通信工作变小，固定同步成本反而上升。

## 一层里有两套并行布局

以 Attention TP=1、Attention DP=$P_e$ 为例。每个 rank 保存 Attention/shared 权重副本、自己 sequences 的 KV Cache，以及一部分 routed experts。

```text
Before MoE layer: Attention-DP layout

rank 0: tokens A, B + KV(A,B)
rank 1: tokens C, D + KV(C,D)
rank 2: tokens E, F + KV(E,F)
...

Router chooses logical experts
          │
          ▼
Global EP dispatch across all ranks
          │
          ▼
Each rank executes its local expert slots
          │
          ▼
Global EP combine
          │
          ▼
Back to Attention-DP layout
```

Attention DP rank 是 token 的 source owner：它维护请求、位置、KV 与 residual stream。Expert rank 只是这一层某些 FFN 的临时执行位置。Combine 结束后，expert output 必须回 source owner，下一层 Attention 不会跟着 token 永久迁移。

这种布局将不同权重类型区别对待：

| 模块 | 常见并行方式 | 主要状态/通信 |
| --- | --- | --- |
| Attention / MLA | DP，必要时叠加 TP/CP | 每请求 KV，本地 Attention 或 TP collective |
| Shared expert | DP，必要时 TP | 每 token 都执行，可与 routed dispatch overlap |
| Routed experts | Wide EP，必要时 ETP | dispatch/combine All-to-All |
| Embedding / LM head | DP 或 TP | logits 与 sampling 路径 |

这也解释了为什么 Wide EP 不等同于“整个模型做 64-way TP”。Routed experts 可以跨 64 ranks，Attention 仍在许多较小 TP groups 或纯 DP ranks 中独立运行。

## vLLM 中 EP=TP×DP 的含义

vLLM 当前 Expert Parallel 部署文档采用一个易理解的 folding：

$$
P_e=P_t\times P_d
$$

其中 $P_t$ 是 Tensor Parallel size，$P_d$ 是 Data Parallel size。启用 EP 后：

- expert layers 跨全部 $P_e$ ranks 分片；
- Attention 在每个 DP replica 内使用 $P_t$-way TP；
- 当 $P_t=1$ 时，Attention weights 在各 DP ranks 完全复制。

例如 TP=2、DP=4 的 8 GPU 部署：

```text
Attention groups:
  DP0: ranks [0,1] TP=2
  DP1: ranks [2,3] TP=2
  DP2: ranks [4,5] TP=2
  DP3: ranks [6,7] TP=2

Routed expert group:
  EP: ranks [0..7] EP=8
```

这种映射让不同 DP replicas 的 token 在 MoE layer 共享全部 expert shards。它是具体 runtime 的当前设计，不代表所有系统必须使用同一公式；但它很好地展示了 Wide EP 的本质：Attention 的数据并行维度可以被“折入”MoE 的 expert parallel domain。

## Parallel Folding 为什么重要

传统 group generator 往往要求 EP 是某个 DP group 的子集，使 Attention 与 MoE 被迫共享相近的 TP/CP/DP 关系。这样会出现：

- Attention 需要高 TP 才能放下权重，expert 却被切成很小的 ETP GEMM；
- 长上下文需要 CP，MoE FFN 对 sequence 切分本身并无特殊需求；
- EP 受 Attention DP size 限制，无法独立扩大；
- 为了同时开启 CP=8 与 EP=8，物理 world size 被不必要地相乘。

Megatron Core 的 MoE Parallel Folding 为 Attention 和 MoE 分别建立逻辑维度：

```text
Attention: TP × CP × DP × PP
MoE:       ETP × EP × EDP × PP
```

两组 mapping 共享同一批物理 ranks，却按层解释成不同 process groups。官方示例说明，传统映射下 EP 受 DP 约束，而 folding 可以让 Attention 保持 TP=4、CP=2、DP=8，MoE 使用 ETP=1、EP=64、EDP=1。

Folding 没有消灭 layout conversion，而是让转换变得显式并可优化。设计时要给每个 rank 列出：

```text
global rank
node / local GPU
attention TP group
attention DP group
context group
expert EP group
expert ETP group
pipeline stage
```

只写一组并行数字，无法判断 collective 是否落在预期的 NVLink 或 RDMA domain。

## DeepSeek-V3/R1 的两个具体布局

DeepSeek 2025 年公开的在线推理系统给出了一个清晰案例。模型每层有 256 个 routed experts，每 token 激活 8 个，并配置 32 个冗余 routed expert 副本，总物理 slots 为：

$$
256+32=288
$$

### Prefill

官方配置使用 Routed Expert EP32，跨 4 个节点：

$$
\frac{288}{32}=9
$$

即每 GPU 9 个 routed experts；MLA/shared expert 使用 DP32。

### Decode

官方配置使用 Routed Expert EP144，跨 18 个节点：

$$
\frac{288}{144}=2
$$

即每 GPU 2 个 routed experts；MLA/shared expert 使用 DP144。

这不是所有 DeepSeek 部署或所有硬件的推荐值，而是官方披露系统在特定 H800 集群、模型与流量下的工程选择。它说明两个通用结论：

1. P/D 解耦后，Prefill 与 Decode 可以拥有完全不同的 EP width；
2. EP 大于 logical expert 数的一半时，冗余副本和每 rank 固定 slot 数会共同决定布局。

不能只复制这组数字。需要用本地网络、batch、权重精度、KV 容量与 SLO 重新推导。

## Prefill 为什么不一定用最宽 EP

Prefill 一次处理大量输入 token，计算以 Attention 和较大 expert GEMM 为主。扩大 EP 有利于聚合 token，但跨节点发送的是每个输入 token 的 hidden state，payload 也很大。

Prefill EP width 的取舍可以抽象为：

$$
T_{prefill}
\approx T_{attention}(N/P_d)
+T_{dispatch}(N,k,H,P_e)
+T_{expert}(N_a/E)
+T_{combine}
$$

过窄：

- 每 GPU expert 权重多；
- 可用于 activation/buffer 的显存少；
- 单 GPU expert work 可能过大；
- 整体 batch 聚合能力受限。

过宽：

- 跨节点 peers 与同步增加；
- 每 rank Attention token 太少；
- 大 payload 更易受 NIC 带宽限制；
- straggler 与故障域扩大。

因此 Prefill 常选择中等 EP width，使用较大 chunk/microbatch 追求吞吐，并通过双 batch overlap 隐藏通信。

## Decode 为什么可能使用更宽 EP

Decode 每个 sequence 每步只产生一个 token，Attention 主要读取 KV，expert GEMM 容易因 batch 太小而低效。扩大 DP/EP domain 可以汇集更多并发 sequences 的 assignments，让每个 expert 获得足够 batch，同时每 GPU 只保存少量 expert weights。

但 Decode 对 TPOT 敏感，小消息固定延迟和最慢 rank 更突出。只有在足够高并发下，更大 expert batch 的收益才可能覆盖跨节点 RTT。

Wide Decode EP 的容量单位不是单请求，而是同时活跃的 sequence 数。可以估算：若目标是每个 logical expert 平均至少收到 $m_{target}$ 个 assignments，一轮需要：

$$
N_{active}
\gtrsim \frac{m_{target}E}{k}
$$

例如 $E=256$、Top-8，希望每 expert 平均 16 rows，则全局一轮约需：

$$
N_{active}\gtrsim\frac{16\times256}{8}=512
$$

这只是均匀路由下的平均值；实际还要处理 expert/rank 倾斜。若在线并发长期低于这个量级，EP144 之类宽度可能无法形成理想 GEMM，应该缩小实例或合并流量，而不是维持空转的大 group。

## Wide EP 的通信矩阵怎样扩张

EP size 为 $P_e$ 时，dispatch 可以表示成 $P_e\times P_e$ 的 count matrix $C$：

$$
C_{ij}=\text{source rank }i
\text{ 发给 destination rank }j\text{ 的 assignments}
$$

随着 $P_e$ 增大：

- matrix 的潜在非零 peer 数增加；
- 每对 peer 的消息可能变小；
- count/notification 状态增多；
- 跨节点流量成为主导；
- 任一 destination 热点影响更多 sources。

若 experts 均匀分布，assignment 在本地 GPU 的概率约 $1/P_e$。一个 8-GPU 节点在 $P_e$ ranks 中所占比例为 $8/P_e$，因此节点内命中比例也会随 EP 变宽下降，除非 group-limited routing 或副本 placement 提高 locality。

若按每个 assignment 独立展开 hidden state，跨节点的端到端 hidden-state 逻辑字节数近似为：

$$
V_{inter}
\approx N_tkHb
\left(1-\frac{G_{local}}{P_e}\right)
$$

其中 $G_{local}$ 是同节点 EP ranks 数，$b$ 是 dispatch dtype bytes。Combine 再产生一轮返回 traffic。这个公式还忽略同一 token 的多个 experts 落到同一远端 rank/转发域时可复用 hidden-state payload，也没有计入元数据、对齐与分层转发，因此不能替代实际 NIC bytes；它只是 assignment-expanded 口径，说明 Wide EP 为什么必须把网络放进第一层容量模型。

## Flat All-to-All 很快遇到 peer scale 问题

假设 18 节点、每节点 8 GPUs 的 EP144。若每张 GPU 直接与所有远端 GPUs 建立细粒度交换，会带来：

- 大量 peer/QP/notification 状态；
- 许多小消息与 RDMA doorbell；
- NIC rail 负载不均；
- 交换机 incast；
- 更高的同步和故障暴露面。

分层通信把节点内 scale-up 和节点间 scale-out 分开：

```text
source GPU assignments
  → node-local permute / aggregation
  → selected NIC rail / RDMA path
  → destination node local scatter
  → destination expert slots
```

是否先聚合、使用 direct GPU-to-GPU、怎样分 rail，取决于 DeepEP/HybridEP 等 backend 与硬件。通用原则是：

1. 节点内尽量利用 NVLink/NVSwitch；
2. 跨节点合并过小 payload，控制 peer fan-out；
3. rank-to-NIC 绑定匹配 NUMA/PCIe 拓扑；
4. EP traffic 与 KV transfer、存储等 traffic 做隔离或优先级管理；
5. 用真实 NIC counters 验证，而不是只看 logical bandwidth。

## Group-limited routing 限制跨节点 fan-out

若模型把 experts 划为 $G$ 个 groups，并规定每 token 只从其中 $g$ 个 groups 选择 experts，runtime 可以把 group 与节点/节点集合对齐。

```text
router selects 2 of 8 expert groups
  → token only needs to reach nodes owning those groups
  → select Top-k experts inside allowed groups
```

这能减少每 token 的跨节点目的集合，也让 EPLB hierarchical placement 有明确边界。DeepSeek-V3 的 EPLB 会尽量把同组 experts 放在同一节点。

但 group-limited routing 是模型架构。Serving runtime 不能为了网络 locality 临时减少合法 groups，也不能把一个未被 router 选中的近端 expert 替换远端 expert。可调整的是同一 logical expert 的 replica，以及 checkpoint 已允许范围内的 physical placement。

## 三种负载必须同时平衡

Wide EP 中，一个 rank 可能在 Attention 阶段是 DP owner，在 expert 阶段又是许多远端 token 的 destination。因此有三类负载：

### Attention compute load

Prefill 主要与输入 token 数和长度有关；Decode 主要与活跃请求、KV 长度和 attention 实现有关。

### Dispatch send load

Source rank 的发送量近似与本地 token assignments 成正比：

$$
S_r\approx N_rkHb
$$

一个 DP rank 分到更多/更长 Prefill 请求，会同时增加 Attention 计算和 dispatch send。

### Expert receive load

由全局 router 与 physical placement 决定：

$$
R_r=\sum_{i}C_{ir}
$$

它与本 rank 自己拥有多少请求没有直接一一对应。

DeepSeek 官方系统分别描述了 Prefill load balancer、Decode load balancer 和 Expert-Parallel load balancer：Prefill 关注 Attention 计算与输入 token，Decode 关注 KV Cache/请求数，EPLB 则最小化最大 dispatch receive/expert load。

只做请求级 round-robin 不能平衡 experts；只做 EPLB 也不能修复某个 Attention DP rank 堆积大量长上下文请求。

## Dual Batch Overlap 怎样隐藏跨节点通信

Wide EP 的 dispatch/combine 很难完全消除，只能寻找独立计算覆盖它。Prefill 可以把 batch 切为两个 microbatches：

```text
time ─────────────────────────────────────────►

microbatch A: dispatch ─ expert ─ combine
microbatch B:      attention ─ dispatch ─ expert ─ combine
microbatch A next:          attention ─ dispatch ─ ...
```

更准确的调度会把 A 的 EP communication 与 B 的独立 Attention/shared expert 计算交错。DeepSeek 公开系统称其为 dual-batch overlap；Decode 因各阶段时长不均，还把 Attention 拆分并形成更细的多阶段 pipeline。

Overlap 的可行条件：

- microbatches 之间没有数据依赖；
- 通信 kernel 不占满计算需要的 SM；
- 两批的 buffer 和 dispatch handle 不互相覆盖；
- HBM、NVLink、NIC 不发生更严重争用；
- 增加的 pipeline depth 不破坏 TPOT/公平性。

评测应比较完整 iteration/layer wall time，以及 overlap 前后单独 GEMM 是否变慢。看到 timeline 上两个彩色区块重叠，不代表被隐藏的时间等于二者交集。

## Expert 副本在 Wide EP 中还有整除作用

冗余 experts 通常用于复制热点，也能让物理 slot 数适配 EP width。以上面的 256 experts、32 replicas 为例：

```text
physical slots = 288

EP32:  9 slots / rank
EP144: 2 slots / rank
```

若没有 32 个副本，256 无法均匀分到 144 ranks。可以允许 uneven slots，但会让权重显存和最大计算容量先天不均；也可以选择其他 EP size。加入 replicas 同时满足热点分流与规则布局，但显存成本要按全部 layers 计算。

副本数量越多，并不保证更平衡。还需要：

- 依据真实 workload 选择复制哪些 logical experts；
- 在副本间分发同一 logical assignment；
- 让 placement 兼顾节点 locality；
- 周期更新计划且安全迁移权重；
- 确保每个副本 model revision 一致。

Wide EP 会放大 placement 更新成本，因为一次 plan 可能涉及多节点、多层的大量 expert bytes。

## KV Cache 为什么仍然是本地 DP 状态

Routed expert FFN 不产生随 sequence 长期保留的 KV；KV 属于 Attention。一个 token 在某层 dispatch 到远端 expert，并不意味着它的 KV ownership 也迁过去。

```text
sequence owner / Attention DP rank
  ├── owns request scheduler state
  ├── owns or references KV Cache
  ├── sends layer hidden state to routed experts
  └── receives combined output back
```

所以 Decode DP load balancing 常以 KV usage 和 active sequence count 为主要信号。迁移 sequence owner 需要处理整条请求与 KV；改变 expert replica 只影响单层瞬时路由，两者代价完全不同。

这也给 P/D 解耦留下空间：Prefill pool 与 Decode pool 可以使用不同 Wide EP layouts，KV 通过独立 connector 传输，expert physical ids 不进入 KV 语义。

## 网络容量怎样估算

设每秒经过 MoE layers 的 token rate 为 $Q$，MoE layer 数为 $L_m$，Top-$k$ 为 $k$，hidden size 为 $H$，dispatch/combine 分别为 $b_d,b_c$ bytes。按 assignment 展开的 hidden-state 逻辑数据率为：

$$
B_{logical}
=Q L_m k H(b_d+b_c)
$$

跨节点 assignment 比例为 $\rho_{inter}$，则相同口径下为：

$$
B_{inter}\approx\rho_{inter}B_{logical}
$$

实际网络一方面要加入 metadata、padding、协议开销、分层转发和不均衡，并考虑双向/rail 分布；另一方面，route-aware dispatcher 对同一 token 的同 destination 命中去重后，hidden-state payload 又可能低于这个 assignment-expanded 估算。容量规划必须用真实 routing trace 与 NIC counters 校准，也不能只用全集群 aggregate bandwidth，因为最热 NIC/rank 决定瓶颈：

$$
B_{required,rail}
\ge \operatorname{P99}(B_{rank/rail})\times headroom
$$

还要把 KV transfer、checkpoint、权重加载和其他 collectives 纳入共享链路。若 EP 与 P/D KV transfer 同时跨同一 fabric，高峰时会互相抬高 TTFT/TPOT。

## 故障域从一台节点扩大到整个 EP Group

Wide EP 的一次 MoE forward 依赖整个 group。任一 rank、NIC rail 或节点失效，都可能让 All-to-All 无法完成。健康单位不再是一张 GPU：

```text
one rank failure
  → abort outstanding dispatch/combine
  → whole EP instance not ready
  → fail or migrate in-flight requests
  → recreate complete process group
  → reload/verify expert placement generation
  → warm up communication buffers and kernels
  → canary before republishing
```

因此冗余容量必须以完整 Wide EP instance 计算。一个 EP144 实例需要 18 个 8-GPU 节点；零散的 10 张空闲 GPU 无法充当 N+1。

Wide group 启动也更慢：权重分片、RDMA registration、process rendezvous、JIT kernel 和 topology validation 都要完成。Autoscaler 不能假定节点分配成功后数秒内就有新容量。

降低故障爆炸半径的选择包括：

- 使用多个较窄 EP instances；
- 每个实例保留足够吞吐 headroom；
- 请求路由跨实例分散；
- 预热完整 spare group；
- drain/升级以 gang 为单位；
- 对 fabric degradation 做 slow-rank 检测。

更宽 EP 的效率收益需要与可用性和恢复成本一起计价。

## Control Plane 需要管理多个版本

Wide EP 不只有 model revision，还包含多份可变化的执行配置：

```text
model_revision
parallel_layout_generation
expert_placement_generation
communication_backend/config
network_topology_snapshot
```

一次安全 rollout 应保证所有 ranks 在同一 barrier/batch boundary 切换。不能出现：

- Router 使用新 logical-to-physical map，部分 destinations 仍装旧权重；
- P worker 写入一个 pool 的 physical expert id，D worker 按另一 layout 解释；
- 旧 dispatch completion 写入新 generation buffer；
- 一半 ranks 已切换 communication backend。

控制面发布 declarative plan，worker group 完成加载、checksum 和 canary 后再整体 ready。失败时回滚完整 generation，而不是逐 rank 修补。

## 何时 Wide EP 不值得

Wide EP 不是稀疏模型的默认最优解。以下场景要谨慎：

- 总 expert 权重单节点已能舒适放下；
- 在线并发低，无法形成足够大的全局 expert batch；
- 跨节点网络带宽或稳定性不足；
- workload 变化快，负载计划持续过期；
- EP group 启动/恢复时间不满足可用性目标；
- KV Cache 才是主要容量瓶颈，扩大 EP 并不能减少它；
- 模型只有少量 experts，副本/peer 管理收益有限；
- TP/PP 已消耗大量网络，EP 再扩展导致争用。

可以从最小可行 EP 开始，在真实 trace 上逐步扩大，寻找 goodput 峰值。吞吐随 EP size 的曲线通常先上升后趋平甚至下降，而非单调增长。

## 基准矩阵

评测 Wide EP 至少扫描四个维度：

### Scale

```text
EP size:        8 / 16 / 32 / 64 / ...
nodes:          1 / 2 / 4 / ...
experts/rank:   derived with replicas
```

### Workload

```text
Prefill token buckets
Decode active sequences
ISL / OSL / KV length distribution
real routing trace and synthetic skew
```

### Data path

```text
dispatcher backend/version
dispatch and combine dtype
communication SM count
flat vs hierarchical transport
overlap enabled/disabled
```

### Placement

```text
redundant expert count
hierarchical/global EPLB plan
local/cross-node assignment ratio
plan update interval
```

输出不能只有 tokens/s。至少包括：

- TTFT、TPOT 和 goodput；
- Attention、dispatch、expert GEMM、combine 分解；
- per-expert/rank/node P50-P99 load；
- NIC/NVLink bytes、utilization 与 slow ranks；
- local vs cross-node assignments；
- communication-compute overlap；
- KV capacity、communication buffer 与 expert weights；
- group startup、recovery 和 plan rollout 时间。

## 一条配置推导路径

部署一个大型 MoE serving pool，可以按以下顺序做：

1. 计算 Attention/shared/routed expert 权重、KV 与 buffer 显存；
2. 找到能够放下权重的最小 EP width；
3. 根据目标流量估算每 logical expert 的 batch，确认扩大 EP 有足够 token；
4. 为 Attention 独立选择 TP/CP/DP，不让 expert ETP 被迫等于 Attention TP；
5. 画出 rank/node/NIC 与所有 process groups；
6. 用真实 routing trace 估算 count matrix 和跨节点比例；
7. 选择 redundant expert 数，使 slot 整除且热点有合理副本；
8. 分别为 Prefill 与 Decode 搜索 EP width 与 dispatcher 路径；
9. 建立请求/Attention DP balance 与 EPLB 两套控制环；
10. 验证 overlap、故障恢复、启动时间和整组冗余；
11. 以满足 SLO 的 goodput，而不是单次峰值吞吐选最终配置。

每次只改变一类变量。若同时把 EP8 改成 EP64、开启 FP8、换 DeepEP、加 32 个 replicas 再打开 dual batch overlap，一个最终 speedup 无法说明哪项有效，也无法安全回退。

## 小结

Wide EP 的目标，是让大规模稀疏模型在每张 GPU 只保存少量 experts，同时从更大的全局请求池聚合 token，形成高效 expert GEMM。它的代价是 routed token 在每个 MoE layer 跨越更大的网络和故障域。

可以用七条原则概括：

1. Attention/KV 按请求保持 DP ownership，routed experts 才跨全局 EP group；
2. Parallel Folding 允许 Attention 与 MoE 使用不同 TP/CP/EP 维度；
3. Prefill 与 Decode 的 token shape 不同，应分别选择 EP width；
4. Group-limited routing、分层通信和 expert placement 共同控制跨节点 fan-out；
5. 请求负载、dispatch send 与 expert receive 是三种独立的均衡问题；
6. EP 变宽只有在全局 token batch 足够时才会改善 expert GEMM；
7. 可用性、启动与备用容量都以完整 EP group 为单位。

至此，MoE 执行链路从 router、Expert Parallel、EPLB、DeepEP 到 Wide EP 串了起来：逻辑路由决定要算谁，物理负载均衡决定副本在哪里，专用 dispatcher 决定 token 怎样抵达，而 Wide EP 决定这些机制怎样跨越整个 GPU 集群仍保持可解释、可恢复和满足 SLO。

## 参考资料

- [DeepSeek-V3/R1 Inference System Overview](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md)
- [DeepSeek EPLB](https://github.com/deepseek-ai/EPLB)
- [DeepEP](https://github.com/deepseek-ai/DeepEP)
- [vLLM: Expert Parallel Deployment](https://docs.vllm.ai/en/stable/serving/expert_parallel_deployment/)
- [Megatron Core: Mixture of Experts and Parallel Folding](https://docs.nvidia.com/megatron-core/developer-guide/nightly/user-guide/features/moe.html)
- [MoE Parallel Folding: Heterogeneous Parallelism Mappings for Efficient Large-Scale MoE Model Training](https://arxiv.org/abs/2504.14960)
