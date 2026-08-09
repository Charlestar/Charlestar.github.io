---
layout: post
title: "DeepEP：把 MoE 的 Dispatch 与 Combine 做成专用数据面"
subtitle: "理解变长通信、两级互联、低延迟路径、FP8 传输与计算重叠"
date: 2026-07-23 09:00:00 +0800
last_modified_at: 2026-08-09
author: iStar
catalog: true
series: moe-communication
series_order: 40
technology_year: 2025
mathjax: true
tags: [AI Infra, MoE, DeepEP, Expert Parallel, RDMA]
---

Expert Parallel 把不同 experts 放在不同 GPU 上。Router 选出 Top-$k$ experts 后，token hidden states 先通过 dispatch 到达 expert owner；expert GEMM 完成后，输出再通过 combine 回到 source rank。这个过程看起来像两次 All-to-All，但直接调用一个通用 collective，通常只能解决“数据最终到达”，不能同时解决 MoE 真正关心的几件事：

- 每轮路由都产生不同的变长 count；
- token 要按 destination 和 local expert 重排；
- 节点内 NVLink 与跨节点 RDMA 应走不同数据路径；
- Prefill 追求大消息吞吐，Decode 追求小消息延迟；
- 通信 kernel 不能把 expert GEMM 需要的 SM 全占掉；
- combine 必须准确复用 dispatch 的逆映射。

DeepEP 的定位，就是把这些需求收进一个面向 Expert Parallel 的 GPU 通信数据面。它不仅提供“发送张量”的能力，还把路由元数据、buffer、通知、低精度 payload、dispatch/combine 配对与异步依赖纳入同一条执行路径。

本文不把某个版本的 API 当作永远不变的标准，而是沿着一轮 MoE forward，解释 DeepEP 为什么存在、各组件解决什么问题，以及接入 serving runtime 时仍有哪些职责不属于通信库。

## 通用 collective 少知道了一层语义

NCCL All-to-All 一类 collective 看到的是 rank 与 buffer；MoE runtime 看到的是 token、expert assignment 和 Top-$k$ 权重。

```text
generic collective
  input:  send buffers + counts
  output: receive buffers

MoE dispatcher
  input:  hidden states + expert ids + routing weights
  output: tokens grouped for each local expert
          + metadata for the reverse combine
```

如果 runtime 自己完成所有 count、prefix sum、permutation、跨节点转发与 inverse permutation，再把中间 buffer 交给 collective，数据会多次往返 HBM，还会产生许多 kernel launch 和 CPU/GPU 同步。通信库即使达到高链路带宽，端到端 MoE layer 仍可能慢在 collective 之外。

DeepEP 选择提升抽象层次：dispatch 直接接收 router 结果，输出按本地 expert 可消费的布局，并返回后续 combine 所需的 handle。这样，优化可以跨越原本分开的 permutation、network transfer 和 notification 边界。

## 一轮 DeepEP 数据流

设 source rank 上的输入为：

```text
x            [num_tokens, hidden]
topk_idx     [num_tokens, top_k]
topk_weight  [num_tokens, top_k]
```

一次 forward 的逻辑路径为：

```text
router output
    │
    ▼
DeepEP dispatch
    ├── count assignments by destination/expert
    ├── calculate layout and offsets
    ├── move hidden states through NVLink/RDMA
    └── produce EPHandle + completion event
    │
    ▼
recv_x grouped by local expert
    │
    ▼
grouped expert GEMM
    │
    ▼
DeepEP combine(handle)
    ├── send outputs back to source ranks
    ├── restore source token / top-k slot
    └── reduce routing-weighted expert outputs
    │
    ▼
MoE layer output in original token order
```

这条路径中有三个核心对象：通信 buffer、dispatch handle 和异步 event。理解它们比背函数参数更重要。

## ElasticBuffer：通信不能每轮临时找内存

变长 routing 不等于可以在每轮随意分配显存。在线 Decode 每层、每步都执行 MoE；若 dispatch 先把 counts 拷回 CPU，再按本轮大小申请 send/receive buffer，会引入 allocator、同步和不可预测的长尾。

DeepEP V2 使用统一的 `ElasticBuffer` 接口承载高吞吐与低延迟 EP 操作。初始化时需要知道一组容量信息：

```text
EP process group
max tokens per rank
hidden size
Top-k
whether dispatch uses FP8
```

buffer 容量上界可抽象为：

$$
M_{buffer}
=f(P_e, N_{max}, H, k, dtype, alignment, protocol)
$$

其中 $P_e$ 是 EP size，$N_{max}$ 是每 rank 最大 token 数。`f` 不只是 payload 乘法，还包含路由元数据、notification、接收布局、对齐和 transport 所需空间。

预分配带来三点收益：

1. 稳定地址便于 GPU/RDMA 数据路径复用；
2. 避免热路径中的动态显存申请；
3. 可以提前发现配置无法容纳峰值，而不是写越界后才失败。

代价是容量规划必须真实。官方 V2 文档也明确提示其 buffer 占用高于 V1。若把 `max tokens per rank` 设置得过大，会挤压 KV Cache 和 expert workspace；设置过小，则在突发 Prefill 或路由倾斜时无法承载。

## EPHandle：Combine 为什么不能重新猜路由

dispatch 后的 expert outputs 已失去原始 token 的连续顺序。要把结果返回 source rank，必须知道每个 row 来自哪里、属于哪个 Top-$k$ slot，以及哪些 rows 是 padding。

DeepEP dispatch 返回的 `EPHandle` 可以理解为这次路由布局的执行凭证：

```text
EPHandle
├── send / receive layout
├── source and destination mapping
├── per-expert receive counts
├── Top-k index / weight association
├── alignment and validity metadata
└── information needed by combine
```

官方示例中，handle 暴露每个本地 expert 收到的 token 数，grouped GEMM 可以据此构造每个 expert 的 $m$ shape；combine 则直接消费同一个 handle，把 expert outputs 沿逆路径送回。

这种配对有一个重要正确性边界：**handle 只对产生它的路由布局有效。** 如果下一层或下一 decode step 的 router decision 已经变化，就不能只因为 tensor shape 相同而复用旧 mapping。

V2 文档展示了“当 gating decisions 保持不变时缓存 handle”的模式。这里的前提不能省略。自回归模型的 hidden state 每步变化，router 选择通常也会变化；只有 runtime 能证明布局不变、路由被预先固定，或 handle 仅复用与具体 expert id 无关的安全部分时，才能跳过重算。

## EventOverlap：异步返回不代表数据已经可用

dispatch 若异步运行，Python 调用返回时网络和 GPU kernel 可能仍在执行。DeepEP 用 event 对象表达通信 stream 与当前 compute stream 的依赖：

```text
recv_x, handle, event = dispatch(..., async=true)

do_independent_work()

event.current_stream_wait()
expert_gemm(recv_x)
```

这里有两个容易混淆的概念：

- **调用异步**：CPU 不等待通信完成；
- **通信计算重叠**：等待前确实存在一段与 dispatch 无依赖的 GPU 计算，并且两者能并发使用硬件资源。

如果 dispatch 后立即 wait，只有异步 API，没有 overlap。如果通信 kernel 占满 SM 或 HBM，独立 GEMM 即使位于另一个 stream 也可能变慢。DeepEP 强调降低或控制通信的 SM 占用，目的之一正是给计算留下资源。

## 为什么要关心通信用了多少 SM

GPU 发起、搬运、重排和同步通信时，常需要执行 CUDA kernels。这些 kernels 会占用 Streaming Multiprocessors；而 expert GEMM 也需要同一批 SM。

若通信单独执行，使用更多 SM 可能更快达到带宽上限；若与 GEMM 重叠，过多通信 SM 会导致：

$$
T_{gemm}^{overlap} > T_{gemm}^{standalone}
$$

最终 layer wall time 可能反而增加。正确目标不是最小化 `dispatch_us`，而是最小化：

$$
T_{MoE}
=T_{route+dispatch+expert+combine}
$$

DeepEP V2 会根据 MoE 配置分析计算 SM/QP 数，也允许调用方覆盖。官方性能表同时报告 bandwidth 与 SM count，就是提醒读者：两条同样达到 90 GB/s 的路径，若一条占 24 个 SM、另一条占 6 个 SM，它们与计算重叠的价值并不相同。

这些数字只能在相同 GPU、NIC、拓扑、message shape 和版本下比较，不能当成任何集群的固定结论。

## 高吞吐与低延迟不是两套数学，而是两种形状

DeepEP V1 文档把 normal kernels 和 low-latency kernels 分开描述；V2 在 `ElasticBuffer` 下统一接口，但 Prefill 与 Decode 的优化目标仍然不同。

### Prefill / 训练形状

一轮有较多 tokens：

- payload 大，链路带宽决定性更强；
- counts 和 launch 固定成本容易摊薄；
- expert GEMM 更大，存在流水化 overlap 空间；
- buffer 峰值与路由倾斜更需要关注。

目标更接近：

$$
\max \frac{\text{logical payload bytes}}{T_{dispatch/combine}}
$$

但仍要同时观察真实 NIC bytes、SM 占用和端到端 layer time。

### Decode 形状

每条活跃 sequence 每步通常贡献一个 token：

- 消息小而频繁；
- notification、CPU sync 和 RTT 占比更高；
- 每个 expert 的 GEMM 可能只有几个 rows；
- 任一 rank 的迟到都会反映到 TPOT。

目标更接近最小化一次 dispatch/combine 的固定延迟，而不是把大 buffer 填满。一个在 8K-token benchmark 达到高带宽的 kernel，不能证明它在 64-token Decode batch 下有更低延迟。

因此接入时至少要用两套基准：实际 Prefill token bucket，以及由线上并发与 continuous batching 形成的 Decode token 数分布。

## 节点内和跨节点是两条物理路径

单节点 EP 可通过 NVLink/NVSwitch 交换；跨节点 EP 还要经过 GPU、NIC、交换网络与远端 GPU。两者的带宽、延迟、peer scale 和拥塞模型不同。

```text
same-node assignment
GPU ── NVLink/NVSwitch ── GPU

cross-node assignment
GPU ── local fabric ── NIC
    ── RDMA fabric ──
NIC ── local fabric ── GPU
```

如果把所有 peers 视为完全相同的平坦网络，容易出现两类浪费：

1. 原本可在节点内完成的 traffic 绕行或复制到跨节点路径；
2. 许多 GPU 各自向远端发送小消息，放大 QP、通知和交换机压力。

DeepEP 面向 NVLink 节点内通信与 RDMA 跨节点通信设计，并支持不同规模的 scale-up/scale-out domain。V2 当前实现转向 NCCL Gin backend，同时保留 hybrid/direct 模式；V1 的 NVSHMEM 路径仍有单独 legacy 文档。部署文档必须固定 commit/tag，因为 transport backend 与依赖要求会随版本变化。

## DeepEP V1 与 V2 的边界不能混写

DeepEP 是快速演进的开源项目。当前仓库说明的 V2 有几项结构变化：

- 高吞吐与低延迟 API 统一到 `ElasticBuffer`；
- EP backend 从 V1 的 NVSHMEM 主路径转向更轻量的 NCCL Gin；
- kernel 采用运行时 JIT 组织；
- 支持更大的 scale-up/scale-out domain；
- 通过分析式方法选择 SM 与 QP 数；
- 在保持吞吐的同时降低部分场景的 SM 使用；
- buffer 消耗高于 V1；
- V1 的某些 0-SM 低延迟 RDMA 能力不再属于 V2 当前路径。

因此，网上常见的 `Buffer`、`Buffer.set_num_sms`、NVSHMEM 配置和 low-latency 专用接口，可能描述的是 V1；当前 README 示例使用的是 V2 `ElasticBuffer` 与 `EPHandle`。文章、部署脚本和故障排查记录应明确版本，不能把两代参数拼在同一配置里。

版本变化不影响核心抽象：route-aware dispatch、paired combine、预分配通信空间和显式异步依赖仍然成立。

## FP8 Dispatch 减少的是 payload，不是全部成本

若 hidden states 原本以 BF16 传输，每元素 2 bytes；使用 FP8 可把主体 payload 降到约一半：

若 $N_a=N_tk$ 为 assignment 数：

$$
V_{BF16}=2N_aH
$$

$$
V_{FP8}\approx N_aH + V_{scale}
$$

其中 `scale` 的数量取决于量化粒度。DeepEP 接口可以让 dispatch 输入由 data 与 scale factors 组成，combine 仍可使用 BF16，从而优先压缩跨网络的 expert 输入。

但端到端收益还要扣除：

- FP8 quantize 和 scale 生成；
- 额外 scale payload；
- 接收端 dequantize 或 FP8 expert kernel 的适配；
- alignment/padding；
- 数值误差对模型输出的影响。

若 Decode payload 很小，固定量化开销可能抵消节省的字节；若网络是主要瓶颈，收益更明显。需要分别对质量、dispatch 时间、expert GEMM 以及完整 TPOT 做消融。

上面的公式里 `k` 已包含在 $N_a$，不能再次相乘。通信容量建模时最常见的错误之一，就是混用 token 数和 assignment 数，导致 Top-$k$ 被漏算或重复计算。

## Router 权重什么时候应用

Top-$k$ router weights 可以在 source rank 保存，也可以随 dispatch 发送，并在 combine 时完成加权归并：

$$
y_t=\sum_{j=1}^{k}p_{t,j}o_{t,j}
$$

DeepEP dispatch/combine 接口允许传递 top-k weights，并由 handle 关联对应关系。具体 runtime 还要决定：

- expert GEMM 前是否需要权重；
- combine kernel 是否融合 multiply + reduce；
- shared expert output 在哪里合并；
- padded assignment 的权重怎样清零；
- low-precision weight 是否影响数值。

通信库可以搬运并组合数据，但 router 的模型语义来自 checkpoint。不能为了方便 buffer layout 改变 Top-$k$ 归一化顺序或丢弃某个 assignment。

## Shared Expert 提供天然的 overlap 候选

一些 MoE 架构除了 routed experts，还包含每个 token 都经过的 shared expert。若 shared expert 的输入已在本地，且与 routed dispatch 没有依赖，可以形成：

```text
stream comm:     routed expert dispatch ──────────────┐
stream compute:  shared expert GEMM ───────────────┐  │
                                                   ▼  ▼
                                             merge outputs
```

这是比“随便找点计算重叠”更可靠的候选，因为 shared expert 与远端 routed expert 从同一输入分叉。实际收益仍取决于 SM/HBM 争用和 merge 依赖；如果通信 kernel 抢占过多 SM，shared GEMM 会拉长。

没有 shared expert 时，也可尝试 chunked dispatch、其他 layer 的独立工作或训练中的相邻 microbatch overlap，但 runtime 必须证明依赖安全。

## Imbalance 决定 buffer 与尾延迟

DeepEP 加速数据移动，不能消除 router 造成的热点。设本轮总 assignments 为 $N_a$，expert $e$ 收到 $n_e$，destination rank $r$ 收到：

$$
L_r=\sum_{e\in\mathcal E_r}n_e
$$

最坏 rank 同时影响：

- receive buffer 峰值；
- 该 rank 的 RDMA/NVLink 入流量；
- grouped GEMM 工作量；
- combine 最晚完成时间。

`num_max_tokens_per_rank` 如果只按平均 $N_a/P_e$ 设置，会在热点出现时低估。若按“所有 token 都去同一 rank”的理论最坏值设置，又可能浪费大量显存。

工程上可以组合：

- 从真实 trace 提取 rank-load 分位数；
- 保留受控 headroom；
- 超过上界时 chunk dispatch，而非写越界；
- 使用模型允许的 expert replica / placement balancing；
- 对极端 overload 做 admission control；
- 记录 buffer overflow 或 fallback，而不是静默 dropping。

DeepEP 的未来工作中也包含借助 EP replay 降低不均衡所需的中间 buffer，这进一步说明 buffer sizing 与 load imbalance 是同一个问题的两面。

## 通信库不能替 runtime 做哪些事

DeepEP 位于 MoE layer 的数据面，不是完整推理引擎。下面这些仍由上层负责：

### Process group 与 expert placement

Runtime 决定哪些 ranks 构成 EP group、每个 expert 在哪里、是否有副本，以及 Attention 使用什么 TP/DP。DeepEP 依赖这份映射，不负责选择模型并行策略。

### Scheduler 与 batch

Continuous batching、Prefill/Decode 分流、请求优先级和 token budget 决定每轮输入形状。通信库不会为等待更大 batch 与 TPOT 之间做产品级取舍。

### Router 语义

Top-$k$、group-limited routing、shared expert、capacity 和 dropping 均由模型/runtime 定义。Dispatcher 只执行给定 assignment。

### Expert GEMM

dispatch 输出 per-expert token counts，具体 grouped GEMM、量化权重和 kernel 选择属于计算层。通信快了以后，许多小 expert GEMM 仍可能成为瓶颈。

### 故障恢复

某 rank 退出时，整个 EP group 的 collective/数据路径通常失效。上层必须摘除完整 worker group、迁移请求并以新 generation 重建 process group；buffer 本身不是持久化恢复状态。

## 集成时的数据契约

一个稳健的 runtime adapter 应明确以下契约：

```text
Input contract
  x layout and dtype
  topk_idx range and ownership map
  topk_weights normalization
  valid token count and padding mask

Dispatch output contract
  recv_x grouped order
  per-expert counts
  alignment rows and validity
  handle lifetime

Combine contract
  expert output dtype/layout
  same handle and request/layer epoch
  destination tensor ownership
  reduction precision
```

特别要处理 handle 生命周期。它引用的 buffer/layout 不能在 combine 完成前被下一轮覆盖；request cancellation、CUDA Graph replay 和多 stream 并发也不能让旧 handle 与新 payload 交叉。

可以给每次 MoE layer execution 加上：

```text
(microbatch_id, layer_id, decode_step, dispatch_epoch)
```

在 debug/验证构建中校验 handle 与 combine 调用一致，避免异步执行下的 ABA 问题。

## 正确性验证应该覆盖极端路由

DeepEP 官方仓库提供测试，但接入一个具体模型仍需做端到端对照。测试输入不能只用均匀随机路由，因为真实 bug 常出现在边界：

1. 所有 assignments 都命中一个本地 expert；
2. 所有 assignments 都命中一个远端 expert；
3. 某 destination 为 0 token；
4. token/expert count 不是对齐倍数；
5. Top-1、Top-2 与模型真实 Top-$k$；
6. 同一 token 的多个 experts 位于同一 rank；
7. 极端 rank imbalance；
8. FP8 dispatch + BF16 combine；
9. 多节点 rank mapping 改变；
10. handle 被错误复用时能够检测或测试失败。

验证不变量包括：

$$
N_{valid,send}=N_{valid,recv}=N_tk
$$

以及分布式 MoE output 与参考实现的数值误差。对 FP8 路径，应先验证 BF16 完全正确，再单独引入量化误差；不要同时调 permutation 和 dtype，避免错误互相掩盖。

## 性能基准要避免“逻辑带宽”误读

DeepEP 官方性能数据使用特定模型形状、GPU、NIC、EP topology 和 dtype，并说明报告的是 bottleneck/logical bandwidth，其中可能包含本地 rank traffic。复现实验至少要记录：

```text
DeepEP commit or release
GPU architecture and count per node
NVLink/NVSwitch topology
NIC model, ports and link rate
EP scale-up × scale-out layout
tokens per rank, hidden size, top-k
dispatch/combine dtype
expert alignment and imbalance pattern
communication SM count
```

至少同时报告：

- dispatch/combine latency 分布；
- logical payload bytes；
- actual cross-node NIC bytes；
- NVLink 与 RDMA utilization；
- communication SM count；
- expert GEMM 单独耗时与 overlap 后耗时；
- 完整 MoE layer wall time；
- serving 的 TTFT、TPOT 与 goodput。

如果只把 `payload / kernel_time` 算成 GB/s，可能得到高于物理 NIC 的数字，因为本地 traffic、双向流量或定义口径被算在分子中。这不一定是造假，但必须说明口径。

## 网络配置也是数据面的一部分

跨节点 EP 会与 TP、KV transfer、checkpoint、存储等 traffic 共享网络。DeepEP 当前文档提出了几项运行建议：

- 用 InfiniBand virtual lanes/service level 隔离 EP 与其他 workload；
- 根据网络条件配置 adaptive routing；
- 关注 congestion control 与优先级策略；
- 正确选择 NIC 和 RDMA 参数。

这些设置不能直接照抄。是否启用 adaptive routing、怎样分 VL、拥塞控制如何配置，要结合交换网络、运维策略和其他租户验证。文章中的 kernel 优化无法补偿错误布线、跨 NUMA 绑定或 NIC oversubscription。

一条最小拓扑审计应回答：

```text
GPU → nearest NIC mapping
EP ranks → nodes and NVLink domains
NIC rails → switch paths
other collectives sharing the same links
RDMA registration and permissions
failure behavior when one rail degrades
```

## 从现有 All-to-All 迁移到 DeepEP

可以分四步进行，而不是一次替换所有通信与量化：

### 第一步：固定 BF16 正确性

- 冻结模型 revision、router 与 parallel mapping；
- 保持原 grouped GEMM 不变；
- 用 DeepEP BF16 dispatch/combine 替换原 dispatcher；
- 做逐层输出和极端路由对照。

### 第二步：建立形状基线

- 收集 Prefill/Decode 的 token-per-rank 分布；
- 记录 per-expert/per-rank imbalance；
- 设定 buffer 上界与 headroom；
- 分别测单节点和多节点。

### 第三步：优化异步与 SM

- 用 event 显式建立 stream dependency；
- 找到真正独立的 shared expert 或 chunked work；
- 扫 communication SM count；
- 以完整 layer wall time 选择配置。

### 第四步：引入低精度与网络调优

- 单独开启 FP8 dispatch；
- 验证模型质量和 scale 开销；
- 配置 traffic isolation/adaptive routing；
- 重新测端到端 SLO，不只比较 microbenchmark。

每一步都保留回退开关。遇到未知路由 shape、buffer overflow 或 backend 不可用时，runtime 可以回到已验证的 dispatcher，而不是让整个模型服务无法启动。

## 故障与降级边界

DeepEP 运行在一个已建立的 EP process group 上。一个 rank 或链路失败后，不应假设本轮 dispatch 能在剩余 ranks 上自动缩容继续：

```text
rank/link failure
  → abort current EP operation
  → mark entire model instance not ready
  → cancel or migrate affected requests
  → destroy/recreate communication group
  → allocate/register fresh buffers
  → run correctness canary
  → publish new instance generation
```

若跨节点网络退化但未完全断开，表现可能是 TPOT 尾延迟上升而非显式 error。需要监控 per-rank dispatch/combine、RDMA retry、link counters 和 barrier skew，找出 slow rank。

Fallback 到较慢 dispatcher 只有在所有 ranks 一致切换、buffer layout 兼容且当前请求边界安全时才可进行。不能让一半 ranks 使用 DeepEP、另一半使用普通 collective。

## 小结

DeepEP 不是“更快的 All-to-All”这么简单。它把 MoE 路由的上层语义下沉到通信数据面：输入是 token 与 expert assignments，输出是可直接供 grouped GEMM 消费的本地布局；dispatch handle 又把同一次路由准确带到 combine。

理解它可以抓住七点：

1. `ElasticBuffer` 用预分配和稳定数据路径换取低抖动，但必须做好峰值与不均衡容量规划；
2. `EPHandle` 是 dispatch/combine 的布局契约，只能在路由确实不变时复用；
3. `EventOverlap` 表达依赖，真正 overlap 还需要独立计算和足够硬件资源；
4. 通信 SM 越多不一定端到端越快，优化目标是完整 MoE layer；
5. Prefill 与 Decode 需要分别测试吞吐和固定延迟；
6. FP8 降低 payload，但不能省略 scale、量化开销和质量验证；
7. V1/V2 backend、API 与依赖差异明显，部署必须固定版本。

DeepEP 解决的是 token 怎样高效穿过 GPU 网络。它不能决定 experts 应该放在哪里，也不能修复 router 长期把流量压到少数 ranks 的问题。下一步要处理的正是 Expert Parallel Load Balancing：如何从 per-layer routing trace 推导热点、复制和重排 experts，又不改变模型的路由语义。

## 参考资料

- [DeepEP 官方仓库与 V2 文档](https://github.com/deepseek-ai/DeepEP)
- [DeepEP V1 Legacy Documentation](https://github.com/deepseek-ai/DeepEP/blob/main/docs/legacy.md)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- [Megatron Core: Mixture of Experts](https://docs.nvidia.com/megatron-core/developer-guide/nightly/user-guide/features/moe.html)
- [Megatron Core: Parallelism Strategies Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)
