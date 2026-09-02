---
layout: post
title: "Expert Parallel：MoE Token 为什么要两次穿过 GPU 网络"
subtitle: "从 EP 分组、All-to-All 通信量到 Prefill/Decode 的并行选择"
date: 2026-07-17 09:00:00 +0800
last_modified_at: 2026-09-03
author: iStar
catalog: true
series: moe-communication
series_order: 20
technology_year: 2017
mathjax: true
tags: [MoE, 专家并行, 分布式推理]
---

MoE 模型把大量 FFN 参数分散到许多 experts 中，每个 token 只激活其中少数几个。稀疏激活降低了单 token 计算量，却没有让其余专家权重从显存里消失。当一张 GPU 放不下所有 experts 时，最直接的做法是让每个 rank 只保存一部分 experts，这就是 Expert Parallel（EP）。

EP 解决了权重容量问题，也改变了 token 的执行路径：Attention 结束时，token 仍在原来的 source rank；router 选中的 expert 却可能在另一个 rank。hidden state 必须先被 dispatch 到 expert owner，计算结束后再 combine 回 source rank，才能继续 residual connection 和下一层。

因此，一个 MoE layer 不是“一次稀疏 GEMM”，而是：

```text
route
  → count
  → permute
  → dispatch
  → local expert GEMM
  → combine
  → unpermute
```

理解 EP 的关键，就是把这两次跨设备交换的形状、代价与正确性条件说清楚。

## EP 与 TP、DP 切分的对象不同

假设一层有 $E$ 个 experts，expert parallel size 为 $P_e$。最简单的均匀放置下，每个 EP rank 保存：

$$
E_{local} = \frac{E}{P_e}
$$

个完整 experts。

这与 Tensor Parallel（TP）不同。TP 把一个矩阵的行或列分到多张 GPU，同一个 token 经过该层时，多张 GPU 协作完成一次矩阵乘；EP 则让不同 expert 的权重归不同 rank 所有，token 根据 router 结果移动到对应 owner。

```text
Tensor Parallel
  one expert W = [W shard 0 | W shard 1 | ...]
  every token participates in the TP collective

Expert Parallel
  rank 0 owns expert 0,1
  rank 1 owns expert 2,3
  rank 2 owns expert 4,5
  rank 3 owns expert 6,7
  each assignment moves to its expert owner

Data Parallel
  each replica owns the same model/expert set
  different requests or token batches stay in different replicas
```

EP 主要按 expert 维度节省权重显存，TP 同时切分 dense/attention 和单个 expert 的内部矩阵，DP 则增加独立副本与 serving 吞吐。大型部署常把三者组合起来，而不是三选一。

## 一个 Top-2 路由例子

设 8 个 experts 分布在 4 个 EP ranks，每个 rank 先持有两个输入 token：

```text
rank 0: t0, t1       owns expert 0,1
rank 1: t2, t3       owns expert 2,3
rank 2: t4, t5       owns expert 4,5
rank 3: t6, t7       owns expert 6,7
```

router 为每个 token 选择两个 experts：

```text
t0 → e1, e6
t1 → e4, e7
t2 → e1, e5
t3 → e2, e4
t4 → e0, e7
t5 → e3, e6
t6 → e2, e5
t7 → e0, e3
```

原来只有 8 个 token，但 Top-2 产生 16 个 **expert assignments**。Expert GEMM 的工作单位不是原 token 数，而是 assignment 数：

$$
N_a = N_t \times k
$$

其中 $N_t$ 是进入 MoE layer 的 token 数，$k$ 是 Top-$k$。

rank 0 上的 `t0` 有一份留给本地 expert 1，另一份发到 rank 3 的 expert 6；`t1` 则分别发往 rank 2 和 rank 3。其他 ranks 做相同处理。dispatch 后，数据按 owner 重组：

```text
rank 0 receives: e0:[t4,t7]  e1:[t0,t2]
rank 1 receives: e2:[t3,t6]  e3:[t5,t7]
rank 2 receives: e4:[t1,t3]  e5:[t2,t6]
rank 3 receives: e6:[t0,t5]  e7:[t1,t4]
```

每个 rank 对本地 experts 运行 grouped GEMM。计算结果随后沿相反方向回到原 token 所在 rank，再用 router 权重合并两份输出。

## Dispatch 前为什么必须 count 和 permute

router 的输出通常是形如 `[num_tokens, top_k]` 的 expert id 和权重。网络传输希望发往同一 destination 的数据连续，因此要先把散乱 assignment 变成分段布局。

### 统计目的 rank

若 experts 均匀连续分配，expert $e$ 的 owner 可以写成：

$$
\operatorname{owner}(e)
=\left\lfloor\frac{e}{E_{local}}\right\rfloor
$$

每个 source rank 统计 `send_count[destination]`，还需要得到其他 rank 将向自己发送多少数据。动态路由使 count 每轮都可能不同，不能预先假定固定大小。

### Prefix sum 计算偏移

对 counts 做 prefix sum，得到每个 destination 在 send buffer 中的起始 offset：

```text
send_count  = [3, 1, 2, 2]
send_offset = [0, 3, 4, 6]
```

### 按目的地写入连续 buffer

每个 assignment 把 hidden state、expert id、router weight，以及恢复原位置所需的索引写到目标区间：

```text
dispatch payload
├── hidden state
├── local expert id
├── source token index
├── top-k slot / routing weight
└── request or validity metadata
```

实现会尽量融合 histogram、prefix sum 和 permutation，避免多次读写 HBM。但无论 kernel 怎样融合，逻辑上仍必须建立一个双射：每个有效 assignment 恰好发送一次，并能在 combine 后回到正确的 token/slot。

## All-to-All-v 才接近真实通信形状

规则的 All-to-All 假定每个 source 发给每个 destination 相同数量数据。MoE 的路由由输入决定，更接近 All-to-All-v：每一对 rank 的发送量都可变化。

可以把一轮通信写成 count matrix：

$$
C_{ij} = \text{rank }i\text{ 发往 rank }j\text{ 的 assignment 数}
$$

理想均匀时，按 assignment 展开的 $C_{ij}$ 大致接近 $N_a/P_e^2$；实际可能出现某一列特别大，说明一个 destination 拥有热点 experts。该 rank 不仅收到更多工作，还通常要接收更多字节、执行更大的 expert GEMM，成为所有 source 的尾部延迟。这里的 $C_{ij}$ 是路由工作量，不一定等于物理传输的 hidden-state 行数：若同一 token 选中的多个 experts 位于同一 destination rank，DeepEP/Flex 一类 dispatcher 可以只发送一份 hidden state，再在目的端展开给这些 experts。

通信前往往还有一轮 count exchange 或对等同步，让接收端分配/定位 buffer。专用 dispatcher 会把元数据交换、payload 传输、GPU 侧同步和 permutation 尽量合并，但动态大小不会凭空消失。

## 先按 assignment-expanded 口径估算通信量

hidden size 为 $H$，传输 dtype 为 $b$ bytes，Top-$k$ 为 $k$。若按每个 assignment 独立展开 hidden state，一次 dispatch 的 hidden-state 逻辑字节数近似为：

$$
V_{dispatch} \approx N_t \times k \times H \times b
$$

若 combine 也逐 assignment 返回相同形状的 expert output，两次往返的 hidden-state 逻辑字节数近似为：

$$
V_{roundtrip}
\approx 2N_tkHb
$$

并非所有 assignment 都跨设备。若 expert 选择与放置近似均匀，本地比例约为 $1/P_e$，在“每个 assignment 独立传输”的估算下，跨 rank 的端到端 hidden-state 逻辑字节数约为：

$$
V_{network}
\approx 2N_tkHb\left(1-\frac{1}{P_e}\right)
$$

这是 assignment-expanded 容量估算，不是物理链路字节的严格上界：同 destination 去重会使它降低，元数据、对齐和分层转发又可能增加链路流量。若 dispatcher 对同一 token 的同 rank 多 expert 命中做去重，令 $D_t$ 为 token $t$ 的不同远端 destination rank 集合，则 dispatch 的 hidden-state 主体更接近：

$$
V_{dispatch,network}
\approx H b\sum_t |D_t|
$$

而不是固定的 $N_tkHb$。反方向若在 expert rank 先完成本地加权归并，combine 也可以复用同样的去重机会。除此之外还需要加入：

- expert id、offset 和 router weight；
- padding 或 alignment；
- count/notification 元数据；
- 协议 header 与重传；
- 节点内转发和跨节点 RDMA 的不同路径；
- 量化 scale 与反量化开销。

### 一个数值例子

设一轮有 2,048 个 token，Top-2，hidden size 7,168，dispatch 使用 FP8 的 1 byte，combine 使用 BF16 的 2 bytes：

$$
V_{dispatch}
=2048\times2\times7168\times1
\approx 29.4\text{ MB}
$$

$$
V_{combine}
=2048\times2\times7168\times2
\approx 58.7\text{ MB}
$$

在逐 assignment 展开的口径下，单层一次往返的 hidden-state 逻辑字节约为 88 MB；实际链路字节还取决于本地命中、同 destination 去重、元数据和转发路径。模型包含许多 MoE layers 时，互联仍很容易进入关键路径。FP8 dispatch 能减少发送字节，但必须连同量化精度、scale 布局和端到端耗时一起验证。

## Combine 不是简单把结果发回来

expert output 返回 source rank 后，还要恢复 token 原顺序并按 router 权重合并：

$$
y_t = \sum_{j=1}^{k} p_{t,j}\,E_{e_{t,j}}(x_t)
$$

combine handle 通常需要保存：

- assignment 在 dispatch buffer 中的位置；
- source rank 与 source token index；
- Top-$k$ slot；
- 每个 expert 的 token count；
- router weight，或能重新取得它的索引；
- padding/无效 token mask。

DeepEP 的接口就让 dispatch 返回可供 combine 使用的 handle，并暴露每个本地 expert 收到的 token 数，供后续 grouped GEMM 构造形状。这不是 API 细节上的偶然，而是 dispatch 和 combine 天生共享路由元数据。

若两个 expert output 到达顺序不同，不能按“收到顺序”相加；必须按稳定 token index 归位。一个 offset 错误可能不会造成 shape mismatch，却会把 `t3` 的输出加到 `t4` 上，是 EP 最危险的静默正确性问题之一。

## 为什么不用 AllGather 收集所有 token

另一种 dispatcher 是让每个 rank AllGather 所有 token，然后在本地挑出属于自己的 assignments。它省去了复杂的变长点对点 permutation，但会把无关 token 也发送到所有 ranks。

简化比较：

| 方案 | 数据移动 | 优点 | 代价 |
| --- | --- | --- | --- |
| AllGather | 每个 rank 得到全部输入 token | shape 规则、实现直接 | EP 越大，冗余数据越多 |
| All-to-All | 只把 assignment 发给 owner | 避免大部分无关 token | 动态 count、permute 与同步复杂 |
| Flex/专用 dispatcher | 按路由发送并融合多级传输 | 可针对跨节点和低延迟优化 | 与硬件、runtime 和版本绑定更深 |

当 EP 很小、Top-$k$ 很大，或 token batch 极小，AllGather 的简单性可能占优；当 experts 多、跨节点通信昂贵时，按路由发送通常更合理。Megatron Core 当前文档也同时保留 AllGather、All-to-All 和 Flex dispatcher，并按 EP 规模与跨节点场景区分适用范围。

所以“MoE 必须 All-to-All”是过度概括。需要比较的是实际 workload 下的端到端 layer time，而不是 collective 名称。

## EP size 不是越大越好

增大 $P_e$ 会降低每张 GPU 保存的 expert 权重：

$$
M_{expert,local}
\approx \frac{M_{expert,total}}{P_e}
$$

但同时会发生：

1. 本地命中比例下降，更多 assignment 跨设备；
2. 每个 expert 收到的 token 数不变或变小，GEMM 更碎；
3. peer 数增加，count/sync 开销上升；
4. EP group 更可能跨越节点边界；
5. 最慢 rank 影响的参与者更多。

选择 EP size 时，先满足权重显存，再在可行集合中比较网络与计算效率。Megatron Core 的通用建议也是把模型并行维度保持在避免 OOM 所需的较小范围，把剩余 GPU 用于数据并行；这不是绝对规则，但揭示了 model parallel communication 的现实成本。

## Expert Tensor Parallel 处理单个 expert 仍然过大

如果一个完整 expert 仍放不进单卡，可以在 expert 内继续使用 Expert Tensor Parallel（ETP）：

```text
EP chooses which expert group owns a token
ETP shards that expert's matrices inside the group
```

于是一次 token 路径同时包含：

- EP dispatch/combine；
- expert 内部 TP collective；
- Attention/共享层自己的 TP collective。

对于 fine-grained MoE，单 expert 的矩阵可能已经较窄，继续切分会让每 rank GEMM 更小、collective 比例更高。若显存允许，ETP=1 往往更容易获得高效 grouped GEMM；若 expert 很大，则必须接受 ETP 的额外通信。

组合 TP 与 EP 时，还要明确 token 在 TP ranks 上是否复制。若每个 TP rank 都持有相同 token，却未经 sequence parallel 去重就参与 EP dispatch，可能把同一个 assignment 重复发送。Megatron Core 因此明确要求 TP+EP 配置配合 sequence parallel；不同 runtime 的具体布局可能不同，但“先确定 token 在每个并行维度上的所有权”是普遍原则。

## Attention 与 MoE 不必使用同一组并行维度

Attention 适合的切分与 experts 适合的切分不一定相同：

- Attention 需要考虑 head 数、KV heads 与长上下文；
- MoE 需要考虑 expert 数、Top-$k$、grouped GEMM 和 All-to-All；
- 高 TP 可能有利于放下 Attention 权重，却把 expert GEMM 切得过小；
- 高 EP 能分散 experts，却不直接降低 KV Cache。

可以用 parallel folding 一类思路让两类层使用不同逻辑分组：

```text
Attention domain: TP × CP × DP
MoE domain:       ETP × EP × EDP
Shared dimension: PP / global world
```

层边界需要完成 layout 转换，但换来的好处是避免让所有层服从同一个不合适的切分。部署前应画出每个 rank 在每种 group 中的身份，而不是只写一个 `world_size=64`。

## 节点内与节点间应分层处理

同一节点的 NVLink/NVSwitch 和跨节点 RDMA 在延迟、带宽与拥塞特征上差异很大。一个 EP group 跨 4 个节点时，平坦 All-to-All 可能让每张 GPU 直接与许多远端 peers 交换小块数据。

分层 dispatcher 可以：

```text
source GPU
  → node-local gather/forward
  → inter-node RDMA
  → destination node local scatter
```

或将节点视为 scale-up domain，在域内走 NVLink、域间走 NIC，并融合通知和数据路径。DeepEP 正是围绕 MoE dispatch/combine 提供高吞吐与低延迟 kernel，并针对节点内、跨节点以及低精度传输优化。

拓扑感知还包括 expert placement。若 router 采用 group-limited routing，模型只允许 token 在少数 expert groups 中选择，就可以把 group 与节点对应，限制跨节点 fan-out。但这是模型语义的一部分，serving runtime 不能自行删除合法候选来省通信。

## Prefill 和 Decode 是两种 EP workload

### Prefill

一次 forward 有较多 token，dispatch payload 大，expert GEMM 的 $m$ 维也更可观：

- 更接近带宽受限的大消息通信；
- count/launch 固定成本容易摊薄；
- grouped GEMM 能形成较大工作块；
- 可通过 chunking 控制单轮 buffer 峰值。

Prefill dispatcher 通常追求 throughput，允许更大的通信 buffer 和更深的流水线。

### Decode

每条 sequence 每步只产生一个 token。Continuous batching 会把许多 sequence 合并，但在线 SLO 限制了 batch 等待：

- payload 较小且高频；
- kernel launch、通知和同步占比上升；
- 每个 expert 可能只收到少数 token；
- 等待热点 rank 会直接抬高 TPOT。

Decode dispatcher 更关注 latency，可能使用不同协议、固定预注册 buffer、低延迟 kernel或较少的 CPU 参与。DeepEP 也把高吞吐与低延迟 dispatch/combine 分为不同路径。

同一个 EP size 在 Prefill benchmark 上表现很好，不代表适合 Decode。P/D 解耦部署可以分别设置并行度：Prefill pool 用更高吞吐配置，Decode pool 用更低通信延迟配置，只要 KV layout 与传输协议兼容。

## 负载倾斜同时放大计算和通信尾延迟

设 expert $e$ 收到 $n_e$ 个 assignments，平均值为 $\bar n=N_a/E$。一个简单的 expert imbalance 指标是：

$$
I_e = \frac{\max_e n_e}{\bar n}
$$

但 EP 真正等待的是 rank，而一个 rank 可能持有多个 experts。rank $r$ 的负载为：

$$
L_r = \sum_{e\in\mathcal E_r} n_e
$$

因此还要计算：

$$
I_r = \frac{\max_r L_r}{N_a/P_e}
$$

expert-level 看似均衡，不代表 rank-level 均衡；两个次热点 experts 恰好放在同一 rank，也会形成 straggler。反过来，复制热点 expert 并让 assignments 在副本间分流，可以改善 rank balance，但会增加权重显存和放置复杂度。

推理 runtime 必须尊重 checkpoint 的 router 分数与 Top-$k$ 选择，不能为了均衡随意改选另一个 expert。可做的通常是副本选择、expert placement、请求混批和模型已经定义的路由约束。

## 通信与计算怎样重叠

dispatch 完成前，某个 expert 不必等待全局所有 token；只要它需要的输入段到齐，就可以开始 GEMM。由此可以构造流水线：

```text
dispatch chunk 0 ──► GEMM chunk 0 ──► combine chunk 0
      dispatch chunk 1 ──► GEMM chunk 1 ──► combine chunk 1
            dispatch chunk 2 ──► ...
```

实际 overlap 受以下条件限制：

- count/offset 是否必须在 payload 前全局确定；
- expert GEMM 能否消费部分到达的数据；
- communication kernel 占用多少 SM；
- NIC、NVLink、HBM 与 GEMM 是否争用同一资源；
- chunk 太小时，通知和 launch 是否反而占主导；
- combine 能否在其他 experts 尚未完成时独立返回。

“使用了两个 CUDA streams”并不证明通信被隐藏。应在 GPU timeline 上测量真正并发的区间，并比较：

$$
T_{layer}
\quad\text{vs}\quad
T_{dispatch}+T_{gemm}+T_{combine}
$$

若三段单独耗时之和明显大于 layer wall time，才说明有有效 overlap；还要确认 overlap 没有拖慢 GEMM 本身。

## 显存账不只有 expert 权重

EP 节省本地 expert weights，但需要额外 buffer：

```text
local expert weights
+ router logits / top-k metadata
+ send and receive payload buffers
+ permuted token buffer
+ grouped GEMM output/workspace
+ combine buffer
+ counts, offsets and handles
+ transport registration / staging memory
```

动态分配这些 buffer 会引入 allocator 抖动，也难以被 CUDA Graph 捕获。工程实现常按最大 token 数或分桶 shape 预分配，再用 valid count 控制实际工作。

buffer 上界不能只用 `max_num_sequences`。Prefill 一条请求可能贡献许多 token，Top-$k$ 又把 assignment 放大 $k$ 倍，还要考虑最坏路由倾斜。若为每个 expert 按全局最坏值分配，内存会极度浪费；若只按平均值分配，热点会溢出。需要在容量因子、动态 buffer、chunking 与拒绝策略之间取舍。

## 正确性检查比性能调优更早

EP 的常见错误不是直接 crash，而是 token 被静默串位。上线性能优化前，应建立以下不变量：

### Assignment 守恒

不允许 dropping 的模型应满足：

$$
\sum_r N_{send,r}
=\sum_r N_{recv,r}
=N_tk
$$

若存在 padding，要分别统计 valid assignments 与 padded rows。

### Expert 所有权一致

接收端每个 assignment 的 expert id 必须属于本 rank，并与 checkpoint placement 一致。

### Combine 归位一致

每个 source token 恰好接收 $k$ 份对应 output；inverse permutation 与 Top-$k$ slot 不可重复或遗漏。

### 单卡参考对齐

在小模型和固定输入上，把分布式输出与单卡/不切分实现逐层比较。测试应覆盖：

- token 全发本地；
- token 全发同一个远端 rank；
- 极端 expert 倾斜；
- 空 expert；
- token 数不是 alignment 倍数；
- Top-1、Top-2 和模型实际 Top-$k$；
- padding、capacity 与无效 token；
- 多节点和 rank 重排。

通信低精度还要单独设置数值误差阈值，不能把所有差异归因于 FP8。

## 评测 EP 应记录什么

只有端到端 tokens/s 很难定位问题。至少记录四组数据：

### Workload

```text
prefill/decode
num_tokens and num_sequences
hidden size, expert FFN size
num_experts, top-k
model and adapter revision
```

### Parallel topology

```text
DP / TP / PP / CP / EP / ETP sizes
experts per rank
rank-to-GPU-to-node mapping
NVLink/NVSwitch and NIC topology
```

### 路由形状

```text
assignments per expert
assignments per destination rank
local vs remote ratio
max-to-mean expert and rank imbalance
empty / padded / dropped assignments
```

### 时间与资源

```text
count + permute latency
dispatch latency and bytes
grouped GEMM latency and m-distribution
combine + inverse permute latency
actual overlap interval
SM, HBM, NVLink and NIC utilization
```

测试矩阵应分开扫 token batch、EP size、Top-$k$、节点数与通信 dtype。报告峰值带宽时，要说明是 logical bandwidth 还是物理 NIC bytes；包含本地 traffic 的逻辑值不能直接与网卡线速比较。

## 选择并行配置的一条推导路径

假设要部署一个 64 experts、Top-2 的模型，可以按以下顺序推导：

1. 计算 Attention、共享权重、全部 experts、KV Cache 和 workspace 的显存；
2. 找到能放下本地 expert shard 的最小 EP size；
3. 检查单 expert 是否仍需 ETP，若能放下则先比较 ETP=1；
4. 为 Attention 独立选择 TP/CP，不强迫 MoE 沿用相同切分；
5. 把 EP group 映射到 NVLink domain，记录何时必须跨节点；
6. 用实际路由 trace 计算 $C_{ij}$、$I_e$ 和 $I_r$；
7. 分别测 Prefill 和 Decode 的 dispatch/GEMM/combine；
8. 若通信主导，再比较 AllGather、All-to-All 与专用 dispatcher；
9. 若 straggler 主导，先处理 expert placement/replica 与请求混批；
10. 最后评估低精度通信、overlap 和更大的 EP。

这个顺序先确定容量可行性，再定位通信还是计算问题，避免一开始同时改变 EP、量化、dispatcher 和 expert placement，最后无法解释收益来自哪里。

## 小结

Expert Parallel 通过“每个 rank 只保存部分 experts”解决 MoE 权重容量，但代价是 token assignment 在每个 MoE layer 进行 dispatch 和 combine 两次交换。

这条数据路径可以归纳为五个判断：

1. Expert 计算的工作单位是 `token × Top-k` 的 assignment；通信 payload 还取决于本地命中和同 destination 去重；
2. 路由动态决定 All-to-All-v 的 count matrix，最热 destination 往往决定尾延迟；
3. EP size 增大能省权重显存，却提高远端比例、peer 数与小 GEMM 风险；
4. Prefill 需要吞吐路径，Decode 需要低延迟路径，不能用同一组大 batch 数据下结论；
5. dispatch handle、generation 和 inverse permutation 的正确性先于任何带宽优化。

下一层优化才是选择具体通信库。知道 payload 怎样形成、哪些元数据必须保存、瓶颈来自跨节点还是 rank 倾斜之后，才能理解 DeepEP 一类 dispatcher 为什么要同时重写 permutation、通知、RDMA 与 combine，而不是简单封装一次 collective。

## 参考资料

- [Megatron Core: Parallelism Strategies Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)
- [Megatron Core: Mixture of Experts](https://docs.nvidia.com/megatron-core/developer-guide/nightly/user-guide/features/moe.html)
- [Megatron Core: MoE API Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/moe.html)
- [DeepEP: Expert Parallel Communication Library](https://github.com/deepseek-ai/DeepEP)
- [vLLM: Expert Parallel Deployment](https://docs.vllm.ai/en/stable/serving/expert_parallel_deployment/)
- [GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://arxiv.org/abs/2006.16668)
