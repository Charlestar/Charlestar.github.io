---
layout: post
title: "Collective Communication：AllReduce、AllGather、ReduceScatter 与 All-to-All"
subtitle: "从 Tensor 所有权和通信量出发，理解分布式训练中的数据重排"
date: 2026-08-20 09:00:00 +0800
last_modified_at: 2026-09-01
author: iStar
catalog: true
series: distributed-training
series_order: 5
technology_year: 2015
mathjax: true
tags: [分布式训练, GPU优化]
---

在分布式训练代码里，`all_reduce`、`all_gather`、`reduce_scatter` 和 `all_to_all` 往往只是几行 API；在系统时间线上，它们却可能决定 GPU 是持续计算，还是集体等待网络。更麻烦的是，同一个名字下还叠着三层不同问题：

1. **数学语义**：哪些 rank 提供什么输入，操作结束后谁拥有哪一部分结果；
2. **通信算法**：Ring、Tree、recursive doubling 等算法如何实现这份语义；
3. **物理执行**：数据经过 NVLink、PCIe、网卡还是交换机，怎样切 chunk、占用 channel，并与 CUDA stream 上的计算排序。

如果把这三层混在一起，就会产生许多似是而非的结论。例如“Data Parallel 就是 AllReduce”“AllReduce 一定使用 Ring”“AllGather 的通信量是模型大小乘 GPU 数”“异步 collective 返回后 tensor 就能立即读取”。这些说法都只在特定定义或前提下成立。

本文先建立 rank、communicator 和 tensor ownership 的统一模型，再逐个推导常见 collective 的输入、输出和通信量，最后把它们落到 DP、FSDP、TP、CP 与 EP。重点不是记 API，而是看到任意一段并行代码时，能够回答三个问题：

```text
collective 前：每个 rank 持有什么？
collective 中：线上至少移动多少数据，关键路径有多少轮？
collective 后：结果复制在哪里，又沿哪个维度分片？
```

## Rank 不是 GPU 的永久名字

一个分布式进程加入通信组后，会在该组中获得从 `0` 到 `p-1` 的 **rank**，其中 `p` 是 group size。rank 是 communicator 内的逻辑编号，不是机器编号，也不是永远固定的 CUDA device id。

例如同一物理 GPU 上的进程可以同时属于多个 group：

```text
world group:  [0,1,2,3,4,5,6,7]
TP group A:   [0,1,2,3]
TP group B:   [4,5,6,7]
DP group 0:   [0,4]
DP group 1:   [1,5]
DP group 2:   [2,6]
DP group 3:   [3,7]
```

物理进程 `world rank 4` 在 `TP group B` 内可能是 local rank 0，在 `DP group 0` 内却是 local rank 1。NCCL 文档特别提醒，Broadcast/Reduce 的 `root` 参数是 communicator 中的 rank，而不是 device number。把 world rank、group rank 和 local CUDA device 混淆，轻则把 shard 排错，重则让不同进程进入不匹配的 collective 而永久等待。

## Communicator 定义了“谁必须一起出现”

Communicator 不只是一个连接句柄，它至少规定了：

- 成员集合及其 rank 顺序；
- collective 的匹配上下文；
- 可使用的传输路径与内部拓扑；
- 与故障、abort 和重建相关的生命周期。

一次 collective 的参与范围是 communicator，而不是整个 job。TP AllReduce 只需要同一个 TP group 的 ranks；DP 梯度同步只需要拥有同一份模型分片的 DP ranks；EP All-to-All 只在共享一组 experts 的 EP group 内发生。

因此，“8 卡训练”的通信量不能只看全局 GPU 数。若设备网格为 `DP=2, TP=4`，TP collective 的 $p$ 是 4，DP collective 的 $p$ 是 2，而且两个 group 可能映射到完全不同的链路。

## Tensor Ownership 才是 Collective 的起点

描述 tensor 时，不只写 shape，还要写 placement。常见状态包括：

| Placement | 每个 rank 持有的内容 | 典型来源 |
| --- | --- | --- |
| Replicated | 相同的完整 tensor | AllReduce 输出、参数副本 |
| Sharded | 完整 tensor 的互不重叠切片 | FSDP 参数、Sequence Parallel activation |
| Partial | 完整结果的一部分贡献，仍需求和 | Row Parallel GEMM 输出 |
| Routed | 按目的 rank 重排后的不等长记录 | MoE expert assignments |
| Root-owned | 只有 root 拥有有效完整结果 | Reduce/Gather 输出 |

`Sharded` 和 `Partial` 尤其容易混淆。假设完整向量为 $y$：

```text
sharded:
rank 0 owns y[0:2]
rank 1 owns y[2:4]

partial:
rank 0 owns y^(0), rank 1 owns y^(1)
and y = y^(0) + y^(1)
```

前者需要拼接，即 AllGather；后者需要逐元素归约，即 AllReduce。ReduceScatter 则把各 rank 的 full-shape `Partial` contributions 一边归约、一边分配最终 owner，直接完成 `Partial → reduced Sharded`；这个过程不要求先产生一份 replicated full tensor。

## 先统一通信量的记号

设 communicator 有 $p$ 个 ranks，元素 dtype 占 $d$ bytes。本文用 $B$ 表示一次操作所处理的**完整逻辑 tensor**大小：

$$
B=N\times d
$$

但不同 collective 的本地输入并不都等于 $B$：

- AllReduce：每个 rank 输入 $B$，输出也是 $B$；
- AllGather：每个 rank 输入 $B/p$ 的 shard，输出 $B$；
- ReduceScatter：每个 rank 输入 $B$ 的 partial tensor，输出 $B/p$；

All-to-All 不对应上述“每 rank 共同描述同一个 $B$”的 placement 变换，因此单独把每个 rank 的发送缓冲区记作 $B_{local}$；全体 ranks 的应用层输入总量是 $pB_{local}$。等长交换时，每个 source→destination 块约为 $B_{local}/p$，每个 destination 从 $p$ 个 sources 合计接收 $B_{local}$，其中包括本地块。

后文“每 rank 线上发送量”不计算本地 self-copy，也不等同于应用层 send buffer 大小。链路重传、协议头、对齐、跨层级转发以及双向收发会让硬件计数与这个理想值不同。

## $\alpha$–$\beta$ 模型：小消息与大消息为什么表现相反

经典 Hockney 风格模型把一次发送 $m$ bytes 的时间近似写成：

$$
T(m)=\alpha+\beta m
$$

其中：

- $\alpha$ 是启动一次通信步骤的固定延迟，包括软件调度、同步和链路启动；
- $\beta$ 是每 byte 的传输时间，即有效带宽的倒数；
- 对 Reduce 类操作，还可加入本地归约代价 $\gamma m$。

这个模型不是精确的硬件模拟，却能解释算法选择：小消息由步骤数和 $\alpha$ 主导，偏好 $O(\log p)$ 轮的 Tree 或 recursive doubling；大消息由 $\beta B$ 主导，偏好让每条链路持续搬运分块数据的带宽型算法。MPICH 的经典 collective 优化论文正是按 message size、process count 和 reduction operator 选择不同算法，而不是寻找一种永远最优的实现。

还要注意，critical-path time、每 rank 发送量、所有 ranks 的 aggregate traffic 和某条瓶颈链路上的 bytes 是四个不同指标。只报其中一个，不能完整说明性能。

## Broadcast：一个 Owner 变成所有人 Replicated

Broadcast 的输入和输出是：

```text
before:
root rank owns X, other receive buffers have no useful X

after:
every rank owns the same X
```

若 $X$ 为 $B$ bytes，信息至少要抵达 $p-1$ 个非 root ranks，因此全局有效交付量为 $(p-1)B$。但这不意味着 root 必须串行发送 $p-1$ 次。树形传播可以让已收到数据的 rank 继续转发，理想深度约为 $\lceil\log_2 p\rceil$；大消息还可切成 chunks，在树上流水传递。

Broadcast 常用于：

- 从某个 rank 分发初始化参数、随机种子或小型控制信息；
- checkpoint 加载后把 root-owned tensor 复制到 group；
- 保持必须完全一致的配置或状态。

它不天然等于 barrier。MPI 标准明确指出，不能依赖 Broadcast 具有特定同步效果；“大家最终拿到同一 tensor”和“大家此刻已经同时到达某一代码点”是不同语义。

## Reduce：所有 Partial 汇入一个 Root

设每个 rank 有同 shape 的输入 $X_r$，Reduce 计算：

$$
Y=\bigoplus_{r=0}^{p-1}X_r
$$

但只有 root 的 receive buffer 获得 $Y$。$\oplus$ 可以是 sum、min、max 等受支持的归约运算。

```text
rank 0: X0 ┐
rank 1: X1 ├─ sum/min/max ─→ Y only on root
rank 2: X2 ┤
rank 3: X3 ┘
```

对梯度训练而言，结果通常还要继续被所有 ranks 使用，所以单独 Reduce 并不常见；对日志聚合、root-only 决策或随后只由 root 保存的统计量，它却比 AllReduce 更符合 ownership。语义上，Reduce 后接 Broadcast 可以构成 AllReduce，但实现库可以使用更高效的一体化算法。

## AllReduce：Partial 变成 Replicated

AllReduce 计算同样的归约：

$$
Y=\bigoplus_{r=0}^{p-1}X_r
$$

区别是每个 rank 最终都得到完整 $Y$：

```text
before: rank r owns partial contribution Xr, each of size B
after:  every rank owns identical reduced Y, size B
```

这正好匹配标准 Data Parallel 梯度：每个副本从不同 mini-batch 得到本地梯度，先求和或平均，再以相同梯度更新相同参数。

### Ring AllReduce 怎样拆成两半

设 $B$ 可以均匀切成 $p$ 个 chunks。Ring AllReduce 通常可理解为：

1. **ReduceScatter**：经过 $p-1$ 轮邻居传输与归约，每个 rank 获得一个已经全局求和的 chunk；
2. **AllGather**：再经过 $p-1$ 轮传输，让所有 ranks 收齐全部 reduced chunks。

四个 ranks 的逻辑环为：

```text
rank 0 → rank 1 → rank 2 → rank 3 → rank 0
```

每轮每个 rank 发送一个约 $B/p$ 的 chunk。忽略 self-copy、协议与 padding，每 rank 的发送量为：

$$
V_{ring\ allreduce}
=2(p-1)\frac{B}{p}
$$

当 $p$ 很大时趋近 $2B$，而不是 $pB$。一个简化的时间模型是：

$$
T_{ring}
\approx 2(p-1)\alpha
+2\frac{p-1}{p}B\beta
+\frac{p-1}{p}B\gamma
$$

Ring 的优势是大消息下每个 rank 的数据量接近带宽下界，且稳定地流过邻接链路；代价是步骤数随 $p$ 线性增长，小消息容易被 $2(p-1)\alpha$ 支配。

### Tree AllReduce 为什么更适合延迟敏感区间

最容易理解的 Tree AllReduce 是先沿树向 root Reduce，再沿树 Broadcast。若每一步传完整 $B$，二叉树关键路径约为：

$$
T_{tree,simple}
\approx 2\lceil\log_2p\rceil(\alpha+B\beta)
$$

它的轮数少，因此小到中等消息更有吸引力；朴素树在靠近 root 的链路上却可能重复搬运完整大消息。实际 GPU collective 库会用分块、双树、多 channel 等方法改善带宽和链路利用率，所以这条公式只用于比较倾向，不是在预测某个 NCCL 版本的确切实现。

NVIDIA 的 NCCL 调优说明也把 Ring 概括为峰值带宽利用率较好、Tree 在中等消息上凭借对数深度表现突出。最终选择还取决于 topology、protocol、channel 数量和运行时成本模型。

### AllReduce 不只是“Reduce 加 Broadcast”的 API 拼接

从语义上：

$$
\operatorname{AllReduce}
=\operatorname{Broadcast}\circ\operatorname{Reduce}
$$

也有：

$$
\operatorname{AllReduce}
=\operatorname{AllGather}\circ\operatorname{ReduceScatter}
$$

第二个恒等式对分片训练尤其重要：如果下一段计算只需要一份 reduced shard，就可以停在 ReduceScatter，不必立即 AllGather。Sequence Parallel 和 FSDP 的价值，部分正来自“不要过早恢复 replicated placement”。

这里说的是有条件的数学等价，而不是任意两个 API 调用都可替换：操作必须使用同一 communicator 与 rank order、匹配的 count/dtype 和 reduction operator；ReduceScatter 还要求输出能按 rank 切成等长 blocks，后续 AllGather 按同一顺序重建。Reduce+Broadcast 也必须约定同一个 root。即使条件都满足，不同算法的浮点归约顺序仍可能不同，因此结果不保证 bitwise identical。

## AllGather：Sharded 变成 Replicated

设完整 tensor $X$ 沿某一维被切成 $p$ 个 rank-ordered shards：

$$
X=[X_0,X_1,\ldots,X_{p-1}]
$$

每个 $X_r$ 大约 $B/p$ bytes。AllGather 后，每个 rank 按 rank 顺序得到完整 $X$：

```text
before: rank r owns Xr, size B/p
after:  every rank owns [X0 | X1 | ... | X(p-1)], size B
```

Ring AllGather 经过 $p-1$ 轮，每 rank 的理想线上发送量是：

$$
V_{allgather}=\frac{p-1}{p}B
$$

容易混淆的是 API 视角：若把“每 rank 输入大小”记为 $b$，那么输出为 $pb$，线上发送量为 $(p-1)b$。两种写法完全一致，因为 $B=pb$。

AllGather 只拼接，不做求和。它依赖 shard 到 rank 的映射；若模型保存时的 shard order 与新 communicator 的 rank order 不一致，即使 collective 成功返回，重建的 tensor 也可能在语义上错误。

## ReduceScatter：Partial 变成 Sharded

每个 rank 输入一个完整 shape 的 partial tensor $X_r$，先逐元素归约为：

$$
Y=\bigoplus_r X_r
$$

再让 rank $r$ 只保留 $Y_r$：

```text
before: every rank owns a full-sized partial contribution, B bytes
after:  rank r owns reduced shard Yr, B/p bytes
```

Ring ReduceScatter 每 rank 的理想线上发送量为：

$$
V_{reducescatter}=\frac{p-1}{p}B
$$

它不是 Reduce 后再由 root Scatter 的必然实现。边传输、边归约、边确定最终 owner，可以避免先在某个 root materialize 完整结果，也能让输出直接进入 sharded optimizer 或 sharded activation 的下一阶段。

## All-to-All：Sharded 不是唯一的分片方式

All-to-All 的输入不是“大家各持有完整 tensor 的同一种 contribution”，而是每个 source rank 为每个 destination 准备不同的数据。等长版本中，每个 rank 的发送缓冲区被分成 $p$ 块：

```text
source rank i send buffer:
[ to rank 0 | to rank 1 | ... | to rank p-1 ]

destination rank j receive buffer:
[ from rank 0 | from rank 1 | ... | from rank p-1 ]
```

若每 rank send buffer 总大小为 $B_{local}$，每个 peer 块约 $B_{local}/p$，排除本地块后的理想线上发送量为：

$$
V_{alltoall}=\frac{p-1}{p}B_{local}
$$

这个公式的系数看起来和 AllGather 相同，但两者的 $B$ 定义和数据语义都不同：AllGather 把每个 rank 的一个全局 tensor shard 分发给所有 peers；All-to-All 则从每个 rank 独立的 $B_{local}$ 中，给不同 peers 发送不同 slices。它更像一次全局 transpose 或 personalized exchange，不能只看系数就断言两种操作搬运了同一规模的全局数据。

## Ragged All-to-All：MoE 的真实输入通常不等长

标准等长 All-to-All 假设所有 source-destination pair 的块大小相同。MoE router 却可能把大量 tokens 发给某个热门 expert，而另一个 expert 几乎没有 token。此时需要 variable-split All-to-All，也就是 MPI 中的 `Alltoallv` 或 PyTorch `all_to_all_single` 的 split-size 形式。

设 $c_{ij}$ 表示 source rank $i$ 发往 destination rank $j$ 的元素数。合法交换必须满足每一对 peers 的发送与接收一致：

$$
sendcount_{i\rightarrow j}
=recvcount_{j\leftarrow i}
$$

常见流程是：

```text
router output
  → count assignments per destination
  → exchange counts
  → prefix-sum offsets
  → pack/permute payload
  → variable-split All-to-All
  → local expert compute
  → reverse variable-split All-to-All
  → unpermute/combine
```

只交换 payload 而没有先对齐 counts，会造成 buffer 越界、数据截断或 collective 不匹配。为了使用更规则的等长通信，有些系统会按 capacity padding；这样减轻了 shape 管理，却会为 padding token 付出网络和计算代价。选择 ragged 还是 padded，不只是 API 偏好，而是负载不均衡、kernel shape 与通信规则性之间的权衡。

## 六种 Collective 的 Ownership 对照

| 操作 | 每 rank 输入 | 结束后的 owner | Ring/理想化每 rank 线上发送量 |
| --- | --- | --- | ---: |
| Broadcast | root 有 $B$ | 所有 ranks 各有 $B$ | 取决于树/链路；全局有效交付 $(p-1)B$ |
| Reduce | 每 rank 有 partial $B$ | root 有 reduced $B$ | 取决于算法 |
| AllReduce | 每 rank 有 partial $B$ | 所有 ranks 各有 reduced $B$ | $2(p-1)B/p$ |
| AllGather | 每 rank 有 shard $B/p$ | 所有 ranks 各有完整 $B$ | $(p-1)B/p$ |
| ReduceScatter | 每 rank 有 partial $B$ | 每 rank 有 reduced shard $B/p$ | $(p-1)B/p$ |
| All-to-All | 每 rank 有按目的地切分的 $B_{local}$ | 每 rank 收到来自各 source 的 $B_{local}$ | $(p-1)B_{local}/p$ |

表中的 bytes 是理解下界的工具，不是 `nccl-tests` 必然显示的数值。不同算法可能让同一 byte 经过多个物理 hops；多 rail、switch offload、双向链路以及层级化算法也会改变“该除以哪条带宽”。

## Data Parallel：梯度为什么落到 AllReduce

标准 DDP 在每个 rank 上保留完整参数，并处理不同 mini-batch。若本地梯度为 $g_r$，全局平均梯度为：

$$
g=\frac{1}{p}\sum_{r=0}^{p-1}g_r
$$

AllReduce 让每个 rank 都得到相同 $g$，随后各自运行相同 optimizer update。此处 gradient 在 collective 前是 `Partial`，后是 `Replicated`。

上式默认每个 rank 的 $g_r$ 使用相同 loss normalization，且包含相同数量的有效样本或 tokens。若 rank $r$ 的有效权重为 $w_r$，sample-wise 或 token-wise 全局平均应是 $\sum_r w_rg_r/\sum_r w_r$，不能无条件把各 rank 的局部平均再等权平均。最后一个不齐的 batch、变长 sequence 和动态丢样本都会触发这条边界。

若全部梯度总大小为 $G$ bytes，Ring AllReduce 的理想每-rank 线上发送量约为：

$$
2\frac{p-1}{p}G
$$

实际 DDP 通常不会等 backward 全部结束后才发送一块 $G$。它把 gradients 装入 buckets，当某个 bucket 的参数梯度都 ready 时就发起 collective，与更早层的 backward 计算重叠。bucket 太小会增加 $\alpha$ 与 kernel launch 开销；太大则推迟首个通信，减少 overlap 窗口。

## FSDP/ZeRO：AllGather 与 ReduceScatter 构成状态生命周期

在 fully sharded data parallel 中，参数平时按 DP ranks 分片。某个模块计算前，需要把 parameter shards AllGather 成完整计算参数；backward 得到的 full-shape partial gradients 则通过 ReduceScatter 直接变成各 rank 应保存的 reduced gradient shard。关键问题是完整参数在 forward 后是否立即 reshard，这会改变一次 iteration 的 AllGather 次数。

```text
parameter shard B/p
  → AllGather
full parameter B for forward
  → release/reshard after forward

parameter shard B/p
  → AllGather before backward
full parameter B for backward

full-shaped partial gradient B
  → ReduceScatter
reduced gradient shard B/p
  → local optimizer update
```

上图对应 PyTorch FSDP `FULL_SHARD`、ZeRO-3 一类“forward 后立即释放完整参数”的典型生命周期。忽略预取、重计算和共享参数等额外因素，两个 AllGather 加一个 ReduceScatter 的理想每-rank 线上发送量为：

$$
3\frac{p-1}{p}B
$$

若配置选择在 forward 后保留完整参数直到 backward 完成，就只需一个 AllGather 和一个 ReduceScatter，通信量才是 $2(p-1)B/p$，与 Ring AllReduce 的理想量级相同；代价是完整参数驻留更久、峰值显存更高。FSDP 的目标不是凭空消除通信，而是在通信量和显存之间选择生命周期，并把 parameters、gradients、optimizer states 的长期 ownership 保持为 sharded。

## Tensor Parallel：Partial Output 为什么需要求和

Megatron-style Tensor Parallel 把一对线性层分别做 column 和 row partition。Row-parallel linear 的每个 rank 只计算 reduction dimension 的一部分：

$$
Y_r=X_rW_r
$$

完整输出为：

$$
Y=\sum_{r=0}^{p-1}Y_r
$$

因此 $Y_r$ 是 `Partial` 而不是普通 shard，需要 AllReduce 才能恢复每 rank 都可使用的 replicated residual。Attention output projection 也有类似落点。

TP collective 频率高，往往一层出现多次，而且处在前向与反向关键路径。即使 activation 比全模型梯度小，频繁的小到中等消息也可能更受 $\alpha$、拓扑和同步等待影响。因此，DP 与 TP 不应只比较“总 bytes”，还要比较 collective 数量、大小分布和是否能 overlap。

## Sequence Parallel：把 AllReduce 的中间状态保留下来

传统 TP block 在 row-parallel 输出后 AllReduce，使 residual 在 TP group 内复制。Sequence Parallel 利用：

$$
\operatorname{AllReduce}
=\operatorname{AllGather}\circ\operatorname{ReduceScatter}
$$

先用 ReduceScatter 完成求和并只保留 sequence shard，让 LayerNorm、dropout、residual add 等 token-local 操作在 sharded activation 上执行；进入需要另一种布局的线性层前再 AllGather。

```text
row-parallel partial activation
  → ReduceScatter(sequence)
sequence-sharded activation
  → local elementwise / normalization work
  → AllGather(sequence)
replicated-or-TP-ready activation
```

它未必减少这一对 collectives 的理想总通信量，主要收益是缩短 replicated activation 的驻留时间。把“通信量”和“峰值显存 placement”分开，才能理解为什么这项技术值得存在。

## Context Parallel：有时并不使用单一 Collective

Context Parallel 把输入 sequence 长期切在多个 ranks 上，但 Attention 需要访问全局上下文。根据实现，通信可能是：

- AllGather K/V 后本地计算完整 attention；
- Ring P2P 逐块传递 K/V，并在线合并 softmax 统计量；
- All-to-All 在 sequence-sharded 与 head-sharded layout 间转置；
- 分层或混合模式，在节点内与节点间选择不同路径。

因此 CP 的正确抽象是“为全局 Attention 建立所需的数据依赖”，不是“CP 等于某一个 collective”。选择方案时要同时比较 K/V bytes、临时显存、causal load balance、通信是否能与 attention tile 计算重叠。

## Expert Parallel：All-to-All 搬运的是 Assignment

在 MoE 中，router 为每个 token 选出 Top-$k$ experts。若一共有 $N_t$ 个 tokens，则通信工作单位通常是：

$$
N_a=N_t\times k
$$

个 expert assignments，而不是 $N_t$ 个原始 tokens。dispatch All-to-All 把 hidden states 发到 expert owners；expert 计算完成后，combine All-to-All 把结果送回原 token owner。

```text
source-token layout
  → route + permute
  → dispatch All-to-All(v)
expert-owner layout
  → grouped GEMM
  → combine All-to-All(v)
source-token layout
  → weighted combine + residual
```

如果 router 极不均衡，平均 bytes 看起来不大，最热 destination 仍会成为 straggler。EP 性能诊断必须同时看 send-count matrix、每 expert token count、padding/capacity、dispatch 与 combine 两个方向，而不能只看一个 All-to-All 的总耗时。

## Pipeline Parallel：相邻 Stage 主要是 P2P

Pipeline Parallel 把层按深度切到不同 stages。相邻 stages 之间通常传 activation 与 gradient，主要语义是 `send/recv`，不是整个 group 参与的 collective。

不过 PP 并没有让 collective 消失：每个 stage 内仍可能有 TP/CP/EP groups；不同 pipeline replicas 之间仍有 DP/FSDP group。一个 rank 同时参与多类 communicator 时，collective 顺序与 stream 依赖变得更复杂，这也是生产环境中“单独 microbenchmark 都很快，组合后却 hang 或抖动”的常见来源。

## Ring、Tree 不是 Collective 的固有属性

AllReduce 是语义，Ring 和 Tree 是实现算法。库可以根据 message size、rank count、topology、可用 channel 与 protocol 选择算法；同一个训练 step 中，不同 bucket 甚至可能走不同路径。

可以用下面的倾向建立直觉：

| 条件 | 常见倾向 | 原因 |
| --- | --- | --- |
| 小消息、rank 多 | Tree/recursive doubling | 减少 $\alpha$ 主导的步骤数 |
| 大消息、规则拓扑 | Ring/带宽型算法 | 每 rank bytes 接近带宽下界 |
| 跨节点且层级明显 | Hierarchical | 先节点内归并，再走稀缺的节点间链路 |
| 不规则 All-to-All | Pairwise/分层/专用交换 | 目的地与大小均不相同 |

这些不是手工强制算法的理由。成本模型还要知道真实拓扑和并发流量；环境变量固定某个算法，可能修复一台集群，也可能破坏另一台集群或另一种 message size。

## Chunk：一块 Tensor 怎样进入流水线

大 tensor 通常不会作为一个不可分割整体走完所有 hops。通信库会把它切成 chunks：当 chunk 0 前进到下一跳时，当前链路可以处理 chunk 1，其他 rank 也可以对已经收到的 chunk 做 reduction。

Chunk 太大：

- pipeline 难以填满，首字节到末级的延迟更高；
- 无法充分利用多条链路与多个执行单元。

Chunk 太小：

- 协议、队列、同步和 kernel bookkeeping 占比升高；
- GPU thread block 做的数据不足；
- outstanding work 过多，资源压力增加。

因此“把 bucket 调小以便更早 overlap”与“让 collective 大到能吃满带宽”之间存在张力。训练框架的 bucket 与通信库内部 chunk 还是两个层次，不能把参数名相同的概念直接等同。

## Channel：一条逻辑 Ring 可以并行成多条数据通道

Channel 可以理解为通信库对数据和执行资源的并行划分。多个 channels 可能使用不同 ring/tree 实例或不同链路，让 GPU、NVLink/NIC 和交换结构同时工作。

增加 channel 不保证线性提速：

- 链路或 NIC 已饱和时，只会增加竞争；
- 更多 communication CTAs 会争用 SM，与计算 overlap 时可能反而拖慢 GEMM；
- 小消息被切得过碎，固定开销放大；
- topology 不对称时，多 channel 可能共同撞在同一个瓶颈 hop。

本文只建立 chunk/channel 的抽象。具体 NCCL 如何选择 algorithm、protocol、CTA 和 channel，适合在后续 NCCL 专题中结合版本与硬件单独讨论。

## Topology：逻辑相邻不等于物理相邻

同一个 communicator 的 ranks 可以跨越：

```text
GPU
 ├─ NVLink / NVSwitch
 ├─ PCIe switch / CPU root complex
 └─ NIC ─ InfiniBand or RoCE fabric ─ remote NIC ─ remote GPU
```

逻辑 Ring 若把经由 NVLink 可直达的 GPU 排成远距离 PCIe 路径，就会在最慢 hop 上受限。跨节点 collective 若让同一 node 多次穿越稀缺 uplink，也可能放大 oversubscription。Tree 的 root、branch placement 和 rail 使用同样重要。

因此 rank mapping 是性能输入，不只是编号。诊断时至少记录 hostname、local device、PCI bus id、NIC affinity、NUMA、communicator rank 与链路类型。本文不展开 NVLink、NVSwitch、PCIe、InfiniBand 和 RoCE 的物理细节，它们将作为后续互联拓扑文章的主线。

## CUDA Stream：API 返回不代表结果可被 Host 或其他 Stream 读取

NCCL collective 通常把通信 kernel enqueue 到给定 CUDA stream，对 host 异步执行。正确依赖应由同一 stream 的先后顺序，或 CUDA event 建立：

```text
compute_stream: produce tensor ─ record(event_ready)
comm_stream:    wait(event_ready) ─ collective ─ record(event_done)
compute_stream: wait(event_done) ─ consume result
```

少了前一个 wait，通信可能读到尚未完成的输入；少了后一个 wait，消费者可能读到尚未完成的结果。调用 `async_op=True` 或拿到 work handle，只表示 host 不阻塞，并不表示数据已就绪。应按框架契约等待 work、stream 或 event，避免用全局 `cudaDeviceSynchronize` 把所有潜在 overlap 一次抹掉。

还要区分“enqueue 已完成”和“GPU 执行已完成”。NCCL 非阻塞 communicator 的 `ncclGroupEnd()` 甚至可能返回 `ncclInProgress`，表示 kernels 仍在后台发射；官方文档要求先轮询 communicator 状态，再执行相关 CUDA 同步操作。

若使用 `ncclGroupStart()`/`ncclGroupEnd()` 聚合多个调用，组内单个 API 返回时甚至可能还没有把 operation enqueue 到 stream；整组只会在最外层 `ncclGroupEnd()` 被整体启动。Grouping 的作用是管理多 GPU 调用或合并 launch，不会替 ranks 自动配对，也不会放宽 ordering contract。即使组内操作跨不同 communicators，各 GPU 仍须保持一致的 host-side issuance order。

## In-place 不是“任意让输入输出指向同一地址”

In-place collective 可以减少额外 buffer，但每个操作都有精确的 offset 约束。例如 NCCL 文档规定：

- AllGather in-place 时，本 rank 的 send pointer 应指向 receive buffer 中属于本 rank 的那一段；
- ReduceScatter in-place 时，receive pointer 应指向 send buffer 中属于本 rank 输出 shard 的 offset；
- AllReduce 可在相同 send/receive buffer 上原地归约。

如果只看到“支持 in-place”就把两个任意重叠 views 传进去，可能破坏尚未发出的数据。还要考虑 allocator 是否在通信完成前复用 storage、tensor 是否 contiguous、offset 是否按 communicator rank 而非 world rank 计算。

## Collective Ordering：所有 Rank 必须进入同一条协议轨道

NCCL 要求一次 collective 的所有 ranks 使用一致的 operation、count 和 datatype 共同形成完整操作；Reduce 类操作还要匹配 reduction op，rooted 操作要匹配 root，variable-split 交换则要保证 peer-wise send/receive counts 对得上。更一般地，同一 communicator 上的 collective 必须以匹配顺序出现：

```text
correct:
rank 0: AllReduce(A) → AllGather(B)
rank 1: AllReduce(A) → AllGather(B)

wrong:
rank 0: AllReduce(A) → AllGather(B)
rank 1: AllGather(B) → AllReduce(A)
```

错误版本不会像 RPC 那样靠方法名自动配对。常见结果是 hang；在参数 shape 恰好兼容时，甚至可能出现更隐蔽的数据错误。MPI 的 collective correctness 规则同样要求 group 成员以相同顺序调用 collective，并指出重叠 communicators 上的阻塞调用可能形成循环依赖。

条件分支尤其危险：若只有检测到 `NaN` 的 rank 进入一个 AllReduce，其他 ranks 跳过，检测逻辑本身就会 hang。应让所有 ranks 执行同一个 collective，把本地布尔量作为输入归约。

## Deadlock 不只来自“少了一个 Rank”

常见死锁来源包括：

- 某个 rank 因 OOM、异常或 dataloader 卡住，根本没走到 collective；
- 不同 ranks 的 control flow 导致 collective 类型或顺序不同；
- split sizes/count/dtype 不一致；
- 多个 overlapping communicators 以相反阻塞顺序获取进展；
- P2P send/recv 与 collective 形成循环依赖；
- 多线程在同一 communicator 上并发发起未定义顺序的操作；
- 一个 stream 等待另一个 stream 的 event，而后者又在等待当前 collective。

调大 timeout 只能把死锁推迟暴露，不能修复协议。排查时应先比对每个 rank 的“collective 序号、group id、op、shape、dtype、stream”，再看网络。

## Reduction 的数值语义也属于正确性

浮点加法不满足严格结合律：

$$
(a+b)+c \neq a+(b+c)
$$

Tree、Ring 或不同 chunk 顺序可能产生末位差异。rank count、rank mapping、算法、拓扑路径或 chunk 策略变化，也可能改变 reduction order；单纯把同一个固定算法与计算重叠，并不必然改变 operand order。对 BF16/FP16，内部是否使用更高精度 accumulator、最终何时 cast 回输出 dtype，同样影响误差。

因此分布式回归测试应使用合理 tolerance，并把“数学上等价”与“bitwise identical”分开。另一方面，明显的大幅偏差不能简单归因于浮点顺序，还应检查 loss normalization、gradient scaling、padding mask、重复/漏样本和 shard layout。

在 MPI 等允许 user-defined reduction 的接口里，operator 还涉及是否交换律、是否结合律；某些高性能算法会重排 operands，若 operator 不满足声明的性质，结果就没有合法语义。NCCL 的自定义能力不是任意 GPU 函数：除预定义 op 外，当前公开接口主要是受约束的 PreMulSum 一类操作。不能把 MPI 的任意 user-op 语义直接投射到 NCCL。

## Collective 的故障语义不是事务

网络错误、peer process 崩溃或 GPU fault 发生时，不能假设 collective 要么全部生效、要么全部回滚。某些 ranks 的 output buffer 可能已被部分写入，某些操作甚至仍在 stream 上排队。

NCCL 官方错误处理说明指出，异步网络错误可能让操作停止进展而永不完成；应用应查询 asynchronous error，调用 `ncclCommAbort`，并重建 communicator。对于 fatal error，原 communicator 不能被当作仍然健康继续使用。

恢复边界应放在更高层：

1. watchdog 检测超时、rank failure 和 communicator error；
2. 停止继续消费可能不完整的输出；
3. 协调所有健康 ranks abort，而不是只让一个 rank 单独退出；
4. 清理 outstanding work 与相关 streams/buffers；
5. 重新建立进程组或缩容 communicator；
6. 从最近一致 checkpoint 重放训练 step。

Collective 只完成数据交换，不提供 optimizer step 的 exactly-once。若故障发生在部分 ranks 更新参数之后，必须靠 checkpoint、step epoch 和训练控制面恢复全局一致性。

## 性能诊断第一步：先画出 Message Size 分布

不要只看“通信占了 30%”。先为每个 communicator 统计：

- op 类型与调用次数；
- payload bytes 与 dtype；
- group size；
- P50/P95/P99 duration；
- 发起时间、完成时间及与 compute overlap 的区间；
- 算法、protocol、channel（在运行库能提供时）；
- local/remote rank mapping。

大量 4 KB collectives 和少量 4 GB collectives 即使总 bytes 相同，调优方向也相反。前者考虑融合、bucket、减少同步点；后者考虑链路带宽、拓扑、分层算法与均衡多 rail。

## 性能诊断第二步：区分 Algorithm Bandwidth 与 Bus Bandwidth

若 payload 为 $B$，简单的应用带宽可以写成：

$$
algbw=\frac{B}{T}
$$

但 AllReduce 为产生一个 $B$ 的结果，每 rank 实际需要发送约 $2(p-1)B/p$。因此基准工具常把算法带宽换算为某种 bus bandwidth，以便估计互联利用率。不同 collective 的换算因子不同。

比较结果前必须确认：

- $B$ 是每-rank 输入、输出，还是全局逻辑 tensor；
- 报告的是单向、双向还是 aggregate bandwidth；
- 是否包含 warmup、同步与校验；
- 是否跨节点以及用了多少 NIC；
- bus bandwidth 的定义是否与工具版本一致。

不能拿 AllReduce 的 `algbw` 直接和 NIC line rate 比，再据此判断链路只利用了一半。

## 性能诊断第三步：看 Timeline 中的等待而不只是 Kernel 时长

某个 NCCL kernel 持续 5 ms，不代表网络独自用了 5 ms。它可能包含：

- 等待最慢 rank 到达；
- 等待上游 compute stream 事件；
- 等待同一 NIC 上的另一 communicator；
- 实际传输与 reduction；
- 下游因错误同步而暴露出来的空洞。

可把 collective 的可见代价拆成：

$$
T_{visible}
=T_{queue}+T_{straggler}+T_{transfer}-T_{overlap}
$$

其中不是严格相加的独立量，但有助于形成诊断顺序。若所有 ranks 的通信 kernel 都晚启动，先查 producer/bucket readiness；若早到 ranks 长时间等待一个 rank，查 compute、dataloader、expert imbalance 或 GPU throttle；若大家同时开始且都慢，再查 topology、contention 与 protocol。

## 性能诊断第四步：做分层对照实验

一个可复现的验证矩阵可以逐层扩大范围：

1. 单 GPU 基线，确认计算与 tensor shape；
2. 单机两 GPU，选择物理近邻与远邻各测一次；
3. 单机全部 GPU，观察 PCIe/NVLink/NVSwitch 路径；
4. 两节点、单 rail；
5. 两节点、多 rail；
6. 完整节点数与生产 rank mapping；
7. 单一 communicator microbenchmark；
8. 与 TP/DP/EP 并发的真实 workload。

每层只增加一个变量，才能判断拐点来自 group size、跨节点、NIC affinity 还是 communicator 并发。不要用一个无计算、单 communicator 的峰值 microbenchmark 宣称真实训练必然达到相同带宽。

## Ragged All-to-All 的专用诊断指标

All-to-All 平均带宽正常时，EP 仍可能被热点 expert 拖慢。建议至少记录：

$$
imbalance
=\frac{\max_j\sum_i c_{ij}}
{\frac{1}{p}\sum_j\sum_i c_{ij}}
$$

以及：

- send/recv count matrix 的最大行、最大列和零元素比例；
- Top-$k$ 后 assignments 数与原 token 数；
- padding ratio 或 dropped-token count；
- pack/permute、count exchange、payload exchange、unpack 的分段耗时；
- dispatch 与 combine 是否对称；
- 最热 expert 的 GEMM batch size 和执行时间。

如果网络阶段很快而 grouped GEMM 在单 rank 堵塞，继续调 NCCL 没有意义；如果 pack kernel 占主导，增加链路带宽也不会解决问题。

## 一份面向代码审查的 Collective Checklist

看到一次 collective，可以按下面顺序审查：

### 语义

- 输入是 replicated、sharded、partial 还是 routed？
- reduction operator 和 normalization 是否正确？
- 输出应该被谁持有，沿哪个维度分片？
- rank order 是否等于 checkpoint/layout 期望的顺序？

### 协议

- 所有 group members 是否一定进入同一个 op？
- op、count、dtype、root、split sizes 是否匹配？
- 条件分支、异常路径和 early return 是否也保持顺序？
- overlapping communicators 是否存在循环等待？

### 生命周期

- producer 在哪个 stream，如何通知 comm stream？
- consumer 如何等待 collective 完成？
- buffer 何时可以释放或被 allocator 重用？
- in-place offset 和 storage overlap 是否满足 API 契约？

### 性能

- group size 与 message size 分布是什么？
- 理想每-rank bytes 和关键路径轮数是多少？
- topology/rank mapping 是否经过慢链路？
- bucket、chunk、channel 与 compute overlap 是否合理？
- 慢的是等待、pack、传输、reduction 还是 unpack？

### 故障

- 是否有有限 timeout 与异步错误轮询？
- 一个 rank 失败后，其他 ranks 如何得知并 abort？
- communicator 能否安全重建或 shrink？
- 从哪个一致 checkpoint/step epoch 恢复？

## 从 Collective 回看五类并行

现在可以把常见并行策略压缩成 ownership transition：

| 并行策略 | 主要输入状态 | 典型通信 | 主要输出状态 |
| --- | --- | --- | --- |
| DP/DDP | gradient partial | AllReduce | gradient replicated |
| FSDP/ZeRO-3 | parameter sharded / gradient partial | AllGather / ReduceScatter | temporary parameter replicated / gradient sharded |
| TP | feature-sharded 或 partial activation | AllGather、ReduceScatter、AllReduce | layer 所需布局 |
| CP | sequence-sharded K/V/activation | P2P Ring、AllGather 或 All-to-All | 保持或变换 token/head layout |
| EP | source-token routed assignments | All-to-All(v) 两次 | expert-owner 后再回 source-token |

这个表比“某并行用某 collective”的口诀更可靠，因为实际框架可以用不同算法完成同一 placement transition。只要前后 ownership 和数学操作一致，底层有优化空间；若前后 ownership 已经描述错了，调再快也只是更快地产生错误结果。

## 后续文章应该继续回答什么

Collective 的抽象到这里已经足够支撑后续三层内容：

1. **GPU 互联拓扑**：PCIe、NVLink、NVSwitch、InfiniBand 与 RoCE 分别提供什么路径，NUMA 和 NIC affinity 怎样改变瓶颈；
2. **GPUDirect RDMA**：远端 GPU buffer 的注册、DMA、NIC 与 CPU bypass 是怎样工作的；
3. **NCCL 内部机制**：如何发现 topology，如何选择 Ring/Tree/PAT、Simple/LL/LL128、channel 与 CTA，怎样做 buffer registration 和故障处理。

这三层不应提前压成一句“用了 NCCL 就会自动最优”。Collective 语义告诉系统必须完成什么；互联告诉它可以走哪些路；NCCL 才负责在特定版本、硬件和运行参数下选择并执行方案。

## 总结

理解 Collective Communication 的核心不是记住函数签名，而是追踪 tensor ownership：

- Broadcast 把 root-owned 变成 replicated；
- Reduce 把多份 partial 汇到 root；
- AllReduce 把 partial 变成 replicated；
- AllGather 把 sharded 变成 replicated；
- ReduceScatter 把 partial 变成 reduced-sharded；
- All-to-All 把按 source 排列的数据重排为按 destination/owner 排列。

在此之上，Ring、Tree、chunk、channel、topology 与 stream ordering 决定实现性能；一致的 collective 顺序、split sizes、buffer 生命周期和故障协调决定系统能否正确结束。DP、FSDP、TP、CP 和 EP 看似是五套并行技术，底层其实都在反复进行这些 ownership transition。

当一段分布式代码变慢时，先问“前后 placement 是什么、每 rank 真正移动多少 bytes、最慢 rank 在等什么”，通常比先搜索一个 NCCL 环境变量更接近根因。

## 参考资料

- [NVIDIA NCCL User Guide: Collective Operations](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)
- [NVIDIA NCCL User Guide: Creating and Managing Communicators](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html)
- [NVIDIA NCCL User Guide: Group Calls](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/groups.html)
- [NVIDIA NCCL User Guide: In-place Operations](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/inplace.html)
- [NVIDIA Technical Blog: Understanding NCCL Tuning to Accelerate GPU-to-GPU Communication](https://developer.nvidia.com/blog/understanding-nccl-tuning-to-accelerate-gpu-to-gpu-communication/)
- [MPI Forum: MPI 4.1 Collective Communication](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node114.htm)
- [MPI Forum: Collective Communication Correctness](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node172.htm)
- [Thakur, Rabenseifner, Gropp: Optimization of Collective Communication Operations in MPICH](https://doi.org/10.1177/1094342005051521)
- [PyTorch: Distributed Communication Package](https://docs.pytorch.org/docs/stable/distributed.html)
- [PyTorch: DistributedDataParallel](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
- [PyTorch: FullyShardedDataParallel](https://docs.pytorch.org/docs/stable/fsdp.html)
- [Megatron Core: Parallelism Strategies Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)
- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)
- [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198)
- [GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://arxiv.org/abs/2006.16668)
