---
layout: post
title: "从 Data Parallel 到 ZeRO/FSDP：训练显存到底怎样被切开"
subtitle: "逐项计算参数、梯度、Optimizer State、Activation 与 Collective 的生命周期"
date: 2026-08-06 09:00:00 +0800
last_modified_at: 2026-09-03
author: iStar
catalog: true
series: distributed-training
series_order: 70
technology_year: 2019
mathjax: true
tags: [分布式训练]
---

把训练从一张 GPU 扩到八张，最直接的方法是让每张卡保存完整模型、读取不同数据，再同步梯度。这种 Data Parallel 提高了吞吐，却没有让单卡少放任何模型状态：parameters、gradients 和 optimizer states 仍被复制八份。模型越大，GPU 数量增加得越多，这种冗余越显眼。

ZeRO（Zero Redundancy Optimizer）提出了一个朴素但影响深远的问题：既然 Data Parallel ranks 最终要得到相同模型，为什么训练过程中的每个瞬间都必须各自持有一整份状态？

答案是把不同状态沿 data-parallel group 分片，只在某段计算真正需要完整 tensor 时临时聚合：

```text
DDP:    parameters replicated + gradients replicated + optimizer replicated
ZeRO-1: parameters replicated + gradients replicated + optimizer sharded
ZeRO-2: parameters replicated + gradients sharded    + optimizer sharded
ZeRO-3: parameters sharded    + gradients sharded    + optimizer sharded
```

PyTorch FSDP 的 `FULL_SHARD` 与 ZeRO Stage 3 共享同一核心思路：模块计算前 all-gather 参数，计算后释放完整参数；backward 结束后用 reduce-scatter 得到本 rank 应保留的梯度 shard，再只更新本地 optimizer shard。

要判断这种方法能否解决 OOM、会增加多少通信、checkpoint 应怎样保存，首先要把训练显存逐项算清楚。

## Data Parallel 保证的数学语义

设 data-parallel world size 为 $N$。第 $r$ 个 rank 在自己的 micro-batch $\mathcal{B}_r$ 上计算 loss 和梯度：

$$
g_r=\nabla_\theta
\frac{1}{|\mathcal{B}_r|}
\sum_{x\in\mathcal{B}_r}\ell(x;\theta)
$$

所有 ranks 通过 AllReduce 得到平均梯度：

$$
g=\frac{1}{N}\sum_{r=1}^{N}g_r
$$

每个 rank 从相同的参数 $\theta$ 和 optimizer state 出发，用相同 $g$ 更新，就会得到相同的新参数。于是下一步仍可独立读取不同数据。

有效 global batch size 通常为：

$$
B_{global}
=
B_{micro}
\times N
\times G
$$

其中 $G$ 是 gradient accumulation steps。若最后一个 batch 大小不同、某些 ranks 跳过数据、loss normalization 或梯度缩放不一致，这条等价关系就会破坏。分布式训练正确性首先是数据与归一化语义，再是 collective 是否成功返回。

## 一次 Mixed-precision Adam 为什么接近 16 Bytes/Parameter

设模型有 $P$ 个可训练参数。经典 FP16/BF16 mixed-precision Adam 的模型状态常近似拆成：

| 状态 | 常见 dtype | Bytes/parameter |
| --- | --- | ---: |
| 计算参数 | FP16/BF16 | 2 |
| 梯度 | FP16/BF16 | 2 |
| FP32 master parameter | FP32 | 4 |
| Adam first moment $m$ | FP32 | 4 |
| Adam second moment $v$ | FP32 | 4 |
| 合计 |  | 16 |

所以模型状态主体约为：

$$
M_{states}\approx16P\ \text{bytes}
$$

7B 参数的理想主体就是约 112 GB（十进制），远超“BF16 模型文件约 14 GB”给人的直觉。

这个 16 bytes 只是便于推导的经典模型，不是所有训练栈的固定常数：

- BF16 optimizer 可能不保存独立 FP32 master copy；
- gradient accumulation dtype 可能是 FP32；
- 8-bit optimizer 会改变 moment 大小并增加 scale metadata；
- frozen parameters 不需要 gradient/optimizer state；
- parameter padding、flattening 与 alignment 会增加实际 bytes；
- weight tying、views 与共享参数会影响去重方式。

因此先从 optimizer 和 runtime contract 读取实际 dtype，再套公式。

## 16P 之外还有哪些显存

训练峰值显存可以写成：

$$
M_{peak}
=
M_{model\ states}
+M_{activations}
+M_{temporary}
+M_{communication}
+M_{allocator}
$$

其中：

- **Activations**：forward 为 backward 保存的中间 tensor，随 micro-batch、sequence length、hidden size 和层数增长；
- **Temporary buffers**：GEMM workspace、attention scratch、fused optimizer 临时区；
- **Communication buffers**：gradient buckets、parameter all-gather windows、reduce-scatter outputs；
- **Allocator overhead**：reserved-but-unused blocks、碎片与 graph pools。

ZeRO 的三阶段主要处理 model states，不会自动把 activation 除以 $N$。一个作业从 parameter OOM 变成 activation OOM，并不表示 ZeRO 失效，而是主要项已经转移。

## DDP 为什么单卡仍是 16P

标准 DistributedDataParallel 在每个 rank 上持有完整：

```text
parameters:       2P
gradients:        2P
master + moments: 12P
----------------------
model states:    16P bytes/rank
```

Backward 时 gradient hooks 按 bucket 触发 AllReduce。一个 ring AllReduce 在概念上可看作：

```text
ReduceScatter: aggregate chunks and leave one reduced chunk per rank
AllGather:      distribute all reduced chunks back to every rank
```

最终每张卡仍得到完整 gradient，因为每张卡都要用完整 optimizer state 更新完整参数。

DDP 的优点是计算路径简单：forward/backward 使用本地完整参数，通信集中在梯度同步，并可与 backward 重叠。只要模型状态本身能放进单卡，它通常是很强的吞吐基线。

## ZeRO Stage 1：只切 Optimizer State

Stage 1 把 FP32 master parameters 与 Adam moments 按 data-parallel ranks 分片。每个 rank 只负责更新约 $1/N$ 的参数区间：

$$
M_{Z1}
\approx
4P+\frac{12P}{N}
$$

这里 replicated 的 $4P$ 是低精度 parameters 与 gradients，sharded 的 $12P$ 是经典 Adam optimizer states。

一次 step 可以理解为：

```text
all ranks hold full low-precision parameters
  → each rank computes local full gradients
  → reduce/scatter or equivalent gradient synchronization
  → owner rank updates its optimizer/parameter shard
  → all-gather updated parameter shards
  → all ranks again hold identical full parameters
```

Stage 1 省掉最大的一项 optimizer redundancy，同时保留完整参数参与 forward/backward。对参数能放下、optimizer state 放不下的场景，它往往用较小执行改动获得明显收益。

## ZeRO Stage 2：Gradient 也不再复制

Stage 2 在 Stage 1 基础上分片 gradients：

$$
M_{Z2}
\approx
2P+\frac{14P}{N}
$$

低精度 parameters 的 $2P$ 仍 replicated；gradients 的 $2P$ 与 optimizer states 的 $12P$ 按 rank 分片。

Backward 不必先生成一份完整 reduced gradient 再丢弃大部分，而可以按 bucket ReduceScatter：

```text
local gradient contributions
  → ReduceScatter
  → rank r retains reduced gradient shard r
  → update matching optimizer shard
  → AllGather updated parameter shards
```

与 DDP 的 `ReduceScatter + AllGather` 分解相比，Stage 2 并不必然增加同量级的总通信 volume；它改变的是结果的驻留位置和 optimizer ownership。性能差异更多来自 bucket、overlap、实现细节和 parameter synchronization 时机。

## ZeRO Stage 3：Parameter 只在计算窗口完整存在

Stage 3 再把低精度 parameters 分片。理想稳态模型状态变为：

$$
M_{Z3}
\approx
\frac{16P}{N}
$$

但 layer 计算不能用缺失的参数。以某个 wrapped module 为单位，runtime 要执行：

```text
local parameter shard
  → pre-forward AllGather full module parameters
  → forward compute
  → optionally reshard/free full parameters
  → pre-backward AllGather full module parameters
  → backward compute
  → ReduceScatter gradients
  → keep local gradient shard
  → local optimizer update
```

所以“parameters sharded”并不表示某张卡在整个 iteration 中从未看到完整 layer parameters。它表示完整参数只在有限窗口内 materialize，并在使用后尽快释放。

峰值显存至少还包含一个或多个 all-gathered modules：

$$
M_{peak,Z3}
\not=\frac{16P}{N}
$$

Wrap granularity、prefetch depth、通信 bucket 和 allocator reuse 决定额外窗口有多大。

## 用 7B、8 GPUs 做一次理想账单

令 $P=7$B、$N=8$，沿用 16 bytes/parameter 的经典 mixed-precision Adam 模型：

| 策略 | 理想 model-state bytes/rank | 近似大小 |
| --- | ---: | ---: |
| DDP | $16P$ | 112 GB |
| ZeRO-1 | $4P+12P/8$ | 38.5 GB |
| ZeRO-2 | $2P+14P/8$ | 26.25 GB |
| ZeRO-3 | $16P/8$ | 14 GB |

这些数字不包含 activations、all-gather windows、communication buckets、CUDA context 与碎片。它们适合判断量级，不能直接拿来把 7B ZeRO-3 塞进 16 GB GPU。

同样，增加 GPU 数只会让 sharded 项继续下降；replicated activation、临时 full module 和每 rank micro-batch 不会按相同公式缩小。

## ZeRO 与 FSDP 的关系

ZeRO 是一套消除 data-parallel state redundancy 的分阶段思想；DeepSpeed ZeRO 是其系统实现。PyTorch Fully Sharded Data Parallel 将类似的全分片语义集成到 PyTorch distributed/autograd 体系。

在 FSDP1 的常用 `ShardingStrategy` 中：

| FSDP strategy | 近似对应 | 行为摘要 |
| --- | --- | --- |
| `NO_SHARD` | DDP | 参数、梯度、optimizer states replicated；梯度 AllReduce |
| `SHARD_GRAD_OP` | ZeRO-2 类 | gradients/optimizer sharded；参数在计算窗口外可分片 |
| `FULL_SHARD` | ZeRO-3 类 | parameters/gradients/optimizer states 全分片 |
| `HYBRID_SHARD` | node 内 ZeRO-3 + node 间复制 | 把频繁 shard collectives 限制在较快域内 |

这只是概念映射，不代表 DeepSpeed 与 FSDP 的配置、hooks、checkpoint、offload 和 bucket 行为完全相同。

FSDP2 的 `fully_shard` 使用 per-parameter DTensor 表示 sharded state，并通过 hooks 在 forward/backward 前 unshard，之后 reshard。它改善了一些 composability 与参数可见性，但部署仍必须按所用版本的官方 contract 编写。

## Wrap Granularity 为什么决定峰值与吞吐

若把整个模型作为一个 FSDP unit：

```text
AllGather entire model
→ compute all layers
→ reshard entire model
```

参数峰值可能接近完整模型，失去 Stage 3 的主要意义。若每个极小子算子都单独 wrap，则 all-gather 次数过多、消息太小、latency 和 hook overhead 上升。

Transformer 常以 block 为基本单位：

```text
block 0 unshard → compute → reshard
block 1 unshard → compute → reshard
...
```

但一个 block 的大小、embedding/LM head 的 weight tying、MoE experts、pipeline stage 和 activation checkpoint boundary 都会影响最佳切法。

Wrap policy 实际决定了三件事：

1. 单次 all-gather 多大；
2. 同时驻留多少 full parameters；
3. 通信能否被相邻 module 的计算隐藏。

它不是纯粹的代码组织选项，而是内存/网络/调度的共同参数。

## Reshard-after-forward 是空间换通信

Stage 3 forward 结束后有两种基本选择。

### 立即 Reshard

释放 full parameters，只保留 shard；backward 到该模块前再 all-gather 一次。

- 峰值更低；
- forward 与 backward 各需一次参数 all-gather；
- 适合参数容量压力大或深层模型。

### 保留到 Backward

forward 后暂时保留 full parameters，backward 直接使用，之后再 reshard。

- 省掉 pre-backward all-gather；
- 参数跨更长生命周期驻留，显存增加；
- 更接近 ZeRO-2/`SHARD_GRAD_OP` 的部分行为。

选择应由模型参数/activation 比例和网络速度决定。GPU 显存宽裕而跨节点网络慢时，保留参数可能更快；显存紧张时则必须接受额外通信。

## AllGather Prefetch 怎样与计算重叠

如果严格按：

```text
wait module i parameters
→ compute module i
→ request module i+1 parameters
```

GPU 会在层间等待通信。FSDP/ZeRO runtime 通常在计算 module $i$ 时预取 $i+1$：

```text
communication stream:    all-gather(i+1) ───── all-gather(i+2)
compute stream:       compute(i) ───────── compute(i+1)
```

但无限 prefetch 会让多个 full parameter buffers 同时驻留，引发 OOM。PyTorch FSDP 的 `limit_all_gathers` 一类 rate limiting 正是为了限制在途 all-gathers；profiling 中出现 CPU 发射间隙未必表示 GPU 真正空闲。

Forward prefetch、backward prefetch 和执行顺序还受 dynamic graph、conditional modules、activation checkpoint recompute 影响。只有已知下一模块且 stream dependencies 正确，重叠才安全。

## ReduceScatter 必须尽早释放 Gradient

Backward 按反向层序产生 gradients。理想 Stage 2/3 实现会在某个 bucket 就绪后立即 ReduceScatter，而不是等待所有 layers 完成：

```text
backward layer L
  → bucket ready
  → ReduceScatter on communication stream
  → free full local gradients

backward layer L-1 continues on compute stream
```

Bucket 太大，通信启动晚且 full gradients 驻留久；bucket 太小，collective latency 和 launch overhead 增加。参数注册顺序与 backward ready 顺序不匹配时，也可能让 bucket 等待一个很晚才产生的 gradient。

需要在 profiler 中检查 collective 是否真正与 backward GEMM 重叠，而不是只确认配置打开了 overlap。

## Gradient Accumulation 会改变生命周期

当 $G>1$ 时，一个 optimizer step 包含多次 forward/backward。DDP 常用 `no_sync()` 跳过前 $G-1$ 次 AllReduce，在本地累积 full gradients，最后一次再同步。

ZeRO-2/3 的 gradients 本来按 collective 分片，简单跳过通信可能迫使每个 rank 暂存 full gradients，破坏内存模型或实现不支持。不同 runtime 对 `no_sync`、coalesced reduction 和 low-precision accumulation 有不同 contract。

因此 gradient accumulation 配置要同时核对：

- 每个 micro-step 是否 ReduceScatter；
- shard 上累积还是 full tensor 上累积；
- loss 是否除以 accumulation steps；
- gradient clipping 在 global norm 还是 local shard 上计算；
- overflow/GradScaler 是否由所有 ranks 一致决定；
- optimizer step 与 scheduler step 的频率。

只看最终 global batch size，无法判断通信与峰值显存。

## Activation Checkpointing 解决的是另一项

Activation checkpointing 不保存所有 forward intermediates，而在 backward 前重算一段 forward：

```text
without checkpoint:
  save activations → backward uses them

with checkpoint:
  save boundary inputs → recompute forward segment → backward
```

它用额外 FLOPs 换 activation memory，与 ZeRO 的 model-state sharding 正交：

| 技术 | 主要减少 | 主要代价 |
| --- | --- | --- |
| ZeRO/FSDP | 参数、梯度、optimizer redundancy | collectives、临时 unshard buffer |
| Activation checkpointing | forward activations | backward recompute |
| Tensor/Sequence Parallel | 单 rank 计算与部分 activations/parameters | 细粒度 collectives |
| CPU/NVMe offload | GPU resident states | PCIe/NVLink-C2C/storage transfer |

若显存 profile 显示 activation 占主导，从 ZeRO-2 升 ZeRO-3 可能收益有限；应调整 micro-batch、sequence parallel 或 checkpoint granularity。

## CPU/NVMe Offload 不是免费显存

ZeRO-Offload/ZeRO-Infinity 可把 optimizer states、parameters 或计算移动到 CPU/NVMe。它扩大容量，却把问题转为分层存储调度：

```text
NVMe ↔ CPU DRAM ↔ pinned buffers ↔ PCIe/NVLink-C2C ↔ GPU HBM
```

每步所需 bytes 如果超过链路可隐藏的带宽，GPU 会等待数据。需要同时考虑：

- PCIe 实际双向带宽与 NUMA locality；
- CPU optimizer 吞吐和 memory bandwidth；
- pinned memory 总量；
- prefetch depth 与 eviction；
- NVMe queue depth、写放大与设备寿命；
- checkpoint I/O 与训练 offload 的竞争。

Offload 适合“否则根本放不下”或 GPU compute 很长、传输可充分重叠的场景。它不应只按 OOM 是否消失评估，还要看 step time、GPU idle 和总成本。

## Hybrid Sharding 为什么匹配物理拓扑

全局 ZeRO-3 会在每个 wrapped module 上跨整个 data-parallel group all-gather/reduce-scatter。若 group 横跨节点，频繁参数通信会经过较慢或拥塞的网络。

Hybrid sharding 可以：

```text
within node / fast NVLink domain:
  shard parameters, gradients, optimizer states

across nodes:
  replicate each shard group and synchronize replicas
```

这样用更多参数副本换取跨节点通信减少。它的最佳 group size 通常对应 NVLink/NVSwitch island、NIC rail 与 NUMA 边界，而不是任意 world-size 因数。

同一作业还可能叠加 Tensor Parallel、Pipeline Parallel 或 Expert Parallel。此时每个 process group 必须明确：

```text
DP group: data replicas / ZeRO-FSDP sharding
TP group: one layer's tensor shards
PP group: pipeline stages
EP group: expert ownership and token dispatch
```

把 FSDP group 错跨到 TP 维度，会同时破坏参数 ownership 与 collective 语义。

## Shared Parameters 与 Tied Weights 为什么麻烦

Embedding 与 LM head 可能共享同一个 Parameter；某些模块会保存 parameter view 或在 forward 外引用权重。FSDP 在 unshard/reshard 期间可能替换 parameter view，旧引用未必仍指向当前 full tensor。

如果共享参数被两个不同 FSDP units 独立管理，可能出现：

- 同一逻辑权重被重复 flatten/shard；
- forward 两处看到不同 storage；
- gradient 被重复或遗漏同步；
- checkpoint 中产生两个不一致条目。

Wrap plan 必须识别共享参数，让 ownership 唯一，并遵守 runtime 对 original parameters、views 和 ignored modules 的限制。不能等到 loss 发散后才从通信日志猜测。

## Mixed Precision 至少有四个 Dtype

“BF16 训练”可能同时包含：

```text
parameter compute dtype
buffer dtype
gradient reduction dtype
optimizer/master dtype
```

还可能有 attention/GEMM accumulator、loss scalar 和 gradient norm dtype。FSDP mixed-precision policy 会决定参数在 all-gather 后转换成什么 dtype、gradients 用什么 dtype ReduceScatter、optimizer 是否保留低精度 grads。

通信 dtype 更低可以减少 bytes，却改变 rounding 与 overflow；参数 shard 若以 FP32 保存、计算前再转 BF16，稳态内存也不同。Manifest 与实验报告应逐项记录，而不是只写 `bf16: true`。

## Checkpoint 不能假设每个 Rank 都有完整模型

DDP 可以让 rank 0 直接保存完整 `state_dict`，因为它本来就持有全部参数。ZeRO-3/FSDP 稳态只持有 shards，保存策略至少有两类。

### Full State Dict

聚合完整参数到一个或少数 ranks，再写传统 checkpoint。

- 兼容性高；
- 聚合时可能 OOM；
- rank 0 内存与 I/O 成为瓶颈；
- optimizer full state 更大。

### Sharded State Dict

每个 rank 写本地 shards 和 metadata，再由 distributed checkpoint 描述全局 tensor。

- 避免单 rank 聚合；
- 可以并行 I/O；
- 恢复到不同 world size 需要 reshard；
- 依赖稳定的 tensor names、placement 和 checkpoint schema。

一致 checkpoint 还必须包含：

- model parameters；
- optimizer moments/master weights；
- LR scheduler 和 global step；
- GradScaler/overflow state；
- RNG states；
- data sampler/dataloader progress；
- parallel mesh 与 sharding metadata；
- tokenizer/model config 与代码 revision。

若只保存模型而丢失 optimizer，作业可以继续 fine-tune，但不是从同一训练轨迹恢复。

## 初始化阶段也可能先 OOM

一个常见失败是：计划用 ZeRO-3 训练 70B，却先在每个 rank 上构造完整 FP32 模型，再进行 sharding。分片还没开始，CPU 或 GPU 已经 OOM。

大模型初始化需要采用 meta device、deferred initialization 或 sharded checkpoint load：

```text
create parameter metadata without full storage
  → assign ownership from device mesh
  → materialize only local shards
  → load matching checkpoint shards
```

随机初始化还要保证不同 ranks 拼接后的全局 tensor 与目标 seed 语义一致。若每个 rank 用相同 seed 独立生成 local shape，结果不一定等于单进程初始化后再切分。

初始化、训练、checkpoint 和恢复必须共享同一套 parameter identity 与 sharding plan。

## 训练 Hang 应按 Collective 顺序排查

ZeRO/FSDP 通过 hooks 动态插入 collectives。只要不同 ranks 进入模块的顺序或次数不同，就可能一部分 rank 在等待 parameter AllGather，另一部分已经进入 gradient ReduceScatter。

常见原因包括：

- data-dependent control flow 在 ranks 间不同；
- 某个 rank 提前遇到空 batch 或异常；
- gradient checkpoint 重算路径不一致；
- unused parameters 在不同 ranks 上不同；
- shared module 被调用次数不同；
- collective group 或 device mapping 配错；
- OOM 后只有一个 rank 退出，其余仍等待 NCCL。

诊断时给每个 collective 记录：sequence number、process group、tensor numel、dtype、module id 和调用阶段。只看到“卡在 NCCL”还不足以定位第一个语义分叉点。

## 性能分析不要只看 GPU Utilization

一次可信的训练 profile 至少分解：

```text
data loading
forward compute
parameter all-gather wait
activation recompute
backward compute
gradient reduce-scatter
optimizer step
parameter synchronization
checkpoint I/O
```

同时记录：

- samples/s 与 tokens/s；
- model FLOPs utilization；
- step time P50/P99；
- peak allocated/reserved memory；
- collective bytes、duration 与 overlap ratio；
- all-gather window 数量和最大并发；
- straggler rank；
- host/NIC/GPU topology；
- loss 与 gradient norm 正确性。

高 GPU utilization 可能包含 recompute 或等待前后的碎片化 kernel，不等于有效训练吞吐。低峰值显存也可能以大量跨节点 all-gather 为代价。最终应比较在相同模型、global batch、sequence length 和数值配置下完成一个有效 token 的时间与成本。

## 选择 DDP、ZeRO-2 还是 ZeRO-3

可以从显存主导项和网络出发：

| 场景 | 优先验证 | 原因 |
| --- | --- | --- |
| 完整 model states 可放单卡 | DDP | 路径简单，通信集中在 gradients |
| Optimizer/gradient 造成 OOM，parameters 能放 | ZeRO-1/2 | 避免频繁 parameter all-gather |
| Parameters 本身无法单卡放下 | ZeRO-3/FSDP full shard | 只有计算窗口需要 full module |
| 节点内快、节点间慢 | Hybrid shard | 限制频繁 shard collectives 的拓扑范围 |
| Activation 占主导 | Activation checkpoint / sequence parallel | 单纯升级 ZeRO stage 改善有限 |
| GPU 容量仍不足但 CPU/NVMe 充足 | Offload | 用传输与主机计算换容量 |

Stage 数字更大不代表训练更先进。能用 ZeRO-2 稳定跑满 GPU 时，切到 ZeRO-3 可能只是增加 parameter communication；能用 DDP 的小模型也未必需要 full shard 的复杂性。

## 一条可执行的落地路径

1. **建立单卡内存分解**：分别测 model states、activations、temporary 和 allocator；
2. **固定数学基线**：保存 loss、gradient norm、global batch、seed 与短程收敛曲线；
3. **先跑 DDP 吞吐基线**：确认数据、AllReduce 与 scaling 正确；
4. **按 OOM 项选择 stage**：不要默认从 ZeRO-3 开始；
5. **设计 wrap/shard plan**：对齐 Transformer blocks、shared weights 与 physical topology；
6. **验证 mixed-precision contract**：parameters、reduce、grads、optimizer 分别记录 dtype；
7. **调 bucket 与 prefetch**：同时观察 overlap 和 peak windows；
8. **加入 activation checkpointing**：单独测 recompute 增量；
9. **验证 accumulation/clipping**：比较单步 gradients 与 optimizer updates；
10. **设计 sharded checkpoint**：完成保存、故障注入、同 world-size 和 resize 恢复；
11. **做多节点 profile**：定位 topology、straggler 和 collective sequence；
12. **用有效 tokens/cost 选配置**：显存利用率只是约束，不是最终目标。

## 常见误区

### “8 张 GPU 做 Data Parallel，单卡模型显存会除以 8”

DDP 复制完整 model states；除以 8 的是每 rank 处理的数据，不是模型状态。

### “ZeRO-3 的峰值显存就是 16P/N”

那只是理想 sharded state 主体。模块计算还需要临时 full parameters、collective buckets、activations 和 workspace。

### “ZeRO 会自动减少 Activation”

ZeRO 分片 model states。Activation 要靠 micro-batch、checkpointing、sequence/context parallel 等手段处理。

### “Stage 越高一定越快”

更高 stage 省更多显存，也把更多参数通信放进 forward/backward。网络或计算粒度不合适时吞吐会下降。

### “FSDP 就是 DeepSpeed ZeRO-3 的另一个名字”

核心全分片思想相近，但 API、parameter representation、hooks、offload、checkpoint 和版本行为不同。

### “能从 checkpoint 读回 loss 就算恢复成功”

还要恢复 optimizer、scheduler、RNG、sampler 与 sharding metadata，并验证下一步 update 与连续运行一致。

## 小结

ZeRO/FSDP 并没有让训练状态消失，而是缩短完整状态的驻留时间，并把 ownership 分散到 data-parallel ranks。

可以抓住十点：

1. DDP 每 rank 复制完整 parameters、gradients 和 optimizer states；
2. 经典 mixed-precision Adam 可用 16 bytes/parameter 做量级估算；
3. ZeRO-1 切 optimizer，ZeRO-2 再切 gradients，ZeRO-3 再切 parameters；
4. ReduceScatter 让每个 rank 只保留自己的 gradient shard；
5. Stage 3 在模块计算前 AllGather，之后按策略 Reshard；
6. Wrap granularity 与 prefetch 同时决定通信效率和峰值 full-parameter windows；
7. Activation checkpointing、parallelism 与 offload 处理的是不同显存项；
8. FSDP/ZeRO group 必须匹配 TP/PP/EP 维度和物理拓扑；
9. Sharded checkpoint 必须保存全局 tensor 的 identity、placement 与训练状态；
10. 最终以正确性、tokens/s、峰值显存、通信 overlap 与恢复能力共同验收。

这篇先在 Data Parallel 维度解决状态复制。下一篇会进入 Tensor Parallel：当单个 layer 的参数或矩阵乘本身也放不进一张卡时，Megatron-LM 怎样用 column/row parallel 切开 MLP 和 Attention，并把 collective 放在数学上恰好可以合并的位置。

## 参考资料

- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
- [DeepSpeed: ZeRO Documentation](https://deepspeed.readthedocs.io/en/stable/zero3.html)
- [PyTorch: FullyShardedDataParallel](https://docs.pytorch.org/docs/stable/fsdp.html)
- [PyTorch: FSDP2 fully_shard](https://docs.pytorch.org/docs/main/distributed.fsdp.fully_shard.html)
- [PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel](https://arxiv.org/abs/2304.11277)
- [PyTorch Distributed Checkpoint](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html)
