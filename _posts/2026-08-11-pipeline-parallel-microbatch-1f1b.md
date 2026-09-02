---
layout: post
title: "Pipeline Parallel：Micro-batch 怎样流过 Transformer Stages"
subtitle: "从 GPipe、1F1B 到 Interleaving，理解流水气泡、Activation 生命周期与并行组合"
date: 2026-08-11 09:00:00 +0800
last_modified_at: 2026-09-03
author: iStar
catalog: true
series: distributed-training
series_order: 50
technology_year: 2018
mathjax: true
tags: [分布式训练, GPU优化]
---

Tensor Parallel 沿 Transformer layer 的宽度切矩阵，一个 layer 由多张 GPU 共同完成；Pipeline Parallel（PP）沿模型深度切层，让不同 GPU 分别持有连续的 layer stages。前者每层都要进行 collective，后者主要在相邻 stages 之间传递 activation 与 activation gradient。

如果只是把前 20 层放在 GPU 0、后 20 层放在 GPU 1，然后按顺序执行，GPU 0 做完 forward 后就会等待 GPU 1，GPU 1 做 backward 时 GPU 0 也在等待。模型虽然放下了，设备利用率却很低。Pipeline Parallel 真正解决的问题不是“怎样把层分开”，而是怎样用多个 micro-batches 填满这些 stages，同时保持同步训练的梯度语义。

一条最简数据流是：

```text
tokens
  → stage 0: embedding + layers 0..N0
  → send activation
  → stage 1: layers N0+1..N1
  → send activation
  → ...
  → last stage: final norm + LM head + loss

backward 沿相反方向传回 activation gradients
```

本文从时间线开始，依次回答四个问题：micro-batch 为什么能形成流水、bubble 怎样估算、1F1B 为什么比全 Forward/全 Backward 更省显存，以及 PP 怎样与 TP、DP、Sequence/Context Parallel 组合。

## 先区分 Mini-batch 与 Micro-batch

假设一次 optimizer step 的 global batch 被 Data Parallel replicas 和 Pipeline micro-batches 共同拆分。忽略额外的 gradient accumulation 层级时：

$$
B_{global}
=
B_{micro}
\times m
\times p_{DP}
$$

其中：

- $B_{micro}$ 是一次进入 pipeline 的样本数；
- $m$ 是一个 optimizer step 内的 micro-batch 数；
- $p_{DP}$ 是 Data Parallel degree。

在本文后续讨论的同步 flush schedules 中，同一条 pipeline 上的 $m$ 个 micro-batches 使用同一版本参数，分别完成 forward/backward，梯度累加后才执行一次 optimizer step。拆成 micro-batches 的目标不是改变数学 batch，而是让不同 stages 同时处理不同 micro-batches。

例如 4 个 stages 可以在同一时刻执行：

```text
stage 0: micro-batch 3 forward
stage 1: micro-batch 2 forward
stage 2: micro-batch 1 forward
stage 3: micro-batch 0 forward
```

每个 micro-batch 仍按 stage 0→1→2→3 的依赖顺序前进，但整个系统已经并行工作。

## 为什么一定会出现 Pipeline Bubble

设 pipeline 有 $p$ 个 stages。第一个 micro-batch 到达 stage $p-1$ 前，后面的 stages 没有输入，这是 warmup；最后一个 micro-batch 离开前面的 stages 后，前面 stages 又会提前空闲，这是 drain。两端的空闲区就是 bubble。

在各 stage Forward/Backward 耗时近似相同、忽略通信与重叠差异的简化模型下，一次 flush schedule 的 bubble fraction 可近似写为：

$$
f_{bubble}
\approx
\frac{p-1}{m+p-1}
$$

直觉是每个 stage 有 $2m$ 个有效 Forward/Backward slots，而整条流水还要支付由深度带来的填充与排空成本。增大 $m$ 能摊薄固定的 $p-1$，但会带来新的代价：

- global batch 随 $m$ 增大，可能改变优化语义；
- 若保持 global batch 不变，$B_{micro}$ 会变小，GEMM 利用率可能下降；
- 更多未完成 micro-batches 会增加 activation、RNG 与调度状态；
- step latency 变长，故障时可能损失更多未提交工作。

所以“micro-batch 越多越好”并不成立。目标是让 $m$ 足够大以接受 bubble，同时让 local GEMM shape、global batch 和 activation memory 仍合理。

## GPipe：先完成全部 Forward，再完成全部 Backward

GPipe 的经典同步 schedule 可以概括为：

```text
F0, F1, F2, ... F(m-1)
然后
B(m-1), B(m-2), ... B0
最后 optimizer.step()
```

这里 `Fi`、`Bi` 表示同一个 micro-batch 在当前 stage 上的 forward/backward。所有 micro-batches 完成 backward 前不更新参数，因此没有跨 micro-batch 的参数版本不一致。

它的优势是执行关系直观，容易验证同步 SGD 语义；主要问题是 activation 生命周期很长。Stage 0 很早完成 `F0`，却要等 `F(m-1)` 穿过整条 pipeline、last stage 开始 backward 后，才能最终收到 `B0`。如果不做 recomputation，每个 stage 可能同时保存接近 $m$ 份 forward activations：

$$
M_{activation,GPipe}
\propto
m\times M_{activation,micro}
$$

Activation Checkpointing 可以少存中间结果、在 backward 前重算，但它用额外计算换显存，并没有缩短 micro-batch 的逻辑生命周期。

## 1F1B：让 Forward 与 Backward 在稳态交替

同步 1F1B（One Forward, One Backward）通常分三段：

1. **Warmup**：先做若干 Forward，把足够多 micro-batches 推向下游；
2. **Steady state**：每完成一个新 micro-batch 的 Forward，就尽快处理一个更早 micro-batch 的 Backward；
3. **Cooldown**：不再注入新 Forward，清空剩余 Backward。

对某个中间 stage，稳态类似：

```text
... F4 → B1 → F5 → B2 → F6 → B3 ...
```

它并不意味着同一张 GPU 真正同时执行一条 Forward 与一条 Backward；通常是交替排入执行流。价值在于更早释放旧 micro-batch 的 activations。Pipeline 前端需要更长 warmup，后端更早进入 Backward，因此各 rank 的 outstanding activations 上限不同，但通常明显小于“先完成全部 Forward”的 $m$ 份。

在理想等时模型中，同步 1F1B 不会凭空消除 flush bubble；它主要改变 Forward/Backward 的排列和 activation memory。吞吐提升常来自：

- 较低峰值显存允许增大 micro-batch；
- 更及时的 buffer 复用；
- Forward/Backward 通信与不同 micro-batch 计算的重叠；
- 更细的 interleaving 进一步缩短 bubble。

## Flush 为什么关系到参数版本正确性

同步 1F1B 常被称为 1F1B-Flush：一个 optimizer step 的全部 micro-batches 都完成 backward，pipeline 排空后才更新参数。于是这些 micro-batches 的 Forward 和 Backward 使用同一参数版本。

早期 PipeDream 一类异步 pipeline 可以跨 optimizer steps 持续运行，减少 flush，但会出现某个 micro-batch 的 Forward 使用旧参数、Backward 时当前参数已经变化的问题。系统需要 weight stashing 或其他一致性机制，这已经不是普通同步训练的简单调度替换。

因此看到“无 bubble”或“不 flush”的 schedule 时，应先问：

1. 参数何时更新；
2. Forward 与 Backward 使用哪个版本；
3. 梯度应用到哪个版本；
4. 该算法是否仍与目标 synchronous SGD 等价。

不能只比较设备时间线的空白面积。

## Interleaved 1F1B：一个物理 Rank 放多个 Virtual Stages

普通 PP 把连续的一大段 layers 放在每个 rank。Interleaved Pipeline Parallel 再把每个物理 rank 的 layers 切成多个 model chunks，也称 virtual pipeline stages。

假设 4 个物理 ranks、每个 rank 2 个 chunks，逻辑顺序可以是：

```text
rank 0 chunk 0 → rank 1 chunk 0 → rank 2 chunk 0 → rank 3 chunk 0
→ rank 0 chunk 1 → rank 1 chunk 1 → rank 2 chunk 1 → rank 3 chunk 1
```

模型不是简单地在 rank 3 结束，而会回到 rank 0 的下一块。更细粒度的虚拟 stages 能降低每个逻辑 stage 的计算时间，让调度器在 bubble 区域插入更多可执行工作。Megatron-LM 的 interleaved 1F1B 正是用 model chunks 减少 pipeline bubble。

代价也很直接：

- stage 边界增多，P2P 发送次数增加；
- 调度表、buffer ownership 与 chunk id 更复杂；
- 小 chunk 可能让 kernel 太碎，降低局部效率；
- rank 间可能出现多次往返，物理拓扑更重要；
- checkpoint 要记录 layer 到 physical rank/chunk 的 logical mapping。

只有当减少的 bubble 大于额外通信与调度成本时，interleaving 才有净收益。

## Stage 不能只按 Layer 数量平均切

若 80 层 Transformer 分到 8 个 stages，直接每 stage 10 层只是起点。实际耗时还包括：

- 第一 stage 的 embedding、position/RoPE 准备和数据输入；
- 最后一 stage 的 final norm、vocab projection 与 distributed loss；
- 不同 layer 的 dense/MoE、attention 形式和 sequence shape；
- activation checkpointing 带来的重算；
- 跨 stage tensor layout conversion；
- 与 TP、CP、EP collectives 的竞争；
- pipeline send/recv 的拓扑路径。

Pipeline throughput 由最慢 stage 决定。若各 stage 单个 micro-batch 的时间为 $t_i$，稳态节拍近似受：

$$
t_{stage}=\max_i t_i
$$

限制。一个 stage 比其他 stages 慢 20%，其余 ranks 会周期性等待，即使理论 bubble 已经很小。

合理切分应先 profile 单层或 layer groups，再按执行时间、显存峰值和通信边界联合分区。Embedding 与 loss 也应计入，而不是只平分 Transformer block 数。

## Stage 边界到底传多少数据

若 stage 边界 activation 的逻辑 shape 是：

$$
[S,B_{micro},H]
$$

单次 Forward 发送量约为：

$$
V_F=S\times B_{micro}\times H\times bytes(dtype)
$$

Backward 需要反向传递同 shape 的 activation gradient，因而还有相近的 $V_B$。对**一个逻辑 stage 边界**，一次 optimizer step 的逻辑流量约为：

$$
V_{step,\mathrm{one\ boundary}}\approx m(V_F+V_B)
$$

若普通 $p$-stage pipeline 的 $p-1$ 个边界具有相同 tensor shape，则把每个边界发送的 payload 相加后，整条 pipeline 的累计逻辑流量近似为：

$$
V_{step,\mathrm{all\ boundaries}}\approx (p-1)m(V_F+V_B)
$$

这里统计的是跨各边界发送的 payload 总和，不等同于关键路径时间；virtual/interleaved stages 还要按实际跨越的逻辑边界次数计算。

实际 local bytes 会受到 TP/SP/CP layout 影响。例如 boundary tensor 已沿 sequence 或 hidden sharded 时，每个 rank 只发送自己的 shard；如果下游要求不同 placement，则还可能伴随 AllGather、ReduceScatter 或 layout transform。

PP 常被称为“通信量低于 TP”，因为它不是每个 layer 都做 collective，但这不代表跨节点发送可以忽略。长 sequence、大 hidden、多个 virtual stages 会让 P2P 流量进入 critical path。

## PP、TP 与 DP 怎样组成设备网格

只考虑三种并行时：

$$
W=p_{TP}\times p_{PP}\times p_{DP}
$$

例如 64 张 GPU 使用 TP=4、PP=4、DP=4：

- 每个 TP group 用 4 张卡共同完成一个 stage 内的 layers；
- 4 个 PP positions 组成一条 16-GPU model pipeline；
- 整条 model pipeline 再复制 4 份，处理不同 data shards。

每个 rank 同时属于多个 process groups，但通信语义不同：

```text
TP group: 同一个 layer/stage 内合并 feature shards
PP group: 相邻 stages 传 activation/gradient
DP group: 相同 TP×PP position 的 replicas 同步梯度或模型状态
```

常见 placement 是把高频 TP collective 放在单节点 NVLink/NVSwitch 域内，让 PP 的相邻 stage 跨节点，再用 DP 扩展 replicas。但最优映射取决于 topology、message size 和 overlap，不能只根据 global rank 连号分组。

## Gradient Accumulation 与 Loss Normalization

Pipeline stages 不应各自独立执行 optimizer step。Last stage 计算每个 micro-batch 的 loss，并沿 backward 把梯度传回；所有 stages 累积同一个 step 的参数梯度，最后共同更新。

若 micro-batches 的有效 token 数不同，简单对每个 micro-batch loss 先求平均、再对 $m$ 次梯度等权相加，可能与对全局有效 tokens 求平均不等价。更稳妥的方式是累积 loss numerator 和 token count，按全局有效 token 数归一化，或明确让每个 micro-batch gradient 按 token count 加权。

此外要保证：

- gradient clipping 使用所有 model shards 的全局 norm；
- GradScaler overflow/skip 决策在 PP、TP、DP ranks 上一致；
- 任一 stage 出现 NaN/Inf 时，所有 stages 都跳过同一次 update；
- learning-rate scheduler 的 step 数只随 optimizer commit 前进。

否则 pipeline 仍会运行，但不同 stages 已不再属于同一个参数版本。

## Activation Checkpointing 与 PP 的关系

PP 减少每张卡保存的 layers，1F1B 减少同时存活的 micro-batches；Activation Checkpointing 减少每个 micro-batch、每个 stage 内保存的中间 activations。三者作用在不同维度：

$$
M_{activation}
\approx
N_{live\ microbatches}
\times
N_{local\ layers}
\times
M_{saved\ per\ layer}
$$

- PP 降低 $N_{local\ layers}$；
- 1F1B 控制 $N_{live\ microbatches}$；
- recomputation 降低 $M_{saved\ per\ layer}$。

它们可以组合，但重算会改变 stage Forward/Backward 耗时，原来均衡的 layer partition 可能重新失衡。做完显存优化后必须重新测 stage time。

## 变长序列会破坏静态节拍

若 micro-batches 的 token 数差异很大，attention 与 MLP 计算时间也会变化。某个超长 micro-batch 会像“慢车”一样依次阻塞所有 stages，bubble 不再是规则三角形。

训练系统通常需要组合：

- 按 token 数而不是样本数构造 micro-batch；
- sequence packing 减少 padding；
- 让同一步内 micro-batches 的计算量尽量接近；
- 对 variable shapes 明确交换 P2P tensor metadata；
- 为不同 shape 准备通信 buffer 与 kernel plan；
- 记录 per-micro-batch、per-stage latency，而不是只看 step 平均值。

静态 shape 假设不一致是 PP hang 的常见来源：发送方发出 `[S,B,H]`，接收方却按另一组维度分配 buffer，结果可能是错误、越界或所有 ranks 永久等待。

## Deadlock 为什么比普通 OOM 更难定位

Pipeline schedule 是分布式状态机。每个 rank 的 send/recv 顺序必须与邻居严格匹配。常见死锁来源包括：

- 某个 rank 因数据耗尽少执行了一个 micro-batch；
- Forward 抛错后其他 ranks 仍在等待 P2P recv；
- last stage 的 loss 分支改变了 backward 次数；
- activation checkpoint 重算走了不同控制流；
- variable sequence length 的 shape handshake 不一致；
- interleaved chunks 的 send/recv tag 或顺序不一致；
- 一个 rank 发现 overflow 并提前跳过，其他 ranks 继续执行。

调试时应为每个事件记录：

```text
global step
micro-batch id
physical rank
pipeline rank
virtual chunk id
forward/backward
peer rank
tensor shape and dtype
send/recv issue and completion time
```

只打印“rank 3 卡住”通常不足以定位是哪一对状态机失配。

## Checkpoint 必须保存完整训练边界

最安全的 checkpoint 时刻是 pipeline 已 flush、optimizer step 已提交、所有 ranks 对 global step 达成一致之后。除了模型参数，还要保存：

- optimizer moments 与 master weights；
- LR scheduler、GradScaler 和 global step；
- data sampler/cursor 与 consumed tokens；
- RNG states，包括 DP/TP/PP 下的 tracker；
- parallel layout 与每个 tensor 的 logical placement；
- checkpoint 是否完整提交的 metadata。

如果在 pipeline 中途保存，还要捕获所有 in-flight micro-batches、activation、gradient 和通信状态，复杂度远高于从 step boundary 恢复。生产系统通常选择一致的 step boundary，再用异步 I/O 缩短停顿，而不是尝试序列化整个流水状态机。

## 怎样读 Pipeline Profiler

端到端平均 MFU 不能说明 bubble、失衡还是通信在浪费时间。至少同时观察：

1. 每个 physical/virtual stage 的 Forward 与 Backward 时间；
2. warmup、steady 1F1B、cooldown 的 slots；
3. P2P send/recv 的 issue、wait 与实际 overlap；
4. 每个 stage outstanding activations 与峰值显存；
5. 最慢 stage 以及其他 stages 的等待时间；
6. micro-batch shape 与 token 数分布；
7. TP/DP collectives 是否与 PP P2P 争用链路；
8. optimizer step、checkpoint 与 data loading 是否形成额外 flush。

可以先从一个简单利用率估算开始：

$$
U_{pipeline}
\approx
(1-f_{bubble})
\times
U_{balance}
\times
U_{compute}
$$

其中 $U_{balance}$ 描述 stage 不均衡，$U_{compute}$ 描述每个 stage 内 kernel 的实际效率。即使理论 bubble 只有 5%，local GEMM 太小或某 stage 慢 20%，整体仍不会接近满载。

## 一条可执行的配置路径

1. **单卡建立数值基线**：固定 data、dropout 与 optimizer，保存 loss/gradient；
2. **先做两段 Pipeline**：验证 activation 与 gradient 的 shape、dtype 和数值；
3. **使用 Flush Schedule**：先保证一个 step 内参数版本一致；
4. **增加 Micro-batches**：观察 bubble、local GEMM 与 global batch；
5. **切换 1F1B**：验证 loss parity，并测 activation 峰值；
6. **按时间重新分层**：把 embedding、loss、recompute 和 MoE 成本计入；
7. **加入 TP**：显式生成 TP×PP groups，验证 boundary placement；
8. **再加入 DP/FSDP**：只在相同 model-position replicas 间同步或分片；
9. **尝试 Interleaving**：比较减少的 bubble 与增加的 P2P；
10. **覆盖异常路径**：数据耗尽、NaN、OOM、rank failure 与恢复；
11. **用 tokens/s、显存和收敛共同验收**：不能只比较 schedule 图是否更满。

## 常见误区

### “把 Layers 平均分给 GPU 就完成了 PP”

模型只是在显存上放下了。没有 micro-batch schedule、均衡分区和匹配的 P2P 状态机，设备大部分时间仍会等待。

### “1F1B 会消除所有 Bubble”

同步 flush 仍有填充与排空成本。1F1B 首先改善 activation 生命周期；interleaving 和更多 micro-batches 才进一步摊薄 bubble。

### “Micro-batch 越多，吞吐越高”

更多 micro-batches 会减小 bubble，但也可能让单次 GEMM 太小、global batch 过大、step latency 和调度开销上升。

### “PP 只需传一份 Activation”

每个 micro-batch 的 Forward 要传 activation，Backward 还要反向传 gradient；virtual stages 会增加边界次数。

### “不 Flush 只是更激进的性能优化”

跨 step 持续运行会触及参数版本与梯度一致性，需要 weight stashing 或新的训练算法，不能默认与同步 SGD 等价。

## 小结

Pipeline Parallel 沿模型深度切分 layers，并用 micro-batches 把不同 stages 填成流水。理解它需要同时看四条线：

1. **计算线**：Forward/Backward 如何经过 warmup、steady 1F1B 与 cooldown；
2. **显存线**：每个 micro-batch 的 activation 从 Forward 存活到对应 Backward；
3. **通信线**：相邻 stages 传 activation/gradient，TP/DP collectives 还会共享拓扑；
4. **一致性线**：所有 micro-batches 完成后才提交 optimizer step，保证参数版本一致。

GPipe 的全 Forward/全 Backward 容易理解，但 activation 驻留多；同步 1F1B 更早回收 activation；interleaved 1F1B 用多个 virtual stages 进一步降低 bubble，却增加调度与 P2P 复杂度。真正的性能取决于 micro-batch shape、最慢 stage、网络映射和 local kernel 效率，而不只是一张理想时间线。

下一篇会继续讨论 Sequence Parallel 与 Context Parallel：它们都沿 token 维切 activation，却为什么只在不同区域生效，以及 Attention 为什么必须额外交换全局上下文。

## 参考资料

- [GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://arxiv.org/abs/1811.06965)
- [PipeDream: Fast and Efficient Pipeline Parallel DNN Training](https://arxiv.org/abs/1806.03377)
- [Memory-Efficient Pipeline-Parallel DNN Training](https://arxiv.org/abs/2006.09503)
- [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473)
- [NVIDIA Megatron Core: Pipeline Parallel Schedules](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.pipeline_parallel.schedules.html)
- [NVIDIA Megatron Core: Parallelism Strategies Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)
