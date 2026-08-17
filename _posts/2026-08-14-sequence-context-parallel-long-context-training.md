---
layout: post
title: "Sequence Parallel 与 Context Parallel：Token 维到底怎样切"
subtitle: "从 LayerNorm Activation 分片到跨卡 Attention，厘清 SP、CP、Ring 与 All-to-All 的边界"
date: 2026-08-14 09:00:00 +0800
last_modified_at: 2026-08-17
author: iStar
catalog: true
series: distributed-training
series_order: 40
technology_year: 2022
mathjax: true
tags: [分布式训练, 注意力机制, GPU优化]
---

Sequence Parallel（SP）与 Context Parallel（CP）都会把 `[sequence, batch, hidden]` activation 沿 token 维切到多张 GPU，因此很容易被当成同一种技术。真正的区别不在“切哪个维度”，而在分片保持到哪里：SP 主要消除 Tensor Parallel ranks 之间重复保存的 LayerNorm、dropout、residual activations；CP 则让网络输入和几乎全部 activations 都保持 sequence-sharded，并专门解决 Attention 对全局上下文的依赖。

可以先记住一句话：

```text
SP：在 TP block 的边界省 activation，进入需要完整 token 范围的计算前再聚合
CP：整个网络都只持有本地 token，Attention 内显式交换远端上下文
```

两者都不是“把 sequence 平均切开就结束”。要判断实现是否正确，必须追踪每个 tensor 的 logical placement、collective 和 global token position。

## 从 Transformer Activation 的 Shape 开始

设一个 Transformer block 的输入为：

$$
X\in\mathbb{R}^{S\times B\times H}
$$

其中 $S$ 是 sequence length，$B$ 是 micro-batch size，$H$ 是 hidden size。若使用 $p$ 张卡沿 sequence 分片，每张卡理想上只保存：

$$
X_r\in\mathbb{R}^{(S/p)\times B\times H}
$$

单份 activation 的理论显存从：

$$
M_X=S\times B\times H\times bytes(dtype)
$$

下降为：

$$
M_{X_r}\approx\frac{M_X}{p}
$$

但这个除以 $p$ 只有在 tensor 整个生命周期都保持分片时才成立。若下一层立刻 AllGather 回完整 $S$，峰值显存仍可能出现完整 activation；若 Attention 还 materialize 大型临时 tensor，真正峰值也不只由 block input 决定。

## 为什么 Tensor Parallel 仍会复制大量 Activation

Megatron-style Tensor Parallel 把 MLP 与 Attention 的线性层沿 hidden/features 或 heads 切开。Column-parallel linear 产生 feature-sharded 输出，row-parallel linear 产生 partial output，传统实现用 AllReduce 恢复 replicated residual：

```text
replicated residual X
  → column-parallel linear
  → feature-sharded hidden
  → row-parallel linear
  → partial [S,B,H] on every rank
  → AllReduce
  → replicated [S,B,H] on every rank
```

AllReduce 后，每个 TP rank 都持有相同的完整 residual。LayerNorm、dropout 和 residual add 虽然计算量不大，却让 $S\times B\times H$ activation 在 TP group 内复制 $p_{TP}$ 份。

TP 已经切开 parameters 和大 GEMM，并不等于所有 activations 都自然分片。Sequence Parallel 正是针对这段 replicated 区域。

## Sequence Parallel：把 AllReduce 拆成 ReduceScatter 与 AllGather

从通信语义看：

$$
\operatorname{AllReduce}
=
\operatorname{AllGather}
\circ
\operatorname{ReduceScatter}
$$

传统 row-parallel output 把 partial results AllReduce 后，所有 ranks 立刻拿到完整 `[S,B,H]`。SP 改成 ReduceScatter：先完成跨 rank 求和，同时让每个 rank 只保留一段 sequence。

```text
row-parallel partial outputs
  → ReduceScatter(sequence)
  → sequence-sharded residual
  → local dropout / residual / LayerNorm
  → AllGather(sequence)
  → next column-parallel linear
```

本来 AllReduce 已经包含等价的 reduce 与 gather 通信。SP 并非额外凭空增加一整套数据搬运，而是推迟 AllGather，让两个 collectives 之间的 activation 保持 sequence-sharded，从而减少驻留显存。

这段区域里的操作必须对 token 独立：

- LayerNorm 通常沿 hidden dimension 归一化，每个 token 可独立计算；
- elementwise activation、bias、dropout 可对本地 tokens 执行；
- residual add 只要两侧使用相同 token shard，也可本地完成；
- token-wise MLP 的非线性本身不需要看其他 tokens。

若某个操作沿 sequence 求统计量，或者需要任意 token 与其他 token 交互，就不能直接把它留在普通 SP 本地区域。

## SP 的 Forward 与 Backward Placement

设 TP size 为 $p$，SP 与 TP 使用同一个 process group。一个简化 block 的 Forward placement 是：

```text
X: sequence-sharded
  → local LayerNorm
  → AllGather sequence
X_full: replicated over sequence, or enters TP-specific layout
  → column/row-parallel Attention or MLP
  → ReduceScatter sequence
Y: sequence-sharded
  → local dropout + residual
```

Backward 按相反方向恢复 tensor placement。Forward 的 AllGather 对应 Backward 的 ReduceScatter；Forward 的 ReduceScatter 对应 Backward 的 AllGather。实现还可能把 collective 与 GEMM 分块重叠，但 logical result 不变。

最常见的错误不是 shape 不匹配，而是 shape 相同、语义错误。例如每个 rank 都有 `[S/p,B,H]`，但：

- 它可能是不同 token ranges 的有效 shards；
- 可能是对同一 token range 的 partial sums；
- 也可能是错误地重复保存同一段 tokens。

因此日志和 API 不能只记录 shape，还要记录 `global_offset`、shard dimension、是否 partial 以及所属 process group。

## SP 为什么没有解决长上下文 Attention

Self-Attention 对每个 Query token $i$ 需要读取所有允许位置的 Key/Value：

$$
O_i
=
\operatorname{softmax}
\left(
\frac{Q_iK^T}{\sqrt d}+M_i
\right)V
$$

即使 rank $r$ 只负责一段 $Q_r$，它仍需要全局 $K,V$ 才能得到精确结果。普通 SP 在 Attention 前把所需 sequence 数据聚合回来，然后仍按 TP heads/features 计算；它只是让 Attention 之外的一段 activation 生命周期被分片。

所以 SP 主要解决“TP ranks 为什么重复保存 token-local activation”，并没有让单个 Attention head 的上下文天然跨设备扩展。真正把 Attention 的 sequence 也切开的，是 Context Parallel 或 Ring Attention 一类算法。

## Context Parallel：让本地 Query 看见全局 KV

CP 把输入和整网 activations 沿 sequence 分给独立的 CP ranks：

```text
rank 0 owns tokens 0..S/p-1
rank 1 owns tokens S/p..2S/p-1
...
rank p-1 owns the final token range
```

Linear、LayerNorm、MLP 等不跨 token 的模块可以直接处理本地 shard。Attention 必须额外通信，使本地 Queries 与全局 Keys/Values 发生交互。

最直观的实现是 AllGather KV：

```text
local Q_r, K_r, V_r
  → AllGather K/V across CP group
  → local Q_r attends to full K/V
  → keep only output O_r for local queries
```

它简单，但每张卡会在 Attention 内临时持有完整 KV，通信与峰值显存可能较高。GQA/MQA 的 KV heads 较少时，这条路径可能仍有吸引力；MHA 或超长上下文下则需要更流式的办法。

## P2P Ring：让 KV Blocks 依次经过每个 Query Shard

Ring-style CP 固定本地 $Q_r$，把 $K,V$ blocks 沿环传递。每收到一块，就计算本地 Query 对该块的 attention contribution，同时发送上一块、接收下一块：

```text
step 0: Q_r × local KV_r
step 1: Q_r × KV_(r-1)
step 2: Q_r × KV_(r-2)
...
```

不能分别对每个 KV block 做 Softmax 后再把 outputs 相加，因为 Softmax denominator 是全局的。Blockwise exact attention 要维护 online statistics。若第 $j$ 块得到 local maximum $m_j$、exponential sum $l_j$ 和未归一化 weighted value $u_j$，合并到 running state 时需要按新的最大值重新缩放：

$$
m'=\max(m,m_j)
$$

$$
l'=e^{m-m'}l+e^{m_j-m'}l_j
$$

$$
u'=e^{m-m'}u+e^{m_j-m'}u_j
$$

最终：

$$
O=\frac{u}{l}
$$

这样无需 materialize 完整 attention matrix，也无需在单卡保存完整 KV。P2P 传输能否隐藏，取决于每个 local attention block 的计算时间是否足以覆盖下一块 KV 的通信。

## All-to-All：在 Sequence 与 Head 维之间转置

另一类 CP 做法使用 All-to-All，在 sequence-sharded 与 head-sharded layout 之间转换。可以把它理解为一次分布式转置：

```text
before A2A:
  each rank has all local heads for S/p tokens

after A2A:
  each rank has a subset of heads for full S tokens
```

每个 rank 获得完整 sequence、较少 heads，于是本地完成这些 heads 的 Attention；输出再通过反向 All-to-All 恢复 sequence shards。DeepSpeed Ulysses 属于这类思路。

它避免环形多步传输，却要求 head 数、CP degree 与 layout 合法匹配。若 GQA 的 KV heads 很少，继续沿 heads 切分可能出现每 rank 不足一个 KV head，需要复制、分组或采用混合路径。

## CP 通信方式不是一个固定算法

现代训练栈通常提供多种 Context Parallel communication：

| 方式 | 核心动作 | 优势 | 主要约束 |
| --- | --- | --- | --- |
| AllGather | 聚合完整 KV 后计算 | 简单，易验证 | 临时 KV 与暴露通信较大 |
| P2P/Ring | KV blocks 环形流动 | 可与 block attention 重叠 | 多步调度、mask 与负载均衡复杂 |
| All-to-All | sequence/head layout 转置 | 通信轮次清晰 | 受 head 数与 A2A 拓扑限制 |
| A2A+P2P | 节点内 A2A、节点间 P2P 等分层组合 | 贴合分层互联 | process groups 与调优更复杂 |

Megatron Core 当前把这些路径暴露为 `p2p`、`all_gather`、`a2a` 与 `a2a+p2p`。选择依据应是模型的 Q/KV head 数、sequence length、节点内外拓扑、通信能否重叠和 kernel block shape，而不是只看算法名字。

## Causal Mask 会产生负载不均衡

双向 Attention 中，每个 Query shard 大致处理相同数量的有效 QK pairs。Causal Attention 只允许 token 看到自己及更早位置。若简单按连续 sequence 分片：

- 早期 Query shard 可见的历史较短；
- 后期 Query shard 要处理更多有效 KV blocks；
- 不同 ranks 的有效 attention work 明显不均衡。

一种思路是让每个 rank 同时拥有靠前和靠后的 token chunks，用互补位置平衡有效三角区域；另一种是在 block schedule 中跳过完全被 mask 的块，并重新安排通信/计算顺序。

无论使用哪种方式，global position 必须保持正确。把本地 token index `0..S/p-1` 直接当成全局位置，会同时破坏 causal mask、RoPE phase 和数据对齐。

## Position、Mask 与 RNG 必须跟随 Global Token

Sequence sharding 后至少有三类隐式状态需要显式化。

### Global Position

Rank $r$ 的第 $j$ 个本地 token 对应：

$$
position=offset_r+j
$$

若使用 packed sequences，offset 还要结合每个 sample 的 segment boundaries，而不是只看物理 tensor 下标。

### Attention Mask

Mask 要根据 global query/key positions、segment id、padding 和 causal/bidirectional 规则生成。环形 KV 走到新 rank 后，它携带的不是“当前本地第几个 block”，而是明确的 global key range。

### RNG Mapping

Dropout mask 应能由 global token、layer、micro-batch 和 RNG stream 唯一决定。Checkpoint 恢复或 CP degree 改变后，若随机数仅依赖 local rank 与 local tensor shape，数据相同也可能得到不同 mask，导致不可复现甚至破坏训练等价性。

## SP、CP 与 TP 怎样组成二维切分

假设 TP=4、CP=2，一份 model replica 需要 8 张 GPU。可以把 ranks 画成二维网格：

```text
             TP position
           0   1   2   3
CP shard 0  0   1   2   3
CP shard 1  4   5   6   7
```

- `[0,1,2,3]` 和 `[4,5,6,7]` 分别是 TP groups，共同切 layer features/heads；
- `[0,4]`、`[1,5]`、`[2,6]`、`[3,7]` 是 CP groups，交换对应 TP position 的上下文；
- SP 通常复用每行的 TP group，让 token-local activations 在 TP ranks 间 sequence-sharded；
- CP 则沿列切完整网络的 token ranges。

加入 PP 与 DP 后，总卡数在不考虑 EP 等额外维度时为：

$$
W=p_{TP}\times p_{CP}\times p_{PP}\times p_{DP}
$$

每种 collective 必须落在正确 group。把 CP KV exchange 错发到 TP group，shape 可能仍合法，却会把不同 heads 而不是不同 token ranges 拼在一起。

## SP 与 CP 同时启用时发生什么

CP 先决定每个 CP rank 拥有全局 sequence 的哪一段；在每个 CP shard 内，TP/SP 还可以进一步处理这段局部 sequence。

若全局 $S$ 个 tokens，CP size 为 $c$，TP/SP size 为 $t$，某些 SP-resident activations 的每 rank token 数可能近似为：

$$
S_{local}\approx\frac{S}{c\times t}
$$

但不能由这个公式推断所有 tensors 都除以 $ct$：

- TP linear 内部可能暂时 AllGather SP shards；
- CP attention 会交换或重排 KV/context；
- parameters 沿 TP 切、在 CP group 内通常复制；
- gradient synchronization 还要结合 DP×CP replicas；
- fused kernel 可能要求额外 padding 与 alignment。

正确做法是逐算子标注 placement，而不是给整个模型贴一个“sequence 已除以 $ct$”的标签。

## GQA、MQA 与 MLA 怎样改变 CP 成本

CP Attention 的通信主要与传输的 K/V 表示相关。MHA 的 KV heads 与 Query heads 同量，KV payload 较大；GQA/MQA 减少 KV heads，因此 AllGather 或 Ring 中每个 token 的 KV bytes 下降。

若每个 token、每层 KV payload 近似为：

$$
V_{KV/token}
=
2\times n_{kv}\times d_h\times bytes(dtype)
$$

减小 $n_{kv}$ 会直接减少 CP 通信。MLA 则可能传递压缩 latent 和位置相关分量，具体收益取决于训练实现是否能在 Attention 计算前后保持压缩表示，而不是在通信前就展开为完整 K/V。

因此同一个 CP degree 在 MHA、GQA 和 MLA 上的瓶颈可能完全不同。配置必须基于真实 boundary tensors 测量。

## Packed Sequence 与变长样本为什么更难

长上下文训练常把多个短样本 pack 到固定 token budget。此时一个 physical sequence tensor 里可能包含多个互相不可见的 segments。CP 分片不能只切连续字节，还要保留：

- 每个 token 的 segment id 与 global-in-segment position；
- 跨 CP shard 的 segment boundaries；
- causal mask 在 segment 处重置；
- load balance 不能把 padding 节省又变成某些 ranks 的空闲；
- checkpoint 后 data loader 要恢复同一 packing/cursor 状态。

若不同 CP ranks 的有效 token 数差异大，local MLP、Attention blocks 与通信 payload 都会失衡。仅让 padded tensor shape 整齐，并不代表有效工作均衡。

## 与 Activation Checkpointing 的取舍

不使用 CP 时，长 sequence OOM 常通过 activation recomputation 缓解：少保存中间 activation，Backward 时重跑 Forward。SP/CP 则通过分片减少每 rank activation。

两者可以组合，但优化目标不同：

- recomputation 用额外 FLOPs 换显存；
- SP 主要去掉 TP ranks 间的 activation replication；
- CP 用额外 context communication 换取全网 sequence 分片；
- selective recomputation 只重算显存占比高、计算相对便宜的算子。

选择时应比较 step time，而不是只看峰值显存。CP 若使 Attention 通信完全暴露，可能比保留较小 CP、增加 selective recomputation 更慢。

## 怎样选择 SP 与 CP

可以按问题来源判断：

### 使用 TP 后，LayerNorm/Residual Activation 复制过多

优先考虑 SP。它复用 TP group，并把 AllReduce 拆成 ReduceScatter/AllGather，通常不改变 Attention 的全局计算方式。

### 单个 Sequence 的 Attention/Activation 无法放入现有 TP 布局

考虑 CP。它让完整网络沿 sequence 分片，但要为 Attention 选择 KV AllGather、P2P Ring、A2A 或分层通信。

### Sequence 不长，主要瓶颈是 Parameter/Optimizer State

SP/CP 不是第一选择。先看 FSDP/ZeRO、TP、PP 与 activation checkpointing。

### Head 数少、GQA/MQA 明显

KV AllGather 可能比想象中便宜；A2A 沿 head 切分反而受 head 数限制。应以真实 Q/KV layout 选择。

### 跨节点网络远慢于节点内 NVLink

考虑把 TP 放在节点内，CP 使用拓扑感知的 P2P 或 A2A+P2P 层次结构，并验证通信/计算 overlap。

## 正确性验证不能只对最终 Loss

小规模 reference 测试应逐层比较：

1. 无 SP/CP 的单卡或复制基线；
2. LayerNorm、residual 和 dropout 后的 global tensor；
3. Q/K/V 的 global position 与 head ownership；
4. Attention online Softmax 的 max、sum 与 output；
5. causal/packed mask 的有效 blocks；
6. input gradient 与 parameter gradient；
7. optimizer update 后的 global weights；
8. checkpoint 恢复后的下一步 loss；
9. 改变 SP/CP degree 后的 reshard parity。

测试 tensor 应刻意包含：不能被 shard size 整除的长度、多个 packed segments、GQA 少 KV heads、dropout、causal 与 bidirectional mask。只用全零 mask、无 dropout、规则长度，会漏掉最危险的 placement 问题。

## 性能分析要看哪些量

至少记录：

- 每 rank peak activation memory 与 temporary KV memory；
- SP AllGather/ReduceScatter bytes、等待时间与 overlap；
- CP 每层的通信类型、payload 与 exposed latency；
- local attention block shape 与 kernel efficiency；
- causal/packed sequence 的有效 work imbalance；
- TP、CP collectives 是否争用同一链路和 SM；
- recomputation FLOPs 与节省的 activation bytes；
- tokens/s、MFU、step time P50/P99 与 loss parity。

“通信 kernel 与 GEMM 在 timeline 上重叠”不等于通信被免费隐藏。两者可能争抢 HBM bandwidth、NVLink 或 SM，导致 GEMM 本身变慢。应比较重叠前后的整个 critical path。

## 常见误区

### “SP 和 CP 都切 Sequence，所以只是不同名字”

SP 主要作用于 TP block 边界的 token-local activations；CP 让整个网络保持 sequence-sharded，并在 Attention 内交换全局上下文。

### “AllGather 之后显存还是完整的，所以 SP 没有意义”

SP 的价值在于缩短完整 activation 的驻留区间，并让 LayerNorm、dropout、residual 等区域保持分片。峰值是否下降要结合具体 buffer 生命周期判断。

### “CP 会把 Attention 计算复杂度除以设备数，所以一定线性加速”

每 rank 计算量下降，但全局 dense Attention 的 $O(S^2)$ 工作没有消失；还增加了 KV/context 通信、负载均衡和 kernel 变小的成本。

### “Local Token Index 就是 Position ID”

Sequence shard 的本地下标必须加 global offset，并处理 packed segment boundaries，否则 RoPE 与 causal mask 都会错误。

### “CP Size 越大，能支持的上下文越长，性能也越好”

更大 CP 会缩小本地 block、增加通信参与者并改变拓扑路径。它扩展的是容量上限，不保证时间线性扩展。

## 小结

Sequence Parallel 与 Context Parallel 的共同点只是沿 token 维分片；它们解决的是不同层次的问题：

1. SP 复用 TP group，把 row-parallel AllReduce 拆成 ReduceScatter 与延迟的 AllGather；
2. 两个 collectives 之间的 LayerNorm、dropout、residual activations 保持 sequence-sharded；
3. 普通 SP 在 Attention 前仍会聚合所需 sequence，不能独立扩展单个 head 的上下文；
4. CP 让输入和整网 activations 沿 sequence 分片，并在 Attention 内交换全局 K/V；
5. CP 可使用 KV AllGather、P2P Ring、All-to-All 或分层混合通信；
6. causal mask、global position、packed segments 与 RNG 必须跟随 logical token；
7. SP、CP、TP、PP、DP 组成多维 mesh，每个 tensor 和 collective 都要落在正确 group；
8. 最优方案取决于激活显存、Q/KV heads、网络拓扑、block shape 与 recomputation 成本。

下一篇会进入分布式 Checkpoint：当参数、Optimizer State 和 RNG 已经分散在多维 device mesh 上，怎样保存一份可验证、可异步提交，并能在不同并行度下恢复的训练状态。

## 参考资料

- [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198)
- [NVIDIA Megatron Core: Context Parallel Package](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html)
- [NVIDIA Megatron Core: Parallelism Strategies Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)
- [NVIDIA Megatron Core: TransformerConfig `cp_comm_type`](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.transformer.transformer_config.html)
- [Ring Attention with Blockwise Transformers for Near-Infinite Context](https://arxiv.org/abs/2310.01889)
- [DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models](https://arxiv.org/abs/2309.14509)
- [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473)
