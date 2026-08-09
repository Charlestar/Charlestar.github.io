---
layout: post
title: "Ring Attention：把超长序列沿设备环流动"
subtitle: "从分块 Online Softmax 到 KV 通信与本地 Attention 计算重叠"
date: 2026-08-09 15:20:00 +0800
last_modified_at: 2026-08-09
author: iStar
catalog: true
series: attention-long-context
series_order: 40
technology_year: 2023
mathjax: true
tags: [AI Infra, Ring Attention, 长上下文, 分布式训练]
---

FlashAttention 让单张 GPU 不再物化完整的 $N\times N$ attention matrix，但一条足够长的序列仍然可能放不进单卡。

原因不只来自 attention score。模型在训练时还要保存输入 hidden states、Q/K/V、输出、归一化统计与反向传播需要的状态；即使使用重计算，单卡也要容纳整条序列的 $O(Nd)$ 激活。序列从 32K 增长到 1M token 时，线性状态本身就足以越过显存上限，dense attention 的计算量更会按 $N^2$ 增长。

Ring Attention 的思路是把序列切成多个连续块，每张设备只持有其中一块 Q/K/V。计算 attention 时，Q 留在本地，K/V block 沿设备组成的环逐站传递；每收到一块 K/V，就运行一次本地 blockwise attention，并用 online Softmax 把局部结果精确合并。

```text
device 0 owns Q0, K0, V0
device 1 owns Q1, K1, V1
device 2 owns Q2, K2, V2
device 3 owns Q3, K3, V3

Q stays local; K/V rotate around the ring.
```

当传输下一块 K/V 的时间不超过当前块 attention 的计算时间，通信就能隐藏在计算后面。经过设备数那么多轮之后，每个本地 Q block 都看过全序列 K/V，得到与标准 dense attention 相同的结果。

本文从单卡显存边界出发，推导环形计算如何保持 Softmax 精确性，再讨论通信/计算重叠、causal mask、反向传播和与其他并行策略的组合边界。

## 单卡 FlashAttention 的边界在哪里

设 batch size 为 $B$、序列长度为 $N$、hidden size 为 $D$、注意力头维度为 $d$。FlashAttention 避免保存每个 head 的 $N\times N$ 中间矩阵，因此 attention 额外显存不再按 $O(N^2)$ 增长。

但模型仍然至少需要处理线性规模的状态：

$$
X,Q,K,V,O\in\mathbb{R}^{B\times N\times D}
$$

训练还会叠加各层激活、梯度、optimizer state 和临时 workspace。Activation checkpointing 能用重算换显存，不能让本地 $N$ 无限增长。

更重要的是，单 GPU 即使勉强放下输入，attention 算术量仍约为：

$$
F_{attn}=O(BN^2dH)
$$

其中 $H$ 是 head 数。FlashAttention 优化 IO 与 kernel 效率，没有改变 dense attention 的二次计算。

于是长上下文扩展有两个相互独立的问题：

1. **容量**：一条序列的线性状态怎样分到多张设备；
2. **执行**：全局 Q/K/V 依赖怎样在多设备上完成，且不被通信拖垮。

Ring Attention 主要解决这两个系统问题，不减少模型原本要计算的 token pair 数。

## 为什么沿序列维切分

假设有 $P$ 张设备，将长度 $N$ 的序列均匀切分，每张设备持有：

$$
N_{local}=\frac{N}{P}
$$

个连续 token。对设备 $p$：

$$
Q_p,K_p,V_p\in\mathbb{R}^{N_{local}\times d}
$$

这样 embedding、RMSNorm/LayerNorm、MLP 等按 token 独立的算子都可以在本地块执行。单卡激活长度从 $N$ 降到 $N/P$，理想情况下可处理的全局序列长度随设备数近似线性增长。

难点集中在 self-attention。设备 $p$ 的每个 $Q_p$ 仍要看到全局：

$$
K=[K_0;K_1;\ldots;K_{P-1}],
\quad
V=[V_0;V_1;\ldots;V_{P-1}]
$$

最直接的做法是 all-gather 所有 K/V 到每张卡，再运行本地 Q 对全局 KV 的 attention。计算是正确的，却让每张卡重新持有长度 $N$ 的 K/V，破坏了序列分片的容量目标。

Ring Attention 不 all-gather 完整 K/V，而是让每张卡任何时刻只接收一个远端 block，算完后再转发。

## 环是怎样工作的

以 4 张设备为例，初始时每张卡持有本地 Q/K/V。环方向设为：

```text
device 0 -> device 1 -> device 2 -> device 3 -> device 0
```

第 0 轮，每张卡计算本地块：

```text
device 0: Q0 x (K0,V0)
device 1: Q1 x (K1,V1)
device 2: Q2 x (K2,V2)
device 3: Q3 x (K3,V3)
```

同时将当前 K/V 发给下一张卡。第 1 轮：

```text
device 0 receives (K3,V3): Q0 x (K3,V3)
device 1 receives (K0,V0): Q1 x (K0,V0)
device 2 receives (K1,V1): Q2 x (K1,V1)
device 3 receives (K2,V2): Q3 x (K2,V2)
```

继续旋转，经过 $P$ 轮后，每个 $Q_p$ 与所有 K/V blocks 都完成 attention。K/V 始终以块为单位流动，每张卡只需本地 block、一个接收 buffer 和一个发送/计算中的 buffer，而不需要完整复制全序列。

可以把 forward 抽象成：

```python
kv = local_kv
state = empty_online_softmax_state(local_q)

for step in range(world_size):
    next_kv = async_send_recv(kv)
    partial = block_attention(local_q, kv, mask_for(step))
    state = merge_online_softmax(state, partial)
    kv = wait(next_kv)

local_output = finalize(state)
```

关键不在 Python 循环，而在 `async_send_recv` 与 `block_attention` 是否能真正并发，以及 `merge_online_softmax` 是否保持全局 Softmax 的数值语义。

## 局部 Softmax 为什么不能直接相加

设备 $p$ 的 Q block 分别对多个 K blocks 计算 score：

$$
S^{(j)}=Q_pK_j^T
$$

若对每个 $S^{(j)}$ 单独做 Softmax，再把输出相加：

$$
\sum_j \operatorname{softmax}(S^{(j)})V_j
$$

结果并不等于全局 attention。每个局部 Softmax 都把自己的概率和归一化为 1，而全局 Softmax 应让所有 K positions 共同竞争同一个分母。

正确合并需要维护每个 query row 的：

- 当前全局最大值 $m$；
- 按该最大值缩放后的指数和 $\ell$；
- 未最终归一化的 value 累积 $\widetilde O$。

对新 score block $S$：

$$
m' = \max(m,\operatorname{rowmax}(S))
$$

$$
\ell'
= e^{m-m'}\ell
+\operatorname{rowsum}(e^{S-m'})
$$

$$
\widetilde O'
=e^{m-m'}\widetilde O
+e^{S-m'}V
$$

所有 K/V blocks 处理完后：

$$
O=\frac{\widetilde O}{\ell}
$$

这与 FlashAttention 跨片上 tiles 合并 Softmax 是同一个代数性质。区别只是边界变了：FlashAttention 的 KV tiles 在单卡 HBM 与 SRAM 间流动，Ring Attention 的 KV blocks 还要跨设备网络流动。

因此 Ring Attention 不是一种近似 attention pattern。只要每个允许的 Q-K pair 都被计算，并按 online Softmax 合并，它得到的仍是标准 dense attention。

## 两层分块如何嵌套

一张设备接收到的 K/V shard 仍可能远大于片上 SRAM，不能一次全部放进一个 kernel tile。实际执行通常包含两层：

```text
设备层 block:
  sequence shard on device j
      |
      +-- GPU tile 0
      +-- GPU tile 1
      +-- GPU tile 2
      +-- ...
```

外层 Ring Attention 负责设备间 sequence shard 的轮转；内层 FlashAttention/块注意力 kernel 负责本地 Q block 与当前 K/V shard 的 IO-aware 计算。

Online Softmax 统计可以在内层 kernel 中处理完一个设备 block后，输出可继续合并的状态，也可以让一个 kernel 针对当前远端 block直接更新本地累计 buffer。具体融合程度取决于实现和编译器。

这说明 Ring Attention 与 FlashAttention 不是竞争方案：

- FlashAttention 解决单设备 HBM ↔ SRAM 的 tile 数据流；
- Ring Attention 解决设备 ↔ 设备的 sequence block 数据流；
- 两者共同保持精确 attention 的全局归一化。

上一代引擎优化可作为下一层分布式算法的本地 primitive。

## 通信为什么可能被隐藏

每个 ring step 同时有两件事：

1. 对当前 K/V block 做本地 attention；
2. 将下一块 K/V 异步传入，同时把当前块传出。

若单步计算时间为 $T_{compute}$，通信时间为 $T_{comm}$，理想流水线的 step 时间近似为：

$$
T_{step}\approx\max(T_{compute},T_{comm})
$$

当：

$$
T_{compute}\ge T_{comm}
$$

通信位于计算阴影内，对关键路径几乎不增加额外时间。若 block 太小或网络太慢，$T_{comm}>T_{compute}$，GPU 会等下一块数据，环就变成通信受限。

用量级看，单步每卡 attention 计算约为：

$$
F_{step}=O\left(\left(\frac{N}{P}\right)^2d\right)
$$

传输一块 K/V 的数据量约为：

$$
M_{step}=O\left(2\frac{N}{P}d\cdot bytes\right)
$$

序列块增大时，计算按 block length 的平方增长，通信按一次方增长，因此足够长的 block 更容易覆盖通信。这也是论文强调 block size 要足够大的原因。

但这不代表序列越长越快。全局 dense attention 总计算仍为二次，只是计算/通信比更有利；绝对训练时间会继续快速增长。

## 双缓冲是重叠成立的基础

如果代码先 `recv`、再 compute、再 `send`，公式上的 overlap 不会自动发生：

```text
recv K/V -> wait -> compute -> send -> wait -> next step
```

常见实现需要至少两个 K/V buffers：

```text
buffer A: 当前 step 正在被 attention kernel 读取
buffer B: 通信流正在接收下一 step 的 K/V
```

当前计算完成后交换角色。还要用独立 stream/event 管理依赖：

```text
communication stream: receive into B -----------+
                                                 |
compute stream      : use A -> record done       |
                                      |          |
                                      +-> swap <-+
```

正确性要求 buffer A 在计算完成前不能被下一次发送或接收覆盖，buffer B 在通信完成前不能被计算读取。若使用 collective library，还要确认它是否真的支持与计算 kernel 并行，以及网络/NVLink DMA 是否与当前 GPU 计算争用 HBM 带宽。

“异步 API 已调用”不等于“时间线上已重叠”。只有 profiler 中 send/recv 与 attention kernel 真实交叠，才能认为通信被隐藏。

## 每张卡到底通信多少数据

一个 K/V block 要沿环经过其他设备。每个 step 每张卡发送和接收一块，持续 $P-1$ 次远端轮转。忽略本地第 0 轮，每卡 forward 通信量近似：

$$
M_{device}
\approx 2(P-1)\frac{N}{P}Hd\cdot bytes
$$

当 $P$ 增大时，$(P-1)/P$ 接近 1，因此每卡总通信量在量级上约为全序列一份 K/V，而不是把完整 K/V 一次性常驻显存。

这里的“没有额外通信开销”应谨慎理解。论文指在满足 overlap 条件时，通信不增加关键路径时间；网络仍然实际传输了数据，占用了链路和 buffer，也可能与其他 TP/DP collectives 竞争。

若拓扑不是物理环，逻辑相邻设备跨越多个交换机，单步带宽和延迟都会恶化。Rank mapping 是性能配置的一部分，而不是无关的编号。

## Causal Attention 让环不再完全对称

Non-causal self-attention 中，每个 Q block 都需要所有 K/V blocks，每张卡做相同数量的 block matmul，负载自然均衡。

Autoregressive causal attention 只允许全局位置 $q$ 看到 $k\le q$。若设备按连续顺序持有：

```text
device 0: earliest tokens
device 1: next tokens
device 2: next tokens
device 3: latest tokens
```

则：

- device 0 的 Q 只需要最早 K/V block 的下三角部分；
- device 1 需要 device 0 全块和本地下三角；
- device 3 几乎需要所有先前 blocks。

工作量形成三角形：早期 rank 少，后期 rank 多。若所有 rank 仍同步走 $P$ 个 ring steps，早期设备会跳过大量未来 K/V 计算，却要等待最慢设备完成，造成负载不均衡。

基础实现可以根据全局 block index 整块跳过：

```python
if kv_block_is_strictly_future(q_block):
    skip_compute()
elif block_crosses_diagonal(q_block, kv_block):
    run_attention_with_causal_mask()
else:
    run_attention_without_elementwise_mask()
```

但跳过只减少 FLOPs，不自动缩短同步关键路径。后来出现的 zigzag、stripe 等 context-parallel 排列，会把早晚 token 交错分到不同 rank，使每张卡包含相近数量的 causal work。它们可以看作对基础 Ring Attention 负载分布的进一步优化，不应倒写成原始环算法天然没有 causal imbalance。

## 位置编码必须使用全局位置

每张卡的本地 token index 从 0 到 $N/P-1$，但模型位置应对应全局范围：

```text
device p local index i -> global position p * N_local + i
```

绝对 position embedding、RoPE 和相对位置 bias 都必须基于正确的全局位置或全局相对距离。

以 RoPE 为例，本地 Q/K 在进入 ring 前通常已按各自全局 position 旋转。K block 流到其他设备时不能根据接收方本地 index 再旋转一次。若使用长上下文 RoPE scaling，所有 ranks 也必须共享相同配置。

常见错误是 attention 数值看似稳定，模型却在跨 shard 边界处性能突然下降；原因不是 ring 通信，而是 position 在每卡重新从 0 开始。

正确性测试应专门构造跨设备边界的依赖，而不仅比较随机输出均值。

## Blockwise Feedforward 解决什么

Transformer block 不只有 attention。MLP 中间维度通常是 hidden size 的数倍，长序列的 FFN activation 同样巨大。

由于 MLP 对 token 独立，sequence sharding 后每张卡可以只对本地 tokens 计算，不需要像 attention 一样交换全局序列。Blockwise Parallel Transformer 进一步把本地序列分块执行 FFN，并结合 rematerialization，降低峰值 activation memory。

因此完整长上下文执行可理解为：

```text
Norm / projection: local sequence shard
Attention: local Q + rotating global K/V blocks
Output projection: local sequence shard
MLP: blockwise over local tokens
```

Ring 描述的是 attention 的全局依赖通信；blockwise Transformer 则把 attention 和 FFN 的峰值显存一起控制。只实现环形 K/V 而让 MLP 仍一次物化巨大中间激活，整体上下文上限仍可能由 FFN 决定。

## Backward 需要让梯度也沿环传播

训练时每个 Q-K/V block interaction 都产生梯度贡献。Forward 中 K/V block 沿环经过所有 Q owners；Backward 必须计算：

$$
dQ=dSK,
\quad dK=dS^TQ,
\quad dV=P^TdO
$$

本地 Q owner 可以累积来自各 K/V blocks 的 $dQ$。对某个流动的 K/V block，来自不同 Q owners 的 $dK,dV$ 则需要在它沿环移动时累积，最终回到原 owner。

概念时间线是：

```text
forward : K/V block visits every Q shard
backward: gradient contribution from every Q shard accumulates for that K/V block
```

实现还要恢复或重算 forward 的 Softmax statistics。像 FlashAttention backward 一样，可以保存每行 logsumexp，再按块重算 score/probability，避免存储全局 attention matrix。

Backward 的通信与计算量通常都高于 forward，buffer 生命周期也更复杂。评估 Ring Attention 训练性能不能只测 forward；否则可能忽略 gradient accumulation、atomic/reduction 与重计算的真实成本。

## 与 Tensor Parallel 怎样组合

Ring Attention 的 sequence/context parallelism 将 token 维分片；Tensor Parallel 将 head、hidden 或权重维分片。二者可以组成二维 device mesh。

例如 16 张 GPU：

```text
context parallel size = 4
tensor parallel size  = 4

logical mesh: CP(4) x TP(4)
```

每个 TP group 共同计算一份 sequence shard 上的模型分片；每个 CP ring 则让对应 head/hidden shard 的 K/V 在 4 个序列 ranks 间轮转。

组合时要明确：

- Q heads 与 KV heads 如何被 TP 分配；
- GQA 的 KV head 数是否能被 TP size 整除；
- ring P2P 与 TP all-reduce/all-gather 是否争用同一链路；
- rank layout 是否让高频 TP 通信留在 NVLink 域内；
- pipeline stage 内是否分别建立 CP ring；
- activation checkpoint 与 sequence shard 的边界。

不能简单地把 TP、CP 各自单独测得的效率相乘。两套通信可能在时间线上重叠，也可能互相阻塞。

## 与 Data Parallel、Pipeline Parallel 的关系

**Data Parallel** 复制模型、处理不同样本；它提高全局 batch 吞吐，不能让单条序列跨副本分布。

**Pipeline Parallel** 把层分到不同 stage；每个 stage 仍要处理该 micro-batch 的完整序列或其 CP shard。它解决模型深度/权重容量，不直接解决 attention 的全局 sequence 依赖。

**Context/Sequence Parallel** 把同一条序列的 token 分给多设备，Ring Attention 属于这一类。

三者可以同时存在：

$$
N_{GPU}=DP\times PP\times TP\times CP
$$

但并行维度越多，batch、head、layer、sequence 的整除约束和通信拓扑越复杂。规划时应先找真正的容量瓶颈：模型权重放不下优先 TP/PP，单序列激活放不下才需要 CP，吞吐副本不足再扩 DP。

## 训练与推理的适用方式不同

Ring Attention 最自然的场景是长序列训练或长 prompt prefill：所有 Q positions 都需要输出，$N^2/P$ 的本地计算足够大，容易覆盖 K/V 通信。

单 token decode 不同。每轮每条序列只有一个新 Q，却要读取完整历史 KV：

```text
Q length = 1
distributed KV length = N
```

若仍让 K/V blocks 围着环逐站传递，单步计算很小，通信难以隐藏，decode latency 还要跨多个 hops。更常见的长上下文 decode 方案可能让 query 广播到 KV shards，各卡计算局部 attention state，再做全局 online-softmax reduction；或者使用 context-parallel 专用 decode kernel。

因此“Ring Attention 支持 inference”不能简化成训练 forward 的环直接适合低延迟 decode。要分别分析：

- prompt prefill 的长 Q；
- chunked prefill 的中等 Q；
- decode 的单/少量 Q；
- KV Cache 是否已按 sequence 分片；
- 每 token collective latency。

同一个数学分解，在三个阶段的计算/通信比完全不同。

## “近乎无限上下文”有哪些前提

论文标题中的 near-infinite 强调系统容量可以随设备数扩展，并不意味着上下文长度没有代价。

### 计算仍是二次

设备数 $P$ 将每卡主要计算降到约 $O(N^2/P)$，若保持每卡本地序列长度不变并让 $N\propto P$，则每卡计算会随 $P$ 增长，而不是保持常数。

### 网络拓扑有上限

更多设备意味着更多 ring steps。跨节点带宽、hop latency、拥塞和故障概率会逐渐成为瓶颈。

### 模型未必会有效使用超长上下文

能执行 1M token 不等于模型已经学会稳定检索、组合和推理 1M token。训练数据、位置编码、curriculum 与评测同样重要。

### 输入和数据管道也要扩展

Tokenizer、样本打包、checkpoint、dataloader 与 host-to-device pipeline 都可能在极长样本下成为瓶颈。

### Dense Attention 不是所有场景的最佳选择

当任务允许近似、局部性或检索结构，稀疏 attention、RAG 或 recurrent state 可能用更低算力解决问题。Ring Attention 的价值是保留标准 dense attention 语义，而不是宣称 dense $N^2$ 永远最经济。

## 如何判断通信是否真的被覆盖

不要只看端到端吞吐后推断 overlap。需要从 profiler 时间线验证：

```text
step k compute      [================]
step k+1 recv          [--------]

理想：recv 完全落在 compute 区间内
```

建议记录：

- 每个 ring step 的 attention kernel duration；
- P2P send/recv duration 与实际带宽；
- compute stream 等待 communication event 的时间；
- communication stream 等待 buffer release 的时间；
- HBM bandwidth、Tensor Core utilization；
- 不同 rank 的 step duration 方差；
- causal 场景的有效/跳过 block 数；
- TP collective 与 ring P2P 的时间重叠。

若时间线出现：

```text
compute -> gap -> compute -> gap
```

优先判断 gap 是网络未完成、rank skew、buffer hazard 还是 host 发射不及时。盲目增大 FlashAttention tile 可能无法解决通信链路问题。

## 一个容量与时间的估算示例

假设全局序列 128K，使用 8 张 GPU 做 CP，每卡持有 16K token。仅从序列线性激活看，每卡容量约降为单卡完整序列的 $1/8$。

每层 attention 需要 8 个 device-block interactions：

```text
1 local K/V block + 7 remote K/V blocks
```

若每个 block attention 计算 3 ms、传输下一 K/V block 2 ms，理想 step 是：

$$
\max(3,2)=3\text{ ms}
$$

8 轮约 24 ms，再加启动、首尾流水和同步成本。若跨节点后传输变成 5 ms：

$$
\max(3,5)=5\text{ ms}
$$

总时间就更接近 40 ms。算法没有变化，拓扑改变已让关键路径从计算受限转为通信受限。

这是示意数字，不是特定硬件 benchmark。它说明部署前应先测目标 block size 的 local kernel 时间和真实 P2P 带宽，再判断 overlap 条件是否成立。

## 正确性测试应覆盖哪些边界

一个分布式 attention 结果接近 reference 只是起点。测试矩阵应包含：

### 分片边界

- $N$ 能被 $P$ 整除与不能整除；
- 尾 rank padding 与有效长度 mask；
- Q/KV block 不同大小；
- sequence packing 中多个文档不能互相 attention。

### Mask 与位置

- non-causal；
- causal 对角块和整块跳过；
- sliding-window / segment mask；
- 全局 RoPE position 与 scaling；
- 跨 rank 边界的相对位置。

### 数值

- FP32 reference 对 FP16/BF16；
- 极端 score 下 online Softmax 稳定性；
- 不同 ring order 的浮点归约差异；
- forward output 与 $dQ,dK,dV$；
- dropout RNG 在分片后的确定性。

### 分布式失败

- 某 rank 超时或提前退出；
- send/recv 顺序不一致造成 deadlock；
- world size / mesh 配置错误；
- checkpoint 恢复后 shard mapping 改变；
- 多 ring 与 TP collective 顺序不一致。

特别要用小 shape 构造可在单卡 reference 上完整计算的输入，逐元素比较分布式输出和梯度。直接用 1M token 做正确性测试，reference 本身就无法运行，难以定位错误。

## 性能实验应怎样组织

至少分别扫描以下变量：

| 维度 | 目的 |
| --- | --- |
| 全局序列长度 $N$ | 观察二次计算与容量扩展 |
| CP size $P$ | 观察 weak/strong scaling |
| 本地 block length $N/P$ | 判断通信能否覆盖 |
| 节点内/跨节点 | 分离 NVLink 与网络影响 |
| causal/non-causal | 暴露负载不均衡 |
| forward/backward | 包含梯度环流和重计算 |
| 单独 CP / CP+TP | 观察通信竞争 |

**Strong scaling** 固定全局 $N$，增加设备，期望单卡计算下降；但 block 变小后通信更难隐藏，效率通常不会线性。

**Weak scaling** 固定每卡本地 token 数，随设备增加全局 $N$；容量近似线性增长，但 dense attention 让每卡需要处理更多远端 blocks，总计算仍增加。

结果应报告：

- tokens/s 或 samples/s；
- 每层 attention forward/backward 时间；
- MFU/HFU 及 FLOPs 计算口径；
- 峰值显存；
- 通信暴露时间；
- 各 rank 最大/最小时间；
- 模型、dtype、mesh、interconnect 与软件版本。

只展示“最大跑到多少 token”只能证明容量，不能证明效率。

## 与其他 Sequence Parallel 方法的区别

“Sequence Parallel” 在不同框架中可能指不同机制。

Megatron 风格的 sequence parallel 常把 LayerNorm、Dropout 等 tensor-parallel 区域外的激活沿 sequence 分片，用 reduce-scatter/all-gather 配合 TP，主要减少重复 activation。

Ring Attention 则让 attention 本身的全局 K/V blocks 沿环移动，使单条 sequence 的 dense attention 跨设备完成。它通常更接近今天所说的 context parallelism。

还有 All-to-All 型方法会重排 sequence/head 维，让每张卡拿到完整 head 的部分 token，再运行本地 attention。它们与 ring 的通信 primitive、拓扑敏感性和 causal 负载平衡不同。

选型时不要只比较名称，要画出：

```text
每张卡开始持有什么 tensor shard
attention 前发生什么 collective/P2P
本地 kernel 看到什么 shape
attention 后怎样还原布局
```

数据布局图比“支持 CP”四个字更能说明真实性能。

## 从 Ring Attention 提炼出的系统方法

Ring Attention 的思想可以从 attention 推广到其他分布式块计算。

### 让大状态流动，小状态留在 owner

Q 与输出累积留在本地，K/V blocks 流动。这样每个输出有稳定 owner，减少最终结果重分布。

### 用可结合的统计量代替完整中间结果

Online Softmax 的 $(m,\ell,\widetilde O)$ 使不同块可精确合并。若一个算法能找到小规模、可结合的 sufficient statistics，就更容易做流式和分布式分块。

### 用算术强度覆盖通信

不是所有通信都需要消除。若当前块计算足够重，异步预取可以让传输离开关键路径。设计 block size 时应同时看 FLOPs 与 bytes。

### 让逻辑拓扑贴近物理拓扑

环只与相邻 rank 通信，但逻辑相邻若在物理网络上很远，优势会丢失。Rank placement 和 collective topology 属于算法实现的一部分。

### 分开容量可扩展与时间可扩展

能用 8 卡放下 8 倍序列，是容量扩展；是否用接近相同时间完成，是性能扩展。二次 attention 下二者本来就不会自动同时成立。

## 小结

Ring Attention 将单 GPU 内的 blockwise attention 扩展到多设备：序列沿设备分片，本地 Q 保持不动，K/V blocks 沿环逐站流动；每收到一块就调用本地精确 attention，并通过 online Softmax 统计合并全局结果。

它成立依赖三项关键条件：

1. 每张卡只保存本地 sequence shard 和少量通信 buffers，线性激活容量随设备数扩展；
2. blockwise attention 计算足够长，下一块 K/V 的传输能用双缓冲隐藏；
3. mask、全局 position、Softmax 统计和 backward 梯度在跨设备边界上保持一致。

Ring Attention 保留 dense attention 语义，但不消除 $N^2$ 计算；causal 场景还会产生早晚 rank 工作不均衡，decode 的短 Q 也未必适合直接沿用训练环。它最适合说明一条重要边界：单卡 kernel 已经足够高效之后，长上下文的下一个瓶颈会转向 sequence placement、网络拓扑和通信/计算流水。

下一篇会转向 DeepSeek-V2 的 Multi-head Latent Attention：它不再沿设备分摊完整 K/V，而是改变每个 token 被缓存的表示，用低秩 latent 与矩阵吸收压缩 Decode 必须反复读取的数据。

## 参考资料

- [Ring Attention with Blockwise Transformers for Near-Infinite Context](https://arxiv.org/abs/2310.01889)
- [RingAttention at ICLR 2024](https://openreview.net/forum?id=WsRHpHH4s0)
- [Official RingAttention implementation](https://github.com/haoliuhl/ringattention)
- [Blockwise Parallel Transformer for Large Context Models](https://arxiv.org/abs/2305.19370)
- [World Model on Million-Length Video and Language with Blockwise RingAttention](https://arxiv.org/abs/2402.08268)
- [Large World Model project](https://largeworldmodel.github.io/lwm/)
