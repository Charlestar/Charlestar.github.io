---
layout: post
title: "Tensor Parallel：怎样把一个 Transformer Layer 切到多张 GPU"
subtitle: "从 Column/Row Parallel MLP 到 Attention、Sequence Parallel 与二维设备网格"
date: 2026-08-09 18:10:00 +0800
last_modified_at: 2026-08-09
author: iStar
catalog: true
series: distributed-training
series_order: 20
technology_year: 2019
mathjax: true
tags: [AI Infra, Tensor Parallel, Megatron-LM, Distributed Training, NCCL]
---

Data Parallel 把 batch 分给不同 GPU，每张卡仍执行完整模型；ZeRO/FSDP 再把模型状态沿 Data Parallel ranks 分片，但一个 layer 真正计算时仍需要临时聚合其完整参数。若单个 Transformer layer 太大，或一次矩阵乘本身就希望由多张 GPU 共同完成，就需要切开 layer 内部的 tensors。

Megatron-LM 推广的 Tensor Parallel（TP）没有把一个 GEMM 随意切成很多小块再到处同步，而是利用 Transformer 的结构，把第一层线性变换按 output features 切开，让中间非线性在各 rank 本地完成；第二层再按 input features 切开，只在 block 输出处求和。

一对线性层的基本数据流是：

```text
replicated input X
  → Column Parallel Linear
  → sharded hidden features
  → local nonlinearity
  → Row Parallel Linear
  → partial outputs
  → AllReduce
  → replicated block output
```

同一结构还能映射到 Attention：Q/K/V projection 按 heads 切开，各 rank 本地计算一部分 heads，output projection 再合并。Tensor Parallel 的关键不是记住 layer 类名，而是始终知道每个 tensor 此刻是 replicated、按 feature sharded，还是只包含尚未求和的 partial result。

## 从一个两层 MLP 开始

忽略 bias，Transformer MLP 可写成：

$$
H=\phi(XA)
$$

$$
Y=HB
$$

其中：

$$
X\in\mathbb{R}^{M\times d}
$$

$$
A\in\mathbb{R}^{d\times f},
\qquad
B\in\mathbb{R}^{f\times d}
$$

$M$ 合并了 batch 与 sequence 维，$d$ 是 hidden size，$f$ 是扩展后的 FFN dimension，$\phi$ 是 GeLU、SiLU 或 gated activation 的一部分。

若用 $p$ 张 GPU，最直观但低效的切法是把第一个 GEMM 的 reduction 维切开：每个 rank 只能得到 partial $XA$，必须先 AllReduce 才能应用 nonlinear $\phi$。非线性不满足：

$$
\phi(U+V)=\phi(U)+\phi(V)
$$

因此 collective 被迫放在两层之间，随后第二层还可能再通信。

Megatron 的切法把 collective 推迟到整个 MLP 末尾。

## Column Parallel：切第一层的 Output Features

按数学记号把 $A$ 沿列切成 $p$ 份：

$$
A=[A_1,A_2,\dots,A_p]
$$

每个 rank 持有：

$$
A_r\in\mathbb{R}^{d\times f/p}
$$

输入 $X$ 在 TP group 内复制，各 rank 独立计算：

$$
H_r=\phi(XA_r)
$$

拼接这些结果正好是完整 hidden：

$$
H=[H_1,H_2,\dots,H_p]
$$

因为每个 output feature 完全由对应的 $A_r$ 产生，非线性也逐元素作用，rank 之间不必在 $\phi$ 前通信。

这称为 Column Parallel Linear。若框架把 Linear weight 存成 `[out_features, in_features]`，物理 tensor 可能沿第 0 维切；“Column”描述的是上面 $XA$ 的数学方向，不应只靠数组维号判断。

Column-parallel forward 的状态是：

```text
input:  replicated across TP ranks
weight: output-feature sharded
output: output-feature sharded
```

## Row Parallel：切第二层的 Input Features

第二层输入 $H$ 已沿 feature dimension 分为 $H_r$。对应地把 $B$ 沿行切开：

$$
B=
\begin{bmatrix}
B_1\\
B_2\\
\vdots\\
B_p
\end{bmatrix},
\qquad
B_r\in\mathbb{R}^{f/p\times d}
$$

完整输出为：

$$
Y=HB
=
\sum_{r=1}^{p}H_rB_r
$$

每个 rank 先计算 partial output：

$$
Y_r=H_rB_r
$$

然后对 $Y_r$ 做 AllReduce sum：

$$
Y=\operatorname{AllReduceSum}(Y_r)
$$

最终每个 rank 获得相同的完整输出，可以与 replicated residual 相加，再进入下一子层。

Row-parallel forward 的状态是：

```text
input:  input-feature sharded
weight: input-feature sharded
output: partial replicated-shape tensor
        → AllReduce
        → valid replicated tensor
```

这里最危险的错误是把 `partial output` 当作正常 replicated output。它在每个 rank 上 shape 相同，却只包含求和的一部分；漏掉 collective 后代码仍能运行，loss 会悄悄偏离。

## 为什么这对切分只需一次 Forward Collective

把两层组合起来：

$$
Y
=
\sum_{r=1}^{p}
\phi(XA_r)B_r
$$

每个 rank 可以完整执行自己的：

$$
X\rightarrow XA_r\rightarrow\phi\rightarrow H_rB_r
$$

只有最终不同 rank 的 partial outputs 要相加。非线性被留在 shard 内，collective 落在 residual boundary 前。

这说明 Tensor Parallel 的设计原则：

```text
让无法跨 shard 分配的操作留在本地完整分量上
让线性可加的 partial results 尽可能晚地合并
```

如果 MLP 使用 SwiGLU，gate/up 两个 projections 通常一起按 output features 切分；每个 rank 本地做 `SiLU(gate_r) * up_r`，再把结果交给 row-parallel down projection。只要 gated operands 使用相同 shard ownership，仍不需要在激活中间 AllGather。

## Backward 中 Collective 出现在哪里

Row-parallel forward 用 AllReduce 合并输出。Backward 接收 replicated 的 $\partial L/\partial Y$，每个 rank 可据此计算自己的 $\partial L/\partial B_r$ 和 sharded $\partial L/\partial H_r$，不需要先合并 input-gradient shards。

Column-parallel backward 则会为 replicated input $X$ 产生 partial input gradients：

$$
\frac{\partial L}{\partial X}
=
\sum_r
\frac{\partial L}{\partial (XA_r)}A_r^T
$$

每个 rank 只有求和的一项，因此要 AllReduce，才能把有效的 $\partial L/\partial X$ 传给更早的 replicated layer。

于是一个 column→row block 的典型对称关系是：

```text
forward:
  column local → row local → AllReduce output

backward:
  row local → column local → AllReduce input gradient
```

实现可能用 AllReduce、ReduceScatter/AllGather 对或融合通信表达相同语义，但 tensor 的 logical placement 必须一致。

## Attention 怎样复用同一结构

标准 multi-head attention 先做 Q/K/V projection：

$$
[Q,K,V]=XW^{QKV}
$$

再按 heads 切分，独立计算：

$$
O_i=\operatorname{Attention}(Q_i,K_i,V_i)
$$

最后拼接 heads 并通过 output projection：

$$
Y=[O_1;O_2;\dots;O_{n_h}]W^O
$$

QKV projection 很适合 Column Parallel：每个 TP rank 持有一部分 heads 的 projection columns，本地生成这些 heads 的 Q/K/V。Attention Softmax 在一个 head 内完成，只要完整 head 不跨 rank，就无需为 score 或 probability 做 TP collective。

Output projection 对应 Row Parallel：每个 rank 持有与本地 heads 对应的 $W^O$ rows，生成 partial hidden output，最后 AllReduce sum。

数据流为：

```text
replicated X
  → column-parallel QKV
  → local heads Q_r, K_r, V_r
  → local attention
  → row-parallel output projection
  → AllReduce
  → replicated attention output
```

这让 MLP 与 Attention 都在 residual boundary 恢复 replicated state，一个 Transformer block 的并行语义清晰且可组合。

## Head 数量为何限制 TP Size

按 heads 切分要求 Query heads 通常能被 TP size 整除：

$$
n_h\bmod p=0
$$

GQA/MLA 还要考虑 KV heads 或 latent ownership。若 $n_{kv}<p$：

- 某些 ranks 可能没有独立 KV head；
- runtime 可以复制 KV heads 给多个 ranks；
- 也可以切 head dimension，但 attention kernel 与通信更复杂；
- checkpoint mapping 必须记录 replicated 与 sharded 部分。

所以不能因为 hidden size 能被 8 整除，就假设 TP=8 对所有 attention 架构都合法。至少检查 Q heads、KV heads、FFN dimension、vocab padding、quantization group 和 kernel tile alignment。

## Embedding 怎样按 Vocabulary 切分

大词表 embedding matrix：

$$
E\in\mathbb{R}^{V\times d}
$$

可以沿 vocabulary rows 切到 TP ranks。Rank $r$ 只保存自己的 token id 区间 $[v_r,v_{r+1})$。对输入 ids：

1. 判断每个 id 是否属于本 rank；
2. 属于则做本地 lookup，否则输出 0；
3. AllReduce sum 合并 embeddings。

因为每个 token id 只属于一个 shard，求和后得到正确向量。

若后续直接使用 sequence-parallel layout，也可以选择 ReduceScatter 等不同输出 placement。关键仍是调用方必须知道结果是 replicated 还是按 sequence sharded。

Weight tying 使 input embedding 与 LM head 共用权重时，两个方向要共享同一 vocab partition，checkpoint 也不能把它们当作两份独立参数。

## Vocab-parallel Cross Entropy 不需要聚合完整 Logits

输出 logits 为：

$$
Z=X E^T\in\mathbb{R}^{M\times V}
$$

若 vocabulary 按 TP 切分，每个 rank 只计算：

$$
Z_r\in\mathbb{R}^{M\times V/p}
$$

直接 AllGather 完整 logits 会产生很大 activation。分布式 Softmax/Cross Entropy 可以只做少量 reductions：

1. 各 rank 计算 local max，再 AllReduce max 得到 global max；
2. 各 rank 计算稳定的 local exp sum，再 AllReduce sum 得到 global denominator；
3. target token 所在 rank 取 target logit，其余 ranks 给 0，再 AllReduce sum；
4. 用 global denominator 与 target logit计算 loss。

公式上：

$$
m=\max_r\max_{j\in\mathcal{V}_r}z_j
$$

$$
D=\sum_r\sum_{j\in\mathcal{V}_r}e^{z_j-m}
$$

$$
\ell=-z_y+m+\log D
$$

这样完整 vocab logits 从未在单 rank materialize。Backward 也能直接生成 local logits gradient shard。

## Tensor Parallel 能节省哪些显存

理想情况下，适合切分的 layer parameters 与其 gradients/optimizer states 可近似除以 $p$：

$$
M_{parameters,TP}\approx\frac{M_{parameters}}{p}
$$

但 activation 不会全部自动除以 $p$：

- Column-parallel outputs 沿 feature sharded；
- Row-parallel collective 后输出通常 replicated；
- LayerNorm、dropout、residual inputs 常 replicated；
- attention scores 按 heads sharded，但每个 head 仍覆盖完整 sequence；
- communication buffers 与临时 partial outputs 仍占空间。

因此原始 Megatron TP 仍有大量 replicated activations。Sequence Parallel 正是为这些 residual/LN/dropout 区域进一步分片。

## Sequence Parallel 把 Replicated Activation 沿 Token 切开

Megatron Sequence Parallel（SP）复用 TP process group，把原本在每个 TP rank 都完整存在的 `[sequence, batch, hidden]` activation 沿 sequence 维切分。

它与完整 Context Parallel 不同：

- SP 主要分片 LayerNorm、dropout、residual 等不跨 token 的区域；
- attention/TP linear 计算前仍会按需要 AllGather sequence；
- CP 则让整网 activations 沿 sequence 分片，并为 attention 交换跨 rank KV/context。

一个简化的 SP+TP block 边界是：

```text
sequence-sharded residual
  → local LayerNorm/dropout work
  → AllGather sequence for column-parallel linear
  → feature-sharded internal activation
  → row-parallel linear partial output
  → ReduceScatter along sequence
  → sequence-sharded residual
```

原本 row-parallel 的 AllReduce 可以分解为 ReduceScatter + AllGather。SP 把 AllGather 放到下一次真正需要 replicated input 之前，期间让 activation 保持 sequence shard。通信量级与 AllReduce 相近，却减少了 replicated activation 驻留。

Backward 的 mappings 对称反转。若 forward placement 注释不清，最容易出现重复 AllGather、错误 residual add 或同一个 token 在多个 ranks 上重复进入 MoE dispatch。

## Sequence Parallel 与 Context Parallel 不可混用概念

两者都切 sequence，却处理不同范围：

| 特性 | Sequence Parallel | Context Parallel |
| --- | --- | --- |
| 常用 group | TP group | 独立 CP group |
| 分片范围 | TP 区域外的部分 activations | 网络输入与几乎全部 activations |
| Attention | 通常聚合后按 TP heads 计算 | 必须跨 CP ranks 交换 K/V 或 attention statistics |
| 主要目标 | 去掉 TP ranks 上的 activation replication | 支持更长 sequence 并降低全网 activation |

SP 不会把单个 attention head 的 sequence 自动切到多卡；CP/Ring Attention 才处理跨设备完整上下文。配置文件里同时出现 TP、SP、CP 时，需要分别画出 process groups 和 tensor placements。

## TP 与 FSDP 怎样组成二维 Sharding

假设总共 16 张 GPU，TP size 为 4，则有 4 份 Tensor-Parallel model replicas。可以再让这 4 份 replicas 组成 DP/FSDP 维度：

```text
TP groups (intra-layer):
  [0,1,2,3]
  [4,5,6,7]
  [8,9,10,11]
  [12,13,14,15]

DP groups (same TP shard across replicas):
  [0,4,8,12]
  [1,5,9,13]
  [2,6,10,14]
  [3,7,11,15]
```

Rank 0、4、8、12 拥有同一个 logical TP shard 的不同 data replicas。FSDP/ZeRO 应在这组 ranks 之间切 optimizer/gradient/parameter state，而不是把不同 TP shards 混在同一个 DP collective 中。

总 world size 在只使用 TP 与 DP 时为：

$$
W=p_{TP}\times p_{DP}
$$

加入 PP、CP、EP 后会形成更多正交或嵌套维度。不能只根据 global rank 邻近关系猜 group；应由显式 device mesh 生成并记录。

## 为什么 TP 更偏好留在单节点高速互联

Data Parallel 梯度通常每 optimizer step 同步，并能用较大 buckets 与 backward 重叠。Tensor Parallel collectives 则出现在每个 Transformer layer 的 forward 与 backward critical path：

```text
layer 0 TP collective
layer 1 TP collective
...
layer L TP collective
```

单次 collective 的 activation bytes 可能小于完整 gradients，但频率高、依赖紧，latency 很难完全隐藏。因此 TP group 通常优先放在 NVLink/NVSwitch 等低延迟高带宽域内，把 DP/ZeRO 或 PP 连接扩展到节点间。

这不是绝对规则：超大 hidden、先进网络与通信重叠可以支持跨节点 TP。但 placement 必须用目标 topology profile 验证，不能只看理论 NIC aggregate bandwidth。

## TP Size 增大为何会出现递减收益

每 rank 的 GEMM 维度随 $p$ 变小：

$$
f_{local}=\frac{f}{p}
$$

参数和 FLOPs 下降，但也会发生：

- Tensor Core tile 利用率降低；
- kernel launch 与 epilogue 固定成本占比上升；
- AllReduce latency/bytes 相对 compute 增大；
- 每 rank heads 太少，attention 并行度不足；
- quantization group、MoE expert、vocab shard 对齐恶化；
- 更多 GPU 被绑定为一个 failure domain，DP replicas 减少。

TP 的目标不是让每个 layer 用尽可能多 GPU，而是找到“单 rank 能放下且 local GEMM 仍够大、collective 可承受”的最小合理 size。剩余 GPU 通常更适合增加 DP throughput 或 PP depth。

## 通信重叠需要 Independent Work

Row-parallel output若必须先 AllReduce 才能做 residual add，整个下一阶段都依赖 collective，难以完全隐藏。现代实现会尝试：

- 把 GEMM 拆成 chunks，边计算边 ReduceScatter；
- 将 weight-gradient GEMM 与 input-gradient collective 重叠；
- 使用独立 communication stream；
- 融合 bias/residual/dropout；
- 调整 SM 分配，避免 NCCL 与 GEMM 完全争抢资源。

但“异步发起”不等于“被隐藏”。如果下一条 kernel 立即 wait，或 NCCL 占用过多 SM/带宽导致 GEMM 变慢，timeline 仍没有净收益。

Profiler 应标出每个 layer 的 GEMM、collective issue、wait 和 dependent consumer，并比较重叠前后的 critical path，而不是只看 NCCL bar 与 GEMM bar 是否在时间轴上有交叠。

## Bias、Dropout 与 RNG 也有 Placement

Column-parallel bias 可以随 output features sharded，本地直接加入。Row-parallel bias属于完整 output，若每个 rank 都在 partial output 上先加一次，AllReduce 后会被重复 $p$ 次；应在求和之后加，或按明确缩放/fusion 处理。

Dropout mask 影响数值可复现：

- feature-sharded activation 的各 shards应使用不重叠、可重建的 RNG 子序列；
- replicated activation 若期望 ranks 一致，mask 也要一致；
- sequence-sharded token ranges 应按 global position 确定 RNG mapping；
- checkpoint 恢复后 RNG tracker 必须回到相同状态。

“每张卡 seed 相同”不足以保证正确；不同 ranks 消耗随机数的 tensor shape 与顺序可能不同。

## Gradient Norm 与 Clipping 必须先还原全局量

每个 TP rank 只有 parameter/gradient shards。若直接计算 local norm 并分别 clip，不同 ranks 会使用不同系数。

对 L2 norm：

$$
\|g\|_2^2
=
\sum_r\|g_r\|_2^2
$$

各 rank 先计算 local squared sum，再沿所有互不重复的 parameter shards 所在 groups AllReduce sum，最后开方得到 global norm。若某些 parameters 在 TP group 内 replicated，必须只计一次；若再叠加 DP，DP replicas 也不能重复计数。

同理，NaN/Inf detection、GradScaler overflow 和 optimizer skip 决策要在相关 ranks 上一致，否则一次 step 后 shards 不再属于同一模型。

## Checkpoint 必须记录 Tensor 的 Logical Placement

对一个 column-parallel weight，checkpoint shards 沿 output dimension；row-parallel weight 沿 input dimension；vocab embedding 沿 vocabulary；LayerNorm 可能 replicated。

Sharded checkpoint metadata 至少要描述：

```yaml
tensor_name: transformer.layers.0.mlp.fc1.weight
global_shape: [ffn_hidden, hidden]
placement:
  tensor_parallel: shard(dim=0)
  data_parallel: replicate
dtype: bf16
padding: <logical vs physical size>
mesh:
  tp: <size>
  dp: <size>
```

从 TP=8 恢复到 TP=4 需要 reshard global tensor：先根据 metadata 理解 logical slices，再组合/重切。简单按文件编号拼接可能在 row/column、QKV interleave、gated MLP 或 vocab padding 上出错。

Optimizer moments 与 parameters 使用相同 logical sharding，ZeRO/FSDP 又会在 DP 维进一步切它们。Checkpoint 工具必须同时理解两维 placement。

## 用一个小 Reference 验证切分

实现 TP 时可以先在单进程中模拟 shards：

```python
# 数学示意：A 沿输出维切，B 沿输入维切
A_shards = split(A, dim=1, parts=tp)
B_shards = split(B, dim=0, parts=tp)

partials = []
for A_r, B_r in zip(A_shards, B_shards):
    H_r = activation(X @ A_r)
    partials.append(H_r @ B_r)

Y_tp = sum(partials)
Y_ref = activation(X @ A) @ B
assert_close(Y_tp, Y_ref)
```

然后逐步替换为多进程 collectives，并验证：

1. forward outputs；
2. input gradients；
3. parameter gradients 的对应 global slice；
4. optimizer update 后的 shards；
5. 保存/恢复后的下一步 loss。

使用无 dropout、固定小 tensor 和 FP64/FP32 reference 可以先排除数值噪声，再加入 mixed precision、fusion 和异步通信。

## 性能 Benchmark 应覆盖哪些 Shape

训练中的 $M$ 通常为 micro-batch × sequence length；推理 Prefill、Decode 的 $M$ 差异更大。TP kernel/collective 测试至少覆盖：

- QKV projection 的 $d\times( Q+K+V )$ shape；
- GQA/MLA 下不对称 head dimensions；
- gated MLP 的双分支 projection；
- row-parallel down/output projection；
- vocab-parallel logits；
- forward、activation-gradient、weight-gradient 三类 GEMM；
- sequence parallel 的 AllGather/ReduceScatter；
- 不同 micro-batch、sequence、gradient accumulation。

端到端报告同时记录：

- tokens/s 与 model FLOPs utilization；
- 每 rank peak memory；
- TP collective time 与 overlap ratio；
- local GEMM efficiency；
- TP group 的拓扑路径；
- load imbalance 与最慢 rank；
- loss/gradient parity。

只测一层大 GEMM 会高估真实收益，因为模型里还有 LayerNorm、RoPE、dropout、collectives、embedding 与 optimizer。

## 怎样选择 TP Size

可以按约束逐步缩小候选集：

1. 参数和单层 temporary 是否能放进单 rank；
2. Query/KV heads、FFN、vocab 与 quantization group 是否可合法切分；
3. local GEMM dimensions 是否仍能高效使用 Tensor Core；
4. TP group 是否落在足够快的互联域；
5. collective 是否能与 compute overlap；
6. 剩余 DP/PP replicas 是否足以达到 global throughput；
7. checkpoint 与目标 serving TP 是否容易转换。

不要只用训练时峰值显存选择。TP=8 即使节省更多参数，也可能因 local shapes 太小而比 TP=4 + DP=2 更慢。

## 一条可执行的集成路径

1. **固定单卡数学基线**：保存每个 block output、gradient 与 optimizer update；
2. **给 tensors 标注 placement**：replicated、feature-sharded、sequence-sharded 或 partial；
3. **先切 MLP**：验证 column→activation→row→sum；
4. **再切 Attention**：按 heads 切 QKV，检查 GQA/MLA ownership；
5. **加入 Vocab Parallel**：不聚合完整 logits，验证 distributed CE；
6. **处理 bias/RNG/shared weights**：覆盖容易静默出错的非 GEMM 状态；
7. **加入 Sequence Parallel**：用 AG/RS 替换 replicated activation 生命周期；
8. **建立 DP×TP mesh**：让 FSDP/ZeRO 只在相同 TP shard replicas 间工作；
9. **实现 sharded checkpoint**：测试 TP resize 与 optimizer reshard；
10. **按 topology placement**：优先把 TP 放进高速互联域；
11. **做 timeline profile**：确认 collective 顺序、wait 与真实 overlap；
12. **用 tokens/s、显存和收敛共同验收**：不能只看单层 TFLOPS。

## 常见误区

### “Tensor Parallel 就是把 Weight 平均切开”

切哪个轴决定输出是独立 shard 还是 partial sum，也决定 collective 放在哪里。随意切分会让 nonlinear 前被迫同步。

### “每个 Rank 上 Shape 一样，就是 Replicated Tensor”

Row-parallel partial outputs shape 相同但数值不完整；必须 AllReduce 后才能作为 replicated output。

### “TP Size 越大，训练一定越快”

更多 shards 会缩小 local GEMM 并增加 collective 相对成本，收益通常递减。

### “Sequence Parallel 等于 Context Parallel”

SP 主要去掉 TP 区域外的 activation replication；CP 让完整网络沿 sequence 分片，并专门处理 attention 的跨 token 依赖。

### “GQA 的 KV Heads 可以像 MHA 一样任意按 TP 切”

当 KV heads 少于 TP ranks 时需要复制、切 head dimension 或采用专门 layout，不能产生空 shard 后假装合法。

### “Checkpoint 文件按 Rank 编号拼起来就能改 TP Size”

不同参数的 shard axes、QKV interleave、gated layout、vocab padding 与 DP sharding都不同，必须依赖 logical metadata reshard。

## 小结

Megatron-style Tensor Parallel 的核心是一对互补矩阵切分：

1. Column Parallel 沿 output features 切第一层，让非线性在本地 shard 上完成；
2. Row Parallel 沿 input features 切第二层，在 block 末将 partial outputs 求和；
3. Attention 把 Q/K/V heads 交给不同 ranks，output projection 再合并；
4. Vocab Parallel 用分布式 max/sum/target reductions 避免完整 logits；
5. Forward 与 backward 的 collectives 在相反边界恢复 replicated gradient/output；
6. Sequence Parallel 用 AllGather/ReduceScatter 让 residual/LN activations 沿 token 分片；
7. TP 与 FSDP/DP 形成正交 mesh，前者切 layer，后者切 batch replicas 的状态；
8. 高频 TP collectives 应尽量留在低延迟高速互联域；
9. TP size 由显存、shape、heads、网络和 global parallel plan 共同决定；
10. 每个 tensor 的 logical placement 必须贯穿 forward、backward、optimizer 与 checkpoint。

Tensor Parallel 沿 layer 宽度切分计算。下一篇 Pipeline Parallel 会沿模型深度切层：micro-batches 怎样在 stages 间形成流水、为什么会有 bubble，以及 1F1B、interleaving 与 activation transfer 怎样改变吞吐、显存和恢复边界。

## 参考资料

- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)
- [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473)
- [NVIDIA Megatron Core: Parallelism Strategies Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)
- [NVIDIA Megatron Core: Tensor Parallel API](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.tensor_parallel.html)
- [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198)
- [NVIDIA Megatron Core: Context Parallel Package](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html)
