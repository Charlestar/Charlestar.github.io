---
layout: post
title: "FlashAttention-3：用异步流水榨出 Hopper 的 Attention 性能"
subtitle: "从 TMA、WGMMA、Warp Specialization 到更准确的 FP8 Attention"
date: 2026-06-23 09:00:00 +0800
last_modified_at: 2026-08-09
author: iStar
catalog: true
series: attention-long-context
series_order: 60
technology_year: 2024
mathjax: true
tags: [AI Infra, FlashAttention, Hopper, FP8]
---

FlashAttention-2 在 A100 上通过更好的 thread-block 并行与 warp 工作划分，把精确注意力推进到接近高性能 GEMM 的利用率。但把同一套执行方式搬到 H100，并不会自动吃到新硬件的峰值。

Hopper 改变了两个关键环节：

- TMA 可以异步把多维 tensor tile 从 HBM 搬到 shared memory，并由硬件处理地址计算和边界；
- WGMMA 让一个 warpgroup 异步发起更大的矩阵乘加，并可直接从 shared memory 读取操作数。

如果 kernel 仍然按“加载 K/V → 等待 → 算 $QK^T$ → 等待 → Softmax → 算 $PV$”串行执行，这些异步能力大部分都会被浪费。论文报告 FlashAttention-2 在 H100 上只达到约 35% 利用率，原因已经从单纯的 IO 和并行度问题，进一步转向**依赖链与流水调度**。

FlashAttention-3 为 Hopper 重新组织 attention kernel：producer warps 专门发起 TMA，consumer warpgroups 专门执行 WGMMA 和 Softmax；两个 consumer groups 交替让 Tensor Core 与特殊函数单元同时工作；单个 group 内又跨 KV block 打破部分串行依赖。最后，它利用 FP8 Tensor Core，并通过 block quantization 和 incoherent processing 控制低精度误差。

本文从 H100 的执行模型出发，逐层还原这三种异步重叠，再解释 FP8 路径为什么不仅是把输入 `cast` 成 8 bit。

## FA2 到 H100 后，瓶颈为什么又变了

FA2 已经保留了 FlashAttention 的核心数据流：

1. Q tile 留在片上；
2. 逐块读取 K/V；
3. 计算局部 $S=QK^T$；
4. 用 online Softmax 更新行最大值、指数和与输出；
5. 不在 HBM 中物化完整 $N\times N$ 矩阵。

它还沿 query sequence 分配更多 thread blocks，并在 block 内采用 split-Q，让各 warp 拥有独立输出行，减少 shared-memory 归约。

这些优化在 Ampere 上有效，但 H100 的 Tensor Core 吞吐和异步能力增长得更快。论文以 H100 SXM5 为例给出：

- FP16 matrix multiply 理论吞吐约 989 TFLOPS；
- exp 等特殊函数吞吐约 3.9 TFLOPS。

对 head dimension 128 的 forward，矩阵乘 FLOPs 大约是 exp 操作数的 512 倍，但矩阵乘硬件吞吐又约是特殊函数的 256 倍。因此 Softmax 中数量很少的 exp，仍可能花掉相当于矩阵乘一半的周期。

这揭示了一个新矛盾：

```text
Tensor Core 越快
-> Softmax 占比越显眼
-> 串行等待 Softmax 的代价越大
```

优化不能只继续减少 HBM IO，还要让 Tensor Core、特殊函数单元和内存搬运引擎在同一时间分别做事。

## 先认识 Hopper 的三个硬件能力

### TMA：把搬运从线程指令中剥离

传统 global-to-shared copy 往往需要一组 threads：

- 计算每个元素的地址；
- 判断 tile 是否越界；
- 从 global memory load 到寄存器；
- 再 store 到 shared memory。

这些线程和地址值都会消耗指令槽与寄存器。TMA（Tensor Memory Accelerator）接收 tensor descriptor 和 tile 坐标后，可以异步执行多维 HBM ↔ shared-memory 搬运，并处理常见边界条件。

发起 TMA 后，warp 不必逐元素参与复制，也不必原地等待数据到达。只要用 barrier 表达“这块 buffer 已填满”，consumer 就能在正确时刻读取。

### WGMMA：Warpgroup 级异步矩阵乘

Ampere 常用的 `mma.sync` 由 warp 协作并带有同步语义。Hopper 的 WGMMA（warpgroup matrix multiply-accumulate）让连续 4 个 warps，即 128 threads，协同发起更大的 Tensor Core 工作。

WGMMA 可以异步执行，部分输入直接来自 shared memory。Consumer 发出一组 WGMMA 指令后，可以在等待结果提交之前安排其他独立指令。

这里的“异步”并不表示可以无视依赖。读取 accumulator 前仍要执行相应 wait；优势是 wait 之前有机会插入 Softmax、地址准备或另一批矩阵乘。

### FP8 Tensor Core：吞吐翻倍，但表示范围更紧

Hopper 的 FP8 WGMMA 理论吞吐约为 FP16/BF16 的两倍。论文关注 E4M3 等格式：指数和尾数位数更少，量化误差与离群值风险更高。

如果 Q/K/V 只使用整 tensor 的一个 scale，少量大幅值会把 scale 拉大，使大量普通值落入很粗的量化间隔。吞吐提升很诱人，但 attention score 对 Q/K 误差又很敏感，不能只看 kernel 是否执行了 FP8 指令。

FA3 的三条主线分别对应这些能力：用 TMA/WGMMA 做 producer-consumer 流水，用异步 WGMMA 隐藏 Softmax，再为 FP8 设计 block scale 与离群值扩散。

## Warp、Warpgroup 和 CTA 各自扮演什么角色

理解 FA3，需要把 Hopper 的执行层级区分开：

```text
thread       : 单个执行线程
warp         : 32 threads
warpgroup    : 4 warps = 128 threads，WGMMA 协作单位
CTA/block    : 多个 warps/warpgroups，共享同一块 shared memory
SM           : 调度 CTA、执行 Tensor Core/SFU/TMA 协作
```

FA3 的一个 CTA 仍负责某个 Q row tile 对多个 K/V tiles 的 attention。不同 CTA 沿 batch、head 和 query sequence 并行，这部分继承 FA2。

真正变化发生在 CTA 内。Warps 不再都执行相同的“load + matmul + Softmax”程序，而是拥有长期稳定的角色：

```text
producer warpgroup  -> 只负责发起 Q/K/V 的 TMA 与管理 buffer 状态
consumer warpgroup  -> 只负责 WGMMA、Softmax、输出累积
consumer warpgroup  -> 与另一组交替处理不同 Q tile/工作阶段
```

这种按角色分工称为 warp specialization。它不是简单增加 warps，而是让编译器和硬件更容易为不同职责安排指令与寄存器。

## 第一层重叠：Producer 与 Consumer 并行

没有流水时，一个 KV tile 的时间线类似：

```text
load K/V j -> wait -> QK j -> wait -> softmax j -> PV j -> next
```

TMA 与 warp specialization 让时间线变成：

```text
producer:  load K/V 0 | load K/V 1 | load K/V 2 | ...
consumer:             compute 0    | compute 1    | compute 2
```

Producer 用 TMA 把未来 K/V tiles 预取到 shared-memory circular buffer；consumer 处理当前已就绪 stage。若计算比单次搬运更长，后续 tile 在 consumer 需要前已经到达，HBM latency 被隐藏。

这与普通双缓冲相似，但 FA3 使用 $s$ 个 stage 的环形 buffer：

```text
stage index = j mod s
```

每个 stage 在两个状态之间循环：

1. consumer 已用完，可以被 producer 覆盖；
2. TMA 已填充，可以被 consumer 读取。

Producer 在写入前等待“empty/consumed” barrier，TMA 完成后提交“full/ready”；consumer 反向执行同样的握手。只要任一方漏掉 barrier phase 或过早复用 stage，就可能读取旧 tile 或覆盖仍在使用的数据。

## 为什么专门的 Producer 不会浪费线程

乍看让一组 warp 只负责搬运，好像减少了参与计算的线程。TMA 改变了这笔账：发起 tensor copy 只需要很少指令，producer 不必为每个元素保留地址和数据寄存器。

Hopper 还支持 `setmaxnreg` 动态调整 warpgroup 的寄存器配额。Producer 可以让出大部分 registers，consumer 得到更多寄存器来保存：

- Q fragments；
- WGMMA accumulators；
- row max 和 row sum；
- output accumulator；
- 多 stage pipeline 的中间 score/probability。

因此分工同时改善了两件事：

- 指令调度更单纯，搬运与计算并行；
- 寄存器从低需求 producer 转移到高需求 consumer。

若在旧架构上只有软件 copy、不能动态合理分配寄存器，完全照搬相同分工未必有收益。Warp specialization 是硬件能力、buffer protocol 与资源布局共同构成的设计。

## Circular Buffer 为什么不能无限加深

更多 stages 可以覆盖更长的 HBM latency，却会增加 shared-memory 占用：

$$
S_{buffer}
\approx s\cdot (B_c\cdot d_K+B_c\cdot d_V)\cdot bytes
$$

还要加上 Q、barrier metadata 与可能的中间 tile。每个 CTA 使用的 shared memory 越多，同一 SM 可驻留的 CTAs 越少；超过硬件上限则 kernel 无法启动。

Pipeline depth 的选择需要权衡：

- 太浅：TMA 未完成时 consumer 停顿；
- 太深：shared memory 增加，occupancy 降低；
- tile 太大：计算/搬运比高，但寄存器和 shared memory 压力大；
- tile 太小：更多调度与 barrier，WGMMA 形状也可能不理想。

因此 `stages=3` 或某个 `BLOCK_N` 不是跨 head dimension、dtype 和 GPU 通用的常数。高性能实现会为常见 shape 准备不同 kernel 配置或调度器。

## 第二层重叠：两个 Consumer Warpgroups Ping-Pong

Producer-consumer overlap 主要隐藏数据搬运，Softmax 仍可能让 Tensor Core 空闲。FA3 用两个 consumer warpgroups 交替安排 GEMM 与 Softmax。

设 attention 的两个矩阵乘为：

$$
GEMM_0: S=QK^T
$$

$$
GEMM_1: O\mathrel{+}=PV
$$

对同一 tile，Softmax 必须等待 $GEMM_0$ 产生 scores，$GEMM_1$ 又必须等待 Softmax 得到 probabilities。这条依赖无法凭空删除。

但不同 consumer warpgroup 处理的是不同工作。FA3 用 barrier 调整发射顺序，使：

```text
时间段 A:
  group 1 -> Softmax / rescale
  group 2 -> WGMMA(QK 或 PV)

时间段 B:
  group 1 -> WGMMA(QK 或 PV)
  group 2 -> Softmax / rescale
```

角色来回交换，所以称为 ping-pong scheduling。Softmax 的 exp 主要使用 multi-function unit，WGMMA 使用 Tensor Core；只要指令能同时驻留和发射，两类硬件可以并行工作。

论文指出实际时间线不会像示意图一样完美整齐，但在 head dimension 128、sequence 8192 的一组 forward 配置中，ping-pong 将约 570 TFLOPS 提升到约 620–640 TFLOPS。这个数字说明它是增量优化，不是全部 FA3 加速来源。

## Ping-Pong 和 Producer-Consumer 不是同一件事

两者都叫 overlap，隐藏的对象不同：

| 机制 | 参与角色 | 主要重叠 |
| --- | --- | --- |
| Warp specialization | Producer 与 consumers | TMA 数据搬运 vs 计算 |
| Ping-pong scheduling | 两个 consumer warpgroups | 一组 Softmax vs 另一组 WGMMA |

Producer 不负责 Softmax；ping-pong 也不负责预取 K/V。Profiler 中若只看到 memcpy/TMA 与 kernel 重叠，不能证明 Tensor Core 与 Softmax 已经交错。

FA3 的 ablation 也分别去掉这两类能力：完整配置、只有 warp specialization、只有 GEMM-Softmax pipeline 的性能不同，说明两个优化互补。

## 第三层重叠：单个 Warpgroup 内跨迭代 Pipeline

即使只有一个 consumer warpgroup，也可以利用 WGMMA 的异步性跨 KV blocks 重排部分工作。

设第 $j$ 个 KV block 的流程为：

```text
S_j = Q K_j^T
P_j = local_softmax_and_rescale(S_j)
O   = O + P_j V_j
```

同一 $j$ 内存在严格依赖。但对相邻 block，可以在寄存器中保存额外 state，形成两级流水：

```text
WGMMA:     QK_j -------- PV_j -------- QK_{j+1} -------- PV_{j+1}
scalar:             softmax_j ---------------- softmax_{j+1}
```

更准确地说，FA3 让某些下一迭代的异步 WGMMA 在当前迭代 Softmax 指令周围发射，并在真正读取结果时才 wait。它需要额外 register buffers 保存处于不同 pipeline stage 的 score/probability fragments。

重排必须遵守 online Softmax 的全局状态：新 block 若改变 row max，旧 output accumulator 需要正确 rescale。可以延迟某些更新，却不能让依赖未来 $m$ 的计算使用旧 scale。

这类优化的难点不是写出更多 async 指令，而是证明每次 wait、barrier、rescale 和 buffer reuse 的顺序仍对应精确 attention。

## 三层流水放在一起看

FA3 的 CTA 内可以抽象成：

```text
HBM
 |
 |  TMA: producer 预取未来 K/V
 v
SMEM circular buffer
 |
 |  WGMMA: consumer groups 异步发起 QK / PV
 v
register accumulators
 |
 |  Softmax: 与另一 group 或相邻迭代的 WGMMA 交错
 v
online-softmax output state
```

对应三种 latency hiding：

1. 当前计算隐藏下一 tile 的 HBM→SMEM；
2. 一个 group 的 Tensor Core 工作隐藏另一 group 的 exp/rescale；
3. 同一 group 用跨迭代指令重排缩短局部依赖气泡。

它们共享同一批有限资源：shared memory、registers、barriers、warp slots 和 instruction issue bandwidth。增加一个 pipeline stage 可能改善一层，却因寄存器压力降低整体 occupancy，所以最终配置必须结合硬件计数器与端到端时间选择。

## Online Softmax 仍是精确合并的核心

FA3 的调度更复杂，但数学结果没有改变。对每个 query row，处理新的 score tile $S_j$ 时维护：

$$
m_j=\max(m_{j-1},\operatorname{rowmax}(S_j))
$$

$$
\ell_j=e^{m_{j-1}-m_j}\ell_{j-1}
+\operatorname{rowsum}(e^{S_j-m_j})
$$

$$
\widetilde O_j=e^{m_{j-1}-m_j}\widetilde O_{j-1}
+e^{S_j-m_j}V_j
$$

最后：

$$
O=\widetilde O_T/\ell_T
$$

FP16/BF16 路径仍保留 FP32 的 Softmax rescaling 和累积关键状态，因此论文中的 FP16 FA3 与 FA2 具有相同量级数值误差，并优于把更多中间结果保存在低精度的朴素实现。

“执行顺序异步”与“数学顺序无约束”是两回事。WGMMA 可以晚完成，Softmax 可以和独立工作重叠，但行统计的逻辑合并顺序必须可追踪。

## FP8 路径的第一个问题：WGMMA Layout

低精度实现不只是把：

```python
q = q.to(torch.float8_e4m3fn)
```

插在函数前面。Hopper FP8 WGMMA 对 operand 在 shared memory 和 registers 中的 layout 有特定要求。Attention 的第一个 GEMM：

$$
QK^T
$$

和第二个 GEMM：

$$
PV
$$

对 P/V tile 的行列方向要求并不天然一致。若在 kernel 外显式 transpose V，会增加一次完整的 HBM 读写，可能吞掉 FP8 的收益。

FA3 在 kernel 内完成 V tile transpose，并利用寄存器排列与 shared-memory 写出布局，让 probability tile 的列置换和 V 的对应行置换相互匹配。只要两边应用一致置换，$PV$ 的数学结果不变。

这种优化说明低精度 Tensor Core 性能常被 layout conversion 决定。理论 WGMMA 吞吐翻倍，不代表端到端自动翻倍；转换若不能融合，额外 bytes 会成为新瓶颈。

## Block Quantization 为什么优于 Per-Tensor Scale

FP8 E4M3 的动态范围和有效精度有限。Per-tensor quantization 为整个 Q 选择一个 scale：

$$
s_Q=\frac{\max|Q|}{q_{max}}
$$

$$
Q_q=\operatorname{round}(Q/s_Q)
$$

若 Q 中只有少量极端 outliers，$s_Q$ 被它们主导，普通 block 的量化 step 过粗。

FA3 本来就按 $B_r\times d$ 或 $B_c\times d$ tiles 处理 Q/K/V，因此为每块保存独立 scale：

$$
s_{Q_i},s_{K_j},s_{V_j}
$$

计算 score block 时补回：

$$
S_{ij}
=\alpha(s_{Q_i}s_{K_j})(Q_{q,i}K_{q,j}^T)
$$

输出乘 V 时再考虑 $s_{V_j}$。Scale 数量从每 tensor 一个增加到每 block 一个，但与 $N\times N$ attention matrix 相比开销很小，局部动态范围却显著收紧。

Block scale 还能与前置的 RoPE/quantize kernel 融合。若 RoPE 本来受显存带宽限制，在同一次读写中完成 scale 统计与 FP8 输出，可能不增加额外关键路径。

## Incoherent Processing 如何摊平离群值

Block quantization 只能让 outlier 的影响局限在一个 block；若每个 block 的某些 channel 都有很大幅值，误差仍然明显。

FA3 对 Q 和 K 乘同一个随机正交矩阵 $M$：

$$
Q'=QM,\quad K'=KM
$$

因为：

$$
MM^T=I
$$

所以：

$$
Q'K'^T
=(QM)(KM)^T
=QMM^TK^T
=QK^T
$$

精确实数运算下 attention scores 完全不变。变化发生在量化前的数值分布：一个 channel 上的尖锐 outlier 被正交变换扩散到多个维度，最大绝对值下降，FP8 的有限 code points 分配得更均匀。

论文选择随机 ±1 对角矩阵与 Hadamard matrix 的组合，使变换可在：

$$
O(d\log d)
$$

完成，而不是普通 dense orthogonal matrix 的 $O(d^2)$。它也可以与 RoPE 等前置操作融合。

“Incoherent” 不是把 attention 随机化，也不是近似丢弃信息；正交变换成对作用于 Q/K，score 在量化前保持不变，目的是改变坐标系，让量化误差不被少数坐标支配。

## 为什么 V 不使用同样的 Q/K 成对抵消

Q/K 的正交变换之所以能抵消，是因为它们在点积中以：

$$
(QM)(KM)^T
$$

成对出现。V 位于：

$$
O=PV
$$

若只对 V 乘 $M$，输出也会被旋转；若再对输出乘 $M^T$ 可以还原，却增加额外操作和 layout 约束。

FA3 对低精度 V 重点使用 block scaling 和满足 WGMMA 的 in-kernel transpose，而 incoherent processing 的主要推导针对 Q/K score 计算。理解每个变换在哪个代数位置可以抵消，比笼统说“QKV 都做随机旋转”更准确。

## FP8 Attention 仍然不等于 FP8 全链路

一个 FA3 FP8 kernel 可能包含不同精度：

```text
Q/K/V storage or GEMM operands : FP8
WGMMA accumulation             : FP32-capable accumulator path
row max / exp / row sum        : FP32 critical state
output                          : FP16/BF16 or configured dtype
scales                          : higher precision scalars
```

“FP8 attention”描述主要矩阵乘 operand 和 Tensor Core 路径，不表示 Softmax exp 也用 FP8，亦不表示最终模型所有层都以 FP8 保存。

部署时要从 profiler/SASS 与接口 metadata 确认真正命中了 FP8 WGMMA，而不是：

- 输入以 FP8 存储、内部转为 BF16 计算；
- shape 不支持后回退；
- quantize/dequantize kernel 占去主要时间；
- 高层框架根本没有 dispatch 到 FA3。

仅看 tensor dtype 或包名不足以证明使用了原生低精度计算。

## 论文中的性能数字应该怎样理解

FA3 论文在 H100 80GB SXM5 上报告：

- FP16 forward 相对 FA2 约 1.5–2.0 倍；
- FP16 backward 相对 FA2 约 1.5–1.75 倍；
- FP16 forward 最高约 740 TFLOPS，即约 75% 理论峰值；
- FP8 forward 接近 1.2 PFLOPS；
- 带 block quantization 与 incoherent processing 的 FP8，在包含 outlier 的测试分布上相对 per-tensor FP8 baseline 将 RMSE 降低约 2.6 倍。

这些结论成立于论文的 shape 和计算口径。Benchmark 将 sequence length 从 512 扫到 16K，同时调整 batch 使总 token 数为 16K，并覆盖 head dimension 64/128/256、causal/non-causal。

不能把“740 TFLOPS”直接换成某个模型端到端 tokens/s，因为模型还包括：

- QKV/output projections；
- MLP 或 MoE；
- normalization、RoPE、sampling；
- TP/CP collective；
- optimizer 与反向传播其他部分；
- 输入 layout 和 quantization。

同样，FP8 的 1.2 PFLOPS 是 attention kernel 主要矩阵乘吞吐，不等于训练成本无条件减半。

## 用 Amdahl 定律估算模型级收益

若原模型 step 中 attention 占比为 $p$，FA3 相对旧 kernel 加速 $s$，其他部分不变：

$$
Speedup_{total}
=\frac{1}{(1-p)+p/s}
$$

假设 attention 占 35%，kernel 加速 1.8 倍：

$$
Speedup_{total}
=\frac{1}{0.65+0.35/1.8}
\approx1.18
$$

若长上下文使 attention 占到 70%：

$$
Speedup_{total}
=\frac{1}{0.30+0.70/1.8}
\approx1.45
$$

所以 FA3 对长上下文更有端到端价值；短序列或 MLP 占主导的模型，kernel microbenchmark 再亮眼，总体提升也会受限。

## 为什么 Causal Shape 可能更难调度

Causal attention 只计算下三角。对角线右上的 KV tiles 可以跳过，理论 FLOPs 约减半，但不同 Q row tiles 的有效 KV blocks 数不同：

```text
早期 Q tile -> 只需少量 K/V tiles
后期 Q tile -> 需要更多 K/V tiles
```

如果静态按 Q tile 分给 CTAs，长任务和短任务混在 grid 中，尾部可能只剩少量 CTAs 占用部分 SM。异步流水还需要足够迭代数才能填满和摊薄首尾开销。

因此 causal 的“少算一半”不会严格带来两倍速度，短 sequence/早期 tile 也难以充分利用多 stage pipeline。现代实现会使用 persistent scheduler、动态 tile 分配或 longest-processing-time 思路改善负载，但这些是实现继续演进的部分，不能只从论文的基本算法推断当前所有版本行为。

## GQA/MQA 为何改变 Kernel 任务数量

GQA 让多个 Q heads 共享一个 KV head。FA3 和 FA2 一样避免在 HBM 显式复制 K/V，而是通过 head index 映射复用。

但当 KV heads 很少、Q length 又很短时，按 batch×head×query-tile 形成的并行任务可能不足。特别是 decode：

```text
Q length = 1
KV heads = 1 or few
```

Attention 工作主要是读取长 KV Cache，FA3 论文的长 Q 训练/prefill pipeline 未必是最佳路径。后续实现加入 GQA packing、split-KV、paged KV Cache 与专用 scheduler，正是为了让 serving shape 也产生足够并行度。

文章讨论的 FA3 核心论文机制应与当前仓库的完整 feature set 区分：论文证明 Hopper 异步流水的方向，仓库后来又持续扩展 inference、paged cache、window、softcap 和更多调度能力。

## Prefill、Decode 和训练的收益不同

### 训练

Q/K/V sequence 都较长，需要 forward 和 backward。FA3 的 block pipeline 有足够迭代，论文也主要在此类形状下验证性能和数值。

### Prefill

长 prompt 同样提供长 Q，forward-only 可以直接受益。若 chunked prefill 的 query chunk 较小，pipeline 填充和 CTA 数量需要重新评估。

### Decode

单 token Q 对长 KV Cache，算术强度低、P/V reduction 和 KV 带宽更突出。需要 split-KV 或 paged/decode-specific kernel；不能直接用训练形状的 1.5–2 倍推断 TPOT。

### Speculative Verification

一次验证多个 draft tokens，Q length 比普通 decode 大、比完整 prefill 小，可能落在另一组最佳 tile/split 参数上。动态 proposal length 还会让 CUDA Graph 和 scheduler shape 更复杂。

调研一个推理框架是否“支持 FA3”时，应问每个阶段具体 dispatch 到什么 kernel，而不是只有一个布尔开关。

## 安装了包，为什么可能没有用上 FA3

运行时 dispatch 常受这些条件影响：

- GPU compute capability；
- CUDA、driver、PyTorch 与编译器版本；
- head dimension 与 value head dimension；
- dtype、causal、dropout、window、softcap；
- varlen、paged KV、GQA 与 decode 形状；
- tensor stride 和 layout；
- kernel 包是 FA2、FA3 beta 还是后续统一实现；
- 上层框架自身的 backend 优先级。

官方仓库的 `hopper/` 实现最初面向 H100/H800 和 CUDA 12.3+，后续主分支不断变化并扩展架构/功能支持。因此不要把某篇旧安装命令当成永久接口。

可靠确认方式是：

1. 锁定 commit 或 release，而不是笼统写“最新版”；
2. 运行官方 tests 覆盖目标 shape；
3. 用 profiler 查看真实 kernel symbol；
4. 对 FP8 检查底层指令/性能，确认不是 storage-only；
5. 记录 fallback 原因；
6. 与框架 reference 做数值对比。

尤其不要为了绕过架构检查手工注释 kernel guard。若硬件缺少对应 TMA/WGMMA 语义，强行发射可能得到非法指令、descriptor 错误或静默错误。

## 正确性验证要覆盖异步边界

异步 kernel 的错误不一定每次复现。测试除了数学 shape，还要触发 buffer 和 scheduler 的边界。

### Tile 尾部

- sequence 不是 `BLOCK_M/BLOCK_N` 整数倍；
- head dimension 64、96、128、192、256 等实际支持组合；
- Q/K 不同长度；
- causal 对角 tile；
- varlen 中包含极短和空有效区间。

### Pipeline 状态

- KV tiles 少于 buffer stages；
- 恰好等于 stages；
- 多次绕回 circular buffer；
- persistent CTA 获取下一任务；
- producer/consumer barrier phase 翻转。

### 数值路径

- FP16/BF16 对 FP32 reference；
- FP8 per-tensor、block scale、incoherent processing 的消融；
- 人工注入 outliers；
- causal/non-causal；
- forward 与 $dQ,dK,dV$；
- 不同随机 seed/正交 sign matrix 的一致语义。

### 并发

- 多 CUDA streams；
- CUDA Graph capture/replay；
- TP/CP 通信与 kernel 并行；
- 不同 batch shapes 连续调用；
- 长时间压力测试，捕获偶发 barrier 或 buffer race。

只跑一个 8K×128 的 forward benchmark，无法证明 kernel 可安全用于生产训练。

## 用 Profiler 看三层流水是否成立

Nsight Systems 适合先看宏观时间线：kernel 是否连续、前后是否存在 layout/quantize 空洞、通信是否与 attention 重叠。

Nsight Compute 再看 kernel 内指标：

- Tensor Core/WGMMA 活跃周期；
- TMA/global load 吞吐；
- shared-memory 使用与冲突；
- registers per thread、spill load/store；
- eligible/active warps；
- barrier stall；
- long scoreboard / memory dependency；
- special-function utilization；
- occupancy 与 active CTAs per SM。

诊断可以按现象进行：

| 现象 | 优先检查 |
| --- | --- |
| TMA 后 consumer 经常等待 | stages 太少、HBM latency、tile 搬运过慢 |
| Tensor Core 中间有规律空洞 | Softmax 未隐藏、WGMMA wait 位置过早 |
| Occupancy 很低 | registers/shared memory 过大、tile 配置不合适 |
| SFU 满、Tensor Core 低 | Softmax 成为暴露瓶颈、ping-pong 不充分 |
| FP8 只比 BF16 略快 | 未命中原生 FP8、转换/transpose 成本、shape 受限 |
| Kernel 快但模型不快 | 前后 layout、量化、MLP/通信占主导 |

“GPU utilization 100%”只说明有 kernel 在运行，不能告诉你 TMA、Tensor Core 和 SFU 是否按预期重叠。

## 一份可复现的 Benchmark 应包含什么

至少公开：

```text
GPU model, form factor, clock and power settings
driver, CUDA toolkit, PyTorch
flash-attention commit/release and build flags
Q/K/V dtype and output dtype
batch, q heads, kv heads
seqlen_q, seqlen_k, head_dim
causal/window/dropout/softcap
forward or backward
warm-up, repetitions, synchronization method
TFLOPs formula and causal FLOPs convention
```

比较 FA2/FA3 时必须让输入 layout、dtype、mask 和输出语义一致。若 FA3 使用 FP8、FA2 使用 BF16，应该同时报告：

- 同精度实现差异；
- 低精度额外速度；
- quantization 与 transform 时间；
- 数值误差或模型指标。

只报告最优 kernel latency 会忽略前置 block scale、Hadamard/RoPE 融合是否真的存在于实际模型路径。

## FP8 数值测试为什么必须贴近模型分布

论文的误差实验使用带少量大 outlier 的合成分布，以检验量化对异常通道的敏感性。结果显示完整 FP8 FA3 的 RMSE 低于 per-tensor FP8 baseline，但这不等于任意模型都固定提高 2.6 倍。

生产验证应采集目标模型多个层、不同训练阶段或请求长度下的 Q/K/V 分布，比较：

- max/percentile 与 kurtosis；
- 每 block scale 分布；
- score/logsumexp 误差；
- attention output cosine/RMSE；
- 单层误差随深度累积；
- loss、perplexity 或下游质量；
- 梯度稳定性与 NaN/Inf；
- 长上下文位置的误差是否更大。

若只挑没有 outlier 的正态分布，per-tensor FP8 本来就可能表现很好，无法验证 incoherent processing 的价值；若只构造极端异常，又可能夸大真实收益。

## FA3 没有改变什么

- 它仍是精确 dense attention 的高性能实现，计算复杂度仍为 $O(N^2d)$；
- 它没有让 H100 以外的硬件自动获得相同 WGMMA/TMA 流水；
- 它没有消除长上下文跨设备需求，Ring Attention/context parallel 仍解决另一层问题；
- 它没有让所有 serving shape 都达到训练 benchmark 的利用率；
- FP8 路径仍需验证模型质量，不能只根据 kernel RMSE 决定上线；
- 异步流水增加了 barrier、buffer 和调试复杂度，不是免费抽象；
- 论文结果不代表某个高层框架当前一定调用同一 commit 的实现。

FA3 的意义不是一个“版本号更大”的替换包，而是展示算法必须随硬件执行模型一起演进。FA1 优化 HBM IO，FA2 优化任务与 warp 划分，FA3 则利用 Hopper 的专用异步搬运、异步矩阵乘和低精度能力重新安排整条依赖链。

## 从 FA3 提炼出的异步 Kernel 设计方法

### 1. 先画依赖图，再谈异步

标出哪些 load、GEMM、Softmax 和 rescale 互相依赖。只有跨 tile、跨 warpgroup 的独立工作可以安全重叠。

### 2. 给每类资源稳定的 Producer

TMA producer、Tensor Core consumers 和 SFU work 有清晰角色，减少所有 warps 都执行复杂混合控制流的调度困难。

### 3. 用环形 Buffer 把延迟转为容量问题

预取未来 tile 能隐藏 latency，但要用 shared-memory stages 付费。Depth 必须通过 occupancy 与 stall 共同选择。

### 4. 让低需求角色归还寄存器

Warp specialization 不只分指令，还应匹配资源。Producer 的低 register 需求可转给保存大 accumulators 的 consumer。

### 5. 低精度优化必须同时设计 Layout 与误差

原生 Tensor Core layout、in-kernel transpose、block scale 和 outlier 处理缺一不可。只有 dtype 转换通常得不到完整吞吐或可接受精度。

### 6. 用消融证明每层流水

分别关闭 warp specialization、ping-pong、intra-group pipeline、block quantization 和 incoherent processing，才能知道收益来自哪里，也便于不同 shape 下选择简化路径。

## 小结

FlashAttention-3 面对的是一个由硬件进步制造的新瓶颈：Hopper 的 Tensor Core 已经足够快，若数据搬运、Softmax 和依赖等待仍串行，精确 attention 无法接近峰值。

它用三层异步组织 kernel：

1. Producer warps 用 TMA 预取 K/V，consumer warpgroups 用 WGMMA 计算，借助 circular shared-memory buffer 重叠搬运与执行；
2. 两个 consumer groups ping-pong，让一组的 Softmax 与另一组的 Tensor Core GEMM 同时发生；
3. 单个 group 内跨 KV block 流水，在真正依赖结果前延后 WGMMA wait。

FP8 路径又增加两项数值设计：每个 Q/K/V tile 独立量化，限制 outlier 的影响范围；Q/K 同乘随机 Hadamard 型正交变换，在不改变 $QK^T$ 的前提下摊平离群值。加上 WGMMA layout 对齐和 in-kernel V transpose，FP8 才从理论吞吐变成可用 kernel。

这篇文章也给 Attention 系列补上了从 2022 到 2024 的连续演进：

```text
FlashAttention   -> 控制 HBM IO
FlashAttention-2 -> 改善 block/warp 并行划分
Ring Attention   -> 将 sequence blocks 扩展到多设备
MLA              -> 压缩每个 token 的 KV 表示
FlashAttention-3 -> 利用 Hopper 异步流水与 FP8
```

下一篇 DeepSeek Sparse Attention 会在 MLA 的压缩表示之上继续减少实际访问的历史位置：先用轻量 indexer 选出 top-k，再执行 sparse MLA，并把新的瓶颈推向选择、离散 gather 与稀疏 kernel。

## 参考资料

- [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608)
- [FlashAttention-3 paper PDF](https://tridao.me/publications/flash3/flash3.pdf)
- [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision — PyTorch](https://pytorch.org/blog/flashattention-3/)
- [Next Generation of FlashAttention — NVIDIA](https://developer.nvidia.com/blog/next-generation-of-flashattention/)
- [FlashAttention official repository](https://github.com/Dao-AILab/flash-attention)
- [FlashAttention-3 at NeurIPS 2024](https://papers.neurips.cc/paper_files/paper/2024/file/7ede97c3e082c6df10a8d6103a2eebd2-Paper-Conference.pdf)
