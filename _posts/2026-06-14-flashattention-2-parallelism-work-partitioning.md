---
layout: post
title: "FlashAttention-2：从 IO-aware 到更好的并行划分"
subtitle: "为什么同一个精确注意力算法，还能通过 Thread Block 与 Warp 重排再快一倍"
date: 2026-06-14 09:00:00 +0800
last_modified_at: 2026-08-09
author: iStar
catalog: true
series: attention-long-context
series_order: 30
technology_year: 2023
mathjax: true
tags: [AI Infra, FlashAttention, GPU Kernel, CUDA]
---

第一代 FlashAttention 已经解决了精确注意力最显眼的 IO 问题：它不再把完整的 $N\times N$ score 和 probability 矩阵写回 HBM，而是分块读取 $Q、K、V$，在片上完成在线 Softmax 和输出累积。

算法没有近似，额外显存从二次规模降到线性规模，速度也明显超过物化 attention matrix 的实现。但在 A100 上，它的实际计算吞吐仍显著落后于优化良好的 GEMM。

问题不再是“是否做分块”，而是分块之后的工作怎样映射到 GPU：

- 只沿 batch 和 head 并行时，长序列、小 batch 可能没有足够 thread blocks 填满所有 SM；
- 一个 thread block 内若按 K/V 切给多个 warps，它们需要频繁通过 shared memory 汇总输出；
- 在线 Softmax 的标量运算远慢于 Tensor Core 矩阵乘法，即使 FLOPs 占比很小也会拖慢 kernel。

FlashAttention-2 的贡献正是在这一层。它保持 IO-aware 精确注意力的基本结构，却重新设计算法细节、thread-block parallelism 和 warp-level work partition，让更多时间落在 Tensor Core 擅长的矩阵乘法上。

本文不再从头推导第一代 FlashAttention，而是集中解释三次重排为什么有效，以及如何判断一项 benchmark 的加速究竟来自算法、并行度还是测试口径。

## “已经减少 IO”为什么还不够

一个 GPU kernel 的性能不只由总 FLOPs 和 HBM 字节数决定。至少还有三层约束：

```text
设备层：有多少 SM、Tensor Core、显存带宽
block 层：能启动多少 thread blocks，occupancy 是否足够
warp 层：寄存器、shared memory、同步与数据交换是否高效
```

第一代 FlashAttention 主要改变数据流：让 $Q、K、V$ 的 tile 在 SRAM/寄存器中尽可能完成 $QK^T$、Softmax 和 $PV$，从而避免巨大的 HBM 往返。

但一个 tile 放进片上之后，仍要回答：

- 哪个 thread block 负责哪片 attention matrix；
- 一个 block 内的 4 或 8 个 warps 各算什么；
- 中间结果存寄存器还是 shared memory；
- 哪些 warp 之间需要同步；
- 长序列能否产生足够多的独立 block。

如果这些问题处理得不好，Tensor Core 可能在等同步，某些 SM 可能没有工作，或者寄存器压力迫使数据 spill 到更慢的存储层级。减少 HBM IO 只是必要条件，并不是自动达到峰值吞吐的保证。

## 先看 A100 上两类 FLOPs 的价格差

FlashAttention-2 论文用 A100 说明了一个容易被总 FLOPs 掩盖的事实：不同类型的浮点运算并不等价。

论文给出的理论峰值是：

- FP16/BF16 Tensor Core matrix multiply：312 TFLOPs/s；
- 非矩阵 FP32 运算：19.5 TFLOPs/s。

二者相差 16 倍。于是 1 个 exp、除法、比较或标量缩放不能按“也只是一个 FLOP”与 Tensor Core fused multiply-add 等价看待。

可以用简化时间模型理解：

$$
T \approx
\frac{F_{matmul}}{P_{tensorcore}}
+\frac{F_{nonmatmul}}{P_{scalar}}
+T_{memory}
+T_{sync}
$$

即使 $F_{nonmatmul}\ll F_{matmul}$，只要 $P_{scalar}$ 小得多，第二项仍可能占据明显时间。在线 Softmax 恰好包含 row max、exp、row sum、rescale 等非矩阵操作。

所以 FA2 的目标不是机械地减少总 FLOPs，而是尽量减少昂贵的非矩阵 FLOPs，并让剩余工作更连续地喂给 Tensor Core。

## 基线：分块注意力在算什么

设单个 head：

$$
Q,K,V\in\mathbb{R}^{N\times d}
$$

将 Q 按行分为大小 $B_r$ 的块 $Q_i$，K/V 按列方向对应分为大小 $B_c$ 的块 $K_j,V_j$。一个 Q block 要遍历所有可见 KV blocks：

$$
S_i^{(j)}=Q_iK_j^T
$$

然后更新当前行块的最大值 $m_i$、指数和 $\ell_i$ 与输出累积 $O_i$。在线 Softmax 使各 KV tile 的局部统计可以精确合并，而不需要保存完整 $S$。

第一代与第二代都建立在这个结构上。FA2 没有把复杂度从 $O(N^2d)$ 变成线性，也没有用稀疏近似；它改变的是循环顺序、状态表示和并行任务边界。

这点很重要：如果把 FA2 解释成一种新的 attention 公式，就会错过它最有价值的系统思想——同一个数学算法，映射到不同硬件层级可以有很大性能差异。

## 改动一：延迟归一化，减少非矩阵运算

在线 Softmax 在合并一个新 KV block 时，要处理最大值变化带来的 rescale。设此前统计为 $m^{old},\ell^{old},O^{old}$，新 block 的 scores 为 $S$，更新后的最大值为：

$$
m^{new}=\max(m^{old},\operatorname{rowmax}(S))
$$

指数和更新为：

$$
\ell^{new}
=e^{m^{old}-m^{new}}\ell^{old}
+\operatorname{rowsum}(e^{S-m^{new}})
$$

如果每一轮都维护已经除以 $\ell$ 的归一化输出，就需要反复对旧输出和新贡献做比例缩放。FA2 改为在循环中维护未最终归一化的输出累积：

$$
\widetilde O^{new}
=e^{m^{old}-m^{new}}\widetilde O^{old}
+e^{S-m^{new}}V
$$

直到遍历完所有 KV blocks 后，才做一次：

$$
O=\operatorname{diag}(\ell)^{-1}\widetilde O
$$

这不是省掉数值稳定性。最大值变化时，旧累积仍乘 $e^{m^{old}-m^{new}}$；只是把不必每轮进行的除法/缩放推迟到最后。

另一个细节是反向传播不必同时保存 row max $m$ 和指数和 $\ell$，只保存：

$$
L=m+\log\ell
$$

也就是 logsumexp。Backward 重算 probability 时可用：

$$
P_{ij}=e^{S_{ij}-L_i}
$$

这些改变看起来不像“换了一个大算法”，却恰好减少了 GPU 不擅长的逐元素操作和状态搬运。

## 因果掩码也应在 Tile 层提前判断

对于 causal attention，位置 $i$ 不能读取未来位置 $j>i$。分块后，tile 可分为三类：

```text
完全在因果边界内：直接计算，不必逐元素检查 mask
与对角线相交：计算时应用 causal mask
完全位于未来区域：整个 tile 跳过
```

用方块表示 attention matrix：

```text
Q rows
  |
  v
  [算][跳][跳][跳]
  [算][边][跳][跳]
  [算][算][边][跳]
  [算][算][算][边] -> K columns
```

只有对角线附近的 block 需要执行逐元素 mask。右上角约一半 tile 可以整体不发起矩阵乘法，左下角 tile 则无需反复判断每个元素。

这个优化第一代结构也能利用，但 FA2 的 row-block 任务划分使 causal tile 边界更自然地成为每个 worker 的循环范围。论文在长序列下观察到 causal 相对 non-causal 可跳过近一半工作，不过实际速度不会严格等于 2 倍，因为仍有 Softmax、边界 tile、launch 和负载不均衡成本。

## 改动二：从 Batch×Head 扩展到 Sequence Parallelism

第一代 FlashAttention 的主要 thread-block 并行维度是 batch 和 attention head。粗略地说，一个 head 由一个 thread block 负责，因此可并行 block 数约为：

$$
N_{blocks}^{FA1}\approx B\times H
$$

对 A100 的 108 个 SM，若 batch=1、heads=32，就只有约 32 个独立 block，无法让所有 SM 都持续有工作。长序列反而使每个 block 内循环很久，但空闲 SM 不能帮忙。

而长上下文训练经常因为显存压力把 batch size 降得很小，这正是 attention 工作最多、外层并行度却最少的组合。

FA2 把 Q 的 sequence row blocks 也变成 thread-block 并行维度：

$$
N_{blocks}^{FA2}
\approx B\times H\times
\left\lceil\frac{N}{B_r}\right\rceil
$$

例如 batch=1、heads=32、sequence=8192、$B_r=128$：

$$
1\times32\times64=2048
$$

个 row-block 任务，调度器有充足工作分发给 108 个 SM。每个 worker 负责自己的 $Q_i$ 与输出 $O_i$，遍历它能看到的 K/V blocks。

## 为什么 Forward 的 Row Blocks 可以独立

对不同 Q row blocks $Q_i$ 和 $Q_{i'}$，它们需要读取相同的 K/V，却写入不同的输出行 $O_i,O_{i'}$。每一行的 Softmax 归一化也彼此独立。

因此：

```text
worker 0: Q rows   0..127   x all visible K/V -> O rows   0..127
worker 1: Q rows 128..255   x all visible K/V -> O rows 128..255
worker 2: Q rows 256..383   x all visible K/V -> O rows 256..383
```

worker 之间不需要合并同一个输出行，可以 embarrassingly parallel。代价是不同 worker 可能重复从 HBM/L2 读取 K/V，但换来了足够 occupancy；GPU cache 与访问顺序还能吸收部分重复读取。

这里体现了性能工程中常见的取舍：理论最少的数据读取，不一定等于实际最快。如果过度追求复用导致只有少数 block 工作，闲置的 SM 比额外读取更昂贵。

## Backward 为什么沿 Column Blocks 划分

反向传播需要计算：

$$
dV=P^TdO,
\quad dP=dOV^T,
\quad dS=P\circ(dP-D)
$$

$$
dQ=dSK,
\quad dK=dS^TQ
$$

FA2 的 backward 让一个 worker 负责 attention matrix 的 column block，也就是一块 K/V。这样它可以在片上累积对应的 $dK_j,dV_j$，避免不同 worker 反复合并这两项。

但多个 column workers 都会为同一 $dQ_i$ 贡献一部分：

$$
dQ_i=\sum_j dS_i^{(j)}K_j
$$

于是 $dQ$ 成为跨 workers 的共享归约，需要 atomic add 或等价通信方式。Forward 的 row-block 并行几乎没有跨 block 写冲突，Backward 则必须在更多并行度与 $dQ$ 合并成本之间权衡。

这也是为什么同一 kernel 的前向和反向不应假定使用相同任务划分。要优先让每个 worker 在片上长期持有它负责累积的梯度，同时为不得不共享的量选择可控的归约机制。

## Occupancy 不是“线程越多越好”

Sequence parallelism 产生更多 thread blocks，但 occupancy 还受每个 block 的资源消耗限制。一个 block 若使用太多寄存器或 shared memory，同一个 SM 上能同时驻留的 blocks 就会减少。

设每个 SM 可用 shared memory 为 $S_{SM}$，一个 block 使用 $S_{block}$，仅从这一项看可驻留 block 数上限为：

$$
R_{shared}=\left\lfloor\frac{S_{SM}}{S_{block}}\right\rfloor
$$

寄存器、最大 threads/warps 与架构限制还会进一步收紧它。增大 $B_r,B_c$ 可以减少 tile 边界和 shared-memory 往返，却会增加片上状态：

- Q/K/V tile 更大；
- score/probability fragment 更大；
- output accumulator 使用更多寄存器；
- Softmax 统计量更多。

超过阈值后可能发生 register spilling，数据被迫落到 local memory；严重时 shared memory 超过硬件限制，kernel 根本无法启动。

所以 FA2 论文使用的典型 block shape 在 64/128 的组合中按 head dimension 和硬件手工选择。Tile 不是越大越好，parallel blocks 也不是越多越好，目标是同时维持数据复用、occupancy 与寄存器可行性。

## 改动三：Warp 从 Split-K 改为 Split-Q

一个 thread block 通常包含 4 或 8 个 warps，每个 warp 有 32 个 threads。确定 block 负责某片 attention 后，还要决定 warps 如何瓜分 tile。

### 第一代的 Split-K

第一代 forward 大致让多个 warps 共享 Q，把 K/V 的不同片段分给各 warp：

```text
warp 0: Q x K0 -> partial score -> partial O0
warp 1: Q x K1 -> partial score -> partial O1
warp 2: Q x K2 -> partial score -> partial O2
warp 3: Q x K3 -> partial score -> partial O3
                                  |
                                  v
                         合并成相同的 O rows
```

因为不同 K slices 都对同一组 output rows 有贡献，各 warp 的 partial outputs 必须汇总。Warp 之间不能只靠各自寄存器完成，需要把中间结果写入 shared memory、同步，再读取和相加。

这就是论文所说的 split-K。这里的 K 是矩阵乘法的 reduction dimension 语义，不应与 Transformer 的 Key 张量名称混为一谈；恰好在 $PV$ 阶段，两者都涉及沿 key-position 维归约。

### FA2 的 Split-Q

FA2 改为让 warps 共享 K/V，各自负责不同 Q rows：

```text
warp 0: Q0 x K -> scores0 -> O0
warp 1: Q1 x K -> scores1 -> O1
warp 2: Q2 x K -> scores2 -> O2
warp 3: Q3 x K -> scores3 -> O3
```

每个 warp 从 score 到 $PV$ 都拥有自己对应的 output rows，不需要与其他 warp 合并同一 $O$。K/V 会被所有 warps 读取，但可通过 shared memory 共享；关键收益是省掉 partial output 的跨-warp 通信。

对 forward 来说，这种划分让：

- output accumulator 更自然地留在各 warp 寄存器；
- shared memory 读写减少；
- warp synchronization 减少；
- 数据所有权从输入片段变成输出片段，更清晰。

Backward 的依赖更复杂，仍需要一定同步，但同样尽量避免会产生大规模中间归约的 split-K 布局。

## Thread Block 与 Warp 是两次不同的划分

FA2 的两类并行优化容易被混为一句“沿 sequence 维并行”，其实发生在两个层次。

| 层次 | FA1 的主要做法 | FA2 的改变 | 目标 |
| --- | --- | --- | --- |
| Thread blocks 之间 | 主要按 batch × head | 再按 sequence tiles | 产生足够任务填满 SM |
| 同一 block 的 warps 之间 | split-K，合并 partial output | split-Q，各自拥有 output rows | 减少 shared-memory 通信 |

Block-level parallelism 解决的是全 GPU 有没有足够工作；warp-level partition 解决的是一个 SM 内部协作是否高效。只改其中一层，仍可能受另一层限制。

这也是分析任何 fused kernel 的通用方法：先问 grid 如何覆盖数据，再问 block 内线程如何拥有输入、累积量和输出。

## GQA/MQA 为什么值得在 Kernel 内直接表达

Multi-Query Attention 与 Grouped-Query Attention 让多个 Q heads 共享更少的 K/V heads。若为了复用普通 MHA kernel 先显式复制 K/V：

```python
K_expanded = repeat_interleave(K, groups, dim=head_dim)
V_expanded = repeat_interleave(V, groups, dim=head_dim)
```

就会增加 HBM 读写和临时显存，抵消减少 KV heads 的目的。

FA2 直接通过 head index 映射，让不同 Q heads 指向共享 K/V head。Forward 无需物化复制；Backward 中多个 Q groups 对同一 K/V head 的梯度则需要求和。

这个细节说明高性能 attention kernel 的接口不能只接收四维 tensor shape，还要理解 head mapping。模型结构在数学上节省的内存，只有运行时不偷偷展开时才能成为真实收益。

## FA2 更偏训练，Decode 需要不同策略

论文的核心 benchmark 是训练和较长 Q sequence 的 forward/backward。在 prefill 或训练中，Q 和 K/V 都有较长 sequence，row-block parallelism 能产生大量工作。

自回归 decode 常见形状却是：

```text
Q length = 1
KV length = thousands
```

只有一个 query row，无法沿 Q sequence 产生许多独立 row blocks；瓶颈更多是尽快读取很长的 KV Cache。官方仓库后续为 decode 加入了专门优化：把 KV 加载拆给多个 thread blocks，再用额外步骤合并结果，并支持原地更新 KV Cache、RoPE 和 paged cache。

因此不能看到“FlashAttention-2 更快”就假定它对所有 LLM Serving 阶段有相同收益：

- 训练：长 Q、需要 backward，FA2 是主要适用场景；
- Prefill：Q 较长，很多并行思路仍有效；
- Decode：Q 很短、KV 很长，需要 split-KV/paged attention 等专用路径；
- Chunked Prefill：介于两者之间，query chunk 和 prefix 长度共同决定 kernel。

做推理 benchmark 时必须分别报告 prefill 与 decode，而不是只用一个 attention latency 概括。

## 变长序列为什么需要 Varlen 接口

真实 batch 中的 sequence lengths 常不相同。把所有样本 padding 到最大长度，会让短序列在 attention 中计算大量无效 tiles。

FlashAttention 仓库提供 varlen 接口，通常将所有有效 token packed 到一维 token 维，并用 cumulative lengths 描述边界：

```text
tokens: [seq0 valid tokens][seq1 valid tokens][seq2 valid tokens]
cu_seqlens: [0, len0, len0+len1, len0+len1+len2]
```

Kernel 据此恢复每条序列的 Q/KV 范围，不让不同样本互相 attention。性能收益取决于长度分布：若本来等长，packed metadata 未必带来额外好处；若长短差异大，避免 padding 的价值会很明显。

对 causal cross-attention 或 decode，`seqlen_q != seqlen_k` 时还要明确 causal mask 如何对齐。官方仓库从 2.1 起采用 bottom-right 对齐语义。升级版本时若模型依赖旧的 top-left 行为，不能只验证 tensor shape，应对边界 mask 做数值测试。

## 如何从 PyTorch 侧确认实际后端

调用高层 `scaled_dot_product_attention` 并不保证一定执行某个版本的 FlashAttention。Backend dispatch 会受这些条件影响：

- GPU architecture 与 CUDA/ROCm 版本；
- dtype 是否为 FP16/BF16 等支持类型；
- head dimension；
- causal mask、dropout、GQA、window 等功能组合；
- tensor stride、contiguity 与 layout；
- training/forward-only；
- 框架编译时包含的 kernel 版本。

因此正确做法是记录框架、驱动、kernel 包版本，并通过 profiler 或 backend 日志确认实际 kernel name。只在配置中写：

```python
attn_implementation = "flash_attention_2"
```

不能替代运行时验证；不满足条件时，框架可能报错，也可能回退到其他实现。

直接使用官方包时，接口选择也影响布局成本。若 Q/K/V 已紧邻存储，packed interface 可以避免显式拼接或拆分；变长 batch 应使用 varlen interface；decode 则要考虑带 KV Cache 的专用函数。API 名称会随版本变化，应以锁定版本的 README 与 tests 为准。

## 正确性验证不能只看 `allclose`

FA2 仍是精确 attention，但浮点归约顺序改变会产生数值差异。测试应分层进行。

### Forward

与高精度 reference 对比输出，覆盖：

- causal / non-causal；
- head dim 32、64、128、256 等实际组合；
- 等长与 varlen；
- MHA、GQA、MQA；
- `seqlen_q != seqlen_k`；
- 非 block size 整数倍的尾部；
- 极端 logits，验证 Softmax 稳定性。

容差要结合 dtype 和 sequence length 制定。只看一个随机 seed、一个 shape 的最大误差，没有足够说明力。

### Backward

分别检查 $dQ,dK,dV$，尤其关注 GQA 共享 K/V heads 的梯度归约。还应测试 deterministic 选项、dropout RNG 一致性和 gradient checkpointing 组合。

### 模型级

至少跑一段固定 seed 的短训练，比较 loss trajectory、是否出现 NaN/Inf、梯度 norm 与显存峰值。Kernel 单测通过不代表 layout、mask 和框架 glue code 没有问题。

## 性能 Benchmark 怎样避免误导

FA2 论文报告在 A100 上相对第一代约 2 倍加速，并达到 forward 最高约 73% 理论峰值；这是特定硬件、shape、精度和统计口径下的结果，不是所有部署的固定倍数。

一份可解释的 benchmark 应至少公开：

```text
GPU model / clocks / power mode
driver, CUDA, PyTorch, flash-attn version
dtype and TF32 settings
batch, heads, q heads, kv heads
seqlen_q, seqlen_k, head_dim
causal, dropout, training or inference
warm-up count and timed iterations
forward / backward / end-to-end scope
```

### 固定 Token 数还是固定 Batch

论文的一组实验随 sequence length 调整 batch，使 batch 内总 token 数保持 16K。这能比较长序列下的 kernel 效率，却与固定 batch 的结果回答不同问题。

### FLOPs 口径

Causal attention 跳过约半个矩阵。若计算 TFLOPs/s 时仍按完整 $N^2$ FLOPs 计数，数字会高估实际执行的有效算术吞吐。论文在 attention microbenchmark 中对 causal FLOPs 除以 2，但模型级 MFU 沿用文献常用公式。引用数字时必须说明口径。

### Kernel 时间与端到端时间

Attention 快 2 倍不代表模型训练快 2 倍。设 attention 占原总时间比例为 $p$，attention 加速 $s$ 倍，Amdahl 定律给出总加速上限：

$$
Speedup_{total}
=\frac{1}{(1-p)+p/s}
$$

若 attention 占 40%、加速 2 倍：

$$
Speedup_{total}=\frac{1}{0.6+0.2}=1.25
$$

Embedding、MLP、collective、optimizer 和 dataloader 都不会因 attention kernel 自动加速。论文的 GPT 训练端到端结果最高约 1.3 倍，正体现了这一差异。

## 用 Profiler 判断瓶颈属于哪一层

若 FA2 没有获得预期收益，可以按层检查。

### Grid 太小

表现为 active SM 数不足，长时间只有少数 blocks。检查 batch × heads × Q row tiles 是否足够，以及是否实际落到了支持 sequence parallelism 的 kernel。

### Shared Memory / Register 压力

表现为理论 blocks 很多，实际 occupancy 仍低，或出现 local-memory traffic。检查每 block registers、dynamic shared memory、spill load/store 和选择的 tile shape。

### Warp Stall

若 barrier、memory dependency 或 scoreboard stall 高，可能仍在等待 shared-memory 通信或数据加载。需要结合 source/SASS 与 Nsight Compute 指标定位，不能只看 GPU utilization 百分比。

### 非矩阵 Pipe 占比

Tensor Core 利用不高但普通 FP/特殊函数单元忙，可能受 Softmax exp、rescale、mask 等影响。小 shape 中固定标量开销尤其明显。

### HBM / L2 限制

FA2 减少不必要 IO，却仍需读取 Q/K/V 与写 O；decode 或极端 shape 仍可能带宽受限。此时继续追求更高 Tensor Core 指标没有意义。

### 上层 Layout 转换

Kernel 本身很快，但调用前后发生 transpose、contiguous、padding 或 QKV pack/unpack，端到端收益会消失。Profiler 时间线必须包含 attention 周边算子，而不是只截取 kernel。

## FA2 没有改变什么

理解边界与理解优化同样重要。

- 它仍计算精确 dense attention，算术复杂度仍是 $O(N^2d)$；
- 它没有让无限长上下文变得免费，序列翻倍仍显著增加计算；
- 它不负责跨 GPU 拆分一条超长序列；Ring Attention 等方法解决的是设备间 sequence parallelism；
- 它不自动减少 decode KV Cache 大小，GQA/MQA 与量化负责另一层问题；
- 它不是任意 GPU 上都能获得论文中的 A100 利用率；
- 它也不意味着 FlashAttention 的每个后续版本都只是在同一 CUDA kernel 上小修小补。

FA2 的核心是把第一代正确的 IO 算法进一步变成更好的 GPU 调度。到了 Hopper，异步 TMA、WGMMA 和 producer-consumer warp specialization 又改变了最合适的数据流水，形成 FlashAttention-3 的主题。

## 从 FA2 提炼出的 Kernel 设计方法

把具体公式抽开，FA2 提供了一套可迁移到其他 fused kernel 的分析顺序。

### 1. 先区分 FLOP 类型

不要把 Tensor Core matmul、标量 FP32、exp/div 和地址计算放在同一个 FLOPs 总数里。硬件吞吐不同，优化价值也不同。

### 2. 找到独立输出所有权

Forward 按 Q rows 切后，每个 worker 独立拥有 O rows，减少跨 worker 归约。良好的划分往往让输出由单一执行单元长期持有。

### 3. 再检查 reduction 的代价

Backward 的 $dQ$ 无法完全避免跨 column workers 合并，于是要显式评估 atomic 成本，而不是假装所有维度都 embarrassingly parallel。

### 4. 把并行度与片上资源一起算

更多 blocks 不代表更高 occupancy；tile 越大也不代表复用越好。寄存器、shared memory 和 spill 必须一起 profile。

### 5. 让算法布局贴近硬件层级

Sequence tiles 对应 thread blocks，Q subtiles 对应 warps，MMA fragments 对应 Tensor Core 指令。每一层都应有清晰的数据所有权与通信边界。

这比背诵某个固定 `BLOCK_M=128` 更持久，因为硬件、dtype 和 head dimension 改变后，固定参数可能失效，分析框架仍然成立。

## 小结

FlashAttention-2 没有推翻第一代，而是回答了第一代成功之后才会暴露的问题：数据已经留在片上，为什么 Tensor Core 仍然没有被充分利用？

答案集中在三项改动：

1. 延迟输出归一化、只保存 logsumexp，减少昂贵的非矩阵 FLOPs；
2. 除 batch 和 head 外再沿 sequence tiles 分配 thread blocks，让长序列、小 batch 也能填满 SM；
3. block 内从 split-K 改为 split-Q，让各 warp 拥有独立 output rows，减少 shared-memory 通信与同步。

Forward 按 row blocks 并行，Backward 按 column blocks 并行并处理 $dQ$ 归约，进一步说明“相同数据维度”不一定适合前后向使用同一任务划分。

FA2 的价值不仅是论文报告的速度数字，更在于它把 attention 优化从 IO complexity 推进到 GPU execution mapping：先决定数据怎样分块，再决定 blocks 怎样占满设备，最后决定 warps 怎样少通信地拥有输出。

下一篇 Ring Attention 会把视角从单 GPU 内部转向多设备：当单卡即使使用 FA2 也放不下一条超长序列时，怎样把 sequence blocks 分散到多卡，并让 KV 通信与本地 FlashAttention 计算重叠。

## 参考资料

- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)
- [FlashAttention-2 paper PDF](https://tridao.me/publications/flash2/flash2.pdf)
- [FlashAttention official repository](https://github.com/Dao-AILab/flash-attention)
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
