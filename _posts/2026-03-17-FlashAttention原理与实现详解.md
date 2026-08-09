---
layout: post
title: "FlashAttention：IO-aware 的精确注意力"
subtitle: "从在线 Softmax 到 GPU 分块流水线"
date: 2026-03-17
last_modified_at: 2026-08-09
author: iStar
catalog: true
series: attention-long-context
series_order: 10
technology_year: 2022
mathjax: true
tags: [注意力机制, FlashAttention, GPU优化]
---

FlashAttention 最容易被误解的地方，是它看起来并没有少算什么：输入仍然是 $Q、K、V$，输出仍然等于标准的 scaled dot-product attention，稠密注意力的算术复杂度也仍是二次方。既然公式没变，它为什么能更快、还更省显存？

答案不在公式本身，而在数据如何经过 GPU。朴素实现把一个巨大的中间矩阵写入显存，再读回来做 Softmax 和后续矩阵乘法；FlashAttention 改变计算顺序，让一个数据块在更快的片上存储中完成尽可能多的工作，并避免物化完整注意力矩阵。它优化的是 **IO 路径**，不是把精确注意力换成近似算法。

## 从标准注意力的三步计算开始

设序列长度为 $N$，每个注意力头的维度为 $d$：

$$
Q,K,V \in \mathbb{R}^{N \times d}
$$

标准注意力为：

$$
S = \frac{QK^T}{\sqrt d}, \qquad
P = \operatorname{softmax}(S), \qquad
O = PV
$$

可以把 Query 理解成当前 token 提出的问题，Key 是每个 token 的索引，Value 是被汇总的内容。$S_{ij}$ 表示第 $i$ 个 Query 与第 $j$ 个 Key 的匹配程度，Softmax 将一整行分数归一化成权重，最后用这些权重对 Value 求和。

如果序列只有 4 个 token，$S$ 和 $P$ 都只是 4×4；当 $N=4096$ 时，每个矩阵已有约 1678 万个元素。对 batch、多头和多层模型，保存这些中间结果的成本会迅速放大。

朴素 GPU 实现往往对应三组 kernel：

```text
Q @ K.T  ──► 将 S 写入 HBM
                    │
                    ▼
               读取 S，做 Softmax，再将 P 写入 HBM
                    │
                    ▼
               读取 P 和 V，计算 P @ V
```

HBM 是 GPU 上容量较大的全局显存；寄存器和 shared memory 容量更小，但离计算单元更近、带宽更高。上面的流程让 $N\times N$ 数据在 HBM 中往返，许多时间不是花在乘加运算上，而是等待数据。

## 计算瓶颈为什么由 IO 决定

判断一个 GPU kernel 更偏向计算受限还是带宽受限，可以看算术强度（arithmetic intensity）：

$$
\text{arithmetic intensity} =
\frac{\text{浮点运算次数}}{\text{从显存搬运的字节数}}
$$

同样的数据若只做一次加法就写回，算术强度很低；读入后连续参与多次矩阵乘法，算术强度就更高。现代 Tensor Core 的计算吞吐增长很快，如果数据供应不上，更多计算单元也只能等待。

FlashAttention 的思路类似在厨房里一次取来本轮需要的食材，在操作台上完成切配、烹饪和装盘，再去仓库取下一批。它不会减少菜品本身的工序，却减少了往返仓库的次数。

具体做法是：

1. 将 $Q$ 按行分块，将 $K、V$ 按序列维分块；
2. 把当前块搬入 SRAM/shared memory 或寄存器；
3. 计算局部 score、Softmax 统计量和输出累积；
4. 只把最终输出与少量归一化信息写回 HBM。

困难在于 Softmax 必须看到整行数据。一个 Query 行可能跨越多个 KV 块，不能分别对每个块做 Softmax 后直接拼起来。

## 在线 Softmax 如何合并多个块

对向量 $x$，数值稳定的 Softmax 通常先减去最大值：

$$
\operatorname{softmax}(x_i)
= \frac{e^{x_i-m}}{\sum_j e^{x_j-m}},
\qquad m=\max_j x_j
$$

减最大值避免较大的指数上溢。问题是：逐块读取时，一开始并不知道全局最大值。在线 Softmax 的关键，是在读到新块后修正旧统计量。

假设已经处理的部分最大值是 $m$，指数和是 $\ell$：

$$
\ell=\sum_{j\in \text{old}}e^{x_j-m}
$$

新块最大值为 $m_b$，相对于它的指数和为 $\ell_b$。合并后的最大值与指数和为：

$$
m' = \max(m,m_b)
$$

$$
\ell' = e^{m-m'}\ell + e^{m_b-m'}\ell_b
$$

两个指数因子把旧块和新块都换算到共同基准 $m'$。输出累积也用同样方式缩放，因此不需要保存此前的全部 score。

例如第一块分数是 $[1,2]$，第二块是 $[3,4]$。第一块可以用 $m=2$ 保存统计量；读到第二块后，全局最大值变成 4，第一块对应的指数和乘上 $e^{2-4}$ 即可。若先分别对两块归一化，每块权重之和都会变成 1，之后已经无法知道两块之间谁更重要。

## 分块算法的完整数据流

对一个 Query block，算法依次遍历 KV blocks，并为每个 Query 行维护：

- 当前最大值 $m_i$；
- 当前指数和 $\ell_i$；
- 尚未除以 $\ell_i$ 的输出累积 $o_i$。

简化伪代码如下：

```text
for each Q block:
    load Q block
    m = -inf
    l = 0
    o = 0

    for each K/V block:
        load K and V block
        s = Q_block @ K_block.T * scale
        apply mask to s

        m_new = max(m, rowmax(s))
        p = exp(s - m_new)
        l_new = exp(m - m_new) * l + rowsum(p)
        o = exp(m - m_new) * o + p @ V_block

        m = m_new
        l = l_new

    O_block = o / l
    write O_block
```

真实 kernel 会把矩阵乘法、mask、位置偏置、dropout、归一化与数据搬运进一步融合。tile 尺寸也不是越大越好：块太小会增加循环与调度开销，块太大则可能超过 shared memory 或寄存器容量，降低 occupancy。实现需要根据 head dimension、dtype、GPU 架构和 mask 类型选择布局。

### 因果掩码带来的额外机会

自回归模型使用 causal mask，第 $i$ 个 token 只能关注不晚于自己的位置。分块后，主对角线右上方的 KV blocks 可以直接跳过；与对角线相交的块则在内部应用 mask。

这并不改变最坏情况下的 $O(N^2)$ 复杂度，但避免了已知无效区域的计算。对非因果注意力、滑动窗口和任意稀疏 mask，合法块的形状又会不同，因此同一个“FlashAttention backend”可能包含许多专门 kernel。

## 为什么反向传播也更省显存

训练时，反向传播需要 Softmax probability 等中间信息。朴素实现会在 forward 保存 $N\times N$ 的 $P$，以便 backward 使用。

FlashAttention 在 forward 只保存输出 $O$ 和每行的归一化统计量；backward 再按块重算局部的 $S$ 与 $P$。这是一种有意识的取舍：

```text
少量额外矩阵计算
        换取
不保存巨大的 attention probability
        以及
更少的 HBM 读写和更低的峰值显存
```

“重算”不必然意味着更慢。当计算单元相对空闲、HBM 搬运才是瓶颈时，多做一些局部计算反而可以缩短整体时间。这也是理解现代 GPU 优化时很重要的一点：FLOPs 更少的程序，不一定更快。

## 它优化了什么，又没有优化什么

对于稠密注意力，FlashAttention 仍要计算大部分 $QK^T$ 元素：

$$
\text{time complexity}=O(N^2d)
$$

它避免在 HBM 中保存完整 $S$ 和 $P$，使额外显存占用随序列长度近似线性增长。论文还从 IO complexity 角度分析了 SRAM 大小时的 HBM 访问量。

因此需要区分三种说法：

- **正确**：FlashAttention 减少了显存访问和中间张量占用；
- **通常正确**：长序列训练或 prefill 会因此获得明显收益；
- **错误**：FlashAttention 把稠密注意力从二次时间变成线性时间。

若希望减少 token 对的数量，需要滑动窗口、块稀疏、Top-k 或线性注意力等另一类算法；它们通常改变注意力模式或引入近似，与 FlashAttention 的“精确但换序计算”不是同一件事。

## 从 FlashAttention-1 到 4

各版本的共同主线是让算法映射更贴合硬件，而不是更改 Attention 的数学定义。

### FlashAttention-1：建立 IO-aware 算法

第一版给出了分块、在线 Softmax 与反向重计算的完整方案，并分析了 HBM 与片上存储之间的访问复杂度。它解决了“如何不物化完整注意力矩阵”的核心问题。

### FlashAttention-2：改善并行划分

第二版减少了非矩阵乘法 FLOPs，并重新划分 thread block 与 warp 的工作，让 GPU 上的并行度和负载均衡更好。算法思想相近，但更高比例的时间能落在高吞吐矩阵乘法上。

### FlashAttention-3：使用 Hopper 的异步能力

第三版围绕 Hopper GPU 设计，通过 warp specialization、Tensor Memory Accelerator（TMA）与异步执行重叠数据搬运和计算，还将块矩阵乘法与 Softmax 流水化。论文同时研究了 FP8 路径及其数值误差控制。

这里的关键不是“版本号更大就适合所有 GPU”。架构专用特性、CUDA 工具链、dtype 和形状支持会决定实际能否调用对应 kernel。

### FlashAttention-4：面向不对称硬件扩展

2026 年发布的 FlashAttention-4 继续研究算法与 kernel pipeline 的协同设计，目标是适应新一代 GPU 中计算吞吐与其他单元并非同比增长的现象。它属于快速演进中的实现，使用时应核对论文、官方仓库和框架集成状态，不应只按名称假设已经替代所有旧后端。

## Prefill 和 Decode 为什么表现不同

LLM 推理分为两个形状差异很大的阶段。

**Prefill** 一次处理提示词中的许多 Query，$Q、K、V$ 都较长，矩阵规模足以发挥 Tensor Core 与 tiling 的效率，避免 $N^2$ 中间张量也非常重要。

**Decode** 每轮通常只产生一个新 Query，却要读取全部历史 KV Cache。单请求的 $Q$ 很短，问题更偏向内存带宽和并行度不足。Serving 引擎会通过 continuous batching，把多个请求组织在一起，并可能选择 paged、split-KV 等专用 decode kernel。

所以，“FlashAttention 在训练中很快”不能直接推出“单请求 decode 也按相同比例加速”。应分别测量 prefill latency、time per output token 和端到端吞吐。

## 在 PyTorch 中使用和确认后端

通常优先使用 PyTorch 的 scaled dot-product attention（SDPA），让框架根据设备、dtype、形状和 mask 自动分派：

```python
import torch
import torch.nn.functional as F

q = torch.randn(2, 16, 1024, 64, device="cuda", dtype=torch.float16)
k = torch.randn_like(q)
v = torch.randn_like(q)

out = F.scaled_dot_product_attention(
    q, k, v,
    is_causal=True,
    dropout_p=0.0,
)
```

调试时可以强制使用特定后端。如果输入不受支持，PyTorch 会给出无法运行的原因，而不是悄悄拿一次成功结果当作证明：

```python
from torch.nn.attention import SDPBackend, sdpa_kernel

with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    out = F.scaled_dot_product_attention(
        q, k, v, is_causal=True, dropout_p=0.0
    )
```

布尔 mask 的语义也要特别核对：不同 API 可能用 `True` 表示“允许参与”，也可能表示“需要屏蔽”。迁移代码时，mask 方向错误会产生数值正常、语义完全错误的结果。

## 怎样验证正确性与性能

先在小尺寸上用 FP32 数学实现作为 reference，再逐步覆盖真实形状。正确性矩阵至少包括：

- causal 与 non-causal；
- 定长与变长序列；
- MHA、GQA 和不同 head dimension；
- 极端 logits、全 mask 行和非对齐长度；
- forward、backward 与 dropout；
- FP16、BF16、FP8 等实际 dtype。

低精度 kernel 的运算顺序不同，不应要求逐 bit 相等。应事先定义最大绝对误差、相对误差或任务级容差，并检查 NaN/Inf。若训练要求可复现，还要核对 backward 是否使用非确定性的原子操作。

性能评测则需要预热、同步 CUDA，并报告输入形状：

```python
starter = torch.cuda.Event(enable_timing=True)
ender = torch.cuda.Event(enable_timing=True)

for _ in range(10):
    F.scaled_dot_product_attention(q, k, v, is_causal=True)
torch.cuda.synchronize()

starter.record()
for _ in range(100):
    F.scaled_dot_product_attention(q, k, v, is_causal=True)
ender.record()
torch.cuda.synchronize()

print("average ms:", starter.elapsed_time(ender) / 100)
```

只测一个 $N$ 不足以支持结论。至少扫描序列长度、batch、head 数、head dimension、dtype 和 mask；同时记录峰值显存，并确认没有因为不支持的 shape 回退到 math backend。最终还要回到模型端到端指标，因为 Attention 之外的 MLP、通信和 KV Cache 管理也可能成为瓶颈。

## 小结

FlashAttention 的精髓可以浓缩成一句话：**保留精确注意力的数学结果，重新安排计算与存储，让数据在片上停留更久。** 在线 Softmax 让分块计算成为可能，反向重计算避免保存二次规模的中间张量，而后续版本则不断把这套算法映射到新的 GPU 流水线。

理解这层区别后，就能避开两个常见判断错误：不要把显存复杂度的改善写成算术复杂度的下降，也不要脱离具体 GPU、输入形状和执行阶段讨论“哪个版本最快”。

## 参考资料

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)
- [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608)
- [FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling](https://arxiv.org/abs/2603.05451)
- [FlashAttention 官方仓库](https://github.com/Dao-AILab/flash-attention)
- [PyTorch scaled_dot_product_attention 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention)
