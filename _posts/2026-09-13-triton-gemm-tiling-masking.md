---
layout: post
title: "Triton GEMM：从一个输出 Tile 到地址、边界与数据复用"
subtitle: "沿一段分块矩阵乘推导 program、stride 和 mask，再解释累加精度与调优为什么必须一起考虑"
date: 2026-09-13
last_modified_at: 2026-09-13
author: iStar
catalog: true
series: gpu-runtime-precision
series_order: 15
technology_year: 2019
mathjax: true
tags: [GPU优化, 工程实践]
---

矩阵乘的数学公式只有一个求和，但性能差别往往藏在公式没有写出的事情里：一个输入元素被加载几次，同一份数据能服务多少乘加，每个计算块占多少资源，以及不能整除的尾部究竟算了什么。

Triton 让程序围绕一块张量描述这些工作，再由编译器把块内操作映射到 GPU 线程与硬件指令。这里讨论的是 GPU 编程语言 Triton，不是同名的推理服务器。其 [2019 年 MAPL 论文](https://research.ibm.com/publications/triton-an-intermediate-language-and-compiler-for-tiled-neural-network-computations)提出了面向 tiled computation 的语言与编译器；本文年份指这项代表工作，不表示当时的语法与现在相同。

下面只处理一个明确算子：$$C=AB$$，其中 $$A\in\mathbb R^{M\times K}$$、$$B\in\mathbb R^{K\times N}$$，没有 bias、activation、批维或量化。示例限定同一 CUDA 设备上的二维 FP16 输入、FP32 累加和 FP16 输出，接口以 Triton v3.1.0 为基准。本文做地址与代数推导，未实际编译执行 GPU kernel，也不声称得到任何超过库实现的性能。

## 为什么输出维度用来并行，K 维留在累加器里

先展开一个输出元素：

$$
C_{mn}=\sum_{k=0}^{K-1}A_{mk}B_{kn}
$$

不同 $$m,n$$ 对应不同输出，彼此没有写入依赖，可以交给不同 program。沿 $$K$$ 的乘积却共同属于同一个输出，必须以某种方式归约。最直接的分块方案是：每个 program 负责 $$B_M\times B_N$$ 个输出，持有同形状 accumulator，沿 $$K$$ 每次推进 $$B_K$$。

设第 $$q$$ 轮的输入块为 $$A_q\in\mathbb R^{B_M\times B_K}$$ 和 $$B_q\in\mathbb R^{B_K\times B_N}$$，那么 program 做的是：

$$
C_{tile}\leftarrow 0,
\qquad
C_{tile}\leftarrow C_{tile}+A_qB_q
\quad\text{for each }q
$$

这里的 program 是一块张量工作的实例，不是一条 CUDA thread，也不是独占一个 SM。普通单 CTA 配置中，块内操作由多个线程协作；线程怎样持有元素、怎样把数据送进矩阵乘指令，需要编译器继续完成。原论文的 [§3 与 §5.2](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf)分别讨论块级表达与分层映射，不能把一段高层 `tl.dot` 当成具体寄存器布局。

输出网格为：

$$
G_M=\left\lceil\frac M{B_M}\right\rceil,
\qquad
G_N=\left\lceil\frac N{B_N}\right\rceil
$$

先采用简单的一维编号。对于 $$p\in[0,G_MG_N)$$：

$$
p_m=\left\lfloor\frac p{G_N}\right\rfloor,
\qquad
p_n=p\bmod G_N
$$

Program 的行区间从 $$p_mB_M$$ 开始，列区间从 $$p_nB_N$$ 开始。每个合法输出只属于一个 program，所以最终写回不需要原子累加；沿 $$K$$ 的所有部分和已经在该 program 内合并。

可以用一个循环不变量核对这段算法：完成前 q 轮之后，累加器只包含这些轮次覆盖的合法 K 索引贡献，既不重复，也不遗漏。下一轮加载新的 K 区间并把越界位置补零，于是不变量继续成立；最后一轮结束时恰好覆盖完整求和范围。输出 ownership 与这个归约不变量一起，构成了分块矩阵乘的正确性骨架，之后调整编号顺序或流水参数也不应破坏它们。

## Shape 说明矩阵含义，stride 才决定地址

假设 $$s_{am},s_{ak}$$ 是 A 两个维度的 stride，$$s_{bk},s_{bn}$$ 是 B 的 stride。对于 typed pointer，以下偏移以元素为单位，不是字节：

$$
\operatorname{addr}(A_{mk})=A_{ptr}+m s_{am}+k s_{ak}
$$

$$
\operatorname{addr}(B_{kn})=B_{ptr}+k s_{bk}+n s_{bn}
$$

输出若新分配为连续的 $$M\times N$$ 矩阵，则地址为 $$C_{ptr}+mN+n$$。如果输出也允许非连续布局，就必须同样传入其 stride，不能只给输入保留布局信息。

为什么不总用 `m * K + k`？看一块顺序存储的数据：

```text
X = [[10, 11, 12],
     [20, 21, 22]]

B = X.T
shape(B)  = (3, 2)
stride(B) = (1, 3)
```

`B[1, 0]` 的正确偏移是 `1 * 1 + 0 * 3 = 1`，值为 11。若看见 shape 是 `(3, 2)`，就擅自使用连续 stride `(2, 1)`，偏移变成 2，读到的却是 12。形状检查完全通过，访问也未必越界，结果仍然错误。

分块指针只是把同一地址式广播到二维。令 `rm` 长度为 $$B_M$$、`rn` 长度为 $$B_N$$，当前 `rk` 长度为 $$B_K$$：

```text
A 地址：rm[:, None] * stride_am + rk[None, :] * stride_ak
       (BM, 1)                    (1, BK)
       └────────────────→ 广播成 (BM, BK)

B 地址：rk[:, None] * stride_bk + rn[None, :] * stride_bn
       (BK, 1)                    (1, BN)
       └────────────────→ 广播成 (BK, BN)
```

先核对广播后的维度，再核对每个轴使用了哪个 stride，通常比盯着完整 kernel 更容易发现错误。这种写法允许普通转置和正 stride 切片，不等于所有布局都具有相同的访存效率。地址正确之后，还要看实际连续访问方向、合并访存和编译器布局选择。

## 三个不能整除的维度，为什么需要不同边界处理

取一个贯穿全文的形状：

$$
M=130,\quad N=70,\quad K=65,
\qquad
B_M=64,\quad B_N=32,\quad B_K=32
$$

输出网格是 $$3\times3$$，共有 9 个 program，每个 program 沿 K 执行 3 轮。右下角 program 的行索引是 `128..191`，列索引是 `64..95`，但真正有效的输出只有 2 行、6 列，共 12 个。最后一轮 K 索引是 `64..95`，其中只有 64 合法。

M、N 边界决定哪些输出存在；K 边界决定一个合法输出的求和应该包含哪些项。显式 mask 版本遵守三条规则：

- 读 A 时同时检查 `rm < M` 和 `rk < K`，无效位置填零。
- 读 B 时同时检查 `rk < K` 和 `rn < N`，无效位置填零。
- 写 C 时检查 `rm < M` 和 `rn < N`，不写不存在的输出。

只在写回处加 mask，无法保护前面已经发生的越界 load。也不能写成 `tl.where(mask, tl.load(ptr), 0)` 来期待惰性求值：Triton v3.1.0 的 [`where` 接口](https://github.com/triton-lang/triton/blob/v3.1.0/python/triton/language/core.py)明确说明两侧都会求值。访问保护必须交给 `tl.load` 本身的 `mask`。

这里的零填充是块内逻辑值，不要求先把整个输入矩阵复制到一个更大的、尺寸整齐的张量。后者会额外增加分配、拷贝和峰值显存，还可能掩盖 kernel 没有正确处理真实边界的问题。本文始终读取原始张量的合法位置；掩码之外生成零参与局部运算，输出仍然只分配真实形状。

K 维填零的依据是乘加中的零贡献，不能随意改成取模。用一个纯数学小例说明：

$$
[1,2,3]\cdot[4,5,6]=32
$$

概念上按长度 2 分块，第二块应为 `[3, 0]` 乘 `[6, 0]`。若把索引 `[2, 3]` 对 K=3 取模，变成 `[2, 0]`，第二块就额外加上 $$1\times4$$，结果成为 36。这是解释求和边界的手算例，不是可以直接交给 `tl.dot` 的合法硬件 tile 配置。

[v3.1.0 官方矩阵乘教程](https://github.com/triton-lang/triton/blob/v3.1.0/python/tutorials/03-matrix-multiplication.py)采用了另一种 M/N 处理：无效行列通过取模重复加载合法元素，最后不写对应无效输出。它们不会进入其他合法输出，因而可以这样处理；同一手法不能照搬到参与合法输出归约的 K 维。本文用三个维度都显式 mask 的版本，让正确性条件更直接可见。

## 把地址与边界装进一个短 kernel

下面让 `rk` 从 `0..BK-1` 开始，每轮增加 `BK`，并从输入基址重新构造指针。另一种实现可以保持局部索引不变，直接每轮给指针累加 `BK * stride`；两种写法任选其一，不要既推进全局 K 索引，又额外推进同一基址，否则 K 偏移会被计算两遍。

索引的整数宽度也是地址合同的一部分。`tl.arange` 默认生成 int32 索引；即使每个维度本身不大，`行号 * stride` 仍可能超过有符号 32 位范围。下面在乘法之前把 program 编号与索引转成 int64，不能等乘法溢出后再转换结果。比如 `65536 * 65536` 的正确元素偏移是 $$2^{32}$$，32 位回绕却会变成 0，行列 mask 无法识别这种已经算错的地址。64 位索引仍要求输入 view 合法、地址计算可表示且 launch grid 符合设备限制，不是任意巨大张量的安全承诺。

```python
import torch
import triton
import triton.language as tl


@triton.jit
def _gemm(
    A, B, C,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    SAM: tl.constexpr, SAK: tl.constexpr,
    SBK: tl.constexpr, SBN: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    grid_n = tl.cdiv(N, BN)
    pid_m = pid // grid_n
    pid_n = pid % grid_n

    rm = pid_m * BM + tl.arange(0, BM).to(tl.int64)
    rn = pid_n * BN + tl.arange(0, BN).to(tl.int64)
    rk = tl.arange(0, BK).to(tl.int64)
    acc = tl.zeros((BM, BN), dtype=tl.float32)

    for _ in range(tl.cdiv(K, BK)):
        a_ptrs = A + rm[:, None] * SAM + rk[None, :] * SAK
        b_ptrs = B + rk[:, None] * SBK + rn[None, :] * SBN
        a = tl.load(
            a_ptrs,
            mask=(rm[:, None] < M) & (rk[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(rk[:, None] < K) & (rn[None, :] < N),
            other=0.0,
        )
        acc = tl.dot(a, b, acc)
        rk += BK

    out = acc.to(tl.float16)
    c_ptrs = C + rm[:, None] * N + rn[None, :]
    tl.store(c_ptrs, out, mask=(rm[:, None] < M) & (rn[None, :] < N))


def gemm(a, b):
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("expected two matrices")
    if not a.is_cuda or not b.is_cuda or a.device != b.device:
        raise ValueError("inputs must share one CUDA device")
    if a.dtype != torch.float16 or b.dtype != torch.float16:
        raise TypeError("this example only accepts FP16 inputs")
    m, k = a.shape
    kb, n = b.shape
    if k != kb or min(m, n, k) <= 0:
        raise ValueError("incompatible or empty shape")
    if min(a.stride() + b.stride()) <= 0:
        raise ValueError("this example requires positive strides")

    # 调用方保证输入为合法的非重叠 view；输出不与输入共享 storage。
    c = torch.empty((m, n), device=a.device, dtype=torch.float16)
    grid = (triton.cdiv(m, 64) * triton.cdiv(n, 32),)
    with torch.cuda.device(a.device):
        _gemm[grid](
            a, b, c, m, n, k,
            a.stride(0), a.stride(1), b.stride(0), b.stride(1),
            BM=64, BN=32, BK=32, num_warps=4, num_stages=3,
        )
    return c
```

这段代码是固定版本接口下的解释性实现，不是官方教程的逐字复现，也没有自动求导 backward。空矩阵、零 stride 广播、其他 dtype 与 backend 均不在其合同内。选择 64、32、32 和对应执行配置，只是为了让前面的形状例子能逐行对应，不代表这些参数最快。

为什么不让多个 program 共用输出 buffer 的同一个区域？因为此处采用普通覆盖写回：每个结果只能有一个写入者。若为了减少分配把 C 与输入共享存储，先完成的 program 可能覆盖其他 program 尚未读取的数据，公式本身就失去依据。输出新分配不仅方便调用，也是在代码边界上排除了这一类跨块读写冲突。

代码还明确了三个精度位置：load 后的矩阵输入是 FP16，`acc` 是 FP32，store 前转换成 FP16。FP32 累加不会恢复输入已经丢掉的位，最终转换又可能舍入甚至溢出。若需要 FP32 输出，必须同时修改输出分配和最终转换；不能只把 wrapper 中的 dtype 改掉，就假设 kernel 已经保留高精度结果。[v3.1.0 的 `dot` 语义实现](https://github.com/triton-lang/triton/blob/v3.1.0/python/triton/language/semantic.py)也限定了块维度与 dtype 组合，手算中的任意小矩阵不能直接当作合法 tile。

## Tile 变大带来复用，也带来更重的累加器

看一个没有尾部的 tile，只统计一轮 K 的逻辑输入载荷，假设 A、B 每元素都占 $$b$$ 字节：

$$
Q_{stage}=b(B_MB_K+B_KB_N),
\qquad
F_{stage}=2B_MB_NB_K
$$

这里一次乘加按 2 FLOP 计数。于是：

$$
I_{stage}=\frac{F_{stage}}{Q_{stage}}
=\frac{2B_MB_N}{b(B_M+B_N)}
$$

一个 A 元素可以参与 $$B_N$$ 个输出列，一个 B 元素可以参与 $$B_M$$ 个输出行，这就是扩大 M/N tile 的复用来源。对于 FP16 方形 tile，令 $$B_M=B_N=T,b=2$$，得到 $$I_{stage}=T/2$$：64×64 对应 32 FLOP/byte，128×128 对应 64 FLOP/byte。

但 $$B_K$$ 在这个比值里相消了。把 K 块加倍，会同时增加本轮加载量和乘加量，不会自动提高同一模型中的复用比例。它仍可能减少循环轮数、改变流水线效率和指令安排，只是收益来自另一条机制。

同时，FP32 accumulator 的逻辑数据量是 $$4B_MB_N$$ bytes。64×64 对应 16 KiB，128×128 对应 64 KiB。这里不是“每线程需要多少寄存器”的结论：数据由线程协作持有，实际 live range、布局、分配和 spill 由编译结果决定。但逻辑规模的增长已经说明，大 tile 的资源代价不是免费的。

大 tile 还减少输出 program 数量。如果矩阵本来不大，网格可能不足以填满设备；每个 program 占用更多资源，也可能减少同时驻留的工作。相反，小 tile 虽然 program 更多，却会降低块内复用并增加调度开销。NVIDIA 的[矩阵乘性能说明](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)把这类 tile 效率、并行度和 wave 尾部现象分开讨论，不能只按单个 tile 的 FLOP/byte 选择参数。

尤其要注意，$$I_{stage}$$ 是 program 逻辑输入载荷模型，不是完整 GEMM 的 HBM 算术强度。它没有计入输出，也没有描述缓存命中、重复 program load 与实际内存 transaction。若要使用 [Roofline]({% post_url 2026-09-09-roofline-gpu-performance-model %})，必须先说明流量穿过哪一级存储边界，不能把这条局部公式直接贴到 HBM 横轴上。

## Mask 保住了正确性，却不保证没有尾部计算成本

再回到 130×70×65 的例子。真正有用的矩阵乘工作量是：

$$
F_{useful}=2\times130\times70\times65=1{,}183{,}000\ \mathrm{FLOP}
$$

若边缘 program 和最后一轮仍执行完整规则块 dot，其块级工作模型为：

$$
F_{tile}
=2\times(3\times64)\times(3\times32)\times(3\times32)
=3{,}538{,}944\ \mathrm{FLOP}
$$

有用工作仅占约 33.4%。这是规则 tile 算法的计数，不是硬件指令计数器结果，也不能据此严格预测时间增加为三倍。它揭示的是：mask 可以抑制无效访存与写回，却不意味着 Tensor Core 自动把一块矩阵乘压缩成任意形状的稀疏运算。

同一个配置在整齐的大矩阵上复用很好，在一批接近边界的小矩阵上却可能浪费很多。判断“这个 tile 更快”时必须带上实际 $$M,N,K$$ 分布；只测一个恰好整除的方阵，通常看不见生产形状里的尾部。

矩阵形状不对称时，也不必坚持方形输出块。宽矩阵和高矩阵能够沿不同方向提供大量独立输出，输入连续方向又可能不同。合理候选应同时考虑哪侧输入值得复用、哪个方向有足够工作以及哪个维度尾部严重，而不是把方形 tile 的简洁公式当作所有形状的最优解。

## 相邻 program 还可能借用同一份缓存数据

同一输出行块、不同输出列块会重复需要 A；同一输出列块、不同输出行块会重复需要 B。单个 program 内已经做了一层复用，相邻 program 之间还有另一层机会：如果共享数据仍在 L2，后来的逻辑 load 不必都重新访问 HBM。

官方教程的 grouped program mapping 会把一组相关输出块安排到邻近编号，增加共享数据在缓存中被及时复用的机会。它改变的是编号到输出块的映射，不改变矩阵乘公式，也不让两个 program 共享同一份 accumulator。

编号相近不等于严格按编号执行，不保证落在同一个 SM，也不保证缓存永不驱逐。因此 grouped ordering 是性能提示，不能用它构建跨 program 的先后依赖。这也是为什么评估时要观察整段 kernel，而不是把局部载荷量直接加总后宣称测到了 HBM 流量。

另一种增加并行度的方法是 split-K：让多个 program 计算同一输出块的不同 K 区间，再合并 partial sums。它与本文单 program 内的 $$B_K$$ 循环不同，会引入额外归约、workspace 或原子操作，并改变浮点求和顺序。增加 $$B_K$$ 不等于增加 K 维并行度；如果输出网格太小，需要先识别这个问题，再判断 split-K 的合并成本是否值得支付。

## 调优对象不是一个神奇的块大小

`BM/BN/BK` 描述张量分块，`num_warps` 描述 program 的线程协作配置，`num_stages` 影响流水调度。它们不是彼此的别名：同一个 tile 使用不同执行配置，可能得到不同的资源占用、访存重叠和代码布局。候选参数还必须满足目标硬件与 `tl.dot` 的合法性条件，不能任意枚举整数。

调优缓存也需要跟输入合同一致。官方示例常以 M/N/K 作为显式 key；Triton v3.1.0 的 [autotuner 实现](https://github.com/triton-lang/triton/blob/v3.1.0/python/triton/runtime/autotuner.py)还会把张量 dtype 纳入缓存键。但这不意味着所有 stride 和 layout 特征都会自动参与性能选择，也不意味着 JIT 特化与 autotune key 是同一个机制。

前面的转置矩阵说明，同样 shape 可以拥有不同连续访问方向。若某个配置只对连续输入调好，再用于实际大量转置或切片输入，数值可以正确，性能却可能回退。应围绕真实布局分组选择或验证配置，而不是只扩大一个形状列表。

首次编译和 autotune 也不是免费发生。Autotune 会重复运行候选 kernel；本文每个 program 覆盖写回自己的输出，因此重复运行不会累加旧 C，但带原子累加或其他副作用的 kernel 必须额外处理状态重置。服务冷启动和稳态运行是不同口径，不应把它们混成一个“平均耗时”。

## 在开始比较速度之前，先让错误有机会暴露

正确性测试应同时覆盖整除与非整除的三个维度、转置和正 stride 切片、较小 K、负数及不同幅值。不要只用全 1 输入：错误 stride 和重复加载可能恰好读到相同数字，测试会放过完全错误的地址式。

可以先对已经转换成 FP16 的相同输入，用更高精度求和建立数学参考，再单独观察最终 FP16 输出舍入。这样能区分输入量化、累积差异和输出转换；容差仍要与 K、数值尺度和后端相匹配，不能用一个很宽的阈值掩盖边界错误。CPU 地址模拟或手算通过，也不证明 Triton 编译、GPU 访存安全和浮点执行已经正确。

本文的配套复算可在仓库根目录运行：

```sh
node scripts/reviews/sep13-triton-gemm-examples.mjs
```

它只使用 Node.js 标准库，检查三轴 mask、输出唯一写入、非连续 stride、32 位偏移回绕反例和本文的运算量数字。矩阵乘参考使用 CPU 数值模型，不执行上面的 Python 代码，不验证 Triton 编译、Tensor Core 累加或 GPU 性能。

计时时先明确是否包含输出分配。上面的 wrapper 每次创建 C，若要测纯 kernel，应预先分配相同输出 buffer，并在预热、编译及调优结束后计时。GPU 执行是异步的，需要使用正确的设备计时或同步边界；输入转换、传输、冷缓存和编译时间是否包含，都应分别说明。

吞吐按真正有用的 $$2MNK$$ 除以时间报告，而不是用 padding 后的块级运算量把数字抬高，同时保留实际 latency。对比库实现时，输入布局、输出 dtype、计算精度与 epilogue 必须一致。若自定义 kernel 融合了 bias 或 activation，应该比较同等完整结果的端到端路径，而不是把多做的功能藏进一个不对等的基线。

Triton 的价值不在于手写 GEMM 就一定快过厂商库，而在于能够围绕真实形状和融合需求表达一条可控的数据路径。读懂这条路径，先要知道每个 program 拥有哪些输出、每个指针指向什么、每个尾块贡献哪些项；这些都成立以后，tile、缓存与流水线的优化才有可解释的对象。
