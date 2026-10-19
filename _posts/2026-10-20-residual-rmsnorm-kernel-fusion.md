---
layout: post
title: "Residual + RMSNorm：一次融合里的数值契约与数据流"
subtitle: "从两个必须保留的输出理解残差归一化，再沿 FP32 中间值、行归约与共享内存分析融合到底省了什么"
date: 2026-10-20
last_modified_at: 2026-10-20
author: iStar
catalog: true
series: gpu-runtime-precision
series_order: 18
technology_year: 2019
mathjax: true
tags: [GPU优化, FlashInfer, LLM推理]
---

“把 residual add 和 RMSNorm 融合，就能少读写一次中间张量。”这句话解释了一个方向，却漏掉了实现中最重要的问题：哪一个中间值可以不保存，哪一个残差结果仍然要留给后面的网络？如果输入是 FP16，加法结果应该先舍入回 FP16，还是保留 FP32 参加归一化？

这两个问题分别决定数据流与数值语义。只看 kernel 数量从两个变成一个，无法确认优化前后算的是不是同一个函数。

下面固定一个局部算子：把一份输入与残差相加，一路保存新的残差，另一路做 RMSNorm。RMSNorm 的年份锚定 [2019 年原论文](https://arxiv.org/abs/1910.07467)；具体融合实现则以 FlashInfer v0.7.2 的固定提交 `d84335d8` 为例，不把算法发表时间当作该实现的发布时间。

## 相加之后，为什么有两个输出

设输入 $$X$$ 与已有残差 $$R$$ 都具有形状 $$M\times D$$，每行对应一组需要归一化的隐藏状态。增益参数 $$\gamma$$ 的形状为 $$D$$，在所有行共享。

先暂时忽略舍入，局部计算是：

$$
Z=X+R,
\qquad
R_{new}=Z,
\qquad
Y=\operatorname{RMSNorm}(Z;\gamma,\epsilon)
$$

它不是只有一个输出 $$Y$$。新的残差 $$R_{new}$$ 还会沿 skip 路径继续使用；$$Y$$ 则可以作为下一个算子的输入。二者具有不同用途，不能因为融合就删除持久残差，也不能把归一化后的 $$Y$$ 当作未归一化残差保存。

```text
X ----+
      +--> Z = X + R --+--> R_new  保留在残差路径
R ----+               |
                      +--> RMSNorm --> Y  送给下一算子
```

这里描述的是一段已经存在的局部计算，不是在建议把模型从 Pre-Norm 改成 Post-Norm。归一化放在网络的什么位置由模型结构决定，kernel fusion 的任务是在保留该结构语义的条件下实现这段计算。

FlashInfer 的 `fused_add_rmsnorm(input, residual, weight, eps)` 使用原地接口：`residual` 被覆盖为 $$R_{new}$$，`input` 被覆盖为 $$Y$$，函数返回 `None`，`weight` 不被修改。[Python 接口的 mutation 声明与分派](https://github.com/flashinfer-ai/flashinfer/blob/d84335d8c54e0c8f9f7e22faa15c7e34a58abf94/flashinfer/norm/__init__.py#L267-L305)明确标出了这两个被修改的参数。

因此，调用以后还需要旧 $$X$$ 或旧 $$R$$ 的代码，必须先安排生命周期或另存副本；不能把这个接口当作普通的“传入两个张量，返回一个新张量”。

## RMSNorm 归一化的是均方根，不是方差

对第 $$m$$ 行，定义：

$$
q_m=\frac1D\sum_{j=0}^{D-1}z_{mj}^{2},
\qquad
\rho_m=\frac1{\sqrt{q_m+\epsilon}},
\qquad
y_{mj}=z_{mj}\rho_m\gamma_j
$$

每行有一个共同的缩放因子 $$\rho_m$$，再对各列乘上自己的 $$\gamma_j$$。这里没有减去行均值；$$q_m$$ 是均方值，不是去均值后的方差。变量名即使写成 `variance`，也不能改变它实际计算的统计量。

取 $$z=[1,1]$$。它的均方值是 1，方差却是 0。RMSNorm 不会把这一行中心化成全零；LayerNorm 的中心化步骤则会产生不同结果。因此，RMSNorm 不是少写一行代码但数学上完全相同的 LayerNorm。

在 $$\gamma_j=1$$ 且忽略浮点误差时，归一化输出的均方值为：

$$
\frac1D\sum_j y_j^2=\frac{q}{q+\epsilon}
$$

只要 $$\epsilon>0$$ 且 $$q$$ 有限，它就不严格等于 1。再乘非统一的 $$\gamma$$ 后，输出更没有“每行单位方差”或“每行均方根必为 1”的保证。

同理，理想公式在 $$\epsilon=0$$ 时，对非零向量的正比例缩放具有不变性；加入固定 $$\epsilon$$ 后，缩放输入会改变它与 $$q$$ 的相对大小，只能在 $$q$$ 远大于 $$\epsilon$$ 等条件下近似保留这种关系。不能把理想分析直接扩成所有浮点实现上的严格性质。

$$\epsilon$$ 放在根号内，`sqrt(q + eps)` 与 `sqrt(q) + eps` 不是同一个函数。它还必须显式匹配：FlashInfer 此 API 的默认值是 `1e-6`，而 [PyTorch 2.8 RMSNorm](https://docs.pytorch.org/docs/2.8/generated/torch.nn.RMSNorm.html) 的 `eps=None` 会使用输入 dtype 对应的 `finfo.eps`。拿两个默认参数直接比较，出现差异时未必是 kernel 算错了。

## FP16 存下来的残差，不一定是归一化使用的值

现在把存储 dtype 记为 $$T$$，取 FP16 或 BF16。本文所选 FlashInfer 实现先把输入和残差提升到 FP32，再完成相加。用类型转换明确写出：

$$
Z_{32}=\operatorname{FP32}(X_T)+\operatorname{FP32}(R_T)
$$

接下来出现两条不同路径：

$$
R_{new,T}=\operatorname{cast}_T(Z_{32})
$$

$$
Y_T=\operatorname{cast}_T\left(
Z_{32}\cdot
\operatorname{rsqrt}\left(\frac1D\sum_j Z_{32,j}^{2}+\epsilon\right)
\cdot\operatorname{FP32}(\gamma_T)
\right)
$$

保存残差时会舍入回 $$T$$，但参加归约和归一化的是尚未舍回 $$T$$ 的 $$Z_{32}$$。这并不等价于先执行低精度 `residual.add_(input)`，再把已经舍入的 `residual` 提升到 FP32 做 norm。

这个差别可以很小，却是确定的语义差别。令输入和旧残差为：

$$
X=[1,2],\quad R=[2^{-11},0],\quad\gamma=[1,1],\quad\epsilon=10^{-6}
$$

binary16 在 1 附近的相邻间距为 $$2^{-10}$$。加法得到的第一个 FP32 值 $$1+2^{-11}$$ 正好在两个 binary16 数之间，round-to-nearest-ties-to-even 会把它舍回 1。于是两条路径保存的新残差相同，归一化输入却不同：

| 计算路径 | 保存的 FP16 残差 | 最终 FP16 输出 |
| --- | --- | --- |
| 使用未舍回 FP16 的 $$Z_{32}$$ 做 norm | `[1.0, 2.0]` | `[0.6328125, 1.2646484375]` |
| 先将和舍回 FP16，再做 norm | `[1.0, 2.0]` | `[0.63232421875, 1.2646484375]` |

只比较 `residual` 会漏掉这个区别，只在最终 $$Y$$ 上使用很宽的容差也可能把契约差异掩盖掉。[该版本的官方参考函数](https://github.com/flashinfer-ai/flashinfer/blob/d84335d8c54e0c8f9f7e22faa15c7e34a58abf94/tests/utils/test_norm.py#L85-L94)同样保留 FP32 相加结果用于归一化，因此这是应当先确定的行为，不是随意选择的测试细节。

下面只用 Python 标准库演示这两条舍入路径。`struct` 的 `e` 与 `f` 分别用来舍入到 binary16 与 binary32；归约采用顺序加法，`rsqrt` 用宿主的开方和倒数近似模拟，再舍入 FP32。它用于解释舍入位置，不模拟 GPU 的树形归约、FMA 或快速 `rsqrt`，也不承诺与设备输出逐 bit 相同。

<!-- example: residual-rmsnorm-rounding -->
```python
import math
import struct


def f16(value):
    return struct.unpack("e", struct.pack("e", value))[0]


def f32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def finite_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("finite numerical inputs are required")
    try:
        value = float(value)
    except OverflowError as exc:
        raise ValueError("input exceeds floating-point range") from exc
    if not math.isfinite(value):
        raise ValueError("finite numerical inputs are required")
    return value


def residual_rmsnorm_fp16(x, residual, weight, eps=1e-6,
                         round_before_norm=False):
    if not x or len(x) != len(residual) or len(x) != len(weight):
        raise ValueError("one nonempty row and matching weight are required")
    if not isinstance(round_before_norm, bool):
        raise ValueError("round_before_norm must be a boolean")
    eps32 = f32(finite_number(eps))
    if not math.isfinite(eps32) or eps32 <= 0:
        raise ValueError("eps must stay finite and positive in FP32")

    # Inputs first acquire the FP16 storage dtype.
    x16, r16, w16 = [[f16(finite_number(v)) for v in row]
                     for row in (x, residual, weight)]
    z32 = [f32(f32(a) + f32(b)) for a, b in zip(x16, r16)]
    r_new = [f16(z) for z in z32]
    norm_input = r_new if round_before_norm else z32
    sum_sq = 0.0
    for z in norm_input:
        sum_sq = f32(sum_sq + f32(z * z))
    mean_sq = f32(sum_sq / len(norm_input))
    inverse_rms = f32(1.0 / math.sqrt(f32(mean_sq + eps32)))
    y = [f16(f32(f32(z * inverse_rms) * f32(w)))
         for z, w in zip(norm_input, w16)]
    return y, r_new


x, residual, weight = [1, 2], [2**-11, 0], [1, 1]
fused_y, fused_r = residual_rmsnorm_fp16(x, residual, weight)
rounded_y, rounded_r = residual_rmsnorm_fp16(
    x, residual, weight, round_before_norm=True)
assert fused_r == rounded_r == [1.0, 2.0]
assert fused_y == [0.6328125, 1.2646484375]
assert rounded_y == [0.63232421875, 1.2646484375]
print(fused_y, rounded_y)
```

示例限定有限、可表示的 binary16 输入与结果，$$\epsilon$$ 转成 FP32 后也必须仍为有限正数；超出表示范围时拒绝计算，不处理溢出后的 Inf/NaN 传播。真实实现的验证还需要独立覆盖这些条件及其预期行为，不能把上述小模型当作完整 IEEE 浮点模拟器。

## 一行归约要先跨过所有列，才能输出任何一列

加法是逐元素独立的，RMSNorm 不是。每一列的输出都需要同一个行统计量 $$q_m$$，而它依赖这一行的全部 $$D$$ 列。这形成了融合实现的主要障碍：读到一个元素、完成相加以后，还不能立即产生最终归一化输出。

一种实现让一个 block 负责一行，把列分给 block 内线程。各线程读自己的元素、计算局部平方和；warp 内交换局部和，再跨 warp 合并，得到这一行的总和。完成同步以后，所有参与线程才能使用同一个 $$\rho_m$$ 写输出。

这段等待期间，已经相加的 FP32 值放在哪里？有三种直接的选择：留在寄存器、存入 shared memory，或在得到统计量后从某种存储重新读取或重算。它们对应不同资源成本，并不是写出一个 `sum` 就自动解决了数据保存。

若采用补齐到 $$B\ge D$$ 列的教学实现，越界列可以以零参加平方和，但均值分母必须仍是实际 $$D$$：

$$
q=\frac{\sum_{j=0}^{B-1}\widetilde z_j^2}{D},
\qquad
\widetilde z_j=0\quad\text{for }j\ge D
$$

例如 $$z=[1,2,3]$$ 补齐为 `[1,2,3,0]`，平方和仍是 14，正确均方值却是 $$14/3$$，不是 $$14/4$$。Mask 保护了越界访问，也必须保护统计意义；“补零后按 tile 长度求平均”会把边界处理写成另一个算子。

## 沿 FlashInfer 的 CUDA 路径追踪中间值

固定版本中的 [CUDA kernel 与 launch 配置](https://github.com/flashinfer-ai/flashinfer/blob/d84335d8c54e0c8f9f7e22faa15c7e34a58abf94/include/flashinfer/norm.cuh#L416-L542)给出了一个具体实现：每个 block 对应一行，线程向量化读取输入与残差，提升到 FP32 相加；保存低精度新残差，同时把未舍回低精度的和存入 FP32 shared memory。

随后完成 warp 内与跨 warp 的归约，把总平方和除以实际 $$D$$，加上 $$\epsilon$$ 并求倒数平方根。最后从 shared memory 再读 FP32 的和，读增益，计算并把结果写回 `input`。

```text
global X, R --load--> FP32 sum --+--> cast T --> global R_new
                               |
                               +--> FP32 shared memory --read--+
                               |                               |
                               +--> square/reduce --> rho ------+--> gamma --> cast T --> global Y
```

所以这条路径减少的是跨 kernel 的全局中间张量，不是“中间值全部常驻寄存器，也没有任何额外访存”。Shared memory 的写入、读取、同步和容量都是真实成本，只是它们与把完整 FP32 中间张量放到 global memory 是不同的数据流。

该 CUDA 配置按元素大小与 $$D$$ 选择向量长度，并为归约暂存与 FP32 行数据申请 shared memory。取 $$D=4096$$、两字节元素，向量长度为 8，512 个线程组成 16 个 warps；动态 shared memory 为：

$$
Q_{smem,capacity}
=4\left(4\left\lceil\frac{16}{4}\right\rceil+4096\right)
=16{,}448\ \mathrm{bytes}
$$

这是每 block 的容量需求，不是整次算子的 HBM 流量。它影响同一 SM 能驻留多少 block；真实驻留数还受到线程数、寄存器与设备限制共同约束，不能只用 shared memory 容量做除法就宣布 occupancy。

公共 API 也不保证永远走这条 CUDA 路径。[同一版本的分派逻辑](https://github.com/flashinfer-ai/flashinfer/blob/d84335d8c54e0c8f9f7e22faa15c7e34a58abf94/flashinfer/norm/__init__.py#L58-L101)会考虑 CuTe DSL 是否可用、设备架构与显式配置；[CuTe 实现](https://github.com/flashinfer-ai/flashinfer/blob/d84335d8c54e0c8f9f7e22faa15c7e34a58abf94/flashinfer/norm/kernels/fused_add_rmsnorm.py#L381-L411)保留 FP32 的 `h` 参与归约和最终计算，采用不同的寄存器、加载和行调度组织。读懂一个后端，不能据此断言本机公共调用实际用了哪个 kernel。

## 融合减少多少字节，必须先固定同一数值契约

设总元素数 $$N=MD$$，存储 dtype 每个元素占 $$s$$ 字节。先建立一个逻辑 global-memory 账本：按每行读取一次增益计数，不考虑跨行缓存复用、实际事务粒度、spill 和其他实现开销。

理想融合实现需要读取 $$X,R,\gamma$$，写回 $$R_{new},Y$$，共：

$$
Q_{fused,logical}=3Ns+2Ns=5Ns
$$

要与之比较，一个保持前面 FP32 中间值契约的理想拆分方案，可以让第一个 kernel 写低精度 $$R_{new}$$，并额外写完整 FP32 $$Z$$；第二个 kernel 读取 FP32 $$Z$$ 做 norm。两阶段的逻辑流量是：

$$
Q_{split,logical}
=\underbrace{(2Ns+Ns+4N)}_{\text{add and two outputs}}
+\underbrace{(4N+Ns+Ns)}_{\text{norm}}
=5Ns+8N
$$

融合省去的是 FP32 $$Z$$ 的一次 global 写与一次 global 读，总共 $$8N$$ 字节。它没有省去后续还需要的新残差写回。

取 $$M=128,D=4096,s=2$$：

| 同一 FP32 中间值契约下的逻辑账本 | 字节数 |
| --- | ---: |
| 拆分实现 $$5Ns+8N$$ | 9,437,184 |
| 融合实现 $$5Ns$$ | 5,242,880 |
| 省去的 global 中间张量读写 $$8N$$ | 4,194,304，即 4 MiB |

CUDA 源码把 FP32 $$Z$$ 转移到 shared memory 后，行数据仍有约 $$8N$$ 字节的逻辑 shared 写读，另有归约暂存。不能把 global 流量下降理解成这些字节在所有存储层级中消失。增益还可能命中缓存，所以表中的逻辑字节也不是实际测得的 HBM 字节。

如果基线是低精度 `add_` 后再做 norm，可以写出另一个 $$6Ns\to5Ns$$ 的简化账本：它主要省去 norm 对低精度残差的一次重读。但这个基线先把和舍回 $$T$$，与本文固定融合路径的数值契约不同。两种比较都可以研究，前提是说清楚语义；不能拿后一种字节数，配前一种正确性主张，组合成一个貌似严谨的加速结论。

更不能从字节比例直接算实际 speedup。多个 kernel 的提交成本、缓存命中、同步和资源占用都可能改变时间，最终仍要在相同输入与输出要求下测量。

## 数据少搬了，也可能被资源和形状限制

当行宽较大，保存 FP32 中间值需要更多寄存器或 shared memory。更高的资源占用可能减少同时驻留的 block，或者让编译器产生 spill。这样，即使少了 global 中间张量，新的执行瓶颈也可能来自并行度、寄存器依赖或局部存储。

行数也很重要。一个 block 一行的方案在 $$M$$ 很小时，只产生很少的 block，设备可能没有足够工作；把每行处理得再精巧，也不会自动增加全设备并行度。反过来，很多很短的行可能适合一 CTA 处理多行，避免让每行独占过多线程。不同后端针对这些形状采用不同策略，是合理的工程选择。

此外，融合后的调度粒度改变了：原来两个算子之间可被编译器或运行时观察的边界消失，输入输出的生命周期与依赖也随之变化。算子局部更快，是否缩短请求关键路径，应结合 [Systems 到 Compute 的剖析方法]({% post_url 2026-10-20-gpu-profiling-nsight-systems-compute %})重新检查，而不是只看 kernel 数量减少。

## 正确性检查要覆盖两个输出和原地行为

输入形状相同，不代表输入布局一定受支持。真实 kernel 需要核对 dtype、device、隐藏维连续性、行 stride、向量化对齐和地址范围；一个后端接受的 view，不能不加验证地套给另一个后端。为支持 stride 而使用宽索引，也要在乘法之前保证地址计算不会溢出。

`input`、`residual` 与增益应使用满足 API 前提的独立存储，不应互相 alias 或部分重叠。这个限制不是看见“原地”就能推翻的：原地只是允许各输出覆盖自己指定的输入位置，并不意味着任意输入之间的重叠都安全。CUDA 路径的 `__restrict__` 指针与读写顺序，也不提供这种保证。

构造测试时，分别比较 $$R_{new}$$ 与 $$Y$$，保留调用前的输入快照，并检查增益没有被修改。行宽应包含 1、3、111 这样的边界以及 4096 等常用宽度；数据应包含全零、近零、正负混合、不同量级和前面的舍入反例。$$\epsilon$$、增益定义及输出 dtype 必须在参考和被测实现之间显式统一。

FP32 归约降低了低精度累计误差，但不同归约顺序与快速倒数平方根仍可产生差异。容差应结合 dtype、范围和实际算法来设定；“不是逐 bit 相等”不自动意味着错误，“一个宽容差通过”也不证明两个实现具有同一数值契约。

性能测试还有一个容易漏掉的状态问题：原地调用会同时改变输入与残差。反复对同一组 buffer 调用，下一轮已经不是上一轮的输入。若目标是固定分布的独立样本，应使用预先准备的 buffer 组，或在明确范围外重置数据；若把重置包含进计时，则应把指标称为包含重置的完整成本。无论选择哪种口径，都不能悄悄让数值状态在迭代中累积，再把它当作固定 workload 的稳态比较。

## 融合的核心是保留语义，改变中间数据的去向

Residual + RMSNorm 只有几行公式，却把 kernel 优化中几种典型约束集中在一起：双输出要求保留残差，FP32 与低精度之间的转换决定舍入位置，行归约决定同步边界，中间值的保存方式决定资源与流量。

理清这些关系以后，融合收益才有具体含义：在保留两个输出与既定数值契约的前提下，把一份跨 kernel 的 global 中间结果转移到 kernel 内部，再判断节省是否超过新增的资源和同步成本。它是一项需要形状、后端与测量共同验证的实现选择，不是一条“融合必然更快”的规则。

本文的 [标准库复算脚本](https://github.com/Charlestar/Charlestar.github.io/blob/master/scripts/reviews/residual-rmsnorm-examples.mjs)提取并执行 FP16 舍入示例，检查归一化性质、真实行宽分母及逻辑字节账本。没有执行 FlashInfer GPU kernel，也没有据此给出设备速度、occupancy 或端到端加速结果。

## 参考资料

- [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
- [FlashInfer fused_add_rmsnorm API](https://docs.flashinfer.ai/generated/flashinfer.norm.fused_add_rmsnorm.html)
- [FlashInfer 固定版本 CUDA 融合实现](https://github.com/flashinfer-ai/flashinfer/blob/d84335d8c54e0c8f9f7e22faa15c7e34a58abf94/include/flashinfer/norm.cuh#L416-L542)
- [FlashInfer 固定版本 Python API 与后端分派](https://github.com/flashinfer-ai/flashinfer/blob/d84335d8c54e0c8f9f7e22faa15c7e34a58abf94/flashinfer/norm/__init__.py)
- [FlashInfer 固定版本 CuTe 融合实现](https://github.com/flashinfer-ai/flashinfer/blob/d84335d8c54e0c8f9f7e22faa15c7e34a58abf94/flashinfer/norm/kernels/fused_add_rmsnorm.py)
- [FlashInfer 固定版本数值参考](https://github.com/flashinfer-ai/flashinfer/blob/d84335d8c54e0c8f9f7e22faa15c7e34a58abf94/tests/utils/test_norm.py#L85-L94)
- [PyTorch 2.8 RMSNorm](https://docs.pytorch.org/docs/2.8/generated/torch.nn.RMSNorm.html)
