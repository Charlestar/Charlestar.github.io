---
layout: post
title: "GPTQ：量化一个权重之后怎样补偿输出误差"
subtitle: "从输入相关的二次重建目标推到约束更新、Cholesky 与低位权重执行"
date: 2026-10-15
last_modified_at: 2026-10-15
author: iStar
catalog: true
series: model-quantization
series_order: 40
technology_year: 2022
mathjax: true
tags: [模型量化, 机器学习, LLM推理]
---

把每个权重舍入到最近格点，确实能让各个权重自己的误差尽量小。但线性层使用的是输入与权重的乘积：两个大小相同的权重误差，如果落在不同输入方向上，造成的输出变化可能完全不同。

GPTQ 从这里出发。它不只问“这个权重离哪个格点最近”，还问“既然这个权重已经必须移到格点上，其余尚未量化的权重能否一起移动，让线性层输出少变一点”。这种补偿发生在离线量化阶段，不需要部署时为每次请求重新调整权重。

理解这个方法，需要把三件事分开：线性层的连续重建目标、逐步做出的离散量化决定，以及把计算高效组织起来的工程算法。某一步的连续补偿最优，不等于最后所有权重都落在格点上时已经全局最优。

## Calibration 让量化看见输入方向

仍用系列统一的方向：$$X\in\mathbb{R}^{M\times K}$$，$$W\in\mathbb{R}^{K\times N}$$，输出 $$Y=XW$$。把 calibration 样本经过当前模型时收集到的线性层输入排成 $$X$$，量化后的权重记为 $$\hat{W}$$。考虑层重建目标：

$$
\mathcal{L}(\hat{W})=\lVert X\hat{W}-XW\rVert_F^2
$$

Frobenius 范数的平方就是把全部样本、全部输出坐标的平方误差相加。这个目标不要求任务标签，也不需要对完整语言模型做梯度反传，但它明确依赖 calibration 数据，不能称作 data-free。[GPTQ 论文 §3](https://arxiv.org/html/2210.17323v2#S3)

对一个输出通道的权重列 $$w\in\mathbb{R}^{K}$$，令 $$\Delta w=\hat{w}-w$$，则：

$$
\mathcal{L}(\hat{w})
=\lVert X\Delta w\rVert_2^2
=\Delta w^\top X^\top X\Delta w
=\frac12\Delta w^\top H\Delta w,
\qquad H=2X^\top X
$$

$$H$$ 是这个线性层重建代理的精确 Hessian，而不是完整语言模型任务 loss 的 Hessian。逐层代理、有限 calibration 以及之后的离散决策，才是方法不能等同于完整任务精确优化的原因。

如果 $$X^\top X$$ 是单位矩阵的常数倍，各个方向彼此无关且权重相同，目标退化成普通权重平方误差的倍数。一般情况下，对角元素反映不同输入方向的能量，非对角元素则刻画它们的关联；补偿正是利用了这种关联。

所有输出通道使用相同的输入 $$X$$，因此共享这份 $$K\times K$$ 统计。实现中可以逐批累积 Gram 矩阵，不必为量化一直保存全部 activation。官方实现的 `add_batch` 还带有整体归一化：它在展平 token 之前累计批次中的样本数，并非直接以展平后的 token 总数 $$M$$ 作分母。因此，不能把其缓冲区逐位等同于未归一化的 $$2X^\top X$$；对固定数据，整体正比例缩放不会改变这里无 damping 目标的最优解。[固定版本 add_batch](https://github.com/IST-DASLab/gptq/blob/2d65066eeb06a5c9ff5184d8cebdf33662c67faf/gptq.py#L32)

不保存全部 activation，并不意味着统计没有成本。这份 Gram 矩阵本身就有 $$K^2$$ 个元素，后续求逆和分解也有开销；GPTQ 的工程改造是在这个基础上减少重复工作，而不是消除二阶统计。

## 两个权重怎样互相补偿

先用一组可手算的数据：

$$
X=\begin{bmatrix}1&1/2\\0&\sqrt{3}/2\end{bmatrix},
\qquad
w=\begin{bmatrix}0.49\\0.49\end{bmatrix}
$$

于是：

$$
H=2X^\top X=\begin{bmatrix}2&1\\1&2\end{bmatrix}
$$

为了只观察补偿，固定重建网格为 $$\{-8,-7,\ldots,7\}$$，最近偶数舍入，步长为 1。这 16 个值可以用 4-bit code 表达；若用 unsigned codes $$0\ldots15$$，可选择 scale 1、zero point 8，重建值仍然相同。这里的网格是人为固定的教学条件，不是为这两个小权重自动 min-max 后得到的网格，也不规定实际 GPTQ artifact 的 packing 必须采用哪种有符号表示。

普通 round-to-nearest，简称 RTN，会得到 $$[0,0]^\top$$，层输出平方误差为 0.7203。

现在先把第一维 0.49 量化成 0。第一维变化为 $$\delta_0=-0.49$$，第二维暂时允许连续变化。设其变化为 $$t$$，则新增误差目标为：

$$
\frac12
\begin{bmatrix}-0.49&t\end{bmatrix}
H
\begin{bmatrix}-0.49\\t\end{bmatrix}
=0.2401-0.49t+t^2
$$

对 $$t$$ 求导，得到 $$-0.49+2t=0$$，所以最优补偿为 $$t=0.245$$。第二个权重从 0.49 移到 0.735；它此时还不在量化格点上，这是算法保留的连续自由度，不是最终的低位权重。

最后再把 0.735 舍入成 1，得到 $$[0,1]^\top$$：

| 状态 | 权重向量 | 层输出平方误差 |
| --- | --- | ---: |
| 原始权重 | $$[0.49,0.49]^\top$$ | 0 |
| 第一维已固定，第二维连续补偿 | $$[0,0.735]^\top$$ | 0.180075 |
| 两维均已量化 | $$[0,1]^\top$$ | 0.2503 |
| 两维直接 RTN | $$[0,0]^\top$$ | 0.7203 |

最终权重的 MSE 反而从 RTN 的 0.2401 增加到 0.2501，输出误差却明显下降。GPTQ 试图保护的是输入作用后的层输出，不是保证每个权重都更接近原值。

## 一次约束补偿的公式从哪里来

设尚未量化的坐标集合为 $$F$$，对应 Hessian 主子矩阵为 $$H_F=H[F,F]$$。先假设它正定，并令：

$$
P_F=H_F^{-1},
\qquad r=w_q-Q(w_q)
$$

$$Q(w_q)$$ 是本轮选中的量化格点，$$r$$ 是“当前连续权重减去格点”。为了将坐标 $$q$$ 固定到该值，必须有 $$\delta_q=-r$$；其他自由坐标仍可移动。

在已经对剩余连续变量完成上轮最优补偿的条件下，其自由方向梯度为零。因此，本轮增加的目标正好是 $$\tfrac12\delta_F^\top H_F\delta_F$$，没有遗漏未处理的一次项。引入拉格朗日乘子 $$\nu$$：

$$
\min_{\delta_F}\ \frac12\delta_F^\top H_F\delta_F,
\qquad e_q^\top\delta_F=-r
$$

一阶条件给出：

$$
H_F\delta_F+\nu e_q=0,
\qquad
\delta_F=-\nu P_Fe_q
$$

再代入被固定坐标的约束，得到：

$$
\nu=\frac{r}{(P_F)_{qq}},
\qquad
\boxed{\delta_F=-\frac{r}{(P_F)_{qq}}(P_F)_{:,q}}
$$

该更新包含 $$q$$ 本身：它的变化恰好是 $$-r$$。实现若已经显式写入量化值，就不能再把同一项重复补偿一次。最小新增目标则为 $$r^2/[2(P_F)_{qq}]$$，这也解释了为什么误差大小与输入敏感性需要一起考虑。[论文 §3 式 2](https://arxiv.org/html/2210.17323v2#S3)

在前面的例子里，$$P=H^{-1}=\begin{bmatrix}2/3&-1/3\\-1/3&2/3\end{bmatrix}$$，第一步更新正是 $$[-0.49,0.245]^\top$$。非对角项决定了补偿方向；不能仅凭“减去误差”猜测其他权重该增还是该减。

## 为什么 Cholesky Factor 不能当作原始逆矩阵

逐步固定坐标后，需要的是当前自由集的逆 $$P_F=(H[F,F])^{-1}$$。它通常不等于最初完整逆矩阵 $$P=H^{-1}$$ 的同名主子块。

GPTQ 的 Cholesky 形式先计算一个上三角矩阵 $$U$$，满足：

$$
P=U^\top U
$$

随后按照固定顺序逐行读取 $$U$$。在坐标 $$q$$ 成为剩余自由集的第一项时，有：

$$
(P_F)_{qq}=U_{qq}^2,
\qquad
(P_F)_{qj}=U_{qq}U_{qj},\quad j\in F
$$

所以刚才的补偿可以改写为：

$$
\delta_j=-\frac{r}{U_{qq}}U_{qj},\qquad j\in F
$$

这个关系并非把逆矩阵随意“开平方”。观察一次分块就能理解。若当前上三角因子为：

$$
U=\begin{bmatrix}u&v^\top\\0&C\end{bmatrix},
\qquad
P=\begin{bmatrix}u^2&uv^\top\\uv&vv^\top+C^\top C\end{bmatrix}
$$

消去第一维后，剩余 Hessian 的逆等于逆矩阵上的 Schur 更新：

$$
P_{\mathrm{next}}
=P_{rr}-\frac{P_{rq}P_{qr}}{P_{qq}}
=C^\top C
$$

每次读取下一行 $$U$$，就对应下一次自由集的逆信息。前面二维例子的 $$U_{11}^2=1/2$$，对应剩余单维 Hessian 2 的倒数；它不是完整逆矩阵中原来的 $$P_{11}=2/3$$。

官方源码在 Cholesky 之后仍把这个因子命名为 `Hinv`。读代码时应根据运算确认含义：此处变量装的是 $$U$$，不能在同一个更新里让分母取 $$U_{qq}$$，分子却误用原始 $$P$$ 的一行。[固定版本 gptq.py](https://github.com/IST-DASLab/gptq/blob/2d65066eeb06a5c9ff5184d8cebdf33662c67faf/gptq.py#L98)

## GPTQ 怎样把逐个补偿变成可扩展算法

如果每个输出通道都自己决定下一次量化哪个坐标，不同通道会留下不同自由集，无法有效共享逆矩阵更新。GPTQ 让各输出通道采用同一个输入坐标顺序，因此同一步的 $$H_F$$ 相同，可以一起处理。

这里要留意布局：论文与常见 `nn.Linear.weight` 使用 $$N\times K$$，称为逐列处理；本文使用 $$K\times N$$，同一操作就是固定一行输入坐标上的所有输出权重。轴的名字改变了，所对应的输入维度没有改变。

另一个瓶颈是反复访问大矩阵。当前坐标的舍入，只需要先收到此前对它的补偿；还没轮到的远端坐标，不必每一步都立即更新。于是 GPTQ 在一个 block 内完成局部顺序处理，记录误差，到 block 结束时批量更新后面的权重。

在本文布局中，block 内每个输入坐标保存一行归一化误差：

$$
E_{j,:}=\frac{W_{j,:}-Q(W_{j,:})}{U_{jj}}
$$

block 内及时应用 $$W_{k,:}\leftarrow W_{k,:}-U_{jk}E_{j,:}$$；处理完一个 block $$B$$ 后，远端部分可以用矩阵乘更新：

$$
W_{\mathrm{tail},:}
\leftarrow W_{\mathrm{tail},:}-U_{B,\mathrm{tail}}^\top E_{B,:}
$$

这不是降低理论乘加次数的魔法，而是把许多低算术强度、反复读写的操作，重新组织成更适合硬件利用的矩阵工作。共享顺序、lazy block updates、Cholesky 重写共同构成了 GPTQ 的可扩展性；“对 Hessian 求逆然后 round”遗漏了主要部分。[论文 §4、Algorithm 1](https://arxiv.org/html/2210.17323v2#S4)

## Damping 与两个不同的分组参数

有限样本不保证 $$X$$ 满列秩。例如只观察到输入 $$[1,1]$$ 时，两个输入方向无法区分，$$X^\top X$$ 为 rank 1，不能直接求逆。加入正对角项：

$$
H_\lambda=H+\lambda I,\qquad\lambda>0
$$

能够在精确算术下让这个半正定矩阵变成正定，但也把目标改成：

$$
\mathcal{L}_\lambda(\Delta w)
=\mathcal{L}(\Delta w)+\frac{\lambda}{2}\lVert\Delta w\rVert_2^2
$$

因此，damping 是带有目标变化的稳定化，而不是证明原问题天然可逆。固定官方实现使用 `percdamp * mean(diag(H))` 作为对角增量，默认比例为 0.01。校准归一化若使 $$H$$ 整体缩放，这条按对角均值定义的 damping 也随之缩放。

实现还会检测统计中对角为零的 dead 坐标，调整其 Hessian 对角，并将对应权重置零。它说明该 calibration 统计没有体现这个方向的输入能量，不说明真实部署里这个方向永远无用；数据覆盖不足时，不能把“dead”当成模型语义事实。[固定版本 dead、damping 处理](https://github.com/IST-DASLab/gptq/blob/2d65066eeb06a5c9ff5184d8cebdf33662c67faf/gptq.py#L77)

此外，两个经常都设为 128 的参数并不是同一件事：

| 参数 | 决定什么 | 是否直接定义量化格点共享范围 |
| --- | --- | --- |
| `blocksize` | 累积多少输入坐标的补偿，再批量更新远端 | 否 |
| `groupsize` | 多少权重共享一组 scale、zero point 等参数 | 是 |

前者主要组织计算，后者改变表示与误差。相同数值只是配置巧合。组参数还可以在补偿过程中重新估计，或者先固定 static groups；有些更新会改变后续格点，因此不能把固定网格推导直接当作所有 exporter 的逐位等价证明。

该实现的 `act-order` 按 Hessian 对角从大到小重排输入坐标，是处理顺序的启发式。它与 true-sequential 等选项属于官方后续实现增强，不应倒写成初版算法不可缺少的定义。量化后必须恢复权重坐标，并让 group 索引、排列与 packing 规则一致，不能只恢复矩阵形状。[固定版本实现](https://github.com/IST-DASLab/gptq/blob/2d65066eeb06a5c9ff5184d8cebdf33662c67faf/gptq.py)、[对应 README](https://github.com/IST-DASLab/gptq/blob/2d65066eeb06a5c9ff5184d8cebdf33662c67faf/README.md)

## 用标准库核对二维补偿

第一段代码只处理有限的二维对称正定 Hessian，使用直接逆公式和 Cholesky 因子演示顺序更新。若输入不能满足这些条件，或求逆、分解、补偿出现非有限值或非零中间量下溢为零，就拒绝继续；即使最终结果在数学上可表示，也不尝试修复中间动态范围。它不是适用于真实大矩阵或病态输入的数值线性代数库。

<!-- example: gptq-two-dimensional-compensation -->
```python
from math import sqrt, isfinite, isclose, fsum


def quantize_int4(value):
    if not isfinite(value):
        raise ValueError("finite value required")
    code = min(15, max(0, round(value) + 8))
    return float(code - 8)  # fixed scale=1, zero=8


def gptq_two(h, weights):
    if len(h) != 2 or any(len(row) != 2 for row in h) or len(weights) != 2:
        raise ValueError("this reference requires dimension two")
    if any(not isfinite(v) for row in h for v in row):
        raise ValueError("finite Hessian required")
    if any(not isfinite(v) for v in weights) or h[0][1] != h[1][0]:
        raise ValueError("finite weights and symmetric Hessian required")
    a, c, b = h[0][0], h[0][1], h[1][1]
    diagonal_product, cross_square = a * b, c * c
    if (not isfinite(diagonal_product) or not isfinite(cross_square)
            or (a != 0 and b != 0 and diagonal_product == 0)
            or (c != 0 and cross_square == 0)):
        raise ValueError("determinant outside supported floating range")
    determinant = diagonal_product - cross_square
    if a <= 0 or b <= 0 or determinant <= 0 or not isfinite(determinant):
        raise ValueError("positive definite Hessian required")
    p = [[b / determinant, -c / determinant],
         [-c / determinant, a / determinant]]
    if (any(not isfinite(v) for row in p for v in row)
            or p[0][0] <= 0 or p[1][1] <= 0
            or (c != 0 and p[0][1] == 0)):
        raise ValueError("inverse outside supported floating range")
    u00 = sqrt(p[0][0])
    u01 = p[0][1] / u00
    if not isfinite(u01) or (p[0][1] != 0 and u01 == 0):
        raise ValueError("factor outside supported floating range")
    factor_square = u01 * u01
    if not isfinite(factor_square) or (u01 != 0 and factor_square == 0):
        raise ValueError("factor outside supported floating range")
    tail = p[1][1] - factor_square
    if tail <= 0 or not isfinite(tail):
        raise ValueError("factor outside supported floating range")
    u = [[u00, u01], [0.0, sqrt(tail)]]
    work, result, trace = list(weights), [], []
    for j in range(2):
        chosen = quantize_int4(work[j])
        residual = work[j] - chosen
        error = residual / u[j][j]
        if not isfinite(error) or (residual != 0 and error == 0):
            raise ValueError("normalized error outside supported floating range")
        for k in range(j, 2):
            change = error * u[j][k]
            if (not isfinite(change)
                    or (error != 0 and u[j][k] != 0 and change == 0)):
                raise ValueError("compensation outside supported floating range")
            work[k] -= change
        work[j] = chosen
        if any(not isfinite(v) for v in work):
            raise ValueError("nonfinite compensation")
        result.append(chosen)
        trace.append(work.copy())
    return result, trace, p, u


x = [[1.0, 0.5], [0.0, sqrt(3) / 2]]
w = [0.49, 0.49]
h = [[2.0, 1.0], [1.0, 2.0]]


def sse(candidate):
    return fsum(fsum(v * (q - old) for v, q, old in zip(row, candidate, w)) ** 2
                for row in x)


q, trace, p, u = gptq_two(h, w)
delta = [-0.49 * p[i][0] / p[0][0] for i in range(2)]
assert isclose(delta[1], 0.245)
assert isclose(trace[0][1], 0.735)
assert isclose(sse(trace[0]), 0.180075)
assert q == [0.0, 1.0]
assert isclose(sse(q), 0.2503)
assert isclose(sse([quantize_int4(v) for v in w]), 0.7203)
assert isclose(u[1][1] ** 2, 0.5)
assert not isclose(u[1][1] ** 2, p[1][1])
print({"trace": trace, "quantized": q, "SSE": sse(q)})
```

这段程序同时核对了原始逆形式与因子形式：第一步用 $$P$$ 直接算出 0.245 的补偿，顺序算法则读取 $$U$$，得到相同轨迹。二维限制是刻意的，读者可以逐项检查每个数，而不是把一个大型求逆调用当作解释。

## 局部连续最优为什么不保证最终优于 RTN

考虑另一组固定数据：

$$
X=\begin{bmatrix}1&1&1\\0&1&1\\0&0&1\end{bmatrix},
\qquad
w=\begin{bmatrix}-0.4\\-0.4\\0.2\end{bmatrix}
$$

仍使用相同的 16 格网格，不启用 damping、act-order 或 grid 重估，按 $$0\to1\to2$$ 顺序处理。精确连续补偿的轨迹为：

| 固定步骤 | 补偿后的权重 | 当前层 SSE |
| --- | --- | ---: |
| 第一维固定到 0 | $$[0,-0.6,0.2]$$ | 0.08 |
| 第二维固定到 -1 | $$[0,-1,7/15]$$ | 约 0.186667 |
| 第三维固定到 0 | $$[0,-1,0]$$ | 0.84 |

最后的 0.84 大于 RTN 的 0.44。穷举全部 $$16^3=4096$$ 个格点向量，唯一全局最优则是 $$[-1,0,0]$$，SSE 为 0.24。

原因在于第一步就把第一维固定到了 0，后面不再回头考虑 -1。每一步都在“先前决定已固定、剩余坐标暂时连续”的条件下最优，最终问题却要求剩余坐标也离散化。中间的较低误差不是最终误差的上界。

下面独立运行这个固定三维反例。$$U$$ 在代码中直接给出，并核对 $$HP=I$$ 和 $$P=U^\top U$$；穷举也使用相同网格，而不是偷偷换更宽范围。

<!-- example: gptq-discrete-counterexample -->
```python
from itertools import product
from math import sqrt, fsum, isclose

x = [[1.0, 1.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]]
w = [-0.4, -0.4, 0.2]
h = [[2.0, 2.0, 2.0], [2.0, 4.0, 4.0], [2.0, 4.0, 6.0]]
p = [[1.0, -0.5, 0.0], [-0.5, 1.0, -0.5], [0.0, -0.5, 0.5]]
u = [[1.0, -0.5, 0.0], [0.0, sqrt(3) / 2, -1 / sqrt(3)],
     [0.0, 0.0, 1 / sqrt(6)]]

for i in range(3):
    for j in range(3):
        assert isclose(fsum(h[i][k] * p[k][j] for k in range(3)),
                       float(i == j), abs_tol=1e-12)
        assert isclose(fsum(u[k][i] * u[k][j] for k in range(3)),
                       p[i][j], abs_tol=1e-12)


def quantize(v):
    return float(min(15, max(0, round(v) + 8)) - 8)


def sse(candidate):
    return fsum(fsum(v * (q - old) for v, q, old in zip(row, candidate, w)) ** 2
                for row in x)


work, trace = w.copy(), []
for j in range(3):
    chosen = quantize(work[j])
    error = (work[j] - chosen) / u[j][j]
    for k in range(j, 3):
        work[k] -= error * u[j][k]
    work[j] = chosen
    trace.append(work.copy())

rtn = [quantize(v) for v in w]
scores = [(sse(candidate), candidate) for candidate in product(range(-8, 8), repeat=3)]
best_loss = min(loss for loss, _ in scores)
best = [candidate for loss, candidate in scores if isclose(loss, best_loss)]
assert len(scores) == 4096
assert work == [0.0, -1.0, 0.0]
assert isclose(trace[1][2], 7 / 15)
assert isclose(sse(work), 0.84)
assert isclose(sse(rtn), 0.44)
assert isclose(best_loss, 0.24) and best == [(-1, 0, 0)]
print({"trace": trace, "sequential_SSE": sse(work),
       "RTN_SSE": sse(rtn), "global_best": best, "best_SSE": best_loss})
```

这个反例不是对论文模型实验的否定，而是明确理论保证的范围：顺序连续补偿既不保证整层离散全局最优，也不保证任意输入、任意固定顺序的最终误差都优于 RTN。真实模型上的收益仍需从对应模型、calibration 和任务评测得到。

## 落在格点上与真正存成低位权重是两步

官方 `gptq.py` 完成补偿后，将量化重建值写回原权重 dtype。此时权重虽然落在有限格点上，张量仍可能是 FP16；单看这一步，不会自动获得低位存储收益。

真正压缩还需要把值变回整数 code，再 packing 到较宽的容器中，同时保存 scale、zero point 和布局。固定官方版本的 `quant.py` 中，`quantize` 返回的是 $$s(q-z)$$ 浮点重建值；`Quant3Linear.pack` 才把 3-bit code 装进整数 buffer。两者处理的是不同阶段。[固定版本 quant.py](https://github.com/IST-DASLab/gptq/blob/2d65066eeb06a5c9ff5184d8cebdf33662c67faf/quant.py)

推理时还要让 kernel 理解这份 packing。GPTQ 本体是 weight-only 方法：activation 不因为权重量化而自动变成 INT4。论文中的 batch-one 矩阵向量路径动态解包、恢复低位权重，再参与计算；主要利用较少的权重流量，不等于所有乘法都换成原生 INT4 Tensor Core 指令。固定代码里的 `Quant3Linear` 也明确限制为单 token 路径，不能拿它代表所有后来的 GPTQ backend。[论文 §5 Practical Speedups、§6](https://arxiv.org/html/2210.17323v2#S5)、[Quant3Linear.forward](https://github.com/IST-DASLab/gptq/blob/2d65066eeb06a5c9ff5184d8cebdf33662c67faf/quant.py#L186)

于是，评价一份“GPTQ 模型”至少需要分开三个事实：采用了什么补偿与分组规则，发布了什么实际编码，以及运行时怎样消费这些编码。层重建目标变小，只回答了第一个问题的一部分，不能代替 packing 正确性、端到端质量与性能。

## 从层重建走向实际模型

Calibration 决定了算法看见哪些输入方向。领域、上下文长度、聊天模板和模型中间状态变化，都会改变这份代理对部署输入的代表性。逐层量化时，后续层的输入还会受到前面已经量化的层影响，因此只对一份孤立、固定的高精度 activation 做拟合，也不等于已经覆盖整个误差传播过程。

论文中的流程会把当前输入再送过已经量化的 Transformer block，生成下一个 block 的 calibration 输入，而不是始终复用原始高精度模型的全部中间状态。这让后续统计看到前面 block 已经引入的误差，但仍不能代替目标任务的端到端评测。[论文 §5 Setup](https://arxiv.org/html/2210.17323v2#S5)

GPTQ 给出的核心工具是：利用输入相关的二阶结构，在逐个加入量化约束时重新分配误差。与 [LLM.int8()]({% post_url 2026-10-15-llm-int8-outlier-decomposition %}) 的运行时异常维拆分不同，这些补偿最终折叠进离线权重；与 [AWQ]({% post_url 2026-08-03-awq-activation-aware-weight-quantization %}) 的缩放保护不同，它直接求解每次约束后的连续补偿方向。

理解连续目标、离散决定和实现布局之间的边界之后，GPTQ 就不再是一句“用 Hessian 做 4-bit”。它是一套围绕层输出重建组织起来的误差补偿方法，而是否适合某个模型，仍需要用真实输入、实际 artifact 和目标 kernel 完成验证。

## 参考资料

- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers v2](https://arxiv.org/html/2210.17323v2)，§3–4、Algorithm 1、§5–6；[2022 年首次公开与版本历史](https://arxiv.org/abs/2210.17323)。
- [官方 gptq.py 固定版本](https://github.com/IST-DASLab/gptq/blob/2d65066eeb06a5c9ff5184d8cebdf33662c67faf/gptq.py)，commit `2d65066eeb06a5c9ff5184d8cebdf33662c67faf`，2023-07-11。
- [同版本 quant.py](https://github.com/IST-DASLab/gptq/blob/2d65066eeb06a5c9ff5184d8cebdf33662c67faf/quant.py) 与 [README](https://github.com/IST-DASLab/gptq/blob/2d65066eeb06a5c9ff5184d8cebdf33662c67faf/README.md)。
