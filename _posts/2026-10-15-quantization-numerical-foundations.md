---
layout: post
title: "模型量化的数值基础：Scale、Zero Point 与误差怎样进入计算"
subtitle: "从一条离散数轴推到整数点积，理解量化粒度、累加精度和 Fake Quant 的边界"
date: 2026-10-15
last_modified_at: 2026-10-15
author: iStar
catalog: true
series: model-quantization
series_order: 10
technology_year: 2017
mathjax: true
tags: [模型量化, 机器学习, 工程实践]
---

把一个权重从 16 bit 压到 4 bit，究竟改变了什么？最直接的答案是存储，但计算里还多了一件事：原来能够区分的数，现在可能落在同一个离散格点上。模型读取的已经不是原来的权重，而是由少量编码和一组缩放参数重建的近似值。

这也是模型量化与普通类型转换的区别。把浮点数直接转换成整数，通常不会保留这个张量的数值尺度；量化则定义一份编码与实数之间的对应关系。权重、运行时 activation 和 KV Cache 都可以采用量化，但它们的数据分布、更新时间和使用方式不同，不能仅凭“都是 4 bit”推断收益或误差相同。

我们先把均匀整数仿射量化讲清楚，再把它代入一次线性层。这个基础能够解释为什么 scale 的方向、zero point 的单位、分组轴和累加类型都会影响结果。FP8 等浮点编码和非均匀 codebook 则有各自的格点结构，不应直接套用全部整数结论。

## 一条离散数轴怎样表示实数

设输入为有限实数 $$x$$，整数编码范围为 $$[q_{\min},q_{\max}]$$。选择正 scale $$s>0$$ 和范围内的整数 zero point $$z$$，定义：

$$
q=\operatorname{clip}\left(\operatorname{RNE}(x/s)+z,
q_{\min},q_{\max}\right)
$$

$$
\hat{x}=s(q-z)
$$

这里 $$q$$ 是保存的 code，$$\hat{x}$$ 是重建值；RNE 表示 round to nearest, ties to even，也就是取最近整数，恰好居中时取偶数。clip 则把越界结果限制到编码端点。这个运算顺序与 [ONNX QuantizeLinear opset 13](https://onnx.ai/onnx/operators/onnx__QuantizeLinear.html#quantizelinear-13) 的整数定义一致。

scale 是相邻重建格点之间的距离。本文采用“反量化时乘 $$s$$”的方向；若某个实现保存的是 $$1/s$$，接入时必须换算，不能只看变量名也叫 `scale`。

以 $$s=0.25,z=0,q\in[-7,7]$$ 为例，可表示的实数为 $$-1.75,-1.50,\ldots,1.75$$：

| 原值 $$x$$ | 舍入后的 $$\operatorname{RNE}(x/s)$$ | 最终 code | 重建值 $$\hat{x}$$ |
| ---: | ---: | ---: | ---: |
| -0.625 | -2 | -2 | -0.50 |
| -0.375 | -2 | -2 | -0.50 |
| -0.125 | 0 | 0 | 0 |
| 0 | 0 | 0 | 0 |
| 0.375 | 2 | 2 | 0.50 |
| 1.60 | 6 | 6 | 1.50 |
| 2.40 | 10 | 7 | 1.75 |

前三个负数展示了 ties-to-even：$$-2.5\to-2$$、$$-1.5\to-2$$、$$-0.5\to0$$。最后一行则经历了饱和，无法再用原来的最近格点表示。

这里使用 15 个 code，是一种便于对称的 4-bit 方案，并没有用完 signed INT4 的全部 16 个编码。同样，常见对称 INT8 权重量化会使用 $$[-127,127]$$，但 INT8 类型本身的完整范围仍然是 $$[-128,127]$$。编码类型和量化 recipe 选择的有效范围应分开记录。

## Zero Point 的单位是编码而不是浮点数

从重建式可以直接得到：当 $$q=z$$ 时，$$\hat{x}=0$$。zero point 说明的是“哪个整数 code 对应实数零”，并不是要给原浮点值加上 $$z$$。

假设使用 UINT8，取 $$s=0.1,z=128$$。此时 $$-1,0,0.2,1$$ 分别编码成 $$118,128,130,138$$。整数 128 在这里表示零；若为了做零填充而直接写入 code 0，重建出来反而是 $$-12.8$$。Jacob 等人采用这类仿射表示的一个原因，就是让零值可以被精确编码，便于处理 padding。[Jacob 等人 §2.1](https://arxiv.org/html/1712.05877v1#S2.SS1)

也可以把重建式写成 $$\hat{x}=sq+b$$，但此时 $$b=-sz$$ 是实数域的加性偏置，单位与 $$x$$ 相同；$$z$$ 则是编码域的整数偏移。这两种表示可以换算，数值却通常不同。文件格式里只有一个名为 `offset` 的字段时，必须先确定它属于哪一域。

舍入的位置同样不能随意移动。以 $$s=1,z=1,x=0.5$$ 为例：

$$
\operatorname{RNE}(0.5)+1=1,
\qquad
\operatorname{RNE}(0.5+1)=2
$$

因为把奇数零点移到 round 内部，会改变 tie 两侧整数的奇偶性。因此，“零点是整数，所以放到 round 前后都一样”对 ties-to-even 并不成立。量化参数完全一样，表达式重排仍可能在边界产生不同 code。

## Scale 同时决定覆盖范围和舍入误差

对称量化常用一个绝对最大值 $$a=\max_i\lvert x_i\rvert$$ 确定 scale。若 code 使用 $$[-127,127]$$，且 $$a>0$$，则可取：

$$
s=\frac{a}{127}
$$

这样可以覆盖当前数据的极值，却把整个张量的分辨率都绑定到了最大幅度上。全零张量需要单独处理，例如约定 $$s=1,z=0$$ 并保存全零 code；不能用 $$a=0$$ 算出零 scale 后继续除法。

非对称量化则可以平移重建区间，让有限 code 更贴近偏向一侧的数据。实际 min-max 参数选择通常需要先把零纳入范围，再把零对应的位置调整到合法整数 code；经调整后的重建端点不必与原始最小值、最大值逐位一致。它不是一种能同时精确保留任意两个端点和实数零的承诺。

量化误差需要分两种情况理解。若 $$x$$ 落在可重建端点构成的区间内，最近格点重建满足：

$$
|\hat{x}-x|\le \frac{s}{2}
$$

这个半步长界描述的是舍入误差。输入超出区间后，这个界不再有保证；偏离端点足够远、舍入 code 也越界时，clip 会把它限制到端点，误差可能远大于 $$s/2$$。例如 $$s=1,q\in[-7,7],z=0$$ 时，100 被重建成 7，误差为 93。原值略微超出实数区间，不一定已经触发 code 的 clip：7.2 舍入后仍为 7，这个区别也会影响后面讨论的梯度 mask。

因此，覆盖所有极值未必让整体平方误差最小。构造 100 个 1 和一个 10，只使用 $$[-1,0,1]$$ 三个对称 code：按最大值取 $$s=10$$，100 个 1 全部变成 0，总平方误差为 100；若改取 $$s=1$$，前 100 个数无损，10 被截成 1，总平方误差为 81。适度 clipping 在这组分布上更合算，但换一组数据就可能相反。

这给 calibration 留下了实际问题：我们要用哪些样本、哪种统计和什么目标选择范围？一个 calibration 集上的较低 MSE，不会自动保证换领域、换长度后的 activation 仍落在同一范围，更不等于模型任务质量必然更好。

## 量化粒度决定谁共享同一把尺子

把权重写成 $$W\in\mathbb{R}^{K\times N}$$，输入写成 $$X\in\mathbb{R}^{M\times K}$$，则线性层输出为 $$Y=XW$$。$$K$$ 是输入与归约维，$$N$$ 是输出通道维。之后提到“通道”时都沿用这个约定。

per-tensor 让整个权重矩阵共享参数。per-output-channel 为每一列 $$W_{:,n}$$ 保存自己的 scale 和 zero point。per-group 则还可以沿每列的 $$K$$ 维分组，让同一输出通道里的不同输入片段分别选择范围。不同框架保存权重时可能转置矩阵，因而实际 axis 编号不能从“per-channel”这个名字推断。[LiteRT 的 per-axis 定义](https://developers.google.com/edge/litert/conversion/tensorflow/quantization/quantization_spec#per-axis_vs_per-tensor)

考虑：

$$
W=\begin{bmatrix}0.1&100\\0.2&80\end{bmatrix}
$$

采用 $$[-7,7]$$ 对称量化。per-tensor 的 scale 为 $$100/7$$，第一列两个小值都会被重建成零。若按输出通道分别量化，第一列改用 $$0.2/7$$，重建为约 $$[0.114286,0.2]^\top$$；第二列仍用 $$100/7$$。改善来自小值不再与 100 共用步长，而不是位宽发生了变化。

分组还会改变存储账单。假设 $$P=KN$$ 个权重都用 4-bit code，每个输出通道沿 $$K$$ 维按最多 $$G$$ 个数分组，每组保存一个 16-bit scale，不保存 zero point，也暂不计 packing padding 和对齐，则总位数为：

$$
B=4P+16N\left\lceil\frac{K}{G}\right\rceil
$$

当 $$K=128,N=1,G=32$$ 时，code 占 512 bit，四个 scale 再占 64 bit，合计 576 bit，即每个权重 4.5 bit。相对只计权重的 FP16 存储，压缩比是 $$16/4.5\approx3.56$$，而不是 4。尾组 padding、zero point、额外高精度层都会继续改变这个数字。

更细的粒度能让每组拥有更合适的数值范围，但也要加载更多参数，并适配 kernel 的布局与缩放位置。它提供了更多选择，并不意味着所有实现的速度、总存储或任务质量都随组变小而单调改善。

## 把量化公式代入一次整数点积

先假设一次点积的所有 $$K$$ 项共享输入 scale $$s_x$$、权重 scale $$s_w$$ 以及各自零点。把重建式代入，可以得到：

$$
\hat{y}
=s_xs_w\sum_{k=1}^{K}(q_{x,k}-z_x)(q_{w,k}-z_w)
$$

展开括号后，整数累加值为：

$$
\begin{aligned}
A={}&\sum_k q_{x,k}q_{w,k}
-z_w\sum_kq_{x,k}\\
&-z_x\sum_kq_{w,k}+Kz_xz_w
\end{aligned}
$$

原始 code 乘积的和只是第一项。如果把其余零点修正全部丢掉，即使两个张量的 scale 都正确，结果也会错误。对静态权重，部分列和可以预先计算；若权重选择 $$z_w=0$$，还可以去掉涉及动态输入之和的那一项，这也是对称权重方案便于实现的原因。[LiteRT 的零点展开](https://developers.google.com/edge/litert/conversion/tensorflow/quantization/quantization_spec#symmetric_vs_asymmetric)

完整算一遍：令 $$q_x=[3,4],z_x=2,s_x=0.5$$，$$q_w=[5,7],z_w=4,s_w=0.25$$。重建输入为 $$[0.5,1]$$，权重为 $$[0.25,0.75]$$，浮点点积为 $$0.875$$。整数路径中：

$$
A=43-4\times7-2\times12+2\times2\times4=7
$$

再乘 $$s_xs_w=0.125$$，同样得到 $$0.875$$。这证明的是两种方式对同一组重建值计算一致，并不证明重建值等于量化前的任意原始输入。

INT8 描述乘法操作数，不表示乘积和也只能保存在 INT8。常见路径使用 INT32 accumulator；若 bias 也要在整数域相加，在输入 scale 固定、权重 scale 按输出通道变化时，bias 的对应 scale 应为 $$s_xs_{w,n}$$，zero point 为零。[Jacob 等人 §2.4](https://arxiv.org/html/1712.05877v1#S2.SS4)

INT32 仍然有范围限制。长归约、极端 code、非零零点的修正项与 bias 都需要一起检查，不能只因每个操作数合法，就认定中间累加永不溢出。

如果输出还要存为整数，就需要根据输出 scale $$s_y$$ 和 zero point $$z_y$$ 再做 requantization，其高层数值关系可写成：

$$
q_y=\operatorname{clip}\left(
\operatorname{RNE}\left(\frac{s_xs_w}{s_y}A\right)+z_y,
q_{y,\min},q_{y,\max}\right)
$$

整数 kernel 可以用定点乘子、移位与相应舍入实现这一步。上式本身没有规定这些中间位宽和舍入细节，因此不能把普通 Python 浮点计算当成它们的 bit-exact 复现。

现在再看沿 $$K$$ 分组的情况。若权重 group 为 $$g(k)$$，则需要计算：

$$
\hat{y}=s_x\sum_gs_{w,g}
\sum_{k\in g}(q_{x,k}-z_x)(q_{w,k}-z_{w,g})
$$

不同 group 的 scale 在归约轴内变化，不能选一个 scale 乘在整次整数点积之外。kernel 必须以分组缩放、分组累加或其他等价方式处理。这正是“数学上增加一组 scale”与“硬件上免费增加一组 scale”之间的区别。

## 权重误差小不等于输出误差小

只量化权重时，定义 $$\Delta W=\hat{W}-W$$，输入保持不变，则：

$$
\Delta Y=\hat{Y}-Y=X\Delta W
$$

误差受到输入方向和幅度的共同影响。令 $$X=[1,100]$$，比较两种可能的权重扰动：

$$
\Delta W_A=\begin{bmatrix}0\\0.1\end{bmatrix},
\qquad
\Delta W_B=\begin{bmatrix}0.2\\0\end{bmatrix}
$$

第一种的权重 MSE 为 0.005，比第二种的 0.02 更低；但输出误差分别为 10 和 0.2。第一种更小的权重误差，恰好落在输入幅度更大的方向上，被放大得更明显。

因此，单看权重直方图或权重 MSE，不能充分判断线性层的输出是否保真。观察 activation、用 calibration 输入衡量层重建误差，都是在补上 $$X$$ 这部分信息。[AWQ]({% post_url 2026-08-03-awq-activation-aware-weight-quantization %}) 和 [SmoothQuant]({% post_url 2026-08-02-smoothquant-w8a8-activation-outliers %}) 便从不同角度利用了输入分布。

若输入和权重都量化，令 $$\hat{X}=X+\Delta X$$，则完整误差还包括：

$$
\hat{X}\hat{W}-XW
=\Delta XW+X\Delta W+\Delta X\Delta W
$$

分别优化两个张量的局部误差，不会自动让三项组合后的模型输出误差最小。继续经过非线性层、残差连接与生成过程后，也不能仅凭一个局部 MSE 数字推导最终任务质量。

## Fake Quant 为什么仍然输出浮点数

Fake quant 的前向执行 Q/DQ：先按量化规则选 code，再恢复成浮点值，让后面的浮点算子看到量化扰动。例如，PyTorch 2.8 的 `fake_quantize_per_tensor_affine` 返回的是模拟量化后的浮点张量，并不是已经 packing 的 INT8 权重。[PyTorch 2.8 API](https://docs.pytorch.org/docs/2.8/generated/torch.fake_quantize_per_tensor_affine.html)

这里有一个训练问题：固定参数后，Q/DQ 关于输入是阶梯函数，真实导数几乎处处为零，在跳变处则通常不可导。若训练直接沿这个导数传播，细小权重变化很难获得有效梯度。

常见做法是 straight-through estimator，简称 STE：前向保留阶梯函数，反向另行指定一个可用的梯度近似。它不是证明 round 的导数为 1，而是主动采用不同于真实导数的代理。

边界规则依实现而定。以 PyTorch v2.8.0 的 per-tensor affine cachemask CPU 路径为例，先得到未 clip 的舍入 code $$q_{\mathrm{raw}}$$，再构造：

$$
m=\mathbf{1}\{q_{\min}\le q_{\mathrm{raw}}\le q_{\max}\},
\qquad dX=dY\,m
$$

这不是简单地判断原值是否位于重建端点之间。取 $$s=1,z=0,q\in[-7,7]$$，原值 7.2 虽然大于端点 7，但舍入 code 仍为 7，mask 为 1；到 7.6 时舍入 code 为 8，mask 才变成 0。该 CPU helper 实际先以 float 计算倒数，再对输入乘倒数的结果舍入；scale 精度、输入 dtype 和贴近 tie 的浮点误差都属于实现边界。后面的标准库代码用于展示这条 mask 规则，不是 PyTorch 各路径的逐位复刻。[固定版本 CPU 实现](https://github.com/pytorch/pytorch/blob/v2.8.0/aten/src/ATen/native/quantized/cpu/kernels/QuantizedOpKernels.cpp#L2515)、[反向传播实现](https://github.com/pytorch/pytorch/blob/v2.8.0/aten/src/ATen/native/quantized/FakeQuantPerTensorAffine.cpp#L110)

把这种模拟放进训练过程，就是量化感知训练（quantization-aware training，QAT）的一种常见做法。优化器仍更新高精度 master weights，fake quant 把部署时的误差带进训练前向；训练结束后还要导出编码、参数与可执行图。与之相对，训练后量化（post-training quantization，PTQ）从已经训练好的权重出发选择量化参数，不要求重新训练整套模型。采用 STE 并不保证任意位宽都能收敛，也不保证恢复原模型精度。

这与冻结量化基座、只训练低秩 adapter 的 [QLoRA](https://arxiv.org/abs/2305.14314) 是不同训练安排。两者都涉及低位宽模型，并不意味着可训练参数、前向边界和部署产物相同。

## 用标准库复算编码和计算

下面三段代码分别可独立运行。它们处理本文的小规模有限数值，用于理解数值关系，不负责打包权重或调用量化硬件。

第一段实现标量 affine Q/DQ，并把“未 clip 的 code 越界”和最终重建值一起返回。范围检查要求所选连续区间能够完整装进 INT8 或 UINT8；全零张量的 scale 选择由调用方处理。

<!-- example: quant-foundations-affine -->
```python
from math import isfinite, isclose


def quantize_scalar(x, scale, zero, qmin, qmax):
    if not isfinite(x) or not isfinite(scale) or scale <= 0:
        raise ValueError("finite x and positive finite scale required")
    if any(type(v) is not int for v in (zero, qmin, qmax)):
        raise ValueError("code bounds and zero point must be integers")
    fits_int8 = -128 <= qmin < qmax <= 127
    fits_uint8 = 0 <= qmin < qmax <= 255
    if not ((fits_int8 or fits_uint8) and qmin <= zero <= qmax):
        raise ValueError("invalid low-bit code range or zero point")
    normalized = x / scale
    if not isfinite(normalized):
        raise ValueError("normalization overflow")
    raw = round(normalized) + zero  # Python round uses ties-to-even.
    code = min(qmax, max(qmin, raw))
    restored = scale * (code - zero)
    if not isfinite(restored):
        raise ValueError("reconstruction overflow")
    return code, restored, not (qmin <= raw <= qmax)


xs = [-0.625, -0.375, -0.125, 0.0, 0.375, 1.60, 2.40]
results = [quantize_scalar(x, 0.25, 0, -7, 7) for x in xs]
assert [row[0] for row in results] == [-2, -2, 0, 0, 2, 6, 7]
assert [row[2] for row in results] == [False] * 6 + [True]

unsigned = [quantize_scalar(x, 0.1, 128, 0, 255)
            for x in [-1.0, 0.0, 0.2, 1.0]]
assert [row[0] for row in unsigned] == [118, 128, 130, 138]
assert isclose(unsigned[2][1], 0.2)
assert round(0.5) + 1 == 1
assert round(0.5 + 1) == 2
print(results)
```

这里用 Python 浮点数保存输入与重建值，所以比较小数时使用容差。Python 的二进制浮点、实际 scale 存储精度和 kernel 可能采用的倒数乘法，都会影响贴近 tie 的数值；显式写出舍入约定只是完整数值契约的一部分。

第二段从同一组 code 出发，核对零点展开、重建后点积，以及改变 weight scale 沿归约维的位置会造成什么差别。

<!-- example: quant-foundations-dot -->
```python
from math import fsum, isclose

qx, qw = [3, 4], [5, 7]
zx, zw = 2, 4
sx, sw = 0.5, 0.25
k = len(qx)

centered = sum((a - zx) * (b - zw) for a, b in zip(qx, qw))
expanded = (sum(a * b for a, b in zip(qx, qw))
            - zw * sum(qx) - zx * sum(qw) + k * zx * zw)
x = [sx * (a - zx) for a in qx]
w = [sw * (b - zw) for b in qw]
assert centered == expanded == 7
assert isclose(sx * sw * centered, fsum(a * b for a, b in zip(x, w)))
assert isclose(sx * sw * centered, 0.875)

# Each position is its own group; scales now vary along reduction K.
group_scales = [0.25, 0.5]
grouped = sx * fsum(s * (a - zx) * (b - zw)
                   for a, b, s in zip(qx, qw, group_scales))
wrong = sx * group_scales[0] * centered
assert isclose(grouped, 1.625)
assert not isclose(grouped, wrong)
print({"centered_sum": centered, "output": 0.875,
       "group_scaled_output": grouped})
```

第三段只模拟前面固定参数的 fake quant 和 rounded-code STE mask。有限差分计算的是前向函数的局部变化；`ste_dx` 则是另外指定的反向规则，两者应当不同。

<!-- example: quant-foundations-ste -->
```python
from math import isfinite


def fake_quant_and_mask(x):
    if not isfinite(x):
        raise ValueError("finite scalar required")
    raw = round(x)  # scale=1, zero=0, code range [-7, 7]
    restored = float(min(7, max(-7, raw)))
    mask = -7 <= raw <= 7
    return restored, mask


def ste_dx(x, dy=1.0):
    return dy * fake_quant_and_mask(x)[1]


x, eps = 0.2, 1e-4
fd = (fake_quant_and_mask(x + eps)[0]
      - fake_quant_and_mask(x - eps)[0]) / (2 * eps)
assert fd == 0.0
assert ste_dx(x) == 1.0
assert fake_quant_and_mask(7.2) == (7.0, True)
assert fake_quant_and_mask(7.6) == (7.0, False)
assert ste_dx(7.2, 3.0) == 3.0
assert ste_dx(7.6, 3.0) == 0.0
print({"finite_difference": fd, "surrogate_gradient": ste_dx(x)})
```

## 一个量化模型需要保存哪些数值约定

现在再看“这是一份 INT4 模型”，至少还需要知道：codes 使用什么范围，scale 是乘子还是倒数，zero point 属于编码域还是实数域，哪些元素共享参数，以及 group 沿哪条轴切分。

进入计算阶段，还要知道舍入发生在哪里、scale 用何种精度保存、整数如何累加，以及输出何时重新量化。进入文件与 kernel 阶段，code 的 packing 顺序、尾组 padding 和布局也必须对齐。相同位宽并不是一份完整的互操作协议。

数值基础的核心不是记住更多格式缩写，而是能沿着 $$x\to q\to\hat{x}\to\hat{Y}$$ 追踪变化：误差在哪里进入，哪些量可以提出求和，哪些修正不能省略，哪些代理只存在于训练反向。当这些关系明确之后，才容易理解 [FP8]({% post_url 2026-08-01-fp8-formats-scaling-llm-inference %}) 为什么换了一种格点结构，[AWQ]({% post_url 2026-08-03-awq-activation-aware-weight-quantization %}) 为什么观察 activation，以及 [QServe]({% post_url 2026-08-04-w4a8-progressive-quantization-kernel %}) 为什么要联合设计多级量化与 kernel。

## 参考资料

- [Jacob et al. Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/html/1712.05877v1)，§2.1–2.4、§3。首次预印本发表于 2017 年，后发表于 CVPR 2018。
- [ONNX QuantizeLinear opset 13](https://onnx.ai/onnx/operators/onnx__QuantizeLinear.html#quantizelinear-13) 与 [DequantizeLinear opset 13](https://onnx.ai/onnx/operators/onnx__DequantizeLinear.html#dequantizelinear-13)。
- [LiteRT 8-bit Quantization Specification](https://developers.google.com/edge/litert/conversion/tensorflow/quantization/quantization_spec)。
- [PyTorch 2.8 fake_quantize_per_tensor_affine](https://docs.pytorch.org/docs/2.8/generated/torch.fake_quantize_per_tensor_affine.html) 及文中链接的 v2.8.0 CPU、backward 实现。
