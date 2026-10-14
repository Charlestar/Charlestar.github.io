---
layout: post
title: "QuaRot 与 SpinQuant：为什么换一个坐标系能改善低比特量化"
subtitle: "从 Hadamard 算例到归一化与 Attention 的等价条件，再到受正交约束的旋转学习"
date: 2026-10-15
last_modified_at: 2026-10-15
author: iStar
catalog: true
series: model-quantization
series_order: 90
technology_year: 2024
mathjax: true
tags: [模型量化, GPU优化, LLM推理]
---

一个异常大的坐标，可能迫使同组所有数值使用很粗的量化步长。[LLM.int8()]({% post_url 2026-10-15-llm-int8-outlier-decomposition %}) 把相关通道拆到高精度分支，[SmoothQuant]({% post_url 2026-08-02-smoothquant-w8a8-activation-outliers %}) 和 [AWQ]({% post_url 2026-08-03-awq-activation-aware-weight-quantization %}) 则通过对角缩放重新分配量化难度。另一条路线是：不只放大或缩小某个坐标，而是把多个坐标混合起来，换一套基来表示相同的计算。

QuaRot 利用正交变换让权重、激活与 KV Cache 更适合低比特表示，并把能够抵消的变换融合到权重中。SpinQuant 在此类旋转参数化上继续追问：随机选择的坐标系不一定最好，能不能直接学习一套对量化网络更合适的坐标系？

理解这两种方法，需要同时把握两件事：**未量化的计算为什么可以保持不变，以及量化后为什么反而会变。** 仅证明一个矩阵乘法等价，还不足以宣称整个 Transformer 改写正确；归一化、残差、非线性与位置编码都会限制旋转应该放在哪里。

## 1. 正交变换保留能量 但不保留最大坐标

沿用行向量约定：输入 $$X\in\mathbb R^{M\times K}$$，权重 $$W\in\mathbb R^{K\times N}$$。若 $$R\in\mathbb R^{K\times K}$$ 满足 $$R^TR=RR^T=I$$，令：

$$
X'=XR,\qquad W'=R^TW,
$$

就有：

$$
X'W'=(XR)(R^TW)=XW.
$$

如果输出也换到另一个正交坐标系 $$S$$，对应权重则是 $$W'=R^TWS$$。这里讨论的是数学布局；PyTorch Linear 通常存储 $$N\times K$$，源码左右乘的方向必须按转置关系理解。

正交变换保持向量的二范数和点积：

$$
\lVert xR\rVert_2=\lVert x\rVert_2.
$$

但它不保持最大绝对坐标 $$\lVert x\rVert_\infty$$。例如归一化 Hadamard 矩阵：

$$
H_2=\frac1{\sqrt2}
\begin{bmatrix}1&1\\1&-1\end{bmatrix}
$$

会把 $$[10,0]$$ 变成 $$[10/\sqrt2,10/\sqrt2]$$。能量没有减少，只是不再集中在一个坐标上。对于按 absmax 决定步长的量化器，这可能让原本被大坐标挤压的小数值获得更合适的格点。

这里的归一化不能省：尺寸为 $$d$$ 的未归一化 Hadamard 矩阵 $$B$$ 满足 $$B^TB=dI$$，只有 $$H=B/\sqrt d$$ 才满足正交条件。归一化矩阵元素是 $$\pm1/\sqrt d$$，不是 $$\pm1$$，也不是 $$\pm\sqrt d$$。本文沿用领域中“旋转”的称呼泛指这种正交变换，它可以包含反射，并不额外要求行列式为 1。[QuaRot v2，§3.1](https://arxiv.org/html/2404.00456v2#S3.SS1)

同样，旋转也可能把能量集中起来：$$[1,1]H_2=[\sqrt2,0]$$，最大坐标反而变大。因此“旋转抑制异常值”描述的是所选变换对目标分布的效果，不能理解为任意正交矩阵对任意输入都降低 absmax。

## 2. 同一个旋转可以让量化更好 也可以更差

考虑一个二维点积，激活和权重都使用各自的 absmax 步长：

$$
\Delta_v=\frac{\max_i\lvert v_i\rvert}{7},\qquad
\mathcal Q(v)_i=\Delta_v\operatorname{clip}\!\left(
\operatorname{RNE}(v_i/\Delta_v),-7,7\right).
$$

这是独立教学合同：最近值舍入采用 ties-to-even，$$[-7,7]$$ 是能由 4-bit 容纳的 15 值子码域，不是全部 16 个码值，也不是官方 packed checkpoint 的格式。$$\mathcal Q$$ 表示浮点 Q/DQ 重建，不代表已经压缩存储；全零组另设正步长为 1、codes 全为 0。

先取 $$x=[10,0.2]$$、$$w=[1,1]^T$$。真实点积是 $$10.2$$，不旋转时，激活量化成 $$[10,0]$$，权重不变，结果为 $$10$$。

同时旋转激活与权重后：

$$
xH_2=\left[\frac{10.2}{\sqrt2},\frac{9.8}{\sqrt2}\right],\qquad
H_2^Tw=\begin{bmatrix}\sqrt2\\0\end{bmatrix}.
$$

激活的两个坐标量化后都成为第一个坐标的大小，但第二个权重恰好为零，最终点积恢复为 $$10.2$$。这展示了有利的格点安排。

现在只把权重换成 $$[0.1,1]^T$$，真实点积变成 $$1.2$$。未旋转的量化结果约为 $$1.428571$$，旋转后却变成约 $$0.801429$$：

| 权重 $$w$$ | 真实点积 | 不旋转的 Q/DQ 点积 | 同时旋转后的 Q/DQ 点积 |
| --- | --- | --- | --- |
| $$[1,1]^T$$ | $$10.2$$ | $$10$$ | $$10.2$$ |
| $$[0.1,1]^T$$ | $$1.2$$ | $$1.428571\ldots$$ | $$0.801429\ldots$$ |

第二个例子里，激活和权重的最大绝对坐标都减小了，最终点积误差仍然变大。量化误差还受格点位置、激活与权重的共同变化，以及误差项的叠加影响，不能只看某一个 absmax 指标。

下面直接执行这两个算例，并同时检查正交、范数与点积恒等式。它使用 Python 标准库浮点数，限于下方有限小向量，拒绝不可表示的步长和非有限中间结果，不提供全动态范围高精度保证；也不复现官方校准、低比特 packing 或模型质量。

<!-- example: rotation-helpful-and-harmful-quantization -->
```python
from math import isclose, isfinite, sqrt

def dot(a, b):
    if not a or len(a) != len(b) or not all(isfinite(v) for v in [*a, *b]):
        raise ValueError("finite vectors with equal nonzero dimensions required")
    result = sum(x * y for x, y in zip(a, b))
    if not isfinite(result):
        raise ValueError("dot product is outside this float reference's range")
    return result

def hadamard2(row):
    if len(row) != 2 or not all(isfinite(v) for v in row):
        raise ValueError("a finite two-element vector is required")
    a, b = row
    result = [(a + b) / sqrt(2), (a - b) / sqrt(2)]
    if not all(isfinite(v) for v in result):
        raise ValueError("Hadamard intermediate is outside this float reference's range")
    return result

def qdq(vector):
    if not vector or not all(isfinite(v) for v in vector):
        raise ValueError("finite nonempty vector required")
    peak = max(abs(v) for v in vector)
    step = peak / 7 if peak else 1.0
    if not isfinite(step) or step <= 0:
        raise ValueError("step is outside this float reference's range")
    result = [step * min(7, max(-7, round(v / step))) for v in vector]
    if not all(isfinite(v) for v in result):
        raise ValueError("reconstruction is outside this float reference's range")
    return result

x = [10.0, 0.2]
rotated_x = hadamard2(x)
assert isclose(dot(x, x), dot(rotated_x, rotated_x), abs_tol=1e-12)
assert max(map(abs, rotated_x)) < max(map(abs, x))
assert max(map(abs, hadamard2([1.0, 1.0]))) > 1.0
for w, expected_plain, expected_rotated in [
        ([1.0, 1.0], 10.0, 10.2),
        ([0.1, 1.0], 10 / 7, 0.8014285714285716)]:
    reference = dot(x, w)
    rotated_w = hadamard2(w)  # H is symmetric; column H^T w has these entries
    assert isclose(dot(rotated_x, rotated_w), reference, abs_tol=1e-12)
    plain = dot(qdq(x), qdq(w))
    rotated = dot(qdq(rotated_x), qdq(rotated_w))
    assert isclose(plain, expected_plain, abs_tol=1e-12)
    assert isclose(rotated, expected_rotated, abs_tol=1e-12)
    if w == [1.0, 1.0]:
        assert abs(rotated - reference) < abs(plain - reference)
    else:
        assert abs(rotated - reference) > abs(plain - reference)
    print("w", w, "true / plain / rotated:", reference, plain, rotated)

raw_x = [x[0] + x[1], x[0] - x[1]]
raw_w = [1.0 + 1.0, 1.0 - 1.0]
assert isclose(dot(raw_x, raw_w), 2 * dot(x, [1.0, 1.0]), abs_tol=1e-12)
assert qdq([0.0, 0.0]) == [0.0, 0.0]
```

未量化时的等价关系是确定的代数事实；量化后则是在不同坐标系里使用不同的网格。这正是 QuaRot 能发挥作用、SpinQuant 还能继续优化的空间。

## 3. 全网络等价需要保持同一套残差坐标

如果每个 Linear 都独立换一套基，却不处理它们之间的算子，模型很快就会失去等价关系。以 pre-norm 残差块为例，想把残差状态一直保存为 $$h'=hR$$，需要让整个子层满足：

$$
F'(hR)=F(h)R.
$$

这样残差相加才保持语义：

$$
hR+F'(hR)=(h+F(h))R.
$$

把子层输出旋回原坐标再与 $$hR$$ 相加，就不是同一件事。

RMSNorm 提供了实现这个目标的重要条件。先去掉逐坐标可学习增益，保留原 epsilon：

$$
\operatorname{RMS}_{\epsilon}(h)
=\frac{h}{\sqrt{\lVert h\rVert_2^2/d+\epsilon}}.
$$

由于分母由范数决定：

$$
\operatorname{RMS}_{\epsilon}(hR)
=\operatorname{RMS}_{\epsilon}(h)R.
$$

同一个 $$\epsilon$$ 不破坏等变关系，但随意修改 $$\epsilon$$ 会改变原函数。RMSNorm 原本还带有 $$\Gamma=\operatorname{diag}(\gamma)$$，一般有 $$R\Gamma\ne\Gamma R$$，所以不能直接把带 gamma 的整个算子当作旋转等变。

正确的思路是先把 gamma 融入所有下游输入投影。原来的计算是：

$$
u=\operatorname{RMS}_{\epsilon}(h)\Gamma W_{\mathrm{in}}+b_{\mathrm{in}}.
$$

在旋转后的残差基中，改用不带 gamma 的 RMS，再令：

$$
W'_{\mathrm{in}}=R^T\Gamma W_{\mathrm{in}},
$$

就能恢复相同的 $$u$$。Attention 的 Q/K/V 如果共用这个 norm，就都要变换；MLP 的 up/gate 如果共用输入，也不能遗漏其中一个。输出投影则改为 $$W'_{\mathrm{out}}=W_{\mathrm{out}}R$$，输出 bias 同样进入新坐标系：$$b'_{\mathrm{out}}=b_{\mathrm{out}}R$$。[QuaRot v2，§3.4 与 Stage 1a](https://arxiv.org/html/2404.00456v2#S3.SS4)

网络两端也需要闭合这套约定：embedding 输出进入旋转后的残差基，最终 norm 的 affine 参数按其消费者融合，LM head 再把旋转坐标映射回原 logits。注入残差的其他向量、共享参数和 bias 都必须遵循同一语义，而不是只修改中间某个权重。

这个论证不能原样推广到任意 LayerNorm。LayerNorm 还要沿坐标减去均值，任意正交变换并不保持“全 1 方向”，去均值与旋转一般不交换。对不同 norm 和架构，需要各自证明变换条件；也不能从理想恒等式推出某份预处理脚本的每个操作都严格等价。原 affine、epsilon、全部消费者与残差基是否正确保留，是实际实现必须满足的条件。[固定版 QuaRot 融合与旋转代码](https://github.com/spcl/QuaRot/blob/5008669b08c1f11f9b64d52d16fddd47ca754c5a/fake_quant/rotation_utils.py)、[norm 实现](https://github.com/spcl/QuaRot/blob/5008669b08c1f11f9b64d52d16fddd47ca754c5a/fake_quant/model_utils.py)

## 4. 不能随意跨过非线性和 RoPE

全局残差旋转让子层入口的坐标更适合量化，但 MLP 内部还会产生新的异常值。对于门控 MLP：

$$
Z=\operatorname{SiLU}(U_{\mathrm{gate}})\odot U_{\mathrm{up}},\qquad
Y=ZW_{\mathrm{down}},
$$

可以在已经形成的 $$Z$$ 后面插入正交变换：

$$
Y=(ZH)(H^TW_{\mathrm{down}}).
$$

输入端的 $$ZH$$ 是在线变换，逆变换可融合进 down-projection。不能把 $$H$$ 随意移到 SiLU、GELU 或门控乘积前面；这些逐元素操作一般不与混合坐标的矩阵乘法交换。QuaRot 因此既有离线融合，也有不能完全消去的在线 Hadamard。[QuaRot v2，Stage 1b](https://arxiv.org/html/2404.00456v2#S4)

Attention 中则有两类配对。首先，在同一个 head 内对 Q/K 同时施加同一个正交变换：

$$
(QH)(KH)^T=QK^T.
$$

只旋转一边不成立；也不能把不同 head 的独立注意力随意混合成一次相同的变换。另一个配对发生在 Value 与输出投影之间：若 $$V'=VH$$、$$W'_o=H^TW_o$$，则 $$PV'W'_o=PVW_o$$。多头情形还要按相应分块组织输出投影，QuaRot 进一步利用了 Kronecker 结构，不能只凭一个 head 的等式省略实现中的 reshape 与索引。

Q/K 还有 RoPE 的位置问题。令 $$Q_{\mathrm{pos}}$$、$$K_{\mathrm{pos}}$$ 表示位置变换后的张量，安全的配对是：

$$
(Q_{\mathrm{pos}}H)(K_{\mathrm{pos}}H)^T
=Q_{\mathrm{pos}}K_{\mathrm{pos}}^T.
$$

如果先旋转再分别做不同位置的 RoPE，一般无法这样抵消。二维就能构造反例：取 $$q=[1,0]$$、$$k=[0,1]$$，Query 的位置变换为单位阵，Key 的为：

$$
P=\begin{bmatrix}0&1\\-1&0\end{bmatrix}.
$$

原点积 $$q(kP)^T=-1$$，在位置变换后同时乘 $$H_2$$ 仍为 $$-1$$；但先乘 $$H_2$$ 再对 Key 乘 $$P$$，点积会变为 $$+1$$。乘上正常的 $$1/\sqrt{D_{\mathrm{head}}}$$ 也不会消除这个差异。

固定版 `QKRotationWrapper` 包装位置函数的输出，对 Q、K 分别执行带 $$1/\sqrt{D_{\mathrm{head}}}$$ 归一化的 Hadamard；Attention 原有的 score 缩放和 mask 仍需保留。两个同为 $$1/\sqrt{D_{\mathrm{head}}}$$ 的因子作用不同：前者使 Hadamard 正交，后者缩放 Attention logits。这也是不能把 Q/K 旋转直接、无条件吸收到 RoPE 之前投影权重中的原因。[QuaRot v2，Stage 1d](https://arxiv.org/html/2404.00456v2#S4)、[固定版 wrapper](https://github.com/spcl/QuaRot/blob/5008669b08c1f11f9b64d52d16fddd47ca754c5a/fake_quant/rotation_utils.py)

下面用一个小型、未量化的网络检查坐标约定：embedding 后有两个 RMSNorm 门控残差块，最后接 norm 与 LM head。它不是完整 Transformer，但覆盖了多块之间的残差基、两个 norm 消费分支、gamma、不同 epsilon 与输出 bias；随后单独检查非线性和 RoPE 的反例。所有比较都采用浮点容差，不声称逐位相等。

<!-- example: rotation-norm-residual-graph-and-rope -->
```python
from math import exp, isclose, isfinite, sqrt

def check_vector(row):
    if not row or not all(isfinite(v) for v in row):
        raise ValueError("finite nonempty vector required")
    return row

def check_matrix(matrix):
    if (not matrix or not matrix[0]
            or any(len(row) != len(matrix[0]) for row in matrix)):
        raise ValueError("nonempty rectangular matrix required")
    for row in matrix:
        check_vector(row)

def transpose(a):
    check_matrix(a)
    return [list(col) for col in zip(*a)]

def right(row, matrix):
    check_vector(row)
    check_matrix(matrix)
    if len(row) != len(matrix):
        raise ValueError("vector and matrix dimensions differ")
    return check_vector([sum(row[k] * matrix[k][n] for k in range(len(row)))
                         for n in range(len(matrix[0]))])

def mm(a, b):
    check_matrix(a)
    return [right(row, b) for row in a]

def add(a, b):
    check_vector(a)
    check_vector(b)
    if len(a) != len(b):
        raise ValueError("vector dimensions differ")
    return check_vector([x + y for x, y in zip(a, b)])

def close(a, b):
    return len(a) == len(b) and all(isclose(x, y, abs_tol=1e-11)
                                   for x, y in zip(a, b))

def rms(row, epsilon):
    check_vector(row)
    if not isfinite(epsilon) or epsilon <= 0:
        raise ValueError("finite positive epsilon required")
    denominator = sqrt(sum(v * v for v in row) / len(row) + epsilon)
    if not isfinite(denominator) or denominator <= 0:
        raise ValueError("RMS intermediate is outside this float reference's range")
    return [v / denominator for v in row]

def gain(row, gamma):
    check_vector(row)
    check_vector(gamma)
    if len(row) != len(gamma):
        raise ValueError("gamma dimensions differ")
    return check_vector([x * g for x, g in zip(row, gamma)])

def scale_rows(matrix, gamma):
    check_matrix(matrix)
    check_vector(gamma)
    if len(matrix) != len(gamma):
        raise ValueError("gamma and matrix dimensions differ")
    return [check_vector([g * v for v in row]) for g, row in zip(gamma, matrix)]

def silu(v):
    if not isfinite(v):
        raise ValueError("finite SiLU input required")
    if v >= 0:
        return v / (1 + exp(-v))
    e = exp(v)
    return v * e / (1 + e)

h = 1 / sqrt(2)
rotation = [[h, h], [h, -h]]
inverse = transpose(rotation)
blocks = [dict(gamma=gamma, epsilon=epsilon,
               wg=[[0.2, -0.4], [0.6, 0.3]],
               wu=[[0.7, 0.1], [-0.2, 0.5]],
               wd=[[0.8, -0.1], [0.4, 0.9]],
               bg=[0.1, -0.2], bu=[-0.05, 0.2], bd=[0.03, -0.04])
          for gamma, epsilon in [([1.2, 0.7], 1e-5), ([0.8, 1.3], 1e-4)]]

def block_forward(state, block):
    normalized = gain(rms(state, block["epsilon"]), block["gamma"])
    gates = add(right(normalized, block["wg"]), block["bg"])
    up = add(right(normalized, block["wu"]), block["bu"])
    if len(gates) != len(up):
        raise ValueError("gate and up dimensions differ")
    z = [silu(g) * u for g, u in zip(gates, up)]
    return add(state, add(right(z, block["wd"]), block["bd"]))

def rotate_block(block):
    result = dict(block)
    for name in ["wg", "wu"]:  # both consumers must receive fused gamma
        result[name] = mm(inverse, scale_rows(block[name], block["gamma"]))
    result["wd"] = mm(block["wd"], rotation)
    result["bd"] = right(block["bd"], rotation)
    result["gamma"] = [1.0] * len(block["gamma"])
    assert result["epsilon"] == block["epsilon"]
    return result

embedding = [[0.3, -0.7], [1.2, 0.4], [-0.8, 1.1]]
rotated_embedding = [right(row, rotation) for row in embedding]
state, rotated_state = embedding[1], rotated_embedding[1]
for block in blocks:
    state = block_forward(state, block)
    rotated_state = block_forward(rotated_state, rotate_block(block))
    assert close(rotated_state, right(state, rotation))

head_gamma, head_epsilon = [0.9, 1.1], 3e-5
head = [[0.2, -0.3, 0.7], [0.6, 0.4, -0.2]]
head_bias = [0.1, -0.05, 0.02]
logits = add(right(gain(rms(state, head_epsilon), head_gamma), head), head_bias)
rotated_head = mm(inverse, scale_rows(head, head_gamma))
rotated_logits = add(right(rms(rotated_state, head_epsilon), rotated_head), head_bias)
assert close(logits, rotated_logits)
print("original / rotated graph logits:", logits, rotated_logits)

x, gamma, epsilon = [1.0, 3.0], [2.0, 0.5], 1e-5
assert close(rms(right(x, rotation), epsilon), right(rms(x, epsilon), rotation))
assert not close(gain(rms(right(x, rotation), epsilon), gamma),
                 right(gain(rms(x, epsilon), gamma), rotation))
centered = [v - sum(x) / len(x) for v in x]
rotated_x = right(x, rotation)
rotated_centered = [v - sum(rotated_x) / len(rotated_x) for v in rotated_x]
assert not close(rms(rotated_centered, epsilon), right(rms(centered, epsilon), rotation))
assert not close([silu(v) for v in right(x, rotation)],
                 right([silu(v) for v in x], rotation))

q, k = [1.0, 0.0], [0.0, 1.0]
position = [[0.0, 1.0], [-1.0, 0.0]]
def dot(a, b):
    check_vector(a)
    check_vector(b)
    if len(a) != len(b):
        raise ValueError("vector dimensions differ")
    result = sum(x * y for x, y in zip(a, b))
    if not isfinite(result):
        raise ValueError("dot product is outside this float reference's range")
    return result
original_score = dot(q, right(k, position)) / sqrt(2)
post_score = dot(right(q, rotation), right(right(k, position), rotation)) / sqrt(2)
pre_score = dot(right(q, rotation), right(right(k, rotation), position)) / sqrt(2)
assert isclose(post_score, original_score, abs_tol=1e-12)
assert isclose(pre_score, -original_score, abs_tol=1e-12)
print("RoPE score, original / post / pre:", original_score, post_score, pre_score)
```

## 5. QuaRot 的旋转与量化是两个阶段

QuaRot 不是用“旋转”替换量化器。它先改写适合改写的网络表示，再在这个表示上量化。理解实际执行时，可以把操作分成三类：

| 操作 | 目的 | 是否需要在推理中显式执行 |
| --- | --- | --- |
| 残差基旋转与相邻权重配对 | 改善残差激活和权重的坐标分布 | 能融合的部分离线写入权重 |
| MLP 内部、位置编码后的 Q/K Hadamard | 处理不能随意穿过非线性或 RoPE 的位置 | 相应位置保留在线变换 |
| 权重与激活、KV 的量化 | 映射到有限码域与目标执行格式 | 权重离线量化，动态数据按路径在线处理 |

归一化 Walsh-Hadamard 的蝶形结构在 $$d=2^n$$ 时可以用 $$O(d\log d)$$ 次操作计算，而不是显式构造稠密矩阵后做 $$O(d^2)$$ 乘法。模型维度不全是 2 的幂，论文还使用已知 Hadamard 大小与 Kronecker 分解；这不等于任意尺寸都能不经适配地套同一个快速 kernel。固定版 Q/K wrapper 对 head dimension 也有自己的尺寸限制。[QuaRot v2，§3.1](https://arxiv.org/html/2404.00456v2#S3.SS1)、[固定版在线变换配置](https://github.com/spcl/QuaRot/blob/5008669b08c1f11f9b64d52d16fddd47ca754c5a/fake_quant/main.py)

论文默认使用 [GPTQ]({% post_url 2026-10-15-gptq-second-order-reconstruction %}) 量化旋转后的权重，也研究了 RTN。旋转解决数值表示的问题，GPTQ 继续处理给定表示下的权重重建误差，二者不是互斥选项。

还要区分不同算子的“4-bit”。QuaRot 的 Linear 路径可以让 INT4 激活与 INT4 权重做矩阵乘，得到 INT32 结果后结合尺度恢复浮点；但论文 Stage 2c 中，Attention 读取低比特 KV 后反量化到 FP16 再计算点积，Query 保持 FP16。不能把“权重、激活、KV 都可用低比特表示”扩写成“整个网络所有矩阵乘都以原生 INT4 指令执行”。[QuaRot v2，Stage 2a–2c](https://arxiv.org/html/2404.00456v2#S4)

## 6. SpinQuant 学习的是哪些坐标系

前面的反例说明：正交性只保证未量化时的计算关系，并不直接评价量化后的质量。即使都使用随机 Hadamard，不同随机符号也会改变数据与网格的相对位置。SpinQuant 的实验发现随机旋转会带来明显质量差异，于是把旋转本身作为需要优化的对象。[SpinQuant v4，§2.2](https://arxiv.org/html/2405.16406v4#S2.SS2)

它把可旋转的位置组织为四类，不能把这四类都描述成独立学习的稠密矩阵：

| 名称 | 作用位置与共享方式 | 在 SpinQuant 中是否学习 |
| --- | --- | --- |
| $$R_1$$ | 全模型共用的残差坐标系，矩阵两维均为模型隐藏维度 | 学习，可融合到相邻权重 |
| $$R_2^{(\ell)}$$ | 每层一套 head-dimension 方阵，在该层各头间共享，连接 Value 与输出投影 | 各层分别学习，可融合 |
| $$R_3$$ | 位置编码后 Q/K 的在线 Hadamard | 固定，不学习为任意稠密矩阵 |
| $$R_4$$ | MLP 内部到 down-projection 前的在线 Hadamard | 固定，不学习为任意稠密矩阵 |

尤其是 $$R_2$$：不是每个 head 各自保存一套互不相关的可学习矩阵，而是**每层一套、层内共享**。固定版 `optimize_rotation.py` 先冻结基础模型参数，再创建全局 `model.R1` 和每层 Attention 的 `R2`，交给优化器的也是这些旋转参数。[SpinQuant v4，§3.1–3.2](https://arxiv.org/html/2405.16406v4#S3.SS1)、[固定版 optimize_rotation.py](https://github.com/facebookresearch/SpinQuant/blob/8f47aa3f00e8662caf1a484153920a07e5281c3a/optimize_rotation.py)

保持 $$R_3$$、$$R_4$$ 为结构化 Hadamard 有执行上的原因：它们留在在线路径上，若随意学成大稠密矩阵，可能把快速变换变成昂贵矩阵乘。$$R_1$$、$$R_2$$ 能融合到权重，因此有更大的坐标系选择空间。

优化目标可以概括为，在基础权重固定的条件下，寻找满足正交约束的旋转，使带量化的网络在校准数据上的任务损失较小：

$$
\min_{R_1^TR_1=I,\;(R_2^{(\ell)})^TR_2^{(\ell)}=I}
\mathcal L_{\mathcal Q}\!\left(W_{\mathrm{fixed}},R_1,
\{R_2^{(\ell)}\};\mathcal D_{\mathrm{cal}}\right).
$$

它没有把基础模型的全部参数重新训练一遍，但仍需要前向、反向与校准数据，不能把这个学习阶段说成零成本。冻结基础权重只是不更新它；旋转后的权重和激活仍依赖 $$R$$，损失梯度需要沿这些运算回传给旋转参数。量化中的 round 也不能当作普通平滑层精确求导：固定版 `STEQuantize` 与 `AsymSTEQuantize` 在反向时直接传回输入梯度，对 scale/zero 返回 None。这是 STE 代理反向，不是量化阶梯的精确导数。[固定版 quant_utils.py](https://github.com/facebookresearch/SpinQuant/blob/8f47aa3f00e8662caf1a484153920a07e5281c3a/utils/quant_utils.py)

学习时哪些位置启用量化，还要与最终使用的量化器配合。论文 §4.3.2 的 GPTQ 配置先不量化权重，用激活量化相关的损失优化旋转，再由 GPTQ 处理权重量化；KV 位宽随对应实验设置区分。因此不能笼统写成“所有旋转学习都同时对 W/A/KV 执行目标位宽量化”。[SpinQuant v4，§4.3.2](https://arxiv.org/html/2405.16406v4#S4.SS3.SSS2)

## 7. Cayley 变换为什么能约束正交性

如果像普通参数一样直接执行 $$R\leftarrow R-\eta G$$，即使旧 $$R$$ 正交，新矩阵也一般不满足 $$R^TR=I$$。这会破坏用转置抵消的基础。

Cayley 变换提供了一种构造方法。取反对称矩阵 $$A^T=-A$$，定义：

$$
C(t)=(I-tA)^{-1}(I+tA),\qquad R_{\mathrm{new}}=C(t)R.
$$

实数有限维下，反对称性使这个变换满足 $$C(t)^TC(t)=I$$，所以旧 $$R$$ 正交时，新 $$R$$ 也正交。这里只说明保正交的原理；$$A$$ 如何从梯度与动量构造、步长与符号怎样定义，才决定它是不是所需的下降更新，不能把任意自选 $$A$$ 的例子当成 SpinQuant 优化器的一步。

例如 $$A=\begin{bmatrix}0&-1\\1&0\end{bmatrix}$$，取 $$t=0.1$$：

$$
C(t)=\frac1{1+t^2}
\begin{bmatrix}1-t^2&-2t\\2t&1-t^2\end{bmatrix}
\approx\begin{bmatrix}
0.980198&-0.198020\\
0.198020&0.980198
\end{bmatrix}.
$$

下面检查这个二维闭式解，以及普通加性扰动为什么不保持正交。代码没有从模型损失计算梯度，也不是官方优化过程的复现。

<!-- example: rotation-cayley-orthogonality-principle -->
```python
from math import isclose, isfinite

def check_matrix(matrix):
    if (not matrix or not matrix[0]
            or any(len(row) != len(matrix[0]) for row in matrix)
            or not all(isfinite(v) for row in matrix for v in row)):
        raise ValueError("finite nonempty rectangular matrix required")

def transpose(a):
    check_matrix(a)
    return [list(col) for col in zip(*a)]

def mm(a, b):
    check_matrix(a)
    check_matrix(b)
    if len(a[0]) != len(b):
        raise ValueError("matrix dimensions differ")
    result = [[sum(a[i][k] * b[k][j] for k in range(len(b)))
               for j in range(len(b[0]))] for i in range(len(a))]
    check_matrix(result)
    return result

def gram_error(matrix):
    gram = mm(transpose(matrix), matrix)
    return max(abs(gram[i][j] - float(i == j))
               for i in range(len(gram)) for j in range(len(gram)))

t = 0.1
denominator = 1 + t * t
cayley = [[(1 - t * t) / denominator, -2 * t / denominator],
          [2 * t / denominator, (1 - t * t) / denominator]]
additive = [[1.0, -t], [t, 1.0]]  # I + tA, not a Cayley transform
assert gram_error(cayley) < 1e-12
assert isclose(gram_error(additive), t * t, abs_tol=1e-12)
composed = [[1.0, 0.0], [0.0, 1.0]]
for _ in range(100):
    composed = mm(cayley, composed)
assert gram_error(composed) < 1e-11
print("Cayley matrix:", cayley)
print("Gram error, Cayley / additive:", gram_error(cayley), gram_error(additive))
print("Gram error after 100 compositions:", gram_error(composed))
```

数学中的精确正交性与实际浮点优化还要分开。固定版 SpinQuant 优化器用负梯度和动量构造方向，`Cayley_loop` 只执行 5 次固定点迭代，并包含行归一化的 epsilon、步长限制与随机触发的 QR 修正。它不是每次显式精确求逆，也不能保证浮点下的 Gram 矩阵永远恰好为单位阵。检查 $$\lVert R^TR-I\rVert$$ 与关闭量化后的输出差，才能看到近似更新和融合在实际数值上的影响。[SpinQuant v4，§3.2](https://arxiv.org/html/2405.16406v4#S3.SS2)、[固定版 optimizer.py](https://github.com/facebookresearch/SpinQuant/blob/8f47aa3f00e8662caf1a484153920a07e5281c3a/train_utils/optimizer.py)

## 8. 换坐标系与高效执行需要分别成立

旋转方法有两层不同的正确性。第一层是不启用量化时，权重融合、norm、残差和在线位置是否满足相应的等价条件；第二层是启用量化后，新的网格是否在目标模型和数据上保留了足够的质量。第一层正确不保证第二层一定改善，第二层某个任务变好也不能掩盖第一层的实现错误。

性能则是另一条判断链。融合进权重的旋转不会额外生成一大块在线矩阵乘，但在线 Hadamard、类型转换和量化仍然需要计算；它们能否与 Linear 或 Attention 有效衔接，要看真实 kernel 与张量布局。QuaRot 论文 §5.2 的主要 GPU 延迟分析采用 RTX 3090 上单个 Transformer block 的 microbenchmark，不能把这些数字当成本机完整模型的端到端结果。[QuaRot v2，§5.2](https://arxiv.org/html/2404.00456v2#S5.SS2)

与 [KIVI]({% post_url 2026-10-15-kivi-kv-cache-axes %}) 对照也很清楚：KIVI 重点改变 K/V 的分组轴与流式缓存策略；旋转方法改变进入量化器的数据坐标。若进一步组合它们，还要重新确认旋转后分布、分组参数与缓存读取路径，不能把两篇各自的优点直接相乘。

QuaRot 说明了一套坐标变换如何从局部恒等式延伸到网络参数化，SpinQuant 则把“选什么坐标系”变成受约束的优化问题。两者都不是让量化误差消失，而是在不任意改写基础网络语义的前提下，寻找更适合有限格点的表示。真正有效的表示，既要在数学上配对正确，也要在量化质量和执行成本上值得采用。

## 参考资料

- [QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs，arXiv v2](https://arxiv.org/html/2404.00456v2)：正交表示、网络改写与两阶段量化；首发为 2024-03-30，本文引用修订日期为 2024-10-29。
- [SpinQuant: LLM Quantization with Learned Rotations，arXiv v4](https://arxiv.org/html/2405.16406v4)：旋转学习、Cayley 优化与 GPTQ 配置；首发为 2024-05-26，本文引用修订日期为 2025-02-20。引用后续修订不意味着其中所有实验和实现细节都已包含在 2024 初版。
- [QuaRot 固定版 rotation_utils.py](https://github.com/spcl/QuaRot/blob/5008669b08c1f11f9b64d52d16fddd47ca754c5a/fake_quant/rotation_utils.py)、[model_utils.py](https://github.com/spcl/QuaRot/blob/5008669b08c1f11f9b64d52d16fddd47ca754c5a/fake_quant/model_utils.py)、[main.py](https://github.com/spcl/QuaRot/blob/5008669b08c1f11f9b64d52d16fddd47ca754c5a/fake_quant/main.py)：融合、norm、在线 wrapper 与量化配置的固定实现。
- [SpinQuant 固定版 optimize_rotation.py](https://github.com/facebookresearch/SpinQuant/blob/8f47aa3f00e8662caf1a484153920a07e5281c3a/optimize_rotation.py)、[rotation_utils.py](https://github.com/facebookresearch/SpinQuant/blob/8f47aa3f00e8662caf1a484153920a07e5281c3a/eval_utils/rotation_utils.py)、[quant_linear.py](https://github.com/facebookresearch/SpinQuant/blob/8f47aa3f00e8662caf1a484153920a07e5281c3a/train_utils/quant_linear.py)：可学习对象与旋转权重的使用方式。
- [SpinQuant 固定版 optimizer.py](https://github.com/facebookresearch/SpinQuant/blob/8f47aa3f00e8662caf1a484153920a07e5281c3a/train_utils/optimizer.py)、[quant_utils.py](https://github.com/facebookresearch/SpinQuant/blob/8f47aa3f00e8662caf1a484153920a07e5281c3a/utils/quant_utils.py)：迭代更新与 STE 代理反向。本文标准库例子只验证各自的数学关系，不代表上述 Torch 实现、模型训练或 GPU kernel 的复现。
