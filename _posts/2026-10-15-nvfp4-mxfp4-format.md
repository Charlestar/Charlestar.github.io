---
layout: post
title: "NVFP4 与 MXFP4：4-bit 浮点如何依靠微块缩放表达真实张量"
subtitle: "从 E2M1 编码与两级 scale 到缩放误差、分组矩阵乘和真实字节成本"
date: 2026-10-15
last_modified_at: 2026-10-15
author: iStar
catalog: true
series: model-quantization
series_order: 110
technology_year: 2025
mathjax: true
tags: [模型量化, GPU优化, LLM推理]
---

“把权重压到 4 bit”只说明每个元素的编码长度，还没有说明这 4 bit 表示什么。INT4 通常把编码解释成整数，[QLoRA 的 NF4]({% post_url 2026-10-15-qlora-nf4-frozen-base %}) 用非均匀查找表表示归一化后的权重，FP4 则把符号、指数和尾数放进同一个四位编码。它们都可以每字节存两个元素，却不具有相同的数值语义。

NVFP4 和 MXFP4 又多了一层：**元素本身只有很小的数值范围，真实张量的范围由元素编码与共享 scale 一起表达。** 因而，理解这两种格式不能只背下 E2M1，还要弄清 scale 自己怎样编码、哪些元素共享一个 scale，以及这些 scale 怎样参与矩阵乘法。

本文以 NVIDIA 2025 年的 NVFP4 专题说明为锚。MXFP4 的历史更早：OCP 在 2023 年的 Microscaling Formats v1.0 中已经定义了它。这里的系列顺序不是“MXFP4 在 2025 年才出现”的意思。[OCP v1.0 规范](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)、[NVIDIA NVFP4 介绍](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)分别给出了两者的格式背景。

## 1. E2M1 的十六种编码究竟表示什么

E2M1 使用 1 位符号 $$s$$、2 位指数 $$e$$ 和 1 位尾数 $$f$$，指数偏置为 1。对非零指数，数值是：

$$
v=(-1)^s2^{e-1}\left(1+\frac f2\right),
\qquad e\in\{1,2,3\}.
$$

指数为零时，没有隐含的前导 1，数值变为：

$$
v=(-1)^s\frac f2.
$$

把符号位暂时固定为零，剩下三位恰好给出下面的表。

| 三位幅值编码 | 指数 | 尾数 | 数值 |
| --- | --- | --- | ---: |
| 000 | 00 | 0 | 0 |
| 001 | 00 | 1 | 0.5 |
| 010 | 01 | 0 | 1 |
| 011 | 01 | 1 | 1.5 |
| 100 | 10 | 0 | 2 |
| 101 | 10 | 1 | 3 |
| 110 | 11 | 0 | 4 |
| 111 | 11 | 1 | 6 |

最高位改成 1，就得到对应的负数，包括负零。因此，E2M1 有 16 种位模式，但只有 15 个不同的实数值。最小正规数是 1，唯一的正非正规数是 0.5，最大有限幅值是 6。这里没有为 Inf 或 NaN 保留元素编码；不能照搬“全 1 指数一定表示特殊值”的习惯，否则会错误地删掉 4 和 6。[OCP v1.0 第 5.3.3 节及表 5](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)规定了这些编码。

非均匀间隔是浮点格式的重要特征。靠近零时相邻数值相差 0.5，到了 4 与 6 之间，间隔已经变成 2。浮点格式并没有消灭量化误差，只是把有限的编码分配到了不同数量级。

### 中点的偶数是编码低位的偶数

最近偶数舍入不能理解成“最终十进制数要是偶数”。例如 0.75 位于 0.5 和 1 之间，应选择编码为 `010` 的 1；1.25 位于 1 和 1.5 之间，也选择 1。决定中点去向的是保留编码的最低位。

七个正中点的结果依次为：

$$
\begin{aligned}
&0.25,\ 0.75,\ 1.25,\ 1.75,\ 2.5,\ 3.5,\ 5\\
&\quad\longrightarrow
0,\ 1,\ 1,\ 2,\ 2,\ 4,\ 4.
\end{aligned}
$$

下面的程序枚举所有位模式，并检查中点、双零和字节打包。它明确采用“有限输入、最近偶数、越界饱和、保留零的符号”的参考约定；NaN 和 Inf 被拒绝。格式没有 NaN 编码，不等于所有转换器都应该用同一种方式处理 NaN 输入。

<!-- example: nvfp4-mxfp4-e2m1-codes-ties-packing -->
```python
import math

MAG = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

def decode4(code):
    if type(code) is not int or not 0 <= code < 16:
        raise ValueError("expected a four-bit code")
    return math.copysign(MAG[code & 7], -1.0 if code & 8 else 1.0)

def encode4(value):
    if not math.isfinite(value):
        raise ValueError("finite input required")
    sign = 8 if math.copysign(1.0, value) < 0 else 0
    magnitude = min(abs(value), MAG[-1])
    # The second key selects the even retained code at an exact midpoint.
    code = min(range(8), key=lambda c: (abs(MAG[c] - magnitude), c & 1))
    return sign | code

def pack4(codes):
    if len(codes) % 2:
        raise ValueError("this reference requires an even logical length")
    for code in codes:
        decode4(code)
    return bytes(codes[i] | (codes[i + 1] << 4)
                 for i in range(0, len(codes), 2))

def unpack4(data):
    if not isinstance(data, (bytes, bytearray)):
        raise ValueError("bytes or bytearray required")
    return [code for byte in data for code in (byte & 15, byte >> 4)]

codes = list(range(16))
assert [encode4(decode4(c)) for c in codes] == codes
assert math.copysign(1.0, decode4(8)) == -1.0
midpoints = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
rounded = [decode4(encode4(x)) for x in midpoints]
assert rounded == [0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0]
assert decode4(encode4(100.0)) == 6.0
assert decode4(encode4(-100.0)) == -6.0
assert decode4(encode4(float.fromhex("0x1.fffffffffffffp+1023"))) == 6.0
packed = pack4(codes)
assert unpack4(packed) == codes
for bad in (float("nan"), float("inf"), -float("inf")):
    try:
        encode4(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("non-finite input was accepted")
try:
    pack4([1, 2, 3])
except ValueError:
    pass
else:
    raise AssertionError("odd length was accepted")
print("values:", [decode4(c) for c in codes])
print("midpoints:", rounded)
print("packed:", packed.hex(" "))
```

最后一行输出 `10 32 54 76 98 ba dc fe`：偶数位置的元素放低四位，奇数位置的元素放高四位。这是本例的布局约定，也是后文所引用 ModelOpt 普通 NV 打包路径的相邻元素顺序，不是所有 FP4 内核必须接受的布局。

双零还要分清“格式能表示”与“转换器会保留”。例如 ModelOpt 0.35.0 的普通 NV 转换用 `x < 0` 提取符号，输入负零不会因此得到负零编码；负的非零小数舍入到零，却仍可带负号。上面的程序读取 sign bit，因而不承诺与该路径逐位相同。[固定版本的 NV 转换实现](https://github.com/NVIDIA/Model-Optimizer/blob/c359cb774db4daab623cddde6c7e0399a49cf7d2/modelopt/torch/quantization/qtensor/nvfp4_tensor.py)展示了这一区别。

## 2. 共享 scale 把很小的码本扩展到真实张量

单独的 E2M1 只能表示幅值不超过 6 的数。引入正 scale 后，一个元素的重构值成为：

$$
\hat x_i=s_g p_i,
$$

其中 $$p_i$$ 是解码后的 E2M1 数值，而不是四位编码作为整数时的值；$$g$$ 表示它所在的块。例如 $$s_g=0.125$$ 时，正数码本成为：

$$
0,\ 0.0625,\ 0.125,\ 0.1875,\ 0.25,\ 0.375,\ 0.5,\ 0.75.
$$

同一个元素编码，放在另一个 scale 的块中就会表示另一个真实数值。共享 scale 可以摊薄元数据成本，但同块元素必须共同适应这套格点。把差异很大的数分在一起，仍会牺牲较小数值的分辨率。

MXFP4 与 NVFP4 的主要区别，可以放在同一层次比较。

| 项目 | MXFP4 OCP v1.0 | NVFP4 本文所用定义 |
| --- | --- | --- |
| 元素 payload | E2M1 | E2M1 |
| 每块元素数量 | 32 | 16 |
| 局部 scale | E8M0 | E4M3 |
| 局部 scale 是否限于 2 的整数次幂 | 是 | 否 |
| 张量级 FP32 scale | 该具体格式不要求 | 另有一个 |
| 完整块的主体成本 | 4.25 bit/元素 | 4.5 bit/元素，另计张量级 scale |

MXFP4 的 E8M0 没有符号位和尾数位。有限编码 $$c\in\{0,\ldots,254\}$$ 表示：

$$
s_g=2^{c-127}.
$$

编码 255 表示 NaN；没有零和 Inf。因此，编码 0 是 $$2^{-127}$$，不是 scale 为零。全零块仍可以用合法的正 scale 配全零 payload 表达。[OCP 第 5.2 节表 1 与第 5.4.1 节表 7](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)明确了 MXFP4 的 32 元素块和 E8M0。

NVFP4 则给每 16 个元素配置一个 E4M3 scale，并用张量级 FP32 scale 调整整体范围。E4M3 有尾数，能表达 0.625、0.875 这样的非 2 幂数值，但有限正数范围远小于 E8M0：本文使用的是最大值为 448、最小正非正规数为 $$2^{-9}$$ 的 E4M3 编码，对应所引代码的 E4M3FN，而不是另一种 E4M3FNUZ 变体，二者区别见 [FP8 格式篇]({% post_url 2026-08-01-fp8-formats-scaling-llm-inference %})。张量级 scale 正是用来把不同数量级的张量适配到局部 scale 范围中，而不是把每个元素都存成 FP8。[NVIDIA 对两级缩放的说明](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)给出了这一设计。

较小的块和更灵活的 scale 增加了表达局部分布的机会，但它们不是“任意张量上误差必然更小”的定理。scale 自己也要舍入，块内还会发生 E2M1 舍入，二者必须一起分析。

## 3. NVFP4 的两个 scale 怎样完成一次量化

固定 ModelOpt 0.35.0 的普通权重路径，令张量绝对最大值为 $$A$$，块内绝对最大值为 $$A_g$$。在各中间尺度编码后均有限且严格为正的范围内，其 max-based 思路可写成：

$$
a=\frac{A}{6\cdot448},
\qquad
b_g^\star=\frac{A_g}{6a},
\qquad
b_g=Q_{\mathrm{E4M3}}(b_g^\star).
$$

随后才量化元素：

$$
p_i=Q_{\mathrm{E2M1}}\left(\frac{x_i}{ab_g}\right),
\qquad
\hat x_i=ab_gp_i.
$$

这里 $$a$$ 是张量级**反量化 scale**，$$b_g$$ 是已编码的局部反量化 scale。最大块的理想 $$b_g^\star$$ 为 448，其他块相对缩小。必须先把 $$b_g^\star$$ 编成真实的 E4M3 数值，再用它量化元素；如果编码时用理想 scale、解码时用舍入后的 scale，便已经是另一套数值算法。[ModelOpt 0.35.0 的 NVFP4QTensor](https://github.com/NVIDIA/Model-Optimizer/blob/c359cb774db4daab623cddde6c7e0399a49cf7d2/modelopt/torch/quantization/qtensor/nvfp4_tensor.py)包含这条普通路径。

有些接口使用相反方向的命名。例如同版本 GEMM 路径的 `global_scale` 是：

$$
\gamma=\frac{6\cdot448}{A}=\frac1a.
$$

对应的输出乘子是：

$$
\alpha=\frac1{\gamma_X\gamma_W}=a_Xa_W.
$$

所以看到代码里“乘 global_scale”，不能不看定义就与本文“除以反量化 scale”对比。二者可以是同一个变换的互为倒数写法。[固定的 nvfp4_gemm.py](https://github.com/NVIDIA/Model-Optimizer/blob/c359cb774db4daab623cddde6c7e0399a49cf7d2/modelopt/torch/quantization/backends/nvfp4_gemm.py)直接展示了这一接口约定。

### 零与极小数需要独立的数值合同

全零张量没有必要执行 $$0/0$$。一个明确的参考约定是选择 $$a=1$$、所有 $$b_g=1$$、所有 payload 为零。非零张量中的全零块也可以直接使用正局部 scale。对于非零的极小块，如果 $$b_g^\star$$ 舍入成零，则必须选择有定义的最小 scale 或拒绝该输入，不能继续除以零。

下面的参考程序选择“全零显式处理、非零 scale 下溢则拒绝”，并只接受最后维已经组成完整块的一维输入。它还把全局 $$a$$ 真正舍入到 FP32，检查舍入之后是否仍为有限正数；其余标量运算使用 Python 的双精度浮点，并非逐步模拟官方 FP32 运算。这些是程序的边界条件，而不是声称每条官方后端已经作出同样选择。ModelOpt 的普通 packed 路径、Triton fake quant 路径和 TensorRT 路径在零值保护及布局上不能直接视为逐位等价。[普通路径](https://github.com/NVIDIA/Model-Optimizer/blob/c359cb774db4daab623cddde6c7e0399a49cf7d2/modelopt/torch/quantization/qtensor/nvfp4_tensor.py)、[Triton 路径](https://github.com/NVIDIA/Model-Optimizer/blob/c359cb774db4daab623cddde6c7e0399a49cf7d2/modelopt/torch/quantization/triton/fp4_kernel.py)分别定义了自己的处理。

## 4. scale 量化误差会再次改变元素格点

设同一张量的另一块含有最大值 2688，因此 $$a=2688/(6\cdot448)=1$$。观察下面八个数，将它们重复两次组成一个 16 元素块：

$$
[0,\ 0.3,\ 0.6,\ 0.9,\ 1.2,\ 1.8,\ 2.4,\ 3.6].
$$

理想局部 scale 为 $$3.6/6=0.6$$。如果它能精确保存，每个输入恰好是 E2M1 码本乘以 0.6。但 0.6 不是 E4M3 数值，最近的候选是 0.625。选择该 scale 后，整个块的格点都会移动。

| 采用的局部 scale | 前八个元素的重构值 | MSE |
| --- | --- | ---: |
| 理想实数 0.6 | 与输入一致，只有计算浮点尾差 | 约 0 |
| E4M3 的 0.625 | 0, 0.3125, 0.625, 0.9375, 1.25, 1.875, 2.5, 3.75 | 0.0053515625 |
| E8M0 候选 1 | 0, 0.5, 0.5, 1, 1, 2, 2, 4 | 0.0575 |

这比较的是同一个局部向量与指定 scale 候选，不是完整 MXFP4 与 NVFP4 的模型性能对比；真正的 MXFP4 还使用 32 元素块。这个例子要说明的是：**块 scale 的舍入本身就是一项误差来源，而且它会改变后续 payload 的选择。**

还要区分“准确编码 max-based scale”和“找到最小块 MSE”。仍取 $$a=1$$，让局部块为：

$$
[0.4,\ldots,0.4,\ 3.6],
$$

其中 0.4 有 15 个。按最大值计算，$$b_g^\star=0.6$$，编码后得到 0.625。此时 0.4 重构成 0.3125，3.6 重构成 3.75，MSE 为：

$$
\frac{15(0.3125-0.4)^2+(3.75-3.6)^2}{16}
=\frac{879}{102400}
=0.008583984375.
$$

另一个合法 E4M3 scale 0.875 会把它们重构为 0.4375 和 3.5，得到：

$$
\frac{15(0.4375-0.4)^2+(3.5-3.6)^2}{16}
=\frac{199}{102400}
=0.001943359375.
$$

它比 max-based 的结果更好。这并不否定默认 encoder 的规则，而是说明：一种格式允许哪些数值，与校准算法怎样搜索这些数值，是两回事。下面完整实现上述参考计算；不需要 Torch 或第三方库。

<!-- example: nvfp4-mxfp4-scales-error-and-encoder -->
```python
import math
import struct

MAG = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

def q4(value):
    if not math.isfinite(value):
        raise ValueError("finite normalized payload required")
    magnitude = min(abs(value), MAG[-1])
    c = min(range(8), key=lambda i: (abs(MAG[i] - magnitude), i & 1))
    return c + (8 if math.copysign(1.0, value) < 0 else 0)

def d4(code):
    if type(code) is not int or not 0 <= code < 16:
        raise ValueError("expected a four-bit code")
    return math.copysign(MAG[code & 7], -1.0 if code & 8 else 1.0)

def positive_e4m3(code):
    # Positive finite codes 0..126; 127 is NaN, not a scale candidate.
    if type(code) is not int or not 0 <= code <= 126:
        raise ValueError("positive finite E4M3 code required")
    exponent, fraction = divmod(code, 8)
    return fraction * 2.0**-9 if exponent == 0 else (
        2.0**(exponent - 7) * (1.0 + fraction / 8.0))

E4 = tuple(positive_e4m3(c) for c in range(127))

def scale_code(value):
    if not math.isfinite(value) or value < 0:
        raise ValueError("finite nonnegative scale required")
    # This reference uses nearest-even plus finite saturation.
    value = min(value, E4[-1])
    return min(range(127), key=lambda c: (abs(E4[c] - value), c & 1))

def reconstruct(values, scale):
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("positive scale required")
    output = [scale * d4(q4(x / scale)) for x in values]
    if not all(math.isfinite(x) for x in output):
        raise ValueError("reconstruction outside scalar range")
    return output

def mse(left, right):
    if not left or len(left) != len(right):
        raise ValueError("equal nonempty vectors required")
    if not all(math.isfinite(x) for x in [*left, *right]):
        raise ValueError("finite vectors required")
    squared = [(x - y) * (x - y) for x, y in zip(left, right)]
    total = sum(squared)
    if not math.isfinite(total) or any(
            x != y and error == 0 for x, y, error in zip(left, right, squared)):
        raise ValueError("squared error outside scalar range")
    result = total / len(left)
    if total > 0 and result == 0:
        raise ValueError("mean squared error underflow")
    return result

def nv_reference(values):
    if not values or len(values) % 16:
        raise ValueError("nonempty complete 16-element blocks required")
    if not all(math.isfinite(x) for x in values):
        raise ValueError("finite input required")
    amax = max(abs(x) for x in values)
    if amax == 0:
        return 1.0, [scale_code(1.0)] * (len(values) // 16), [
            q4(x) for x in values], list(values)
    try:
        a = struct.unpack("!f", struct.pack("!f", amax / (6 * 448)))[0]
    except OverflowError as exc:
        raise ValueError("global scale outside FP32") from exc
    if not math.isfinite(a) or a <= 0:
        raise ValueError("global scale underflow or overflow")
    scales, codes, output = [], [], []
    for start in range(0, len(values), 16):
        block = values[start:start + 16]
        block_amax = max(abs(x) for x in block)
        c = scale_code(block_amax / (6 * a)) if block_amax else scale_code(1.0)
        b = E4[c]
        if b == 0:
            raise ValueError("nonzero block scale rounded to zero")
        factor = a * b
        block_codes = [q4(x / factor) for x in block]
        rebuilt = [factor * d4(code) for code in block_codes]
        if not all(math.isfinite(x) for x in rebuilt):
            raise ValueError("reconstruction outside scalar range")
        scales.append(c)
        codes.extend(block_codes)
        output.extend(rebuilt)
    return a, scales, codes, output

pattern = [0.0, 0.3, 0.6, 0.9, 1.2, 1.8, 2.4, 3.6] * 2
a, scales, _, output = nv_reference([2688.0] * 16 + pattern)
assert a == 1.0 and E4[scales[1]] == 0.625
for scale, expected in ((0.6, 0.0), (0.625, 0.0053515625), (1.0, 0.0575)):
    error = mse(pattern, reconstruct(pattern, scale))
    assert math.isclose(error, expected, rel_tol=1e-12, abs_tol=1e-28)
    print("scale / MSE:", scale, error)
assert output[16:] == reconstruct(pattern, 0.625)

counterexample = [0.4] * 15 + [3.6]
_, scales, _, output = nv_reference([2688.0] * 16 + counterexample)
default_error = mse(counterexample, output[16:])
alternative_error = mse(counterexample, reconstruct(counterexample, 0.875))
assert E4[scale_code(0.875)] == 0.875
assert math.isclose(default_error, 879 / 102400)
assert math.isclose(alternative_error, 199 / 102400)
assert alternative_error < default_error
assert nv_reference([0.0] * 16)[3] == [0.0] * 16
assert nv_reference([2688.0] * 16 + [0.0] * 16)[3][16:] == [0.0] * 16

bad_cases = ([1.0] * 17, [float("nan")] * 16, [1e-50] * 16,
             [2688.0] * 16 + [1e-10] * 16)
for values in bad_cases:
    try:
        nv_reference(values)
    except ValueError:
        pass
    else:
        raise AssertionError("input outside the reference contract was accepted")
print("max-based / alternative MSE:", default_error, alternative_error)
print("zero, shape and tiny-scale checks: passed")
```

这里通过另一个块确定 $$a=1$$，不是把每个局部块都单独当作一个张量。如果每个块重新计算全局 scale，问题本身就变了。

## 5. MXFP4 也不是只有一种 scale 选择算法

E8M0 限制了 scale 的可表示集合，却没有唯一规定怎样根据一组输入选择它。OCP 的格式定义将详细的 scale 算法排除在范围之外；第 6.3 节给出的推荐转换方案，是把块最大值向下取整到 2 的幂，再除以元素格式中最大的 2 幂。对 E2M1，后者是 4，而不是最大值 6。[OCP 的范围及转换章节](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)同时允许其他算法。

例如块最大值为 7，按上述方案会取：

$$
s_g=\frac{2^{\lfloor\log_2 7\rfloor}}4
=\frac44=1.
$$

归一化后的 7 超过 E2M1 最大值 6，饱和重构为 6。这个方案并不保证永远没有 clipping。

ModelOpt 0.35.0 的 MXFP4 普通 packed 路径则采用：

$$
s_g=2^{\lceil\log_2(A_g/6)\rceil},
$$

并限制最小指数。对于最大值 7，这会取 scale 为 2，把覆盖范围扩到 12，但格点也变粗了。这是偏向避免范围溢出的 encoder 选择，不是“MXFP4 格式必须这样取 scale”。此外，该版本普通 MX packed 转换在中点的处理与前面的独立 RNE 参考约定不同，不能仅凭格式名就宣称不同软件路径逐位相同。[固定的 mxfp4_tensor.py](https://github.com/NVIDIA/Model-Optimizer/blob/c359cb774db4daab623cddde6c7e0399a49cf7d2/modelopt/torch/quantization/qtensor/mxfp4_tensor.py)包含具体算法。

因此，讨论一个量化结果时，至少要同时说明格式和 encoder。格式决定合法的重构值，encoder 决定到底选哪一组。AWQ 搜索、裁剪、校准数据和误差目标仍有作用，不会因为 payload 变成浮点就自动失去意义。

## 6. 分组轴决定 scale 能在矩阵乘里提出多远

沿用 $$Y=XW$$，其中 $$X\in\mathbb R^{M\times K}$$，$$W\in\mathbb R^{K\times N}$$。设输入在每一行的 K 维分块，权重在每一列的 K 维分块，并且两者块边界对齐。对 NVFP4，可写成：

$$
\hat X_{mk}=a_Xb^X_{mg}p^X_{mk},
\qquad
\hat W_{kn}=a_Wb^W_{gn}p^W_{kn},
\quad k\in g.
$$

一个输出元素于是变成：

$$
\hat Y_{mn}
=a_Xa_W\sum_g
\left(
b^X_{mg}b^W_{gn}
\sum_{k\in g}p^X_{mk}p^W_{kn}
\right).
$$

scale 只能从**同一个块内部的点积**提出。不同块的乘积 $$b^X_{mg}b^W_{gn}$$ 不同，就不能把它们合并成一个输出外积，再去乘整个 K 维的一次点积。两边分组不一致时，应按它们的共同细分区域处理，而不是强行套用同一组号。

例如有两个 16 元素块，payload 点积分别为 16 和 -32，输入块 scale 为 1、2，权重块 scale 为 0.5、2，全局 scale 乘积为 1。正确结果是：

$$
1\cdot0.5\cdot16+2\cdot2\cdot(-32)=-120.
$$

如果错误地把第一组 scale 当作整个 K 维的 scale，就会算成 $$0.5(16-32)=-8$$。这不是微小舍入误差，而是改变了运算。

MXFP4 的分组点积同理，只是没有上述 NV 张量级 scale。OCP 规定了共享 scale 与块点积的数值关系，但把内部累加精度和运算顺序留给实现；精确实数等式不意味着所有 GPU 内核与逐元素浮点反量化后相乘都逐位一致。[OCP 第 6.1–6.2 节](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)给出了这些边界。

### 转置不仅改变形状 还可能改变同组元素

如果一个矩阵按每行连续 32 个元素做 MX 分组，转置后再按行分组，组成员通常已经变了。因此一般不能认为：

$$
Q(W)^T=Q(W^T).
$$

下面使用一个真正的 $$32\times32$$ 矩阵：除左上角为 6，其余全部为 0.3。每行取前述 ceil-power-of-two scale，并使用独立的 RNE payload 参考。先量化再转置，在位置 $$[0,1]$$ 得到 0.25；先转置再量化，在同一位置得到 0.5。区别来自 0.3 是否与 6 共享同一块，不是转置公式出错。

前一种计算中，该元素所在的原矩阵行全是 0.3，scale 为 $$2^{-4}=0.0625$$，归一化数值 4.8 舍入成 4，重构为 0.25。后一种计算中，它与 6 共享 scale 1，归一化数值 0.3 舍入成 0.5。只转置 payload，不重新解释其对应的分组和 scale，不能把一种结果变成另一种。

这也解释了为什么权重按 $$N\times K$$ 存储时，源码的“最后维分组”可以对应数学权重 $$K\times N$$ 的每列 K 分组。但若一个方法直接 `view(-1, G)`，它只是在展平顺序上切块。要把它解释成逐行 K 分组，必须额外保证每行最后维可整除 G，不能只检查总元素数。

下面的 `mx_scale` 实现数学上的 ceil-power-of-two 规则，先用 `frexp` 定位指数，再比较 $$6\cdot2^e$$ 阈值，避免 `log2` 在临界点附近舍回整数。它拒绝需要超过 E8M0 最大指数的输入；这仍是独立 CPU 参考，不是对固定 ModelOpt 路径的逐位模拟。

<!-- example: nvfp4-mxfp4-groups-transpose-and-byte-budget -->
```python
import math

MAG = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

def qdq4(value):
    if not math.isfinite(value):
        raise ValueError("finite payload required")
    magnitude = min(abs(value), MAG[-1])
    code = min(range(8), key=lambda c: (abs(MAG[c] - magnitude), c & 1))
    return math.copysign(MAG[code], value)

def mx_scale(amax):
    if not math.isfinite(amax) or amax < 0:
        raise ValueError("finite nonnegative maximum required")
    if amax == 0:
        return 1.0
    exponent = max(-127, math.frexp(amax)[1] - 3)
    if amax > 6.0 * 2.0**exponent:
        exponent += 1
    if exponent > 127:
        raise ValueError("E8M0 range exceeded")
    return 2.0**exponent

def mx_rows(matrix):
    # Independent RNE reference, not a bit-exact ModelOpt exporter.
    if not matrix or not matrix[0] or len(matrix[0]) % 32:
        raise ValueError("complete row-wise groups of 32 required")
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise ValueError("rectangular matrix required")
    if not all(math.isfinite(x) for row in matrix for x in row):
        raise ValueError("finite matrix required")
    output = []
    for row in matrix:
        rebuilt = []
        for start in range(0, width, 32):
            block = row[start:start + 32]
            amax = max(abs(x) for x in block)
            scale = mx_scale(amax)
            rebuilt.extend(scale * qdq4(x / scale) for x in block)
        output.append(rebuilt)
    return output

def transpose(matrix):
    if not matrix or not matrix[0] or any(
            len(row) != len(matrix[0]) for row in matrix):
        raise ValueError("nonempty rectangular matrix required")
    return [list(column) for column in zip(*matrix)]

weight = [[0.3] * 32 for _ in range(32)]
weight[0][0] = 6.0
first_quantize = transpose(mx_rows(weight))
first_transpose = mx_rows(transpose(weight))
assert first_quantize[0][1] == 0.25
assert first_transpose[0][1] == 0.5
print("transpose counterexample:", first_quantize[0][1], first_transpose[0][1])

group_size = 16
px, pw = [1.0] * 16 + [2.0] * 16, [1.0] * 16 + [-1.0] * 16
bx, bw = [1.0, 2.0], [0.5, 2.0]
ax, aw = 0.5, 2.0
group_dots = [sum(px[k] * pw[k] for k in range(g * 16, (g + 1) * 16))
              for g in range(2)]
group_result = ax * aw * sum(bx[g] * bw[g] * group_dots[g] for g in range(2))
direct_result = sum((ax * bx[k // group_size] * px[k]) *
                    (aw * bw[k // group_size] * pw[k]) for k in range(32))
wrong_result = ax * aw * bx[0] * bw[0] * sum(group_dots)
assert group_result == direct_result == -120.0
assert wrong_result == -8.0
print("group / direct / wrong dot:", group_result, direct_result, wrong_result)

def storage_bytes(elements, group_size, global_bytes):
    if (type(elements) is not int or elements <= 0 or
            type(group_size) is not int or group_size not in (16, 32) or
            type(global_bytes) is not int or global_bytes < 0 or
            elements % group_size):
        raise ValueError("this budget requires complete blocks")
    return elements // 2 + elements // group_size + global_bytes

mx_bytes = storage_bytes(1024, 32, 0)
nv_bytes = storage_bytes(1024, 16, 4)
assert (mx_bytes, nv_bytes) == (544, 580)
assert storage_bytes(16, 16, 4) == 13
print("1024 elements, MX / NV bytes:", mx_bytes, nv_bytes)
print("isolated 16-element NV tensor, bits/value:", 13 * 8 / 16)
```

## 7. 格式正确不等于字节可以交给任意内核

E2M1 定义四位怎样解码，块格式定义哪些元素共享 scale；二者还没有规定 GPU 喜欢怎样读取字节。OCP 明确不规定块在物理内存中的布局，scale 可以与 payload 连续存储，也可以分开存储。[OCP 第 5.1 节](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)区分了数值格式与物理布局。

同样是 NVFP4，普通参考打包与 TensorRT 产生的 scale buffer 也未必相同。固定 ModelOpt 的解码路径遇到 TensorRT 布局时需要专门转换 scale 布局。因此，把 payload 的高低四位顺序调对，只完成了布局合同的一部分；scale 的顺序、分组轴、张量形状和 kernel tile 的要求也要一致。[NVFP4QTensor 的 TensorRT 分支与解码](https://github.com/NVIDIA/Model-Optimizer/blob/c359cb774db4daab623cddde6c7e0399a49cf7d2/modelopt/torch/quantization/qtensor/nvfp4_tensor.py)说明了这种实际差异。

尾块同样不是“补几个零”就结束。若按行补齐最后维，解码必须先恢复补齐后的布局，再按原逻辑形状逐行裁剪；原始 shape 作为 metadata 保存下来，并不自动让一次 reshape 删除 padding。本文程序因此直接拒绝不完整块，第一段打包也拒绝奇数长度，没有提供可任意补齐和导出的 checkpoint 格式。

硬件支持 FP4 也不能证明某个高层张量乘法已经使用 FP4 Tensor Core。以本文固定的 ModelOpt 0.35.0 为例，NV GEMM 入口还检查后端导入、GPU、RealQuantLinear、具体 recipe 和 block16，并要求该路径的输入第二维为 64 的倍数、权重第一维为 32 的倍数。这是**这个版本入口的条件**，不是所有 Blackwell 算子的统一形状定律。[固定 GEMM 入口检查](https://github.com/NVIDIA/Model-Optimizer/blob/c359cb774db4daab623cddde6c7e0399a49cf7d2/modelopt/torch/quantization/backends/nvfp4_gemm.py)比仅看张量 dtype 提供了更具体的信息。

与上一篇 [QServe 的 W4A8]({% post_url 2026-08-04-w4a8-progressive-quantization-kernel %}) 对照，这个区别尤其重要：那里可以用 INT4 存储、重建 INT8 后执行 INT8 GEMM；这里的 E2M1 payload 加块 scale 是另一套数值语义。四位存储相同，不代表计算指令和尺度处理相同。

## 8. 四位 payload 的实际成本不止四位

设一个张量有 $$T$$ 个元素，且完全由完整块组成。不计算任何对齐和布局附加项时，MXFP4 的主体成本为：

$$
4T+8\frac T{32}
=4.25T\ \mathrm{bits}.
$$

NVFP4 则还要计算每块的 E4M3 scale 和一个 FP32 张量级 scale：

$$
4T+8\frac T{16}+32
=4.5T+32\ \mathrm{bits}.
$$

也就是每元素：

$$
4.5+\frac{32}{T}\ \mathrm{bits}.
$$

前面的程序给出：1024 个元素的 MX 主体为 544 字节，NV 主体为 580 字节；只有 16 个元素的独立 NV 张量则需要 8 字节 payload、1 字节块 scale、4 字节全局 scale，合计 13 字节，即每元素 6.5 bit。张量越大，全局 scale 越容易摊薄，但块 scale 的成本不会消失。

如果选择逐行补齐的布局，$$R\times K$$ 矩阵需要的块数是 $$B=R\lceil K/G\rceil$$，不是先展平再计算的 $$\lceil RK/G\rceil$$。只计完整块的 payload 与局部 scale，主体字节数为 $$B(G/2+1)$$，NV 再为一个全局 scale 加 4 字节。例如两行、每行 17 个值，NV 按 G=16 补齐后共有 4 块，需要 $$4(8+1)+4=40$$ 字节。这是指定 padding 布局下的账单，不是前面拒绝尾块的程序已经实现了该 exporter。

这些仍不是整个模型的进程显存。尾部 padding、scale 重排与对齐、文件头、未量化层和运行时 workspace 都可能增加成本。E8M0 的大范围也不代表任意 scale 与 payload 的乘积都能放进 FP32：例如 scale 为 $$2^{127}$$ 时，乘以 E2M1 的 6 已超出 FP32 最大有限值；OCP 对这种越界表示保留实现定义行为，不能把 scale 有限误写成所有重构值都有限。

NVFP4 和 MXFP4 的核心不是“用四位浮点直接替换所有数值”，而是用**元素码本、共享尺度和分组运算共同组成一种张量表示**。格式决定可表示什么，encoder 决定怎样逼近原值，布局和 kernel 决定这些字节怎样参与计算。本文的标准库程序验证的是这些数值关系、反例和字节账单，不是官方 Torch/CUDA 后端、模型质量或 GPU 性能复现；涉及 KV Cache 的部署问题，可以继续阅读 [NVFP4 KV Cache 专题]({% post_url 2026-06-02-nvfp4-kv-cacheblackwell %})。
