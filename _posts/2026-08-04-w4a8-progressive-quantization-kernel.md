---
layout: post
title: "W4A8：4-bit Weight 怎样喂给 8-bit Tensor Core"
subtitle: "从两级整数表示、119 保护范围与 byte lane 到 QServe 的 GEMM 和 SmoothAttention"
date: 2026-08-04 09:00:00 +0800
last_modified_at: 2026-10-15
author: iStar
catalog: true
series: model-quantization
series_order: 100
technology_year: 2024
mathjax: true
tags: [模型量化, GPU优化, LLM推理]
---

4-bit 权重首先节省的是存储与读取字节数，但读取出来的 nibble，不一定就是矩阵乘指令能够直接消费的操作数。QServe 的整数 W4A8 路线选择了一条明确的桥：把 packed UINT4 在寄存器中恢复成 signed INT8，与 INT8 激活相乘、用 INT32 累加，最后结合尺度输出 FP16。

这条桥不能只在数学上成立。恢复每组权重的移位、乘法、零点修正如果太昂贵，Tensor Core 就会等待操作数，省下的带宽反而被转换开销吃掉。因此，QServe 把量化规则、权重布局和 CUDA 指令放在一起设计。

下面讨论的是 QServe 论文在 A100/L40S 上研究的整数 W4A8，以及一个固定版本的官方 kernel。它不是所有名为 W4A8 的实现都采用的通用路线；后面会看到，FP8 activation 的同名 recipe 就有另一套数值合同。

## 1. 同一层计算中有四种不同的精度

沿用本系列的布局：

$$
Y_{M\times N}=X_{M\times K}W_{K\times N}.
$$

$$M$$ 是本次参与计算的 token 行数，$$K$$ 是输入与 reduction 维，$$N$$ 是输出通道。decode 的小 $$M$$ 计算往往需要反复读大权重；prefill 或更高并发使 $$M$$ 增大，同一权重被更多行复用，计算吞吐就更重要。

W4A16 减少权重流量，但典型路径要把权重恢复为 FP16/BF16；W8A8 能利用 INT8 计算，权重流量却仍是 8-bit。整数 W4A8 希望兼顾两者，但必须把格式转换成本算进去。QServe 论文的 Roofline 是这个设计的动机，不是对所有 GPU 和 shape 的速度保证。[QServe v3，§3.1–3.2](https://arxiv.org/html/2405.04532v3#S3)

具体到本文的固定 GEMM 路径，可以分清四层：

| 层次 | 表示 | 作用 |
| --- | --- | --- |
| 权重存储 | packed UINT4 | 降低权重容量和读取流量 |
| 乘法操作数 | INT8 激活与恢复出的 INT8 权重 | 交给目标 MMA 指令 |
| 点积累加 | INT32 | 容纳整数乘加的部分和 |
| 输出 | 结合尺度转换为 FP16 | 交给后续浮点算子 |

沿 $$K$$ 迭代的 GEMM main loop 会反复加载和处理操作数。随 group 改变的参数不能随意拖到最后再处理；可以延后的参数，则应尽量避免在这个循环里重复支付。这正是两级表示与 per-channel/per-group 路径差异的出发点。

## 2. 两级量化的目标是恢复 INT8 而不是恢复 FP16

对每个输出通道，先用浮点尺度 $$s^{(0)}_n$$ 得到一个 INT8 中间目标 $$q^{(0)}_{kn}$$。第二级把这个目标按 $$K$$ 轴分组，压成 UINT4 code $$u_{kn}$$：

$$
W_{kn}\approx s^{(0)}_n q^{(0)}_{kn},\qquad
q^{(0)}_{kn}\approx s^{(1)}_{g(k),n}\left(u_{kn}-z_{g(k),n}\right).
$$

其中 $$s^{(1)}$$ 是整数 group scale，$$z$$ 是整数零点，$$u\in\{0,\ldots,15\}$$。因此最终重建为：

$$
\widehat W_{kn}
=s^{(0)}_n s^{(1)}_{g(k),n}\left(u_{kn}-z_{g(k),n}\right).
$$

$$g(k)$$ 表示输入通道所在组。两个尺度的轴不一样：$$s^{(0)}$$ 沿输出通道定义，$$s^{(1)}$$ 随 $$K$$ 上的组变化。第二级也不保证无损恢复第一次的 INT8 codes，所以不能因为公式中间出现过 INT8，就把最终权重称为第一次量化的精确结果。[QServe v3，§4.1](https://arxiv.org/html/2405.04532v3#S4.SS1)

若激活已经表示为 $$\widehat X_{mk}=s^X_m a_{mk}$$，其中 $$a$$ 是 INT8 codes，则 per-group 路径计算：

$$
\widehat Y_{mn}
=s^X_m s^{(0)}_n\sum_k a_{mk}\widetilde q_{kn},\qquad
\widetilde q_{kn}=s^{(1)}_{g(k),n}\left(u_{kn}-z_{g(k),n}\right).
$$

主循环只需形成合法的 signed INT8 $$\widetilde q$$，不用完整恢复浮点权重。INT32 累加结束后，$$s^X_m s^{(0)}_n$$ 才进入 epilogue。

算一组具体数值。取浮点权重 $$[-1.19,-0.40,0.30,1.19]$$，第一级尺度为 $$s^{(0)}=0.01$$，得到目标 codes $$[-119,-40,30,119]$$。为说明第二级，暂采用独立的简化整数编码规则：

$$
s=\max\!\left(1,\operatorname{RNE}((h-l)/15)\right),\qquad
z=\operatorname{clip}\!\left(\operatorname{RNE}(-l/s),0,15\right),
$$

$$
u_i=\operatorname{clip}\!\left(\operatorname{RNE}(q_i/s)+z,0,15\right),\qquad
\widetilde q_i=(u_i-z)s,
$$

这里 $$l,h$$ 是本组目标 codes 的最小、最大值，RNE 为 ties-to-even。这个例子中 $$s=16,z=7$$：

| 项目 | 四个元素 |
| --- | --- |
| 第一级目标 $$q^{(0)}$$ | $$[-119,-40,30,119]$$ |
| 第二级 UINT4 $$u$$ | $$[0,5,9,14]$$ |
| 恢复后的 INT8 $$\widetilde q$$ | $$[-112,-32,32,112]$$ |
| 最终浮点权重 | $$[-1.12,-0.32,0.32,1.12]$$ |

再给定激活 codes $$a=[2,-1,3,4]$$、尺度 $$s^X=0.25$$，整数点积为 $$352$$，输出是 $$352\times0.25\times0.01=0.88$$。原始浮点权重与这组重建激活的点积为 $$0.92$$。这既展示了尺度怎样回到输出，也展示了第二级量化确实引入了误差。

上述规则是用于手算和证明范围的教学合同，不是完整的 QoQ 参数搜索器。例如纯正窄组 $$[100,119]$$ 在这套规则下会严重截断成 $$[15,15]$$，不能据此承担一般的质量优化。真实校准还涉及 clipping、通道变换和参数搜索；也不能把这几行 RNE 编码器当成官方转换器在任意原始权重上的逐位复现。

## 3. 为什么第一级要给第二级预留范围

即使第一级 codes 本身合法，第二级重建也可能越界。例如目标范围 $$[-113,120]$$，按上述规则得到 $$s=16,z=7$$。对 $$120$$ 先做 $$\operatorname{RNE}(120/16)=8$$，再加零点得到 $$u=15$$，重建就是 $$128$$，超过 signed INT8 的上界。

这里的顺序尤其重要：$$\operatorname{RNE}(7.5)+7=15$$，而 $$\operatorname{RNE}(7.5+7)=\operatorname{RNE}(14.5)=14$$。把零点移进 round，在 ties-to-even 下不总是同一规则。后者会重建成 $$112$$，不能用来替前者证明安全。

QServe 把第一级对称范围收紧为 $$[-119,119]$$。对刚才声明的简化规则，可以给出一个把 clamp 也考虑进去的完整证明，而不依赖随机测试：

1. 第一阶段整数 $$q\in[-119,119]$$，所以组跨度至多 $$238$$，按本例舍入规则有 $$1\le s\le16$$。
2. 先不考虑 UINT4 clamp，令 $$r=\operatorname{RNE}(q/s)s$$。最近值舍入给出 $$\lvert r-q\rvert\le s/2\le8$$，因此 $$r\in[-127,127]$$。
3. 实际重建 $$\widetilde q=(\operatorname{clip}(\operatorname{RNE}(q/s)+z,0,15)-z)s$$，等于把 $$r$$ 投影到 $$[-zs,(15-z)s]$$。因为 $$0\le z\le15$$，这个区间包含零；投影只会把越界的 $$r$$ 向零推，不会将其推到更远的同号极值。因此 $$\widetilde q$$ 仍在 $$[-127,127]$$。

注意，半步误差界对应第二步的**未 clamp 重建**；clamp 后不能无条件沿用同一个误差界。范围安全不等于误差必然很小，也不证明任意给定的 $$u,s,z$$ 组合都安全。例如 $$u=15,s=16,z=0$$ 重建为 $$240$$，就不满足最终 signed INT8 合同。保护范围必须与产生参数和 codes 的具体规则一起理解。[QServe v3，Protective Quantization Range](https://arxiv.org/html/2405.04532v3#S4.SS1)

下面的标准库例子完成两级数值计算，并枚举所有整数端点及组内整数值。整数 RNE 使用 `divmod`，避免把有限枚举的判断建立在浮点除法的巧合上。它只检查本节的教学合同，不运行 LMQuant 或 CUDA kernel；小 group 也不是下文固定 dispatch 所支持的原生形状。

<!-- example: w4a8-two-level-reference-and-protected-range -->
```python
from math import isclose, isfinite


def rne_ratio(numerator, denominator):
    if (type(numerator) is not int or type(denominator) is not int
            or denominator <= 0):
        raise ValueError("integer numerator and positive integer denominator required")
    quotient, remainder = divmod(numerator, denominator)
    up = 2 * remainder > denominator or (
        2 * remainder == denominator and quotient % 2 != 0)
    return quotient + int(up)


def clamp(value, lower, upper):
    return min(upper, max(lower, value))


def encode_group(codes):
    if not codes or not all(type(v) is int and -119 <= v <= 119 for v in codes):
        raise ValueError("nonempty protected integer code group required")
    lower, upper = min(codes), max(codes)
    scale = max(1, rne_ratio(upper - lower, 15))
    zero = clamp(rne_ratio(-lower, scale), 0, 15)
    unsigned = [clamp(rne_ratio(v, scale) + zero, 0, 15) for v in codes]
    restored = [(u - zero) * scale for u in unsigned]
    return unsigned, scale, zero, restored


weights = [-1.19, -0.40, 0.30, 1.19]
assert all(isfinite(v) for v in weights)
scale0 = max(abs(v) for v in weights) / 119
codes0 = [clamp(round(v / scale0), -119, 119) for v in weights]
u, scale1, zero, restored = encode_group(codes0)
assert codes0 == [-119, -40, 30, 119]
assert (u, scale1, zero, restored) == ([0, 5, 9, 14], 16, 7, [-112, -32, 32, 112])
activation_codes, activation_scale = [2, -1, 3, 4], 0.25
integer_sum = sum(a * w for a, w in zip(activation_codes, restored))
output = activation_scale * scale0 * integer_sum
assert integer_sum == 352 and isclose(output, 0.88, abs_tol=1e-12)
print("stage0 / u4 / restored:", codes0, u, restored, "output:", output)
assert encode_group([0, 0]) == ([0, 0], 1, 0, [0, 0])
assert encode_group([100, 119])[3] == [15, 15]  # safe does not mean accurate

# The two round orders are intentionally different at this tie.
assert rne_ratio(120, 16) + 7 == 15
assert rne_ratio(120 + 7 * 16, 16) == 14
assert (15 - 7) * 16 == 128

count, observed_min, observed_max = 0, 128, -129
for lower in range(-119, 120):
    for upper in range(lower, 120):
        scale = max(1, rne_ratio(upper - lower, 15))
        zero = clamp(rne_ratio(-lower, scale), 0, 15)
        assert 1 <= scale <= 16
        for target in range(lower, upper + 1):
            unsigned = clamp(rne_ratio(target, scale) + zero, 0, 15)
            value = (unsigned - zero) * scale
            assert -128 <= value <= 127
            count += 1
            observed_min = min(observed_min, value)
            observed_max = max(observed_max, value)
assert count == 2303960
print("enumerated cases / observed range:", count, observed_min, observed_max)
```

## 4. 一个 byte 中间值可以大于 127

安全范围还要和寄存器里的运算顺序对应。固定 per-group kernel 并不是先对每个 nibble 做 signed 减法、再逐个乘尺度，而是先展开为 byte lane，计算 $$u\times s$$，再加已经缩放的零点偏移 $$-zs$$。

在本节范围合同里 $$u\le15,s\le16$$，所以乘积至多是 $$240$$。**240 不是合法 signed INT8，但它是合法 unsigned byte。** 四个 UINT4 先分别放入一个 32-bit word 的四个 byte，同乘同一个小标量；只要每个乘积小于 256，就不会从一个 byte 乘出进位污染旁边的 lane。

随后用逐 byte 加法加入 $$(-zs)\bmod256$$。CUDA 的 `__vadd4` 分别对四个 byte 相加、忽略各自溢出，既不是饱和加法，也不是允许进位穿到相邻 byte 的普通 32-bit 加法。只有最终数学重建值在 signed INT8 域内，所得 byte 的有符号解释才对应预期的 MMA 操作数。[固定 per-group kernel](https://github.com/mit-han-lab/omniserve/blob/02b2925aa6fa3b92b06316a1524b7f38922cd9c8/kernels/csrc/qgemm/w4a8_per_group/gemm_cuda.cu)、[CUDA 12.8 SIMD 文档](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SIMD.html)

例如仅检查 lane 算术，取 $$u=15,s=16,z=8$$：先得到 unsigned $$240$$，再加偏移 byte $$128$$，按模 256 得到 $$112$$，正好等于 $$(15-8)16$$。这组局部参数用于说明 byte 运算，并不声称它们来自上一节简化编码器的某个实际 group。

真实 artifact 也不是“4-bit zero point 另占半个 byte”那么简单。固定版 Python 类的 per-group 分支里：

| 字段 | 数学含义 | 实际容器 |
| --- | --- | --- |
| `qweight` | 重排后打包的 UINT4 codes | int8 tensor，两个 nibble 一 byte |
| `s1_scales` | 本文的第一级 $$s^{(0)}$$ | 每输出通道 FP16 |
| `s2_scales` | 本文的整数 $$s^{(1)}$$ | 每组一 byte |
| `s2_zeros` | $$(-zs^{(1)})\bmod256$$，不是 raw $$z$$ | 每组一 byte 的二补码位模式 |

名称中的 `s1/s2` 与本文数学上从 0 开始的层级不同，读代码时不能错一层。[固定版 w4a8_linear.py](https://github.com/mit-han-lab/omniserve/blob/02b2925aa6fa3b92b06316a1524b7f38922cd9c8/omniserve/modeling/layers/quantized_linear/w4a8_linear.py)

布局还为目标 kernel 的消费顺序服务。该版本以 $$32\times32$$ 的逻辑权重 tile 重排、打包，scale 与 scaled-zero 也相应重排。按源码的 $$W_{\mathrm{store}}[n,k]$$ 方向看，一个 tile 开头的部分 byte 对应：

| tile 内 byte 下标 | low nibble | high nibble |
| --- | --- | --- |
| 0 | $$W_{\mathrm{store}}[0,0]$$ | $$W_{\mathrm{store}}[16,0]$$ |
| 4 | $$W_{\mathrm{store}}[8,0]$$ | $$W_{\mathrm{store}}[24,0]$$ |
| 8 | $$W_{\mathrm{store}}[0,16]$$ | $$W_{\mathrm{store}}[16,16]$$ |

所以第一个 byte 并不是逻辑矩阵里相邻输入通道的两个权重。数学公式正确而解包布局错误，仍会得到完全错误的矩阵。Python 构造器有 `group_size` 参数，也不代表 kernel 支持任意组大小：该固定 per-group dispatch 明确使用 $$G=128$$，packing 还有 tile 整除要求，不能把本文的小数值示例直接送进这个 kernel。

固定两条 GEMM 路径使用的指令包含：

```text
mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32
```

这明确区分了 INT4 存储、signed INT8 乘法和 INT32 累加。epilogue 再把累加值与尺度转成浮点输出，而不是从名字 W4A8 推断存在 UINT4×INT8 的直接乘法。[固定 per-group kernel](https://github.com/mit-han-lab/omniserve/blob/02b2925aa6fa3b92b06316a1524b7f38922cd9c8/kernels/csrc/qgemm/w4a8_per_group/gemm_cuda.cu)、[固定 per-channel kernel](https://github.com/mit-han-lab/omniserve/blob/02b2925aa6fa3b92b06316a1524b7f38922cd9c8/kernels/csrc/qgemm/w4a8_per_chn/gemm_cuda.cu)

## 5. 哪些修正可以移到 epilogue

该实现的 per-channel 分支不使用前面的整数第二级 group scale，而是直接表示整条输出通道的 UINT4 权重。如果一个输出通道共享同一个零点和权重尺度，代数展开是：

$$
\sum_k\widehat x_k(u_k-z)s_W
=s_W\sum_k\widehat x_ku_k-zs_W\sum_k\widehat x_k.
$$

前一项可以先做 INT8 dot product，后一项用该 token 的输入行和修正。这里的精确恒等式使用的是**量化重建输入的行和** $$\sum_k\widehat x_k$$。

QServe 的 per-channel 路径还做了一步额外选择：用未量化输入的行和 $$\sum_k x_k$$ 替代它。二者通常接近，却不是同一个数，所以从论文式 (12) 到式 (13) 不能整段称为严格等价。[QServe v3，§5.2.2](https://arxiv.org/html/2405.04532v3#S5.SS2.SSS2)

取 $$X=[1/10,1/5]$$、$$\widehat X=[1/8,1/4]$$、$$u=[2,3]$$、$$z=1,s_W=1/2$$。严格针对重建输入的结果为 $$5/16$$；保留第一项却改用原始 $$X$$ 行和修正后，结果为 $$7/20$$，差了 $$3/80$$。这个额外误差与 UINT4 的重建误差需要分开看。

固定版 `generalLayerNorm_fuse_sum` 的相关路径在 INT8 转换前累计归一化输出的行和，再传入后续 per-channel GEMM；这支持“不是 INT8 codes 行和”的读法，也不意味着浮点累计与存储是无舍入的精确实数运算。[固定版 layernorm_kernels.cu](https://github.com/mit-han-lab/omniserve/blob/02b2925aa6fa3b92b06316a1524b7f38922cd9c8/kernels/csrc/layernorm_kernels.cu)

per-group 路径则不同：$$s_g,z_g$$ 都沿 $$K$$ 改变，不能只减一次全局零点修正，也不能把所有 group scale 一起提到 GEMM 外面。最小的例子是 $$x=[1,2]$$、$$u-z=[1,1]$$，两个组的尺度分别为 $$2,3$$，结果是 $$1\times2+2\times3=8$$；错误地提取第一组尺度会得到 $$2(1+2)=6$$。

所以该固定 group kernel 在主循环中按组恢复 INT8，最后仅把 $$s^X_m s^{(0)}_n$$ 放到 epilogue。它的 `input_sum` 参数只是为了接口一致，并不用于这条 group 路径；不能拿 per-channel 行和修正替代实际 group 解码。[固定版 Linear 两个 forward 入口](https://github.com/mit-han-lab/omniserve/blob/02b2925aa6fa3b92b06316a1524b7f38922cd9c8/omniserve/modeling/layers/quantized_linear/w4a8_linear.py)

下面分别验证单个 $$32\times32$$ tile 的布局映射、byte lane 运算和上述修正差异。tile 示例只说明这份固定布局的局部索引，不是完整 checkpoint exporter，也不证明其他尺寸和内核可直接使用。

<!-- example: w4a8-tile-bytes-lanes-and-correction -->
```python
from fractions import Fraction


def tile_offset(n, k):
    if type(n) is not int or type(k) is not int or not (0 <= n < 32 and 0 <= k < 32):
        raise ValueError("integer indices within this 32 by 32 tile required")
    n1, n2, n3 = n // 16, (n // 8) % 2, n % 8
    k1, k2, k3 = k // 16, (k // 4) % 4, k % 4
    offset = ((((n3 * 4 + k2) * 2 + k1) * 2 + n2) * 4 + k3)
    return offset, 4 * n1


def pack_tile(values):
    if len(values) != 32 or not all(len(row) == 32 for row in values):
        raise ValueError("a 32 by 32 tile is required")
    packet = bytearray(512)
    for n in range(32):
        for k in range(32):
            code = values[n][k]
            if type(code) is not int or not 0 <= code <= 15:
                raise ValueError("integer UINT4 codes required")
            offset, shift = tile_offset(n, k)
            packet[offset] |= code << shift
    return bytes(packet)


def unpack_tile(packet):
    if not isinstance(packet, (bytes, bytearray)) or len(packet) != 512:
        raise ValueError("exactly 512 tile bytes required")
    restored = [[0] * 32 for _ in range(32)]
    for offset, byte in enumerate(packet):
        rest, k3 = divmod(offset, 4)
        rest, n2 = divmod(rest, 2)
        rest, k1 = divmod(rest, 2)
        n3, k2 = divmod(rest, 4)
        n, k = 8 * n2 + n3, 16 * k1 + 4 * k2 + k3
        restored[n][k], restored[n + 16][k] = byte & 15, byte >> 4
    return restored


tile = [[(5 * (n // 16) + 7 * (n % 16) + 7 * (k // 16) + 3 * (k % 16)) % 16
         for k in range(32)] for n in range(32)]
packet = pack_tile(tile)
assert unpack_tile(packet) == tile
for offset, low_n, high_n, k in [(0, 0, 16, 0), (4, 8, 24, 0), (8, 0, 16, 16)]:
    assert packet[offset] == tile[low_n][k] + (tile[high_n][k] << 4)
print("tile bytes 0 / 4 / 8:", [packet[i] for i in [0, 4, 8]])

# Four byte lanes share a scale, as in the local register operation.
unsigned, scale, zero = [0, 7, 8, 15], 16, 8
word = sum(u << (8 * i) for i, u in enumerate(unsigned))
product_word = word * scale
assert all(((product_word >> (8 * i)) & 255) == u * scale
           for i, u in enumerate(unsigned))  # no cross-byte carry
zero_byte = (-zero * scale) & 255
lane_bytes = [(((product_word >> (8 * i)) & 255) + zero_byte) & 255
              for i in range(4)]
signed = [b if b < 128 else b - 256 for b in lane_bytes]
assert signed == [(u - zero) * scale for u in unsigned] == [-128, -16, 0, 112]
normal_add = (product_word + sum(zero_byte << (8 * i) for i in range(4))) & 0xFFFFFFFF
lane_add = sum(b << (8 * i) for i, b in enumerate(lane_bytes))
assert normal_add != lane_add  # ordinary 32-bit addition is not __vadd4
print("unsigned product / final s8:", [u * scale for u in unsigned], signed)

x = [Fraction(1, 10), Fraction(1, 5)]
xhat = [Fraction(1, 8), Fraction(1, 4)]
u, z, sw = [2, 3], 1, Fraction(1, 2)
main = sw * sum(a * b for a, b in zip(xhat, u))
exact = main - z * sw * sum(xhat)
extra_approximation = main - z * sw * sum(x)
assert exact == Fraction(5, 16)
assert extra_approximation == Fraction(7, 20)
assert extra_approximation - exact == Fraction(3, 80)
assert 1 * 2 + 2 * 3 == 8 and 2 * (1 + 2) == 6
print("channel correction, exact / original-row-sum:", exact, extra_approximation)
```

## 6. SmoothAttention 为什么要尊重 RoPE 的二维配对

W4A8 是 Linear 的路径，KV4 则压缩 Attention 的历史状态。启用其中一个，不代表另一个自动获得同样的容量或计算收益。QServe 为低比特 Key 引入 SmoothAttention，把 Key 的部分幅度迁移给未量化的 Attention Query：

$$
Q'=Q\Lambda,\qquad K'=K\Lambda^{-1},\qquad
\Lambda=\operatorname{diag}(\lambda),\quad \lambda_i>0.
$$

因为 $$\Lambda$$ 是正的可逆对角矩阵：

$$
Q'(K')^T=QK^T.
$$

这里 Query 不量化，指 Attention 点积所用的 Query 表示，不是说生成 Query 的投影层输入也没有 A8 量化。论文的 block 边界保持 FP16，相关 Linear 输入仍有自己的 INT8 路径；KV4 的压缩与读取转换需要独立处理。[QServe v3，§4.2 与 §5.1](https://arxiv.org/html/2405.04532v3#S4.SS2)

若在 RoPE **之后**执行上述互逆缩放，任意正的对角 $$\Lambda$$ 都能在点积中抵消。但若想把缩放融合进 RoPE 之前的 Q/K 投影权重，就要求它与位置旋转交换。RoPE 的每个二维旋转对必须共享同一个 $$\lambda$$。

论文所用的 split-half 布局配对 $$i$$ 与 $$i+D/2$$，所以约束是 $$\lambda_i=\lambda_{i+D/2}$$；使用相邻通道配对的实现则应按它实际的配对关系处理，不能机械复制 split-half 下标。论文据配对通道的 Key 幅度共同选择尺度，常用迁移指数为 $$0.5$$。若统计全零，教学参考还需显式使用正下界，不能令 $$\lambda=0$$ 后再求逆。[QServe v3，SmoothAttention](https://arxiv.org/html/2405.04532v3#S4.SS2)

这一点与 [QuaRot/SpinQuant]({% post_url 2026-10-15-quarot-spinquant-rotation %}) 很适合对照：一般 Hadamard 不能穿过不同位置的 RoPE，所以在那里把 Q/K 配对旋转放到位置编码后；SmoothAttention 使用对角缩放，并约束每个二维对为同一标量，从而可以满足交换条件。这不是“任何变换都能融合”，而是为融合选择了特定结构。

仍取二维 Query $$[1,0]$$、Key $$[0,1]$$，两位置分别不旋转与旋转 90 度。原点积为 $$-1$$。在位置变换之后用 $$\operatorname{diag}(2,3)$$ 互逆缩放，点积不变；若把它提前，结果却是 $$-2/3$$。提前使用配对等值的 $$\operatorname{diag}(2,2)$$ 才仍为 $$-1$$。这个有限浮点小例拒绝非有限中间值、非零分量缩放成零等超出参考范围的情形，不是任意动态范围的 exporter。

<!-- example: w4a8-smoothattention-rope-pairing -->
```python
from math import isclose, isfinite, sqrt


def dot(a, b):
    if not a or len(a) != len(b) or not all(isfinite(v) for v in [*a, *b]):
        raise ValueError("finite vectors with equal nonzero dimensions required")
    result = sum(x * y for x, y in zip(a, b))
    if not isfinite(result):
        raise ValueError("dot product is outside this float reference's range")
    return result


def rope90(row):
    if len(row) != 2 or not all(isfinite(v) for v in row):
        raise ValueError("a finite two-element RoPE pair is required")
    return [-row[1], row[0]]


def scale_pair(query, key, scales):
    if not query or len(query) != len(key) or len(query) != len(scales):
        raise ValueError("head dimensions differ")
    if (not all(isfinite(v) for v in [*query, *key])
            or not all(isfinite(s) and s > 0 for s in scales)):
        raise ValueError("finite inputs and positive finite scales required")
    q_new = [q * s for q, s in zip(query, scales)]
    k_new = [k / s for k, s in zip(key, scales)]
    for original, transformed in [(query, q_new), (key, k_new)]:
        if any(not isfinite(v) or (x != 0 and v == 0)
               for x, v in zip(original, transformed)):
            raise ValueError("scaling is outside this float reference's range")
    return q_new, k_new


q, k = [1.0, 0.0], [0.0, 1.0]
original = dot(q, rope90(k))  # query's position transform is identity
post_q, post_k = scale_pair(q, rope90(k), [2.0, 3.0])
pre_q, pre_k = scale_pair(q, k, [2.0, 3.0])
paired_q, paired_k = scale_pair(q, k, [2.0, 2.0])
assert original == -1.0 and dot(post_q, post_k) == original
assert isclose(dot(pre_q, rope90(pre_k)), -2 / 3, abs_tol=1e-12)
assert dot(paired_q, rope90(paired_k)) == original

# A declared teaching floor for an all-zero Key pair, not an exporter promise.
key_pair_maxima, epsilon = [0.0, 0.0], 1e-6
common_scale = sqrt(max(epsilon, *key_pair_maxima))
assert common_scale > 0
_, _ = scale_pair(q, k, [common_scale, common_scale])
print("original / post / invalid pre / paired pre:", original,
      dot(post_q, post_k), dot(pre_q, rope90(pre_k)), dot(paired_q, rope90(paired_k)))
```

这些点积恒等式不等于 KV4 无损：Key 与 Value 之后的实际量化仍会引入误差。它们只说明，如果缩放位置和配对条件正确，就没有在量化之前额外改错 Attention 函数。

## 7. 真实转换器接收的不是任意原始 FP 权重

有了前面的数值参考，很容易误以为官方 `from_linear` 会接收任意 FP16/BF16 权重，自动搜索两级尺度并执行本文全部 RNE、clamp。这个理解不符合固定代码的输入流程。

该版本 `checkpoint_converter.py` 读取的是 LMQuant 已经导出的 fake-quant `model.pt` 与 `scale.pt`，从中获取两级尺度和零点，再传入 `W4A8OF16LinearDynamicInputScale.from_linear`。`from_linear` 的职责包括从已有浮点重建参数恢复 codes、检查范围、转换类型与重排打包；它的第二级恢复并没有再次完整执行本文教学编码器的 RNE+clamp 搜索。[固定版 checkpoint_converter.py](https://github.com/mit-han-lab/omniserve/blob/02b2925aa6fa3b92b06316a1524b7f38922cd9c8/scripts/ckpt_converter/checkpoint_converter.py)、[固定版 from_linear](https://github.com/mit-han-lab/omniserve/blob/02b2925aa6fa3b92b06316a1524b7f38922cd9c8/omniserve/modeling/layers/quantized_linear/w4a8_linear.py)

特别是第一步除回浮点尺度后，输入已经包含第二级量化误差，恢复的是最终 $$\widetilde q$$，不一定是原先被保护的 $$q^{(0)}$$。代码实际检查的是 $$[-128,127]$$，不是重新验证 $$q^{(0)}\in[-119,119]$$；最终值例如 121 仍可能合法。原始保护区间与 raw zero 的有效域属于上游参数合同，也不能由这一步检查或 tensor dtype 自动认证。

因此要区分“量化参数怎样搜索”和“按已知参数生成 kernel artifact”。即使旁边有一个 generic pseudo quantizer，也不能把它直接当成完整 QoQ 量化器。本文的整数证明、单 tile pack 与真实转换入口各自回答不同问题，不承诺彼此是任意输入下的逐位替代实现。

## 8. FP8 activation 的 W4A8 不是这一套整数表示

`W4A8` 只告诉位宽，没有告诉整数还是浮点。一个具体例子是 ModelOpt 0.35.0 固定配置 `W4A8_AWQ_BETA_CFG`：权重配置先做 INT4 block128，再接 FP8 量化，输入 activation 使用 FP8 `(4,3)`，即 E4M3 类表示。[ModelOpt 固定版 config.py](https://github.com/NVIDIA/Model-Optimizer/blob/c359cb774db4daab623cddde6c7e0399a49cf7d2/modelopt/torch/quantization/config.py)

它与本文的 UINT4→整数 group scale→signed INT8 operand，不是同一个数值流水线。TensorRT-LLM 1.1.0 文档也列出了 W4A8 AWQ 等 recipe，但模型与硬件支持要按该版本的表格阅读，不能把名字相同理解为 checkpoint、activation dtype 或 kernel 可以互换。这些是固定版本的例子，不是对当前推荐配置或所有后端支持的宣告。[TensorRT-LLM 1.1.0 Quantization](https://nvidia.github.io/TensorRT-LLM/1.1.0/features/quantization.html)

## 9. 把格式合同与执行收益分开核对

对前面的固定 per-group artifact，忽略 bias 与 padding，权重主体和 metadata 的平均成本是：

$$
\frac12+\frac2G+\frac2K\quad\text{bytes/weight}.
$$

其中每组有一 byte 整数尺度、一 byte scaled-zero，每个输出通道还有一份 FP16 的第一级尺度。取 $$G=128,K=4096$$，结果是 $$0.51611328125$$ bytes/weight，而不是只看权重 nibble 的 $$0.5$$。这笔预算也不包含 KV Cache、临时 workspace 和其他浮点模块。

数值与布局检查可以直接围绕本文链条展开：第一级目标与第二级重建是否符合声明规则，最终 signed byte 是否对应数学值，真实 tile 与 scale/zero 是否同步重排，实际 kernel 是否消费同一套元数据。真实 GEMM 还要使用它支持的 $$M/N/K$$ 和 group size，与对应的浮点重建参考比较；单 tile 往返并不能证明整个 kernel 正确。

性能则要再看目标 GPU、shape 与动态激活量化的代价。小 $$M$$ 节省权重流量的收益，可能被大 $$M$$ 下的转换吞吐、寄存器占用和调度改变；KV4 的容量变化也不能全部归到 Linear W4A8 GEMM。本文的标准库执行没有运行官方 converter、Torch、CUDA kernel 或模型推理，不将论文的任务质量与吞吐结果当作本机结果。

W4A8 的关键不只是把两个权重装进一个 byte，而是规定这个 byte 怎样变成合法、可高效供应的 8-bit 操作数。两级整数范围、逐 byte 算术、group 与 channel 修正位置、SmoothAttention 的融合条件，以及真实 converter 的输入假设，必须共同成立。只有把这些细节连起来，4-bit 存储与 8-bit 计算才是一条明确的执行路线，而不只是两个位宽标签。

## 参考资料

- [QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving，arXiv v3](https://arxiv.org/html/2405.04532v3)：本文方法与系统说明；该修订为 2025-05-01，研究首发于 2024 年。
- [固定版 w4a8_linear.py](https://github.com/mit-han-lab/omniserve/blob/02b2925aa6fa3b92b06316a1524b7f38922cd9c8/omniserve/modeling/layers/quantized_linear/w4a8_linear.py)、[checkpoint_converter.py](https://github.com/mit-han-lab/omniserve/blob/02b2925aa6fa3b92b06316a1524b7f38922cd9c8/scripts/ckpt_converter/checkpoint_converter.py)：字段表示、转换输入与 tile packing。原 QServe 官方仓库名称已重定向为 OmniServe，这里固定同一 commit 而非跟随默认分支。
- [固定版 per-group GEMM](https://github.com/mit-han-lab/omniserve/blob/02b2925aa6fa3b92b06316a1524b7f38922cd9c8/kernels/csrc/qgemm/w4a8_per_group/gemm_cuda.cu)、[per-channel GEMM](https://github.com/mit-han-lab/omniserve/blob/02b2925aa6fa3b92b06316a1524b7f38922cd9c8/kernels/csrc/qgemm/w4a8_per_chn/gemm_cuda.cu)、[行和与量化路径](https://github.com/mit-han-lab/omniserve/blob/02b2925aa6fa3b92b06316a1524b7f38922cd9c8/kernels/csrc/layernorm_kernels.cu)：指令、byte lane、epilogue 与 dispatch。
- [CUDA Math API 12.8 SIMD Intrinsics](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SIMD.html)：`__vadd4` 的逐 byte 加法语义。
- [ModelOpt 0.35.0 固定配置](https://github.com/NVIDIA/Model-Optimizer/blob/c359cb774db4daab623cddde6c7e0399a49cf7d2/modelopt/torch/quantization/config.py)、[TensorRT-LLM 1.1.0 Quantization](https://nvidia.github.io/TensorRT-LLM/1.1.0/features/quantization.html)：用于区分同名 FP8 W4A8 路线，不作为跨版本兼容性承诺。
