---
layout: post
title: "QLoRA：冻结的 4-bit 权重怎样参与一次微调"
subtitle: "从 NF4 编码和双重量化的存储账单，推到 LoRA 梯度、显存峰值与合并后的模型"
date: 2026-10-15
last_modified_at: 2026-10-15
author: iStar
catalog: true
series: model-quantization
series_order: 60
technology_year: 2023
mathjax: true
tags: [模型量化, 机器学习, 工程实践]
---

LoRA 只训练少量附加参数，已经省去了基座权重的梯度与优化器状态，为什么微调大模型仍然需要很多显存？因为基座虽然不更新，每次前向和反向计算依然要用到它。若基座仍以 BF16 保存，这部分权重存储并不会随着可训练参数变少而消失。

QLoRA 进一步压缩这份冻结基座：用低位编码保存权重，在计算时恢复到较高精度，让梯度穿过固定的线性运算，最终只更新 LoRA 参数。因此，它不是“用梯度下降训练一堆 4-bit 整数”，也不是把整个微调过程的张量都改成 4-bit。

这条路径把几个不同问题连在一起：NF4 决定权重如何编码，double quantization 压缩编码所需的 scale，LoRA 决定训练哪些参数，paged optimizer 则管理部分内存峰值。先看清一层计算，才能理解每种技术究竟省了哪笔账。

## 量化基座与可训练支路怎样相加

令输入 $$X\in\mathbb{R}^{M\times K}$$，基座权重 $$W\in\mathbb{R}^{K\times N}$$。量化后解码得到固定矩阵 $$\hat{W}$$，添加低秩参数 $$A\in\mathbb{R}^{K\times r}$$、$$B\in\mathbb{R}^{r\times N}$$，则一层输出为：

$$
Y=X\hat{W}+cXAB
$$

$$c$$ 是固定缩放常数，例如常见 LoRA 定义里的 $$\alpha/r$$。第一项来自冻结的量化基座，第二项来自可训练 adapter。这里的“低秩”表示更新矩阵写成两个较小矩阵的乘积，不表示输入或输出也被压到 4-bit。

存储与计算使用不同精度。NF4 的 code 每个只有 4 bit，但矩阵乘可以消费解码后的 BF16 等数值；adapter、activation、梯度与优化器状态也有各自的 dtype。QLoRA 原论文采用低位存储与通常为 BF16 的计算类型，应与“原生 4-bit 乘法完成全部训练”区分。[QLoRA §2–3](https://arxiv.org/html/2305.14314v1#S3)

$$\hat{W}$$ 已经包含基座量化误差，训练看到的是这个固定算子，而不是原始未量化 $$W$$。这个差别在训练后的 merge 阶段仍然重要。

## NF4 保存的是索引而不是均匀整数值

NF4 使用一张包含 16 个非均匀重建值的 codebook，4-bit code 表示查哪一项。它不是均匀 signed INT4，也不是把 4 bit 拆成 sign、exponent、fraction 的 E2M1 浮点编码。

它的构造受零中心正态分布的分位数启发，把较多格点安排在零附近，并保留精确零。实际码表有 7 个负值、一个零、8 个正值，并非逐项正负对称。例如：

| Code 索引 | 归一化重建值，约 |
| ---: | ---: |
| 0 | -1 |
| 3 | -0.39491749 |
| 7 | 0 |
| 8 | 0.07958030 |
| 9 | 0.16093020 |
| 15 | 1 |

零附近的间隔与端点附近不同。正态假设解释的是这类码表的设计动机，不是“真实每个 block 都正态”的保证，也不是对任意输入、误差指标都达到全局最小 MSE 的定理。[原论文附录 E](https://arxiv.org/html/2305.14314v1#A5)、[bitsandbytes 0.47.0 NF4 表](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.47.0/bitsandbytes/backends/utils.py)

对一个有限权重 block，取绝对最大值 $$a=\max_i\lvert w_i\rvert>0$$，先归一化，再选择最近 code：

$$
u_i=w_i/a,
\qquad
j_i=\mathop{\arg\min}_{j\in\{0,\ldots,15\}}
\lvert u_i-C_j\rvert,
\qquad
\hat{w}_i=aC_{j_i}
$$

$$C$$ 是码表。这里 $$a$$ 是反量化乘子，与 [整数仿射量化]({% post_url 2026-10-15-quantization-numerical-foundations %}) 中的整数 zero point 不是同一种参数；NF4 的零已经由 code 7 对应的表项给出。

例如 block $$[0,0.2,-0.4,1]$$ 的 $$a=1$$，得到索引 $$[7,9,3,15]$$，重建约为 $$[0,0.16093020,-0.39491749,1]$$。保存四个浮点重建值并不压缩，真正的 payload 是四个 4-bit 索引，加上 scale 与必要 metadata。

## 从 NF4 索引打包成字节

下面的标准库参考处理一整个教学 block，使用完整码表、最近值选择，并约定距离相同时取较小索引。为全零 block 选 scale 1、全零值索引 7；这是显式的安全约定，不把所有 kernel 的零通道处理假定成相同实现。

打包格式也明确规定：前一个 code 放高 nibble，后一个放低 nibble；奇数尾部补 code 7，并保存原逻辑长度。实际 backend 的布局必须按其协议解释，不能仅凭都是 NF4 就交换字节文件。

<!-- example: qlora-nf4-block-and-packing -->
```python
from math import isfinite, isclose

NF4 = (
    -1.0, -0.6961928009986877, -0.5250730514526367,
    -0.39491748809814453, -0.28444138169288635, -0.18477343022823334,
    -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725,
    0.24611230194568634, 0.33791524171829224, 0.44070982933044434,
    0.5626170039176941, 0.7229568362236023, 1.0,
)


def encode_nf4_block(values):
    values = list(values)
    if not values or any(not isfinite(v) for v in values):
        raise ValueError("nonempty finite block required")
    amax = max(abs(v) for v in values)
    scale = amax if amax else 1.0
    codes = [min(range(16), key=lambda j: (abs(v / scale - NF4[j]), j))
             for v in values]
    restored = [scale * NF4[j] for j in codes]
    if any(not isfinite(v) for v in restored):
        raise ValueError("nonfinite reconstruction")
    return codes, restored, scale


def pack_codes(codes):
    codes = list(codes)
    if any(type(j) is not int or not 0 <= j < 16 for j in codes):
        raise ValueError("4-bit integer codes required")
    length = len(codes)
    if length % 2:
        codes.append(7)
    return bytes((codes[i] << 4) | codes[i + 1]
                 for i in range(0, len(codes), 2)), length


def unpack_codes(payload, length):
    if not isinstance(payload, (bytes, bytearray)):
        raise ValueError("byte payload required")
    if type(length) is not int or length < 0 or len(payload) != (length + 1) // 2:
        raise ValueError("payload and logical length disagree")
    if length % 2 and payload[-1] & 15 != 7:
        raise ValueError("unexpected padding code")
    codes = [j for byte in payload for j in (byte >> 4, byte & 15)]
    return codes[:length]


codes, restored, scale = encode_nf4_block([0.0, 0.2, -0.4, 1.0])
payload, length = pack_codes(codes)
assert codes == [7, 9, 3, 15]
assert payload == bytes([0x79, 0x3F])
assert unpack_codes(payload, length) == codes
assert isclose(restored[1], 0.16093020141124725)
odd = [7, 15, 0]
assert unpack_codes(*pack_codes(odd)) == odd
assert encode_nf4_block([0.0, 0.0])[0] == [7, 7]
assert encode_nf4_block([0.0, 0.0])[2] == 1.0

# The same quantizer need not preserve newly merged floating weights.
merge_codes, merge_restored, _ = encode_nf4_block([0.53, -0.44])
assert merge_codes == [15, 1]
assert not isclose(merge_restored[1], -0.44)
print({"codes": codes, "bytes": payload.hex(), "restored": restored,
       "requantized_merge": merge_restored})
```

参考使用 Python 双精度数查这张表。bitsandbytes 0.47.0 的 default 路径会将 codebook 转成相应输入 dtype 后比较，设备 kernel 也可能具有不同边界细节。因此它验证码表与打包逻辑，不承诺与每个 backend 在所有临界输入上逐 bit 相同。[固定版本 default 算子](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.47.0/bitsandbytes/backends/default/ops.py)

## Double Quantization 压缩的是 Scale

每 64 个 4-bit 权重保存一个 FP32 absmax，单看 scale 就增加：

$$
\frac{32}{64}=0.5\ \text{bit/weight}
$$

权重主体加 metadata 是 4.5 bit/weight，而不是正好 4。Double quantization 再量化这些 scale：每个一级 scale 用 8 bit 表示，每 256 个一级 scale 额外保存一个 FP32 二级 scale。忽略全局 offset、尾部与 headers，平均为：

$$
4+\frac8{64}+\frac{32}{64\times256}
=4.126953125\ \text{bit/weight}
$$

相比 4.5，每个权重约节省 0.373046875 bit。它不是先把 weight 压成 4-bit、再把 weight 压成 2-bit，而是压缩“如何重建这些 weight”的参数。

固定 bitsandbytes 0.47.0 的 `compress_statistics` 路径先计算一级 absmax 的均值 offset，减去 offset，再调用 blocksize 256 的 blockwise 量化；恢复时解码二级表示并加回 offset。这里使用默认 signed 8-bit dynamic codebook，不应误写为“把所有 scale 强制 cast 成 FP8 E4M3”。论文的 8-bit 浮点表述与具体软件的码表、容器和转换规则需要分开。[quantize_4bit 与 quantize_blockwise](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.47.0/bitsandbytes/functional.py)

对含 $$P>0$$ 个权重的张量，令一级 block 数 $$n_1=\lceil P/64\rceil$$、二级 block 数 $$n_2=\lceil n_1/256\rceil$$。若按前述 nibble 打包，并给全局 offset 计 32 bit，一份简化账单是：

$$
8\left\lceil\frac P2\right\rceil+8n_1+32n_2+32
$$

它尚未计格式头、码表存储、布局对齐和未量化模块，因此渐近的 4.126953125 不能当作任何小张量的精确文件大小。

Scale 再量化也有误差。如果一级索引保持不变，scale 偏差为 $$\delta a$$，则该 block 每个权重额外变化 $$C_{j_i}\delta a$$。同组误差会相关地传播，metadata 压缩并不是数值上完全免费。[QLoRA §3 Double Quantization](https://arxiv.org/html/2305.14314v1#S3)

## 冻结权重为什么仍然需要输入梯度

把前向写成 $$Z=XA$$、$$Y=X\hat{W}+cZB$$，给定输出梯度 $$G=\partial\mathcal{L}/\partial Y$$。Adapter 支路先得到 $$dZ=cGB^\top$$，于是：

$$
\frac{\partial\mathcal{L}}{\partial A}=cX^\top GB^\top,
\qquad
\frac{\partial\mathcal{L}}{\partial B}=cA^\top X^\top G
$$

输入同时经过两条支路，因此：

$$
\frac{\partial\mathcal{L}}{\partial X}
=G\hat{W}^\top+cGB^\top A^\top
$$

冻结 $$\hat{W}$$ 意味着不为基座计算参数更新、保存对应优化器状态，不意味着丢掉第一项输入梯度。若上游还有可训练 adapter，梯度仍需穿过这个固定线性算子返回上游。

因此，不能把“冻结基座参数”简单替换成“整个基座前向都套 `no_grad`”。前者允许对需要梯度的输入求导，后者可能切断上游学习路径。同样，这里不需要对被冻结的 NF4 索引或离散选择求 STE；它与更新 master weights、以 fake quant 模拟误差的 QAT 是不同问题。

bitsandbytes 0.47.0 的 `MatMul4Bit.backward` 会在需要时解码权重来计算输入梯度，而打包权重的梯度为 `None`。注意源码函数参数也叫 A、B，但那里 A 是输入、B 是打包权重，不是本文 LoRA 的两个矩阵。[固定版本 MatMul4Bit](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.47.0/bitsandbytes/autograd/_functions.py)

## 一次完整反向与有限差分

取一个输入 token $$x=[1,2]$$，冻结 NF4 codes $$[15,0]$$、scale 0.5，解码得到 $$\hat{w}=[0.5,-0.5]^\top$$。令 rank 为 1，$$A=[0.1,0.2]^\top$$、$$B=[0.3]$$、$$c=1$$，目标值为 1。

基座输出为 -0.5，adapter 输出为 0.15，总输出 -0.35。使用 $$\mathcal{L}=\tfrac12(y-1)^2$$，输出梯度为 -1.35，可以算得：

$$
dA=[-0.405,-0.81]^\top,
\qquad dB=-0.675,
\qquad dx=[-0.7155,0.594]
$$

下面对输入、A、B 的全部五个标量分别做中央有限差分，再只更新 adapter，并核对基座 packet 与 scale 没有改变。

<!-- example: qlora-frozen-base-gradients -->
```python
from math import fsum, isclose
from hashlib import sha256
from struct import pack

base_packet, base_scale = bytes([0xF0]), 0.5
endpoint_table = {0: -1.0, 15: 1.0}
indices = [base_packet[0] >> 4, base_packet[0] & 15]
base = tuple(base_scale * endpoint_table[j] for j in indices)
before = sha256(base_packet + pack("<d", base_scale)).digest()
c, target = 1.0, 1.0
x, a, b = [1.0, 2.0], [0.1, 0.2], 0.3


def prediction(x, a, b):
    return fsum(v * w for v, w in zip(x, base)) + c * b * fsum(
        v * t for v, t in zip(x, a))


def loss_flat(parameters):
    output = prediction(parameters[:2], parameters[2:4], parameters[4])
    return 0.5 * (output - target) ** 2


def gradients(x, a, b):
    g = prediction(x, a, b) - target
    dx = [g * (w + c * t * b) for w, t in zip(base, a)]
    da = [c * v * g * b for v in x]
    db = c * fsum(v * t for v, t in zip(x, a)) * g
    return dx, da, db


parameters = x + a + [b]
dx, da, db = gradients(x, a, b)
analytic = dx + da + [db]
eps = 1e-6
for i, expected in enumerate(analytic):
    plus, minus = parameters.copy(), parameters.copy()
    plus[i] += eps
    minus[i] -= eps
    numeric = (loss_flat(plus) - loss_flat(minus)) / (2 * eps)
    assert isclose(numeric, expected, rel_tol=1e-7, abs_tol=1e-9)
assert isclose(prediction(x, a, b), -0.35)
assert all(isclose(v, t) for v, t in zip(da, [-0.405, -0.81]))
assert isclose(db, -0.675)
assert all(isclose(v, t) for v, t in zip(dx, [-0.7155, 0.594]))

_, da_zero, db_zero = gradients(x, a, 0.0)
assert da_zero == [0.0, 0.0] and not isclose(db_zero, 0.0)
learning_rate = 0.05
updated_a = [v - learning_rate * g for v, g in zip(a, da)]
updated_b = b - learning_rate * db
assert loss_flat(x + updated_a + [updated_b]) < loss_flat(parameters)
assert sha256(base_packet + pack("<d", base_scale)).digest() == before
print({"output": prediction(x, a, b), "dX": dx, "dA": da, "dB": db,
       "B_zero_dA": da_zero, "B_zero_dB": db_zero})
```

有限差分验证的是这个固定小算子的导数，不是 BF16 kernel 或模型微调效果。一个常见初始化也值得注意：若 B 初始化为零、A 非零，第一步 $$dA=0$$，但 $$dB$$ 可以非零；B 更新后，A 才可能收到非零梯度。合法的零初始化效应不能直接诊断为“梯度断了”。

实际代码还需要确认冻结范围。`Linear4bit` 在该版本显式冻结的是量化 weight，不能据此认定 bias、normalization 或其他基座参数也已全部冻结；训练准备过程应按目标参数集合完成设置。它的默认 `quant_type` 还是 `fp4`，选择 NF4 必须明确指定，不能从类名推断。[bitsandbytes 0.47.0 Linear4bit](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.47.0/bitsandbytes/nn/modules.py)

## Paged Optimizer 管理哪一部分内存

冻结基座省掉了基座的梯度与优化器状态，但训练峰值还包括 adapter 参数、其梯度与优化器状态、保存的 activation、反向中间值、解码 workspace 和分配器开销。梯度 checkpointing 能用重新计算换取部分 activation 存储，却不会让这些项目都消失。

Paged optimizer 使用统一内存机制，使相应优化器状态可以在 CPU 与 GPU 之间按页迁移，以缓解某些峰值。它不是把所有 activation 自动放到无限大内存里，也不保证迁移没有延迟；host RAM、互连、page fault 和访问时序都会影响代价。[QLoRA §3 Paged Optimizers](https://arxiv.org/html/2305.14314v1#S3)

因此，存储预算应先分开 frozen base 的 payload 与 scales、adapter 的训练状态，以及随 batch、序列长度和目标模块变化的 activation，再测峰值。把 4.126953125 乘参数量，只近似计算了低位权重这一项，不能据此宣布某块 GPU 一定够用。

论文给出过单张 48GB GPU 微调 65B 模型的设置，但不是任意模型、长度与 batch 的通用容量保证。其附录 G 的内存图还明确限定 batch 1、序列长 512、使用 checkpointing；图中的输入梯度估算不包含 attention，不能原样当成完整训练峰值。[原论文附录 G](https://arxiv.org/html/2305.14314v1#A7)

## Merge 后还是不是训练时的函数

训练完成时，模型函数由 $$\hat{W}$$ 和 adapter 一起定义。至少有三种不同的交付方式：

$$
\begin{aligned}
\text{保留两支路：}&\quad X\hat{W}+cXAB\\
\text{高精度合并：}&\quad X(\hat{W}+cAB)\\
\text{合并后再量化：}&\quad X\,\operatorname{DQ}(Q(\hat{W}+cAB))
\end{aligned}
$$

前两者在实数代数下等价，实际还受计算 dtype、舍入和 kernel 顺序影响。第三种新增了一次量化，不应默认与训练函数相同。

取反向例子中梯度更新前的 $$A$$、$$B$$，LoRA 增量 $$cAB=[0.03,0.06]^\top$$。高精度 merge 得到 $$[0.53,-0.44]^\top$$，对输入 $$[1,2]$$ 的输出仍为 -0.35。用前面的 NF4 量化器重新计算 absmax，会得到 scale 0.53、codes $$[15,1]$$，第二个权重约变成 -0.368982，输出约 -0.207964。这个量化例已经包含在第一段代码中，说明重新导出需要重新评测。

另外，将 adapter 直接加到原始未量化 $$W$$ 上，得到 $$W+cAB$$，也不是训练时的 $$\hat{W}+cAB$$。只有在 $$W=\hat{W}$$ 等相应条件下才相同。基座 revision、量化参数、tokenizer 与模板必须一起匹配，不能把 adapter 当作对任意同名基座都等价的补丁。

## 省下显存不等于证明任务质量

NF4 的设计、double quant 的 metadata 预算与梯度恒等式，解释的是方法为什么可行，不保证任意数据、rank 和 target modules 下都达到同样结果。冻结的量化误差、adapter 表达能力与训练分布共同影响最终函数。

原论文 §8 还明确指出，并未建立 33B、65B 尺度上 QLoRA 与 16-bit full finetuning 等效的直接结论。能够在有限显存中完成大模型微调，与在所有规模上已经证明和全参微调质量一致，是两种主张。[Limitations and Discussion](https://arxiv.org/html/2305.14314v1#S8)

QLoRA 最重要的结构可以沿一层计算记住：低位 code 与 scale 描述一个固定基座；解码后的权重参与前向，也参与必要的输入反向；只有 adapter 参数获得更新。将存储、求导与交付方式分别说清楚，才能既理解节省来自哪里，也知道哪些实验仍需在真实任务上完成。

## 参考资料

- [QLoRA: Efficient Finetuning of Quantized LLMs v1](https://arxiv.org/html/2305.14314v1)，§2–3、§8、附录 E/G；[2023 年首次公开与版本历史](https://arxiv.org/abs/2305.14314)。
- [bitsandbytes 0.47.0 NF4/FP4 码表](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.47.0/bitsandbytes/backends/utils.py) 与 [default 量化路径](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.47.0/bitsandbytes/backends/default/ops.py)。
- [同版本 functional.py](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.47.0/bitsandbytes/functional.py)，double quantization、offset 与二级码表。
- [Linear4bit 模块](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.47.0/bitsandbytes/nn/modules.py) 与 [MatMul4Bit 反向](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.47.0/bitsandbytes/autograd/_functions.py)。
