---
layout: post
title: "SmoothQuant：把 Activation Outlier 迁移到 Weight"
subtitle: "从等价通道缩放到 W8A8 INT8 GEMM、Q/DQ 融合与 Serving 收益"
date: 2026-08-02 09:00:00 +0800
last_modified_at: 2026-10-15
author: iStar
catalog: true
series: model-quantization
series_order: 50
technology_year: 2022
mathjax: true
tags: [模型量化, GPU优化, LLM推理]
---

INT8 矩阵乘已经很成熟，但让语言模型的一次线性层同时使用 INT8 权重和 INT8 activation，仍可能遇到明显的质量下降。困难之一是 activation 的少数通道幅度很大，而常用 kernel 又希望输入整行或整个 tensor 共享 scale。

Transformer 的 activation 里经常存在少数幅度很大的通道。若整个 tensor 共用一个 INT8 scale，这些 outliers 会把量化范围拉得很宽，大多数普通值只能挤在少量整数格点上；若缩小 scale 来照顾普通值，outliers 又会被截断。两种选择都会累积误差。

SmoothQuant 不删除 outlier，也不像 [LLM.int8()]({% post_url 2026-10-15-llm-int8-outlier-decomposition %}) 那样为异常维度保留运行时浮点支路。它在量化前通过数学等价的通道缩放，重塑 activation 与 weight 的数值范围。原始线性计算保持不变，之后两侧落到整数格点上的误差却可以重新分配。

完整路径可以概括为：

```text
calibration data
  → collect activation statistics
  → choose per-input-channel smoothing factors
  → fold factors into weights / preceding operator
  → quantize transformed weights to INT8 offline
  → quantize transformed activations at runtime
  → execute a real INT8 × INT8 GEMM
  → rescale, fuse epilogue, and produce the next tensor
```

只完成前四步会得到 INT8 weights，却不一定得到 W8A8 加速；只有运行时和 kernel 真正消费 INT8 activation，W8A8 的计算路径才闭合。

## INT8 量化究竟损失在哪里

先固定本文的对称均匀量化约定。设浮点 tensor 为 $$x$$，量化 scale 为 $$s_q>0$$，采用最近偶数舍入 RNE：

$$
q=\operatorname{clip}
\left(
\operatorname{RNE}\left(\frac{x}{s_q}\right),
-127,127
\right)
$$

反量化近似值为：

$$
\hat{x}=s_q q
$$

有的 runtime 会使用完整的 $$[-128,127]$$ 编码范围，或引入 zero point 做非对称量化；这里使用对称形式便于说明。全零 tensor 约定正 scale 1、全零 codes，不能由 absmax 算出 scale 0 再继续除法。编码、舍入和饱和的完整关系可对照 [量化数值基础]({% post_url 2026-10-15-quantization-numerical-foundations %})。误差主要来自两处：

- **Rounding error**：连续实数只能落到离散整数格点；
- **Clipping error**：超出表示范围的值被截到边界。

scale 大，覆盖范围宽，但相邻格点间距也大；scale 小，普通值更精细，但极值更容易饱和。对近似均匀、范围稳定的 weight，这个折中比较容易处理。对带有稳定通道 outlier 的 activation，一个 tensor 共用一个 scale 时就会很困难。

例如，一个 activation tensor 的绝大多数值落在 $$[-2,2]$$，少量 outlier 接近 40。若 scale 要覆盖 40，则一个整数步长约为：

$$
s_q\approx\frac{40}{127}\approx0.315
$$

大量小值会被粗粒度舍入。若按 2 来定 scale，分辨率提高约 20 倍，但 outlier 会严重饱和。SmoothQuant 要改变的正是这个分布，而不是 INT8 编码规则本身。

## 等价变换怎样迁移量化难度

考虑一个不写 bias 的线性层：

$$
Y=XW
$$

令 $$X\in\mathbb{R}^{M\times K}$$、$$W\in\mathbb{R}^{K\times N}$$。为每个 input channel 选择有限正数 $$s_j$$，组成 $$K\times K$$ 对角矩阵 $$S$$：

$$
Y=XW=(XS^{-1})(SW)
$$

定义：

$$
X'=XS^{-1},\qquad W'=SW
$$

则有：

$$
X'W'=XW
$$

对第 $$j$$ 个 input channel 来说：

$$
X'_j=\frac{X_j}{s_j},\qquad W'_j=s_jW_j
$$

若希望压小某个幅度异常大的 activation channel，可以为它选择大于 1 的 $$s_j$$，同时放大对应输入方向的权重。两者相乘时缩放抵消，所以实数代数下输出不变；实际浮点改写则应在数值容差内比较。

这条等式有三个重要边界：

1. $$S$$ 沿 **input-channel / GEMM K 维**定义，左右两边必须使用同一组因子；
2. 变换在量化前是数学等价的，量化后误差会重新分布，但不能保证消除；
3. smoothing factor 与后续 INT8 quantization scale 不是同一个概念。

第三点尤其容易混淆。$$s_j$$ 改造 tensor 的分布，通常离线确定；$$s_q$$ 把改造后的 tensor 映射到 INT8，可能静态保存，也可能在线计算。

具体看两个输入通道：$$X=[10,1]$$，$$W=[0.1,1]^\top$$，原始输出为 2。选 $$S=\operatorname{diag}(10,1)$$，得到 $$X'=[1,1]$$、$$W'=[1,1]^\top$$，乘积仍然是 2。

现在对两侧分别做 per-tensor 对称 INT8 量化。变换前输入 scale 为 $$10/127$$，code 为 $$[127,13]$$；权重 scale 为 $$1/127$$，code 为 $$[13,127]$$。重建后点积为：

$$
\hat{y}=\frac{10}{127^2}(127\times13+13\times127)
\approx2.047244
$$

变换后两侧都是 $$[1,1]$$，都可以被 code 127 与 scale $$1/127$$ 精确重建，输出恢复为 2。这个构造展示了等价变换如何改善某次量化，不保证任意张量都能通过一个 $$S$$ 消除全部误差。

## Alpha 在 Activation 与 Weight 之间做平衡

SmoothQuant 使用 calibration statistics 估计每个 activation input channel 的最大幅度，同时观察 weight 对应通道的幅度。一种常见表达是：

$$
s_j=
\frac
{\max(\lvert X_j\rvert)^\alpha}
{\max(\lvert W_j\rvert)^{1-\alpha}}
$$

其中 $$\alpha\in[0,1]$$ 控制 activation 与 weight 之间的量化难度分配。令 $$a_j=\max\lvert X_j\rvert>0$$、$$b_j=\max\lvert W_j\rvert>0$$，变换后的对应通道最大幅度为：

$$
s_j=\frac{(a_jb_j)^\alpha}{b_j},\qquad
a'_j=\frac{a_j}{s_j}=(a_jb_j)^{1-\alpha},\qquad
b'_j=b_js_j=(a_jb_j)^\alpha
$$

这给出三个直观的端点与中点：

- $$\alpha=0$$ 时，$$s_j=1/b_j$$，weight 对应输入通道的最大幅度归一到 $$1$$，activation 接下 $$a_jb_j$$ 的通道间差异；这通常不是保持原始 weight 不变。
- $$\alpha=1$$ 时，$$s_j=a_j$$，activation 通道的最大幅度归一到 $$1$$，weight 接下这些差异。
- $$\alpha=0.5$$ 时，两边的最大幅度均为 $$\sqrt{a_jb_j}$$。它平衡了这两个范围统计，但不保证真实量化误差或任务质量达到最优。

因此，[原论文第 4 节](https://arxiv.org/html/2211.10438v7#S4) 所说的迁移强度，应理解为跨通道量化难度的分配，不能解释为每个通道都随 $$\alpha$$ 增大而绝对放大。由 $$\partial s_j/\partial\alpha=s_j\ln(a_jb_j)$$ 可知，只有 $$a_jb_j>1$$ 时 $$s_j$$ 才随 $$\alpha$$ 增大；若 $$a_jb_j<1$$，方向相反。例如 $$a_j=64$$、$$b_j=0.01$$ 时，$$\alpha=0,0.5,1$$ 对应 $$s_j=100,80,64$$。合适的 $$\alpha$$ 仍取决于模型、模块、量化粒度和目标 kernel。

假设某通道的 activation 最大幅度为 64，weight 最大幅度为 1，且 $$\alpha=0.5$$：

$$
s_j=\frac{64^{0.5}}{1^{0.5}}=8
$$

在这组 $$a_jb_j>1$$ 的数值下，activation 最大幅度从 64 降到 8，weight 最大幅度从 1 增到 8。乘积没变，两侧的范围却更均衡。

前面的两个输入通道分别满足 $$a_jb_j=1$$，所以 $$\alpha=0.5$$ 恰好给出 $$s=[10,1]$$，并把两侧通道范围都变成 1。

上述纯公式要求统计值为正。固定官方实现会对 weight statistics 及最终 smoothing factor 使用最小值 clamp，以处理零通道或过小值；这类约定不能和正值公式混成“任意输入逐 bit 等价”。[固定版本 smooth.py](https://github.com/mit-han-lab/smoothquant/blob/c61476d728e42ae0d8a35e7e78494edcac3237b5/smoothquant/smooth.py)

下面用标准库复算两个通道与 alpha 端点。`smoothing_factor` 是采用上述 clamp 顺序的 Python 浮点参考；官方代码还会把统计值和因子转换到权重 dtype，因此这里不承诺与 FP16/BF16 运算逐 bit 相同。正且有限的因子也不能独自保证后续的除法、权重乘法或参数改写不溢出、不下溢，实际实现仍需检查所用 dtype 的范围。

代码模拟量化与反量化，输出仍为 Python 浮点数；全零量化向量使用本文约定的 scale 1，不复制 `fake_quant.py` 的 epsilon 处理。它不调用 INT8 GEMM，也不是覆盖任意动态范围的模型量化库。

<!-- example: smoothquant-two-channel-int8 -->
```python
from math import fsum, isfinite, isclose


def smoothing_factor(a, b, alpha):
    if not all(isfinite(v) for v in (a, b, alpha)):
        raise ValueError("finite statistics and alpha required")
    if a < 0 or b < 0 or not 0 <= alpha <= 1:
        raise ValueError("nonnegative ranges and alpha in [0, 1] required")
    factor = max(a ** alpha / max(b, 1e-5) ** (1 - alpha), 1e-5)
    if not isfinite(factor) or factor <= 0:
        raise ValueError("unrepresentable smoothing factor")
    return factor


def qdq_int8(values):
    values = list(values)
    if not values or any(not isfinite(v) for v in values):
        raise ValueError("nonempty finite vector required")
    amax = max(abs(v) for v in values)
    scale = amax / 127 if amax else 1.0
    if scale <= 0 or not isfinite(scale):
        raise ValueError("unrepresentable quantization scale")
    codes = [min(127, max(-127, round(v / scale))) for v in values]
    restored = [scale * code for code in codes]
    if any(not isfinite(v) for v in restored):
        raise ValueError("nonfinite reconstruction")
    return codes, restored, scale


x, w = [10.0, 1.0], [0.1, 1.0]
s = [smoothing_factor(abs(a), abs(b), 0.5) for a, b in zip(x, w)]
xs = [a / factor for a, factor in zip(x, s)]
ws = [b * factor for b, factor in zip(w, s)]
dot = lambda a, b: fsum(v * t for v, t in zip(a, b))
assert s == [10.0, 1.0]
assert isclose(dot(x, w), 2.0) and isclose(dot(xs, ws), 2.0)
qx, xhat, _ = qdq_int8(x)
qw, what, _ = qdq_int8(w)
before = dot(xhat, what)
after = dot(qdq_int8(xs)[1], qdq_int8(ws)[1])
assert qx == [127, 13] and qw == [13, 127]
assert isclose(before, 2.047244094488189)
assert isclose(after, 2.0)
assert qdq_int8([0.0, 0.0]) == ([0, 0], [0.0, 0.0], 1.0)
for alpha, expected in [(0.0, 100.0), (0.5, 80.0), (1.0, 64.0)]:
    assert isclose(smoothing_factor(64.0, 0.01, alpha), expected)
assert smoothing_factor(0.0, 0.0, 0.5) > 0
print({"factors": s, "original": dot(x, w),
       "INT8_before": before, "INT8_after": after})
```

## 为什么 Weight 更适合接住 Outlier

把量化难度从 activation 移到 weight，不意味着 weight 不会受伤，也不要求 weight 一定采用 per-channel 量化。前面的两通道构造中，输入和权重都是 per-tensor，仍然因为范围重新平衡而得到更小输出误差。

Activation 由请求在运行时产生。若每个 input channel 都单独动态统计和量化，会引入 reduction、scale 计算、内存读写与复杂 kernel 接口。很多执行路径因此只给 activation 使用 per-tensor 或 per-token scale。

Weight 是静态的，变换、范围选择和 packing 可以离线完成；如果 backend 支持，也能为不同 output channel 或 group 保存独立量化参数。Activation 则随请求动态产生，若为每个 input channel 在线统计和编码，尺度又沿 GEMM 的归约轴变化，就不再是一次整数点积后统一缩放的简单路径。

SmoothQuant 利用的是这种不对称性：用离线变换把通道间范围差异分摊到两侧，让支持高效 INT8 GEMM 的粒度也有机会保留精度。它不是以“所有 weight 天生容易量化”为前提，更不是靠给 weight 无限细粒度就能免费解决问题。

还要分清两条轴：SmoothQuant 的 $$s_j$$ 沿 input channel 重塑 $$X$$ 与 $$W$$；weight 的 INT8 scale 可能沿 output channel 或某种 group 计算。前者保证等价变换，后者决定整数编码精度，它们不能互换。

## Calibration 阶段真正收集什么

SmoothQuant 是 post-training quantization，不需要重新训练模型，但需要有代表性的 calibration data。对每个目标线性层，至少要得到 activation input channels 的范围统计，例如：

```text
# 统计流程示意，不是可执行框架程序
amax = zeros(hidden_size)

for batch in calibration_data:
    x = capture_linear_input(batch)
    amax = maximum(amax, abs(x).amax(over_all_dims_except_hidden))
```

只拿无代表性的随机 token 或极短文本做 calibration，统计出的范围未必覆盖生产分布。数据集至少要考虑：

- 真实输入语言和领域；
- 常见与长尾 prompt 长度；
- system prompt、tool schema、代码块等结构；
- 长上下文位置上的 activation 分布；
- MoE 模型中访问频率不同的 experts；
- 多模态模型的不同输入模态与 projector。

最大值对异常样本非常敏感，有些实现会比较 percentile、clipping 或其他稳健统计。但改变统计方式会改变论文 recipe，需要以质量实验验证，不能只因为 histogram 看起来更平滑就认为更好。

Calibration set 只用于选择变换和量化参数；最终质量必须在未参与 calibration 的 validation set 与任务集上测量，否则容易把参数调到样本本身。

## 离线 Smoothing 怎样消除额外算子

直接根据公式执行 $$X'=XS^{-1}$$，会在每个请求中增加一次逐元素乘法。它的 FLOPs 很少，却可能产生额外 kernel launch 和显存读写，抵消低精度收益。

更好的方式是尽可能把 $$S^{-1}$$ fold 进产生 $$X$$ 的前序参数，把 $$S$$ fold 进当前 linear weight。例如在可等价合并的 normalization / linear 边界上，离线改写相邻参数，使运行时自然产生 $$X'$$。当前权重则保存为已经平滑并量化后的 $$W'$$。

具体看 LayerNorm 的仿射部分。令均值、方差归一化后的向量为 $$z$$，原输出为 $$X=z\odot\gamma+\beta$$。若将参数改为：

$$
\gamma'=\gamma\oslash s,
\qquad
\beta'=\beta\oslash s
$$

则新输出自然变为 $$X'=X\oslash s$$。这里 $$\oslash$$ 表示逐元素相除。只除 gamma、忘记 beta，不能得到相同变换。当前 Linear 的 bias 则不需要随 $$S$$ 修改，因为 $$X'W'+b=XW+b$$；它与前序 LayerNorm 的 beta 不是同一个参数。[固定版本 folding 实现](https://github.com/mit-han-lab/smoothquant/blob/c61476d728e42ae0d8a35e7e78494edcac3237b5/smoothquant/smooth.py#L19)

若同一份归一化输出送给 Q、K、V 三个投影，必须使用同一组 $$s$$，同时变换三份权重。计算 weight 统计时也应考虑这组共享 consumers，而不是为了每个投影分别把同一个 norm 除三次。RMSNorm 没有 LayerNorm 这份 beta，其可折叠的仿射参数按对应结构处理；固定代码分别实现这两种情况。

理想 graph 变为：

```text
before:
previous op → X → multiply by S^-1 → quantize → INT8 GEMM with S·W

after folding:
modified previous op → X' → quantize → INT8 GEMM with transformed W'
```

folding 必须尊重模型拓扑。一个 activation 若分叉给多个 consumers，或同时经过 residual path，不能只修改一条支路而假设仍然等价。工程实现应沿 graph 确认所有 consumers，并逐层比较 smoothing 前后的高精度输出。

同样，不能任意把缩放穿过 GELU、SiLU 等非线性函数。对一个选定的 norm→linear 边界成立，不等于把整个 Transformer 任意相邻算子重写都成立。

验证离线变换最直接的方法是先不做 INT8：

$$
\operatorname{maxabs}(XW-X'W')<\epsilon
$$

若此时已经不一致，问题属于轴、转置、bias、folding 或 graph rewrite，而不是量化精度。

下面只取归一化之后的 $$z$$ 为已知值，专门核对仿射参数与三个共享投影，不涉及均值、方差的实现。正确结果保持三个输出不变；随后刻意漏掉 beta 或 K 分支的权重变换，检查输出确实改变。

<!-- example: smoothquant-shared-norm-folding -->
```python
from math import fsum, isclose

z = [1.0, -1.0]
gamma, beta = [2.0, 3.0], [0.5, -0.5]
s = [10.0, 1.0]
weights = {"q": [0.1, 1.0], "k": [0.2, -0.5], "v": [-1.0, 0.3]}
biases = {"q": 0.7, "k": -1.0, "v": 0.4}


def affine(g, b):
    return [value * gain + offset for value, gain, offset in zip(z, g, b)]


def linear(x, w, bias):
    return fsum(a * b for a, b in zip(x, w)) + bias


x = affine(gamma, beta)
new_gamma = [g / factor for g, factor in zip(gamma, s)]
new_beta = [b / factor for b, factor in zip(beta, s)]
xs = affine(new_gamma, new_beta)
new_weights = {name: [value * factor for value, factor in zip(w, s)]
               for name, w in weights.items()}
original, folded = {}, {}
for name in weights:
    original[name] = linear(x, weights[name], biases[name])
    folded[name] = linear(xs, new_weights[name], biases[name])
    assert isclose(original[name], folded[name], abs_tol=1e-12)
assert all(isclose(a / factor, b)
           for a, factor, b in zip(x, s, xs))

wrong_beta_x = affine(new_gamma, beta)
wrong_beta_q = linear(wrong_beta_x, new_weights["q"], biases["q"])
wrong_k = linear(xs, weights["k"], biases["k"])
assert not isclose(wrong_beta_q, original["q"])
assert not isclose(wrong_k, original["k"])
print({"original": original, "folded": folded,
       "forgot_norm_beta": wrong_beta_q, "forgot_k_weight": wrong_k})
```

## Smoothing 之后为何仍要在线量化 Activation

SmoothQuant 把 activation range 变得容易处理，但 GEMM 输入仍然是浮点 tensor，必须经过 Quantize 才成为 INT8。运行时大致有三种策略。

### Static Per-tensor Scale

Calibration 阶段为某个 activation tensor 固定一个 scale。线上无需再做 amax reduction，开销最低，但流量分布漂移时可能饱和，也可能因范围留得太宽而损失分辨率。

### Dynamic Per-tensor Scale

每次执行时根据当前 tensor 的 amax 计算 scale。它适应请求分布，但 reduction 和 quantization 是真实成本；若不能与上游算子融合，小 batch Decode 尤其容易被固定开销主导。

### Dynamic Per-token Scale

每个 token row 使用独立 scale，可减少不同 token 范围差异造成的误差。代价是更多 scale metadata、更复杂的 GEMM 接口和额外的统计工作。是否支持取决于硬件与 runtime，不是导出 checkpoint 时写一个配置就会自动生效。

SmoothQuant 解决 channel outlier，per-token scale 解决 token 之间的范围差异，两者处理的轴不同，可以组合，也有各自成本。

论文把三种 activation 选择组织为 O1、O2、O3。它们是不同量化配置，不是依次执行的三个阶段：

| 配置 | Weight 量化粒度 | Activation 量化粒度与时机 |
| --- | --- | --- |
| O1 | per-tensor | per-token dynamic |
| O2 | per-tensor | per-tensor dynamic |
| O3 | per-tensor | per-tensor static |

这三项的 weight 均为 per-tensor，不能把 O1 自动解释成 per-channel weight。SmoothQuant 的变换可以与其他粒度组合，后续模型实验和不同实现也可能使用别的配置，但不应改写原表的含义。O3 不再在线估计 activation scale，仍然必须把当前浮点 activation 编成整数；省掉统计不等于省掉 Quantize。[论文 v7 表 2](https://arxiv.org/html/2211.10438v7#S5.SS1)

## 从 Q/DQ Graph 到真实 INT8 GEMM

在显式量化 graph 中，Quantize/Dequantize 节点表达低精度边界：

```text
BF16 activation
  → Q: activation to INT8
  → INT8 GEMM with INT8 weight
  → scale using s_x and s_w
  → bias / residual / activation epilogue
  → BF16 output or requantized INT8 output
```

若 $$X\in\mathbb R^{M\times K}$$、$$W\in\mathbb R^{K\times N}$$，activation 按 token row 使用 scale $$s_{x,m}$$，weight 按 output channel 使用 $$s_{w,n}$$，对称整数点积可以写成：

$$
Y_{mn}\approx s_{x,m}s_{w,n}\sum_{k=1}^{K}(q_x)_{mk}(q_w)_{kn}
$$

两侧各用一个标量的 per-tensor scale 是它的特例。scale 必须在这次 $$K$$ 维求和内保持不变才能移到 epilogue；若按 input channel 或 K-group 变化，就要对局部部分和分别处理，不能直接套用上式。这里也未包含非对称 zero-point 修正。[SmoothQuant v7 §3、式 (2)](https://arxiv.org/html/2211.10438v7#S3)

整数乘加通常需要更宽的 accumulator，随后再应用 scale 并转换到输出 dtype；准确的 accumulator、rounding 和 saturation 规则由 kernel contract 决定。

Q/DQ 节点只是语义，不保证性能。优化器需要把它们与 GEMM、bias、activation 或 residual epilogue 融合。如果实际执行变成：

```text
INT8 weight → dequantize to BF16 → BF16 GEMM
```

那么只有权重存储可能变小，并没有使用 INT8 Tensor Core。若又单独 launch activation quantization 和 output dequantization，端到端速度甚至可能更差。

确认计算路径时，应查看 runtime engine、kernel 名称或 profiler trace，而不是只检查 checkpoint dtype。

一个具体区别来自官方仓库。固定 commit `c61476d728e42ae0d8a35e7e78494edcac3237b5` 的 `fake_quant.py` 在浮点 buffer 上执行 round 与重建，`W8A8Linear.forward` 再调用浮点 `F.linear`。它适合分析量化误差，不是已经实现 packed INT8 kernel 的性能程序。[固定版本 fake_quant.py](https://github.com/mit-han-lab/smoothquant/blob/c61476d728e42ae0d8a35e7e78494edcac3237b5/smoothquant/fake_quant.py)

但不能反过来概括为“论文的 PyTorch 实现只是 fake quant”。论文 §5.1 明确说明，PyTorch/Hugging Face 与 FasterTransformer 两个性能 backend 都用 CUTLASS INT8 GEMM 实现 Linear 和 BMM。误差模拟脚本与真实性能路径是不同产物，不能因为框架名字相同就混为一谈。[论文 v7 Implementation](https://arxiv.org/html/2211.10438v7#S5.SS1)

## W8A8 的内存收益应该怎样算

单看 dense weights：

```text
BF16: 2 bytes / parameter
INT8: 1 byte / parameter + scales and metadata
```

因此权重主体接近减半，但不能直接宣称整个服务显存减半。总显存还包括：

```text
model weights
+ KV Cache
+ runtime activations
+ CUDA Graph pools
+ temporary workspaces
+ allocator fragmentation
+ communication buffers
```

SmoothQuant 不会自动压缩 KV Cache。若服务主要受长上下文 KV 容量限制，W8A8 权重节省对最大并发的提升会小于权重占主导的短上下文场景。

Activation 是否常驻 INT8 也取决于 graph。若每层 GEMM 后马上转回 BF16，主要收益来自 GEMM 输入和 weight；若算子链能够保留低精度并融合，才可能进一步减少 activation 带宽。

## Prefill 与 Decode 的收益为什么不同

Prefill 通常有较大的 $$M$$ 维，GEMM 更接近 compute-bound。目标 GPU 的 INT8 Tensor Core 吞吐、shape、并行切分和融合都合适时，W8A8 更容易产生计算吞吐收益。

Decode 的 batch 较小时，矩阵乘经常更受 weight memory bandwidth 限制。INT8 权重比 BF16 小约一半，理论上有带宽优势；但在线 activation reduction、quantize、scale 读取、kernel launch 和调度延迟可能占据更大比例。

因此应分别报告：

- Prefill latency 与 TTFT；
- Decode step latency 与 TPOT；
- 不同 batch / sequence bucket 的 tokens/s；
- 在同一延迟 SLO 下的 goodput；
- quantize、GEMM、epilogue 各自的时间。

只报告一个大 batch 离线吞吐，会掩盖真实在线请求中的固定开销。

## Tensor Parallel 与分片之后还等价吗

量化 artifact 要与并行分片的轴一致。若 weight 先 SmoothQuant、再 INT8 quantize、再 tensor-parallel shard，部署端必须以同一顺序解释 smoothing factors、quantization scales 和 packed layout。

常见风险包括：

- exporter 与 runtime 对 weight 转置约定不同；
- input-channel smoothing factors 在 column/row parallel 中切错轴；
- per-channel weight scales 没随 shard 一起切分；
- padding 后的 hidden dimension 未同步处理 metadata；
- AllReduce/ReduceScatter 前后错误地重复 dequant 或 requant；
- 某些 rank 走 INT8，另一些因 shape 不支持而 fallback。

最低限度要在单卡 reference、每个 shard 的局部输出、collective 后的完整输出三个层面比较误差。

## SmoothQuant、FP8 与 AWQ 的边界

三者都在解决低精度推理，但执行契约不同：

| 方法 | 典型路径 | Activation | 核心问题 | 主要收益 |
| --- | --- | --- | --- | --- |
| SmoothQuant | W8A8 INT8 | INT8 | 把 activation outlier 迁移到 weight | 权重容量、带宽与原生 INT8 GEMM |
| FP8 | FP8 weight + FP8 activation | FP8 | 用浮点编码、scale recipe 覆盖动态范围 | FP8 Tensor Core 计算与较低带宽 |
| AWQ | W4A16 | BF16/FP16 | 用 activation 判断哪些 weights 更需保护 | 更强的权重压缩，常偏向 Decode 带宽 |

SmoothQuant 的 activation-aware 体现在重塑 activation/weight 分布，并在运行时量化 activation；AWQ 的 activation-aware 主要用于离线判断 weight importance，典型 W4A16 路径并不把 activation 量化到 INT4。

FP8 有指数，较容易容纳跨数量级数值；INT8 是等距格点，依赖 smoothing 与 scale granularity 缓解 outlier。FP8 是否优于 INT8 仍取决于硬件代际、kernel、模型质量和部署目标。

## 分开验证变换误差与量化误差

SmoothQuant 有一个特别有用的诊断顺序：先关掉量化，仅比较 folding 前后的高精度输出。若 gamma、beta、分叉 consumer 或 scale 轴有一处不匹配，此时就会暴露；不应通过修改 alpha 来掩盖 graph rewrite 错误。

只有变换对齐后，才用 Q/DQ 参考比较不同 alpha、范围统计与 activation recipe。逐层记录 rounding、saturation、zero rate 和输出误差，可以区分是迁移过多损伤了 weight，还是 activation 仍被少数通道主导。校准集外的长序列、领域输入和结构化 prompt 需要单独观察，不能只看样本内平均指标。

接着将相同参数交给真实 INT8 kernel，对齐 codes、scale 广播、累加、padding 和输出转换。这里的比较对象不是原始 BF16，而是选定数值合同下的 Q/DQ reference；否则很容易把本来就存在的量化误差和 kernel 实现错误混为一谈。

最后才把变化放回完整模型，观察 perplexity、代码、数学、多语言、长上下文与结构化输出等具体任务。某个局部误差下降不保证任务质量单调上升，某个平均分不变也不保证长尾请求没有退化。

## 保存变换状态而不只保存 INT8 权重

一份可复现的 SmoothQuant artifact 至少要能回答：基座和 calibration 来自哪个版本，采用什么 alpha、统计与零通道 clamp，哪些 norm/linear 参数已经完成 folding，以及 activation 选用 O1/O2/O3 还是其他 recipe。

“是否已经 fold”尤其重要。若权重与 norm 已被变换，加载时又根据同一组 smoothing factors 再变换一遍，就会重复改写。相反，若只发布平滑后的权重，却没有对应前序参数，也不能恢复原来的等价关系。

随后才是编码与执行合同：weight/activation 各自的粒度、scale 方向与精度、zero point、rounding、packing 版本，以及 TP 分片和目标 kernel。相同的 W8A8 名称不能代替这些信息。性能检查也应包含统计、量化与合并后的完整单层，再回到真实 Prefill/Decode 流量，而不是把单个整数 matmul 的时间当作服务加速。

## 小结

SmoothQuant 的核心不是“把异常值消掉”，而是利用线性层的通道缩放不变性，让 activation 和 weight 在有限整数格点上分摊误差。alpha 管理的是范围统计的平衡，并不直接指定某个可保证的模型精度。

数学等价性、参数 folding 和真实 W8A8 是三个连续但不同的环节。先把共享 consumer、gamma/beta 与输入轴改写正确，再选择编码和运行时粒度，最后让真实 kernel 消费 INT8 operands，才构成完整路径。

W8A8 把 weight 与 activation 都压到 8-bit，适合与原生 INT8 矩阵乘结合。[AWQ]({% post_url 2026-08-03-awq-activation-aware-weight-quantization %}) 用 activation 统计保护低比特权重，而 [W4A8]({% post_url 2026-08-04-w4a8-progressive-quantization-kernel %}) 进一步讨论 INT4 存储怎样供应 INT8 计算。阅读时应继续区分“权重占几个 bit”和“乘法使用什么精度”。

## 参考资料

- [SmoothQuant v7 正文](https://arxiv.org/html/2211.10438v7)，§3–4、表 2、§5.1 Implementation；[首次公开与版本历史](https://arxiv.org/abs/2211.10438)。
- [官方 smooth.py 固定版本](https://github.com/mit-han-lab/smoothquant/blob/c61476d728e42ae0d8a35e7e78494edcac3237b5/smoothquant/smooth.py)，LayerNorm/RMSNorm 与共享 consumers 的折叠。
- [同版本 fake_quant.py](https://github.com/mit-han-lab/smoothquant/blob/c61476d728e42ae0d8a35e7e78494edcac3237b5/smoothquant/fake_quant.py)，浮点 Q/DQ 误差模拟路径。
