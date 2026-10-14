---
layout: post
title: "LLM.int8()：异常通道为什么需要单独计算"
subtitle: "从逐向量量化到混合精度分解，理解少数大幅度特征怎样影响整行 INT8 表示"
date: 2026-10-15
last_modified_at: 2026-10-15
author: iStar
catalog: true
series: model-quantization
series_order: 20
technology_year: 2022
mathjax: true
tags: [模型量化, GPU优化, LLM推理]
---

假设一次线性层的输入只有三个特征：$$[0.11,0.2,100]$$，三个权重都为 1。原始点积等于 100.31。若把输入用一个对称 INT8 scale 编码，100 能够被准确表示，前两个小数却都会变成零，最后输出只剩下 100。

这说明 outlier 的影响不只是“那个大数比较难表示”。它还会拉大同一组数共享的步长，让其他特征失去分辨率。大数自己保留了下来，小数对输出的贡献反而丢失了。

LLM.int8() 在 2022 年提出了一个有针对性的处理方式：大部分维度继续使用逐向量 INT8 量化，少数异常特征维度拆出去，以浮点计算其贡献，再将两部分相加。要理解它，首先需要分清逐向量量化能解决什么，以及为什么它仍然无法隔离一行里的异常通道。

## 每个点积分别拥有自己的尺度

沿用 [量化数值基础]({% post_url 2026-10-15-quantization-numerical-foundations %}) 的记号：输入 $$X\in\mathbb{R}^{M\times K}$$，权重 $$W\in\mathbb{R}^{K\times N}$$，输出 $$Y=XW$$。输入的一行对应一个 token 的特征向量；一个 batch 中的多个序列可以在这里展平成多行。权重的一列对应一个输出通道。

如果整个输入矩阵只用一个 scale，那么某个 token 出现的大值会影响其他 token。逐向量量化将这件事拆开：为每一行输入、每一列权重分别选择对称 absmax scale。

$$
s_{x,m}=\frac{\max_k\lvert X_{mk}\rvert}{127},
\qquad
s_{w,n}=\frac{\max_k\lvert W_{kn}\rvert}{127}
$$

全零向量单独约定 scale 为 1、code 全为零。本文的 scale 是反量化时的乘子；论文中的 normalization constant 可以使用其倒数，两个方向不能直接混用。

量化后的每个元素为：

$$
(Q_X)_{mk}=\operatorname{RNE}(X_{mk}/s_{x,m}),
\qquad
(Q_W)_{kn}=\operatorname{RNE}(W_{kn}/s_{w,n})
$$

这里使用 $$[-127,127]$$ 的对称 code 和最近偶数舍入。点积完成后恢复尺度：

$$
\hat{Y}_{mn}
=s_{x,m}s_{w,n}\sum_{k=1}^{K}(Q_X)_{mk}(Q_W)_{kn}
$$

输入行 scale 和权重列 scale 对每个输出元素形成一个乘积。若把它们分别视为长度为 $$M$$、$$N$$ 的列向量，也可以写成：

$$
\hat{Y}=(s_xs_w^\top)\odot(Q_XQ_W)
$$

$$\odot$$ 表示逐元素相乘。$$s_xs_w^\top$$ 是外积形成的 $$M\times N$$ 缩放矩阵，不是把两个尺度数组随意做一次普通矩阵乘法。这个组合就是 LLM.int8() 使用的 vector-wise quantization 思路。[论文 §3.1](https://arxiv.org/html/2208.07339v2#S3.SS1)

它的价值是避免不同 token、不同输出通道的极值互相牵制。代价也相对清楚：每个点积的整个 $$K$$ 维仍使用同一对 scale，可以在整数归约完成后恢复尺度，而不必为每个乘法项单独加载一份缩放参数。

## 一行里的大数仍然会挤掉普通特征

回到 $$x=[0.11,0.2,100]$$。即使已经按行量化，这三个值仍共享：

$$
s_x=\frac{100}{127}\approx0.787402
$$

除以这个步长后，三个数约为 $$[0.1397,0.254,127]$$，舍入得到 $$[0,0,127]$$。权重 $$[1,1,1]^\top$$ 可以被对应的权重 scale 精确重建，但无法补回已经消失的输入贡献。

下面可以独立复算这一步：

<!-- example: llm-int8-single-row -->
```python
from math import fsum, isclose

x = [0.11, 0.2, 100.0]
w = [1.0, 1.0, 1.0]
sx = max(abs(v) for v in x) / 127
sw = max(abs(v) for v in w) / 127
qx = [round(v / sx) for v in x]
qw = [round(v / sw) for v in w]
original = fsum(a * b for a, b in zip(x, w))
quantized = sx * sw * sum(a * b for a, b in zip(qx, qw))
assert qx == [0, 0, 127]
assert isclose(original, 100.31)
assert isclose(quantized, 100.0)
print({"codes": qx, "original": original, "int8": quantized})
```

一种想法是给 activation 的每个输入通道单独缩放。但这些 scale 沿矩阵乘的归约轴 $$K$$ 变化，不能像前面的行列 scale 一样，全部移到点积之外。它当然可以实现，却需要另一种 kernel 数据流。

LLM.int8() 选择的是另一条路：如果真正制造困难的维度很少，就让它们离开 INT8 主路。论文对所测 Transformer 的分析发现，大幅度特征具有明显的通道集中性；这是经验观察，不是说任意模型的异常维都必然一样少，也不是给所有后来模型划出一个固定参数规模的分界线。[论文 §4](https://arxiv.org/html/2208.07339v2#S4)

## 先按列分解再决定计算精度

给定正阈值 $$\alpha$$，本文参考实现选取：

$$
O=\{k:\exists m,\ \lvert X_{mk}\rvert\ge\alpha\},
\qquad
R=\{0,\ldots,K-1\}\setminus O
$$

$$O$$ 是当前输入矩阵中异常特征列的并集，$$R$$ 是其余列。论文 §3.2 用“幅度大于阈值”描述选取，后面的参考代码明确采用 $$\ge$$，与 bitsandbytes 0.47.0 的列检测边界一致。为避免歧义，代码还约定 `threshold=0` 禁用拆分，而不是把所有零以上的列选中。

在引入量化之前，矩阵乘本身就有严格分解：

$$
XW=X_{:,R}W_{R,:}+X_{:,O}W_{O,:}
$$

这是沿归约维分成两组求和。没有删除任何特征，也不需要训练模型来补偿被删掉的值。

接着只对第一项的两个子矩阵进行 INT8 逐向量量化；第二项用浮点路径计算。这便得到概念上的混合精度输出：

$$
\hat{Y}=\hat{Y}_R+Y_O,
\qquad
Y_O=X_{:,O}W_{O,:}
$$

这里的 $$Y_O$$ 用实数代数说明分解；实际 FP16 计算也存在乘法、累加和相加舍入，不能把“高精度支路”说成实数意义上的无误差支路。[论文 §3.2、图 2](https://arxiv.org/html/2208.07339v2#S3.SS2)

整列拆分与只挑超阈值元素不同。假设输入为：

$$
X=\begin{bmatrix}
0.11&0.2&100\\
0.2&-0.3&0.5
\end{bmatrix}
$$

阈值设为 6，第三列被第一行的 100 选中。第二行同一列的 0.5 虽然没有超阈值，也会跟随这条列索引进入浮点分解项。若 INT8 主路仍计算这部分，再把完整异常列的浮点贡献加回来，就会重复计算；若只删掉大元素、补回时却遗漏小元素，也不再对应上述矩阵恒等式。

## 同一个点积怎样从 100 恢复到接近 100.31

在三维例子中，选出 $$O=\{2\}$$ 后，常规部分只剩 $$[0.11,0.2]$$。为这部分重新计算 activation scale：

$$
s_{x,R}=\frac{0.2}{127}
$$

两个值量化成 $$[70,127]$$。对应权重仍为 $$[1,1]^\top$$，因此常规部分的重建输出为：

$$
\hat{y}_R=\frac{0.2}{127}(70+127)
\approx0.3102362205
$$

异常部分计算 $$100\times1=100$$，最后得到：

$$
\hat{y}\approx100.3102362205
$$

| 计算方式 | 输出 | 相对原始点积的误差 |
| --- | ---: | ---: |
| 原始点积 | 100.31 | 0 |
| 不拆分的逐向量 INT8 | 100 | -0.31 |
| 只对常规维量化再相加 | 100.3102362205 | 约 0.0002362205 |

改善不是因为多保存了输入的大值——它在第一种 INT8 路径里已经被准确保留——而是因为普通特征终于可以使用较小步长。残余误差来自常规部分的量化，因此这个例子也没有制造“混合之后完全无损”的假象。

这也不是“拆分后所有矩阵的输出误差都单调下降”的定理。重新选择范围会改变舍入落点，原来不同乘积之间的误差抵消也可能随之消失；实际任务质量仍需在目标模型上检验。

## 一份能看见两条支路的矩阵参考

下面实现上述教学分解：常规维单独重算输入行 scale 和权重列 scale，异常维读取原始浮点权重。输入矩阵必须非空、矩形、维度匹配且元素有限；阈值有限且非负。常规维或异常维为空时，对应支路返回零矩阵；全零向量使用合法 scale 1。

代码使用 Python 浮点数和任意精度整数展示数值关系，不模拟 FP16、INT32 溢出或 CUDA 的布局与执行顺序。它会拒绝无法用 Python 浮点表示的 scale，以及有非零累加值时先下溢为零的 scale 乘积，避免在恢复尺度时静默丢失整项；这不是覆盖任意动态范围的通用数值库。

<!-- example: llm-int8-matrix-decomposition -->
```python
from math import fsum, isfinite, isclose


def checked_matrix(a):
    if not isinstance(a, (list, tuple)) or not a:
        raise ValueError("nonempty matrix required")
    if any(not isinstance(row, (list, tuple)) or not row for row in a):
        raise ValueError("nonempty rows required")
    if any(len(row) != len(a[0]) for row in a):
        raise ValueError("rectangular matrix required")
    try:
        result = [[float(v) for v in row] for row in a]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("finite numeric entries required") from exc
    if any(not isfinite(v) for row in result for v in row):
        raise ValueError("finite entries required")
    return result


def finite_sum(values):
    try:
        value = fsum(values)
    except (ValueError, OverflowError) as exc:
        raise ValueError("nonfinite accumulation") from exc
    if not isfinite(value):
        raise ValueError("nonfinite accumulation")
    return value


def quant_vector(values):
    maximum = max((abs(v) for v in values), default=0.0)
    scale = maximum / 127 if maximum else 1.0
    if scale <= 0 or not isfinite(scale):
        raise ValueError("unrepresentable scale")
    codes = [min(127, max(-127, round(v / scale))) for v in values]
    return codes, scale


def int8_branch(x, w, columns):
    # x and w have already passed matrix and dimension validation.
    m, n = len(x), len(w[0])
    qx = [quant_vector([row[k] for k in columns]) for row in x]
    qw = [quant_vector([w[k][j] for k in columns]) for j in range(n)]
    result = []
    for i in range(m):
        row = []
        for j in range(n):
            acc = sum(a * b for a, b in zip(qx[i][0], qw[j][0]))
            scale_product = qx[i][1] * qw[j][1]
            if not isfinite(scale_product) or (scale_product == 0 and acc != 0):
                raise ValueError("unrepresentable scale product")
            value = scale_product * acc
            if not isfinite(value):
                raise ValueError("nonfinite rescaling")
            row.append(value)
        result.append(row)
    return result


def mixed_mm(x, w, threshold=6.0):
    x, w = checked_matrix(x), checked_matrix(w)
    if len(x[0]) != len(w):
        raise ValueError("matrix dimensions do not match")
    try:
        threshold = float(threshold)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("finite nonnegative threshold required") from exc
    if not isfinite(threshold) or threshold < 0:
        raise ValueError("finite nonnegative threshold required")
    k, n = len(w), len(w[0])
    outliers = [j for j in range(k) if threshold > 0
                and any(abs(row[j]) >= threshold for row in x)]
    selected = set(outliers)
    regular = [j for j in range(k) if j not in selected]
    low = int8_branch(x, w, regular)
    high = [[finite_sum(row[j] * w[j][c] for j in outliers)
             for c in range(n)] for row in x]
    output = [[finite_sum([low[i][c], high[i][c]])
               for c in range(n)] for i in range(len(x))]
    return {"outlier_cols": outliers, "regular": low,
            "outlier": high, "output": output}


x = [[0.11, 0.2, 100.0], [0.2, -0.3, 0.5]]
w = [[1.0], [1.0], [1.0]]
result = mixed_mm(x, w)
assert result["outlier_cols"] == [2]
assert result["outlier"] == [[100.0], [0.5]]
assert isclose(result["output"][0][0], 100.31023622047244)
assert mixed_mm([[6.0]], [[2.0]], 6.0)["outlier_cols"] == [0]
assert mixed_mm([[6.0]], [[2.0]], 0.0)["outlier_cols"] == []
assert mixed_mm([[0.0, 0.0]], [[1.0], [2.0]])["output"] == [[0.0]]
assert mixed_mm([[7.0]], [[2.0]])["regular"] == [[0.0]]
assert mixed_mm([[1.0]], [[2.0]])["outlier"] == [[0.0]]
print(result)
```

注意第二行的浮点支路也有 0.5。它展示了“异常列的并集”如何作用于整个输入矩阵。代码没有让两条支路分别按照各自想象的元素集合计算，而是共享同一组列索引。

如果关闭常规支路的量化，仅沿相同列集合拆分并相加，浮点容差内应恢复原始矩阵乘；开启量化以后，变化才来自明确的量化位置。用这种顺序检查，可以把索引错误与量化误差分开。

## 论文里的浮点支路不等于任意运行时都保留原始权重

前面的参考直接对应原理图：普通列走量化，异常列使用原始浮点输入和权重。实际运行时还必须解决权重缓存、设备布局、算子调度和内存驻留，因此需要区分概念与特定实现。

以固定的 bitsandbytes **0.47.0** 为例，`int8_mixed_scaled_mm` 的 default 路径从原输入中抽取异常列，却从已经量化的 `CB` 反量化得到对应权重，再调用浮点 `addmm`。也就是说，这条浮点支路避免了异常 activation 的 INT8 编码，不代表它使用的 weight 完全没有经历量化。源码中的模块权重按 $$N\times K$$ 保存，与本文 $$K\times N$$ 记号转置。[0.47.0 的混合路径](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.47.0/bitsandbytes/backends/default/ops.py#L30)

范围统计也有区别。该版本的 default `int8_vectorwise_quant` 先把超阈值元素临时置零来计算每行 absmax，之后才把 selected columns 的 INT8 code 整列置零。教学参考则先删除整个 $$O$$，再为 $$R$$ 重算 scale。在前面两行输入的例子里，第二行的 0.5 没有超阈值，仍会参与该 default 路径的统计，row absmax 是 0.5；教学参考将第三列整体移出，第二行的常规维 absmax 是 0.3。两种约定会得到不同 code。[0.47.0 default 量化路径](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.47.0/bitsandbytes/backends/default/ops.py#L102)

因此，前面的程序是展示论文式列分解的参考，不是 bitsandbytes 0.47.0 的逐位复现。权重来源、统计集合和运算精度都必须先对齐，才适合比较数值结果。

从一次调用的生命周期看，activation 来自当前输入，需要动态统计与量化；静态权重的量化结果则可以缓存。主路生成累加值、恢复 scale 后，和异常列的浮点乘积合并，再产生输出。高精度输出并不说明主路没有使用低精度乘法；反过来，输入和权重的存储 dtype 是 INT8，也不保证底层一定调用 INT8 Tensor Core。

例如同一版本的 CUDA matmul 路径，在内部归约维不能被 4 整除时，会把 code 转成 FP32 做矩阵乘，再转成 INT32 输出。这里的数值输入仍是量化 code，执行算术却走了 fallback。这个条件是该版本源码的事实，不能扩展成所有 bitsandbytes 版本或所有硬件的通则。[0.47.0 CUDA matmul](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.47.0/bitsandbytes/backends/cuda/ops.py#L23)

因此，评估一个真实部署时，应分别问“存储了什么”“对哪些数做乘法”“累加和输出用什么类型”，而不是用 `load_in_8bit` 一个名字代替整条执行路径。

## 少数异常维怎样影响容量和延迟

若异常维确实很少，大部分乘积仍可使用 INT8，浮点支路只处理一个较窄的矩阵。这给容量与计算留下了同时改善的可能性，但“很少”必须由当前模型和输入决定。

对于固定阈值和同一批已知数值，向输入矩阵追加 token 行，异常列并集只能保持或增大。一行触发的列会影响同次矩阵乘中的其他行；batch、长度和输入分布变化后，不能默认每层永远只处理六七个异常维。

存储估算也必须匹配运行时。只看 INT8 权重主体，$$KN$$ 个参数约占 $$KN$$ byte，此外还有 scale、其他模块与布局开销。若另行保存异常列的原始 FP16 权重，需要为这些列增加存储；若为了动态选列而长期保留整份 FP16 master weights，就更不能只按一份 INT8 权重估算。前述从 `CB` 反量化的实现，则不是长期另存相同的原始 FP16 列。实际峰值还包括 activation、KV Cache、临时 buffer 和并发请求状态。

延迟方面，量化、统计、列选择、搬运、第二次矩阵乘与输出合并都要花时间。大矩阵上的 INT8 乘法收益可能覆盖这些开销，小 shape 则可能做不到。原论文也专门讨论了量化带来的加速与减速，不能把历史某个模型和设备上的倍数直接当作当前部署保证。[论文附录 D](https://arxiv.org/html/2208.07339v2#A4)

最后，还要分清论文中 performance 的语境。其主要质量结论对应所测模型上的 perplexity 与零样本任务表现；接近 FP16 baseline 的任务分数，不表示所有 logits 逐 bit 相同，也不是延迟不变的同义词。

## LLM.int8() 改变的是精度分配而不是所有算子的类型

LLM.int8() 的对象是 Transformer 中的线性投影与前馈矩阵乘。量化 Q、K、V 的投影权重，不等于把后续 $$QK^\top$$、softmax、$$PV$$ 的完整 attention function 都量化；它也不会自动把 KV Cache 改成 INT8。[论文 §6](https://arxiv.org/html/2208.07339v2#S6)

这种方法与 [SmoothQuant]({% post_url 2026-08-02-smoothquant-w8a8-activation-outliers %}) 的差别也很直接：前者把少数维度分配给另一条计算路径，后者通过等价缩放改变 activation 和 weight 的量化难度分布。与 [FP8]({% post_url 2026-08-01-fp8-formats-scaling-llm-inference %}) 相比，LLM.int8() 主路仍是均匀整数表示，不是给每个值增加一个浮点指数。

贯穿这篇文章的三维算例展示了它最重要的直觉：异常值不必自己出错，也能让共享 scale 的普通值出错。把少数异常列拆出去，首先是在为普通维度取回分辨率；能否进一步兑现容量、速度与任务质量，则取决于这份分解怎样落到具体模型和运行时。

## 参考资料

- [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/html/2208.07339v2)，§2–4、§6、附录 D；[首次公开日期与版本历史](https://arxiv.org/abs/2208.07339)。
- [bitsandbytes 0.47.0 functional API](https://huggingface.co/docs/bitsandbytes/v0.47.0/en/reference/functional)。
- [bitsandbytes 0.47.0 default 算子](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.47.0/bitsandbytes/backends/default/ops.py) 与 [CUDA 算子](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.47.0/bitsandbytes/backends/cuda/ops.py)。
