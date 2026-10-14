---
layout: post
title: "AWQ：为什么 4-bit 权重量化要先观察 Activation"
subtitle: "从两通道算例理解等价缩放、组内耦合、前向搜索与真实 W4A16 执行"
date: 2026-08-03 09:00:00 +0800
last_modified_at: 2026-10-15
author: iStar
catalog: true
series: model-quantization
series_order: 70
technology_year: 2023
mathjax: true
tags: [模型量化, GPU优化, LLM推理]
---

把一组权重映射到 4-bit 整数，首先要解决的是“有限的格点怎样分配”。按最大值确定范围、再就近取整，看起来已经合理：最大的权重没有溢出，其余权重也落在这个范围里。然而，权重数值接近，不代表量化它们的代价接近。一个很小的权重，如果经常乘上幅度很大的输入，它的少量舍入误差也可能明显改变输出。

AWQ（Activation-aware Weight Quantization）从这里出发：离线观察线性层的输入激活，用等价的通道缩放重新安排权重的数值分布，再进行规则的低比特分组量化。它要保护重要通道，但不希望最终格式变成“到处散落少量 FP16 权重”的不规则混合精度布局。

这也解释了名字中容易产生的误解：Activation-aware 表示**量化决策参考激活**，并不表示激活也量化到 4-bit。AWQ 的典型部署目标是 W4A16，即低比特权重配合 FP16/BF16 激活。下面从一次真实取整的计算开始，逐步看缩放怎样起作用，以及它为什么不是越大越好。

## 1. 权重误差为什么必须放回输入中衡量

沿用本系列的矩阵方向：

$$
X\in\mathbb R^{M\times K},\qquad
W\in\mathbb R^{K\times N},\qquad Y=XW.
$$

$$M$$ 是参与本次计算的 token 数，$$K$$ 是输入通道，$$N$$ 是输出通道。将量化再反量化后的权重记为 $$\widehat W$$，定义误差 $$E_W=W-\widehat W$$，就有：

$$
Y-\widehat Y=XE_W.
$$

一个输入通道的误差贡献，不只是权重差多少，还取决于这一列输入有多大。例如两个通道的权重误差同为 $$0.02$$，输入幅度分别是 $$0.2$$ 和 $$8$$，单项输出误差的幅度就是 $$0.004$$ 和 $$0.16$$。只比较权重自身的 MSE，会漏掉这个差异。

AWQ 使用的通道统计量是激活绝对值的平均值：

$$
a_j=\frac1M\sum_{t=1}^{M}\lvert X_{t,j}\rvert.
$$

论文的保护实验发现，按激活而非权重幅度选择少量通道保留 FP16，可以显著改善其特定低比特实验的质量；表 1 使用的是 OPT、INT3、group size 128 等设置。因此“保护约 1%”是理解方法的实验线索，不是所有模型统一采用的阈值，也不是最终 AWQ 格式必须残留 1% FP16 权重。[AWQ v5，§3.1](https://arxiv.org/html/2306.00978v5#S3.SS1)

不过，$$a_j$$ 只是一个易于收集的统计代理，不是完整的重要性度量。考虑两个输入：

$$
X_1=[1,1],\qquad X_2=[1,-1],\qquad
E_W=\begin{bmatrix}1\\-1\end{bmatrix}.
$$

它们各通道的绝对值统计完全相同，但输出误差分别是 $$0$$ 和 $$2$$。原因是通道之间会相关，误差也会抵消或叠加。严格的层输出平方误差包含 $$X^TX$$ 的交叉项：

$$
\lVert XE_W\rVert_F^2
=\operatorname{tr}\!\left(E_W^TX^TXE_W\right).
$$

AWQ 并不是只凭一个幅度统计就宣告量化完成：幅度用来构造缩放候选，候选还要通过前向输出误差来选择。

## 2. 先固定一个能手算的 4-bit 量化合同

为了把缩放机制算清楚，下面采用独立的对称教学量化器：一个非零 group 共享步长 $$\Delta$$，使用整数子码域 $$[-7,7]$$，最近值舍入采用 ties-to-even：

$$
\Delta=\frac{\max_i\lvert w_i\rvert}{7},\qquad
q_i=\operatorname{clip}\!\left(
\operatorname{RNE}(w_i/\Delta),-7,7\right),\qquad
\widehat w_i=\Delta q_i.
$$

这 15 个整数可由 4-bit 容纳，但**没有使用全部 16 个码值**。它不是官方 artifact 的编码声明，也不是在调用官方仓库的对称分支。后文会单独说明固定实现实际使用的 unsigned codes、zero point 与 packing。

全零组单独设 $$\Delta=1$$、codes 全为 $$0$$，保证步长为正且重建准确。所有例子只接收有限实数；NaN、Inf 不是可参与 absmax 搜索的普通输入。教学代码中的每个数组就是一个完整 group，不涉及补齐尾组。真实格式是否允许尾组，必须由它自己的 shape 合同决定。

现在取一个 token、两个输入通道和一个输出通道：

$$
X=[10,1],\qquad
W=\begin{bmatrix}0.1\\1\end{bmatrix},\qquad
Y=10\times0.1+1\times1=2.
$$

两个权重属于同一组，$$\Delta=1/7$$，得到：

$$
q=\begin{bmatrix}1\\7\end{bmatrix},\qquad
\widehat W=\begin{bmatrix}1/7\\1\end{bmatrix},\qquad
\widehat Y=\frac{10}{7}+1=2.428571\ldots
$$

误差主要来自原本更小的权重 $$0.1$$：它乘上的输入恰好大得多。若只按权重绝对值决定保护对象，很容易忽视它。

## 3. 等价缩放能改善结果，也能把结果重新变差

给输入通道选择正缩放 $$s_j$$，令 $$S=\operatorname{diag}(s)$$。在实数代数中：

$$
XW=(XS^{-1})(SW).
$$

也就是输入第 $$j$$ 列除以 $$s_j$$、权重第 $$j$$ 行乘以 $$s_j$$。没有量化时，两者严格抵消；在有限精度计算中则应检查相应的浮点误差。加入量化后，计算变成：

$$
\widehat Y_S=(XS^{-1})\mathcal Q(SW),
$$

其中 $$\mathcal Q$$ 表示量化再反量化，不只是整数编码。取 $$S=\operatorname{diag}(10,1)$$：

$$
XS^{-1}=[1,1],\qquad
SW=\begin{bmatrix}1\\1\end{bmatrix}.
$$

两个权重都落在格点上，重建后输出恰好是 $$2$$。重要的不是“把大输入变小”这个动作本身，而是**改变了同一 group 内两个权重的相对分布**，使有限格点更适合这次计算。

但如果进一步取 $$S=\operatorname{diag}(100,1)$$，结果会怎样？此时：

$$
XS^{-1}=[0.1,1],\quad
SW=\begin{bmatrix}10\\1\end{bmatrix},\quad
\Delta'=10/7.
$$

重建权重变成 $$[10,10/7]^T$$，输出再次变成 $$2.428571\ldots$$。第一个通道虽然被准确表示，却把整组的步长抬高，损害了另一个通道。

| 通道缩放 $$s$$ | 变换后权重 | group 步长 | 量化后的输出 |
| --- | --- | --- | --- |
| $$[1,1]$$ | $$[0.1,1]^T$$ | $$1/7$$ | $$2.428571\ldots$$ |
| $$[10,1]$$ | $$[1,1]^T$$ | $$1/7$$ | $$2$$ |
| $$[100,1]$$ | $$[10,1]^T$$ | $$10/7$$ | $$2.428571\ldots$$ |

更一般地，在没有 clipping、按最近值舍入的前提下，变换域权重的单元素重建误差至多为 $$\Delta'_g/2$$。还原到原通道单位后，误差界变成：

$$
\left\lvert W_{j,n}-\frac{\mathcal Q(SW)_{j,n}}{s_j}\right\rvert
\le\frac{\Delta'_g}{2s_j}.
$$

所以保护效果取决于 $$\Delta'_g/s_j$$，而不是只看 $$s_j$$ 是否增大。如果同组步长没有同比增大，这个误差界可以变小；但实际舍入误差会随格点位置跳变，界变小也不等于每个元素的实际误差都单调下降。论文对显著通道的分析与表 2 同样讨论了过强缩放伤害其他通道的现象。[AWQ v5，§3.2](https://arxiv.org/html/2306.00978v5#S3.SS2)

下面的标准库代码直接计算上述结果，并搜索下一节要介绍的 20 个候选。这里沿用教学对称量化器，**不是官方 AWQ 校准或 GPU 推理复现**。

<!-- example: awq-channel-scaling-and-grid -->
```python
from math import isclose, isfinite, sqrt

def sym_qdq(values):
    if not values or not all(isfinite(v) for v in values):
        raise ValueError("a group must contain finite values")
    peak = max(abs(v) for v in values)
    step = peak / 7.0 if peak else 1.0
    if not isfinite(step) or step <= 0:
        raise ValueError("step is outside this float reference's range")
    codes = [min(7, max(-7, round(v / step))) for v in values]
    restored = [step * q for q in codes]
    if not all(isfinite(v) for v in restored):
        raise ValueError("nonfinite reconstruction")
    return codes, restored, step

def scaled_forward(x, w, scales):
    if not (len(x) == len(w) == len(scales)):
        raise ValueError("channel counts must match")
    if not x or not all(isfinite(v) for v in x + w):
        raise ValueError("finite nonempty inputs required")
    if not all(isfinite(s) and s > 0 for s in scales):
        raise ValueError("scales must be finite and positive")
    scaled_x = [a / s for a, s in zip(x, scales)]
    scaled_w = [a * s for a, s in zip(w, scales)]
    for original, transformed in [(x, scaled_x), (w, scaled_w)]:
        if any(not isfinite(v) or (old != 0 and v == 0)
               for old, v in zip(original, transformed)):
            raise ValueError("transform outside this float reference's range")
    codes, restored, step = sym_qdq(scaled_w)
    products = [a * b for a, b in zip(scaled_x, restored)]
    if any(not isfinite(v) or (a != 0 and b != 0 and v == 0)
           for a, b, v in zip(scaled_x, restored, products)):
        raise ValueError("product outside this float reference's range")
    y = sum(products)
    if not isfinite(y):
        raise ValueError("nonfinite output")
    return y, codes, step

x, w = [10.0, 1.0], [0.1, 1.0]
target = sum(a * b for a, b in zip(x, w))
for scales, expected in [([1.0, 1.0], 17 / 7),
                         ([10.0, 1.0], 2.0),
                         ([100.0, 1.0], 17 / 7)]:
    y, codes, step = scaled_forward(x, w, scales)
    assert isclose(y, expected, abs_tol=1e-12)
    print("scales", scales, "codes", codes, "step", step, "y", y)

activation_meanabs = [abs(v) for v in x]  # one calibration token here
history = []
for i in range(20):
    ratio = i / 20
    raw = [max(a ** ratio, 1e-4) for a in activation_meanabs]
    normalizer = sqrt(max(raw) * min(raw))
    scales = [v / normalizer for v in raw]
    y, _, _ = scaled_forward(x, w, scales)
    history.append(((y - target) ** 2, ratio, y))
best = min(history, key=lambda item: item[0])
assert len(history) == 20 and history[-1][1] == 0.95
assert best[0] < history[0][0]
print("best (MSE, ratio, output):", best)

assert sym_qdq([0.0, 0.0]) == ([0, 0], [0.0, 0.0], 1.0)
assert sym_qdq([-1.0, 1.0])[0] == [-7, 7]
assert sym_qdq([0.5, 1.5, 7.0])[0] == [0, 2, 7]  # ties-to-even
for common in [0.1, 3.0, 10.0]:
    y, _, _ = scaled_forward(x, w, [common, common])
    assert isclose(y, 17 / 7, abs_tol=1e-12)
try:
    sym_qdq([float("nan")])
except ValueError:
    pass
else:
    raise AssertionError("non-finite input was accepted")
```

这组输入的 20 候选搜索选中 $$\alpha=0.85$$，输出约为 $$2.008955$$，比不缩放改善，但没有达到手工选择 $$[10,1]$$ 的精确结果。搜索只比较规定的候选，不是在所有正缩放向量中求全局最优；这里候选的通道尺度比最多为 $$10^{0.95}$$，并不包含手工例子的比值 $$10$$。

最后的公共缩放测试揭示了一个重要性质：在这个理想 absmax 量化器里，把整个 group 同乘一个正数，步长也同比改变，不能免费改善相对分辨率。AWQ 真正利用的是**通道之间不一样的缩放**。

这份 Python 浮点参考只用于检查上述有限算例。它会拒绝关键变换或乘积产生非有限值、非零量下溢为零等超范围情况，即使另一种运算顺序可能恢复出有限结果，也不尝试修复；这不是任意动态范围的模型量化库，更不是 FP16/BF16 逐 bit 仿真。

## 4. 20 个候选怎样从统计量变成量化参数

论文把高维缩放向量限制在一个简单的搜索族中：

$$
s_j=a_j^{\alpha}.
$$

$$\alpha$$ 越大，通道间激活幅度差异对缩放的影响越强。它不是梯度训练出来的逐权重参数，也不是直接指定“哪些权重保留高精度”。

固定到官方 `llm-awq` 的 commit `d6e797a`，`_search_module_scale` 使用 20 个比例 $$\alpha\in\{0,0.05,\ldots,0.95\}$$，不含 $$1$$。每个候选先执行：

$$
u_j=\max(a_j^{\alpha},10^{-4}),\qquad
s_j=\frac{u_j}{\sqrt{\max_i u_i\;\min_i u_i}}.
$$

注意 clamp 在归一化之前。因此最终 $$s_j$$ 不保证仍大于等于 $$10^{-4}$$。全零激活通道在正指数下先被 clamp 为正值；$$\alpha=0$$ 的候选在该实现中得到全 1 向量。这里的正下界防止除零，但不代表任意极端统计量在有限精度中都安全，实际校准仍应检查有限性和数值范围。

对每个候选，实现临时放大相关 Linear 的输入通道权重，执行 pseudo quantization，再除回通道缩放。这样在保持原输入 $$X$$ 不变的情况下，就能评价等效权重 $$S^{-1}\mathcal Q(SW)$$。它比较原模块输出和候选模块输出的 MSE，将输出差转换为 FP32 后平方取均值，然后恢复原模块状态，避免候选之间累积修改。检查对象还可能是包含多个投影的 Attention 或 MLP，而不只是一个独立 Linear。[固定版 auto_scale.py](https://github.com/mit-han-lab/llm-awq/blob/d6e797a42b9ef7778de8ee2352116e0f48a78d61/awq/quantize/auto_scale.py)

因此，“AWQ 不做 reconstruction”需要讲清楚比较对象：它不像 [GPTQ]({% post_url 2026-10-15-gptq-second-order-reconstruction %}) 那样，利用二阶信息对未量化权重连续补偿，也不通过反向传播训练这些缩放参数；但它**确实根据校准数据上的前向输出误差搜索 scale 和 clipping**。不反传并不等于不看输出，更不等于校准数据完全无关。

激活平均幅度决定搜索空间，前向 MSE 决定这个空间里选哪个候选。前者压缩了搜索成本，后者补上了仅看通道幅度遗漏的一部分相互作用。不过，这仍不是完整任务质量的全局优化；某些语言、模态或 MoE expert 没有被校准样本覆盖时，局部统计和候选选择也可能失真。

## 5. Clipping 优化的是哪一种误差

通道缩放之后，一个 group 中仍可能有少数大权重拉宽范围。Clipping 允许牺牲这些端点，换取主体权重更细的步长。对上面的教学量化器，给定阈值 $$c_g>0$$：

$$
w_i^{\mathrm{clip}}=\operatorname{clip}(w_i,-c_g,c_g),\qquad
\widehat w_i=\mathcal Q(w_i^{\mathrm{clip}}).
$$

它的目标不是让裁剪后的权重最接近原权重，而是让**量化后的计算**尽量保留原输出。

固定版 `auto_clip.py` 的默认参数是 `n_grid=20, max_shrink=0.5`，实际循环只有 10 个候选，阈值为原 group absmax 的 $$1,0.95,\ldots,0.55$$ 倍，不含 $$0.5$$。它按 output channel 和输入 group 分别比较该 group 的部分点积输出，在采样 token 上求 MSE，再独立选择阈值；并且按模块名跳过包含 `q_`、`k_`、`query`、`key`、`Wqkv` 的 Linear，不是所有投影一律裁剪。[固定版 auto_clip.py](https://github.com/mit-han-lab/llm-awq/blob/d6e797a42b9ef7778de8ee2352116e0f48a78d61/awq/quantize/auto_clip.py)

这个代理很实用，但它与整层输出 MSE 有区别。设某输出通道的第 $$g$$ 组误差为：

$$
e_g=X_{:,G_g}\left(W_{G_g,n}-\widehat W_{G_g,n}\right).
$$

独立组目标关注各个 $$\lVert e_g\rVert_2^2$$；整层目标却是：

$$
\left\lVert\sum_g e_g\right\rVert_2^2
=\sum_g\lVert e_g\rVert_2^2
+2\sum_{g<h}e_g^Te_h.
$$

前者没有保留组间交叉项。下面是仍用对称教学合同构造的明确反例：

$$
X=[1,2,-1,0.5],\qquad
W=[0.1,0.2,0.1,0.2]^T,
$$

每两个权重一组，原始输出为 $$0.5$$。先定义便于下表阅读的“近似输出减原输出”误差，它与上式 $$e_g$$ 相反，但不影响平方误差。

| 组 | 不裁剪时的部分输出误差 | 独立搜索选中的收缩系数 | 搜索后的部分输出误差 |
| --- | --- | --- | --- |
| $$[0.1,0.2]$$，输入 $$[1,2]$$ | $$+1/70$$ | $$0.95$$ | $$-2/175$$ |
| $$[0.1,0.2]$$，输入 $$[-1,0.5]$$ | $$-1/70$$ | $$0.8$$ | $$-2/175$$ |

每组的绝对误差都从约 $$0.014286$$ 降到 $$0.011429$$，但原先相互抵消的总误差变成同向叠加：输出从准确的 $$0.5$$ 变成 $$0.477142857\ldots$$。下面完整执行这 10 个候选的搜索，不靠手工指定最优阈值。

<!-- example: awq-clipping-local-versus-global -->
```python
from math import isclose, isfinite

def quantize_clipped(group, factor):
    if not group or not all(isfinite(v) for v in group):
        raise ValueError("finite nonempty group required")
    if not isfinite(factor) or not 0 < factor <= 1:
        raise ValueError("invalid clipping factor")
    peak = max(abs(v) for v in group)
    if peak == 0:
        return [0.0 for _ in group]
    threshold = peak * factor
    if threshold == 0 or not isfinite(threshold):
        raise ValueError("threshold is outside this float reference's range")
    step = threshold / 7
    if not isfinite(step) or step <= 0:
        raise ValueError("step is outside this float reference's range")
    clipped = [min(threshold, max(-threshold, v)) for v in group]
    restored = [step * min(7, max(-7, round(v / step))) for v in clipped]
    if not all(isfinite(v) for v in restored):
        raise ValueError("nonfinite reconstruction")
    return restored

def dot(a, b):
    assert len(a) == len(b)
    return sum(x * y for x, y in zip(a, b))

groups = [([1.0, 2.0], [0.1, 0.2]),
          ([-1.0, 0.5], [0.1, 0.2])]
original, baseline, selected = [], [], []
local_before = local_after = 0.0
for x, w in groups:
    reference = dot(x, w)
    original.append(reference)
    base = dot(x, quantize_clipped(w, 1.0))
    baseline.append(base)
    candidates = []
    for i in range(10):
        factor = 1 - i / 20
        prediction = dot(x, quantize_clipped(w, factor))
        candidates.append(((prediction - reference) ** 2, factor, prediction))
    loss, factor, prediction = min(candidates, key=lambda item: item[0])
    selected.append(prediction)
    local_before += (base - reference) ** 2
    local_after += loss
    print("factor", factor, "before", base - reference,
          "after", prediction - reference)

target = sum(original)
assert isclose(target, 0.5, abs_tol=1e-12)
assert isclose(sum(baseline), target, abs_tol=1e-12)
assert isclose(sum(selected), 0.5 - 4 / 175, abs_tol=1e-12)
assert local_after < local_before
assert (sum(selected) - target) ** 2 > (sum(baseline) - target) ** 2
print("group proxy:", local_before, "->", local_after)
print("whole output:", target, "->", sum(baseline), "->", sum(selected))
```

这不是“AWQ clipping 总会变差”的结论，也不是官方非对称量化的实验结果。它说明了一件更具体的事：**独立 group 误差改善，不足以证明整层输出必然改善**。同样，整层 MSE 改善也不能直接推导完整模型的生成质量。理解优化目标的范围，比把某个局部指标称为“精度”更重要。

## 6. 逆缩放放在哪里，取决于计算图

到目前为止，公式都把 $$XS^{-1}$$ 显式写出来。如果每个 Linear 前都单独加除法，可能引入额外计算和中间张量。实际模型改造会尽量把逆缩放吸收到前面的算子，但这个操作必须保持所有消费分支的语义。

对于 LayerNorm 输出 $$X=Z\odot\gamma+\beta$$，可以把 $$\gamma$$ 和 $$\beta$$ 都除以 $$s$$，并让所有消费该输出的 Linear 输入通道权重乘以 $$s$$。不能只改 $$\gamma$$ 而遗漏 $$\beta$$；Q、K、V 共享这个输入时也不能只修改其中一个投影。RMSNorm 没有相应 bias 时，就没有那一项需要处理。当前 Linear 的输出 bias 不属于这个输入通道变换，不应跟着任意乘除。

如果前面是另一个可直接配对的 Linear，则需要相应缩放其输出行和 bias；若中间经过 reshape、分支、门控乘法或共享参数，还要结合具体图结构判断，不能把“前一个 Linear”机械当作总能吸收的位置。

尤其不能把缩放随意穿过 GELU：

$$
\operatorname{GELU}(h/s)\ne\operatorname{GELU}(h)/s
$$

一般成立的是“先计算非线性输出，再做逆缩放”。固定实现的 `ScaledActivation.forward` 正是返回 `act(x) / scales`，而非 `act(x / scales)`；随后下游 Linear 权重乘以对应尺度。[固定版 qmodule.py](https://github.com/mit-han-lab/llm-awq/blob/d6e797a42b9ef7778de8ee2352116e0f48a78d61/awq/quantize/qmodule.py)

这个小例子只检查浮点图变换，不做任何量化。正确写法保持输出，错误地穿过 GELU 则改变原模型函数。

<!-- example: awq-scaled-activation-folding -->
```python
from math import erf, isclose, sqrt

def gelu(value):
    return 0.5 * value * (1 + erf(value / sqrt(2)))

def linear(row, weights):
    assert len(row) == len(weights)
    return [sum(row[k] * weights[k][n] for k in range(len(row)))
            for n in range(len(weights[0]))]

h = [-1.0, 0.5]
scales = [2.0, 0.5]
weights = [[0.4, -0.3], [0.7, 0.8]]  # K x N
scaled_weights = [[scales[k] * v for v in row]
                  for k, row in enumerate(weights)]
reference = linear([gelu(v) for v in h], weights)
correct = linear([gelu(v) / s for v, s in zip(h, scales)], scaled_weights)
wrong = linear([gelu(v / s) for v, s in zip(h, scales)], scaled_weights)
assert all(isclose(a, b, abs_tol=1e-12) for a, b in zip(reference, correct))
assert max(abs(a - b) for a, b in zip(reference, wrong)) > 0.01
print("reference:", reference)
print("after activation, correct:", correct)
print("before activation, wrong:", wrong)
```

因此，“等价缩放通常可以融合”不等于“所有缩放都自动零成本”。有些位置能够通过参数重写吸收，有些需要保留输出缩放或由 kernel 显式融合；如何处理是计算图与实现的问题，不是代数恒等式独自能保证的。

## 7. 从浮点 Q/DQ 到真正的 AWQ Artifact

前面的 $$[-7,7]$$ 例子方便计算，但部署不能只凭“4-bit”三个字决定格式。固定版 `real_quantize_model_weight` 要求 `zero_point=True`，对应 `WQLinear` 仅支持 4-bit。其 pseudo quantization 对每个 group 计算：

$$
\Delta=\frac{\max(\max(w)-\min(w),10^{-5})}{15},\qquad
z=\operatorname{clip}\!\left(-\operatorname{RNE}(\min(w)/\Delta),0,15\right).
$$

随后先舍入 $$w/\Delta$$，再加 $$z$$ 并截断到 $$[0,15]$$，最后重建 $$\Delta(q-z)$$。零组由正下界避免除零。这是源码的具体数值顺序，不应拿另一套“相似的 min-max 公式”当成逐位相同实现。[固定版 quantizer.py](https://github.com/mit-han-lab/llm-awq/blob/d6e797a42b9ef7778de8ee2352116e0f48a78d61/awq/quantize/quantizer.py)

该文件的 `zero_point=False` 分支还包含在本分支赋值前引用 `min_val` 的断言，正常断言开启时不能据此宣称“官方对称路径已经可直接运行”。本文对称代码是自包含的数值参考，与该分支无关；这也不改变上面非对称 real-quant 入口的合同。

Pseudo quantization 返回的是浮点重建权重。它能估计舍入后的输出，但这些权重仍放在浮点 tensor 中，并没有因为数值落在低比特格点上就自动节省存储。真正转换还要生成整数 codes、打包、保存 group scales 和零点信息，并替换执行算子。

这里也要区分矩阵方向：本文的 $$W$$ 是 $$K\times N$$；PyTorch Linear 的存储通常是 $$N\times K$$，所以源码按最后一维连续分组，对应本文沿输入 $$K$$ 轴分组。输入通道缩放在本文左乘 $$S$$，在源码里体现为每个输出行的对应输入列乘以 scale。把两个方向混用，会出现形状能广播、语义却完全错误的问题。

固定版 `WQLinear` 不只是把 8 个 nibble 顺序塞进一个 int32：它有输入维重排、组内次序重排、输出行 interleave，并把 packed 权重存为 int16 tensor；scales 和 `scaled_zeros` 也有转置与 padding，其中后者保存的是 $$-\Delta z$$ 形式的浮点偏移。`forward` 按输入行数选择 GEMV 或 GEMM CUDA 入口。[固定版 qmodule.py 的 packing 与 forward](https://github.com/mit-han-lab/llm-awq/blob/d6e797a42b9ef7778de8ee2352116e0f48a78d61/awq/quantize/qmodule.py)

这意味着，group size、输入轴、整除/对齐要求、zero-point 表示、scale dtype、packing 次序和排除量化的模块共同定义 artifact。这个固定实现的量化入口要求 group size 整除输入维，kernel packing 也有额外 shape 约束，不能把教学代码任意长度的数组直接当成可部署 shape。两个文件都叫 AWQ，不代表可以互换 decoder。

一个具体的对齐检查是：先用 artifact 的 codes、scales 和零点规则解出小块浮点权重，再与量化前搜索所使用的 Q/DQ 参考比较，最后比较相同输入下真实算子的输出。若参考正常、packed 路径异常，应优先检查格式与 kernel 合同，不能把它混成“AWQ 算法精度不好”。

## 8. W4A16 节省的是什么，不能从名字推导什么

W4A16 首先描述权重存储与激活的精度组合，不自动规定累加精度、最终输出 dtype，也不保证硬件直接执行某种 INT4×FP16 指令。TinyChat 在论文所述平台上采用 on-the-fly dequantization，把解包、反量化与矩阵计算融合，避免先把全部重建权重写回 DRAM；具体 operand、累加器和输出仍要看所用 kernel，而不是只看算法名称。[AWQ v5，§4.2](https://arxiv.org/html/2306.00978v5#S4.SS2)

只看权重 payload，16-bit 到 4-bit 是 4 倍压缩；加入 metadata 就不是精确 4 倍。例如若采用上述固定实现、group size 为 128、scale 和 `scaled_zeros` 都用 FP16，忽略 padding 时，每个权重平均占用：

$$
4+\frac{16+16}{128}=4.25\ \text{bits}.
$$

相对 16-bit 权重主体约为 $$16/4.25\approx3.76$$ 倍压缩。这只是这个 metadata 合同下的预算，不是所有 AWQ 格式统一的 bits/weight；padding、未量化的 embedding/norm/head、bias、运行时 workspace 还要另外计入。

同样，AWQ 不自动量化 KV Cache。权重变小能给 KV 留出更多显存，但每个 token 的 KV 容量仍由模型结构和 KV dtype 决定。长上下文、高并发时，节省权重容量和降低单步权重带宽是两个不同收益，不能都归成“推理提速”。

从计算上看，decode 小 batch 的线性层往往反复读取权重，压缩可能减少主要带宽流量；prefill 中较多 token 复用相同权重，计算占比上升，解包、反量化与 W4 kernel 的吞吐会影响收益。group size 也同时影响数值与系统：小 group 能更局部地设置范围，但 metadata 更多、载入与索引更复杂，不能声称“缩小 group 一定让实际输出误差或端到端延迟单调变好”。

论文 TinyChat 的加速结果属于它的模型、硬件、baseline 与软件实现组合，不能直接转移成另一套 runtime 的速度承诺。应把同输入下的数值对齐，与相同负载下的 prefill、decode 和显存结果分别确认：加载成功、文件变小和任务质量保持，是三件不同的事。

## 9. 把 AWQ 放回量化方法之间的关系

[SmoothQuant]({% post_url 2026-08-02-smoothquant-w8a8-activation-outliers %}) 也使用 $$XW=(XS^{-1})(SW)$$，但典型目标是 W8A8：把难处理的激活幅度差异转移一部分给权重，帮助激活进入 INT8。AWQ 的典型目标是 W4A16：激活仍保留高精度，用它来判断怎样为低比特权重安排更合适的分组数值分布。同一个代数恒等式，服务的是不同的量化瓶颈。

GPTQ 则从校准输入对应的层重建目标出发，用二阶信息在逐步量化时补偿其余权重。AWQ 限制在激活统计构造的 scale 候选和 clipping 候选中，前向评价而不反向训练。二者都可能产出 grouped 4-bit 权重，但方法、格式和 kernel 兼容性不是同一层概念，不能只凭同样写着 W4 就互换实现或直接比较结论。

理解 AWQ 可以抓住一条连续的因果链：输出误差经过输入放大，因此观察激活；观察结果用来构造通道缩放，但缩放改变整组步长，所以要做前向搜索；局部搜索只优化代理目标，所以仍要检查整体质量；而数值质量要转成真实节省，还必须有匹配的 packing 和 W4A16 执行路径。

AWQ 的价值不是找出一批“永远重要”的权重，而是在规则低比特格式的约束下，利用输入信息重新分配量化误差。把这件事与任意精度混用、完整模型重训练、KV Cache 压缩和端到端加速区分开，才能准确理解它解决了什么。

## 参考资料

- [AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration，arXiv v5](https://arxiv.org/html/2306.00978v5)：本文依据的方法与系统说明；该修订发布于 2024-07-18，研究最初发表于 2023 年。
- [固定版 auto_scale.py](https://github.com/mit-han-lab/llm-awq/blob/d6e797a42b9ef7778de8ee2352116e0f48a78d61/awq/quantize/auto_scale.py)：通道统计、20 候选搜索、模块输出目标与参数折叠。
- [固定版 auto_clip.py](https://github.com/mit-han-lab/llm-awq/blob/d6e797a42b9ef7778de8ee2352116e0f48a78d61/awq/quantize/auto_clip.py)：分组部分输出误差与 clipping 候选。
- [固定版 quantizer.py](https://github.com/mit-han-lab/llm-awq/blob/d6e797a42b9ef7778de8ee2352116e0f48a78d61/awq/quantize/quantizer.py)、[qmodule.py](https://github.com/mit-han-lab/llm-awq/blob/d6e797a42b9ef7778de8ee2352116e0f48a78d61/awq/quantize/qmodule.py)：Q/DQ、zero-point、packing 和执行入口；全文源码边界固定在同一个 commit，不代表其他版本的支持矩阵。
