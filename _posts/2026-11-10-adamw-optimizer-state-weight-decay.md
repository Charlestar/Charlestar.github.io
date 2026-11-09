---
layout: post
title: "AdamW：梯度怎样变成参数更新，优化器状态又保存了什么"
subtitle: "从梯度的指数移动平均推导偏差修正与解耦衰减，再追踪逐参数步数、显存和断点恢复"
date: 2026-11-10
last_modified_at: 2026-11-10
author: iStar
catalog: true
series: distributed-training
series_order: 49
technology_year: 2017
mathjax: true
tags: [分布式训练, GPU优化]
---

反向传播得到梯度以后，参数还没有更新。梯度只回答当前目标在各个坐标上怎样变化；优化器需要决定沿哪个方向、走多远，以及此前积累的信息怎样影响这一步。

Adam 用两份跨 step 保留的统计量调整更新，AdamW 则把参数衰减从这些统计量中分离出来。因此，同样的当前梯度，配上不同的历史状态，可能得到不同更新；只保存模型权重，也不能一般性地恢复相同训练轨迹。

下面讨论最小化目标、实数 dense gradient、固定两个衰减系数、没有 AMSGrad 的基础 AdamW。公式采用 PyTorch 的学习率与 `weight_decay` 约定，框架行为固定对应 v2.8.0。[AdamW 论文](https://arxiv.org/html/1711.05101v3)最初发表于 2017 年预印本，正式发表于 ICLR 2019；它改变的是衰减与自适应更新的关系，并不重新定义反向传播。

## 梯度之外，为什么还要保留两份历史

设更新前参数为 $$\theta_{t-1}$$，当前梯度为 $$g_t=\nabla_\theta\mathcal L_t(\theta_{t-1})$$。这里的 $$\mathcal L_t$$ 可以是本次训练数据对应的目标；梯度归约、累积与解除 loss scaling 应先遵守训练循环的约定，再交给优化器。

Adam 为参数的每个坐标保存一阶状态 $$m_t$$ 和二阶状态 $$v_t$$，初始为零：

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
$$

$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
$$

平方、乘除与后面的平方根均按元素执行，$$0\le\beta_1,\beta_2<1$$。第一式保留带符号的梯度方向，第二式保留梯度平方的尺度。$$v_t$$ 不是减去均值后的方差，也不是 Hessian；把它称为二阶原始矩，更能避免误解。[Adam 算法 1](https://arxiv.org/html/1412.6980v8#S2)定义了这两条递推。

把第一式展开：

$$
m_t=(1-\beta_1)\sum_{i=1}^{t}\beta_1^{t-i}g_i
$$

近期梯度权重大，较早梯度逐渐衰减，但没有一个突然截断的窗口。取 $$\beta_1=0.9$$，前三次之后是 $$m_3=0.1g_3+0.09g_2+0.081g_1$$。如果梯度在正负之间来回抖动，带符号的平均可能较小，而平方平均仍然很大；这会压低该坐标的归一化更新。

两份状态也保留了不同时间尺度。较大的 $$\beta_2$$ 让平方尺度变化更缓慢，但它并不保证永远适合当前梯度分布。网络进入新阶段、数据变化或梯度突然增大时，历史统计怎样追随新信号，本来就是优化行为的一部分。

## 零初始化为什么需要偏差修正

上面的三个权重只加到 0.271，而不是 1。原因不是漏了哪次梯度，而是初始化状态为零，尚未积累到长期平衡。一般地，已有梯度的权重总和是：

$$
(1-\beta)\sum_{j=0}^{t-1}\beta^j=1-\beta^t
$$

因此 Adam 使用：

$$
\widehat m_t=\frac{m_t}{1-\beta_1^t},
\qquad
\widehat v_t=\frac{v_t}{1-\beta_2^t}
$$

用常梯度 $$g_i=c$$ 检查最直接：$$m_t=c(1-\beta_1^t)$$，修正后恰好是 $$c$$；平方状态修正后是 $$c^2$$。这消除了零初始化造成的权重缺失。在平稳矩假设下可以进一步解释其期望；非平稳训练中，它并不保证估计等于“当前真实均值”。[Adam §3](https://arxiv.org/html/1412.6980v8#S3)讨论的正是初始化偏差。

忽略 weight decay 时，更新方向为：

$$
u_t=\frac{\widehat m_t}{\sqrt{\widehat v_t}+\varepsilon},
\qquad
\theta_t=\theta_{t-1}-\eta_tu_t
$$

$$\eta_t$$ 是这次更新的学习率，$$\varepsilon$$ 位于平方根之外。在常非零梯度、$$\varepsilon=0$$ 的理想算例中，$$u_t=c/\lvert c\rvert$$；梯度幅值被尺度归一化，更新不再简单等于学习率乘原梯度。

但不能由此说 Adam 对所有梯度缩放都严格不变。这一理想性质要求整段梯度历史按同一个正比例缩放；非零 $$\varepsilon$$、改变缩放关系的梯度裁剪、未同比缩放的耦合 L2 项和浮点舍入都可能破坏它。在相同初始参数、相同学习率与衰减日程、零初始 moments 且 $$\varepsilon=0$$、分母非零的条件下，AdamW 的解耦衰减本身不破坏这种不变性。

尤其要注意，$$\sqrt{\widehat v_t}+\varepsilon$$ 与 $$\sqrt{\widehat v_t+\varepsilon}$$ 不是同一公式，尺度很小时差异很大。

重排偏差修正也必须连同 $$\varepsilon$$ 一起移动。设 $$c_1=1-\beta_1^t$$、$$c_2=1-\beta_2^t$$，等价式是：

$$
\frac{m_t/c_1}{\sqrt{v_t/c_2}+\varepsilon}
=\frac{\sqrt{c_2}}{c_1}
\frac{m_t}{\sqrt{v_t}+\varepsilon\sqrt{c_2}}
$$

只改变外面的步长，却遗漏分母中的 $$\sqrt{c_2}$$，会得到另一种数值规则。比较实现时，先把代数形式化到同一口径，比只比较变量名可靠。

## L2 正则为什么不能直接充当 Adam 的参数衰减

对不带动量的 SGD，给目标增加 $$\lambda\lVert\theta\rVert_2^2/2$$，梯度变成 $$g_t+\lambda\theta_{t-1}$$，于是：

$$
\theta_t
=\theta_{t-1}-\eta_t(g_t+\lambda\theta_{t-1})
=(1-\eta_t\lambda)\theta_{t-1}-\eta_tg_t
$$

这里 L2 梯度与直接收缩参数可以写成同一更新。但若把 $$g_t+\lambda\theta_{t-1}$$ 送入 Adam，它会同时进入 $$m_t$$ 与 $$v_t$$。正则项不仅改变当前方向，还改变未来的归一化尺度；它不再等价于参数独立乘一个收缩系数。[AdamW §2](https://arxiv.org/html/1711.05101v3#S2)把这两条路径明确分开。

AdamW 先用数据梯度形成 moments，再单独更新参数：

$$
\boxed{
\theta_t=(1-\eta_t\lambda)\theta_{t-1}
-\eta_t\frac{\widehat m_t}{\sqrt{\widehat v_t}+\varepsilon}
}
$$

这里“解耦”表示 $$\lambda\theta$$ 不进入梯度 moments，不表示收缩与学习率无关。按这套约定，学习率变化也会改变每步收缩量；要把系数理解成不反号的收缩，还需 $$0\le\eta_t\lambda\le1$$。

不同资料的同名参数也未必相同。原论文算法 2 将基础学习率 $$\alpha$$ 与调度倍率分开，而衰减乘调度倍率，不再乘 $$\alpha$$。若基础学习率 $$\alpha>0$$，要与本文 `lr × weight_decay` 形式匹配，需把论文的衰减系数除以 $$\alpha$$。比较配置时必须先映射公式，不能直接照搬一个名为 $$\lambda$$ 的数。

## 两个坐标的一步更新能看出什么

取 $$\theta_0=(1,2)$$、$$g_1=(1,1)$$，并设 $$\beta_1=0.9,\beta_2=0.99,\eta=0.1,\lambda=0.1$$。为了手算清楚，此例取 $$\varepsilon=0$$，且分母非零。

AdamW 得到 $$m_1=(0.1,0.1)$$、$$v_1=(0.01,0.01)$$，偏差修正后两者都是 $$(1,1)$$，因此：

$$
\theta_1=0.99(1,2)-0.1(1,1)=(0.89,1.88)
$$

若改成把 L2 项耦合进 Adam，输入 moments 的梯度是 $$(1.1,1.2)$$；第一步归一化方向仍为 $$(1,1)$$，但没有独立收缩，结果为 $$(0.9,1.9)$$。两者不仅当前参数不同，保存的 moments 也已经不同，后续轨迹不会因下一步设置相同梯度就重新一致。

衰减的执行顺序同样重要。设当前已算好的方向为 $$u_t$$，正确式是 $$(1-\eta\lambda)\theta-\eta u_t$$。如果先减更新，再把新参数整体收缩，会变成：

$$
(1-\eta\lambda)(\theta-\eta u_t)
=(1-\eta\lambda)\theta-\eta u_t+\eta^2\lambda u_t
$$

多出来的交叉项会连梯度更新一起缩小。用上述第一个坐标代入，正确结果是 0.89，后一种顺序则是 0.891。融合 kernel 可以合并内存访问，但不能随意改变这个更新合同。

## 三种不同的更新状态

继续第一个坐标，在第二步传入显式零梯度。历史状态仍然衰减：

$$
m_2=0.09,\qquad v_2=0.0099
$$

修正后 $$\widehat m_2\approx0.473684$$、$$\widehat v_2\approx0.497487$$，得到 $$\theta_2\approx0.8139419853$$。当前梯度为零，历史动量和独立衰减却仍然让参数移动。

[PyTorch v2.8.0 的 Adam 更新入口](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/adam.py)只把 `grad is not None` 的参数纳入本轮处理，因此缺失梯度是另一种情况：该参数的 moments 和 step 都不推进，也不执行这次 decay。AdamW 在[对应入口](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/adamw.py)选择解耦衰减路径。

| 传给某个参数的状态 | 参数值 | moments | 该参数 step |
| --- | --- | --- | --- |
| `grad=None` | 跳过本次更新 | 不变 | 不变 |
| 显式零梯度，学习率非零 | 仍可能移动 | 按系数衰减 | 增加 |
| 有梯度，学习率为零 | 不因本次公式改变 | 继续吸收梯度 | 增加 |

这里的 step 属于每个 `Parameter` 张量，moments 则与该张量同形状。条件分支、稀疏参与的模块或阶段性解冻，会使不同参数的步数不同。外层训练循环调用了一次 `step()`，不等于每个参数都完成了一次有效更新；AMP 检测到非有限梯度而跳步时也需要维持对应状态语义，见[混合精度训练]({% post_url 2026-09-13-mixed-precision-training-loss-scaling %})。

冻结参数也要处理已有梯度。将 `requires_grad` 改为 `False` 不会自动删除此前留下的 `.grad`；若旧梯度仍存在、参数又留在 optimizer 里，它可能仍被消费。应明确清理梯度、参数组与恢复时机，而不是把“以后不再求梯度”误认为“优化器立即忽略一切旧状态”。

纯衰减还有一个容易忽视的累计效应。若 moments 初始为零、每次提供显式零梯度且 $$\varepsilon>0$$，则：

$$
\theta_T=\theta_0\prod_{t=1}^{T}(1-\eta_t\lambda)
$$

相同 `weight_decay` 配上不同的更新次数或学习率日程，累计收缩就可能不同。改变梯度累积次数时，不应只核对有效 batch，还要意识到更新次数和调度单位是否一起改变。

## 用一个标量实现追踪完整状态

下面的 reference 将状态写成 $$(\theta,m,v,t)$$，只依赖 Python 标准库。为保持输入合同简单，它要求数值可转换为有限双精度浮点数、非负学习率和衰减、$$0\le\beta<1$$、$$\varepsilon>0$$，并拒绝非有限中间量；这比某些框架允许 $$\varepsilon=0$$ 的接口更窄。

<!-- example: adamw-scalar -->
```python
import math


def adamw_scalar(state, grad, *, lr=0.1, beta1=0.9, beta2=0.99,
                 weight_decay=0.1, eps=1e-12):
    theta, m, v, t = state

    def finite_real(value):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        try:
            return math.isfinite(value)
        except OverflowError:
            return False

    values = [theta, m, v, t, lr, beta1, beta2, weight_decay, eps]
    if grad is not None:
        values.append(grad)
    if not all(finite_real(x) for x in values):
        raise ValueError("finite real values required")
    if (isinstance(t, bool) or not isinstance(t, int) or t < 0
            or v < 0 or lr < 0 or weight_decay < 0 or eps <= 0
            or not 0 <= beta1 < 1 or not 0 <= beta2 < 1):
        raise ValueError("invalid state or hyperparameters")
    if grad is None:
        return state
    theta, m, v, lr, beta1, beta2, weight_decay, eps = map(
        float, (theta, m, v, lr, beta1, beta2, weight_decay, eps))
    grad = float(grad)
    t += 1
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad * grad
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    if not all(math.isfinite(x) for x in (m, v, m_hat, v_hat)):
        raise ValueError("non-finite moment update")
    denominator = math.sqrt(v_hat) + eps
    if not math.isfinite(denominator):
        raise ValueError("non-finite denominator")
    theta = ((1 - lr * weight_decay) * theta
             - lr * m_hat / denominator)
    if not all(math.isfinite(x) for x in (theta, m, v)):
        raise ValueError("non-finite update")
    return theta, m, v, t
```

<!-- example: adamw-state-example -->
```python
initial = (1.0, 0.0, 0.0, 0)
first = adamw_scalar(initial, 1.0)
missing = adamw_scalar(first, None)
zero = adamw_scalar(first, 0.0)
assert missing == first
assert first[3] == 1 and zero[3] == 2
assert math.isclose(first[0], 0.89, abs_tol=1e-11)
assert math.isclose(zero[0], 0.813941985271064, abs_tol=1e-11)
print(first, missing, zero)
```

它把状态转移显式暴露出来：在任意边界保存完整 tuple，再恢复并继续输入相同梯度，会延续同一计算；只恢复 $$\theta$$ 并把其余三项清零，则开始了另一条优化轨迹。生产实现还会涉及 dtype、设备、参数组和梯度同步，不能仅凭标量结果推断所有执行路径逐位一致。

仓库中的 `node scripts/reviews/adamw-examples.mjs` 会提取并执行这两段实际代码，再用直接展开历史加权和的独立 oracle 检查 moments、更新与恢复。它验证的是 CPU 标量算例和输入边界，不执行 PyTorch/CUDA 优化器，也不证明模型收敛或训练性能。

## 优化器状态为什么会超过模型文件

设有 $$P$$ 个可训练参数元素，参数、梯度、两个 moments 都采用 FP32。主体字节数为：

| 类别 | 大小 | 生命周期 |
| --- | ---: | --- |
| 参数 | $$4P$$ bytes | 跨 step 保留 |
| 参数梯度 | $$4P$$ bytes | 按训练循环清理或累积 |
| 一阶状态 $$m$$ | $$4P$$ bytes | 跨 step 保留 |
| 二阶状态 $$v$$ | $$4P$$ bytes | 跨 step 保留 |

合计是 $$16P$$，但不能把这四项全部称为 optimizer moments。两个 moments 本身是 $$8P$$，另外还有每个参数张量的 step 等小状态。对十亿参数，moments 约为 8 GB，四项主体约为 16 GB，均为十进制；activation、通信、临时张量和 allocator 尚未计入。

v2.8.0 的状态通常在参数首次拥有梯度并进入更新时初始化，而不是构造 optimizer 对象时就全部分配。这解释了为什么前向和反向已经通过，第一次 `step()` 仍可能触发显存峰值。初始化使用与参数关联的张量创建规则；AdamW 这个名字不自动承诺 FP32 moments，也不会自动给低精度参数生成 FP32 master copy。

显式混合精度方案可能采用“低精度计算参数 + FP32 master”，原生 autocast 则可能让参数本来就保持 FP32。这些不同组成已经在[混合精度的状态账单]({% post_url 2026-09-13-mixed-precision-training-loss-scaling %})中展开；[ZeRO/FSDP]({% post_url 2026-08-06-data-parallel-zero-fsdp-memory-sharding %})改变的则是这些状态由哪些 ranks 持有，不改变每次更新需要哪些数学信息。

## Fused 改变执行方式，不会让 moments 消失

逐参数、foreach 与 fused 路径，都可以实现上述目标，但发起算子的方式和临时存储不同。逐参数方式容易产生许多小操作；foreach 把多组张量一起处理；fused 路径进一步合并更新中的操作与内存访问。具体分派和支持条件取决于版本、设备与 dtype。

用一个粗略账本理解融合动机：若 FP32 更新把 $$\theta,g,m,v$$ 各读一次，再把 $$\theta,m,v$$ 各写一次，则流量主体是：

$$
Q_{model}=16P+12P=28P\ \text{bytes}
$$

这是单次理想读写的估算，省略缓存、临时中间量、AMSGrad 与通信，不是设备测得的流量。把更新拆成许多算子，可能反复读写状态；融合尝试减少这些额外搬运与启动，却仍然需要跨 step 保存 $$m,v$$。[Roofline]({% post_url 2026-09-09-roofline-gpu-performance-model %})中的存储边界与实际流量区别，在这里同样适用。

Foreach 还可能增加临时 tensorlist 的峰值内存，因此“批量处理更多参数”不等于“一定更省显存”。评价不同路径应固定参数布局、dtype、梯度与更新语义，同时比较 step 时间、完整训练时间和峰值。浮点运算重排也可能带来微小数值差异，不能把 fused 自动理解成与另一条路径逐位相同。

## 参数分组与恢复决定更新能否连续

参数组可以使用不同学习率、衰减和其他超参数。把 bias、norm 参数排除在衰减之外，是许多训练配方采用的选择，但不是 AdamW 公式强制规定的规则。按维度或名称分组时，应实际检查命中的参数、遗漏项以及共享权重，确保同一可训练参数不会被意外重复更新。

恢复训练时，模型权重之外还需要 moments、各参数 step、参数组与调度状态。PyTorch v2.8.0 的 [optimizer state_dict 实现](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/optimizer.py)按保存的组与参数顺序映射状态，并不会仅凭 `param_names` 就自动完成任意重排后的按名匹配。重建参数组顺序、冻结集合或模型结构后，要显式核对对应关系。

如果一份参数对应上了另一份同形状的 moments，加载可能不因形状而失败，训练却已经从错误历史继续。分布式重分片更要求参数与其 moments 共享同一逻辑映射，见[分布式 Checkpoint]({% post_url 2026-08-17-distributed-checkpoint-resharding-async-save %})。

最有辨识力的验证不是只看恢复后 loss 是否有限，而是在同一初始状态与梯度序列上比较连续执行和中途保存恢复的 $$\theta,m,v,t$$。再分别覆盖零梯度、缺失梯度、零学习率、不同参数组和实际 dtype；这样才能区分更新公式、状态映射与执行精度的问题。

AdamW 因而不只是“Adam 加一个 decay 参数”。它定义了梯度如何进入历史、历史如何归一化当前更新、收缩如何独立作用，以及这些状态怎样跨 step 延续。理解这条状态链，才能解释训练为什么移动、为什么停住，以及恢复以后为什么没有沿着原来的路径继续。

## 参考资料

- [Adam: A Method for Stochastic Optimization](https://arxiv.org/html/1412.6980v8)
- [Decoupled Weight Decay Regularization](https://arxiv.org/html/1711.05101v3)
- [PyTorch v2.8.0 AdamW](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/adamw.py)
- [PyTorch v2.8.0 Adam](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/adam.py)
- [PyTorch v2.8.0 Optimizer](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/optimizer.py)
