---
layout: post
title: "DPO：成对偏好怎样直接改变语言模型的概率"
subtitle: "从 KL 正则与奖励差推导 pair loss，再追踪序列 log probability、Reference 与数据边界"
date: 2026-10-13
last_modified_at: 2026-10-13
author: iStar
catalog: true
series: distributed-training
series_order: 105
technology_year: 2023
mathjax: true
tags: [分布式训练, 机器学习, 工程实践]
---

同一个问题，模型给出两份回答：一份准确解释 BPE 使用已经学好的合并顺序，另一份却声称“每次收到新输入，都会重新合并出现最多的字符对”。我们知道第一份更合适，这个相对判断怎样变成语言模型的训练信号？

监督微调可以只保留较好的回答，让模型学习生成它；成对偏好还提供了额外信息：面对相同问题，哪些回答应当比另一些回答更受支持。DPO（Direct Preference Optimization）用一个直接依赖生成概率的损失学习这种关系，训练时可以使用固定的离线偏好数据。

[2023 年首次公开的 DPO 论文](https://arxiv.org/abs/2305.18290)从带 KL 正则的奖励优化出发，推导出这一目标。我们先在有限回答集合中推导关系，再看一条回答的概率怎样从 token logits 算出来。本文限定原始 sigmoid DPO、固定 Reference、不加 label smoothing 或其他混合损失；数值代码都是标准库算例，没有运行真实模型偏好训练。

## 一条偏好记录表达了什么

设问题为 $$x$$，被选中的回答为 $$y_w$$，另一个回答为 $$y_l$$。下标 w、l 只是 winner、loser 的约定，也常对应数据字段 `chosen` 和 `rejected`：

```json
{
  "prompt": "BPE 在处理新文本时，怎样决定先合并哪一对符号？",
  "chosen": "使用训练时确定的合并顺序，优先应用当前可用且 rank 较小的规则。",
  "rejected": "重新统计这段新文本中的字符对，优先合并当前出现次数最多的一对。"
}
```

这是我们自己构造的技术问答。偏好标准是原理是否正确，并不是哪段文字更长。换成摘要、代码或客户服务，标准可能还包括任务完成度、表达与格式，但必须先确定我们希望模型学到什么。

数据里的 `chosen` 也不自动等于事实真值。一份不够好的回答可以击败一份更差的回答；评审者也可能偏爱流畅却错误的解释。优化器只看标签与概率，不能替我们发现标签的标准或事实本身有问题。

监督微调以示范回答的负对数似然作为训练目标；DPO 的一条记录同时使用两份回答，并通过 Reference 比较它们相对原模型的变化。它训练的仍是生成模型，并非额外加一个“这两段谁更好”的分类头。

## Reference 为什么出现在目标里

记待训练的生成策略为 $$\pi_\theta(y\mid x)$$，固定参考策略为 $$\pi_{\mathrm{ref}}(y\mid x)$$。常见做法从一份 SFT 模型开始，复制它作为 Reference，并将 policy 初始化为相同状态。Reference 不是正确答案库，而是一份提供基准概率分布的模型。

如果已经有奖励函数 $$r(x,y)$$，只追求最高奖励可能让策略偏离原有分布；奖励在陌生输出上的误差还可能被放大。带 KL 正则的目标把奖励与偏离成本放在一起：

$$
\max_\pi\ \mathbb E_{x}\left[
\mathbb E_{y\sim\pi(\cdot\mid x)}r(x,y)
-\beta D_{\mathrm{KL}}\big(\pi(\cdot\mid x)\Vert\pi_{\mathrm{ref}}(\cdot\mid x)\big)
\right],\qquad\beta>0
$$

这里 KL 的方向是 policy 相对于 Reference，即 $$D_{\mathrm{KL}}(\pi\Vert\pi_{\mathrm{ref}})$$。正则项并不证明答案正确，它约束的是概率分布的偏离。这是 [DPO 原文 §3–4](https://arxiv.org/html/2305.18290v3#S3)采用的起点。

现在固定一个 prompt，用有限的完整回答集合 $$\mathcal Y$$ 作为这一节抽象模型的整个支持集。令 $$p(y)=\pi(y\mid x)$$、$$q(y)=\pi_{\mathrm{ref}}(y\mid x)$$，两者在这个集合上都归一化，即 $$\sum_{y\in\mathcal Y}p(y)=\sum_{y\in\mathcal Y}q(y)=1$$；假设每个 $$q(y)>0$$，奖励固定且有限。这不是从真实模型中随意挑几个候选、直接把其未经归一化的概率当作完整分布。目标写成：

$$
J(p)=\sum_y p(y)r(y)-\beta\sum_y p(y)\log\frac{p(y)}{q(y)}
$$

这里 $$p(y)\ge0$$，概率为 0 的项按连续极限约定取 0。

定义：

$$
Z=\sum_yq(y)\exp(r(y)/\beta),\qquad
p^*(y)=\frac{q(y)\exp(r(y)/\beta)}{Z}
$$

有限集合与有限奖励使这里的 $$Z$$ 有限。展开 $$D_{\mathrm{KL}}(p\Vert p^*)$$ 可以得到：

$$
\begin{aligned}
D_{\mathrm{KL}}(p\Vert p^*)
&=\sum_y p(y)\log\frac{p(y)}{q(y)}
-\frac{1}{\beta}\sum_y p(y)r(y)+\log Z\\
J(p)&=\beta\log Z-\beta D_{\mathrm{KL}}(p\Vert p^*)
\end{aligned}
$$

KL 非负，所以在这些前提下，$$p=p^*$$ 取得最大值。最优分布是在 Reference 上，按 $$\exp(r/\beta)$$ 对回答重新加权。

这份有限分布推导没有证明神经网络一定能表达 $$p^*$$，也没有证明有限数据、有限步数的优化会找到它。推广到所有 token 序列，还需要检查支持集与可归一化条件；不能省略这些前提，就将闭式解当作任何训练运行的收敛保证。

## 两份回答相减，消掉了未知的归一化项

从上式反解奖励：

$$
r(x,y)=\beta\log\frac{\pi^*(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}+\beta\log Z(x)
$$

直接用这个奖励仍有困难：$$Z(x)$$ 涉及整份回答分布。但偏好通常只比较同一问题的两份回答，奖励差中与回答无关的 $$\beta\log Z(x)$$ 恰好抵消。

采用 Bradley–Terry 偏好模型，令：

$$
P(y_w\succ y_l\mid x)=\sigma(r(x,y_w)-r(x,y_l)),\qquad
\sigma(z)=\frac{1}{1+\exp(-z)}
$$

这是对偏好概率的一种建模假设。它没有规定评审者永远一致，也没有把奖励解释成客观正确率。

用可训练的 $$\pi_\theta$$ 参数化前面的 log-ratio，定义相对变化：

$$
\Delta_\theta=
\left[\log\pi_\theta(y_w\mid x)-\log\pi_\theta(y_l\mid x)\right]
-\left[\log\pi_{\mathrm{ref}}(y_w\mid x)-\log\pi_{\mathrm{ref}}(y_l\mid x)\right]
$$

再令 $$m=\beta\Delta_\theta$$。一条标记为 $$y_w\succ y_l$$ 的样本，其负对数似然为：

$$
\ell_{\mathrm{DPO}}=-\log\sigma(m)
=\log(1+\exp(-m))
=\operatorname{softplus}(-m)
$$

对训练集平均这些损失，就是原始 DPO 的目标。这与 [论文正文 Eq.7](https://arxiv.org/html/2305.18290v3#S4)及[作者代码中的 `preference_loss`](https://github.com/eric-mitchell/direct-preference-optimization/blob/main/trainers.py)一致。

Reference 固定后，计算一条 pair 的标量损失只需四个数：policy 对 chosen/rejected 的 log probability，以及 Reference 对同两份回答的 log probability。原始 DPO 不需要先训练独立 reward model，再通过在线采样优化它；奖励差被直接写成了生成模型概率的函数，policy 分数仍须保留通向模型参数的梯度。

$$\sigma(m)$$ 表示这个偏好模型对 chosen 胜过 rejected 的支持，既不是生成 chosen 的绝对概率，也不是 chosen 事实正确的概率。这三个量不能互换。

## 相对更受支持，不代表绝对概率一定上升

用三份完整回答 $$[w,l,o]$$ 构造一个有限分布。假设 Reference 和旧 policy 都是 `[0.4,0.4,0.2]`，新 policy 是 `[0.3,0.1,0.6]`，取 $$\beta=1$$：

| 分布 | chosen 概率 | rejected 概率 | 二者概率比 |
| --- | ---: | ---: | ---: |
| Reference / 旧 policy | 0.4 | 0.4 | 1 |
| 新 policy | 0.3 | 0.1 | 3 |

旧 margin 为 0，损失为 $$\log2\approx0.693147$$；新 margin 为 $$\log3$$，损失为：

$$
-\log\sigma(\log3)=-\log(3/4)=\log(4/3)\approx0.287682
$$

pair loss 明显降低，chosen 的绝对概率却从 0.4 降到了 0.3。这是目标函数允许的一种分布变化，不是声称一次独立三分类 SGD 更新必然走这条轨迹。

因此，DPO 鼓励的是相对 Reference 的偏好 margin。实际模型共享参数，许多样本同时施加梯度，不能保证每个 chosen 的绝对概率都逐步上升；更不能从训练 pair loss 下降，直接推断任意新问题的生成答案都会更正确。

## 梯度怎样区分已经学会和仍然排错的 pair

对 $$\Delta_\theta$$ 求导：

$$
\frac{\partial\ell}{\partial\Delta_\theta}
=-\beta\sigma(-\beta\Delta_\theta)
$$

Reference 固定时，对模型参数的梯度为：

$$
\nabla_\theta\ell
=-\beta\sigma(-m)
\left[\nabla_\theta\log\pi_\theta(y_w\mid x)
-\nabla_\theta\log\pi_\theta(y_l\mid x)\right]
$$

若 chosen 的相对变化已经远高于 rejected，$$m$$ 大而为正，前面的权重很小；若隐式奖励仍把 rejected 排在 chosen 前面，权重更大。每条 pair 的力度由当前 margin 决定，并非永远对 chosen/rejected 施加相同幅度的更新。

下面用稳定的 sigmoid 和 softplus 复算。输入是我们指定的完整回答 log probabilities，函数返回损失、对 delta 的导数与 margin，不计算神经网络梯度。

<!-- dpo-example: loss-and-gradient -->
```python
import math


def sigmoid(value):
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def dpo_pair(policy_w, policy_l, ref_w, ref_l, beta):
    values = [policy_w, policy_l, ref_w, ref_l, beta]
    if not all(math.isfinite(value) for value in values) or beta < 0:
        raise ValueError("Finite log probabilities and nonnegative beta required")
    # beta=0 is accepted only to inspect the degenerate boundary case.
    if beta == 0:
        return math.log(2.0), 0.0, 0.0
    delta = (policy_w - policy_l) - (ref_w - ref_l)
    if not math.isfinite(delta):
        raise ValueError("Delta exceeds this scalar example's numeric range")
    margin = beta * delta
    if not math.isfinite(margin):
        raise ValueError("Margin exceeds this scalar example's numeric range")
    loss = max(0.0, -margin) + math.log1p(math.exp(-abs(margin)))
    derivative = -beta * sigmoid(-margin)
    return loss, derivative, margin


ref_w = ref_l = math.log(0.4)
old = dpo_pair(ref_w, ref_l, ref_w, ref_l, beta=1.0)
new = dpo_pair(math.log(0.3), math.log(0.1), ref_w, ref_l, beta=1.0)
print(round(old[0], 6), round(new[0], 6))  # 0.693147 0.287682
print(round(new[1], 6))                   # -0.25
assert math.isclose(old[0], math.log(2))
assert new[0] < old[0]
```

稳定写法避免了直接计算 `exp(-margin)` 时常见的溢出；若输入虽有限，但 log-ratio 相减或乘 beta 已超出浮点范围，这个标量示例会明确报错，而不是把非有限值当作有效结果。

正则推导中的 $$\beta$$ 与 pair loss 里的 $$\beta$$ 有关联，但实际训练效果不能简化成“越大，每一步就越保守”。在固定奖励的精确最优分布里，增大 $$\beta$$ 会让重加权更温和；在 DPO 的单样本梯度里，权重却是 $$\beta\sigma(-\beta\Delta)$$。

例如固定 $$\Delta=0$$，$$\beta=0.1$$ 与 1 的权重分别为 0.05 和 0.5；固定 $$\Delta=5$$，对应权重又变成约 0.037754 和 0.006693。它随 beta 的变化不是对所有 margin 都同一个方向。数据、学习率、参数化与训练步数还会共同影响最终 KL。

特别地，固定有限 log-ratios 时，$$\beta=0$$ 使损失恒为 $$\log2$$，参数梯度为 0。它不会退化成 SFT；Reference-free 的其他设置，也不是把这个系数设成零就得到的同一个算法。

## 一份回答的 log probability 来自哪些 token

回答 $$y=(y_1,\ldots,y_T)$$ 的概率遵循自回归链式分解：

$$
\log\pi_\theta(y\mid x)
=\sum_{t=1}^{T}\log P_\theta(y_t\mid x,y_{<t})
$$

这里采用自然对数。每一项都对给定 prompt 和已经出现的回答前缀求条件概率，并不是假设各 token 独立。

训练时可以采用 teacher forcing：将已知回答输入模型，在相应位置读取“下一个目标 token”的概率。一份完整回答不需要为了计算这个数再次逐 token 采样生成。

看一条具体序列，假设本例将 EOS 纳入完整回答事件：

```text
输入位置        0     1    2    3    4    5    6
input IDs      BOS   p0   p1   y0   y1  EOS  PAD
loss mask       0     0    0    1    1    1    0

logits 位置 2 预测 y0
logits 位置 3 预测 y1
logits 位置 4 预测 EOS
```

对应常见 labels 表示，可以将不计入回答分数的位置设为 `-100`，再让 logits 与下一位置的 labels 对齐。Prompt 不计入这个 log probability，仍然是 Attention 可见的条件；不能因“不计 loss”就把它从上下文遮掉。

下面直接给出每个 logits 位置对下一个实际 token 的概率，省略真实模型前向与词表维度。EOS 和 PAD 故意采用同一 ID，依靠位置 mask 区别：

<!-- dpo-example: sequence-logprob -->
```python
import math

ids = [0, 11, 12, 20, 21, 2, 2]  # BOS, p0, p1, y0, y1, EOS, PAD
loss_mask = [False, False, False, True, True, True, False]
next_token_probs = [0.9, 0.7, 0.5, 0.25, 0.8, 0.6]

target_positions = [i + 1 for i in range(len(ids) - 1) if loss_mask[i + 1]]
logp = math.fsum(math.log(next_token_probs[pos - 1]) for pos in target_positions)

print(target_positions)  # [3, 4, 5]
print(round(logp, 6))    # -2.302585
print(round(logp / len(target_positions), 6))  # -0.767528
assert math.isclose(logp, math.log(0.5 * 0.25 * 0.8))
```

三个有效条件概率的乘积是 0.1，完整回答 log probability 为 $$\log0.1\approx-2.302585$$。除以 token 数得到的 -0.767528，是几何平均 token 概率的对数，已不是完整回答概率的对数。

原始 DPO 使用有效 completion token 的 log probability 总和。改成平均值，会改变 pair margin 与训练目标，应作为另一种明确的归约选择，而不是无声地当成数值优化。[作者 `_get_batch_logps`](https://github.com/eric-mitchell/direct-preference-optimization/blob/main/trainers.py)与 [TRL v0.23.1 的默认归约](https://github.com/huggingface/trl/blob/v0.23.1/trl/trainer/dpo_trainer.py)可用来对照这一点。

是否包含 EOS 由数据合同确定；PAD 应按实际 padding 位置排除。若 EOS 与 PAD 共用 ID，直接删掉所有相同 ID，会把真实终止 token 也删掉，进而改变我们正在比较的回答事件。

## 训练 batch 中需要四个分数，但只更新 policy

同一条 pair 的计算关系是：

```text
prompt + chosen  ─┬─ policy    → log πθ(yw|x)   保留梯度
                 └─ Reference → log πref(yw|x) 固定

prompt + rejected ┬─ policy    → log πθ(yl|x)   保留梯度
                 └─ Reference → log πref(yl|x) 固定

四个 log probabilities → 相对 margin → pair loss → 更新 policy
```

实现可以将 chosen 和 rejected 拼成一个 2B 的 batch 一起 forward，而不是启动两次独立调用；Reference 也能对同样的序列批处理。固定 Reference 的计算不保留训练梯度，但仍需要读取权重、执行前向并计算相应分数。

这条固定数据训练路径省掉了每个优化步骤都要执行的在线 rollout，也省掉独立 reward model 的拟合环节。它并没有让偏好采集和生成评估免费：两份候选要从某处获得，训练后的真实行为还要通过生成检查。

完整 RLHF/GRPO 的模型与数据流可对照 [后训练系统文章]({% post_url 2026-08-18-rlhf-grpo-system-dataflow %})。DPO 的工程区别是把当前这一步变成了已知序列上的概率计算和反向传播，而不是把所有后训练任务都变成同一种算法。

冻结的 Reference 还可以预计算 log probabilities，减少后续重复前向。[TRL v0.23.1 文档](https://huggingface.co/docs/trl/v0.23.1/en/dpo_trainer)提供了相关配置；这是该固定版本的能力说明，不是最新接口承诺。

这些缓存必须绑定 Reference 权重、Tokenizer 与模板、实际 token IDs、attention/position 条件、completion mask、截断和数据版本。Reference 评分还应关闭随机 dropout 等训练态行为，避免同一样本的基准随调用改变。文字记录看起来没变，但 EOS 或截断规则变了，旧分数就不再描述当前训练样本。采用 Adapter 共享模型存储时，也必须确认 Reference 路径实际使用的是选定的基准权重。

## 原始 prompt 相同，还不够保证比较条件相同

最容易隐藏错误的部分，往往发生在 pair loss 之前。

第一种错误是输入边界漂移。我们在 [Tokenizer 与 BPE]({% post_url 2026-10-13-tokenizer-bpe-model-input %})中看过：整段文字编码不一定等于分段编码后拼接。训练究竟先编码完整 prompt+answer，还是分别编码后拼 IDs，需要读取具体实现。

例如 [TRL v0.23.1 的 `tokenize_row`](https://github.com/huggingface/trl/blob/v0.23.1/trl/trainer/dpo_trainer.py)分别编码三个文本字段，再由后续路径拼接 IDs。它采用了明确的分段合同，不能因为数据字段叫 prompt，就自动认为最终序列必然等于部署时整段重新编码的结果。尤其要检查 assistant 开始标记、空格与特殊 token 是否一致。

第二种错误是截断改变了标签的对象。假设 chosen 与 rejected 唯一不同之处在最后一段，而上限裁掉了那一段，两份有效回答已经相同。这时，在同一确定性条件与评分规则下，$$\Delta=0$$、损失为 $$\log2$$，pair 的参数梯度相互抵消。原文字段仍不同，训练却没有留下任何排序信息。

反过来，若为保留回答末尾而从整段左侧裁剪，长回答与短回答可能留下不同长度的 prompt。两份回答看到了不同问题上下文，“同一个 x”的前提也随之改变。检查应落到最终 token 条件和有效 completion，而不能只对原 JSON 的 prompt 做字符串比较。

第三种错误是 mask 与 shift 不一致。首个 completion token 的概率来自最后一个 prompt 位置的 logits；shift 少一位，就可能把 prompt token 算进回答，或者漏掉首个回答 token。Padding 与 EOS 相同、跨样本 packing、不同有效长度，会进一步使这种错位更难从张量 shape 看出来。

因此，先抽取一小批 pair，逐项展示最终 IDs、位置、可见性、shifted targets 与累计 log probabilities，比只观察一个不断下降的训练 loss 更有信息。

## 评估应该检查学到了哪种偏好

一个可解释的训练集，需要在同一问题和同一标准下比较两份回答。长度、格式与措辞可能成为捷径：如果 chosen 总是更长，模型可能学到多写；如果标签来自另一个模型，该模型的系统性误判也会进入训练。

划分评估数据时，应防止同一问题及其近似改写同时进入训练和测试。只把同一 prompt 的另一对候选留到测试，仍可能高估对新问题的泛化。技术问答可以额外保留概念反例，代码任务则可以保留可执行测试结果，帮助检查偏好标签是否服务于实际目标。

训练日志中的隐式奖励通常写为：

$$
\widehat r_\theta(x,y)=\beta\left[\log\pi_\theta(y\mid x)-\log\pi_{\mathrm{ref}}(y\mid x)\right]
$$

其 margin 是两份隐式奖励之差；相应 reward accuracy 统计 chosen 的隐式奖励是否高于 rejected，见 [TRL v0.23.1 的日志定义](https://huggingface.co/docs/trl/v0.23.1/en/dpo_trainer#logged-metrics)。这检查的是模型是否学会数据中的排序，不是“生成内容事实正确率”。即使这个指标变好，也需要看 held-out pair 和真实生成任务。

实际评估可以同时检查任务正确性、偏好胜率、生成长度分布、重复与异常输出，以及基础能力是否退化。修改 beta 或训练轮数时，比较相同评估条件下的结果；只比较训练 loss，容易把更强地拟合标签误当成更好的模型。

离线偏好还限制了模型获得的直接反馈：未覆盖的任务、数据中没有出现的行为，不会因为这个损失形式漂亮，就自动得到可靠指导。模型可以通过共享表示泛化，但这种泛化需要验证。

DPO 把成对比较转成了一份可以直接计算的生成概率目标。理解奖励差为什么能消掉归一化项，能说明算法从何而来；理解最终 token、条件与 mask 怎样产生那四个分数，则能说明一段训练代码究竟在优化什么。两层都正确，训练曲线才有清楚的含义。
