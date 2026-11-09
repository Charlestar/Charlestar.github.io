---
layout: post
title: "融合 LM Head 与交叉熵：不保存整张 Logits 怎样仍算对损失和梯度"
subtitle: "从词表投影的张量寿命出发，推导稳定归一化、分块梯度与提前反向计算的接口边界"
date: 2026-11-10
last_modified_at: 2026-11-10
author: iStar
catalog: true
series: gpu-runtime-precision
series_order: 60
technology_year: 2024
mathjax: true
tags: [GPU优化, 分布式训练]
---

Transformer 主体输出的 hidden states 只有 hidden size 那么宽，最后的 LM head 却要为每个位置生成整个词表的分数。词表十几万、参与训练的位置有几千个时，这次投影会突然形成一张远大于 hidden states 的 logits 张量。

采用 mean 或 sum 归约时，交叉熵最终只返回一个标量，但朴素训练路径可能把整张 logits 以及相关中间结果保留到 backward。融合 LM head 与交叉熵的关键，是让投影结果尽快被损失和梯度计算消费，不必让全部位置的词表分数同时长期存在。

它不意味着只计算目标 token 的分数：完整 softmax 的分母仍然依赖全词表。下面先推导这个依赖，再解释 [Liger 的 Fused Linear Cross Entropy](https://arxiv.org/html/2410.10989v1#S3)怎样沿 token 维分块，以及词表 tile 路线又改变了哪一层存储。

## 从 Hidden States 到词表，形状怎样放大

设已经完成 next-token 标签对齐，共有 $$R$$ 行训练位置，hidden size 为 $$H$$，词表大小为 $$V$$。采用 PyTorch Linear 的权重存储约定：

$$
X\in\mathbb R^{R\times H},\qquad
W\in\mathbb R^{V\times H},\qquad
Z=XW^T\in\mathbb R^{R\times V}
$$

这里先省略 bias。$$W$$ 的第 $$j$$ 行对应词表中第 $$j$$ 类，$$z_{ij}$$ 是位置 $$i$$ 对该类的 logit。它是未归一化分数，不是概率。

取 $$R=8192,H=4096,V=131072$$，并假设这些张量采用 BF16。一份 $$X$$ 是 64 MiB，权重 $$W$$ 是 1 GiB，而完整 $$Z$$ 达到 2 GiB。如果再物化一份 FP32 logits，则该副本有 4 GiB。模型主体已经能够运行，不代表训练末端的这一峰值也一定能放下。

要解决的问题因此很具体：损失与反向是否必须等到完整 $$R\times V$$ 矩阵全部生成，才能开始工作？如果各行的损失可以独立计算，就有机会边投影、边消费，只保留一个较小的行块。

## 一行交叉熵需要什么信息

先讨论 hard-label、无 class weights、无 label smoothing 的交叉熵。设目标类为 $$y_i$$，则：

$$
p_{ij}=\frac{e^{z_{ij}}}{\sum_{k=0}^{V-1}e^{z_{ik}}},
\qquad
\ell_i=-\log p_{i,y_i}
$$

直接计算指数可能溢出。取 $$m_i=\max_jz_{ij}$$、$$s_i=\sum_j e^{z_{ij}-m_i}$$，得到稳定形式：

$$
\ell_i=(m_i-z_{i,y_i})+\log s_i,
\qquad
p_{ij}=\frac{e^{z_{ij}-m_i}}{s_i}
$$

目标 logit 只贡献一项，但 $$s_i$$ 仍然遍历全部词表。融合只改变何时生成、保存和消费这些分数，不会把这个稠密目标变成 sampled softmax。

最大值和指数和也可以在线合并。若一段已有统计为 $$(m,s)$$，另一段为 $$(\widetilde m,\widetilde s)$$，则：

$$
m'=\max(m,\widetilde m),\qquad
s'=s e^{m-m'}+\widetilde s e^{\widetilde m-m'}
$$

两部分先换到共同最大值，再相加，便能得到同一归一化量。这是 [Online Normalizer](https://arxiv.org/html/1805.02867v2#S3)可以按词表片段累计的原因。它为更细的 tile 实现提供了工具，但“只保留统计”与“反向不再需要分数”仍是两件事。

实现时保留 $$(m,s)$$ 还有一个数值细节。若 logits 全为 $$10^{16}$$，正确损失为 $$\log V$$；先把 $$m+\log s$$ 舍入成一个浮点数，再减同样巨大的目标 logit，可能丢失低位。直接计算 $$(m-z_y)+\log s$$ 能避免这次不必要的相消，但不能恢复输入 logits 在更早计算中已经丢掉的差异。

## 梯度说明了哪些大张量可以及时消费

设有效标签行集合为 $$\mathcal I$$，$$N_{valid}=\lvert\mathcal I\rvert>0$$，训练损失采用有效行均值：

$$
\mathcal L=\frac1{N_{valid}}\sum_{i\in\mathcal I}\ell_i
$$

对每行 logits 求导。因为 log-sum-exp 的导数是 softmax，目标 logit 的导数是对应位置的 1，得到：

$$
D_{ij}=\frac{\partial\mathcal L}{\partial z_{ij}}
=\frac{\mathbf1[i\in\mathcal I]}{N_{valid}}
\left(p_{ij}-\mathbf1[j=y_i]\right)
$$

这是最小化负 log-likelihood 的符号：概率减 one-hot。再沿线性层反传：

$$
\frac{\partial\mathcal L}{\partial X}=DW,
\qquad
\frac{\partial\mathcal L}{\partial W}=D^TX
$$

若有 bias，其梯度再对 $$D$$ 的行求和。由此可以看到，一块 logits 在变成 $$D$$ 后，就能立刻与 $$W$$、对应的 $$X$$ 块相乘，贡献输入和权重梯度；不必为了等待其他 token 的 logits 而继续保留它。

被忽略的行在 head 这条路径中 $$D_i=0$$，但不意味着对应 token 从模型上下文中消失。它仍可能通过 attention 为其他位置提供信息，并通过那些计算路径收到梯度。Loss mask 与上下文可见性不能互相代替。

## 按 Token 分块，改变的是张量寿命

设每块最多包含 $$C$$ 行，对第 $$c$$ 块执行：

$$
Z_c=X_cW^T
\longrightarrow
(\mathcal L_c,D_c)
\longrightarrow
\left(D_cW,D_c^TX_c\right)
$$

输入梯度写入该块对应的行，权重梯度则跨块相加：

$$
\nabla_W\mathcal L=\sum_c D_c^TX_c
$$

```text
当前 X 块
  → 生成 C×V logits
  → 归一化与交叉熵
  → 得到当前 D 块
  → 写对应 dX，累加 dW
  → 释放当前词表中间量，进入下一块
```

关键不是 Python 循环本身，而是循环结束后哪些张量仍然存活。先生成完整 $$Z$$ 再对它分块求 loss，完整 logits 并没有消失；即使逐块生成 logits，如果普通 autograd 图保留了每块反向所需的大张量，再把所有 loss 相加后才 backward，峰值也不一定按预期下降。

[Liger v0.5.2 的 FLCE 实现](https://github.com/linkedin/Liger-Kernel/blob/v0.5.2/src/liger_kernel/ops/fused_linear_cross_entropy.py)在每块中计算局部梯度，再保留最终输入、权重等梯度供 backward 使用。这里的融合是扩大算子之间可共同规划的边界，并不表示整段训练末端只启动一个 GPU kernel：分块投影、交叉熵和梯度矩阵乘仍是不同工作。

## 提前计算局部梯度，不等于上游梯度永远是 1

假设上层还构造了 $$\mathcal J=f(\mathcal L)$$，那么真正传回来的标量是 $$a=\partial\mathcal J/\partial\mathcal L$$。链式法则要求：

$$
\frac{\partial\mathcal J}{\partial X}=a\nabla_X\mathcal L,
\qquad
\frac{\partial\mathcal J}{\partial W}=a\nabla_W\mathcal L
$$

因此前向中可以预先计算局部导数，backward 再乘实际的 $$a$$。Loss weighting、梯度累积归一化或 loss scaling 都可能使它不等于 1。省略这次乘法，普通 `loss.backward()` 或许看不出问题，换一个训练目标组合就会出错。

在固定 v0.5.2 实现中，保存的是 detach 后的局部梯度，上游标量不为 1 时还会原地缩放保存区。这种设计应按普通一阶、标量 loss 的单次 backward 理解，不能直接承诺高阶求导或任意重复 backward。它也没有授权提前更新权重：整个反向、梯度合并和优化器 step 的训练顺序仍需保持。

## 每块平均以后，为什么还要重新加权

全局均值的分母是 $$N_{valid}$$，不是块数，也不一定是总行数。若第 $$c$$ 块有 $$n_c$$ 个有效标签，并先算出局部平均 $$\overline\ell_c$$，正确组合是：

$$
\mathcal L=\sum_{c:n_c>0}\frac{n_c}{N_{valid}}\overline\ell_c
$$

例如一块有两个有效位置、平均 loss 为 1，另一块只有一个有效位置、loss 为 4，全局平均为 2，而不是两个块均值的平均 2.5。尾块较短、prompt labels 被忽略或整块没有有效标签，都会使“每块权重相同”失效。

Liger 该版本使用有效标签数的比例修正块内 mean；下面的 CPU reference 则直接把每个有效行除以全局计数，表达同一数学目标。空有效块应贡献零，不能先计算一个 $$0/0$$ 再指望后面乘零修复。

`ignore_index` 的含义也要准确：它忽略对应目标行，不会把这个索引代表的词表类别从其他行 softmax 分母中删除。无 class weights 的 hard-label mean 按有效 target 计数；其他选项应另外遵守 [CrossEntropyLoss 的归约定义](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/loss.py)。整个输入没有有效 target 时，均值没有定义，训练循环需要明确跳过或拒绝这次更新。

Next-token 对齐发生在这些计算之前。应先在样本内部让当前位置 hidden state 对应下一个 target，再 flatten；不能先把多个样本压成一条长数组，再整体 shift 产生跨样本监督。对于 packed sequence，还要处理段边界，具体见[序列打包与有效监督]({% post_url 2026-09-13-sequence-packing-variable-length-attention %})。

## 显存与计算量需要分别记账

回到 $$R=8192,H=4096,V=131072$$ 的 BF16 例子。若取 $$C=256$$，单份块 logits 是 64 MiB，而不是完整矩阵的 2 GiB。但其他必要对象仍然存在：

| 对象 | 形状 | 本例大小 |
| --- | --- | ---: |
| 完整 logits | $$R\times V$$ | 2 GiB |
| 单份块 logits | $$C\times V$$ | 64 MiB |
| 输入及同 dtype 输入梯度，各一份 | $$R\times H$$ | 64 MiB |
| 权重及同 dtype 权重梯度，各一份 | $$V\times H$$ | 1 GiB |

若权重梯度用 FP32，单份变成 2 GiB。模型主体保存的 activation、临时缩放块、库 workspace 与 allocator 还会影响峰值。因此“2 GiB 变成 64 MiB”只比较一类中间张量，不是整个训练显存减少 32 倍。

固定版本的 Liger 还会在原地计算块内 logits 梯度之后，生成一份缩放后的块用于输入梯度计算；64 MiB 不能直接当作该实现的精确临时峰值。其 loss 统计使用 FP32，也不意味着所有梯度 buffer 都采用 FP32，实际梯度创建和保存仍与输入、权重 dtype 有关。

假设 $$X,W$$ 都需要梯度，主要 GEMM 运算为前向投影、输入梯度和权重梯度各一次：

$$
F_{GEMM}\approx2RHV+2RHV+2RHV=6RHV
$$

分块不消灭这些全词表乘加。块变小可以缩短 logits 生命周期，却也可能降低 GEMM 形状效率、增加启动和权重重复访问；块太大则削弱容量收益。应在目标 shape、dtype 与训练配置下观察折中，而不是认为分得越细越快。

## 一个可以逐项检查的分块实现

下面以标准库实现上面的数学合同：标签已经对齐，权重不带 bias，输出为标量 `mean` 或 `sum`。输入矩阵为 Python 列表，其中数值须能转为有限双精度浮点数；即便输入有限，乘加或输出仍可能溢出，代码会拒绝非有限 logits、损失和梯度。这个 reference 跳过被忽略行的 head 投影；全忽略时 `mean` 显式拒绝，`sum` 返回零及零梯度。代码保留 $$(m,s)$$，并直接计算局部梯度，不构建自动微分图。

<!-- example: flce-reference -->
```python
import math


def checked_real(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("real number required")
    try:
        value = float(value)
    except OverflowError as error:
        raise ValueError("number exceeds float range") from error
    if not math.isfinite(value):
        raise ValueError("finite number required")
    return value


def finite_sum(values):
    try:
        return checked_real(math.fsum(values))
    except (OverflowError, ValueError) as error:
        raise ValueError("non-finite sum") from error


def online_stats(values):
    m, s = -math.inf, 0.0
    for value in values:
        new_m = max(m, value)
        s = s * math.exp(m - new_m) + math.exp(value - new_m)
        m = new_m
    return m, s


def linear_ce_chunked(x, weight, targets, chunk_size=2,
                      reduction="mean", ignore_index=-100):
    if (isinstance(chunk_size, bool) or not isinstance(chunk_size, int)
            or chunk_size <= 0 or reduction not in ("mean", "sum")):
        raise ValueError("invalid chunk or reduction")
    if not weight or not weight[0]:
        raise ValueError("nonempty weight required")
    vocab, hidden = len(weight), len(weight[0])
    if (len(x) != len(targets)
            or any(len(row) != hidden for row in weight + x)):
        raise ValueError("shape mismatch")
    for row in weight + x:
        for value in row:
            checked_real(value)
    if isinstance(ignore_index, bool) or not isinstance(ignore_index, int):
        raise ValueError("integer ignore_index required")
    if any(isinstance(y, bool) or not isinstance(y, int)
           or (y != ignore_index and not 0 <= y < vocab) for y in targets):
        raise ValueError("invalid target")
    count = sum(y != ignore_index for y in targets)
    if reduction == "mean" and count == 0:
        raise ValueError("mean needs an effective target")
    denominator = count if reduction == "mean" else 1
    dx = [[0.0] * hidden for _ in x]
    dw = [[0.0] * hidden for _ in weight]
    losses = []
    for start in range(0, len(x), chunk_size):
        stop = min(start + chunk_size, len(x))
        logits = [[finite_sum(a*b for a, b in zip(x[i], w))
                   for w in weight] if targets[i] != ignore_index else None
                  for i in range(start, stop)]
        for offset, scores in enumerate(logits):
            i, target = start + offset, targets[start + offset]
            if target == ignore_index:
                continue
            m, s = online_stats(scores)
            losses.append(checked_real(
                ((m - scores[target]) + math.log(s)) / denominator))
            dz = [(math.exp(z-m)/s - (j == target)) / denominator
                  for j, z in enumerate(scores)]
            dx[i] = [finite_sum(dz[j]*weight[j][h] for j in range(vocab))
                     for h in range(hidden)]
            for j in range(vocab):
                for h in range(hidden):
                    dw[j][h] = checked_real(dw[j][h] + dz[j] * x[i][h])
            del dz
        del scores, logits
    return finite_sum(losses), dx, dw


def scale_saved_gradients(dx, dw, upstream):
    upstream = checked_real(upstream)
    return ([[checked_real(upstream*checked_real(v)) for v in row] for row in dx],
            [[checked_real(upstream*checked_real(v)) for v in row] for row in dw])
```

用三行输入检查目标行、忽略行和尾块：

<!-- example: flce-small-example -->
```python
x = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
weight = [[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]]
targets = [0, -100, 1]
loss, dx, dw = linear_ce_chunked(x, weight, targets, chunk_size=2)
assert math.isclose(loss, 0.5626709415555414)
assert dx[1] == [0.0, 0.0]
scaled_dx, scaled_dw = scale_saved_gradients(dx, dw, 0.3)
assert math.isclose(scaled_dx[0][0], 0.3 * dx[0][0])
print(loss, dx, dw)
```

第一行损失约为 0.407606，第三行约为 0.717736，平均值约为 0.562671。改变块大小不应改变数学目标；对于同一组输入，可以进一步用中心差分检查 $$X,W$$ 的每个梯度，并覆盖 `sum`、不均匀有效计数、空有效块和非 1 的上游标量。这种小型对照把归一化、梯度方向与块间累加分开暴露，便于定位错误。

仓库中的 `node scripts/reviews/fused-linear-ce-examples.mjs` 会提取并执行这两段代码，对照独立的完整矩阵计算和有限差分。Python 列表、行级临时量与高精度求和并非 GPU 缓冲区实现；该检查验证有限算例的数学与边界，不测量显存、吞吐或实际 Triton/CUDA 行为。

## 沿词表切 Tile 与沿 Token 分块有什么不同

上述路线仍然会生成一份 $$C\times V$$ 的块 logits，只是同时存活的 token 行数更少。另一条路线把 token 与 vocabulary 都进一步切成片上 tile，边做投影边更新每行归一化统计，不把这些词表 tile 全部写回 HBM。

[Cut Cross-Entropy](https://arxiv.org/html/2411.09009v2#S4)采用了这种设计方向：保存每行所需统计，在 backward 重算相关 tile，再形成梯度。它避免的是更细粒度的 logits 物化，不是跳过全词表 normalizer。Forward 保存多少、backward 重算多少、梯度怎样累加，需要作为同一条数据流设计。

CCE 中的梯度过滤还属于额外数值策略，而不是原始 dense CE 在实数上的恒等变换。其后续[预训练讨论](https://arxiv.org/html/2411.09009v2#S5)涉及过滤与低精度累加对质量的影响，所以不能把早期特定微调结果推广成任意训练中的无损保证。比较 token chunk 与 vocabulary tile 时，应先对齐损失、梯度和精度合同，再讨论峰值与吞吐。

## 接入模型时，必须保持哪些接口

最直接的适用场景是：调用者只需要标量训练损失，以及这条损失对 hidden states 和 head 权重的梯度。如果其他消费者仍需要完整 logits，例如某些蒸馏分布、KL 目标或自定义统计，直接替换接口就会丢失它们所需的数据。应一起重设计消费者，而不是让融合函数悄悄改变训练目标。

Tied embedding 也需要特别处理。LM head 与输入 embedding 共用参数时，提前算出的 $$\nabla_W\mathcal L$$ 只是 head 分支贡献，还要与 embedding 分支及其他共享路径的梯度相加；它不是可以立即覆盖最终 `.grad` 的完整答案。

词表并行则引入另一条边界：每个 rank 只拥有部分词表时，归一化需要全局最大值与指数和等统计，不能各自对本地词表独立 softmax。具体 collective 推导见 [Tensor Parallel 的词表交叉熵]({% post_url 2026-08-08-megatron-tensor-parallel-transformer %})；本地省掉 logits 物化不会自动解决跨 rank 的全局归一化。

最后，数学上可实现的合同不等于某个库版本支持所有变体。这里的标量 mean/sum、普通一阶求导，与 `reduction="none"`、高阶梯度、重复 backward、class weights 或额外 loss 项是不同接口。部署时应逐项核对所选版本，并比较完整训练 step 的 loss、输入梯度、共享权重梯度与一次参数更新，而不是只看前向返回的标量。

融合 LM head 与交叉熵真正利用的，是损失结构与张量生命周期之间的关系：每个词表分数需要参与计算，但不必让所有分数同时长期保存。只要全局分母、局部导数和上游链式法则保持一致，就可以把一张巨大的中间矩阵改成按块产生、立即消费的数据流。

## 参考资料

- [Liger Kernel: Efficient Triton Kernels for LLM Training](https://arxiv.org/html/2410.10989v1)
- [Liger v0.5.2 Fused Linear Cross Entropy](https://github.com/linkedin/Liger-Kernel/blob/v0.5.2/src/liger_kernel/ops/fused_linear_cross_entropy.py)
- [Liger v0.5.2 Cross Entropy Kernel](https://github.com/linkedin/Liger-Kernel/blob/v0.5.2/src/liger_kernel/ops/cross_entropy.py)
- [Online Normalizer Calculation for Softmax](https://arxiv.org/html/1805.02867v2)
- [PyTorch v2.8.0 CrossEntropyLoss](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/loss.py)
- [Cut Your Losses in Large-Vocabulary Language Models v2](https://arxiv.org/html/2411.09009v2)
