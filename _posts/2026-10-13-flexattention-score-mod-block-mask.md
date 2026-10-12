---
layout: post
title: "FlexAttention：Attention 规则怎样变成可跳过的 GPU 工作"
subtitle: "沿着 score_mod、mask_mod 与 BlockMask，理解规则组合、编译、块稀疏和 Decode 的位置边界"
date: 2026-10-13
last_modified_at: 2026-10-13
author: iStar
catalog: true
series: attention-long-context
series_order: 65
technology_year: 2024
mathjax: true
tags: [注意力机制, GPU优化, 工程实践]
---

给 Attention 加一条规则通常很容易：只看过去、不跨文档、限制最近若干个 token，或者给远处的位置加一个偏置。把这些规则同时放进高效 kernel，却没有那么直接。表达式只有几行，执行时仍可能生成一张庞大的分数矩阵，或者把本来不可见的位置全部计算一遍。

FlexAttention 将这个问题分成两部分：程序描述每个连接的分数与可见性，编译器把规则融入 Attention kernel，再利用块级信息跳过整块不可见的工作。PyTorch 在 [2024 年的发布文章](https://pytorch.org/blog/flexattention/)中介绍了这个接口，后续的 [MLSys 2025 论文](https://proceedings.mlsys.org/paper_files/paper/2025/file/61a9278dfef5f871b5e472389f8d6fa1-Paper-Conference.pdf)给出了编程模型与实现。

我们从一个连接的数学含义出发，走到块级执行、样本打包和 KV Cache。本篇接口示例固定为 PyTorch 2.8.0，便于核对行为；它不是“当前最新版”的声明。标准库代码用于复算数学与索引，CUDA 示例在本文环境未执行，不构成 GPU 性能测试。

## 先明确一个连接经过了哪些操作

设一个 head 的 Query 和 Key 维度为 $$d$$，query 位置为 $$i$$、key 位置为 $$j$$。使用默认缩放时，原始分数为：

$$
s_{ij}=\frac{q_i^Tk_j}{\sqrt d}
$$

随后定义两类规则：分数变换 $$g$$，以及可见性 $$M_{ij}\in\{\mathrm{True},\mathrm{False}\}$$。最终进入 softmax 的分数是：

$$
\widetilde s_{ij}=
\begin{cases}
g(s_{ij},b,h,i,j),&M_{ij}=\mathrm{True}\\
-\infty,&M_{ij}=\mathrm{False}
\end{cases}
$$

其中 $$b,h$$ 分别标识 batch 和 head。先假设可见连接经变换后得到有限分数；对于至少有一个可见 key 的 query，输出为：

$$
o_i=\sum_{j:M_{ij}}\frac{\exp(g(s_{ij},b,h,i,j))}{\sum_{k:M_{ik}}\exp(g(s_{ik},b,h,i,k))}v_j
$$

这份表达式是语义定义，不要求实现将整个矩阵存到显存。它也明确了顺序：先处理已经缩放的 QK 分数，再按可见性排除连接，最后归一化与汇总 Value。

在 [PyTorch 2.8 API](https://docs.pytorch.org/docs/2.8/nn.attention.flex_attention.html)里，`score_mod(score, b, h, q_idx, kv_idx)` 负责分数变换，`mask_mod(b, h, q_idx, kv_idx)` 返回连接是否允许。`True` 表示允许参与，不能套用某些其他接口中 `True` 表示屏蔽的约定。

`score` 已包含所设定的缩放；如果又在 `score_mod` 中除以一次 `sqrt(d)`，就改变了 Attention 的温度。想改缩放或偏置可以显式这样做，但必须与希望实现的模型规则一致。

## score_mod 改变分数，操作次序也改变结果

先自行定义一个教学组合：对分数做 soft-capping，再加入距离偏置。

$$
g(s,b,h,i,j)=c\tanh(s/c)+\alpha_h(p_j-p_i),\qquad c>0
$$

$$p_i,p_j$$ 是逻辑位置，$$\alpha_h$$ 是每个 head 的系数。第一项把未加偏置的分数限制在 $$(-c,c)$$ 内；第二项在 causal 场景中可以对较远的历史位置施加负偏置。这是我们定义的组合，不是声称某个已有模型恰好采用它。

如果先加偏置再 soft-cap，变成 $$c\tanh((s+\alpha_h(p_j-p_i))/c)$$，得到的是另一套规则。非线性操作通常不能任意交换顺序，所谓“把几个 Attention 技巧组合起来”必须先明确组合后的数学式。

硬 mask 也有同样的问题。把非法连接的分数设为 0，仍会让它带着 $$\exp(0)=1$$ 进入 softmax。设为 $$-\infty$$ 可以排除它，但若之后对所有位置再执行 soft-cap：

$$
c\tanh(-\infty/c)=-c
$$

非法位置重新变成了有限分数，获得非零概率。下面用两个连接直接观察这个错误：

<!-- flex-example: softcap-order -->
```python
import math


def softmax(values):
    maximum = max(values)
    if maximum == -math.inf:
        raise ValueError("This reference requires at least one visible key")
    weights = [math.exp(value - maximum) for value in values]
    total = sum(weights)
    return [weight / total for weight in weights]


cap = 5.0
scores = [0.0, 0.0]
allow = [True, False]

correct = [cap * math.tanh(s / cap) if ok else -math.inf
           for s, ok in zip(scores, allow)]
masked_first = [s if ok else -math.inf for s, ok in zip(scores, allow)]
incorrect = [cap * math.tanh(s / cap) for s in masked_first]

print(softmax(correct))  # [1.0, 0.0]
print([round(p, 6) for p in softmax(incorrect)])  # [0.993307, 0.006693]
assert softmax(incorrect)[1] > 0
```

分数变换与可见性独立定义，既让这个顺序更清楚，也向执行器提供了“哪些连接根本不需要计算”的信息。[v2.8.0 前向 kernel](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_inductor/kernel/flex_attention.py)在相关路径中先执行 `score_mod`，再对 partial block 施加元素 mask。

## mask_mod 定义合法连接，BlockMask 定义块级工作

如果只在 `score_mod` 里写“非法处返回 `-inf`”，可以定义正确的数值结果，却不能据此保证执行器已经跳过那些 QK 计算。要利用结构稀疏性，FlexAttention 从 `mask_mod` 建立 `BlockMask`。

为什么还要一份块级表示？GPU 上的矩阵乘法按 tile 组织，逐元素问“这里是否允许”并不等于可以逐元素省掉一次高效矩阵乘法。只要一个 tile 内仍有有效连接，通常就需要访问这块工作，再处理其中被屏蔽的位置。

把长度为 6 的 causal Attention 按 2×2 教学 tile 分开：

```text
                  KV 块
               [0,1] [2,3] [4,5]
Q 块  [0,1]       P     0     0
      [2,3]       F     P     0
      [4,5]       F     F     P

0：整块不可见    F：整块可见    P：部分可见
```

三种块承担不同工作：

| 块类型 | QK 与 Value 汇总的相关块工作 | 元素可见性检查 | score_mod |
| --- | --- | --- | --- |
| 完全屏蔽 `0` | 可以跳过 | 无需逐元素处理 | 无需处理 |
| 完全可见 `F` | 需要计算 | 可略过该 mask 检查 | 仍然执行 |
| 部分可见 `P` | 需要计算 | 保留元素 mask | 仍然执行 |

“Full block 可以省 mask”不等于“Full block 可以省分数变换”。距离偏置、soft-cap 等操作，在全可见连接上仍有语义作用。三类块的区别见 [论文 §4.2](https://proceedings.mlsys.org/paper_files/paper/2025/file/61a9278dfef5f871b5e472389f8d6fa1-Paper-Conference.pdf)。

这个小例子有 36 个可能连接，其中 21 个满足 causal 条件。块级则跳过 3 块、访问 6 块，对应 24 个 tile 单元格；访问块里仍有 3 个位置需要屏蔽。可见元素比例与访问块比例不同，不能把“15 个元素被屏蔽”直接写成 GPU 时间下降 15/36。

用标准库可以复算分类：

<!-- flex-example: block-counts -->
```python
length, tile = 6, 2
counts = {"F": 0, "P": 0, "0": 0}

for q_start in range(0, length, tile):
    row = []
    for k_start in range(0, length, tile):
        visible = [k <= q
                   for q in range(q_start, q_start + tile)
                   for k in range(k_start, k_start + tile)]
        kind = "F" if all(visible) else ("P" if any(visible) else "0")
        counts[kind] += 1
        row.append(kind)
    print(" ".join(row))

allowed = sum(k <= q for q in range(length) for k in range(length))
visited_cells = (counts["F"] + counts["P"]) * tile * tile
print(counts)              # {'F': 3, 'P': 3, '0': 3}
print(allowed, visited_cells)  # 21 24
```

2×2 是为了看清计数采用的纸面尺度，不是实际 GPU tile 的推荐配置。这段代码没有构造 PyTorch `BlockMask`，也没有模拟 Tensor Core 吞吐。

## 编译器把局部规则接进 Attention 模板

[FlexAttention 论文 §4.1](https://proceedings.mlsys.org/paper_files/paper/2025/file/61a9278dfef5f871b5e472389f8d6fa1-Paper-Conference.pdf)的实现思路是：捕获规则的计算图，将局部变换生成的代码放进经过优化的 Attention 模板。模板处理分块、在线 softmax、数据读取等共同工作，规则描述具体模型的差异。

因此，`score_mod` 看起来像一个标量函数，并不意味着运行时 Python 会对每对 token 调用一次。`torch.compile` 使这些可捕获的张量操作进入编译后的执行路径，避免在 Python 层遍历整张矩阵。

这也解释了接口为什么有表达边界：适合写的是与当前分数和索引有关、能被捕获的运算。任意 Python I/O、依赖 `.item()` 的逐元素控制流，或需要整行概率才能做的后处理，不能仅凭函数签名就变成同一类局部变换。这个接口覆盖许多 scaled dot-product Attention 变体，不是任意新注意力算法都能直接装进来的万能容器。

高效执行路径可以不在显存中保存完整 QK 分数与概率矩阵，继承 [FlashAttention 的分块与在线归一化思路]({% post_url 2026-03-17-FlashAttention原理与实现详解 %})。这里讨论的是 Attention kernel 的中间量；BlockMask 如何构造，还有另一份成本账。

## 一个固定版本的调用示例

下面示范 causal sliding window 与分数变换。我们将窗口 $$W=32$$ 定义为“最多看 32 个位置，包含 query 自身”，所以可见条件是：

$$
0\le p_i-p_j<W
$$

例如 $$W=3,p_i=5$$ 时，只能看位置 3、4、5。若写成 `<= W`，最多会包含 W+1 个位置；这并非语法错误，但采用的是另一种窗口定义。

代码使用 PyTorch 2.8.0 的公开入口，Q/K/V 为相同头数与维度的 CUDA FP16 张量。该版本的接口仍标为 prototype，需要兼容的 PyTorch、编译依赖与 CUDA 环境；本文仅核对接口和语义，未执行这段 GPU 程序。

<!-- flex-example: pytorch-api -->
```python
import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

W, CAP, ALPHA = 32, 5.0, 0.1


def window_mask(b, h, q_idx, kv_idx):
    distance = q_idx - kv_idx
    return (distance >= 0) & (distance < W)


def score_mod(score, b, h, q_idx, kv_idx):
    return CAP * torch.tanh(score / CAP) + ALPHA * (kv_idx - q_idx)


q, k, v = [torch.randn(1, 2, 256, 64, device="cuda", dtype=torch.float16,
                     requires_grad=True) for _ in range(3)]

# Small explanatory input. The default mask-construction path has its own cost.
block_mask = create_block_mask(
    window_mask, B=None, H=None, Q_LEN=256, KV_LEN=256, device="cuda"
)
compiled_attention = torch.compile(flex_attention)
out = compiled_attention(q, k, v, score_mod=score_mod, block_mask=block_mask)
out.sum().backward()
print(out.shape)  # torch.Size([1, 2, 256, 64])
```

mask 与 batch/head 无关，因此 `B=None,H=None` 让构造时的表示沿这两维广播。如果不同样本的边界或不同 head 的窗口不一样，就不能继续假设这个广播条件成立。

这里改变了分数与可见性，得到的是我们定义的新规则。把它替换进一个按其他规则训练的模型，会改变模型行为；API 能正确执行，不证明模型质量保持不变。

## Packing 让物理邻居不再等于上下文邻居

把两条长度为 3 的独立样本连成一条物理序列：

```text
物理位置    0  1  2  3  4  5
样本编号    0  0  0  1  1  1
段内位置    0  1  2  0  1  2
```

如果希望保留独立样本的 causal window 语义，条件应同时满足：

$$
M_{ij}=(\mathrm{segment}_i=\mathrm{segment}_j)
\land(0\le p_i-p_j<W)
$$

只按整条物理序列施加 causal 条件，会允许第二条样本看第一条。只重置位置，却没有样本身份条件，也不会自动删除这些连接；位置编码与可见性承担不同职责。

同理，写了正确的 Attention mask，也不代表 labels 和 loss mask 已经正确处理样本边界。完整训练语义见 [序列打包与变长 Attention]({% post_url 2026-09-13-sequence-packing-variable-length-attention %})。

这种组合适合用 `mask_mod` 描述：读取对应的段编号和逻辑位置，再做布尔运算。每批 packing 边界变化时，所需 BlockMask 也可能变化；若所有层采用同一规则，可以让这一批共享构造结果，避免逐层重复计算。

## Decode 时，q_idx 等于零不代表位置是零

Prefill 的 Q 和 K 同时覆盖一段序列，常使 `q_idx` 与文本位置看起来相同。Decode 却经常只有一个 query：Q 的序列长度为 1，所以它的张量下标始终是 0，而当前 token 的逻辑位置已经是 5、500 或更大。

设当前 query 位于全局位置 5，KV 预分配容量为 8，但只填充位置 0 到 5。使用 $$W=3$$ 时，它应该看 3、4、5。若直接写 `kv_idx <= q_idx`，只会留下位置 0；若只检查容量，也可能读到尚未填充的状态。

下面限定为连续、从位置 0 开始的 KV 布局，复算正确坐标：

<!-- flex-example: decode-coordinates -->
```python
q_idx, q_offset = 0, 5
capacity, valid_length, window = 8, 6, 3
q_position = q_idx + q_offset

visible = [k for k in range(capacity)
           if k < valid_length and 0 <= q_position - k < window]
print(visible)  # [3, 4, 5]
assert visible == [3, 4, 5]
```

若 KV 已经因滑动窗口丢掉旧位置，它的张量下标也不再等于全局位置；若采用分页或循环布局，还需要从存储位置映射到逻辑位置。不能把物理 page index 直接代入距离偏置。PyTorch 的 [FlexAttention for Inference](https://pytorch.org/blog/flexattention-for-inference/)专门讨论了 query offset 与 KV Cache 的适配。

这个区别还会影响 BlockMask：更新被闭包捕获的 offset 张量，只改变规则读到的数值，不会自动改写已经构造好的 skipped/full/partial 元数据。原来被跳过的一块如今变得可见时，旧块结构会漏算。需要重建或按正确的推理映射更新块结构，而不能只看 Python 函数现在返回什么。

## 构造缓存与编译缓存，是两种缓存

`torch.compile` 的结果保存执行图或 kernel；BlockMask 保存某次可见性结构的块信息。编译产物仍可用，不意味着旧 BlockMask 也仍可用。

例如，相同 shape 的两批 token，第一批的样本边界在位置 3，第二批在位置 4。一个依赖段编号的 `mask_mod` 可以继续具有相同计算图，但矩阵的可见连接已经改变，块元数据需要重新检查。缓存键只保存序列长度，会把两份不同结构误认为相同。

相反，固定长度、固定 causal 规则，可以提前构造并复用；只更新某个 head 的偏置数值、未改变可见性结构时，也不必因此重新构造 BlockMask。是否重编译还取决于 shape、dtype、闭包内容、Python 常量和编译器 guards，不能把“某些捕获张量可以更新”扩写成“任何变化都无需重新编译”。

构造阶段还可能有显著峰值内存。在 [v2.8.0 前端源码](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/attention/flex_attention.py)中，默认未编译的 `create_block_mask` 先调用 `create_mask` 生成元素级表示，再归约成块级结构。因此，BlockMask 最终不保存完整元素矩阵，并不代表它从构造到执行都不曾产生 dense 临时 mask。

同一版本已将 `_compile=True` 标为将移除的旧参数，建议用 `torch.compile(create_block_mask)`；首发博客中的旧构造建议不能直接当作这个版本的现状。编译构造或直接利用已知结构生成元数据，是另一个优化问题，需要单独验证正确性与成本。

## 正确性检查应早于性能比较

实现一套组合规则后，可以先用很小的 dense reference 显式算出分数与可见性，检查 forward 结果，再核对相应梯度。Reference 必须遵守同样的缩放、变换次序、位置与 mask；拿新窗口规则的输出和旧 full-attention 输出比较，没有理由期待相同。

至少应覆盖首 token、窗口左右边界、段首、非方形 Q/K、有效 KV 长度与容量不同，以及分数变换后仍有硬屏蔽这些情况。使用较高精度 reference 有助于定位错误，但 fused kernel 改变运算顺序后不必逐位一致，应采用与 dtype 和任务相符的容差。

全屏蔽 query 也需要明确约定。普通数学 softmax 无法对一整行 $$-\infty$$ 进行有效归一化；本篇的有效 query 都至少看见自身，且其变换后分数有限。v2.8.0 相关前向路径有保护处理，但不能因此假定任意手写 reference 或其他后端都会输出同样结果。即使布尔 mask 中存在 `True`，若自定义 `score_mod` 又把所有允许连接变成 `-inf`，仍要按全屏蔽行处理；NaN 或正无穷则需要另外检查。Padding query 应按数据合同处理，并排除相应 loss。

`ROWS_GUARANTEED_SAFE` 之类选项表达的是调用者已经证明“每行至少有一个有效位置”的性质，不是一个可以随手打开的提速按钮。数据发生变化后，该性质仍需成立。

性能测量则至少包含三项：首次编译、mask 构造，以及稳定运行的 kernel。若同一编译产物调用 N 次，同一 BlockMask 复用 K 次，可以用一份简单成本分解理解总收益：

$$
T_{\mathrm{平均}}
\approx\frac{T_{\mathrm{compile}}}{N}
+\frac{T_{\mathrm{mask}}}{K}
+T_{\mathrm{kernel}}+T_{\mathrm{其他}}
$$

这只是成本记账，不是 GPU 延迟预测。短序列或每层都改变结构时，构造与调度成本可能盖过跳块收益；元素很稀疏、却在每个 tile 都留少量连接时，也可能几乎没有整块可以跳过。Prefill 的大矩阵乘法与单 query Decode 的执行形状，还应分别测量。

FlexAttention 让模型规则和执行方式之间多了一条清楚的连接：局部分数变换进入编译模板，可见性进入块结构，完全不可见的块成为可以省掉的工作。规则、坐标和缓存生命周期都明确之后，才能判断一次提速究竟来自少存中间量、少做块计算，还是来自可以摊薄的编译与构造成本。
