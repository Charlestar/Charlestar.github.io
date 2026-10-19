---
layout: post
title: "SnapKV：用观察窗口为长提示词选择 KV Cache"
subtitle: "从完整因果注意力里的投票与池化推导保留索引，再区分有损选择、GQA 存储、RoPE 位置和生成阶段的增长"
date: 2026-10-20
last_modified_at: 2026-10-20
author: iStar
catalog: true
series: kv-cache-memory
series_order: 35
technology_year: 2024
mathjax: true
tags: [KV Cache, 注意力机制, LLM推理]
---

一份很长的提示词进入模型以后，后续每生成一个 token，都可能重新读取大量历史 K/V。假如问题只涉及其中几段信息，能否在开始生成前选出更小的缓存，让后面的 attention 少读一些位置？

SnapKV 选择从提示词末尾寻找线索：观察最后一小段 query 如何关注完整提示词，为此前的位置投票，通过池化让筛选偏向得分较高的邻近信息，再按 head 分别选择位置。它不压缩每个数的 bit width，而是直接删去一部分位置，所以节省缓存的同时，也改变了后续 attention 的输入集合。

[论文](https://arxiv.org/abs/2404.14469)首版发布于 2024-04-22，本文使用 2024-06-17 的 v2 方法描述，并对照[官方实现的固定提交 `e216ddc8`](https://github.com/FasterDecoding/SnapKV/tree/e216ddc84c5bd210378cbdbbba12ba02102aa640)。重点不是背下一组压缩率，而是弄清楚分数从哪里来，选中的索引指向什么，以及这个小缓存怎样接入真实的解码过程。

## 要缩小的是提示词缓存，不是忘记位置的来源

先考虑一个无 padding 的样本和一个 attention head。提示词长度为 $$L$$，末尾 $$W$$ 个 token 构成 observation window，前面的 $$P=L-W$$ 个 token 构成 prefix。压缩后提示词缓存预算为 $$C$$，其中观察窗口全部保留，prefix 只能再留 $$C-W$$ 个位置。

例如 $$L=8,W=2,C=5$$ 时，位置 `0..5` 参与筛选，位置 `6,7` 全部保留。要从六个 prefix 位置中选择三个，而不是从八个位置里任意选五个。

这个策略利用的是一种经验关联：提示词末尾的 query 已经接触到前面的上下文，其 attention 模式可以为随后生成阶段提供线索。论文研究了这种模式的稳定性，但它不是对所有提示词和未来 query 都成立的定理。问题改变、生成轨迹改变，原先得票低的位置仍有可能变得重要。

在多层、多 head 模型里，筛选分别发生在各层和各 head。它们看到不同的表示与注意力分布，因此通常不会得到一份全模型共享的 token 索引表。

## 投票必须来自完整提示词的因果 softmax

令 $$q_t,k_j\in\mathbb R^D$$，观察窗口 query 的位置是 $$t=P,\ldots,L-1$$。先对完整提示词计算 logits，并施加因果约束：

$$
\ell_{tj}=\frac{q_t^\top k_j}{\sqrt D},
\qquad
A_{tj}=\frac{\exp(\ell_{tj})}{\sum_{u=0}^{t}\exp(\ell_{tu})}
\quad\text{for }j\le t
$$

未来位置 $$j>t$$ 的权重为零。注意分母包括当前 query 能看到的全部 prefix 和观察窗口位置，而不是只包括 prefix。

得到完整 attention 概率以后，再截取 prefix 列，并沿观察窗口 query 求和：

$$
s_j=\sum_{t=P}^{L-1}A_{tj},
\qquad 0\le j<P
$$

每个 $$s_j$$ 是 prefix 位置的票数，不是某个 token 的固有重要性。它取决于当前层、当前 head、提示词及观察窗口。一个 query 主要关注自身或附近观察窗口时，分给 prefix 的总票本来就少；如果先删掉观察窗口列再做 softmax，就会把这份较弱的 prefix 关注强行归一化成总和为 1，改变不同 query 之间的投票权重。

一个反例可以直接看到选中对象翻转。设 prefix 只有 A、B 两个位置，后面有两个观察 query。下表给的是 logits 取指数后的正权重，`—` 表示因果 mask：

| query | A | B | 观察位置 2 | 观察位置 3 |
| --- | ---: | ---: | ---: | ---: |
| 位置 2 | 9 | 1 | 90 | — |
| 位置 3 | 1 | 3 | $$2/9$$ | $$2/9$$ |

完整 softmax 后，A 得票 $$0.09+0.225=0.315$$，B 得票 $$0.01+0.675=0.685$$，应选 B。如果只对 prefix 的 A、B 归一化，则变成 A 得票 $$0.9+0.25=1.15$$、B 得票 $$0.1+0.75=0.85$$，反而选 A。变化不是数值误差，而是分母被改成了另一个算法。

[官方 `update_kv`](https://github.com/FasterDecoding/SnapKV/blob/e216ddc84c5bd210378cbdbbba12ba02102aa640/snapkv/monkeypatch/snapkv_utils.py)也是先计算观察 query 对完整 K 的注意力，完成 mask 与 softmax，然后才切出 prefix 累计分数。

## 池化把一个峰值扩展成邻域偏好

如果只按原始票数挑位置，模型可能留下一个片段中得分最高的 token，却丢掉相邻的名字、数字或代码。SnapKV 在 top-k 之前对一维位置分数做 pooling，让邻域共同影响筛选。

以宽度 3、步长 1、两端补零的 average pooling 为例：

$$
\widetilde s_j=\frac{s_{j-1}+s_j+s_{j+1}}3
$$

越界分数按零处理，分母仍是 3。这对应 [PyTorch avg_pool1d](https://docs.pytorch.org/docs/2.8/generated/torch.nn.functional.avg_pool1d.html) 默认 `count_include_pad=True` 的边界规则，而不是在边缘只除以两个有效邻居。

然后选择 $$\widetilde s$$ 最高的 $$C-W$$ 个 prefix 索引，与全部观察窗口 K/V 拼接。Pooling 并不把预算从 $$C$$ 扩张成“每个峰再多留几个邻居”；它先改变各位置的得分，最终仍受 top-k 数量约束，也不保证选出完整的连续片段。

固定版本的官方代码默认 `avgpool`，同时支持 `maxpool`。论文 §5.2 的一个消融设置使用宽度 5 的 max pooling。平均池化、最大池化与窗口宽度都是具体配置，不能把某个实验设置说成所有 SnapKV 实现唯一的定义。

## 从 Q 和 K 开始算一次完整选择

继续使用 $$L=8,W=2,C=5,D=2$$。为了能手算，构造观察窗口两行 logits 的指数权重如下：

```text
位置          0  1   2   3  4  5  6  7
query 6       1  2  10   8  1  1  2  —
query 7       1  1   8  12  1  1  1  1
```

这可以由真实的点积形式构造：令 $$q_6=[\sqrt2,0]$$、$$q_7=[0,\sqrt2]$$，每个 $$k_j$$ 的两维分别是两行正权重的自然对数，除以 $$\sqrt D$$ 后恰好得到对应 logit。位置 7 对 query 6 被 mask，其人为设置的值不会参与 softmax。

第一行有效权重之和是 25，第二行是 26。Prefix 的六个票数和池化分数为：

| prefix 位置 | 原始票数 $$s_j$$ | 宽度 3 的平均池化分数 |
| --- | ---: | ---: |
| 0 | 0.078462 | 0.065641 |
| 1 | 0.118462 | 0.301538 |
| 2 | 0.707692 | 0.535897 |
| 3 | 0.781538 | 0.522564 |
| 4 | 0.078462 | 0.312821 |
| 5 | 0.078462 | 0.052308 |

原始分数最高的是位置 3，池化以后最高的是位置 2；邻近位置 4 也因靠近峰值得到较高分。最终选出 prefix 的 `[2,3,4]`，再保留 `[6,7]`，共五组 K/V。

下面的完整 CPU 示例从 Q/K 点积开始，不直接把预设的分数塞进 top-k。它只处理单个无 padding head，且要求输入和点积累加处于有限 Python 浮点可表示的范围；多个 head 可以分别调用。Python 浮点与稳定的小索引 tie-break 用于解释算法，不复刻官方低精度转换和 GPU `topk` 的 bitwise 行为。[PyTorch topk 文档](https://docs.pytorch.org/docs/2.8/generated/torch.topk.html)也没有保证相同分数的索引稳定性。

<!-- example: snapkv-select -->
```python
import math


def select_prompt_kv(queries, keys, values, window, capacity,
                     kernel_size=3, pooling="avg"):
    length = len(keys)
    if not length or len(queries) != length or len(values) != length:
        raise ValueError("Q, K and V need the same nonempty prompt length")
    dimension = len(keys[0])
    if not dimension:
        raise ValueError("head dimension must be positive")
    for tensor in (queries, keys, values):
        for row in tensor:
            if len(row) != dimension:
                raise ValueError("all head vectors must have the same dimension")
            if any(isinstance(x, bool) or not isinstance(x, (int, float))
                   or not math.isfinite(x) for x in row):
                raise ValueError("finite numerical vectors are required")
    for number in (window, capacity, kernel_size):
        if isinstance(number, bool) or not isinstance(number, int) or number <= 0:
            raise ValueError("window, capacity and kernel must be positive integers")
    if window > length or capacity <= window:
        raise ValueError("require window <= prompt length and capacity > window")
    if kernel_size % 2 != 1 or pooling not in ("avg", "max"):
        raise ValueError("use an odd kernel and avg or max pooling")

    if length < capacity:
        indices = list(range(length))
        return {"indices": indices, "keys": [list(r) for r in keys],
                "values": [list(r) for r in values], "votes": [], "pooled": []}

    prefix = length - window
    attention_rows = []
    for t in range(prefix, length):
        logits = [math.fsum(a*b for a, b in zip(queries[t], key))
                  / math.sqrt(dimension) for key in keys]
        if not all(math.isfinite(x) for x in logits):
            raise ValueError("non-finite dot product")
        largest = max(logits[:t+1])
        weights = [math.exp(logits[j] - largest) if j <= t else 0.0
                   for j in range(length)]
        total = math.fsum(weights)
        attention_rows.append([w / total for w in weights])

    votes = [math.fsum(row[j] for row in attention_rows)
             for j in range(prefix)]
    radius = kernel_size // 2
    pooled = []
    for j in range(prefix):
        neighbors = [votes[p] for p in range(j-radius, j+radius+1)
                     if 0 <= p < prefix]
        score = (math.fsum(neighbors) / kernel_size if pooling == "avg"
                 else max(neighbors))
        pooled.append(score)
    selected = sorted(range(prefix), key=lambda j: (-pooled[j], j))[:capacity-window]
    indices = selected + list(range(prefix, length))
    return {"indices": indices, "keys": [list(keys[j]) for j in indices],
            "values": [list(values[j]) for j in indices],
            "votes": votes, "pooled": pooled}


a = [1, 2, 10, 8, 1, 1, 2, 1]
b = [1, 1, 8, 12, 1, 1, 1, 1]
keys = [[math.log(x), math.log(y)] for x, y in zip(a, b)]
queries = [[0.0, 0.0] for _ in keys]
queries[6], queries[7] = [math.sqrt(2), 0.0], [0.0, math.sqrt(2)]
values = [[float(j), -float(j)] for j in range(len(keys))]
result = select_prompt_kv(queries, keys, values, window=2, capacity=5)
assert result["indices"] == [2, 3, 4, 6, 7]
assert result["values"] == [values[j] for j in [2, 3, 4, 6, 7]]
print(result["indices"])
```

Gather 必须对 K、V 使用同一组索引；只重排 K 或只重排 V，会把注意力权重配到错误的内容上。索引也不是全局通用的：另一个 head 如果主要关注前几个位置，就应得到另一组选择。

选中位置不一定按原始时间顺序存储。对已经具有正确位置表示的历史 K/V，单 query 读取全部保留历史时，K、V 同步置换不改变该注意力求和。但这不授权任意改写位置编码或多 query 因果 mask；更复杂的批处理和分块解码仍要单独验证。

## 删除位置为什么会改变输出

固定一个未来 query，设其在完整缓存上的 attention 概率为 $$\alpha_j$$。保留位置集合为 $$S$$，完整 softmax 给这部分位置的总概率质量为：

$$
m=\sum_{j\in S}\alpha_j
$$

只要 $$m>0$$，删去其他位置再做 softmax，会把保留位置的权重重新归一化为 $$\alpha_j/m$$，而不是简单保留原权重不变。将完整输出拆成两部分，在 $$0<m<1$$ 时有：

$$
O=mO_{keep}+(1-m)O_{drop}
$$

$$
O_{keep}-O=(1-m)(O_{keep}-O_{drop})
$$

这里 $$O_{keep}$$ 和 $$O_{drop}$$ 分别是两组内部归一化后的 V 加权和。如果所选范数下所有 $$\lVert V_j\rVert\le V_{max}$$，由三角不等式可得到这个局部 attention 输出的界：

$$
\lVert O_{keep}-O\rVert\le 2V_{max}(1-m)
$$

这是对固定 Q/K/V 的直接代数分解，不是 SnapKV 的模型级质量保证。它说明真正相关的是未来 query 在保留集合上的概率质量，以及被删内容与保留内容的差异；观察窗口得票高，并不保证每个未来 query 的 $$m$$ 都高。误差还可能在后续层与自回归过程中传播。

最小反例是两个相同 logits，对应 $$V=[0,10]$$。完整 attention 输出为 5，只留第一个位置以后输出变为 0。没有量化、没有舍入，也已经发生变化。

论文中的 hit rate 是依据 attention 阈值判断重要位置后计算的命中比例，不是上式的概率质量本身，也不等于任务准确率。它能研究筛选模式，但不能替代长文档问答、引用完整性、多跳推理或代码任务的实际质量评估。

## 当前 prefill 仍读完整 K/V，压缩缓存服务后续 decode

在固定提交的 [Llama FlashAttention 接入代码](https://github.com/FasterDecoding/SnapKV/blob/e216ddc84c5bd210378cbdbbba12ba02102aa640/snapkv/monkeypatch/llama_hijack_4_37.py#L73-L96)中，流程是先计算并应用 RoPE，再准备 head 对齐的 K/V；prefill 时把压缩结果写入 `past_key_value`，但当前 attention 计算继续使用局部变量中的完整 K/V。

所以它并不是先删掉大半提示词，再拿剩余位置完成本次 prefill attention。当前 prefill 的完整计算仍要支付，额外的观察窗口打分和筛选也要支付；节省主要面向后续使用缓存的解码阶段。

提示词缓存预算 $$C$$ 也不等于整个生成过程的总缓存上限。该实现不会每步重新筛选已经生成的全部历史；对于触发压缩的 $$L\ge C$$ 分支，当已有 $$g$$ 个新增 token 的 K/V 进入缓存时，物理缓存长度为：

$$
L_{cache}=C+g
$$

短提示词 $$L<C$$ 则旁路筛选，长度为 $$L+g$$；两种情况统一写成 $$\min(L,C)+g$$。这里 $$g$$ 指已经跑过相应 forward 并追加 K/V 的 token 数，不是刚采样、尚未被下一次 forward 处理的 token。若生成很长，新增部分仍会增长。“固定提示词缓存”不能写成“解码多久都只占 $$C$$ 个位置”。

## 缓存槽位不能代替 RoPE 的原始位置

假设原提示词长 8，压缩后留下 `[2,3,4,6,7]`，它们可能存进物理槽位 `0..4`。原始位置 6 的 key 已经按位置 6 旋转，不能因为它现在位于槽位 3，就把它重新当作位置 3 的 key。

类似地，在尚未追加任何生成 token 时，下一个输入 token 的逻辑位置是 8，不是压缩缓存长度 5。已有 $$g$$ 个生成 token 追加以后，下一逻辑位置是 $$L+g$$，不是 $$C+g$$。

```text
原始位置       2   3   4   6   7     下一个逻辑位置 8
物理槽位       0   1   2   3   4     下一写入槽位   5
```

位置之间的距离参与 RoPE 的点积关系。把物理长度当作已处理的逻辑长度，即使所有张量 shape 都能对上，也会改变模型看到的位置关系。实现因此需要区分逻辑进度、缓存长度、原位置索引和已经编码的位置表示。

对于单 query 读历史，不一定每次还要把原始索引显式传给 kernel，因为位置可能已经编码进 K；但在生成输入准备、mask、相对位置偏置、分页映射或多 token 验证里，这些语义仍需保持一致，不能一概只看 compact tensor 的长度。

## GQA 下的缓存节省不能只算 C 除以 L

原生 GQA 用 $$H_{kv}$$ 个 K/V heads 服务 $$H_q$$ 个 query heads，通常 $$H_q>H_{kv}$$。如果同组 query heads 各自选择不同位置，保留后的 K/V 序列就可能不再相同；“一组共享一份缓存”与“每个 query head 任意独立选择”之间存在真实的表示冲突。

固定提交的 Llama 接入选择先 `repeat_kv`，把 K/V 展开到 $$H_q$$ 个 heads，再逐 head 压缩并保存。因此，持久缓存账本应按展开后的 head 数计数，不能一边使用 query-head 独立选择，一边仍假设缓存按原生 $$H_{kv}$$ 保存。

假设各层形状相同，batch 为 $$B$$，层数为 $$N_{layer}$$，每元素 $$s$$ 字节，原生 GQA 的完整缓存逻辑容量为：

$$
Q_{full}=2BN_{layer}H_{kv}(L+g)Ds
$$

按上述展开实现计算，则是：

$$
Q_{SnapKV,expanded}=2BN_{layer}H_q\bigl(\min(L,C)+g\bigr)Ds
$$

两者比值为：

$$
\frac{Q_{SnapKV,expanded}}{Q_{full}}
=\frac{H_q}{H_{kv}}\frac{\min(L,C)+g}{L+g}
$$

取 $$B=1,N_{layer}=32,H_q=32,H_{kv}=8,D=128,s=2$$，提示词 $$L=16384$$、预算 $$C=1024$$，已有 $$g=512$$ 个新增 token 进入缓存：

| 持久 K/V 存储方式 | 逻辑容量 |
| --- | ---: |
| 原生 GQA，不删位置 | 2112 MiB |
| 固定官方接入中的展开 head 后压缩 | 768 MiB |
| 假设仍按 8 个 KV heads 保存的另一种设计 | 192 MiB |

第三行不是这个接入代码已经实现的收益。要保留原生 GQA 共享，必须另行设计组内共享的选择策略或能表达非共享选择的存储与 kernel；它可能改变算法、预算或执行成本。展开方案在这组数值下仍然节省缓存，但其比例是 $$4/11$$，而不是只看序列长度得出的 $$1/11$$。

这些数值只统计持久 K/V，不包括权重、workspace、分配器碎片与临时 attention 张量，也不代表实际峰值显存。短提示词旁路筛选并不会撤销这条接入路径中的 head 展开；随着 $$g$$ 继续变大，展开头数的成本也越来越重要，小提示词缓存不保证始终比原生 GQA 完整缓存更小。

## 少读缓存以前，还要支付一次选择成本

观察窗口需要计算约 $$BH_qWL$$ 个 attention 分数，点积工作量随 $$BH_qWLD$$ 增长。它避免了为筛选单独物化完整 $$L\times L$$ 矩阵，但没有让这项额外工作归零。

例如单样本、32 个 query heads、观察窗口 32、提示词长 16384 时，一份 $$H_q\times W\times L$$ 的 FP32 分数张量就是 64 MiB。真实实现还可能有低精度 logits、softmax 临时结果、top-k 和 gather buffer，而且当前 prefill 仍持有完整 K/V。因此，持久缓存减小不等于 prefill 峰值显存按同样比例减小。

这也影响时间收益。长提示词、后续需要生成较多 token 的请求，可能更有机会摊薄一次选择成本；输出很短时，额外打分可能占据更大比例。首 token 延迟、每 token 解码延迟和完整请求延迟应该分别测量，不能只展示后续 attention 的局部速度。

把它与 [KV 量化]({% post_url 2026-10-15-kivi-kv-cache-axes %})组合也不是简单把两个压缩比例相乘：选位置的分数是否来自量化后的表示、gather 与布局转换怎样执行、低精度 kernel 是否仍支持新布局，都需要重新核对。一个减少位置数量，一个减少每个数的存储位宽，两者引入的误差来源也不同。

## 把算法接入服务以前，需要验证哪些语义

首先检查实际注意力接口，而不是只把示例里的 `gather` 搬过去。本文 CPU 例限定单样本、无 padding 的完整因果注意力；固定官方 `update_kv` 会重新构造观察窗口的因果 mask，不能据此宣称它自动处理任意 padding、ragged batch、packed sequence 或任意外部 mask。

其次检查 cache 与生成状态：短提示词是否正确旁路，预算是否给观察窗口留足空间，K/V 是否同索引，head 映射是否一致，新增 token 是否正确追加，逻辑位置是否独立于物理长度。跨层或跨请求复用也要谨慎：筛选依赖当前提示词和 query，两个请求共享原始 prefix，不表示它们应共享同一份已经剪枝的缓存。

质量评估应在相同模型、模板、位置设置和生成参数下与完整缓存比较，覆盖细节检索、多处证据、长答案与任务切换，而不只观察一个平均分。对不能容忍信息遗漏的请求，还需要明确完整缓存或其他策略的回退条件。

固定仓库 README 标注测试过 `transformers==4.37.0`，并列出 `flash-attn==2.4.0`。这是一套具体历史接入，不是对当前所有 Transformers、分页缓存或推理引擎版本的兼容承诺。迁移时应重新核对 cache API、位置状态与 attention kernel 约束。

SnapKV 最值得保留的思路，是把“全部历史都很长”转化成“用提示词末尾的实际注意力，为后续缓存做一次有条件的选择”。理解这项选择时，算法分数、逻辑位置与物理存储必须同时在场：忽略任何一个，都可能得到看似省显存、实际上换了语义或算错容量的实现。

仓库中的 [CPU 复算脚本](https://github.com/Charlestar/Charlestar.github.io/blob/master/scripts/reviews/snapkv-examples.mjs)会提取本文代码，检查完整 softmax、因果 mask、池化、两个 head 的不同选择、K/V 配对、位置与容量账本。它没有执行官方 GPU 实现、加载模型或复现论文的任务质量与速度实验。

## 参考资料

- [SnapKV 论文 v2](https://arxiv.org/html/2404.14469v2)
- [SnapKV 官方固定版本筛选与池化代码](https://github.com/FasterDecoding/SnapKV/blob/e216ddc84c5bd210378cbdbbba12ba02102aa640/snapkv/monkeypatch/snapkv_utils.py)
- [SnapKV 官方固定版本 Llama 接入](https://github.com/FasterDecoding/SnapKV/blob/e216ddc84c5bd210378cbdbbba12ba02102aa640/snapkv/monkeypatch/llama_hijack_4_37.py)
- [SnapKV 官方固定版本 README](https://github.com/FasterDecoding/SnapKV/blob/e216ddc84c5bd210378cbdbbba12ba02102aa640/README.md)
- [PyTorch 2.8 avg_pool1d](https://docs.pytorch.org/docs/2.8/generated/torch.nn.functional.avg_pool1d.html)
- [PyTorch 2.8 topk](https://docs.pytorch.org/docs/2.8/generated/torch.topk.html)
