---
layout: post
title: "KIVI：Key 和 Value 为什么选择不同的量化轴"
subtitle: "从 2-bit 分组误差到 residual buffer、混合缓存注意力与真实容量预算"
date: 2026-10-15
last_modified_at: 2026-10-15
author: iStar
catalog: true
series: model-quantization
series_order: 80
technology_year: 2024
mathjax: true
tags: [模型量化, KV Cache, LLM推理]
---

权重量化压缩的是模型参数，KV Cache 量化压缩的则是每个请求持续增长的历史状态。一个模型的权重已经降到 4-bit，并不意味着长上下文也能低成本运行：每生成一个 token，Attention 仍要读取这个请求在各层积累的 Key 和 Value。上下文变长、并发提高之后，这部分容量和读取流量会越来越重要。

KIVI 在 2024 年提出的方案不是删掉一部分历史 token，而是用很低的位宽表示历史 KV，同时保留一小段高精度状态。它最有辨识度的选择是：**Key 按通道量化，Value 按 token 量化。** 两者在缓存中看起来形状相同，为什么不能直接复用同一套分组方式？答案需要同时看数值分布、Attention 中的计算位置，以及缓存逐 token 增长时的状态变化。

## 1. Key 改变注意力分布 Value 改变加权内容

先只看一个 head，省略 batch 和层编号。已有 $$T$$ 个可见 token，每个 head 有 $$D$$ 个通道：

$$
K,V\in\mathbb R^{T\times D},\qquad q\in\mathbb R^{1\times D}.
$$

一次 decode 的 Attention 可以写为：

$$
s=\frac{qK^T}{\sqrt D},\qquad
p=\operatorname{softmax}(s),\qquad o=pV.
$$

$$s$$ 是各 token 的 logits，$$p$$ 是总和为 1 的注意力权重，$$o$$ 是该 head 的输出。这里的 $$K$$ 表示实际进入点积的 Key；模型使用 RoPE 时，也要遵循其对应的旋转与缓存位置约定。

令量化重建后的缓存为 $$\widehat K=K+E_K$$、$$\widehat V=V+E_V$$。Key 误差首先进入 logits：

$$
\widehat s-s=\frac{qE_K^T}{\sqrt D}.
$$

它经过 softmax 后，会改变“关注哪些 token”。Value 则影响这些 token 提供的内容。记 $$\widehat p=\operatorname{softmax}(\widehat s)$$，输出误差有一个精确分解：

$$
\widehat o-o=(\widehat p-p)V+\widehat pE_V.
$$

第一项来自注意力分布改变，第二项来自用这个分布读取了近似的 Value。因此，单看 $$K$$、$$V$$ 各自的重建 MSE，不足以说明最终 Attention 输出的误差。例如 Key 误差若只是给所有 logits 加上同一个常数，softmax 会把这次统一平移抵消；而某个 Value token 即使只占整个矩阵的一小行，也可能恰好承载大部分注意力权重。

KIVI 根据所测模型的分布和实验发现，Key 有较稳定的通道异常值，而 Value 没有同样明显的模式；两者更合适的量化轴不同。这个观察是方法动机，不是“任何矩阵、任何任务中某种轴一定更优”的数学定理。[KIVI v2，§3.1–3.2](https://arxiv.org/html/2402.02750v2#S3.SS2)

## 2. 2-bit 量化保存的原点不是整数 zero point

先把单个 group 的数值合同写清楚。对一组有限实数，KIVI 使用的 min-max 表示可以写为：

$$
m=\min_i x_i,\qquad
\Delta=\frac{\max_i x_i-m}{2^b-1},
$$

$$
c_i=\operatorname{RNE}\!\left(\frac{x_i-m}{\Delta}\right),\qquad
\widehat x_i=m+\Delta c_i.
$$

当 $$b=2$$ 时，codes 是 $$0,1,2,3$$。论文将 $$m$$ 称为 zero-point，但这里它是**原数值单位下的浮点原点**，不是 [仿射量化基础]({% post_url 2026-10-15-quantization-numerical-foundations %}) 中以整数码为单位的 $$z$$。两种写法不能直接混用。

例如一组范围是 $$[-1,1]$$，四个重建格点为 $$-1,-1/3,1/3,1$$，并没有精确的零格点。把 $$m$$ 换成整数 $$z$$ 而不改网格，会改变原来的量化规则；$$-m/\Delta$$ 也未必是整数。

如果整个 group 都相同，就会出现 $$\Delta=0$$。下面的 CPU 参考显式约定：常量组保存原点 $$m$$，令正步长为 1、codes 全为 0，准确重建这个常量。这是教学参考的特殊分支。固定版 `new_pack.py` 的相关路径直接计算差值再相除，没有相同的常量分支，不能把参考代码的处理自动视为所有真实 kernel 的行为。[固定版 new_pack.py](https://github.com/jy-yuan/KIVI/blob/876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6/quant/new_pack.py)

KIVI 的 asymmetric 包含两个需要分开的含义：数值上，每组使用自己的 min 和 max，不要求正负对称；策略上，Key 与 Value 采用不同量化轴。它不是把同一套对称 INT2 简单应用到全部缓存。

## 3. 同一个大通道为什么会污染一整行

在 $$T\times D$$ 的矩阵中，本文明确使用以下分组定义，避免只说“沿通道维”而产生歧义：

| 缓存 | 共享一套参数的元素 | 固定什么 | 在哪一维收集 $$G$$ 个元素 |
| --- | --- | --- | --- |
| Key per-channel | $$K[t:t+G,d]$$ | 通道 $$d$$ | 时间轴上的连续 token |
| Value per-token | $$V[t,d:d+G]$$ | token $$t$$ | 该 token 的连续通道 |

先构造一个 Key：

$$
K=\begin{bmatrix}
100&0&1&-1\\
101&1.5&1.5&-0.5\\
102&0.5&2&0\\
103&1&2.5&0.5
\end{bmatrix}.
$$

如果每行的 4 个通道共享 2-bit 参数，第一行的最小值是 $$-1$$、最大值是 $$100$$，步长为 $$101/3$$。普通通道的 $$0$$ 和 $$1$$ 都会落到最低码，整行重建成 $$[100,-1,-1,-1]$$。一个大通道决定了整个 token 的数值范围。

如果改为每列跨 4 个 token 共享参数，每列的范围都很窄。第一列用格点 $$100,101,102,103$$，第二列用格点 $$0,0.5,1,1.5$$，其余列也恰好落在自己的网格上。这一构造中，按通道量化可以完全重建 Key。

这里的“完全重建”来自特意选择的格点数值；真实模型当然不会都恰好落在格点上。它展示的是隔离机制：一个持续偏大的通道，不再把其他通道的步长一起拉粗。如果再取 $$q=[0,1,0,0]$$，原始 Attention 最大概率落在第二个 token，按 token 量化后却落在第四个；按通道重建则保持原分布。这样不仅看到了权重范围变化，也看到了它如何影响这次读取历史的决策。

Value 还要考虑另一种污染。构造：

$$
V=\begin{bmatrix}
-100&-100&-100&-100\\
1&0&1&0\\
100&100&100&100\\
0&0&0&0
\end{bmatrix},\qquad p=[0,1,0,0].
$$

这里人为给定 $$p$$，只为隔离 Value 误差，并不是声称某个有限的 Q/K softmax 恰好产生了这个 one-hot 分布。原输出是第二行 $$[1,0,1,0]$$。

按 token 量化时，第二行独立使用 $$[0,1]$$ 的范围，可以准确表示。若改为每列跨 token 共用范围，其他两行的 $$-100$$、$$100$$ 会把步长拉到 $$200/3$$，第二行的元素全被重建成约 $$33.3333$$。完全没有被注意力选中的 token，反而决定了重要 token 的重建分辨率。

这不是说 Value 没有异常值就可以随意量化，而是说：**按 token 分组，可以避免其他 token 的幅度直接决定本 token 的网格。** 一个 token 内部仍可能有难处理的分布，所以还需要分组与高精度残留，而不是只换一条轴就保证质量。

下面每行或每列的 4 个元素是一组，两种分组使用相同 group size。代码同时提供一个低位在前、每 byte 放 4 个 code 的教学 packing，并显式保存逻辑长度；它不是官方 INT32 artifact 布局。参考只处理有限、非空的规则小矩阵，拒绝不可表示的步长、非有限重建与中间结果；它不提供全动态范围高精度保证，也不承诺与 FP16/BF16 后端逐位相同。

<!-- example: kivi-axes-and-2bit-reconstruction -->
```python
from math import exp, isclose, isfinite, sqrt

def quantize_group(values):
    if not values or not all(isfinite(v) for v in values):
        raise ValueError("finite nonempty group required")
    origin, top = min(values), max(values)
    if top == origin:
        codes, step = [0] * len(values), 1.0
    else:
        step = (top - origin) / 3
        if not isfinite(step) or step <= 0:
            raise ValueError("step is outside this float reference's range")
        codes = [min(3, max(0, round((v - origin) / step))) for v in values]
    restored = [origin + step * c for c in codes]
    if not all(isfinite(v) for v in restored):
        raise ValueError("reconstruction is outside this float reference's range")
    return codes, origin, step, restored

def check_matrix(matrix):
    if (not matrix or not matrix[0]
            or any(len(row) != len(matrix[0]) for row in matrix)
            or not all(isfinite(v) for row in matrix for v in row)):
        raise ValueError("finite nonempty rectangular matrix required")

def qdq_rows(matrix):
    check_matrix(matrix)
    return [quantize_group(row)[3] for row in matrix]

def qdq_columns(matrix):
    check_matrix(matrix)
    columns = [quantize_group(list(col))[3] for col in zip(*matrix)]
    return [list(row) for row in zip(*columns)]

def weighted(probabilities, rows):
    check_matrix(rows)
    if (len(probabilities) != len(rows)
            or not all(isfinite(p) and p >= 0 for p in probabilities)):
        raise ValueError("finite nonnegative probabilities must match Value")
    result = [sum(p * row[d] for p, row in zip(probabilities, rows))
              for d in range(len(rows[0]))]
    if not all(isfinite(v) for v in result):
        raise ValueError("weighted result is outside this float reference's range")
    return result

def pack2(codes):
    if any(type(c) is not int or not 0 <= c <= 3 for c in codes):
        raise ValueError("2-bit integer codes required")
    data = bytearray((len(codes) + 3) // 4)
    for i, c in enumerate(codes):
        data[i // 4] |= c << (2 * (i % 4))
    return bytes(data), len(codes)

def unpack2(data, length):
    if (not isinstance(data, (bytes, bytearray)) or type(length) is not int
            or not 0 <= length <= 4 * len(data)):
        raise ValueError("invalid logical length")
    return [(data[i // 4] >> (2 * (i % 4))) & 3 for i in range(length)]

key = [[100.0, 0.0, 1.0, -1.0], [101.0, 1.5, 1.5, -0.5],
       [102.0, 0.5, 2.0, 0.0], [103.0, 1.0, 2.5, 0.5]]
key_token, key_channel = qdq_rows(key), qdq_columns(key)
assert key_channel == key
assert key_token[0] == [100.0, -1.0, -1.0, -1.0]
print("Key per-token first row:", key_token[0])
print("Key per-channel first row:", key_channel[0])

query = [0.0, 1.0, 0.0, 0.0]
def attention_probabilities(keys):
    check_matrix(keys)
    if len(keys[0]) != len(query):
        raise ValueError("Key and Query dimensions differ")
    logits = [sum(q * k for q, k in zip(query, row)) / sqrt(len(query))
              for row in keys]
    if not all(isfinite(v) for v in logits):
        raise ValueError("logits are outside this float reference's range")
    peak = max(logits)
    shifted = [v - peak for v in logits]
    if not all(isfinite(v) for v in shifted):
        raise ValueError("logit differences are outside this float reference's range")
    unnormalized = [exp(v) for v in shifted]
    return [v / sum(unnormalized) for v in unnormalized]

p_original = attention_probabilities(key)
p_token = attention_probabilities(key_token)
p_channel = attention_probabilities(key_channel)
assert p_channel == p_original
assert max(range(4), key=p_original.__getitem__) == 1
assert max(range(4), key=p_token.__getitem__) == 3
assert max(abs(a - b) for a, b in zip(p_original, p_token)) > 0.1
print("Key attention, original / per-token:", p_original, p_token)

value = [[-100.0] * 4, [1.0, 0.0, 1.0, 0.0],
         [100.0] * 4, [0.0] * 4]
p = [0.0, 1.0, 0.0, 0.0]
out_token = weighted(p, qdq_rows(value))
out_channel = weighted(p, qdq_columns(value))
assert out_token == [1.0, 0.0, 1.0, 0.0]
assert all(isclose(v, 100 / 3, abs_tol=1e-12) for v in out_channel)
print("Value output, per-token:", out_token)
print("Value output, per-channel:", out_channel)

codes = quantize_group([100.0, 101.0, 102.0, 103.0])[0]
packet, length = pack2(codes)
assert packet == bytes([0xE4]) and unpack2(packet, length) == codes
packet, length = pack2([0, 1, 2])
assert unpack2(packet, length) == [0, 1, 2]  # padding is not a fourth value
assert quantize_group([5.0] * 4)[3] == [5.0] * 4
assert quantize_group([0.0] * 4)[3] == [0.0] * 4
print("first-column codes:", codes)
```

本例的矩阵尺寸完整成组。后面讨论真实流式缓存时，不足组的 Key 不会被当成一组随意量化；它们先留在 residual buffer 中。Value 则要求通道维能按目标格式分组或采用实现明确支持的补齐规则。

## 4. 时间分组带来了一个流式问题

按 token 量化 Value 很自然：新 token 到来后，它的所有通道已经齐了，可以立即确定这一行的 min 和 max。但是 Key 按通道沿时间分组，新 token 的同组后续成员还没有出现。若每来一个 token 都重新量化整个历史，不仅重复计算，还会改变旧 codes 和参数，破坏稳定的追加布局。

KIVI 因此把缓存分成两部分：已经封存的量化历史，以及尚未封存的高精度残留。这里的 full precision 是模型原有的 FP16/BF16 等浮点表示，并不意味着额外保存 FP32 或数学精确实数。

要分清两个参数：$$G$$ 是量化 group size，$$R$$ 是 residual length。Key 的一个 group 是同通道的 $$G$$ 个连续 token；Value 的一个 group 是同 token 的 $$G$$ 个连续通道。$$R$$ 则控制残留状态的封存或保留策略，要求 Key 的 $$R$$ 能被 $$G$$ 整除，但 $$R$$ 不必等于 $$G$$。论文主要实验使用 $$G=32$$、$$R=128$$。[KIVI v2，§3.3 与 §4.1](https://arxiv.org/html/2402.02750v2#S3.SS3)

Key 的策略是攒满一段再整体封存：新 Key 追加到 residual；达到 $$R$$ 个 token 后，把这一段按每 $$G$$ 个 token 分组量化，接到历史后面，然后清空 Key residual。于是单 token decode **持久化之后**的残留长度会在 $$0$$ 到 $$R-1$$ 之间变化，不是始终保留最近 $$R$$ 个 Key。

Value 不需要等待时间上的同组数据，却仍为质量保留最近的一段浮点状态。它使用 FIFO：新 Value 加到末尾，长度超过 $$R$$ 后，把最老的一行按通道分组量化并移入历史。成熟状态下，持久化的 Value residual 长度为 $$R$$。

这两种策略并不对称。论文所说 Key 的平均高精度窗口约为 $$R/2$$，也不能读成每一步的最低保护长度。按刚才的持久化时点，在完整周期中均匀观察，长度平均值是 $$\left(R-1\right)/2$$，封存之后甚至可以是 0。

## 5. 当前计算和持久状态是两个时点

仅知道上面的缓存形状，还不足以还原一次 Attention。必须问：**是在算当前输出之前封存，还是算完之后封存？**

论文 Appendix A 的 Algorithm 1，先追加新 K/V，再处理 Key 封存和 Value FIFO，然后用更新后的混合缓存计算 Attention。固定到官方 commit `876b4d2`，`LlamaFlashAttention_KIVI` 的 decode 路径采用不同的操作顺序：

```text
论文 Algorithm 1
  追加 K/V → 调整两种缓存 → 计算混合缓存 Attention → 保存状态

固定 Flash decode
  追加 K → 计算当前 logits → 封存满段 K → 全局 softmax
  → 追加 V → 计算当前输出 → 移走最老 V → 保存状态
```

因此，刚好使 Key residual 达到 $$R$$ 的新 Key，在该固定实现的**当次 logits**中仍走浮点分支，尽管随后保存的缓存里，它已经进入量化历史。Value 的当前计算也可能暂时用到 $$R+1$$ 行浮点状态，计算结束才把最老的一行移走。[论文 Algorithm 1](https://arxiv.org/html/2402.02750v2#A1)、[固定版 llama_kivi.py](https://github.com/jy-yuan/KIVI/blob/876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6/models/llama_kivi.py)

用 $$R=4,G=2$$ 展示固定 Flash decode 的顺序，从空状态逐个追加 token，部分关键时点如下。表格记录浮点分支的长度，量化历史与浮点残留合起来始终包含全部可见 token：

| 追加后的总 token 数 | 当前 logits 用到的 K 浮点长度 | 保存后的 K 浮点长度 | 当前输出用到的 V 浮点长度 | 保存后的 V 浮点长度 |
| --- | --- | --- | --- | --- |
| 3 | 3 | 3 | 3 | 3 |
| 4 | 4 | 0 | 4 | 4 |
| 5 | 1 | 1 | 5 | 4 |
| 8 | 4 | 0 | 5 | 4 |
| 9 | 1 | 1 | 5 | 4 |

特别看第 9 步：当前 Key 分界是“8 个历史量化 token + 1 个浮点 token”，Value 却是“4 个历史量化 token + 5 个浮点 token”。不能拿 Key 的边界去切分 Value 的注意力概率。

下面只保存 token 编号，专门检查状态转移，不搭建完整缓存 runtime。`k_compute` 和 `v_compute` 是当前计算使用的快照，`kq/kr/vq/vr` 是持久化状态。示例的 $$R=4,G=2$$ 是状态教学尺寸，**不能直接交给官方原生 2-bit packing**：其 INT32 容器每个装 16 项，kernel 还可能有额外尺寸要求。

<!-- example: kivi-streaming-compute-versus-persist -->
```python
def trace(residual_length=4, group_size=2, steps=9):
    if (type(group_size) is not int or type(residual_length) is not int
            or group_size <= 0 or residual_length <= 0
            or residual_length % group_size):
        raise ValueError("positive G and R with R divisible by G required")
    if type(steps) is not int or steps < 0:
        raise ValueError("nonnegative integer step count required")
    kq, kr, vq, vr = [], [], [], []
    snapshots = []
    for token in range(1, steps + 1):
        kr.append(token)
        k_compute = (kq.copy(), kr.copy())  # current logits are formed now
        if len(kr) == residual_length:
            assert len(kr) % group_size == 0
            kq.extend(kr)
            kr = []
        # global softmax belongs here, after the logits have been formed
        vr.append(token)
        v_compute = (vq.copy(), vr.copy())  # current output is formed now
        if len(vr) > residual_length:
            assert len(vr) == residual_length + 1
            vq.append(vr.pop(0))

        expected = list(range(1, token + 1))
        assert k_compute[0] + k_compute[1] == expected
        assert v_compute[0] + v_compute[1] == expected
        assert kq + kr == expected and vq + vr == expected
        assert len(kq) % residual_length == 0
        assert 0 <= len(kr) < residual_length
        assert len(vr) == min(token, residual_length)
        snapshots.append((k_compute, v_compute, kr.copy(), vr.copy()))
        print(token, "compute K/V full:", len(k_compute[1]), len(v_compute[1]),
              "persist K/V full:", len(kr), len(vr))
    return snapshots

states = trace()
assert [len(s[2]) for s in states] == [1, 2, 3, 0, 1, 2, 3, 0, 1]
assert [len(s[3]) for s in states] == [1, 2, 3, 4, 4, 4, 4, 4, 4]
key_compute, value_compute, _, _ = states[-1]
assert key_compute == (list(range(1, 9)), [9])
assert value_compute == (list(range(1, 5)), list(range(5, 10)))
```

Prefill 也应区分当前计算与留给 decode 的状态。固定 Flash 路径先用原始浮点 K/V 完成 prefill Attention，随后才构造量化历史与 residual。它不是把当前 prompt 的自注意力直接替换成“全部低比特 KV”的计算。[固定版 prefill 分支](https://github.com/jy-yuan/KIVI/blob/876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6/models/llama_kivi.py)

## 6. 混合缓存也只能有一个全局 softmax

Key 被分成量化历史 $$\widehat K_q$$ 与浮点残留 $$K_r$$，可以分两次点积得到 logits：

$$
s_q=\frac{q\widehat K_q^T}{\sqrt D},\qquad
s_r=\frac{qK_r^T}{\sqrt D}.
$$

但归一化必须覆盖全部可见 token：

$$
p=\operatorname{softmax}\!\left(\operatorname{Concat}(s_q,s_r)\right).
$$

不能分别计算 $$\operatorname{softmax}(s_q)$$、$$\operatorname{softmax}(s_r)$$ 再把两段输出直接相加。两段概率会各自总和为 1，相当于丢掉了它们原本应有的相对概率质量。实现可以融合或分块计算 softmax，但各块必须共享等价的全局归一化结果，而不是独立归一化。

取得完整 $$p$$ 之后，再**按 Value 自己的 token 分界**切为 $$p_{vq}$$、$$p_{vr}$$：

$$
o=p_{vq}\widehat V_q+p_{vr}V_r.
$$

固定实现正是拼接 Key 两条路径的 logits 后统一 softmax，再按 `value_full_length` 切概率；量化部分的矩阵乘法使用融合反量化入口，而不是必须把整个缓存先完整解压回显存。[固定版模型路径](https://github.com/jy-yuan/KIVI/blob/876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6/models/llama_kivi.py)、[固定版 matmul.py](https://github.com/jy-yuan/KIVI/blob/876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6/quant/matmul.py)

接着第 9 步的分界，构造一个可以精确解释的 logits 例子：前 8 个 token 的 logit 为 0，最后一个为 $$\ln9$$。全局分母是 $$8+9=17$$，前 8 个各得 $$1/17$$，最后一个得 $$9/17$$。如果 Value 第一维依次是 $$1,2,\ldots,9$$，正确输出就是：

$$
\frac{1+\cdots+8+9\times9}{17}=\frac{117}{17}
\approx6.882353.
$$

错误地对两段分别 softmax，再直接相加，则得到 $$4.5+9=13.5$$。下面假定量化历史已经被解出为给定浮点块，只检查混合路径和归一化合同，不模拟 CUDA kernel。Key 的两段长度为 8 和 1，Value 的两段长度为 4 和 5。

<!-- example: kivi-global-softmax-and-value-boundary -->
```python
from math import exp, isclose, isfinite, log, sqrt

def dot(a, b):
    if not a or len(a) != len(b) or not all(isfinite(v) for v in a + b):
        raise ValueError("finite vectors with equal nonzero dimensions required")
    result = sum(x * y for x, y in zip(a, b))
    if not isfinite(result):
        raise ValueError("dot product is outside this float reference's range")
    return result

def softmax(logits):
    if not logits or not all(isfinite(v) for v in logits):
        raise ValueError("finite nonempty logits required")
    peak = max(logits)
    shifted = [v - peak for v in logits]
    if not all(isfinite(v) for v in shifted):
        raise ValueError("logit differences are outside this float reference's range")
    exponentials = [exp(v) for v in shifted]
    total = sum(exponentials)
    return [v / total for v in exponentials]

def weighted(probabilities, rows):
    if (not rows or not rows[0] or len(probabilities) != len(rows)
            or any(len(row) != len(rows[0]) for row in rows)
            or not all(isfinite(v) for row in rows for v in row)
            or not all(isfinite(p) and p >= 0 for p in probabilities)):
        raise ValueError("probabilities must match this Value partition")
    result = [sum(p * row[d] for p, row in zip(probabilities, rows))
              for d in range(len(rows[0]))]
    if not all(isfinite(v) for v in result):
        raise ValueError("weighted result is outside this float reference's range")
    return result

query = [sqrt(2), 0.0]
key_quant_restored = [[0.0, 0.0] for _ in range(8)]
key_full = [[log(9), 0.0]]
value_quant_restored = [[float(t), -float(t)] for t in range(1, 5)]
value_full = [[float(t), -float(t)] for t in range(5, 10)]
old_logits = [dot(query, k) / sqrt(2) for k in key_quant_restored]
new_logits = [dot(query, k) / sqrt(2) for k in key_full]
probabilities = softmax(old_logits + new_logits)
v_boundary = len(value_quant_restored)  # 4, not the Key boundary 8
old_out = weighted(probabilities[:v_boundary], value_quant_restored)
new_out = weighted(probabilities[v_boundary:], value_full)
mixed = [a + b for a, b in zip(old_out, new_out)]
dense = weighted(probabilities, value_quant_restored + value_full)
assert all(isclose(a, b, abs_tol=1e-12) for a, b in zip(mixed, dense))
assert isclose(mixed[0], 117 / 17, abs_tol=1e-12)
assert isclose(sum(probabilities), 1.0, abs_tol=1e-12)

wrong_p = softmax(old_logits) + softmax(new_logits)
wrong = weighted(wrong_p, value_quant_restored + value_full)
assert isclose(sum(wrong_p), 2.0, abs_tol=1e-12)
assert isclose(wrong[0], 13.5, abs_tol=1e-12)
try:
    weighted(probabilities[:len(key_quant_restored)], value_quant_restored)
except ValueError:
    pass
else:
    raise AssertionError("the Key boundary was incorrectly accepted for Value")
print("global probabilities:", probabilities)
print("mixed output:", mixed, "separate-softmax error:", wrong)
```

混合路径与完整重建参考相等，只能证明这个分段计算本身没有再引入错误，并不证明量化缓存等于原始缓存。前面分解中的 $$E_K$$、$$E_V$$ 仍然存在，softmax 也不会把它们自动消除。

## 7. 2-bit 不等于整个服务显存缩小八倍

先看未量化 KV。若有 $$L$$ 层、batch 为 $$B$$、每层保存 $$H_{kv}$$ 个 KV head、head dimension 为 $$D$$，每个浮点元素占 $$f$$ bytes，则长度为 $$T$$ 的缓存主体占用：

$$
M_{\mathrm{full}}=2LBH_{kv}TDf.
$$

这里是实际保存的 $$H_{kv}$$，不是 Query head 数 $$H_q$$。GQA/MQA 模型的 KV head 数较少，不能按每个 Query head 都保存一份 KV 来计算。这个容量公式本身也不表示某个固定实现已经支持所有 GQA 路径。

再看量化后的预算。每组不只有 2-bit codes，还要保存浮点原点与步长。如果二者各用 FP16、$$G=32$$，每个量化元素的平均成本是：

$$
2+\frac{16+16}{32}=3\ \text{bits}.
$$

即便完全没有 residual 和 padding，量化部分相对 16-bit 也只是约 $$16/3=5.33$$ 倍，而不是 8 倍。固定版 pack 路径为 min/max 分配与输入相同 dtype 的 tensor，步长也由它们计算；因此 metadata 的实际 bytes 应按所用 dtype 计入，不要一律假定是 FP32 或忽略不计。[固定版 new_pack.py](https://github.com/jy-yuan/KIVI/blob/876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6/quant/new_pack.py)

更完整地，设持久化后 Key、Value 的浮点残留长度分别为 $$r_K,r_V$$，codes 位宽为 $$b$$，每个原点或步长占 $$e$$ bytes，浮点残留每元素占 $$f$$ bytes。暂不计 padding，每层、每 KV head、每个请求的缓存成本为：

$$
M=D\left[
(2T-r_K-r_V)\left(\frac b8+\frac{2e}{G}\right)
+(r_K+r_V)f
\right].
$$

这个式子假设 Key 的量化时间长度与 Value 的通道维都完整成组；整批还要乘上层数、请求数和 KV head 数。

例如持久状态 $$T=4096,D=128,G=32,R=128$$，恰好有 $$r_K=0,r_V=128$$，并取 $$b=2,e=f=2$$。未量化的单层单 KV head 是 2 MiB；量化历史约 378 KiB、浮点残留 32 KiB，总计 410 KiB，约压缩 5 倍。真实分配还包括 packing 对齐、tensor metadata、临时 logits 和 workspace；模型权重更不会因为量化 KV 同步缩小。

短上下文下 residual 占比很高，收益会进一步降低。长上下文下量化历史占比提高，但写入量化、读取反量化和 Attention kernel 是否高效，又决定容量节省能否变成延迟或吞吐收益。论文中的吞吐改善还包含显存释放后允许更大 batch 的收益，不能换算成任意单请求、固定 batch 的加速承诺。[KIVI v2，§4.2.4](https://arxiv.org/html/2402.02750v2#S4.SS2.SSS4)

## 8. KIVI 保留了历史位置 但改变了历史数值

KIVI 没有主动删掉旧 token，也没有把模型改成更少的 KV heads。它与 KV eviction、GQA/MQA、分页管理可以影响不同维度的成本，但组合起来需要实际适配缓存格式和执行路径，不能只在配置中叠几个开关。

例如固定版普通 `LlamaAttention_KIVI` 对 KV 分组数有明确限制；Flash 分支虽然在若干位置使用 `repeat_kv`，也不能据此推导所有短缓存、量化历史为空或非空的 GQA 状态都已验证兼容。量化 axes 的原理、容量公式与某个实现的全路径支持，是三种不同结论。这里的代码只执行标准库数值和状态示例，不构成对真实 Torch/Triton/CUDA 路径、任务质量或 GPU 性能的复现。[固定版模型实现](https://github.com/jy-yuan/KIVI/blob/876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6/models/llama_kivi.py)

论文中的 tuning-free 指该方法不依赖 AWQ 那样的离线校准集和参数搜索：min/max 随在线到达的 group 确定，而不是先训练一个量化器。它不等于无需选择 $$G$$、$$R$$，也不等于“2-bit 对所有输入无损”。group size、残留长度和模型分布都会影响结果。长文本检索、位置敏感任务以及持续 decode 中的缓存演化，应针对量化后的模型判断，不能只看一个静态矩阵的重建误差。

若继续比较 [NVFP4 KV Cache]({% post_url 2026-06-02-nvfp4-kv-cacheblackwell %})，重点会转向另一种数值表示、缩放与硬件执行；若继续读 [QuantSpec]({% post_url 2026-05-21-QuantSpec-Self-Speculative-Decoding-with-Hierarchical-Quantized-KV-Cache-for-Long-Context-Inference-Acceleration %})，还会遇到草稿和目标的不同缓存视图，以及接受、拒绝和回滚。KIVI 的单向追加示例不能自动证明推测解码的事务状态也正确。

KIVI 值得记住的不是“KV 可以压到 2-bit”这一句话，而是其完整设计关系：Key 用跨 token 的通道分组隔离大通道，Value 用单 token 内的分组限制跨 token 污染；Key 等待整段封存，Value 保留最近 FIFO；两种缓存的分界不必相同，但它们共同参与一次全局归一化的 Attention。数值轴、时间轴和计算时点三者对齐，才是一套能解释清楚的 KV 量化方法。

## 参考资料

- [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache，arXiv v2](https://arxiv.org/html/2402.02750v2)：方法、实验与 Appendix A；该修订为 2024-07-25，研究首发于 2024 年 2 月。
- [固定版 models/llama_kivi.py](https://github.com/jy-yuan/KIVI/blob/876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6/models/llama_kivi.py)：prefill、decode、两种缓存边界与状态更新顺序。
- [固定版 quant/new_pack.py](https://github.com/jy-yuan/KIVI/blob/876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6/quant/new_pack.py)：min-max 量化、分组轴与 packing。
- [固定版 quant/matmul.py](https://github.com/jy-yuan/KIVI/blob/876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6/quant/matmul.py)：量化缓存矩阵乘法入口。上述代码引用均固定在同一 commit，不代表其他版本或其他后端的兼容性矩阵。
