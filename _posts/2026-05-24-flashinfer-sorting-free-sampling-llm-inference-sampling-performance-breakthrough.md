---
layout: post
title: "FlashInfer Sorting-Free Sampling：无需显式排序的 GPU 采样"
subtitle: "从目标分布到 Dual Pivot Rejection Sampling"
date: 2026-05-24 12:00:00 +0800
last_modified_at: 2026-08-09
author: iStar
catalog: true
series: gpu-runtime-precision
series_order: 80
technology_year: 2026
mathjax: true
tags: [GPU优化, FlashInfer, LLM推理]
---

大模型每生成一个 token，最后都要从词表分布中选出一个索引。Greedy decoding 只取最大值；随机生成则常先应用 temperature、top-k、top-p 或 min-p，再按剩余概率采样。

这一步看似比 Transformer 前向小得多，却处在每个 decode step 的串行路径上。词表可能有十几万项，动态 batch 中每个请求的过滤参数又不同。如果实现先对整行 logits 排序、累计、mask、恢复原顺序并再次采样，中间读写和 kernel launch 会成为可见开销。

FlashInfer 的 Sorting-Free Sampling 换了一个视角：**采样最终只需要一个服从过滤后分布的 token，并不需要完整的有序词表。** 它通过 inverse transform sampling、rejection sampling 与双 pivot 收缩，在一个 fused kernel 内找到合法样本，避免显式全排序。

这篇文章从分布语义出发，逐步推导算法为什么正确、dual pivot 为什么收敛，再讨论 GPU 实现、随机数状态和服务集成。

## 采样流水线究竟在计算什么

模型输出 logits $z_i$。应用温度 $T$ 后得到概率：

$$
p_i
= \frac{\exp(z_i/T)}{\sum_j\exp(z_j/T)}
$$

随后过滤器定义一个允许集合 $S$，最终分布为：

$$
\tilde p_i
=
\begin{cases}
\dfrac{p_i}{\sum_{j\in S}p_j}, & i\in S\\
0, & i\notin S
\end{cases}
$$

Sampling kernel 的合同是从 $\tilde p$ 采出一个 token。排序只是一种构造 $S$ 的手段，并不是输出要求。

### Top-k

令 $S_k$ 是概率最大的 $k$ 个 token：

$$
S_k=TopK(p,k)
$$

最终只在 $S_k$ 内按原概率比例采样。$k=1$ 等价于 greedy 的候选集合，$k=V$ 则不做过滤。

### Top-p

将概率从大到小记作：

$$
p_{(1)}\ge p_{(2)}\ge\cdots\ge p_{(V)}
$$

Top-p 取累计概率达到阈值 $p_{nucleus}$ 的最小前缀：

$$
m
= \min\left\{r:\sum_{j=1}^{r}p_{(j)}\ge p_{nucleus}\right\}
$$

$$
S_p=\{(1),\ldots,(m)\}
$$

分布尖锐时候选很少，分布平坦时需要更多 token。

### Min-p

Min-p 不看累计和，而是相对最大概率设置门槛：

$$
S_{minp}
= \{i:p_i\ge p_{base}\cdot \max_j p_j\}
$$

它会随本轮最大概率动态缩放，和 top-p 的 nucleus 语义不同。

## Top-k 与 top-p 的组合顺序会改变分布

常见的 `top_k_first` 语义是：

1. 先保留 top-k；
2. 在这 $k$ 项内部重新考虑累计概率并应用 top-p；
3. 对最终集合归一化采样。

`joint` 则在 rejection 的每一轮同时检查 top-k 与 top-p 条件。两者的候选集合不一定相同，因此服务配置、回归测试和缓存 key 都要记录 `filter_apply_order`，不能只记录 $k,p$ 两个数。

举一个小分布：

```text
token: A    B    C    D
prob : .40  .30  .20  .10
```

若 top-k=3、top-p=0.6，按“top-k 后 top-p”的 nucleus 会保留 A 与 B；joint 语义则必须按具体 API 定义判断。即使参数相同，顺序不同也可能改变可采 token。

## 传统排序路径为什么昂贵

直观实现大致是：

```python
sorted_probs, sorted_idx = sort(probs, descending=True)
keep_k = arange(vocab_size) < top_k
keep_p = cumsum(sorted_probs) <= top_p
mask = combine(keep_k, keep_p)
filtered = where(mask, sorted_probs, 0)
sampled_sorted_index = multinomial(filtered)
token_id = sorted_idx[sampled_sorted_index]
```

真实代码还要确保 top-p 至少保留一个 token、处理 ties，并将 mask 恢复到原词表顺序。

对词表大小 $V$，全排序通常需要 $O(V\log V)$ 工作以及多轮全局内存交换。即使 radix sort 的渐进形式不同，它仍会为了得到完整顺序搬运远多于“一个样本”所需的数据。

GPU 上额外问题包括：

- 小 batch 时可并行的 row 数少；
- sort、cumsum、mask 与 multinomial 之间有多个 launch；
- 中间 tensor 多次写入 HBM；
- 每请求不同 $k,p$ 造成动态控制；
- 每步都在 latency-critical decode path 重复。

Sorting-free 的目标不是证明排序永远慢，而是让采样成本更接近“扫描并选一个 token”，而不是“把整个词表排好再选一个 token”。

## 不带过滤时：inverse transform sampling

Categorical sampling 可以从 CDF 反演。先取：

$$
u\sim Uniform(0,1)
$$

定义 prefix sum：

$$
F_j=\sum_{i=0}^{j}p_i
$$

找到第一个满足 $F_j>u$ 的位置 $j$，它被选中的区间长度恰好为 $p_j$，因此：

$$
Pr(sample=j)=p_j
$$

FlashInfer 在 GPU 上让一个 thread block 处理一条概率分布，用 CUB 的 block reduce/scan primitive 并行求和。若 $V$ 大于单个 tile，一次处理一个词表块：

```text
running_sum = 0
for tile in vocabulary:
    tile_sum = reduce(tile.probs)
    if running_sum + tile_sum > u:
        block_scan(tile) -> locate exact token
        return token
    running_sum += tile_sum
```

一旦累计质量超过 $u$ 就提前停止，不必始终扫描完整词表。

## 如何在不知道 top-k 边界时拒绝错误样本

Top-k 采样的难点是：不排序时，最小 top-k 概率阈值 $\tau_k$ 未知。但算法可以逐渐抬高一个 pivot $\tau$。

从所有 $p_i>\tau$ 的 token 中按其概率质量采样。若样本属于最终 top-k，直接接受；若不属于，就把它的概率作为更高 pivot，下一轮不再考虑更小的 token。

```text
round 1: pivot = 0
         在全部 token 中采样 -> 抽到非 top-k r1

round 2: pivot = p[r1]
         只在 p_i > pivot 中采样 -> 抽到非 top-k r2

round 3: pivot = p[r2]
         候选继续缩小 -> 抽到 top-k j -> accept
```

Top-p 的判定略有不同：对当前 pivot，计算高于它的累计质量 $q$。若 $q\ge top_p$，门槛还不够高；若 $q<top_p$，当前样本可落在所需 nucleus 边界。Min-p 则检查 pivot 与最大概率比例。

这些 round 被融合进一次 CUDA kernel，而不是 CPU 反复发起多个 kernel。

## 拒绝采样为什么不会改变 top-k 分布

设真正的 top-k 集合为 $T$，其总概率质量为：

$$
Z=\sum_{j\in T}p_j
$$

在当前 pivot $\tau$ 下，仍可能被抽到的非 top-k 质量记作：

$$
W(\tau)
= \sum_{r\notin T,\ p_r>\tau}p_r
$$

总可采质量为：

$$
S(\tau)=Z+W(\tau)
$$

对某个 $j\in T$，这一轮直接抽中它的概率是 $p_j/S(\tau)$。抽到坏样本 $r$ 时并不输出，只提高 pivot 后继续。设最终返回 $j$ 的概率为 $Q_j(\tau)$，有递推：

$$
Q_j(\tau)
= \frac{p_j}{S(\tau)}
+ \sum_{r\notin T,p_r>\tau}
\frac{p_r}{S(\tau)}Q_j(p_r)
$$

代入候选解 $Q_j(\tau)=p_j/Z$：

$$
\frac{p_j}{S}
+ \frac{W}{S}\frac{p_j}{Z}
= \frac{p_j}{S}\left(1+\frac{W}{Z}\right)
= \frac{p_j}{Z}
$$

因此最终输出分布恰好是 top-k 内按原概率重新归一化的分布。被拒绝的 round 增加计算量，却不会给坏 token 分配最终概率。

这也解释了正确性标准：sorting-free 与 sorting reference 应该服从同一目标分布，不要求相同 seed 下逐 token 输出完全一样，因为两种算法消耗随机数的路径不同。

## 单 pivot 的尾延迟问题

普通 rejection sampler 在常见分布上往往几轮就结束，但缺少严格的 round 上界。若连续抽到概率彼此很接近的坏 token，pivot 每次只提高一点，收敛可能很慢。

平均 kernel latency 看起来不错，少数请求却会经历更多 round，形成 TPOT 长尾。Serving 系统更关心这种可预测性，而不只是平均吞吐。

FlashInfer v0.2.3 引入 Dual Pivot Rejection Sampling，用第二个 pivot 保证每次失败都显著缩小搜索区间。

## Dual Pivot 怎样收缩范围

定义 $f(x)$ 判断概率阈值 $x$ 是否已经满足过滤边界。维护：

$$
low\leftarrow0,\qquad high\leftarrow\max_i p_i
$$

并保持：

$$
f(low)=0,\qquad f(high)=1
$$

每一轮：

1. 在 $p_i>low$ 的质量中做 inverse sampling，得到 token $j$；
2. 令 $pivot_1=p_j$；
3. 再取 $pivot_2=(pivot_1+high)/2$；
4. 根据两个 pivot 的有效性更新区间。

三种情况是：

```text
f(pivot1) = 1
  -> sampled token valid, return j

f(pivot1) = 0, f(pivot2) = 1
  -> low = pivot1, high = pivot2

f(pivot1) = 0, f(pivot2) = 0
  -> low = pivot2
```

若没有接受，新的不确定区间至少缩小一半。因此 round 数有：

$$
O\left(\log\frac{1}{\epsilon}\right)
$$

的浮点精度相关上界，$\epsilon$ 是可区分的最小数值尺度。Dual pivot 的价值不是再改变目标分布，而是给搜索过程提供更可控的最坏情况收敛。

## 一个简化的 top-k 例子

概率为：

```text
token: A    B    C    D    E
prob : .35  .28  .18  .12  .07
top-k = 2
```

目标集合是 `{A,B}`，归一化后：

$$
P(A)=\frac{0.35}{0.63},\qquad
P(B)=\frac{0.28}{0.63}
$$

第一轮从全部质量采样。如果抽到 D，D 不在 top-2，`low` 至少提高到 0.12；下一轮 E 已被排除。若再抽到 C，`low` 提高到 0.18；余下 A/B 都是合法集合，最终输出仍按 0.35:0.28 的比例分配。

真正实现不应先排序来判断 D/C 是否 top-2，否则失去意义。它通过统计高于 pivot 的 token 数/质量来实现过滤判定。

## GPU kernel 的关键不是公式，而是数值细节

一次 fused kernel 要完成：

- 每请求初始化/读取 RNG state；
- 分块读取概率；
- 按 pivot 过滤并 reduce 总质量或计数；
- inverse transform 定位 token；
- 评估 top-k/top-p/min-p 条件；
- 更新两个 pivot 并在 kernel 内重试；
- 输出 sample 与可选 valid 标志。

### 浮点 prefix sum 可能不单调

理论上非负概率的 CDF 单调递增；并行浮点加法不满足结合律，不同 reduction 顺序可能产生微小回退或边界错位。FlashInfer 官方博客特别指出，这会导致无效 token，必须在 token 定位逻辑中显式保护，而不是假设 block scan 天然满足实数算术性质。

### Tie 需要明确定义

当多个 token 概率恰好等于 top-k 边界，集合身份取决于 tie-break 规则。Sorting reference、selection predicate 与统计验证必须一致，否则两边都“看似合理”却不是同一个分布。

### 退化分布必须可报告

全零、含 NaN、过滤后没有有效质量的 row 不应悄悄返回 token 0。当前 API 提供 `check_nan` 和某些接口的 `return_valid`，调用方应依据业务契约处理无效行。

## 当前 FlashInfer API 应怎样选择

官方 sampling 模块同时提供不同输入和输出语义的函数，不能只按名字里有 `top_k` 就互换。

### 直接采样

- `top_k_sampling_from_probs`
- `top_p_sampling_from_probs`
- `top_k_top_p_sampling_from_probs`
- `top_k_top_p_sampling_from_logits`
- `min_p_sampling_from_probs`

`from_probs` 版本要求输入已经是合法概率；当前文档中 fused top-k/top-p probability 接口期望 float32。`from_logits` 版本会承担 logits 到采样路径的相应处理，具体融合和 dtype 以安装版本为准。

### 只过滤/归一化

- `top_k_renorm_probs`
- `top_p_renorm_probs`
- `top_k_mask_logits`

这些函数返回过滤后的 tensor，适合后面还要执行其他 logits processor 或返回处理后 logprobs 的路径。`renorm + sampling_from_probs` 应与对应 fused sampler 在分布语义上等价，但性能和随机数消耗不一定相同。

### 选择 top-k 值本身

当前 FlashInfer 还有 radix-based `flashinfer.top_k`，它是 top-k selection API，可以返回 values/indices，和 sorting-free categorical sampling 不是同一个问题。需要返回完整 top-k 列表时，rejection sampler 只输出一个 token，自然无法替代 selection。

## Seed、offset 与 deterministic 的边界

Sampling API 可接收 `torch.Generator`，也可以显式传 `seed` 与 `offset`。CUDA Graph 场景中，tensor 形式的 seed/offset 更容易在不改变捕获图 shape 的情况下更新。

显式管理时有一个硬要求：每次调用必须推进随机状态。反复使用相同 seed 与 offset，会重复相同随机序列，使不同 decode step 高度相关。

`deterministic=True` 表示选择确定性的 kernel 路径，帮助同一环境重放；完整可重现性仍依赖：

- 相同输入 logits/probs；
- 相同每请求 seed 和 offset；
- 相同 batch 到 row 的映射；
- 相同过滤顺序与参数；
- 相同 FlashInfer/CUDA/GPU 环境；
- 相同的随机数消费路径。

不要把它扩展成“跨实现、跨 GPU、跨版本的 token 序列必然 bitwise 相同”。排序版与 rejection 版即使目标分布相同，也可能消费不同数量的随机数。

## 动态 batch 中的 RNG 所有权

服务系统每一轮都会重新组合 batch。若 RNG state 只绑定 row index，而不是 request ID，就会发生：

```text
step t:   row 0 -> request A
step t+1: row 0 -> request B
```

B 意外继承 A 的随机状态。正确设计应让每个 sequence/request 持有 seed/counter，再在形成 batch 时收集到对应 row。Beam、parallel samples 和 speculative decoding 还需要为每个分支划分不重叠的 counter 空间。

FlashInfer 的 `indices` 参数允许多个输出映射到同一行概率分布，这对 parallel sampling 很有用；但 RNG 仍要按输出实例独立推进，不能因为共享 probs 就共享同一随机样本。

## 统计正确性应该怎样测试

随机 sampler 不能用“一次输出与 reference 相同”验证。适合的测试分四层。

### 精确枚举目标分布

使用很小的词表，手工计算 top-k/top-p/min-p 过滤后的 $\tilde p$。覆盖：

- $k=1$ 与 $k=V$；
- $p$ 接近 0 与 $p=1$；
- 最大概率已经超过 top-p；
- 大量相同概率；
- top-k-first 与 joint；
- 每个 batch row 使用不同参数。

### 硬支持集检查

任何被 mask 的 token 都不应出现。这个检查不需要大量样本，一次非法输出就说明 predicate、tie 或数值边界错误。

### 比较经验分布

重复采样得到频率 $\hat p$，使用 total variation distance：

$$
TVD(\tilde p,\hat p)
=\frac12\sum_i|\tilde p_i-\hat p_i|
$$

也可结合卡方或置信区间。阈值应随样本数与低概率类别调整，不能给所有分布套固定而无统计依据的容差。

### 重放与批次重排

分别验证：相同状态可重放、offset 更新后不重复、batch row 重排不会改变每个 request 的 RNG 流、CUDA Graph replay 时 seed tensor 能生效。

## 性能测试需要构造不同概率形状

Rejection round 数依赖分布形态，只随机生成一组 logits 不够。至少应覆盖：

- 尖锐分布：一个 token 占大部分质量；
- 平坦分布：大量概率接近；
- 长尾分布；
- top-k 很小/很大；
- top-p 很低/接近 1；
- 混合 batch，每行参数不同；
- 32K、128K 及目标模型真实词表；
- batch size 从 1 到线上上限。

记录 mean/P95/P99 kernel latency、平均/最大 rejection rounds、HBM 流量与 occupancy。Dual pivot 的价值应在尾延迟上体现，而不只是平均值。

随后在完整服务中测 TPOT、吞吐与 scheduler overhead。Sampling 只占 decode step 的一部分，kernel-level 大幅加速会受 Amdahl 定律限制；当模型前向或跨 GPU 通信占主导时，端到端变化可能较小。

## 与 vLLM 等引擎集成时

推理引擎通常还要处理：

- repetition/presence/frequency penalty；
- bad words 与 allowed token mask；
- grammar/structured output；
- temperature；
- top-k/top-p/min-p；
- logprobs 返回；
- speculative verification；
- greedy 与 random request 混批。

这些 processor 的顺序就是 API 语义。若先用 FlashInfer fused sampler，后面又要应用 grammar mask，就已经太晚；若为了返回 processed logprobs 必须物化过滤后分布，则“直接输出一个 token”的路径也未必适用。

因此 backend 选择应由实际 processor pipeline 决定。当前 vLLM 源码包含 FlashInfer sampling 路径，但是否命中取决于版本、安装、硬件和本轮参数。应检查目标版本代码与 runtime metrics，不要依赖一个并不存在的通用 `sampling_backend` 配置。

## 小结

FlashInfer Sorting-Free Sampling 的关键不是把 sort 换成另一个 selection API，而是意识到最终目标只是从过滤后分布取一个样本：

1. inverse transform sampling 在概率质量上定位 token；
2. rejection sampling 逐步排除不满足 top-k/top-p/min-p 的候选；
3. dual pivot 保证失败时显著缩小阈值区间；
4. fused CUDA kernel 在一次 launch 内完成多轮搜索；
5. 正确性表现为目标分布等价，而非跨算法 token-by-token 相同。

真正落地还依赖浮点 scan 的数值保护、tie 语义、每请求 RNG state、过滤器顺序与统计测试。只有这些条件都对齐，“无需排序”才既更快，也仍然是在采样同一个分布。

## 参考资料

- [Sorting-Free GPU Kernels for LLM Sampling](https://flashinfer.ai/2025/03/10/sampling.html)
- [FlashInfer Sampling API](https://docs.flashinfer.ai/api/sampling.html)
- [Top-k + Top-p Sampling API](https://docs.flashinfer.ai/generated/flashinfer.sampling.top_k_top_p_sampling_from_probs.html)
- [FlashInfer Top-k Selection API](https://docs.flashinfer.ai/api/topk.html)
- [The Curious Case of Neural Text Degeneration（Nucleus Sampling）](https://arxiv.org/abs/1904.09751)
- [FlashInfer sampling source](https://github.com/flashinfer-ai/flashinfer/tree/main/include/flashinfer/sampling.cuh)
