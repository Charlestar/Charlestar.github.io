---
layout: post
title: "Chunked Prefill：长 Prompt 为什么要切片执行"
subtitle: "从 Prefill/Decode 干扰到 Token Budget 与 Stall-Free Scheduling"
date: 2026-06-06 09:00:00 +0800
last_modified_at: 2026-09-03
author: iStar
catalog: true
series: serving-scheduling
series_order: 20
technology_year: 2023
mathjax: true
tags: [推理调度, LLM推理]
---

Continuous Batching 让推理引擎可以在模型迭代之间移走已完成请求、接纳新请求，但它留下了一个棘手的问题：**新请求第一次进入 GPU 时，往往不是一条轻量的 decode，而是一整段 prompt 的 prefill。**

当 32 条请求正在稳定生成时，插入一个 32K-token prompt，可能让所有请求很久都拿不到下一个 token。若为了保护生成流畅度而永远优先 decode，这个长 prompt 又可能一直得不到首 token。

Chunked Prefill 的思路很直接：不要求一个 prompt 在一次调度中全部算完，而是沿 token 维切成若干片段。调度器每轮只处理其中一段，将剩余 token 留给后续迭代：

```text
完整 prefill:  [---------------- 8192 tokens ----------------]

分块 prefill:  [2048] [2048] [2048] [2048]
                  ^      ^      ^      ^
              decode 可以在这些调度边界持续推进
```

切片并不是简单地把文本拆成四个独立请求。每一块都必须继承前面块生成的 KV Cache，attention 也要看到此前所有 token。它真正改变的是执行计划，而不是模型语义。

本文将依次回答：prefill 为什么会阻塞 decode，沿 token 维切分为什么仍能得到同一结果，chunk 与 decode 如何组成 batch，token budget 如何控制延迟，以及生产环境中应怎样选择和验证 chunk size。

## Prefill 和 Decode 为什么会互相干扰

对一条长度为 $L_p$ 的 prompt，prefill 会让所有输入 token 通过模型，并为每一层生成 K/V。线性层、MLP 等计算大致随本轮 token 数增长；causal attention 则要计算 prompt 内的依赖关系。

Decode 阶段每轮通常只处理一个新 token，却要读取此前累积的 KV Cache。把二者放到 GPU 的视角中，会看到明显不同的工作形态：

| 阶段 | 本轮新 token 数 | 主要特征 | 常见瓶颈 |
| --- | ---: | --- | --- |
| Prefill | prompt 长度或 chunk 长度 | 大矩阵乘法，token 并行度高 | 计算吞吐 |
| Decode | 每请求通常 1 个 | 小矩阵乘法，读取大量历史 KV | 显存带宽、launch overhead |

这只是常见倾向，不是对所有模型和 batch 的绝对判断。小 prompt 的 prefill 可能没有充分利用 GPU；大 decode batch 也可能把矩阵乘法做得很大。但在典型在线负载中，长 prefill 与 decode 的执行时间差异足以制造明显干扰。

假设 16 条 decode 请求每轮耗时 20 ms，一个新长 prompt 的完整 prefill 耗时 240 ms。若调度器直接执行完整 prefill：

```text
decode round      : 20 ms
long prefill      : 240 ms
next decode round : 20 ms
```

正在生成的请求会出现约 260 ms 的 token 间隔，而此前节奏可能只有 20 ms。这个突然拉长的间隔常被称为 generation stall。

如果不断有长 prompt 到达，简单的 prefill-first 会反复制造 stall；简单的 decode-first 则可能让 waiting queue 中的 prompt 长时间无法开始。这不是 Continuous Batching 的批次可变性能够自动解决的，它需要控制每次允许插入多少 prefill 工作。

## 为什么不能只把 Decode Batch 做大

一种直觉是：既然 prefill 影响生成，就等所有 decode 结束后再做 prefill。但在线服务通常不会出现整齐的空档。

只要请求持续到达，系统中就可能一直存在 running decode：

```text
t0: A、B 开始 decode
t1: C 等待 prefill
t2: D 到达，A、B 仍在 decode
t3: E 到达，B、D 仍在 decode
...
```

若规则是“没有 decode 才执行 prefill”，C 可能无限等待。反过来，若每个新请求都立即完整 prefill，已有请求的 TPOT 会随到达流量剧烈抖动。

更重要的是，decode-only batch 不一定高效。每条请求只提供一个新 token，batch 较小时无法形成足够大的矩阵乘法；模型权重仍要从显存读取，算力却可能没有吃满。Prefill 有大量并行 token，恰好可以提高本轮 arithmetic intensity。

所以目标不是把 prefill 从 decode 旁边赶走，而是形成一种有界混合：

> 用足够大的 prefill chunk 提高 GPU 利用率，同时把 chunk 限制在不会长时间阻塞 decode 的范围内。

Sarathi 把这种思路称为让 decode “piggyback” 在 prefill chunk 上，并用 decode-maximal batching 尽量把正在生成的请求覆盖到每个混合批次中。

## 沿 Token 维切分为什么仍然正确

设 prompt 为：

$$
x_1,x_2,\ldots,x_L
$$

完整 prefill 会一次处理全部 $L$ 个 token。Causal attention 保证位置 $t$ 只能关注 $1\ldots t$：

$$
Attention(q_t,K_{1:t},V_{1:t})
$$

现在选择 chunk size $C$，先计算 $x_1\ldots x_C$，保存各层的 $K_{1:C},V_{1:C}$；下一块计算 $x_{C+1}\ldots x_{2C}$ 时，每个 query 同时访问：

- 前一块已经缓存的 $K_{1:C},V_{1:C}$；
- 当前块中因果掩码允许访问的早期位置。

第二块结束后，缓存扩展到 $K_{1:2C},V_{1:2C}$。重复这个过程，最后一个位置看到的上下文与完整 prefill 相同。

因此，从模型计算图的语义看：

$$
Prefill(x_{1:L})
\equiv
Chunk(x_{1:C}) \rightarrow Chunk(x_{C+1:2C}) \rightarrow \cdots
$$

这里的等价指相同 causal 依赖与模型结果；不同 kernel、归约顺序或精度可能产生常见的浮点细微差异。Chunked Prefill 不应删除跨块 attention，也不能把每个 chunk 当成独立 prompt 重新使用 position 0。

实现需要为每块提供至少这些信息：

```text
token IDs of current chunk
absolute positions
number of already computed tokens
KV block table for previous chunks
slot mapping for newly written K/V
causal attention metadata within current chunk
```

只有文本切片，没有 KV 连续性和正确 position，得到的就不是原模型的 prefill。

## 一条长 Prompt 如何穿过多轮调度

假设 prompt 有 7000 token，每轮全局 token budget 为 2048，同时有 32 条 decode 请求，每条本轮需要 1 token。

调度器先为 decode 预留 32 token，剩余：

$$
2048-32=2016
$$

于是长 prompt 的执行进度可能是：

| 轮次 | Decode token | Prefill chunk | Prefill 累计进度 |
| ---: | ---: | ---: | ---: |
| 1 | 32 | 2016 | 2016 / 7000 |
| 2 | 32 | 2016 | 4032 / 7000 |
| 3 | 32 | 2016 | 6048 / 7000 |
| 4 | 32 | 952 | 7000 / 7000 |

第 4 轮完成后，请求才得到用于采样首个输出 token 的最终 hidden state。也就是说，chunked prefill 可以缩短每次 stall，却不会让总 prefill 计算消失；这个请求的 TTFT 至少跨越四轮。

这条时间线揭示了一个核心取舍：

- chunk 越小，每轮对 decode 的阻塞越短，ITL 更平滑；
- chunk 越小，长 prompt 需要的轮数越多，TTFT 和调度开销可能上升；
- chunk 越大，prefill 更快完成，但单轮执行时间和 decode 尾延迟可能升高。

因此 chunk size 不是越小越好，也不是只要“能装进显存”就越大越好。

## Sarathi 的 Decode-Maximal Batching

2023 年的 SARATHI 工作提出 chunked-prefill 与 decode-maximal batching。它不是简单地轮流运行一块 prefill、一轮 decode，而是构造混合批次：批次中包含一个 prefill chunk，并尽量装入等待执行的 decode token。

例如固定 chunk 为 512 token，当前有 24 条 decode：

```text
batch = 512 prefill tokens + 24 decode tokens
```

Prefill chunk 提供较大的矩阵乘法，让 GPU 接近计算饱和；decode token 与它共批次执行。论文观察到，在相应实验条件下，把 decode 捎带到 prefill 上的增量成本可以显著低于单独执行 decode batch。

为什么会这样？模型权重在一轮中本来就要为 prefill 读入并参与矩阵乘法，加入少量 decode token 只是扩大 token 维度，许多操作能共享同一轮权重访问和 kernel launch。相较之下，单独执行小 decode batch 仍需走完整模型层，却缺少足够并行工作。

Sarathi 将这种批次设计用于两个目标：

1. 让正在生成的请求尽量每轮都被覆盖，减少 generation stall；
2. 让不同 pipeline micro-batch 的执行时间更接近，减少 pipeline bubble。

“Stall-free” 是调度目标，而非任何 chunk size 下都能保证的结果。如果 prefill chunk 本身仍耗时 200 ms，decode 即使被放在同一 batch 中，也要等这轮完成才能拿到 token。chunk 需要根据目标 token interval 和硬件实测时间来定。

## 固定 Chunk Size 与动态 Token Budget

理解 Chunked Prefill 时，经常会混淆两个相近但不同的控制方式。

### 固定 Chunk Size

Sarathi 的基本描述将 prefill 切成近似相等的块，例如每块 512 token。它带来较稳定的 batch 计算量，尤其有利于 pipeline parallel 中平衡 micro-batch。

```text
prompt 2300 tokens -> 512 + 512 + 512 + 512 + 252
```

固定 chunk 容易 profile，却可能浪费每轮剩余容量。若当前只有 4 条 decode，与当前有 200 条 decode 时使用相同 chunk，批次总工作量会明显不同。

### 动态填满 Token Budget

现代调度器常先给 running 请求分配工作，再把本轮剩余 token budget 交给 prefill。于是实际 chunk 大小由当轮负载决定：

$$
C_t = \min(P_{remaining}, B_{token}-D_t)
$$

其中：

- $P_{remaining}$ 是 prompt 尚未计算的 token 数；
- $B_{token}$ 是一轮 token budget；
- $D_t$ 是本轮 decode 或其他运行请求占用的 token 数。

这种方式适应性更强，但每轮执行时间会随 $D_t$ 和上下文分布变化。为了避免单条长 prompt 吞掉几乎全部预算，还可以增加 per-request threshold 或限制同时进行的 partial prefills 数量。

两种方式都属于 chunked prefill。前者强调规则形状和 stall-free batch，后者强调统一 token 调度下的容量填充，不能只看到配置名就假定实现完全相同。

## vLLM V1 如何表达这件事

vLLM V1 的统一 scheduler 不必维护互斥的“prefill 请求类型”和“decode 请求类型”。它更关注两个进度：

- `num_computed_tokens`：模型已经实际计算到哪里；
- `num_tokens_with_spec`：prompt、已生成 token 和当前 speculative tokens 一共要求计算到哪里。

待完成工作近似为：

$$
num\_new\_tokens
= num\_tokens\_with\_spec
- num\_computed\_tokens
$$

对于新 prompt，这个差值可能是数千；对于普通 decode，通常是 1。Scheduler 在全局 `max_num_scheduled_tokens` 内给每个 request 分配本轮 token 数：

```python
{
    "decode-A": 1,
    "decode-B": 1,
    "prompt-C": 2046,
}
```

若 C 还剩 6000 prompt token，就先推进 2046，下一轮继续追赶。这个表示同时适用于 chunked prefill、prefix cache 命中和 speculative decoding，减少了为每种优化设计独立状态机的需要。

当前 vLLM 官方调优文档说明，在可用场景中 V1 默认启用 chunked prefill；基础策略先调度 pending decode，再把 `max_num_batched_tokens` 的剩余预算给 prefill，放不下的 prefill 自动切块。

这些是当前版本行为，不应被写死为所有引擎或历史版本的通用定义。升级时仍要核对目标版本的 SchedulerConfig 与调度源码。

## Token Budget 不是执行时间 Budget

用 token 数限制一轮工作很便宜，却只是执行时间的近似。

下面三种批次都可能有 2048 个 scheduled token：

```text
2048 prefill tokens from one short-context request
1024 prefill tokens + 1024 single-token decodes
2048 decodes with widely different context lengths
```

它们的运行时间通常不同。原因包括：

- decode attention 读取量随历史上下文增长；
- prefill attention 成本受 chunk 与已有 prefix 长度共同影响；
- GEMM shape、kernel 选择和 padding bucket 不同；
- 多 GPU collective 的消息形状不同；
- prefix cache、量化和 MoE 路由会改变执行路径。

因此 `max_num_batched_tokens=4096` 不代表每轮具有固定时延。它更像一个稳定、低开销的容量护栏。

若系统需要严格控制 P99 ITL，可以在 token budget 之上加入执行时间预测、deadline slack 或直接做 prefill/decode 资源分离。FairBatching 属于前一种方向，DistServe 等 disaggregated serving 属于后一种方向。

## Chunk Size 怎样影响 TTFT、TPOT 与吞吐

调优时可以先建立三个近似关系。

设 prompt 长度为 $L_p$，每块大小为 $C$，需要的 chunk 数为：

$$
N_{chunk}=\left\lceil\frac{L_p}{C}\right\rceil
$$

若第 $j$ 个 chunk 的执行时间为 $T_j(C)$，中间还夹有调度与等待开销 $Q_j$，则 prefill 完成时间近似为：

$$
T_{prefill,chunked}
=\sum_{j=1}^{N_{chunk}}(T_j(C)+Q_j)
$$

Chunk 越小，$N_{chunk}$ 越大，重复调度、kernel 边界和排队机会增加，TTFT 可能上升。但已有 decode 请求在任一轮受到的最大阻塞更接近单个 $T_j(C)$，所以 TPOT 尾部更容易控制。

Chunk 越大，GPU 对 prefill 的效率可能更高，长 prompt 更快完成；代价是每轮 stall 更长，可共存的 decode 或其他 prefill 数量也减少。

实际曲线通常不是单调的。某些 chunk size 恰好落在高效 GEMM tile 或 CUDA Graph bucket 上，另一些则触发不同 attention kernel。必须用目标模型、目标 GPU 与真实长度分布测量，而不能把论文中的 512 或文档示例中的 2048 当成普适最优值。

## KV Cache 分配必须跟着 Chunk 增长

完整 prefill 可以在开始前为整段 prompt 估算 KV 空间；chunked prefill 则让请求逐轮增长。假设 block size 为 16 token，并且请求此前恰好停在 block 边界，本轮增加 1000 token 需要：

$$
\left\lceil\frac{1000}{16}\right\rceil=63
$$

个新的物理 KV blocks。若此前最后一个 block 还有空位，新分配数量可能少于 63；实际还要考虑尾块余量、多组 KV cache 与混合 attention 类型。这里的 63 是边界对齐条件下的例子，不是所有请求的固定下界。

调度器通常要先 reserve blocks，再把 chunk 交给 worker。若只按第一块的空间接纳大量长请求，可能出现：

```text
很多长请求各完成一个小 chunk
-> KV Cache 被半成品占满
-> 没有请求能获得后续 blocks
-> 频繁 preemption / recompute
```

这是一种 over-admission。每个请求单看都能执行第一块，整体却在后续增长时造成 cache thrashing。

可选的防护方式包括：

- 限制同时处于 partial prefill 的请求数；
- admission 时检查完整输入长度是否可容纳；
- 为 KV pool 保留 watermark；
- 根据 prompt 剩余长度与 preemption 成本选择请求；
- 在抢占频繁时降低入口并发，而不是继续缩小 chunk。

Chunked Prefill 降低的是单轮计算阻塞，不会减少整段 prompt 最终需要的 KV Cache 容量。

## Prefix Cache 命中后应该怎样切

若 prompt 前 6144 token 已经命中 prefix cache，完整输入为 8192 token，那么真正需要 prefill 的只有后 2048 token：

```text
[ cached 6144 ][ uncached 2048 ]
```

调度器应先把命中的完整 blocks 计入 `num_computed_tokens`，再只对未命中后缀分块。若 token budget 为 1024，可能执行两块，而不是重新切完整 8192。

这要求 cache lookup、computed progress 与 chunk boundary 使用一致的 block 语义。常见边界包括：

- 只复用完整 KV blocks，尾部不足一块的 token 需要重算；
- cache key 必须包含会改变 KV 的模型或 adapter 身份；
- position、RoPE scaling 和多模态输入 hash 必须一致；
- prompt logprobs 可能要求对缓存 token 重新得到输出信息。

Prefix caching 减少总 prefill 工作，chunked prefill 控制剩余工作每轮如何进入 GPU。它们互补，但分别优化“算多少”和“每次算多少”。

## Attention Kernel 面临的特殊形状

对第一块 prompt，attention 类似普通 prefill：query 和 KV 都来自当前前缀。对后续块，query 长度是 chunk size，KV 长度则是此前 prefix 加当前块：

```text
chunk 1: query 0..511,    KV 0..511
chunk 2: query 512..1023, KV 0..1023
chunk 3: query 1024..1535, KV 0..1535
```

这既不是纯 decode 的单 query，也不是从位置 0 开始的完整 prefill。Kernel 需要支持带 prefix 的 causal attention，并把旧 K/V 从 paged cache 读取、新 K/V 写回正确 slots。

一个混合 batch 中还可能同时存在：

- 单 token decode；
- 不同 prefix 长度的 prefill chunks；
- 刚好完成 prefill、需要采样首 token 的请求；
- speculative verification 的多 token query。

运行时通常把所有 query token packed 成一维布局，再通过 cumulative sequence lengths、query lengths、context lengths 与 block tables 描述边界。若仍依赖二维 `[batch, max_seq_len]` 并统一 padding，chunked prefill 的效率优势会被大量无效 token 抵消。

## Pipeline Parallel 为什么尤其在意规则 Chunk

Pipeline Parallel 将模型层分布在多个 stage 上，并让不同 micro-batch 在 stage 间流动。吞吐取决于各 micro-batch 的执行时间是否接近。

若某个 micro-batch 是长 prefill，另一个只有少量 decode：

```text
micro-batch P: 180 ms
micro-batch D:  20 ms
```

快 stage 完成 D 后仍要等待慢 prefill 推进，形成 pipeline bubble。把长 prefill 切成多个相近大小的 chunk，再让 decode 分散搭载到这些 chunk 上，可以缩小 micro-batch 时长差异。

这也是 Sarathi 除了在线 token stall 之外强调的另一项收益：chunking 同时是一种 pipeline load balancing 手段。

不过多 stage 环境还要考虑：

- 每个 stage 的层计算量并不完全相同；
- KV Cache 分布在 stage 对应层上；
- scheduler 必须保证各 rank 使用一致的 batch metadata；
- chunk 太小会增加 pipeline 调度和通信频次；
- MoE 层的动态路由可能破坏固定 token 数对应固定时间的假设。

因此规则 chunk 只是减少方差的起点，不能代替逐 stage profiling。

## 长 Prompt 并发时的公平性问题

只有一条长 prompt 时，剩余 token budget 全给它通常合理。若有十条长 prompt 同时等待，调度器还要决定 partial prefills 的并发方式。

### 串行完成

先连续推进 P1 的 chunks，P1 完成后再处理 P2。它让最早请求较快获得首 token，但后面的 TTFT 很长。

```text
P1-1 P1-2 P1-3 P1-4 | P2-1 P2-2 ...
```

### 轮转推进

每轮换一个 prompt：P1-1、P2-1、P3-1。等待更均匀，却让每条请求都更晚完成全部 prefill；同时更多半成品占用 KV Cache。

```text
P1-1 P2-1 P3-1 P1-2 P2-2 P3-2 ...
```

### 长短区分

允许短 prompt 越过很长 prompt，可改善整体 TTFT 或 short-job goodput，但必须定义 aging，避免长请求永远被跳过。

这说明 chunked prefill 是执行机制，不自带唯一公平策略。`max_num_partial_prefills`、long-prefill threshold、FCFS/priority 与 admission policy 会共同决定最终行为。

## 多模态输入不能总在任意 Token 位置切开

文本 token 的依赖边界相对规则，多模态请求还包含 image/audio encoder 的输出或占位区间。若一个图像特征对应连续的一组 embedding，调度器未必允许从中间切开：

```text
[text][ image embedding span ][text]
              ^
         可能不允许从这里断开
```

原因可能包括 encoder output 尚未完整产生、attention metadata 需要完整 item、processor 使用不可分割的 feature group，或 cache 生命周期不同。

因此引擎往往需要额外的 encoder compute budget、encoder cache，以及“是否允许 partial multimodal input”的配置。不能把文本 chunked prefill 的 token budget 逻辑原封不动地套到所有多模态模型。

验证多模态场景时，应专门覆盖：

- chunk boundary 落在媒体占位区间前后；
- 相同图片的 processor/cache 命中；
- 多图请求的 encoder budget；
- 取消请求后的 encoder cache 回收；
- 文本 TTFT 与媒体加载时间的分段观测。

## Chunked Prefill 与 P/D 分离的边界

Chunked Prefill 仍让 prefill 和 decode 共享同一组 GPU。它通过时间切片降低干扰，但两类工作仍争夺：

- model weights 与 SM；
- HBM bandwidth；
- KV Cache 容量；
- scheduler token budget；
- CUDA Graph 与 kernel execution path。

当尾部 ITL 需要严格可预测，或者 prompt 长度和到达流量波动极大，仅靠固定 token budget 很难稳定控制每轮时间。Prefill/Decode disaggregation 会让不同 worker 分别执行两阶段，再传输 KV Cache。

两者不是简单的先进替代落后：

| 方案 | 优势 | 主要成本 |
| --- | --- | --- |
| Chunked Prefill | 单实例内完成，无 KV 跨节点传输；混合批次可提高利用率 | 干扰仍存在，chunk size 难统一 |
| P/D 分离 | 资源可独立扩缩，decode 尾延迟更可控 | KV 传输、路由和集群复杂度 |

低并发或单机部署中，chunked prefill 往往更直接；长上下文、高负载和严格 SLO 场景中，P/D 分离可能更容易建立容量模型。本系列后续会单独讨论 DistServe、KV transfer 和 disaggregated serving。

## 如何选择 Token Budget 与 Chunk 上限

没有脱离负载的统一参数，但可以按可验证的过程选择。

### 1. 先测单项执行曲线

固定模型、精度和并行配置，分别测量：

- 不同 prefill chunk size 的执行时间；
- 不同 decode batch size 与 context length 的执行时间；
- 混合 batch 的执行时间；
- scheduler/metadata 准备的 CPU 时间。

不要只测总吞吐。需要得到类似：

```text
T_prefill(chunk_tokens, prefix_length)
T_decode(num_seqs, context_distribution)
T_mixed(prefill_tokens, decode_seqs, contexts)
```

### 2. 用 TPOT 目标给单轮时间设上界

若交互业务希望 P99 TPOT 不超过 100 ms，单个混合 batch 的目标耗时应明显低于 100 ms，还要为排队、通信、采样和抖动留余量。

从 profiling 曲线中选择不会频繁跨过该上界的 chunk 区间，而不是直接从显存能容纳的最大 token 数开始。

### 3. 回放真实长度与到达分布

把短/长 prompt、短/长 output、prefix cache 命中和突发到达一起回放，逐步扫描 `max_num_batched_tokens`、`max_num_seqs` 与 partial-prefill 限制。

### 4. 以 Goodput 选择工作点

比较同时满足 TTFT 和 TPOT SLO 的完成请求数，而不是选择 raw tok/s 最大点。一个参数可能提高 10% 吞吐，却让大量流式请求出现不可接受的停顿。

### 5. 检查过载后的退化方式

流量超过容量时，观察系统是有界排队、明确拒绝，还是 KV thrashing、反复抢占和全体超时。后者说明 admission 与 cache policy 有问题，继续微调 chunk size 不能根治。

## 监控时要把一个 Prefill 拆开看

只记录“request 正在 prefill”会掩盖关键进度。建议至少采集：

- prompt 总 token 与已 computed token；
- 每轮 scheduled prefill token；
- partial-prefill request 数；
- prefill 等待时间、计算时间和跨越的 engine steps；
- 每轮 decode token 数与 mixed batch 比例；
- KV blocks 分配、释放和 preemption；
- TTFT、TPOT 与 per-step GPU duration；
- token budget 未用满的原因。

最后一项尤其有价值。某轮只调度了 512/4096 token，可能因为：

- 没有更多请求；
- `max_seqs` 已满；
- KV blocks 不足；
- multimodal item 不可切；
- 当前策略限制长 partial prefill；
- distributed rank 需要对齐；
- 某些 shape 不能进入目标执行图。

不区分原因，只看到“GPU 利用率低”，很容易错误地继续增大 token budget。

## 常见故障模式

### Chunk 太大：Decode 仍然卡顿

表现为平均 TPOT 尚可，但每次长 prompt 到达都伴随 P99 ITL 峰值。应把 per-step duration 与 scheduled prefill tokens 对齐，确认是否由大 chunk 引起，再评估缩小 budget 或做时间感知调度。

### Chunk 太小：TTFT 与 CPU 开销上升

长 prompt 跨越太多 engine steps，scheduler、input preparation 和 kernel launch 的固定成本被反复支付。GPU 可能呈现大量细碎执行，prompt 完成率反而下降。

### Partial Prefill 过多：KV Cache Thrashing

许多请求都算了一点，却都无法完成。KV usage 长期接近 100%，preemption/recompute 持续上升。应收紧 admission 或 partial-prefill 并发，而不是让更多请求进入第一块。

### Token Budget 看似合理，实际时间波动很大

通常与 context length、MoE 路由、kernel shape 或多 GPU collective 有关。需要从纯 token 估算升级为 workload-aware profiling，或把不同工作拆到不同资源池。

### Prefix Cache 与 Chunk 进度不一致

表现为重复计算、position 错误或 cache block 越界。要检查命中 token 是否只按完整 blocks 计入、computed progress 是否与 slot mapping 同步，以及 cache identity 是否包含所有模型条件。

### 请求取消后仍保留半成品

Partial prefill 已分配多轮 KV blocks。取消路径若只移除 waiting entry 而未释放 cache，会造成缓慢显存泄漏。需要测试在每个 chunk boundary 和 GPU 在途期间取消。

## 一个完整的验证矩阵

上线前可以用下面的维度组合建立回归集。

| 维度 | 建议覆盖 |
| --- | --- |
| Prompt | 32、512、4K、32K、接近 max model length |
| Output | 1、32、256、长生成 |
| 到达 | 单请求、稳定并发、prefill burst、decode 稳态中插入长请求 |
| Cache | 冷缓存、完整命中、部分命中、eviction 后重算 |
| 并行 | 单 GPU、TP、PP，必要时 DP |
| 资源 | KV 充足、接近容量、触发 preemption |
| 功能 | LoRA、speculative decoding、多模态、priority、取消 |
| 参数 | 多组 token budget、max seqs、partial-prefill 限制 |

正确性检查包括：

- chunked 与 unchunked 在 greedy decoding 下输出语义一致；
- position IDs、RoPE 与 attention mask 正确；
- prompt logprobs 等接口语义符合目标版本；
- prefix cache 命中与未命中结果一致；
- 分布式 rank 不发生 batch metadata 分歧；
- 取消、OOM 和 worker failure 后 KV blocks 可回收。

性能检查则至少报告 TTFT/TPOT 分位数、goodput、吞吐、每轮时间、KV usage 和 scheduler CPU overhead，并给出具体模型、硬件、软件版本与长度分布。

## 小结

Chunked Prefill 解决的是 Continuous Batching 中最容易破坏交互体验的一次重工作：新请求的长 prompt。

它沿 token 维把 prefill 拆成多轮，每一块继承前块的 KV Cache 和绝对位置，因此保持 causal attention 语义；调度器再将有限 prefill 工作与 decode token 组成混合 batch。Sarathi 的 decode-maximal batching 说明了这种组合为何既能减少 generation stall，也能利用 prefill 的计算密集特征承载更多 decode。

真正落地时，需要同时看四个层次：

```text
模型语义：跨 chunk 的 attention 与 position 必须连续
调度策略：chunk 如何分享 token/time budget
内存系统：partial prefill 如何增长和回收 KV blocks
GPU runtime：混合变长输入如何 packed 并高效执行
```

Chunk 越小并不天然越好。它用更多轮次和调度开销换取更短的单次阻塞；chunk 越大则可能提高 prefill 效率，却恶化 decode 尾延迟。合理参数只能来自目标环境下的执行曲线、真实流量回放与 SLO goodput。

有了 Continuous Batching 与 Chunked Prefill 两个基础，后面的 EngineCore 才能进一步讨论调度状态如何穿过进程边界；异步连续批处理会解释 CPU 调度如何与 GPU 执行重叠；FairBatching 则会把固定 token budget 推进到面向 TTFT/TPOT 契约的时间预算。

## 参考资料

- [SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills](https://arxiv.org/abs/2308.16369)
- [Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve](https://arxiv.org/abs/2403.02310)
- [vLLM Optimization and Tuning: Chunked Prefill](https://docs.vllm.ai/en/latest/configuration/optimization/)
- [vLLM V1 Scheduler source](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py)
- [vLLM SchedulerConfig source](https://github.com/vllm-project/vllm/blob/main/vllm/config/scheduler.py)
- [TensorRT-LLM: Paged Attention, In-flight Batching, and Request Scheduling](https://nvidia.github.io/TensorRT-LLM/features/paged-attention-ifb-scheduler.html)
