---
layout: post
title: "FairBatching：面向 LLM 推理的公平批次形成"
subtitle: "从延迟契约、进度余量到 Prefill Admission Budget"
date: 2026-05-17 12:00:00 +0800
last_modified_at: 2026-09-02
author: iStar
catalog: true
series: serving-scheduling
series_order: 50
technology_year: 2026
mathjax: true
tags: [推理调度, LLM推理]
---

在线大模型服务中的“公平”，并不是把一半 GPU 时间留给 prefill、另一半留给 decode。两类任务的工作形态不同，用户对它们的等待预期也不同：新请求希望尽快看到第一个 token，正在生成的请求则希望后续 token 连续到达。

FairBatching 讨论的是一个更具体的问题：**当 prefill 与 decode 同时争夺一个批次时，怎样利用请求已经取得的进度，在不破坏 decode 流畅度的前提下及时接纳新请求。**

这篇文章沿着一次真实调度决策展开。先建立延迟指标，再解释固定优先级为什么会失效，最后还原 FairBatching 的 deadline、slack、批次时间预算和 Prefill Admission Budget。

> FairBatching 目前来自 2025 年 10 月提交的研究预印本，论文原型基于 vLLM 0.10.1.1。下文会区分论文提出的机制、便于理解的示例，以及当前 vLLM 已公开的能力，避免把研究原型误写成稳定 API。

## 从一次请求到一串 token

一个生成请求进入推理引擎后，通常经历两个阶段。

**Prefill** 一次处理整段输入，建立每层的 KV Cache，并计算第一个输出 token。输入越长，这一步需要处理的 token 越多。

**Decode** 每轮只新增一个或少数几个 token，但每次都要读取此前积累的 KV Cache。一次生成会进行很多轮 decode，因此单轮不大，却不能长时间得不到调度。

用户感受到的延迟也相应分成两部分：

- **TTFT（Time To First Token）**：请求到达至第一个 token 输出的时间；
- **TBT/ITL（Time Between Tokens / Inter-Token Latency）**：相邻输出 token 之间的时间；
- **TPOT（Time Per Output Token）**：从开始生成到当前进度的平均每 token 时间。

对请求 $i$，论文把第 $j$ 个输出 token 的时刻记作 $OutputTime_{i,j}$，到达时刻记作 $ArrivalTime_i$。于是：

$$
TTFT_i = OutputTime_{i,0} - ArrivalTime_i
$$

相邻 token 间隔为：

$$
TBT_{i,j} = OutputTime_{i,j} - OutputTime_{i,j-1}
$$

TTFT 描述“多久开始回答”，TBT 描述“回答是否卡顿”，TPOT 则描述请求从整体上是否跟得上目标进度。三者不能互相替代。

## 冲突发生在同一个 GPU 批次里

假设某一时刻队列中有三个请求：

| 请求 | 当前状态 | 延迟目标 | 眼前风险 |
| --- | --- | --- | --- |
| $P$ | 等待 8K token 的 prefill | TTFT 不超过 500 ms | 再不执行就会超时 |
| $D_1$ | 已输出 20 个 token | 平均每 50 ms 一个 token | 已经领先目标进度 |
| $D_2$ | 刚输出第一个 token | 平均每 50 ms 一个 token | 余量较少 |

这是用于解释的假设场景，不是论文实验数据。

如果调度器永远优先 decode，$D_1$ 与 $D_2$ 会很流畅，但 $P$ 的 TTFT 可能不断增长。如果立即执行完整的 8K prefill，$P$ 可以开始生成，两个 decode 请求却可能出现明显停顿。

现代引擎通常支持 **chunked prefill**：把长输入切成若干块，与 decode token 放进同一批次。它解决了“能否混合执行”的工程问题，却没有自动回答“这一轮应该放多少 prefill”。切得过小，prefill 仍可能长期排队；切得过大，decode 尾延迟仍会恶化。

因此，chunked prefill 是执行机制，batch formation 才是调度策略。

## 固定优先级为什么不够

最直接的策略有两种。

### Prefill 优先

新请求到来就优先做 prefill，TTFT 通常较好。但输入突发时，大量 prefill 会阻塞正在生成的请求，使 TBT 抖动。对聊天和实时交互，这种中途停顿往往比略慢但均匀的输出更明显。

### Decode 优先

只要存在 decode，就先保证它们每轮推进，再用剩余容量处理 prefill。vLLM 的 chunked-prefill 文档也描述了类似的基础取舍：优先调度 decode，将 `max_num_batched_tokens` 的剩余预算交给 prefill。

这能保护流式输出，却存在一个容易忽略的问题：**decode 请求可能在空闲时期提前积累了大量进度，而固定优先级不会把这部分“领先”让给新到达的 prefill。**

回到前面的例子。假设 $D_1$ 的目标是平均每 50 ms 生成一个 token，但在过去 500 ms 内已经生成了 20 个 token。它相当于领先了约 10 个 token。此时短暂放慢 $D_1$，未必会破坏它的整体延迟契约；若调度器仍机械地每轮优先它，$P$ 就可能在本可利用的余量面前违反 TTFT。

FairBatching 所说的 prefill starvation，重点正是这种**没有利用历史进度造成的等待**，而不是系统永远不执行 prefill 的字面意义。

## 为什么只盯 TBT 会误判

TBT 直观，却有一个不适合单独承担调度目标的性质：更早生成某个 token，反而可能让下一次间隔看起来更差。

例如一条稳定输出的时间线是：

```text
token 1:  50 ms
token 2: 100 ms
token 3: 150 ms
```

每个 TBT 都是 50 ms。现在系统在空闲时把 token 2 提前到 60 ms：

```text
token 1:  50 ms
token 2:  60 ms
token 3: 150 ms
```

请求明明更早拿到了 token 2，token 2 到 token 3 的 TBT 却变成了 90 ms。如果策略要求每个间隔都不超过 50 ms，它会认为 token 3 已经违规，并立即抢占其他工作；此前提前 40 ms 的收益完全没有被计入。

这就是论文所谓的非单调性。FairBatching 改用随输出进度增长的时间包络，把“已经提前完成的 token”折算成以后可以消费的 slack。

## 用时间包络描述服务目标

设请求 $i$ 的 TTFT 目标为 $ttft\_slo_i$，之后每个输出 token 的平均时间目标为 $tpot\_slo_i$。为避免序号产生 off-by-one，下面严格沿用调度器的零基 output index：$j=0$ 表示第 1 个输出 token，$j=1$ 表示第 2 个。索引为 $j$ 的 token deadline 是：

$$
token\_ddl_{i,j}
= arrival_i + ttft\_slo_i + tpot\_slo_i \cdot j
$$

这条直线就是请求的进度包络。请求当前需要生成的下一个 token 对应一个 deadline，距离这个 deadline 的时间为：

$$
slack_i = request\_ddl_i - current\_time
$$

- slack 很小或为负，请求接近或已经违反目标；
- slack 很大，请求此前进展较快，可以暂时把资源让给更紧迫的工作；
- 每成功生成一个 token，下一个 deadline 沿包络向后移动一个 TPOT。

假设请求在 0 ms 到达，TTFT 目标为 300 ms，TPOT 目标为 50 ms。第 1 个输出 token 的索引是 $j=0$，deadline 为 300 ms；第 6 个输出 token 的索引是 $j=5$，所以 deadline 是：

$$
300 + 5 \times 50 = 550\ \text{ms}
$$

若前 6 个 token 已经完成，调度器接下来要生成的是第 7 个 token，此时 $next\_output\_idx=6$，对应 deadline 为 600 ms。若当前时间为 430 ms，它还有 170 ms slack。即使它在本轮没有被调度，只要之后仍能在包络内追上，就不等于体验目标已经受损。

这种定义把公平从“每一轮都分到相同资源”改成了“每个请求都沿着自己的延迟契约推进”。

## 从 token 容量切换到时间容量

仅用 `max_num_batched_tokens` 限制批次并不够。相同的新 token 数，在不同上下文长度下可能有完全不同的执行时间：

- FFN 和投影层的成本主要随本轮新 token 数增加；
- attention 还要读取历史 KV，成本受上下文总长度影响；
- 模型、并行方式和 GPU 会改变两部分的比例。

FairBatching 使用一个经过离线 profiling 校准的线性估计器：

$$
batch\_time = a
+ b \cdot total\_new\_tokens
+ c \cdot total\_context
$$

其中 `total_new_tokens` 表示批次中新处理的 prefill/decode token 总量，`total_context` 汇总与 attention 相关的上下文规模，$a,b,c$ 则由具体部署环境拟合。

它不是宣称所有内核都严格线性，而是在调度热路径中用一个足够便宜的模型回答：**再加入这个请求后，批次是否还能在紧迫请求的 deadline 前结束？**

线上若更换 GPU、量化格式、并行策略或 attention backend，都应重新校准，而不能直接沿用另一套环境的系数。

## 自适应确定本轮时间预算

在形成批次前，调度器先查看所有 decode 请求的 slack。论文给出的初始时间预算可概括为：

$$
init\_time\_budget =
\max\left(
\min_i slack_i,
\min_i tpot\_slo_i
\right)
$$

第一项保证批次执行时间不会轻易跨过最紧迫 decode 请求的 deadline。第二项给预算设置一个下限，避免 decode 突发时最小 slack 过小，导致批次被切得极碎、GPU 利用率和调度开销急剧恶化。

这不是一个固定的 prefill 比例：

- decode 请求普遍领先时，时间预算变大，可以接纳更多 prefill；
- 某个 decode 接近 deadline 时，预算收紧；
- 极端高负载下 slack 普遍不足，策略自然趋近 decode 优先，以保护已有请求。

## 一个批次怎样形成

FairBatching 把候选工作分成三组：

1. **紧迫 decode**：slack 已小到需要优先保护；
2. **prefill**：等待建立 KV Cache的新请求或尚未完成的 prefill chunk；
3. **非紧迫 decode**：已经积累一定进度余量的生成请求。

组内按 slack 从小到大排序，然后依次尝试加入批次：先紧迫 decode，再 prefill，最后用剩余时间接纳非紧迫 decode。每加入一项，都用执行时间模型重新估计批次是否超过预算与系统容量。

下面的伪代码只表达论文机制，不对应当前 vLLM 的可复制接口：

```python
def form_batch(running_decodes, waiting_prefills, capacity):
    min_tpot = min(req.tpot_slo for req in running_decodes)
    min_slack = min(req.next_deadline - now() for req in running_decodes)
    time_budget = max(min_slack, min_tpot)

    urgent, non_urgent = partition(
        running_decodes,
        lambda req: req.slack < time_budget + min_tpot,
    )

    urgent.sort(key=lambda req: req.slack)
    waiting_prefills.sort(key=lambda req: req.slack)
    non_urgent.sort(key=lambda req: req.slack)

    batch = []
    for group in (urgent, waiting_prefills, non_urgent):
        for work in group:
            candidate = batch + [work]
            if fits_memory(candidate, capacity) and \
               predict_time(candidate) <= time_budget:
                batch.append(work)

    return batch
```

真实实现不会如此简短。它还需要处理 chunk 大小、KV block 分配、抢占、已计算 token 数、并行 worker 同步和空批次等边界。这里最重要的是三段顺序背后的含义：

- 先守住已经接近违约的生成请求；
- 再让等待首 token 的请求使用可消费的余量；
- 最后才给已经领先目标的 decode 继续“囤积进度”。

## Prefill Admission Budget 把视角扩展到集群

单个实例能公平形成批次，不代表负载均衡器可以无限向它发送新请求。若实例已经没有足够时间和显存接纳 prefill，继续路由只会把 TTFT 违规藏进本地队列。

论文因此提出 **Prefill Admission Budget（PAB）**。它在预留已有 decode 与在途 prefill 的资源后，估计当前 TTFT 目标内还能接纳多少 prefill 工作，并把这个容量暴露给上层调度器。

PAB 可以被理解为“本实例此刻对新 prompt 的可承诺容量”，而不是 GPU 的静态 token 上限。上层可以据此：

- 把请求路由到仍有 admission budget 的副本；
- 在所有副本预算不足时排队或限流；
- 避免只依据队列长度或 GPU 利用率做出误判。

这一步把本地 batch formation 与集群 admission control 连起来了。前者决定下一批执行什么，后者决定是否应该让更多工作进入该实例。

## 接入真实推理引擎时的边界

把论文策略放入生产系统，需要逐项解决以下问题。

### KV Cache 不是无限的

时间模型认为某项工作可以加入，不代表 KV Cache 仍有可分配 block。调度器必须先确认显存容量，必要时决定抢占、换出或拒绝。发生抢占后，重计算/恢复成本也应反馈到执行时间估计中。

### Speculative decoding 改变“下一步”大小

普通 decode 每次通常推进一个 token；推测解码可能提出并验证一棵候选树，实际计算量和接受 token 数都不固定。deadline 按已提交 token 推进，而预算模型要按本轮真实验证工作估算，二者不能混在一起。

### 取消与失败必须回收预算

客户端断开、超时或模型 worker 失败后，请求不能继续占用 KV block、deadline 队列或 PAB。资源提交最好具有可回滚的状态边界，避免“账面已释放、GPU 仍占用”或相反的双重错误。

### 不同请求可能有不同 SLO

交互式聊天、离线生成和高优先级租户不应被强行套用同一 TTFT/TPOT。异构 SLO 可以直接进入 deadline 计算，但必须限制租户配置范围，防止一个不现实的目标长期吞噬全部容量。

### 调度器本身也在延迟路径上

每轮遍历所有请求、尝试大量组合或调用复杂预测器，都会增加 CPU overhead。需要记录 scheduler duration，并为队列建立合适的数据结构。若形成批次花掉的时间接近一个 TPOT，GPU 侧策略再精细也失去意义。

### 时间预测必须持续校准

序列长度分布、prefix cache 命中率和内核选择都会造成漂移。线上应比较预测时间与实测时间，监控残差，并在误差过大时采用保守预算或退化策略。

## 怎样验证策略确实改善了服务

平均 tok/s 无法说明公平调度是否成功。一个系统可能吞吐很高，却让大量请求错过延迟目标。更合适的实验至少覆盖以下维度。

### 工作负载

- 到达过程：平稳、突发和周期性流量；
- prompt 长度：短对话、长文档以及混合分布；
- output 长度：短回答与持续生成；
- SLO：统一目标与按业务区分的异构目标；
- prefix cache：冷缓存与不同命中率。

### 对照策略

- FCFS；
- prefill-first；
- decode-first；
- chunked prefill 的固定 token budget；
- 只看 TBT 的调度；
- FairBatching 的 envelope 与 PAB。

### 指标

- TTFT、TBT/TPOT 的 P50、P95、P99；
- TTFT 与 TPOT SLO 违规率；
- goodput，即满足延迟目标的有效请求或 token 产出；
- 总吞吐、GPU 利用率和 KV Cache 使用率；
- scheduler CPU 时间与执行时间预测误差；
- 不同长度、租户和优先级分组后的尾延迟。

其中 goodput 比单独的吞吐更接近目标：超出 SLO 的大量 token 即使计入 tok/s，也没有兑现系统承诺。

还要特别观察饱和点前后。低负载时，几种策略都可能表现良好；只有在 prefill 突发与 decode 队列并存时，历史进度能否转化为 slack 才真正影响结果。

## 如何理解论文结果

FairBatching 论文在多个模型和三组真实流量 trace 上比较了它与已有策略，报告了 TTFT/TPOT SLO 达成和 goodput 的改善。这些结果证明该机制值得研究，但不能直接换算为任意模型、GPU 和服务目标下的固定收益。

阅读具体数字时应同时核对：

- 使用的模型与硬件；
- prompt/output 长度分布；
- 请求到达率及是否已经过载；
- TTFT、TPOT SLO 的具体倍数或阈值；
- baseline 是否使用相同的 chunked prefill、KV Cache 和并行配置。

论文是预印本，原型也绑定一个特定 vLLM 版本。若准备落地，合理路径是先复现实验与 profiling，再将 deadline/slack 机制适配到当前引擎调度状态，而不是照搬旧版本补丁。

## 小结

FairBatching 的关键并不是提出又一种“prefill 占多少比例”的经验参数，而是把请求进度变成可计算的时间余量：

1. TTFT 约束首 token，TPOT 包络约束后续平均进度；
2. 提前完成的 decode token 会积累 slack；
3. 批次时间预算由最紧迫请求动态决定；
4. 紧迫 decode 先被保护，prefill 随后消费可用余量，非紧迫 decode 最后填充；
5. PAB 再把本地可接纳的 prefill 容量交给集群调度。

由此，“公平”不再是让每类工作每轮都获得相同份额，而是在资源紧张时，仍然尽可能兑现每个请求各自的延迟契约。

## 参考资料

- [FairBatching: Fairness-Aware Batch Formation for LLM Inference](https://arxiv.org/html/2510.14392)
- [Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve](https://arxiv.org/abs/2403.02310)
- [vLLM Optimization and Tuning: Chunked Prefill](https://docs.vllm.ai/en/latest/configuration/optimization/)
- [vLLM V1 Scheduler source](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py)
