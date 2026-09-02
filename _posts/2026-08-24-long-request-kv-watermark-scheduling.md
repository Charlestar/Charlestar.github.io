---
layout: post
title: "长请求治理：Chunked Prefill 之后，KV Cache 怎样避免被长输出占满"
subtitle: "从输入公平调度、动态 KV 准入到高水位保护，理解长尾请求的完整生命周期"
date: 2026-08-24 09:00:00 +0800
last_modified_at: 2026-09-03
author: iStar
catalog: true
series: serving-scheduling
series_order: 30
technology_year: 2023
mathjax: true
tags: [LLM推理, 推理调度, KV Cache]
---

在线 LLM 服务里，长请求往往只占很小比例，却能决定整个集群的尾延迟。一条超长输入会让 GPU 长时间执行 Prefill，后到的短请求迟迟看不到首 Token；一条超长输出则会在 Decode Pool 里停留几千轮，每轮继续增长 KV Cache，最终让新请求连显存准入都做不到。

这两类问题看起来都叫“长请求”，资源形态却不一样：

```text
长输入：短时间内集中消耗 Prefill Compute，并一次建立大量 KV
长输出：长时间占用 Decode Slot，每轮追加 KV，持续消耗带宽与容量
```

Chunked Prefill 能把长输入拆成多个有界片段，让短请求穿插执行；它不会缩短总输入，也不会限制输出随后增长到多长。只完成 Chunked Prefill，系统仍可能在 Decode 阶段被少量长序列拖入 KV 高水位。

因此完整的长请求治理需要三条控制线同时工作：

1. **执行公平**：长输入在 Chunk 边界让出 GPU，但不能永久饥饿；
2. **内存准入**：新请求进入前，要考虑 Active Sequence 的未来 KV 增长；
3. **高水位保护**：当预测失效或输出异常变长时，系统有逐级、可解释的降压动作。

本文沿着一条请求从等待、Prefill、Decode 到结束的生命周期，说明 Token Budget、Time Budget 与 KV Budget 怎样配合，以及哪些“保护动作”会改变请求语义、必须由接口契约明确授权。

## 1. 平均长度为什么掩盖不了长尾

假设 95% 请求输入 1K Token、输出 200 Token，剩余 5% 请求输入或输出 32K。平均值看起来可能还能接受，但资源驻留不是简单平均：

- 长 Prefill 的单轮 Kernel 时间远大于短 Decode；
- 长输出持有 KV 的时间是普通请求的数十倍；
- KV 未释放前，后续请求无法使用这些 Block；
- 一次长 Kernel 会同时推高一批 Decode 请求的 TPOT；
- 多个长请求在同一实例相遇，会产生局部热点。

排队系统的尾部由服务时间分布和资源相关性决定。仅用平均 ISL/OSL 配容量，会漏掉“长输入与长输出是否出现在同一请求”“是否集中到同一租户”“是否在同一时间 Burst”等关键结构。

工作负载至少要保存联合分布：

$$
P(ISL,OSL,arrival,tenant,priority,prefix\_hit)
$$

而不是分别采样两个边缘平均值。

## 2. 长输入与长输出分别伤害什么指标

TTFT 可以粗略分解为：

$$
TTFT=T_{queue,P}+T_{prefill}+T_{handoff}+T_{first\_decode}
$$

长输入主要增加 Prefill 计算与排队。如果同一 GPU 上还有 Active Decode，一个完整长 Prefill 还会让这些请求等待更久才得到下一 Token。

TPOT/ITL 则依赖每轮 Decode 的排队与执行：

$$
TPOT_i\approx
\frac{T_{last\_token,i}-T_{first\_token,i}}
{N_{output,i}-1}
$$

长输出本身未必让单轮变慢，但它长期留在 Active Batch 中，KV Length 持续增加；达到较大并发时，每轮 Attention 读取的总 KV 更多，其他请求也会受到影响。

因此“短请求优先”不能只看输入长度，也不能只在 Prefill Queue 排序。系统要同时管理首 Token Deadline、逐 Token 进度和内存驻留时间。

## 3. 一条请求应有显式状态机

长请求治理很容易在多个队列中失去一致状态。可以先把生命周期写清楚：

```text
WAITING_ADMISSION
  → PREFILL_READY
  → PREFILL_RUNNING
  → PREFILL_PAUSED
  → DECODE_READY
  → DECODING
  → DECODE_PREEMPTED
  → FINISHED / CANCELLED / REJECTED / ABORTED
```

每个状态至少绑定：

- 已完成的 Token 进度；
- 当前 KV Block 与逻辑位置；
- 还需要的 Prompt Token；
- 已生成和允许生成的最大 Token；
- Deadline/Priority/租户配额；
- 抢占次数与累计服务时间；
- 是否可以重算、Swap 或迁移；
- 客户端是否仍连接。

调度器不能只维护“这是 Prefill 还是 Decode”。Prefix Cache 命中、Chunked Prefill、推测 Token 和恢复请求都可能让进度一次移动不同数量。

## 4. Chunked Prefill 改变的是执行粒度

若 Prompt 长度为 16K，Chunk Size 为 2K，完整 Prefill 不再一次执行，而是：

```text
round 1: prompt[0:2K]
round 2: prompt[2K:4K]
...
round 8: prompt[14K:16K]
```

每块必须读取之前块生成的 KV，并使用全局 Position。它不是八条独立 Prompt，也没有减少总 Attention 工作。

收益来自每轮占用变得有界：Scheduler 可以在 Chunk 之间插入短 Prefill 与 Decode Token，避免一个 16K Kernel 连续阻塞所有请求。

但“支持切块”只是执行能力，真正的公平性取决于谁获得下一块。如果长请求每轮都立刻获得下一个 Chunk，它仍然会连续占用 Prefill 预算；如果永远优先新到短请求，长请求又可能无法完成。

## 5. 固定 Chunk Size 不能直接代表公平

相同 Token 数的 Chunk，执行时间可能差很多：

- 当前 KV Length 不同；
- Attention Backend 与模型结构不同；
- Batch 中 Decode 数量不同；
- Prefix Cache 命中改变 Query Length；
- MoE Expert 路由和通信负载不同；
- MultiModal Span 可能不能任意切分。

因此仅设置 `chunk_size=2048`，只能约束 Token Work，不能保证每轮 GPU 时间相同。

更完整的调度会同时使用：

```text
token budget：本轮最多推进多少 Token
time budget：本轮预计允许占用多少时间
memory budget：本轮最多新增多少 KV Block
```

三者分别约束吞吐形状、延迟干扰和容量安全。

## 6. 基于配额的让出与恢复

一种可解释策略是给长 Prefill 发放有限执行配额：

1. 请求获得一个或若干 Chunk；
2. 配额耗尽后，在 Chunk 边界保存进度并回到 Ready Queue；
3. 短请求先使用释放的时间/Token 预算；
4. 长请求按累计等待或让出次数提高恢复优先级。

可以定义一个简单 Aging Score：

$$
priority_i
=base_i
+\alpha\cdot wait_i
+\beta\cdot yields_i
$$

其中 `base` 由业务优先级决定，`wait` 防止长时间排队，`yields` 补偿已经多次主动让出的请求。

公式不是唯一方案。关键不变量是：短请求能穿过长工作，长请求也有可证明的最终进展；不能只优化 P50，却让某些请求无限等待。

## 7. Prefill 暂停后必须保留什么

在 Chunk 边界暂停长 Prefill 时，已经完成的前缀 KV 是继续执行的断点。恢复必须保持：

```text
computed_tokens
logical positions
KV block table / ownership
prefix cache references
attention mask state
model / adapter / tokenizer identity
```

如果暂停时释放 KV，恢复就要重算已完成前缀；如果保留 KV，它会在等待期间占用容量。Scheduler 实际是在 Compute Waste 与 Memory Residency 之间选择。

长 Prefill 数量少时，保留断点通常合理；系统接近 KV 高水位时，可能把低优先级暂停请求的 KV Offload 到 CPU，或释放后允许重算。无论哪种方式，都要在状态中记录，不能让 Scheduler 以为 Block 仍在 GPU。

## 8. Prefix Cache 命中会改变长输入的真实工作

两条 32K Prompt 的成本可能完全不同：

```text
request A: 30K prefix hit + 2K uncached
request B: 0 prefix hit + 32K uncached
```

调度优先级和 Chunk 数量应基于 `uncached_tokens`，而不是原始 ISL。Cache-Aware Router 还要把命中位置与实例负载一起考虑：为了多命中 2K Token，把请求送到已经严重拥塞的 Worker，可能反而恶化 TTFT。

Prefix Cache 命中减少 Prefill Compute，却不会消除最终 Active Context 的 KV 容量。如果命中的 Block 需要重新 Onboard 到 HBM，Admission 仍要为它们分配可读位置。

## 9. Prefill 结束不是内存压力结束

请求完成 Prefill 时，已经拥有 \(N_{in}\) 个上下文位置。之后每生成一个 Token，KV Cache 再增加一个位置：

$$
N_{KV}(t)=N_{in}+N_{generated}(t)
$$

对一条输出上限 32K 的请求，刚进入 Decode 时只看到输入 KV，后续还可能增加 32K。多个请求同时增长时，当前空闲 Block 并不能代表未来安全。

这正是长输出与长输入治理的分界：Chunked Prefill 可以控制“输入 KV 多快建立”，却不能阻止 Decode 在几千轮内把 Pool 推向满载。

## 10. 为什么不能为 `max_tokens` 全额预留

最保守方案是在请求进入时为 `input_tokens + max_tokens` 预留全部 KV。这样不会中途 OOM，却会严重降低并发：大量请求会提前 EOS，预留空间长期闲置。

完全不考虑未来增长则走向另一端：当前 Batch 可以装下，但几十轮后所有请求一起增长，Scheduler 只能频繁抢占。

更实用的是动态准入：

```text
current committed KV
+ bounded near-term growth reserve
+ safety watermark
<= physical KV capacity
```

Near-term Reserve 可以按未来若干 Scheduler Iteration、长度分位数、Priority 与可抢占性估算。它不能提供绝对不抢占保证，但能避免在明显不可持续的状态下继续接纳新请求。

## 11. 把 KV Pool 划成多个压力区间

可以用三个水位表达逐级压力：

```text
normal       : usage < low watermark
pressure     : low <= usage < high
critical     : high <= usage < emergency
exhausted    : no allocatable block
```

水位不是为了制造更多配置，而是让动作按风险递增：

| 区间 | 推荐动作 |
| --- | --- |
| Normal | 正常准入与批处理 |
| Pressure | 降低新 Prefill/Sequence 准入，偏向完成 Active Request |
| Critical | 停止低优先级准入，路由分流，选择性抢占/Offload |
| Exhausted | Fail closed；不得无 Block 继续写 KV |

阈值需要留出 Kernel Workspace、异步释放延迟和 Block 对齐，不能把理论 100% 容量当作安全终点。

## 12. 高水位时先做什么，后做什么

一条相对温和的降压顺序是：

```text
1. 停止把新请求路由到该实例
2. 暂停本地新 Prefill Admission
3. 让接近完成的 Active Decode 优先释放 KV
4. 降低低优先级并发或迁移可迁移请求
5. Offload / Recompute 可恢复状态
6. 仅按明确契约终止异常或可重试请求
```

顺序背后的原则是先停止增加压力，再帮助已有工作完成，最后才采取会增加计算、传输或改变客户端结果的动作。

如果一到高水位就随机杀请求，系统虽然不 OOM，却把容量问题变成不可预测的业务失败。

## 13. Engine 和 Global Router 必须形成闭环

本地 Engine 最早知道 KV Block、Active Sequence 和每轮时间；Global Router 决定新流量继续去哪台实例。两者需要共享至少：

```text
ready / draining state
waiting prefill work
active decode sequences
KV used / free / reserved blocks
preemption rate
predicted TTFT / TPOT
prefix cache locations
watermark state
```

如果 Engine 已进入 Critical，Router 仍根据旧指标持续发请求，本地 Admission 再聪明也只能堆积或拒绝。Cache Event、Load Event 和 Endpoint Lease 要有版本与过期时间，避免控制面把失联 Worker 当成低负载节点。

## 14. Decode 抢占有三种不同成本

KV 不足时，可以对某条 Active Request 采取不同策略。

### 14.1 Recompute

释放 KV，稍后从已提交 Token 前缀重新 Prefill。它节省传输，但浪费 GPU Compute；上下文越长，重算越贵。

### 14.2 Swap/Offload

把 KV 移到 CPU/SSD，恢复时再 Onboard。它保留计算结果，却消耗 PCIe/网络与 Host Memory，可能干扰前台请求。

### 14.3 Migrate

把请求状态转移到另一个 Decode Worker。除了 KV，还要迁移 Token、Position、Sampling RNG、Grammar/Stop 状态和 Adapter 身份；目标端必须有兼容模型与容量。

选择可以基于成本：

$$
C_{recompute}
\quad vs \quad
C_{offload}+C_{onboard}
\quad vs \quad
C_{migrate}
$$

并加入 Deadline 与目标端排队。不存在对所有长度和互联都最优的固定模式。

## 15. 抢占对象不能只选“最大 KV”

释放最大请求能最快回收空间，却可能杀伤一个即将完成、已经等待很久的高价值请求。

Victim Score 可以综合：

- 可释放 KV Bytes；
- 已完成比例与预计剩余 Token；
- 重算/传输成本；
- 业务 Priority 与 Deadline；
- 累计抢占次数；
- 客户端是否允许重试；
- 目标 Worker 的 Prefix/Adapter Locality。

系统还应限制同一请求的抢占次数。反复释放、恢复、再释放会产生 Thrashing，既不完成请求，也持续消耗资源。

## 16. “异常长输出”必须由接口契约定义

输出很长不代表模型异常。代码生成、长报告和推理任务本来就可能需要大量 Token。服务端不能因为系统压力大，就在未声明的情况下截断结果。

可安全执行的硬限制包括：

- 客户端传入的 `max_tokens`；
- 模型或服务公开的最大上下文；
- 明确的租户 Token/成本配额；
- 客户端取消或 Deadline 到期；
- 已声明的安全与内容策略。

“检测到循环输出”也要谨慎。重复字符串可能是有效数据。若提供重复检测，应作为显式产品选项，返回独立 `finish_reason`，并允许任务类型配置，而不是隐藏在 Scheduler 中。

## 17. 结束原因要区分容量保护与模型完成

至少需要区分：

```text
stop          正常 EOS 或 Stop Sequence
length        达到请求/服务长度上限
cancelled     客户端主动取消
deadline      SLO/业务 Deadline 到期
preempted     状态被保存或将重算，尚未最终失败
overload      准入前因容量不足拒绝
aborted       运行中按明确策略终止
error         运行时故障
```

把高水位终止伪装成 `stop`，会让调用方以为回答完整。可恢复抢占也不应该先向客户端发送最终错误，再后台继续。

## 18. FairBatching 解决的公平与内存公平不同

FairBatching 通过 TTFT/TPOT 进度与 Slack 决定批次形成，主要约束时间上的服务公平。KV 高水位还带来内存公平：一条长序列可能在很多轮内占用远多于其他请求的 Block。

两者要联合：

```text
time entitlement：请求多久应获得一次执行
memory entitlement：请求可以长期占用多少可抢占状态
```

只做时间公平，可能让所有请求都按时推进，直到 KV 一起增长并撞墙；只做内存上限，又可能频繁抢占长但重要的请求。

租户级策略可以给交互、批处理、Agent 和离线作业不同权重，同时保留 Aging，避免低优先级永远无法完成。

## 19. SLO 优先级不能等同于静态队列优先级

静态 High/Low Priority 简单，却不反映请求离 Deadline 还有多远。一个低等级请求可能已经等了很久，而一个高等级请求刚刚到达且余量充足。

可用紧迫度：

$$
urgency_i
=\frac{predicted\_remaining\_service_i}
{deadline_i-now}
$$

值越大，越可能错过目标。但预测要考虑它当前 KV Length、所在 Worker、Batch Shape 与可能的抢占成本。

硬业务优先级仍可作为权重或保留容量，而不是完全被模型预测取代。若预测失效，Scheduler 应回退到可解释的 Priority + Aging 策略。

## 20. P/D 分离后长请求跨越两个准入器

P/D 架构中，Prefill Admission 成功不代表 Decode 一定有容量。若 P 持续接收长输入并快速产出大型 KV，而 D Pool 已接近高水位，会出现：

```text
P finished
→ KV waiting for handoff
→ D cannot admit
→ P/host buffer retained
→ transfer queue and memory grow
```

因此 P Admission 需要看到 D 的预测容量，或者系统要为已接受的 Prefill 预留 Decode Handoff Credit。

Credit 可以按 KV Bytes、目标 D Class 和有效期发放。没有 Credit 时，P 应延迟或拒绝低优先级请求，而不是算完再发现无处可去。

## 21. Cache-Aware 路由也要看长输出驻留

Router 常把请求送到 Prefix 命中最多的 Worker。但一个拥有长 Prefix 的 Worker 可能同时承载许多长输出，KV 已接近 Critical。

更完整的成本函数是：

$$
Cost_j=
C_{uncached\_prefill,j}
+C_{queue,j}
+C_{transfer,j}
+C_{future\_KV,j}
+C_{SLO\_risk,j}
$$

命中价值只是其中一项。Router 还应对持续偏向同一 Worker 设置 Load Penalty 和 Max Wait，防止 Cache Locality 制造热点。

## 22. 多租户环境要防止 KV 容量被单方占满

只限制 QPS 无法控制长请求：一个租户每秒只发一个请求，也可能每条生成数万 Token。

更有意义的配额包括：

- 并发 Active Sequence；
- 输入/输出 Token Rate；
- KV Token-seconds 或 Block-seconds；
- 可驻留/可 Offload KV Bytes；
- 每分钟抢占与重算预算；
- Priority Class 的保留容量。

`KV Token-seconds` 同时反映大小和时间：占 32K KV 一秒，与占 1K KV 三十二秒具有相同的一阶容量面积。它比单次 Token 数更接近长期资源成本。

配额拒绝应发生在 Admission，并返回明确可重试信息；不要等到显存满后随机牺牲其他租户的 Active Request。

## 23. 取消必须尽快传播到 KV Allocator

客户端关闭连接后，如果 Router、Engine 与 Worker 之间的取消信号延迟，模型仍可能继续生成，KV 也继续增长。

取消路径需要：

```text
client disconnect
→ request cancellation epoch
→ stop future scheduling
→ cancel/ignore in-flight result safely
→ release KV / prefix refs / offload buffers
→ publish capacity event
```

GPU Kernel 通常不能任意中断，但本轮结束后必须阻止下一轮。异步 Pipeline 还要防止迟到的 Kernel Completion 把已释放 Block 再次写成有效状态。

取消回收延迟本身应该成为指标，特别是在长输出和工具调用场景中。

## 24. 可观测性要把“正在增长”展示出来

只看当前 `kv_cache_usage_pct`，无法判断水位是在下降还是几十轮后必然撞满。需要同时记录：

```text
active_sequences by priority/tenant
current_kv_tokens / blocks
reserved_growth_blocks
kv_growth_tokens_per_second
low/high/emergency watermark state
waiting_prefill_tokens
prefill_yield_count / wait age
preemption by recompute/offload/migrate
recomputed_tokens / transferred_bytes
admission_reject reason
cancel_to_blocks_freed_ms
TTFT / TPOT / E2E by ISL-OSL bucket
```

Trace 应能回答一条请求为什么暂停、何时恢复、Block 在 GPU/CPU/远端哪个层级，以及最终结束原因。

## 25. 长请求基准不能只使用一条 1M Prompt

单请求最大上下文测试只证明“能算”，不能证明“能稳定承载”。验证矩阵至少包含：

1. **短输入短输出基线**；
2. **长输入短输出**，观察 TTFT 与 Chunk 公平；
3. **短输入长输出**，观察 KV 增长与 TPOT；
4. **长输入长输出**，验证端到端最坏资源；
5. **少量长请求混入大量短请求**；
6. **长请求 Burst 集中到单租户**；
7. **Prefix 高命中但目标 Worker 高水位**；
8. **P/D Handoff 时 D 无 Credit**；
9. **Swap/Recompute/Migrate 交叉**；
10. **取消、Deadline 和 Worker Failure**。

每组都要比较平均值和分位数，并检查低优先级请求最终是否完成。吞吐上升但 P99 或饥饿恶化，不能称为治理成功。

## 26. 正确性测试要覆盖暂停、恢复与释放

长请求路径有大量状态转换，适合做不变量检查：

- `computed_tokens` 单调增加，除非显式进入重算状态；
- 每个逻辑 KV Block 只有合法 Owner/Refcount；
- Offload 成功前不能释放唯一 GPU 副本；
- Migrate Commit 后旧 Worker 不得继续生成；
- Position、RoPE、Sampling RNG 与 Grammar 状态在恢复后连续；
- Cancel/Finish 后 Block 最终回到基线；
- 报告 `stop` 的结果确实正常闭合；
- 达到 Exhausted 时 Fail Closed，不覆盖仍在使用的 Block。

还要做长时间混沌测试：持续混合 Add、Yield、Resume、Preempt、Cancel、Migrate 和 Finish，观察是否存在缓慢 KV 泄漏。

## 27. 一条可落地的调度顺序

可以把每轮决策写成分层过程：

```text
1. 收集已完成/取消请求并释放 KV
2. 更新 Active Request 的进度、增长预测与 Deadline
3. 评估本地水位和全局 Router 状态
4. Critical 时先执行降压，不接纳新增长
5. 选择紧迫 Decode，保证逐 Token 进度
6. 为可完成或高价值 Prefill 分配 Chunk/Time/KV Budget
7. 用剩余容量填充非紧迫 Decode/Prefill
8. 对暂停请求执行 Aging，并记录恢复资格
9. 提交 Block Allocation 与执行计划
10. Kernel 完成后原子更新进度与 Ownership
```

顺序可以按引擎实现调整，但“回收在分配之前”“高水位先止压”“Allocation 与执行原子对齐”是重要的不变量。

## 28. 与现有技术文章的关系

这篇文章并不替代已有组件：

```text
Continuous Batching → 请求怎样逐轮加入/离开 Batch
Chunked Prefill     → 长输入怎样切成可调度片段
FairBatching        → TTFT/TPOT 时间余量怎样形成 Batch
PagedAttention      → KV Block 怎样按需分配
Mooncake/NIXL       → KV 怎样跨内存层和 Worker 移动
本文                → 长输入与长输出如何形成统一保护闭环
```

真正的生产系统需要这些层共享同一份 Sequence State、KV Identity 和 SLO 语义。各自优化一个局部指标，却不交换状态，最终仍会在边界处失效。

## 29. 结语

长请求治理不是简单地限制最大 Token，也不是把长 Prompt 切块之后就结束。长输入的核心矛盾是一次重计算对其他请求的阻塞，长输出的核心矛盾则是状态在时间和容量上的持续驻留。

Chunked Prefill、Quota 与 Aging 让长输入在 Chunk 边界公平让出；动态 Growth Reserve、P/D Credit 和分级水位让系统在 KV 真正耗尽前停止加压；Recompute、Offload 与 Migrate 提供不同成本的恢复手段；明确的 Finish Reason 又保证容量保护不会伪装成正常答案。

当 Token Budget、Time Budget 和 KV Budget 同时进入调度器，并且 Router 能及时看到 Engine 的水位和增长趋势时，少量长请求才不会把局部压力放大成全局排队。系统也不需要在“让长请求霸占资源”和“粗暴杀掉长请求”之间二选一，而是可以用可测量、可恢复、可审计的方式管理它们的完整生命周期。

## 参考资料

- [快手万擎大模型推理成本和性能优化实践](https://zhuanlan.zhihu.com/p/2067652898524345525)
- [SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills](https://arxiv.org/abs/2308.16369)
- [Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve](https://www.usenix.org/conference/osdi24/presentation/agrawal)
- [FairBatching: Fairness-Aware Batch Formation for LLM Inference](https://arxiv.org/html/2510.14392)
- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- [vLLM Optimization and Tuning](https://docs.vllm.ai/en/latest/configuration/optimization/)
- [TensorRT-LLM KV Cache System](https://nvidia.github.io/TensorRT-LLM/features/kvcache.html)
