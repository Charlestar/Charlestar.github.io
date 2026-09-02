---
layout: post
title: "Program-Aware Serving：Agent 等待工具时，GPU 状态该放在哪里"
subtitle: "从 Call 依赖、PLAS/ATLAS 调度到 KV Preserve、Swap 与 Recompute，理解 Agent 工作流的推理数据面"
date: 2026-08-29 09:00:00 +0800
last_modified_at: 2026-09-03
author: iStar
catalog: true
series: model-serving-agents
series_order: 40
technology_year: 2024
mathjax: true
tags: [LLM推理, 推理调度, KV Cache]
---

传统 Chat Completion 把一次 HTTP 请求视为完整工作：Prompt 进入模型，Token 连续生成，请求结束后释放状态。Agent 工作流却会把一个用户任务拆成许多相互依赖的 LLM Call：模型决定调用搜索，等待网络结果，再把结果加入上下文继续推理；也可能并行派生多个子任务，最后汇总、反思和重试。

```text
user task
  → LLM call A
  → tool search ── wait ──► result
  → LLM call B
  ├─ LLM call C1 ─┐
  └─ LLM call C2 ─┴─► LLM call D
  → final answer
```

如果 Serving 层只看到一串互不相关的 API 请求，它会错过两个重要事实：

1. Call B、C、D 的完成时间共同决定同一个 Program 的端到端延迟；
2. 工具等待期间，前一轮生成的 KV Cache 暂时不计算，却可能很快再次使用。

“把单次请求跑得更快”不一定让整个 Agent 更快。一个已经执行了几十次 Call 的长 Program，可能不断把后续 Call 插到队列前面，让刚到达的短 Program 长期等待；另一边，如果每次工具调用都销毁 KV，恢复后要重算全部历史；若所有等待任务都把 KV 固定在 GPU，又会把显存留给当前无法运行的 Program。

Program-Aware Serving 要做的，是让推理数据面认识到 Call 的上层生命周期：它不需要理解业务答案，却要知道 Program Identity、依赖关系、累计服务量、暂停原因和可恢复状态。本文从 Call 与 Program 的区别开始，结合 Autellix 的程序级调度和 InferCept 的中断式 KV 管理，构建一条从 Agent Runtime 到 GPU Scheduler 的完整链路。

## 1. 先区分 Request、Call、Turn、Thread 与 Program

这些术语在不同框架里经常混用，可以先给出本文使用的定义：

| 概念 | 含义 |
| --- | --- |
| Request | 一次进入推理 API 的传输单元 |
| LLM Call | 一次具有输入、采样参数与输出的模型调用 |
| Turn | 用户、模型或工具在会话中的一次交互 |
| Thread | Program 中可顺序执行的一条依赖链 |
| Program | 为完成同一用户目标而形成的动态调用图 |

单线程 ReAct Agent 可以在一个 Program 内顺序执行多次 Call；Map-Reduce 或多 Agent 协作会产生多个并行 Thread；工具、代码执行和人工输入形成 External Interrupt。

Program 的完整 DAG 往往事先未知。模型输出决定是否调用工具、生成几个分支、何时结束，因此 Scheduler 不能要求提交时提供精确剩余 Call 数和输出长度。

## 2. 为什么 Call 级优化不等于 Program 级优化

假设两个 Program：

```text
Program A: A1 → A2 → A3 → A4
Program B: B1
```

如果 A1 先到达，A2 在 A1 完成后又被当成“新请求”放到最高优先级，A 的后续 Call 可能连续占用资源，B 一直无法完成。

从 Call 角度，每次排队都可能合理；从 Program 角度，已经获得大量 GPU 服务的 A 挡住了只需一次 Call 的 B，造成 Program-level Head-of-line Blocking。

优化目标也不同：

- Call Latency：某次模型调用多久完成；
- Program Latency：用户任务从开始到最终结果多久完成；
- Makespan：并行 Program 所有必要分支何时汇合完成；
- Program Throughput：单位时间完成多少个完整任务。

如果基准只统计 Call tokens/s，系统可能很快地产生许多中间结果，却没有更快完成任何用户任务。

## 3. Program 是运行时展开的动态 DAG

Autellix 将 Agentic Program 抽象为包含 LLM Call 与 External Interrupt 的动态 DAG：

```text
node:
  LLM_CALL | TOOL | CODE | HUMAN | JOIN

edge:
  data dependency
  control dependency
```

单线程 Program 是一条逐步展开的路径；多线程 Program 会 Fork、Join，某些分支还可能提前取消。

Serving 层不一定要拿到完整 Prompt 内容或工具结果，只要收到足够的调度元数据：

```json
{
  "program_id": "p-123",
  "thread_id": "t-2",
  "call_id": "c-8",
  "parent_call_ids": ["c-3"],
  "priority_class": "interactive",
  "deadline": "...",
  "resume_from": "state-7"
}
```

这些字段必须由可信 Agent Runtime 生成。不能让模型正文伪造 `priority=critical` 或冒充其他租户的 Program。

## 4. 程序级调度需要累计“已经获得多少服务”

不知道 Program 最终会运行多久时，无法直接使用最短剩余作业优先。Autellix 的 PLAS（Program-Level Attained Service）采用非 Clairvoyant 思路：按 Program 已经累计获得的服务量排序。

令 Program \(p\) 已完成和正在执行 Call 的累计服务为：

$$
A_p(t)=\sum_{c\in p} service_c(t)
$$

Service 可以用模型执行时间、Decode Steps 或经过校准的 GPU Work 表示。PLAS 优先选择 \(A_p\) 较小的 Program：

$$
priority(c)=-A_{program(c)}
$$

直觉是：已经消耗很多服务的长 Program 不应让每个新 Call 都获得“全新请求”的最高优先级；累计服务较少的 Program 更可能较快完成。

PLAS 不需要事先知道剩余长度，但不是在预测绝对最短 Program。它是一种减少程序级阻塞的在线近似，仍需结合业务优先级与防饥饿机制。

## 5. 为什么普通 MLFQ 仍可能看不见 Program

Multi-Level Feedback Queue 会降低长 Call 的优先级，让短 Call 穿过，从而缓解 Call-level Head-of-line Blocking。

问题在于工具返回后产生的 A2 可能被视为一条全新短请求，再次进入最高优先级队列。A1 的历史服务没有传递给 A2：

```text
MLFQ sees: A2 is new
Program scheduler sees: A already consumed A1 service
```

因此 Program Identity 必须跨越 Call 边界。否则再精细的单 Call Age、Token Count 或 Queue Level，都无法知道一组请求共同属于已经运行很久的任务。

## 6. 多线程 Program 还要认识 Critical Path

Map-Reduce、Tree Search 和多 Agent 协作会产生并行 Thread：

```text
          ┌─ C1 ─ C2 ─┐
A ─ B ───┤            ├─ E
          └─ D1 ──────┘
```

若 C 分支远慢于 D，继续优先执行已经不阻塞 Join 的 D1 后续工作，可能不能缩短 Makespan。

Autellix 的 ATLAS（Adaptive Thread-Level Attained Service）把累计服务扩展到 Thread，并利用 Program 内 Thread 的最大累计服务近似 Critical Path。它在没有完整未来 DAG 的情况下，让阻塞 Program 推进的关键 Call 获得更合理的顺序。

实现需要 Agent Runtime 报告 Fork/Join 和 Thread 关系。只给所有并行 Call 相同 `program_id`，却没有依赖结构，Scheduler 仍无法判断哪个分支卡住了 Join。

## 7. Program Scheduler 不应直接猜业务 DAG

从 Prompt 文本或工具名推断依赖关系很脆弱：

- 同名工具可能服务完全不同流程；
- 模型可能动态取消或新增分支；
- 日志里的自然语言不是事务提交记录；
- 错误推断会优先执行本不该运行的 Call。

更稳妥的分工是：

```text
Agent Runtime:
  定义 program/thread/call identity
  维护依赖、工具副作用与 checkpoint

Serving Control Plane:
  维护累计服务、队列、placement 与 SLO

Inference Engine:
  执行 call、管理 token/KV 与抢占
```

上层提供语义，下层提供资源事实。二者通过版本化事件连接，而不是让任一层猜测另一层的内部状态。

## 8. 工具调用为什么形成一次 Interception

模型生成 Tool Call 后，正常 Decode 暂停：

```text
LLM decode
  → emit tool call
  → wait for network / code / human
  → append result
  → resume LLM
```

等待时间可能从毫秒到分钟，甚至永远不返回。等待期间 GPU 不需要为该 Program 计算 Token，但它的上下文很可能在恢复时继续使用。

这产生了一个资源问题：已建立的 KV Cache 应该留在 GPU、换到其他层级，还是释放并在恢复时重算？

InferCept 把三种基础策略称为 Preserve、Swap 与 Discard，并通过成本模型动态选择。

## 9. Preserve：用显存换恢复延迟

Preserve 在工具等待期间保留 GPU KV：

```text
interrupt
→ mark sequence suspended
→ keep KV blocks pinned
→ tool returns
→ append tool result and resume
```

优点是恢复快，不需要传输或重算。代价是暂时不可执行的请求仍占用 HBM，降低其他请求的并发。

它适合：

- 等待时间很短；
- 上下文很长，重算成本高；
- 当前 KV 压力低；
- Program Deadline 紧；
- 恢复概率高。

“工具通常很快”不能作为唯一判断。P99 网络延迟、人类确认和第三方限流可能让少量 Preserve 长时间占住显存。

## 10. Discard/Recompute：用计算换容量

Discard 释放 KV，只保存 Token/Checkpoint；工具返回后重新 Prefill 历史：

```text
interrupt → free KV
tool returns → rebuild prompt → recompute KV → continue
```

优点是等待期间不占 KV。代价与历史长度成正比，还会把恢复流量重新送入 Prefill Queue。

它适合：

- 上下文较短；
- 工具等待很长或返回概率低；
- GPU KV 已高压；
- Prefix Cache/外部 KV Store 能复用大部分历史；
- Prefill Capacity 充足。

恢复时必须使用已提交 Token 与一致的 Chat Template、Tokenizer、Model/Adapter Revision。重新让模型生成工具调用前的文本，可能得到不同结果；正确做法是重放确定的历史 Token，只重算 K/V。

## 11. Swap/Offload：用传输换计算

Swap 将 KV 从 HBM 移到 CPU DRAM、SSD 或远端 KV Store，工具返回后再 Onboard：

```text
GPU KV ── offload ──► host / remote
                           │
tool returns               │
GPU KV ◄── onboard ────────┘
```

它保留计算结果，却消耗 PCIe、网络和下层容量。若一次性传输巨大 KV，前台 Decode 可能因带宽争用停顿。

InferCept 的一种关键优化是把 Swap 按层或按预算流水化，与前台模型计算重叠，并限制每轮传输量不超过链路可承受的 Swap Budget。

Swap 不是 Preserve 与 Discard 之间自动最优的折中。短 Context 可能重算更快，超短等待可能保留更快，慢网络下远端恢复甚至比重新 Prefill 更贵。

## 12. 三种策略可以放进同一个成本模型

对一次中断 \(i\)，可估算：

$$
C_{preserve}
=M_{KV,i}\cdot T_{wait,i}\cdot price_{HBM}
$$

$$
C_{discard}
=T_{recompute,i}\cdot price_{GPU}
$$

$$
C_{swap}
=T_{offload,i}+T_{onboard,i}
+C_{contention,i}
$$

真实目标不只是金钱成本，还可以包含 Deadline 违约与其他请求被阻塞的机会成本。

关键输入是预测等待时间，但工具延迟通常长尾且不稳定。成本模型应使用分布或置信区间，并允许到达 TTL 后重新评估：开始先 Preserve，等待变长后再 Offload；而不是在中断发生时做一次不可更改的决定。

## 13. 状态迁移必须有提交边界

从 GPU Offload 到 CPU 时不能先释放唯一 GPU 副本，再希望传输成功。正确流程类似事务：

```text
allocate target
→ copy KV + metadata
→ verify completion/checksum
→ atomically publish new owner
→ release old owner
```

Onboard、跨 Worker Migration 同理。元数据至少包含：

- Model、Adapter、Tokenizer 与 KV Layout Revision；
- Layer/KV Group/TP Rank Ownership；
- Logical Position 与 Block Table；
- Dtype、Scale、Stride 与 Checksum；
- Program/Thread/Call 与 State Epoch；
- 当前 Owner、Lease 与恢复状态。

只有 Bytes 没有身份，无法安全恢复。

## 14. 工具结果到达时不一定应该立即抢占别人

工具返回意味着 Program 变为 Runnable，不代表它必须立刻运行。如果所有恢复请求都插队，频繁使用快速工具的长 Program 会压住普通请求。

恢复 Call 应回到 Program-Aware Scheduler：

```text
tool result arrives
→ validate program epoch and dependency
→ materialize/resume state
→ enqueue with program attained service
→ scheduler decides execution order
```

高优先级 Deadline 可以提高紧迫度，但仍应受租户配额和显存准入约束。恢复事件只是状态转换，不是无限优先权。

## 15. Program Identity 如何跨 API 与 Engine 传播

一条可追踪链路可以使用：

```text
tenant_id
program_id
thread_id
call_id
attempt_id
state_epoch
parent_ids
```

`attempt_id` 区分同一逻辑 Call 的重试，`state_epoch` 防止旧工具结果或迟到 Worker Completion 覆盖新状态。

这些字段需要进入：

- Agent Runtime Checkpoint；
- API Header/Request Metadata；
- Global Scheduler Queue；
- Engine Sequence State；
- KV Cache/Offload Manifest；
- Trace 与计费记录。

跨层只传 `request_id`，每轮新请求都换 ID，就无法聚合 Program 的累计服务与端到端延迟。

## 16. 重试不能让累计服务归零

如果失败后创建一个新 `program_id`，长 Program 可以通过重试反复获得最高优先级，也会丢失公平性与配额历史。

合理语义是：

```text
logical call c-8
  attempt 1: worker failure
  attempt 2: resume/recompute

program attained service
  includes useful and policy-defined wasted service
```

是否把失败计算全部计入优先级需要策略，但至少不能无条件归零。否则错误频繁的 Program 反而获得更多资源。

外部工具副作用还需要独立幂等键。Serving 层恢复 LLM Call，不应自动重复一笔已经成功的支付、删除或消息发送。

## 17. 并行分支怎样进入跨 Engine 调度

多线程 Program 的 Calls 可能被分散到多个 Engine，以获得并行性和 Prefix Locality。Placement 可以同时考虑：

$$
Cost(engine)=
queue\_delay
+uncached\_prefill
+kv\_transfer
+critical\_path\_delay
+memory\_pressure
$$

若所有分支都放在同一 Engine，可能形成局部拥塞；完全打散又会丢失公共 Prefix，并增加 Join 前的数据与状态协调。

全局 BFD（Best-Fit Decreasing）一类装箱可以作为启发式：按预测资源需求把 Program/Thread 放到最合适的容量槽。但 Agent DAG 动态展开，预测随时会变，Placement 必须支持重新评估、迁移和保守回退。

算法名不能替代成本模型。BFD 优化的对象是 GPU Time、KV Bytes、Prefix Locality 还是 Deadline，需要明确。

## 18. 依赖关系允许哪些 Call 提前执行

Scheduler 只能运行依赖已经满足的 Node：

```text
READY(call)
= all parents committed
  and program epoch valid
  and not cancelled
  and required tool outputs available
```

模型推测未来会需要某工具，不代表可以提前产生带副作用的调用。对于纯读取、可幂等的工具，上层 Runtime 可以做 Speculative Tool Execution，但这属于业务语义决策，不应由 GPU Scheduler 擅自执行。

Join Node 还要定义失败策略：任一分支失败即取消其他分支，还是允许部分结果汇总？Serving 层只按上层提交的状态执行和释放资源。

## 19. Program SLO 需要从终点反推 Call 紧迫度

若 Program Deadline 为 \(D_p\)，当前时间为 \(t\)，关键路径预计剩余服务为 \(R_p\)，可以定义 Slack：

$$
slack_p=D_p-t-R_p
$$

负 Slack 表示即使立刻执行也可能违约。Scheduler 可以结合 PLAS 与 Slack：

```text
base fairness: lower attained service first
urgency boost: smaller slack first
hard policy: tenant / priority / safety constraints
```

这比把每个 Call 都设置同一个静态 Priority 更贴近 Program 终点。

但 \(R_p\) 很难准确预测。模型应输出区间与置信度，预测漂移时退回 PLAS/ATLAS + Aging，而不是让错误 Deadline 模型饿死其他任务。

## 20. “最短 Program 优先”仍需要防饥饿

持续到来的短 Program 可能让长 Program 一直排队。公平策略需要 Aging 或最低服务保证：

$$
score_p
=w_1\cdot attained_p
-w_2\cdot waiting_p
-w_3\cdot urgency_p
$$

还可以按租户设置 Weighted Fair Share，防止一个租户用大量短 Program 占满所有完成槽。

要同时观察：

- Program Completion Time 分布；
- Slowdown（完成时间/独占执行时间）；
- 最大等待时间；
- 各租户获得的 GPU Service；
- 被抢占、Offload 与重算次数。

只优化平均 Program Latency，可能牺牲少量长 Program 到不可接受的程度。

## 21. Tool Wait 是显存回收窗口，也是预取窗口

中断开始后，系统有机会把 KV Offload；预计工具即将返回时，又可以提前 Prefetch：

```text
tool wait begins
→ decide preserve/offload/discard
→ observe elapsed time and tool progress
→ prefetch before predicted completion
→ tool result arrives
→ enqueue runnable call
```

若预取过早，KV 又在 GPU 空等；过晚则 Program 等待 Onboard。可以根据工具类型、历史延迟分布和实时进度选择 Prefetch Lead Time。

但是工具服务并不总暴露进度，预测也可能被第三方抖动打破。Prefetch 必须受独立带宽与 KV 水位预算约束，不能为了一个可能恢复的 Program 挤掉正在 Decode 的请求。

## 22. TTL 不应只有一个全局常数

简单策略是中断后保留 KV \(T\) 秒，超时再 Offload/Discard。统一 TTL 无法适配：

- 1ms 计算器；
- 数百毫秒检索；
- 数十秒图片生成；
- 几分钟人工审批；
- 大小不同的上下文与不同 SLO。

更合理的 TTL 取决于：

$$
TTL_i=f(tool\_class,
context\_bytes,
recompute\_cost,
queue\_pressure,
deadline,
latency\_distribution)
$$

系统还要在等待中动态更新。若原本预计快速返回的 API 已超过 P99，继续 Preserve 的收益会迅速下降。

## 23. Prefix Cache 与 Suspended KV 的语义不同

Prefix Cache 是可供一个或多个未来请求复用的已完成前缀；Suspended KV 属于一个未完成 Program 的恢复状态。

两者可能共享相同 Block Store，却有不同生命周期：

| 状态 | Owner | 可否逐出 | 恢复语义 |
| --- | --- | --- | --- |
| Reusable Prefix | Cache/Tenant | 可重算后逐出 | Miss 只增加 Prefill |
| Suspended Program KV | Program Attempt | 按策略迁移/重算 | 错用会破坏继续生成 |
| Active Decode KV | Running Sequence | 需先抢占/提交 | 下一轮立即读取 |

Suspended State 还绑定 Sampling、Stop、Grammar 与工具调用位置，不能因为 Token Prefix 相同就让另一个 Program 获得完整私有状态。

跨租户 Prefix 共享也要遵守隐私策略，不能让 Cache Hit 延迟成为敏感工作流的侧信道。

## 24. Structured Output 与 Tool Call 状态也要一起暂停

工具调用可能由 Grammar-Constrained Decoding 生成。暂停点除了 KV，还可能包含：

- Grammar Matcher State；
- 已输出但尚未提交的 Token；
- UTF-8/JSON 流式边界；
- Sampling RNG 与 Speculative Decode State；
- Stop Sequence 的部分匹配；
- Tool Name 与 Arguments 的解析进度。

理想情况下，只在完整 Tool Call 已提交后进入 External Interrupt，减少半结构恢复的复杂度。若支持生成中途暂停，以上状态必须序列化或通过已提交 Token 确定性重放。

只恢复 KV 而让 Grammar 从初始状态开始，下一 Token Mask 可能完全错误。

## 25. 取消 Program 要沿 DAG 传播

用户取消最终任务后，仍在运行的分支、等待中的工具和 Offload KV 都要收回：

```text
cancel program epoch
→ stop scheduling ready calls
→ cancel safe external tools
→ mark running calls cancelled
→ discard late tool results
→ release active/suspended KV
→ close joins and emit final state
```

取消不能只发给当前 HTTP Call。否则后台分支仍会继续消耗 GPU，工具结果回来后还可能创建新的 Call。

对已经产生外部副作用的工具，取消只阻止后续工作，不能假装副作用不存在。Agent Checkpoint 必须保存其提交记录。

## 26. Worker 故障时恢复的是 Program 的哪个边界

可以把可恢复状态分三层：

```text
Program checkpoint：DAG、工具结果、业务提交记录
Call checkpoint：已提交 token、sampling/grammar 状态
Engine state：KV blocks、runner slot、kernel in-flight
```

GPU Worker 崩溃后，Engine State 通常丢失；若有外部 KV 副本，可以 Onboard，否则从 Call Token 重算。Program Runtime 再决定 Call 是否可重试，以及哪些工具不能重复。

State Epoch 防止旧 Worker 复活后提交过期 Token。恢复后的累计服务也要延续，不能让故障 Program 重获零 Attained Service。

## 27. 可观测性要同时看 Call 和 Program

建议使用一组可关联事件：

```text
PROGRAM_STARTED
CALL_READY / CALL_SCHEDULED / CALL_COMPLETED
THREAD_FORKED / THREAD_JOINED
INTERRUPT_STARTED / INTERRUPT_COMPLETED
KV_PRESERVED / KV_OFFLOADED / KV_DISCARDED
KV_PREFETCH_STARTED / RESUME_READY
CALL_PREEMPTED / CALL_RETRIED
PROGRAM_COMPLETED / PROGRAM_CANCELLED
```

指标至少包括：

- Program E2E Latency、Makespan 与完成率；
- 每个 Program 的 Calls、Tokens 与 GPU Service；
- Call Queue/Execution/Tool Wait/Resume 分解；
- PLAS/ATLAS Queue Level 与最大等待；
- Preserve/Swap/Discard 选择比例；
- Suspended KV Bytes 与各层驻留时间；
- Offload/Onboard Bytes、延迟与前台干扰；
- 重算 Token、TTL 预测误差；
- 租户公平与 SLO 达成。

只有 Program Trace 才能解释“每次 LLM Call 都很快，用户为什么仍等了两分钟”。

## 28. 评测必须使用真实 DAG 与工具延迟

把每个 Program 简化成固定三次相同长度 Call，会掩盖动态分支与中断。

测试集应覆盖：

| Program | 关键压力 |
| --- | --- |
| 单线程 ReAct | 多次 Call/Tool 交替 |
| 长短 Program 混合 | Program-level HoL Blocking |
| Map-Reduce | Fork/Join 与 Critical Path |
| Tree Search | 动态宽度、取消分支 |
| 人工审批 | 极长且高方差 Interrupt |
| 快速计算工具 | Preserve 的收益 |
| 大上下文检索 Agent | Swap/Recompute 成本 |
| 多租户 Burst | 公平与 Priority |

工具延迟要使用分布和故障，而不是固定 Sleep。还要分别比较：

- FCFS/MLFQ 与 PLAS/ATLAS；
- Discard、Always Preserve、Always Swap 与动态策略；
- 冷/热 Prefix、不同 KV 水位和带宽；
- 单 Engine 与多 Engine Placement。

最终报告 Program Completion Time、Throughput、GPU Goodput、KV Bytes-time 与 SLO，而不只报告 Call Token/s。

## 29. 正确性不变量比平均加速更重要

Program-Aware Serving 至少要验证：

- 同一逻辑 Call 只有一个有效 Commit；
- 未满足依赖的 Call 不能执行；
- 迟到工具结果不能跨 State Epoch 生效；
- Resume 后 Token、Position、RNG 与 Grammar 连续；
- Offload/Migrate 只有在目标验证后才切换 Owner；
- Program Cancel 后不再创建新 Call；
- 外部副作用使用业务幂等键；
- 租户 Priority 与 Program Metadata 不可伪造；
- 所有 Active/Suspended KV 在 Finish/Cancel 后最终释放；
- 调度优化不改变模型采样分布或工具参数语义。

混沌测试应随机插入 Tool Timeout、Worker Failure、重复结果、网络分区、Program Cancel 和 KV Store Eviction，再检查状态机是否收敛。

## 30. 一条可落地的实现顺序

### 第一步：先建立跨 Call 的 Identity 与 Trace

让 Agent Runtime 稳定传递 Program/Thread/Call/Attempt/Epoch，先观察 Program E2E 和工具等待分布，不立即改变调度。

### 第二步：实现可恢复的 Interception

从完整 Tool Call 边界暂停；先支持 Preserve 与 Discard，验证 Token/KV 重放正确，再加入受预算控制的 Swap。

### 第三步：引入 Program 累计服务

在单 Engine 上比较 FCFS、MLFQ 与 PLAS，加入 Aging、租户权重和 Deadline Guard；验证不会让长 Program 饥饿。

### 第四步：扩展到 DAG 与 Critical Path

让 Runtime 报告 Fork/Join，加入 ATLAS 类 Thread 统计；取消无效分支并验证 Join 语义。

### 第五步：建立多 Engine Placement

把 Prefix Locality、Queue、KV 水位与 Program Critical Path 放进路由；支持 State Epoch、迁移与故障恢复。

### 第六步：再做预测、Prefetch 与自适应

收集足够工具延迟和 Program Trace 后，预测 TTL/剩余服务；始终保留带资源上限的保守策略。

## 31. 与 Agent Framework 的边界

LangGraph 一类框架解决的是业务图、Checkpoint、Interrupt 与工具节点怎样编排；Program-Aware Serving 解决的是这些节点调用模型时，GPU 队列、KV 与跨 Engine 资源怎样安排。

```text
Agent layer:
  what must happen next
  what state is committed
  which side effects are safe

Serving layer:
  when/where an LLM call runs
  where resumable KV lives
  how GPU service is shared
```

两层都需要暂停/恢复，却不能共用一个模糊的“checkpoint”概念。Agent Checkpoint 保存业务事实，KV Checkpoint 保存可重算的模型执行状态。前者决定正确性，后者主要决定性能。

## 32. 结语

Agent 将 LLM Serving 的调度单位从独立请求扩展成动态 Program。Call 之间有依赖，工具调用制造不可预测的空窗，分支完成时间共同决定用户何时得到最终结果。只优化单 Call 的 Batch 和 Kernel，不足以解决程序级阻塞和等待期间的 KV 浪费。

PLAS/ATLAS 提供了一种不预知完整 DAG 也能利用累计服务和关键路径的方法；Preserve、Discard 与 Swap 则把工具中断期间的状态管理转化为显存、计算和传输之间的成本选择。二者通过可信的 Program Identity、State Epoch 和结构化事件连接，才形成完整的数据面。

最终目标不是让 Serving 层理解 Agent 的业务意图，而是给它刚好足够的生命周期信息：哪些 Call 属于同一任务、哪个分支阻塞终点、当前为什么暂停、恢复需要什么状态，以及这个 Program 已经获得多少 GPU 服务。

当这些信息成为一等状态后，工具等待不再只是一个断开的 HTTP 请求，KV 也不再只能在“永远留在显存”和“全部丢掉重算”之间选择。整个 Agent Program 才能作为可调度、可恢复、可观测的系统对象运行。

## 参考资料

- [快手万擎大模型推理成本和性能优化实践](https://zhuanlan.zhihu.com/p/2067652898524345525)
- [Autellix: An Efficient Serving Engine for LLM Agents as General Programs](https://arxiv.org/abs/2502.13965)
- [InferCept: Efficient Intercept Support for Augmented Large Language Model Inference](https://proceedings.mlr.press/v235/abhyankar24a.html)
- [Parrot: Efficient Serving of LLM-based Applications with Semantic Variable](https://www.usenix.org/conference/osdi24/presentation/lin-chaofan)
- [CacheTTL: Efficient and Robust Multi-Turn LLM Agent Scheduling with KV Cache Time-to-Live](https://arxiv.org/abs/2511.02230)
- [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104)
