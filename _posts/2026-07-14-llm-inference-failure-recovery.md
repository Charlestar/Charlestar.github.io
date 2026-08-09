---
layout: post
title: "分布式 LLM 推理故障恢复：一条流式请求怎样活下来"
subtitle: "从健康检测、请求迁移到 KV 状态、优雅下线与过载保护"
date: 2026-07-14 09:00:00 +0800
last_modified_at: 2026-08-09
author: iStar
catalog: true
series: distributed-inference
series_order: 60
technology_year: 2025
mathjax: true
tags: [AI Infra, 故障恢复, 分布式推理, KV Cache, Kubernetes]
---

一条流式生成请求已经输出了 127 个 token，此时负责 decode 的 GPU 进程退出。系统应该怎样处理？

最简单的答案是让编排系统重启进程，但这只能恢复一份计算资源，不能自动恢复这条请求。请求已经经过路由，可能在 Prefill worker 上建立过 KV Cache，又把状态传给了 Decode worker；客户端还收到了部分响应。若没有显式设计，重启后的进程既不知道这条请求，也不知道客户端已经看到哪里。

故障恢复因此不是“Pod 能否重新拉起”，而是三个不同层次的问题：

1. **服务恢复**：健康实例能否继续接收新请求；
2. **请求恢复**：执行中的请求能否在其他实例继续；
3. **语义恢复**：继续后的输出是否没有缺失、重复或不可接受的漂移。

这三个目标的代价逐级增加。一个系统可以很快恢复新流量，却中断所有在途请求；也可以重放请求，却无法保证随机采样得到逐 token 完全相同的结果。设计恢复机制前，必须先说清楚自己承诺的是哪一层。

## 先看一条请求携带了哪些状态

把一次生成简化成“prompt 进去、token 出来”，很容易误以为只要保存文本就能恢复。实际执行状态至少包括：

```text
请求语义
├── prompt token
├── 已提交的输出 token
├── 模型、版本、Adapter 与生成参数
├── 随机数状态与采样历史
├── 结构化输出的语法状态
└── 工具调用及其外部副作用

执行状态
├── KV Cache 与 block table
├── scheduler 中的队列位置
├── speculative decoding 的草稿状态
├── tensor / pipeline / expert parallel collective
└── 正在进行的 KV transfer

交付状态
├── 客户端连接
├── 已发送的 stream frame
├── 客户端已确认的连续序号
└── 取消、超时与 deadline
```

恢复设计的第一步不是把所有状态都持久化，而是判断哪些状态是事实、哪些可以重算、哪些一旦丢失就无法透明恢复。

| 状态 | 能否重建 | 典型策略 |
| --- | --- | --- |
| prompt 与已确认输出 token | 通常可以 | 作为恢复日志，重新 Prefill |
| KV Cache、block table | 可以，代价可能很高 | 命中则复用，失效则重算 |
| 权重的本地副本、编译缓存 | 可以 | 从共享存储或镜像重新加载 |
| 随机数与采样状态 | 只有显式保存才可靠 | 序列化 RNG 与 sampler 状态，或接受输出漂移 |
| grammar / FSM 状态 | 依赖实现 | 保存状态，或从完整 token 前缀确定性重放 |
| beam、multi-choice 状态 | 很难只靠一条前缀恢复 | 保存完整候选树，或明确不支持迁移 |
| 已执行的外部工具调用 | 不能靠模型重算撤销 | 幂等键、事务日志与补偿操作 |
| 已写到客户端但未确认的数据 | 无法单方面判断 | 序号、确认点和去重协议 |

这个分类给出一个关键原则：**KV Cache 是加速计算的派生状态，token 序列才更接近请求的可恢复事实。** KV 丢失会让恢复变慢；错误复用一份不属于该请求或版本的 KV，却会直接破坏输出正确性。

## 恢复目标需要可度量

传统存储系统常用 RTO 和 RPO 描述恢复。在流式推理中可以做相似映射：

- **服务 RTO**：故障发生后，多久能重新接受足够的新请求；
- **请求 RTO**：某条受影响请求多久恢复输出；
- **token RPO**：最多允许多少已生成 token 被重算、丢失或重复交付；
- **语义连续性**：恢复后是否必须与原执行逐 token 相同。

如果产品只承诺“失败后可重试”，请求 RTO 就等于客户端重新发起请求的时间，系统实现最简单。如果承诺流式连接尽量不断开，frontend 就必须在 worker 之外保存恢复所需的请求状态，并能够把执行迁移到其他 worker。

逐 token 完全一致则要求更高。即使 prompt 和输出前缀相同，以下因素也可能改变后续 token：

- RNG 状态没有恢复；
- 批处理组成变化导致浮点归约顺序变化；
- speculative decoding 的 draft 路径改变；
- 模型副本、Adapter 或 kernel 版本不同；
- grammar processor 没有恢复到同一状态。

因此，接口文档应区分“请求可以继续”和“输出位级确定”。后者通常需要确定性执行、版本固定和更多状态复制，不能从“支持迁移”自然推导出来。

## 故障域决定恢复边界

分布式推理不是一个进程，而是多条相互连接的路径：

```text
Client
  │
  ▼
Gateway / Frontend ── Router / Discovery
  │                         │
  ├──── Prefill worker ─────┤
  │          │ KV transfer  │
  └──── Decode worker ──────┘
             │
             ├── TP / PP / EP ranks
             └── KV Store / Metadata

Control Plane: autoscaler / planner / operator
```

不同节点失效，损失的状态和恢复动作不同。

| 故障点 | 对新请求的影响 | 对在途请求的影响 | 主要恢复动作 |
| --- | --- | --- | --- |
| Gateway / frontend | 入口容量下降 | 连接与交付进度可能丢失 | 多副本、会话恢复或客户端重连 |
| Router / discovery | 路由决策受阻 | 已建立的数据路径可能继续 | 本地快照、租约与多副本控制面 |
| Prefill worker | 新 prompt 无法完成 Prefill | 尚未发布的 KV 丢失 | 在其他 P worker 重算 |
| KV transfer | P/D 交接卡住 | D worker 不能安全开始 | 重传、换源或重新 Prefill |
| Decode worker | decode 容量下降 | 流式请求直接受影响 | 从已提交前缀迁移或终止 |
| 任一 TP/PP/EP rank | 整个并行实例不可用 | collective 中的请求通常失败 | 整组摘流、终止 collective、成组重启 |
| KV 元数据 | 缓存感知路由退化 | 可能读到陈旧位置 | epoch 校验、快照重建、回退为 miss |
| Planner / autoscaler | 暂时不能调整容量 | 不应中断现有生成 | 保持最后一份安全配置 |

这张表也说明，不能只统计“GPU worker 可用率”。入口、发现服务、元数据和传输层都可能成为请求恢复链路上的单点。

## 健康检查不是一个布尔心跳

进程存在不代表实例可以接流量；HTTP 端口可达也不代表 GPU 能完成推理。至少要区分两类信号：

- **Liveness**：进程是否已进入不可自愈状态，需要被重启；
- **Readiness**：实例此刻是否应出现在可路由集合中。

一个正在优雅下线的 worker 可以仍然 live，但必须立刻 not ready；它要继续完成旧请求，却不能接受新请求。反过来，一个因队列过长暂时拒绝流量的 worker 也未必应该被杀掉。

成熟的检查可以形成三层：

1. 进程与事件循环是否响应；
2. 模型、GPU、通信组和依赖是否就绪；
3. 在空闲或专用通道上执行一条极小的真实推理 canary。

第三层能发现“服务端口正常，但 CUDA context、collective 或模型执行已经失效”的情况。NVIDIA Dynamo 的健康检查也把外部健康端点与最小推理 canary 分开处理，而不是仅检测进程可达性。

检查频率和阈值需要权衡：

- 过慢会延长错误路由时间；
- 过快会让短暂拥塞被误判成死亡；
- canary 与真实流量共用拥塞队列时，过载可能被误判为硬故障；
- 所有探针同一时刻执行，会制造周期性抖动。

可以给探针加入 jitter，区分 timeout、OOM、GPU XID、网络断连和主动 drain，并让 Router 根据故障类型决定“暂时降权”还是“立即摘除”。

## 租约和 epoch 阻止故障实例复活

只靠 worker 主动注销不够。进程崩溃或网络隔离时，它无法发出注销消息，因此服务发现通常需要带 TTL 的租约：worker 持续续租，续租停止后，endpoint 自动从可路由集合中移除。

租约解决“多久认为它死了”，epoch 解决“回来的是不是同一个执行世代”。假设 worker `decode-7` 崩溃后重启，名字与地址可能不变，但它的内存、KV block 和在途请求已经全部消失。新进程必须使用更高的 generation：

```text
endpoint = (worker_id, generation)

old: (decode-7, 41)
new: (decode-7, 42)
```

Router、KV metadata 和 transfer completion 都应携带 generation。任何来自 41 的延迟事件到达时，都必须被 42 拒绝。否则会出现典型的 ABA 问题：控制面看到同一个名字，以为旧状态仍然有效。

generation 最好延伸到完整并行实例，而不是单个 rank。对于一个 8-way tensor parallel worker group：

```text
instance_id = model-A/tp-group-12
instance_generation = 9
rank = 0..7
```

只有 8 个 rank 都完成同一 generation 的 rendezvous、权重加载与 canary，整个 endpoint 才能 ready。

## 计划内下线：先停止路由，再停止进程

滚动升级、缩容和节点维护都属于可预期中断。正确顺序是：

```text
1. endpoint 标记为 not ready / 从发现服务注销
2. Router 停止分配新请求
3. 等待路由缓存与连接池收敛
4. 完成、迁移或取消在途请求
5. 释放 KV lease、通信组和临时资源
6. 在超时边界内退出进程
```

如果先发送终止信号，再等待 Router 发现故障，中间仍会有新请求进入已经无法服务的实例。优雅下线的本质不是“多等几十秒”，而是让数据面状态按正确顺序收敛。

drain 需要一个有限 deadline。无限等待会让升级永远卡在长请求上；立即终止又浪费本可以完成的工作。常见策略是：

- 剩余时间充足的短请求原地完成；
- 可迁移请求转移到健康 worker；
- 超过 drain deadline 的请求取消并返回可识别错误；
- 到达 hard deadline 后进程必须退出。

在 Kubernetes 中，`preStop` 和 `terminationGracePeriodSeconds` 可以承载这段窗口，但应用仍需主动改变 readiness 并执行 drain。PodDisruptionBudget 只限制一部分**自愿中断**的并发数量，不能防止节点掉电、硬件故障等非自愿中断，也不能替代应用层请求恢复。

## 非预期崩溃：新流量恢复与旧请求恢复是两条线

worker 突然退出后，系统应并行执行两件事：

```text
新请求路径：
lease 到期 / 主动故障通知
  → endpoint 摘除
  → Router 选择其他健康实例

在途请求路径：
stream 断连 / backend error
  → frontend 冻结已提交进度
  → 判断请求是否可迁移
  → 在健康 worker 重放或返回终止错误
```

只完成第一条线，服务总可用性看起来可能很好，但所有 worker 重启都会切断一批流式请求。只完成第二条线，却没有及时摘除 endpoint，则迁移请求还可能再次被路由到故障实例。

NVIDIA Dynamo 的请求迁移设计把 migrator 放在 frontend 预处理与 backend 之间。它拦截请求和响应，持续积累已生成 token；初始连接失败或流中断时，可以把请求前缀重新发给健康 worker。同时，失败 worker 会被本地临时抑制，以覆盖服务发现尚未完成传播的短窗口。

这种迁移不是把 GPU 寄存器或 CUDA context 搬到另一张卡，而是**从可重放的语义状态重新执行**。因此恢复会产生额外 Prefill 工作和延迟，也必须设置最大迁移次数，避免同一请求在集群抖动时无限漂移。

## 用“已提交 token 前缀”恢复 decode

设原始 prompt 为 $P$，已经生成的 token 为：

$$
Y = (y_1, y_2, \ldots, y_n)
$$

如果 frontend 已确认前 $k$ 个 token 构成连续、不可回退的交付前缀，那么恢复请求可以把：

$$
P' = P \Vert (y_1, \ldots, y_k)
$$

重新送入健康 worker。新 worker 对 $P'$ 执行 Prefill，重建 KV，然后从位置 $k+1$ 继续 decode。

这里的难点不在拼接 token，而在“$k$ 到底是多少”。需要至少区分：

- worker 已经生成；
- frontend 已经收到；
- frontend 已经写入连接；
- 客户端已经确认。

如果没有客户端确认协议，连接断开的一瞬间可能无法知道最后一个 frame 是否已被用户收到。系统只能在两种语义间选择：

- **at-least-once**：宁可重发边界 token，由客户端按序号去重；
- **at-most-once**：宁可跳过不确定 frame，可能造成缺口。

更稳妥的流协议会给每个 frame 加上稳定的 `request_id` 和单调递增的 `sequence_no`：

```json
{
  "request_id": "req-8f...",
  "sequence_no": 128,
  "token_id": 29871,
  "text_delta": " recovery"
}
```

客户端记录最高连续序号，重连后带回 checkpoint；frontend 只从下一序号继续发送。若接口仍是一次性 HTTP stream，至少也要在服务端明确“断流后由客户端整体重试”的语义，不能暗示透明的 exactly-once。

## 为什么有些请求不能只靠 token 重放

token 前缀对普通单序列生成很有用，但不等于完整的执行快照。

### 随机采样

temperature、top-p、top-k 相同，不代表恢复后的随机流相同。若要求同一输出，需要保存 RNG key/counter、采样器消费位置，并尽量固定执行的数值路径。否则应把恢复后的后缀视为一个同样合法、但可能不同的采样结果。

### 结构化输出

JSON Schema、正则或 CFG 解码通常维护 grammar automaton 状态。如果该状态能从完整 token 序列确定性重放，就可以重建；如果 processor 还有隐藏状态，就必须显式序列化。

### 多候选和 beam

`n > 1`、best-of 或 beam search 同时维护多个候选、分数和父子关系。一条最终可见前缀无法还原完整候选树。Dynamo 当前文档也把多候选和部分 guided decoding 场景列为请求迁移限制，这恰好说明“保存输出 token”并非通用恢复协议。

### 工具调用

模型已经触发付款、创建工单或写数据库后，恢复时再次生成相同 tool call 可能造成重复副作用。外部调用必须使用幂等键：

```text
idempotency_key = request_id + tool_call_sequence
```

tool executor 记录调用状态与结果，恢复执行只能读取已完成结果或继续未完成事务，不能盲目再次调用。

## Prefill/Decode 解耦下的故障窗口

P/D 解耦把一次请求跨越两个 worker pool，恢复点也随之增加。

### Prefill 尚未完成

P worker 在产生可发布的 KV 之前失败，最安全的动作是在其他 P worker 重做 Prefill。此时没有对外 token，通常不涉及交付去重。

### Prefill 完成，但 KV 尚未发布

KV 写入应有两阶段可见性：先写 block 和校验信息，全部完成后再原子发布 manifest。部分写入不能进入可路由索引。

```text
ALLOCATED → WRITING → SEALED → PUBLISHED
                   ↘ ABORTED
```

只有 `PUBLISHED` block 才能被 D worker 消费。P worker 在 `WRITING` 阶段崩溃，后台回收器根据 lease 清理孤儿 block。

### KV 已发布，D worker 尚未接管

如果 KV 位于远端共享层或有多个副本，另一个 D worker 可以读取它；如果 KV 只存在于故障 P worker 的显存，则必须重新 Prefill。调度器要把“metadata 中存在记录”和“至少有一个可达数据副本”分开判断。

### transfer 进行到一半

传输层不能因为收到部分 completion 就把整段 KV 标记 ready。每个 transfer 应绑定：

```text
(request_id, model_revision, source_generation,
 destination_generation, block_range, transfer_epoch)
```

失败后可从其他 source 重传，也可回退到 recompute。迟到的旧 transfer completion 因 generation 或 epoch 不匹配而被丢弃。

### Decode 已经开始流式输出

D worker 失败后，frontend 使用已提交 token 前缀迁移。如果剩余 deadline 不足以覆盖重新 Prefill，系统应快速失败，而不是消耗更多 GPU 后才超时。

## KV Cache 的正确恢复语义：宁可 miss，不可错用

KV block 是否可复用，至少取决于：

- 模型名称与精确 revision；
- tokenizer 与 prompt token 序列；
- Adapter/LoRA 身份和版本；
- attention/KV layout 与数据类型；
- RoPE、位置编码和上下文参数；
- block 内容 hash；
- worker、allocation 与 publish generation。

其中任何字段不匹配，都应该按 cache miss 处理。恢复期间不要为了命中率放宽校验，因为 stale KV 往往不会立即触发 crash，而是悄悄生成错误结果。

远端 KV 层还要处理三个常见问题：

1. **悬挂引用**：metadata 指向已经释放的 block；
2. **迟到删除**：旧 generation 的回收消息删掉新 block；
3. **孤儿数据**：block 写完，但 manifest 未发布或请求已经取消。

对应机制是带 generation 的引用、compare-and-delete、短 lease 加续租，以及定期 mark-and-sweep。缓存控制面不可用时，服务应优先退化为重新计算，而不是继续信任无法验证的新缓存位置。

## 一个 rank 失败，通常意味着整个并行实例失败

Tensor Parallel、Pipeline Parallel 和 Expert Parallel 通过 collective 共同完成一次 forward。一个 rank OOM 或网络断开后，其余 rank 即使进程仍在，也可能永远阻塞在 NCCL collective 中。

因此，健康和恢复单位应是完整 worker group：

```text
rank failure detected
  → abort outstanding collectives
  → whole instance becomes not ready
  → revoke instance lease and KV ownership
  → fail or migrate all in-flight requests
  → restart all ranks with a new group generation
  → rendezvous + load + canary
  → publish one ready endpoint
```

不能让幸存 rank 各自加入不同世代的新组，也不能因为 rank 0 的健康端点返回成功就认为整个实例可用。编排系统需要 gang-aware rollout：扩容、缩容和替换都以完整拓扑为单位。

“N+1 GPU”也未必提供故障余量。若每个服务实例需要 8 张同节点 GPU，那么冗余单位至少是一整组满足拓扑条件的 8 GPU 资源；若模型跨两个节点，备用容量还要包含网络和可调度节点组合。

## 控制面故障不应立刻打断数据面

Planner、autoscaler 或 operator 决定未来怎样放置资源，但已经在生成的请求不应依赖它们逐 token 在线工作。控制面暂时不可用时：

- Router 继续使用最后一份未过期的安全 endpoint 快照；
- 现有 P/D 数据通道继续运行；
- 禁止基于不完整信息进行激进扩缩容；
- 恢复后通过全量 snapshot 对账，而不是只依赖可能丢失的增量事件。

服务发现和消息总线本身需要高可用部署。Dynamo 的系统级故障设计也把 etcd 与 NATS 的韧性独立列出，因为 worker 自动重启无法修复一个不可用的发现平面。

网络分区比直接 crash 更棘手：worker 可能仍能计算，却无法续租或上报状态。为避免 split-brain，租约失效后它必须停止接受新请求；恢复连接时以新 generation 重新注册，而不是继续沿用旧所有权。

## 过载保护是故障恢复的一部分

许多“故障”并非硬件损坏，而是突发流量把队列、KV HBM 或 transfer buffer 填满，继而触发 OOM、超时和重试风暴。没有 admission control 的自动重试会形成正反馈：

$$
\text{overload}
\rightarrow \text{timeout}
\rightarrow \text{retry}
\rightarrow \text{more load}
\rightarrow \text{more timeout}
$$

保护应同时存在于两层：

- Router 尽量不把请求发给已经繁忙或内存不足的 worker；
- worker 保留最终的并发、token 和队列硬上限，不能完全相信上游视图。

队列达到阈值时，应快速返回明确的 retryable overload，而不是排队到客户端 deadline 后才失败。Dynamo 的请求拒绝设计同样使用“路由规避 + worker 硬限制”两层机制；具体 HTTP 状态码可能随接口版本变化，客户端不应只依赖一个数字，而应识别稳定的错误类型和 `Retry-After` 等提示。

重试策略需要：

- exponential backoff 与 jitter；
- 每条请求的 migration/retry budget；
- 集群级 retry token bucket；
- 保留原始 deadline，不因重试无限延长；
- 高低优先级隔离，避免批处理流量挤掉交互请求。

拒绝一小部分新请求，往往是在保护大量在途请求和恢复余量。

## 取消请求必须沿执行图传播

客户端断开并不意味着 GPU 工作自动停止。Frontend 应把 cancel 传播到请求的所有子任务：

```text
client disconnect
  → frontend request context cancelled
  → pending Prefill cancelled
  → KV transfer cancelled
  → Decode sequence removed from scheduler
  → KV leases released
  → tool / child requests cancelled where safe
```

否则系统会出现 ghost work：客户端已经离开，worker 仍继续生成数千 token，占用 KV 和 batch slot，最终加剧过载。取消和迁移之间也要互斥；一旦收到更高 epoch 的 cancel，迟到的 migration completion 不能让请求重新复活。

## 把故障处理写成状态机

散落在 callback 中的“出错就重试”很难处理并发事件。可以为请求定义显式状态：

```text
ADMITTED
  → PREFILLING
  → TRANSFERRING
  → DECODING
  → COMPLETED

任何运行态
  → MIGRATING → PREFILLING
  → CANCELLING → CANCELLED
  → FAILED
```

状态转换需要 compare-and-swap 的 request epoch：

```text
request_epoch = 7

migrate: 7 → 8
cancel:  8 → 9
```

所有 worker response、KV publish 和 stream frame 都携带 epoch。旧执行在网络恢复后产生的结果会因 epoch 过期而被丢弃。这个机制把“哪个副本当前有权继续生成”从时间猜测变成可验证规则。

终态也要保持幂等：同一完成事件重复到达不能重复结算 token、释放两次引用或再次调用工具。

## 恢复容量要在故障前预留

请求迁移会制造额外工作。设正常 Prefill 负载为 $L_p$，失败实例上有 $N_f$ 条请求需要重放，其恢复输入长度为 $s_i$，单位 token Prefill 成本为 $c_p$，希望在 $T_r$ 内消化，那么额外恢复能力近似为：

$$
C_{recovery} \ge \frac{c_p \sum_{i=1}^{N_f} s_i}{T_r}
$$

如果集群平时已经以 99% 利用率运行，迁移只会把故障从一个实例扩散到其余实例。容量规划应包含：

- 完整并行实例的 N+1 或故障域冗余；
- worker 冷启动、权重加载和编译预热时间；
- 请求重放产生的 Prefill 峰值；
- KV 远端读取与重传带宽；
- 节点维护期间的 PDB 与拓扑约束；
- 过载时保留给高优先级和恢复流量的配额。

autoscaler 的启动速度通常慢于秒级故障恢复。预热 spare、较低的稳态利用率和快速负载拒绝，往往比“故障后再申请 GPU”更可靠。

## 可观测性要能回答请求去了哪里

只有 Pod restart count，无法判断恢复是否真的有效。一组更接近语义的指标包括：

| 维度 | 指标示例 |
| --- | --- |
| 检测 | failure detection latency、lease expiry、canary failure reason |
| 摘流 | stale route count、requests sent after not-ready |
| 迁移 | attempts、success rate、replayed tokens、migration latency |
| 交付 | duplicate frame、sequence gap、client reconnect success |
| KV | orphan blocks、stale generation reject、transfer retry、recompute tokens |
| drain | active requests at start/end、deadline forced kill、drain duration |
| 过载 | shed requests、queue saturation、retry amplification factor |
| 分布式组 | rank failure、collective abort latency、group restart time |

日志至少关联以下字段：

```text
request_id
request_epoch
frontend_id
worker_instance_id + generation
prefill_attempt / decode_attempt
model_revision + adapter_revision
stream_sequence_no
failure_reason + recovery_action
```

一次请求跨越多个 worker 时，trace span 应覆盖 route、queue、Prefill、KV publish、transfer、Decode、migration 和 client write。这样才能判断 TTFT 变长是恢复重算、路由抖动，还是 KV transfer 超时造成。

## 不做故障注入，就不知道恢复路径能否工作

健康检查、租约和迁移代码在正常环境中几乎不会被走到，最容易在真正故障时同时失效。可以建立逐层演练矩阵：

| 注入故障 | 应验证的结果 |
| --- | --- |
| kill 空闲 D worker | endpoint 在目标时间内摘除，新请求不再命中 |
| kill 正在生成的 D worker | 可迁移请求恢复；frame 无不可解释缺口或重复 |
| kill P worker | 未发布 KV 不可见，请求在其他 P worker 重算 |
| 中断 KV transfer | block 不进入 ready；重传、换源或 recompute |
| kill 一个 TP rank | 整个 group 摘流，collective 不永久挂起 |
| 隔离 worker 与 discovery | lease 到期后停止接流量，重连使用新 generation |
| 暂停 KV metadata | cache 退化为 miss，不读取未验证位置 |
| 暂停 Planner | 现有请求继续，恢复后能全量对账 |
| 注入突发流量 | bounded queue 生效，重试不形成放大 |
| 客户端中途断开 | cancel 传播，KV 与 scheduler slot 被释放 |
| 滚动升级节点 | 先摘流后 drain，PDB 约束自愿中断并发 |

演练不仅看“最终恢复”，还要检查时间边界和资源泄漏：故障后多久摘流、迁移消耗多少 token、是否留下孤儿 KV、hard deadline 时还有多少请求被杀死。

建议先在单组件和合成流量下验证，再进行整节点、网络分区和控制面故障。生产演练需要设定 blast radius、自动停止条件和回滚路径。

## 一条流式请求的完整恢复过程

回到开头：请求已经输出 127 个 token，D worker 突然退出。一条设计完整的恢复路径可以是：

1. backend stream 断开，frontend 记录故障时间和当前 request epoch；
2. worker 的 endpoint 被主动事件或 lease 到期摘除，Router 本地也暂时抑制它；
3. frontend 确认客户端已提交到 `sequence_no=127`；
4. 检查请求不含不可重放的候选树或未记录副作用，并且还有 retry budget；
5. request epoch 从 3 增加到 4，旧执行的迟到 frame 全部失效；
6. Router 选择模型、Adapter 与能力兼容的健康 D worker；
7. 若有经过严格校验的 KV 副本则复用，否则用 prompt 加 127 个 token 重新 Prefill；
8. 恢复 RNG、grammar 等已保存状态，或按接口约定接受后缀漂移；
9. 新 worker 从 `sequence_no=128` 继续生成；
10. 恢复耗时、重放 token 数和故障原因进入 trace 与指标。

如果任一前提不成立——例如工具副作用未知、迁移次数耗尽、deadline 不足或没有完整并行实例——系统应快速、明确地结束请求，而不是伪装成成功继续。

## 实施顺序

故障恢复可以按风险从低到高分阶段落地：

### 第一阶段：服务恢复

- readiness 与 liveness 分离；
- endpoint 租约和 generation；
- 完整并行实例作为健康单位；
- bounded queue、worker 硬上限和明确拒绝；
- 优雅下线先摘流再 drain；
- client cancel 全链路传播。

### 第二阶段：普通请求迁移

- frontend 保存 token 化 prompt、参数与输出前缀；
- request epoch 和最大迁移次数；
- 普通单候选生成的 replay；
- KV 丢失时可靠回退到 recompute；
- stream frame 序号和客户端去重。

### 第三阶段：复杂语义

- RNG、sampler 与 grammar 状态快照；
- multi-choice / beam 的完整候选状态；
- speculative decoding 的可重放边界；
- 工具调用幂等与外部事务日志；
- 跨 frontend 的会话恢复。

### 第四阶段：持续验证

- 组件级故障注入；
- 整个 GPU 节点与网络故障域演练；
- 控制面和元数据不可用演练；
- 恢复容量压测；
- 以 token RPO、请求 RTO 和恢复 goodput 设 SLO。

## 小结

分布式 LLM 推理的故障恢复，不能停留在“容器会重启”。真正的恢复边界跨越请求语义、KV 派生状态、流式交付进度和多 rank 执行组。

设计时可以抓住六条原则：

1. 把新流量恢复、在途请求恢复和输出语义连续性分开承诺；
2. 以已确认 token 前缀作为普通生成的恢复事实，KV Cache 可验证则复用，否则重算；
3. 用 lease、generation 和 request epoch 拒绝迟到的旧状态；
4. 以完整 TP/PP/EP 实例为健康、摘流和冗余单位；
5. 先摘流再 drain，并用 admission control 阻止过载演化成级联故障；
6. 用故障注入验证时间边界、交付语义和资源回收，而不只验证进程重新出现。

当这些机制连起来，一条流式请求才有可能在 GPU 进程、节点甚至部分控制面故障后继续；而无法继续的请求，也会以明确、可观测、不会扩大故障的方式结束。

## 参考资料

- [NVIDIA Dynamo: Fault Tolerance](https://docs.nvidia.com/dynamo/latest/user-guides/fault-tolerance)
- [NVIDIA Dynamo: Request Migration](https://docs.nvidia.com/dynamo/latest/user-guides/fault-tolerance/request-migration)
- [NVIDIA Dynamo: Request Cancellation](https://docs.nvidia.com/dynamo/latest/user-guides/fault-tolerance/request-cancellation)
- [NVIDIA Dynamo: Health Checks](https://docs.nvidia.com/dynamo/dev/observability/health-checks)
- [NVIDIA Dynamo: Graceful Shutdown Architecture](https://docs.nvidia.com/dynamo/dev/knowledge-base/design-documents/fault-tolerance/graceful-shutdown-architecture)
- [NVIDIA Dynamo: Request Rejection Architecture](https://docs.nvidia.com/dynamo/dev/knowledge-base/design-documents/fault-tolerance/request-rejection-architecture)
- [Kubernetes: Disruptions](https://kubernetes.io/docs/concepts/workloads/pods/disruptions/)
- [Kubernetes: Pod Lifecycle](https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/)
- [Kubernetes: Configure a PodDisruptionBudget](https://kubernetes.io/docs/tasks/run-application/configure-pdb/)
