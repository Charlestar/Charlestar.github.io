---
layout: post
title: "NVIDIA Dynamo：把多套推理引擎组织成一个系统"
subtitle: "从请求路由、KV 状态到 P/D 扩缩容与 Kubernetes 编排"
date: 2026-07-08 09:00:00 +0800
last_modified_at: 2026-08-09
author: iStar
catalog: true
series: distributed-inference
series_order: 30
technology_year: 2025
mathjax: true
tags: [分布式推理, Kubernetes, KV Cache]
---

vLLM、SGLang 与 TensorRT-LLM 能把一组 GPU 上的模型执行得很快，但数据中心级推理还需要回答引擎之外的问题：一条请求该去哪个副本，prefill 与 decode 要不要分离，两个资源池各扩多少实例，KV Cache 在哪台 worker 上，worker 重启后怎样停止使用过期状态，以及怎样把模型服务映射到 Kubernetes 的 Pod、网络和拓扑约束。

NVIDIA Dynamo 位于这层。它不是另一个 attention kernel，也不是要替换现有 inference engine；它把不同 backend 变成可发现、可路由、可分离、可扩缩和可恢复的分布式推理系统。引擎继续负责模型 forward、Paged KV Cache、continuous batching 与 sampling，Dynamo 负责跨实例的 request、control 和 state 协作。

2025 年最初发布时，Dynamo 的核心被概括为 Planner、Smart Router、Distributed KV Cache Manager 与 NIXL。此后开源项目持续演进，增加了 Kubernetes Operator、KVBM、更多路由/部署模式和故障处理。具体 CRD 字段与命令会变化，但架构问题相对稳定。

本文不按组件逐项抄文档，而是沿一条请求经过的三条路径理解 Dynamo：

```text
request path   谁接收请求、选择 worker、执行 P/D 并返回 token
control path   谁观察负载、决定资源数量与 placement
state path     谁跟踪 KV、worker membership、事件与故障后的失效
```

## 为什么单个高性能引擎还不够

先看最简单的多副本部署：

```text
               ┌─► Engine Pod A
HTTP Gateway ──┼─► Engine Pod B
               └─► Engine Pod C
```

如果 Gateway round-robin，请求数量可能均匀，计算成本却不均匀：

- A 已缓存请求的 20k-token 公共前缀；
- B 没有缓存，但队列为空；
- C 缓存命中最多，却正在处理多个长输出；
- 新请求是长 prefill、短回答；
- 另一个请求 prompt 很短，却会生成很久。

HTTP 层看到的都是一个连接，LLM runtime 看到的是完全不同的 future work。路由不理解 token、KV 与 active decode，就会同时错过 cache reuse 和 load balance。

再加入 P/D disaggregation：

```text
Frontend → Prefill Pool → KV transfer → Decode Pool → token stream
```

系统还要选择 P 与 D 的配对、把首 token/采样状态交接、管理 KV transfer 和两个池的独立容量。普通无状态服务发现只能告诉你 Pod 存活，不能完成这套推理语义。

## 三条路径为什么要分开

### Request Path

Request path 在每次请求和 token stream 上发生，必须尽量短、稳定：

```text
client
  │
  ▼
Frontend / Gateway
  │
  ▼
Router decision
  │
  ├─ aggregated ─► one engine worker
  │
  └─ disaggregated ─► P worker ─► D worker
                                      │
                                      ▼
                                  token stream
```

它负责协议、路由、执行与流式返回。Planner 不应位于每个 token 的同步关键路径，否则控制器短暂抖动会直接阻塞生成。

### Control Path

Control path 在秒到分钟尺度观察：

- 请求率、input/output length；
- P queue token、D active KV 与 step latency；
- TTFT/ITL SLO；
- worker 数量和 GPU 拓扑；
- 扩容冷启动时间与成本。

Planner 根据这些信号计算目标 replica/placement，Operator 或其他 connector 把目标应用到底层资源。

### State Path

State path 保存快速路由和恢复所需的软状态：

- worker membership 与 health/lease；
- 每台 worker 的 KV block/prefix 事件；
- 全局 prefix index；
- external/offloaded KV 位置；
- in-flight request 与 P/D handoff 状态。

它允许 Router 快速判断 cache overlap，也必须在 worker 失败时让过期状态及时消失。State path 可以容忍 cache miss 和重建，却不能容忍把错误 KV 当作正确 KV 使用。

三条路径的交互是：

```text
                    metrics / events
Request Path ─────────────────────────► State + Control
     ▲                                      │
     │ worker set / routing index / targets │
     └──────────────────────────────────────┘
```

## Request Path 从 Frontend 开始

Dynamo Frontend 可以直接作为 OpenAI-compatible 入口，也可以与 Kubernetes Gateway API Inference Extension 组合：外部 Gateway 负责认证、限流和边缘策略，Endpoint Picker Plugin 选择目标，再把流量送到相应 worker sidecar/Frontend。

两种拓扑的差别在 routing boundary：

```text
Dynamo-native:
client → Dynamo Frontend → integrated Router → workers

Gateway API:
client → Gateway → EPP worker selection → Frontend sidecar → workers
```

它们不应该同时各做一次互相冲突的负载均衡。部署时要明确谁拥有最终 worker selection，哪些 request hints 可以穿透，以及 retry 后是否仍保持 session/KV locality。

## Smart Router 的目标不是“命中最长前缀”

对请求 $r$ 与 worker $i$，设：

- $P_r$ 为 prompt block 数；
- $H_{ri}$ 为该 worker 已有的可复用 prefix blocks；
- $Q_i$ 为当前 prefill/decode 负载；
- $C_i$ 为 worker 容量与设备能力。

可以构造简化成本：

$$
Cost(r,i)=
w_p(P_r-H_{ri})
+w_d\widehat{DecodeLoad}_i
+w_q\widehat{Queue}_i
$$

Router 选择较低成本 worker。实际 Dynamo cost model 和可调参数随版本演进，这个式子只表达原则：未命中的 prefill 计算、active decode 与排队都要计价。

### 只看 Cache 会形成热点

系统 prompt 首次落到 A 后，所有共享前缀请求都选 A；A 越有缓存，越被继续选择，队列越来越长，其他 worker 空闲。

```text
cache-only positive feedback

more hits on A → more requests to A → more cache on A → more hits on A
```

### 只看 Load 会反复重算

Least-loaded 每次把请求发往空闲 worker，多轮会话可能在副本间跳动，长前缀反复 prefill。

### 成本模型要比较复用收益与等待

如果 A 能省 10k-token prefill 但要排队 2 秒，B 需要重算却能立即执行，谁更好取决于模型 profile 与 TTFT deadline。Router 应估计 work，而不只是把 cache overlap 当作绝对优先级。

## Router 怎样知道 Worker 有哪些 KV

### Event-backed Index

Worker 在 KV block allocate/store/free/evict 时发布事件，Router/Indexer 用这些事件维护全局 prefix registry，常以 radix/prefix structure 进行匹配：

```text
worker A: block stored(hash h1, h2, h3)
worker B: block evicted(hash h1, h2)
                  │
                  ▼
            global KV index
                  │
new prompt hashes ─┴─► per-worker overlap
```

事件需要 worker epoch 和顺序信息。若 A 重启后重新使用同一名字，旧实例迟到的 `store` 事件不能污染新实例状态；event gap 也要通过 snapshot/reconciliation 修复。

### Prediction-based Index

若 worker 不发布完整事件，Router 可以根据自己曾经发送的请求推测哪些 prefix 可能仍存在。这减少基础设施依赖，但 eviction、preemption 和进程内 cache policy 会让预测逐渐偏离真实状态。

错误预测通常应导致一次 cache miss 和重算，而不是读错数据。worker/connector 在实际加载时仍要验证 block 是否存在；Router 的 index 是 placement hint，不是 correctness authority。

### Load-only Fallback

当 event path 不健康或某 backend 不支持 KV reporting，系统应能退回 load-aware/round-robin，而不是停止服务。降级会增加 prefill 计算，但保持 request path 可用。

## Prefix Identity 必须跨 Backend 一致

Router 要比较 KV overlap，所有 worker 必须对相同执行输入得到相同 block identity。至少包含：

- 精确 token IDs，而不是原始文本；
- block boundary 与 hash chain；
- model/revision；
- tokenizer、chat template 与 special tokens；
- LoRA/adapter；
- 影响 hidden state 的位置和模型配置；
- 多模态输入的内容 identity；
- KV layout/dtype compatibility namespace。

如果 Python 进程使用随机化 hash seed，而系统直接依赖进程 hash，两个 worker 对同一 token block 可能得到不同结果。官方部署文档会对相关 backend 给出一致性配置；更稳妥的原则是使用显式、稳定、带版本的内容哈希。

## Aggregated 与 Disaggregated 是两种 Request Graph

### Aggregated

一个 worker 完成 prefill 和 decode：

```text
Frontend → Router → Engine Worker(P+D) → stream
```

优点是权重只保存一份、KV 不跨 worker；短 prompt、低并发、小模型或慢网络下通常更简单。

### Disaggregated

两个池分别执行：

```text
Frontend → Router
               │
               ▼
         Prefill Worker
               │ KV + first-token state
               ▼
          Decode Worker → stream
```

Dynamo 让两个 worker pool 独立设置 replica、GPU 数和并行策略，并使用 NIXL 完成 KV transfer。P 的扩容压力主要来自 input length、context 与 prefix reuse；D 的压力来自并发、output length 和 active KV memory。

分离并不总是更快。官方文档同样建议在小模型、短 prompt、低 concurrency 或缺少高速 fabric 时比较 aggregated layout，不能把 disaggregation 当成默认正确答案。

## P/D Request Handoff 传的不只是 KV Bytes

Prefill 完成后，D 需要：

- 已计算 token 数与 position；
- P 产生的 KV block descriptors；
- 第一个 token 或下一分布相关状态；
- sampling 参数与 RNG/sequence state；
- stop、grammar 与 request deadline；
- source worker identity 与 transfer epoch。

NIXL 负责 memory transfer，Dynamo/engine connector 负责这些 request metadata。两端必须使用相同模型、KV dtype、block size 与 layout；否则可能直接传输失败，更危险的是 bytes 能复制但 attention 以错误 stride/head 分片解释。

同节点可以走 CUDA IPC/NVLink 等路径，跨节点通常需要 RDMA-capable fabric 与正确 device plugin/网络配置。传输 backend 的选择是 NIXL 层问题，P/D request 生命周期属于 runtime。

## KVBM 在 Engine Cache 之上补什么

Inference engine 已经有自己的 GPU KV block manager。Dynamo KVBM（KV Block Manager）的目标是把 block 生命周期扩展到更广的 memory hierarchy 与共享场景：

```text
attention hot path
       │
       ▼
GPU HBM blocks
       │ offload / onboard
       ▼
pinned CPU DRAM
       │
       ▼
local SSD / remote storage tiers
```

KVBM 关心：

- block layout 与 model dimensions；
- allocate/register/match 等状态；
- offload/onboard 调度；
- external store/backend；
- 与 NIXL registration/transfer 集成；
- block events 与 metrics。

它不等同于 Router 的 KV index。Router 持有“某 worker 可能有某 prefix”的轻量 registry；KVBM/engine cache 才持有实际 block 和生命周期。

它也不等同于 Mooncake Store/LMCache 等所有外部存储。Dynamo 允许不同 KV management 方案接入，具体 backend 支持和成熟度随 release 变化。设计中应明确谁是 block allocator、谁负责 eviction、谁发布 event，避免两个管理器同时认为自己拥有同一块内存。

## 多层 KV 的价值与代价

假设从某层加载 $S$ 个 token 的 KV 耗时：

$$
T_{load}(tier,S)
$$

重新 prefill 耗时：

$$
T_{prefill}(S)
$$

只有：

$$
T_{lookup}+T_{load}+T_{onboard}<T_{prefill}
$$

复用才降低 TTFT。命中 SSD/remote storage 不一定比短 prompt 重算快。

多层 cache 还要决定：

- write-through 还是按价值选择 offload；
- HBM eviction 后是否保留 host copy；
- 热 block 是否复制；
- 长 prefix 与短热门 prefix 谁更值得容量；
- worker failure 后 external copy 是否可继续使用；
- 多租户数据能否共享。

因此 KVBM 是 memory system，不是一项无代价的开关。

## Planner 为什么不能只做 HPA

通用 HPA 常用 CPU/GPU utilization 或 request count。LLM 服务真正约束的是 TTFT 与 ITL，它们受序列长度、KV capacity、P/D queue 和 transfer 共同影响。

两个流量窗口即使 request/s 相同：

```text
Window A: ISL 500,  OSL 2000  → Decode-heavy
Window B: ISL 16000, OSL 100  → Prefill-heavy
```

需要扩的池完全相反。单个 `replicas` 指标无法表达。

Dynamo Planner 消费 runtime metrics，分别计算 target prefill/decode replica。当前文档提供 throughput、latency、load 和 SLA-oriented 等优化目标；实现和算法在版本间会迭代，但控制回路可以概括为：

```text
metrics
  │
  ▼
estimate / predict workload
  │
  ▼
performance model + correction factor
  │
  ▼
target (num_prefill, num_decode)
  │
  ▼
Kubernetes/runtime connector applies scaling
  │
  └──────── observed result feeds back ────────┐
                                               │
                                               └─► next interval
```

## SLA Planner 的输入不能只靠 Request Count

更完整的 workload vector 是：

$$
W_t=(\lambda_t,
E[ISL_t],E[OSL_t],
Q_{P,t},U_{KV,D,t},
TTFT_t,ITL_t)
$$

其中：

- $\lambda$：到达率；
- ISL/OSL：输入/输出长度分布；
- $Q_P$：prefill queue/token work；
- $U_{KV,D}$：decode active KV utilization；
- TTFT/ITL：实际 SLO 反馈。

Planner 可以用预部署 profile/AIConfigurator 的性能模型估计不同拓扑，也可以用在线观测校正预测。目标不是让所有 GPU 100% 利用，而是在 SLA 与成本之间选择容量。

## 扩缩容延迟让控制器必须预测

创建 GPU Pod 需要调度、拉镜像、加载数十到数百 GB 权重、建立并行组、编译/捕获 kernel 与健康检查，可能远慢于流量 burst。纯反应式规则：

```text
queue high → start worker → worker ready after burst is over
```

既错过 SLO，又可能在负载下降后留下昂贵空闲副本。

Planner 需要：

- 预测下一窗口的 request/ISL/OSL；
- 把 cold-start time 加入 lookahead；
- 设置 min replica/headroom；
- 对 scale-up/scale-down 使用不同阈值；
- 加入 cooldown 与最小稳定时间；
- 让 P/D 联合变化，避免只扩一侧；
- 在模型误差增大时回退保守策略。

当前 target 与实际 ready capacity 也要分开。Kubernetes 中 replicas 已增加，不等于 worker 已加载模型并进入 Router endpoint set。

## DynamoGraphDeployment 与 Operator 做什么

在 Kubernetes 中，Dynamo 用声明式资源描述 Frontend、worker 和连接关系。Operator 将期望状态转化为 deployment/pod group、service discovery 和配置。

```text
DynamoGraphDeployment
  ├─ Frontend component
  ├─ Prefill worker component
  ├─ Decode worker component
  ├─ dependencies / endpoints
  └─ scaling and placement policy
             │
             ▼
       Dynamo Operator reconcile
             │
             ▼
Pods / Services / EndpointSlices / worker metadata
```

多节点 tensor/pipeline/expert parallel worker 不是一组可以任意独立重启的 Pod。它们需要 gang scheduling、共同 readiness 和 topology-aware placement。Dynamo 文档中的 Grove/PodClique 一类资源用于表达这些 worker group 和 scaling group。

这里的关键原则是：**扩的是一个可工作的分布式 instance，而不是零散 GPU Pod 数。** 只启动部分 rank 的 replica 会占用资源却无法接收请求。

## Backend-Agnostic 的真实含义

Dynamo 支持 vLLM、SGLang、TensorRT-LLM 等 backend，意味着 request/control/state 抽象可以接入不同 engine，不意味着所有功能组合完全等价。

Connector/integration 至少要适配：

- OpenAI request 与 engine sampling schema；
- stream/cancel/error 语义；
- engine KV event；
- P/D mode 和 transfer metadata；
- block layout、dtype 与 parallel rank；
- health、metrics 与 graceful drain；
- structured output、LoRA、speculative decode 等能力。

现行架构既有 integrated backend（Dynamo worker 与 engine 同进程），也探索 sidecar 连接 stock engine 的方式。同进程能深度访问 KV 与 scheduler，升级耦合更强；sidecar 隔离版本并复用原生 server，能暴露的内部状态较少。

所以选择 backend 时应查对应版本的 feature/compatibility matrix，而不是只看首页列出的名称。

## Failure 首先发生在 State 失真

### Worker Crash

worker 失联后，Router 必须停止分配新请求，KV index 要移除其 block，P/D in-flight transfer 要失败或重路由，客户端请求要重试、迁移或明确终止。

### Stale KV Event

一个 worker 重启后进程名相同、内存已重建。旧 epoch 的迟到事件如果仍被接受，Router 会把请求送去命中不存在的 cache。事件必须携带 instance/epoch，discovery lease 过期时清理关联状态。

### Graceful Scale Down

直接删除 D worker 会中断所有 active decode：

```text
stop new routing
      │
      ▼
drain / migrate / finish active requests
      │
      ▼
release KV and transfer leases
      │
      ▼
remove endpoint and terminate
```

若输出很长，完全 drain 可能耗时很久。平台需要 deadline、migration 能力或可接受的取消策略。

### Control Plane Failure

Planner/Operator 短暂不可用时，现有 request path 应尽量继续运行在最后一个稳定配置；否则一个 autoscaler 故障会放大成推理中断。控制器恢复后再 reconcile actual vs desired。

### Overload

故障减少容量时，盲目继续接收会把排队扩散到全部 worker。load shedding/admission control 应在 deadline 已不可满足前拒绝或降级，而不是让请求超时后才释放资源。

## Dynamo 与 llm-d 的边界

两者都处于 inference engine 之上的分布式/云原生层，也都关注 KV-aware routing、P/D disaggregation 与 Kubernetes，因此并不是简单的“底层库 vs 上层平台”关系。

可以从生态与控制边界理解：

### Dynamo

提供自己的 distributed runtime、Frontend/Router、Planner、KVBM、NIXL 集成和 Kubernetes Platform/Operator，把数据面与控制面组成一套 NVIDIA 主导的开放推理栈，并连接多种 engine。

### llm-d

围绕 Kubernetes、Gateway API Inference Extension、vLLM 与社区组件组织推理服务，强调 Kubernetes-native scheduling/routing 与可组合生态。

二者功能会有重叠，也可能共享标准或底层组件。选择时应比较：

- 当前目标 engine 与模型的成熟支持；
- 是否需要 KVBM/NIXL/Planner 的一体化路径；
- 现有 Kubernetes Gateway、调度与可观测体系；
- 多厂商硬件/软件要求；
- 升级节奏与团队可维护边界；
- 故障与数据面的实测能力。

不要同时启用两套都认为自己拥有最终 worker selection 或 replica target 的 controller，除非明确划分 authority。

## 怎样评估 Dynamo 而不是评估底层 Engine

### Baselines

至少分开：

```text
1. single engine / one replica
2. multiple engines + round-robin
3. multiple engines + load-aware routing
4. Dynamo KV-aware routing
5. Dynamo disaggregated P/D
6. P/D + KVBM / external KV tiers
7. P/D + Planner autoscaling
```

每增加一层，说明哪项指标变化。否则无法知道收益来自 engine kernel、cache routing、P/D 还是更多 GPU。

### 固定 Model 与 Backend

同一模型 revision、量化、TP/PP/EP、engine version 和 sampling 参数。Dynamo 官方发布中的大倍数常来自特定硬件、DeepSeek-R1 workload、context/generation 配置与完全不同的 P/D parallelism；这些结果不能代表任意部署。

### Workload

保留真实：

- 到达时间与 burst；
- ISL/OSL joint distribution；
- session/prefix overlap；
- tenant/model/LoRA；
- reasoning、tool use 和 multimodal；
- cancel/timeout。

随机生成互不相同 prompt 会让 KV-aware routing 看不到价值；所有请求共用同一 prompt 又会夸大它。

### Metrics

```text
TTFT / ITL / E2E P50-P99
SLO attainment and per-GPU goodput
prefill recomputed tokens
KV overlap predicted vs actual
P/D queue and transfer visible latency
HBM/host/storage KV utilization
router decision latency
scale decision → ready capacity time
request retry/migration/failure rate
GPU-hours and cost per qualified request
```

## 一条可落地的上线顺序

### 1. 先稳定 Backend

在固定副本上验证模型正确性、TP/PP、吞吐和 sampling。不要把 engine 本身的问题带进分布式 control plane。

### 2. 加 Frontend 与 Load-aware Routing

验证 streaming、cancel、retry、worker discovery 和 graceful drain。先不依赖 KV events，建立可用基线。

### 3. 打开 KV-aware Routing

检查 block hash、事件完整性、预测/实际 hit、热点和 index 恢复。故意断开 event path，验证能安全降级。

### 4. 评估 P/D

用 DistServe 一类 goodput 方法选择初始 P/D ratio，测 NIXL path、KV layout 和生命周期。比较 tuned aggregated baseline。

### 5. 再加入 KVBM/External Tier

以净节省 prefill 时间衡量，不只看 hit rate。隔离 model/adapter/tenant namespace。

### 6. 最后启用 Planner 自动应用

先让 Planner shadow mode 输出建议，与人工 capacity model 对比；误差可控后再自动 scale-up，最后谨慎开放 scale-down。所有变更都要有上下限和回滚。

这种顺序让每层的正确性和收益可归因。一次性部署完整 stack，遇到 TTFT 波动时很难知道是 Router、KV、transfer、Planner 还是 engine。

## 小结

NVIDIA Dynamo 解决的是“怎样把多套高性能推理引擎变成一个持续运行的分布式系统”。Request path 通过 Frontend、Router 与 P/D workers 处理低延迟执行；control path 由 metrics、performance model、Planner 和 Operator 调整资源；state path 用 worker discovery、KV events、prefix index、KVBM 与 NIXL 支撑复用和故障恢复。

KV-aware routing 的核心不是最长前缀优先，而是比较复用省下的 prefill 与目标 worker 的排队/active decode；KVBM 的核心不是无限扩容 HBM，而是让 block 在多层 memory 中以明确 owner 和状态 offload/onboard；Planner 的核心不是追求满卡，而是根据 ISL/OSL、P queue、D KV 和 TTFT/ITL 分别调整两个池。

Dynamo 不替代 vLLM、SGLang 或 TensorRT-LLM，也不让它们的功能差异消失。它提供跨 backend 的协作框架，具体组合仍要经过兼容性和故障验证。与 llm-d 等平台比较时，也应关注 authority、生态和运维边界，而不是只比较组件名称。

真正成熟的部署不是“把所有特性打开”，而是让三个路径各自可观测、可降级：Router 丢失 KV state 时退回负载路由，Planner 故障时保持稳定容量，external cache 失败时允许重算，P/D transfer 不可用时有明确失败或 aggregated fallback。这样，分布式优化才不会把一次可恢复的 cache miss 放大成整个服务中断。

## 参考资料

- [NVIDIA Dynamo 官方仓库](https://github.com/ai-dynamo/dynamo)
- [Dynamo Overall Architecture](https://docs.nvidia.com/dynamo/dev/knowledge-base/overview)
- [Dynamo Routing Concepts](https://docs.nvidia.com/dynamo/latest/components/router/routing-concepts)
- [Dynamo Planner](https://docs.nvidia.com/dynamo/latest/components/planner)
- [Dynamo KVBM Design](https://docs.nvidia.com/dynamo/v1.0.1/design-docs/component-design/kvbm-design)
- [Dynamo Disaggregated Serving](https://docs.dynamo.nvidia.com/dynamo/dev/kubernetes/disaggregated-serving/overview)
- [Introducing NVIDIA Dynamo（2025）](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/)
