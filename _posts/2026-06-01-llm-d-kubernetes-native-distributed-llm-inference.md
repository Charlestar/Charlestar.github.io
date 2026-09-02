---
layout: post
title: "llm-d：Kubernetes 原生分布式 LLM 推理栈"
subtitle: "沿一条请求理解智能路由、KV Cache 与 Prefill/Decode 解耦"
date: 2026-06-01 12:00:00 +0800
last_modified_at: 2026-09-03
author: iStar
catalog: true
series: distributed-inference
series_order: 50
technology_year: 2025
mathjax: true
tags: [分布式推理, Kubernetes, KV Cache]
---

在 Kubernetes 上启动几个 vLLM Pod 并不困难，困难的是决定下一条请求应该进入哪个 Pod。

普通 HTTP 服务的实例通常近似无状态，请求处理时间也相对接近，因此 round-robin 能够工作。LLM 推理打破了这两个前提：一个 Pod 可能已经缓存了某段长前缀，另一个 Pod 正在处理超长 prefill，第三个 Pod 虽然空闲，却需要从头计算整个对话。只看连接数或 Pod 数量，无法判断真实代价。

llm-d 处理的正是模型服务器之上的这一层：它不重新实现 attention kernel 或 continuous batching，而是在 Kubernetes 中组织模型服务器、收集推理状态，并让每条请求去往更合适的执行位置。理解 llm-d，可以从一条请求在系统中的旅程开始。

## 请求到达时，系统真正要做四次判断

假设用户继续一段长对话，新的 prompt 与上一次共享 20K token 前缀。请求进入集群后，理想系统需要回答：

1. 哪些 Pod 正在服务这个模型，且当前健康可用？
2. 哪个 Pod 已经持有最长的可复用 KV 前缀？
3. 缓存收益是否足以抵消该 Pod 的排队时间？
4. 这次请求应在同一实例完成，还是交给独立 prefill/decode 池？

基础路径如下：

```text
client
  │
  ▼
Gateway / L7 Proxy
  │  将请求元数据交给 EPP
  ▼
llm-d Endpoint Picker
  │  发现、过滤、评分、选择 endpoint
  ▼
Model Server Pod
  │  vLLM / SGLang 执行模型
  ▼
streaming response
```

启用 Prefill/Decode 解耦后，一次请求还会分成两段：

```text
Router 选择 decode endpoint
       │
       ├── 缓存足够：decoder 直接完成请求
       │
       └── 未缓存后缀较长：再选择 prefill endpoint
                            │
                            ▼
                     prefill 生成 KV
                            │
                         KV transfer
                            │
                            ▼
                     decoder 连续生成
```

llm-d 的核心组件、KV 索引和解耦协议，都是为这几次判断服务的。

## llm-d 与模型服务器、Kubernetes 各管什么

分布式推理常把多个层次混在一起。先划清边界：

| 层次 | 主要职责 | 不负责什么 |
| --- | --- | --- |
| GPU kernel | attention、GEMM、collective、量化计算 | 请求级路由与 Pod 生命周期 |
| 模型服务器 | batching、KV block、模型执行、采样、流式输出 | 跨副本的全局放置决策 |
| llm-d | 推理感知路由、流量控制、P/D 编排、参考部署路径 | 替代 vLLM/SGLang 的执行引擎 |
| Kubernetes | Pod 放置、服务发现、资源、扩缩容、故障恢复 | 理解 prompt 和 KV 命中价值 |

Kubernetes scheduler 决定 Pod 放在哪台机器上，llm-d Router 决定一条已到达的请求进入哪个 endpoint。前者处理资源放置，后者处理请求放置；二者都叫“调度”，时间尺度和信息来源却不同。

当前 llm-d 也不仅限于 vLLM，架构中的 Model Server 可以是 vLLM 或 SGLang 等执行引擎。具体 well-lit path、KV 协议和已验证硬件组合仍有差异，不能把“架构可接入”理解为所有组合已经达到相同性能。

## 三个核心对象怎样把数据面连起来

llm-d 当前架构以 Router、InferencePool 和 Model Server 为三个核心概念。

### InferencePool：比 Kubernetes Service 多一层推理语义

`InferencePool` 用 label selector 定义哪些 Model Server Pod 属于同一个逻辑池，并为 EPP 提供 endpoint discovery。Pod 扩缩容或 readiness 改变时，候选列表随之更新。

它还把 Gateway 与 EPP 连起来：Gateway 把 `InferencePool` 当作后端，资源中的 endpoint picker 引用告诉代理应咨询哪个 EPP。可以把它理解为“面向推理的 Service”，但它本身不替请求选择最终 Pod，选择逻辑在 EPP 中。

同一个池内可以用 Pod label 表达 variant，例如：

```text
same base model / same InferencePool
├── role=prefill
├── role=decode
└── role=prefill-decode
```

variant 不是另一种模型，而是同一模型服务能力或成本特征的逻辑分组。EPP 可以先按角色过滤，再在组内评分。

### Proxy：负责可靠转发，不承担 LLM 决策

Proxy 是成熟的 L7 数据面，处理连接、TLS、HTTP streaming 和请求转发。典型实现是 Envoy，也可以由符合接口的网关或云负载均衡器承担。

请求到达后，Proxy 暂停转发，通过 Envoy `ext-proc` 协议把请求信息交给 EPP。EPP 返回目标 Pod 地址，Proxy 再把原请求发到该地址。这样无需为了加入 LLM 路由逻辑而重写一个完整网络代理。

### EPP：在请求路径上作放置决策

Endpoint Picker（EPP）读取请求、endpoint 和推理状态，通过插件流水线完成：

```text
parse request
    │
    ▼
flow control / queue
    │
    ▼
Profile Handler
    │
    ├── Filter：排除不健康或角色不匹配的 endpoint
    ├── Score：缓存亲和、负载、延迟等信号打分
    └── Pick：从候选中选出最终 endpoint
```

其 Data Layer 异步观察 Kubernetes API、探测模型服务器指标，并维护 prefix tree、KV index 或 latency predictor 等状态。请求路径只读取这些已更新的数据，不应在每次请求里重新查询整个集群。

这个分层有一个重要后果：Proxy 可用不等于 EPP 可用。`InferencePool` 可以配置 EPP 失联时 FailOpen 或 FailClose。前者保可用性但可能退化为非智能路由，后者保策略一致性但会拒绝流量，需根据业务风险明确选择。

## 为什么 round-robin 会浪费已经计算过的 prompt

Transformer 对 prompt 做 prefill 后，会为每层、每个 token 保存 K/V 状态。后续 generation 只需在此基础上追加 token。若另一个请求拥有相同前缀，模型服务器的 Automatic Prefix Caching 可以复用已有 block，跳过那部分 prefill。

问题在于缓存属于具体 Pod。假设：

```text
Pod A: 已缓存 system prompt + conversation history，queue=4
Pod B: 无缓存，queue=0
```

选择 A 能省掉长 prefill，却要排队；选择 B 立即开始，却要重算全部前缀。最优决策取决于两种成本的比较，而不是“缓存命中永远优先”。

可以用一个概念性目标表达：

$$
\widehat{T}_{finish}(e)=
\widehat{T}_{queue}(e)+
\widehat{T}_{uncached\ prefill}(e)+
\widehat{T}_{decode}(e)
$$

对 endpoint \(e\)，路由器希望选取预计完成时间更低者。前缀命中减少第二项，运行请求数和队列深度影响第一项，硬件与 batch 状态同时影响三项。

llm-d 的默认路径可以组合 prefix-aware 与 load-aware scorer；更进一步的 latency predictor 会基于在线数据预测 TTFT/ITL。无论采用启发式还是预测模型，都应避免让热门前缀把全部请求吸到同一 Pod，形成 cache locality 与排队热点之间的反转。

## Prefix-aware routing 有“近似”和“精确”两条路径

llm-d 当前文档明确区分两种实现，它们不是同一套机制的精度开关。

### 近似路径：从请求历史推断缓存位置

轻量路径不要求模型服务器发送 KV 事件，也不在 EPP 内运行真实 tokenizer。它会用字符/token 比例近似分块，对 prompt 构造 rolling hash，并记录此前把这些前缀路由到了哪个 Pod。

```text
prompt characters
  -> approximate token blocks
  -> rolling hashes
  -> EPP in-memory LRU
  -> estimated prefix match per Pod
```

优点是依赖少、部署简单；缺点是 EPP 记录的是“曾经把请求发到哪里”，不是真实缓存状态。Pod 可能已经因为 HBM 压力淘汰 block，或者重启后清空缓存，索引仍在短时间内认为它存在。

这种路径适合先验证流量中是否确实存在前缀复用价值，也可用于无需精确 block 状态的简单场景。

### 精确路径：用 token 与 KVEvents 建立全局索引

精确路径先把 prompt 转成真实 token IDs，再订阅模型服务器发出的 block add/evict 事件：

```text
request
  -> render/tokenize -> exact token IDs
                           │
model servers -> KVEvents  │
        │                  │
        ▼                  ▼
        global KV-block index
                  │
                  ▼
        exact prefix match by Pod/tier
```

当前文档中的 vLLM 路径使用 render endpoint 获取 token，并通过 ZMQ 接收 `KVEvents`。Indexer 维护哪些 token block 位于哪些 Pod、哪些存储层；prefix scorer 据此计算命中。

精确不等于没有时序问题。路由决策发生后，新 block 尚未生成，对应事件也尚未到达。为填补这个窗口，索引可以加入 speculative entry；如果请求失败或实际状态变化，系统必须依靠后续事件和生命周期处理保持一致。

两条路径的取舍可以归纳为：

| 维度 | 近似索引 | 精确索引 |
| --- | --- | --- |
| prompt 表示 | 字符/token 估算 | 真实 token IDs |
| 状态来源 | EPP 的路由历史 | Model Server KVEvents |
| 能否观察真实淘汰 | 不能直接观察 | 可以 |
| 外部依赖 | 少 | tokenizer/render、事件通道、Indexer |
| P/D block 选择 | 信息有限 | 可定位具体 KV block |

先选择哪条路径，应由流量和运维预算决定，而不是因为“精确”一词天然更好。

## 分层 KV Cache 延长的是状态寿命，不是无限显存

HBM 中 KV block 的访问最快，但容量有限。并发升高或上下文变长时，LRU 会淘汰较冷 block；同一会话稍后返回，就需要重新 prefill。

分层缓存把淘汰从“删除”改成“下沉”：

```text
HBM
 │  eviction
 ▼
CPU RAM
 │  colder / larger working set
 ▼
local NVMe or shared filesystem
```

请求再次出现时，系统比较两种代价：从慢层拉回 KV，还是重新计算前缀。设需要恢复的 KV 大小为 \(S_{kv}\)，有效传输带宽为 \(B\)，固定 I/O 开销为 \(L\)：

$$
T_{restore}\approx L+\frac{S_{kv}}{B}
$$

只有当 \(T_{restore}<T_{recompute}\) 时，命中慢层才有性能意义。远端存储提供更大容量、跨副本共享和重启后保留，但网络拥塞、随机 I/O 与并行请求可能使恢复变慢。

当前 well-lit path 中，vLLM 可通过 `OffloadingConnector` 使用 HBM → CPU → filesystem 层级，SGLang 可使用 HiCache。EPP 的全局索引负责知道 block 在哪里，实际数据读写仍由模型服务器及其 connector 完成。索引不是存放 KV tensor 的数据库。

共享存储还带来容量治理问题。connector 不一定替共享层执行全局淘汰；若只配置写入而没有配额、TTL 或外部 eviction controller，缓存会把存储占满。

## Prefill 与 Decode 为什么值得拆开

一条 LLM 请求包含两个资源特征明显不同的阶段：

- Prefill 并行处理整段 prompt，大矩阵计算较多，常偏 compute-bound；
- Decode 每次追加一个 token，反复读取模型权重与 KV，常偏 memory-bandwidth-bound。

共置执行时，长 prefill 可能与正在 decode 的请求争用 GPU，造成 inter-token latency 抖动。P/D 解耦让两类 worker 分别扩缩容、选择并行度与 kernel，并避免长 prompt 直接干扰 decode 池。

但拆开后必须支付 KV 转移成本。若模型有 \(L\) 层、每 token KV 元素数为 \(d_{kv}\)、数据类型为 \(b\) bytes，长度 \(s\) 的 prompt 所需 KV 规模可粗略表示为：

$$
S_{kv}\approx 2Lsd_{kv}b
$$

系数 2 对应 key 与 value。GQA、MLA、量化和具体布局会改变实际大小，但“上下文越长，需转移状态越多”的趋势不变。

所以 P/D 解耦不是免费地把两个阶段放到不同机器，而是在以下两者之间交换：

```text
收益：阶段专用资源 + 隔离 prefill 干扰 + 独立扩缩容
代价：第二次路由 + KV 传输 + 协议编排 + 故障状态
```

短 prompt、低并发或普通以太网环境中，共置可能更简单也更快；长上下文、明显 P/D 比例失衡且拥有 RDMA 网络时，解耦才更容易兑现收益。

## llm-d 如何决定一次请求是否真的走 P/D

当前 EPP 的 disaggregated profile 并不是无条件为每条请求选择两个 Pod。典型流程先选择 decode endpoint，再由 decider 检查该 decoder 已缓存多少 prompt：

1. `decode-profile` 过滤并选择 D endpoint；
2. decider 根据 D 上的缓存命中和未缓存后缀判断是否解耦；
3. 若 D 已有足够前缀，直接走 decoder-only；
4. 若未缓存后缀较长，再运行 `prefill-profile` 选择 P endpoint；
5. Proxy 将 P/D 地址交给 decode Pod 中的 routing sidecar。

这一步把 prefix routing 与 disaggregation 连接起来。一个拥有完整前缀的 decode Pod 不需要再让 prefiller 重算并传输 KV；只有预期节省大于编排与传输成本时才值得拆分。

路由 sidecar 负责把一次外部请求改写成模型服务器理解的多阶段协议。vLLM 与 SGLang 的交互方式并不相同：当前 llm-d 文档中，vLLM `nixlv2` 走顺序的 prefill→decode 协议；SGLang 路径使用启动协调信息让 P 与 D 并发工作。因此故障处理和取消传播也要按引擎分别验证。

## NIXL 传的是 KV 数据，不是任务消息

P/D 解耦的数据面需要搬运大块 GPU/CPU memory。使用普通任务队列只能传“去哪取”的消息，不能替代高带宽 KV transfer。

llm-d 当前重点集成 NIXL（NVIDIA Inference Transfer Library）。NIXL 提供统一内存传输接口，并可落到 UCX、UCCL、libfabric 等后端，使用 IB、RoCE、EFA 等传输。开启 GPUDirect RDMA 时，NIC 可直接访问已注册的 GPU KV 内存，避免 CPU staging buffer。

```text
Prefill GPU KV memory
   │ register
   ▼
NIC ───────── RDMA fabric ───────── NIC
                                      │
                                      ▼
                              Decode GPU KV memory
```

TCP fallback 适合功能测试，不应据此判断生产性能。生产验收至少要测：

- KV transfer latency 与有效 GB/s；
- 传输期间 prefill/decode kernel 是否被阻塞；
- 多请求并发时 NIC、PCIe 与 RDMA queue 是否饱和；
- model parallel collective 与 KV transfer 是否争用同一网络；
- transfer 失败、超时和请求取消后 block 是否回收。

网络带宽足够也不保证解耦有效。KV 注册、元数据往返、sidecar hop 和同步点都可能进入 TTFT 的关键路径。

## Wide Expert Parallelism 怎样进入这条请求路径

超大 MoE 模型的总专家权重可能跨越多节点。若单纯增加 Tensor Parallel，attention KV 可能随 TP 复制，且跨节点 all-reduce 成本很高。DP/EP 组合采用另一种划分：attention 在不同 DP rank 独立执行，MoE token 只 dispatch 到持有对应专家的 rank。

```text
attention: data parallel, each rank owns its requests/KV
MoE:       router -> sparse dispatch -> expert compute -> combine
```

这能避免 attention KV 在 TP rank 间复制，并让跨节点通信聚焦于稀疏 expert assignment。它同时要求低延迟、高吞吐的 A2A 通信和正确拓扑。

llm-d 的 WideEP 路径进一步与 P/D 解耦结合：

1. EPP 选择 prefill 与 decode variant；
2. `LeaderWorkerSet` 管理一个跨节点 vLLM worker group；
3. prefill 组以 DP/EP 执行 prompt；
4. NIXL 将 KV 传到 decode 组；
5. decode 组以适合 decode 的 DP/EP kernel 继续生成。

这不是普通 Deployment 多加几个副本就能得到的并行方式。一个逻辑 model-server endpoint 可能对应一组必须共同启动的 worker；健康检查、滚动升级和 gang scheduling 都要把整组视作一个单元。

## 流量控制发生在 endpoint 选择之前

LLM 请求的服务时间差异很大：100 token 摘要与长推理可能共用同一池。如果所有请求直接涌入模型服务器，内部队列已经过载后，路由器再选“最空 Pod”也无法恢复尾延迟。

EPP 的 flow control 可以在池饱和时将请求保留在入口队列，按 priority band、flow 和 ordering policy 决定放行顺序。它要解决两个问题：

- admission：当前是否还应向后端发新请求；
- fairness：多个租户、实时流量与 batch 流量如何分享 dispatch 机会。

入口队列并不会减少总工作量，却能防止每个 Pod 都积累不可见的长队列，并为优先级/SLO 提供统一控制点。相应地，EPP 变成有状态的关键路径组件，HA、重启时队列丢失和背压协议都需要演练。

模型服务器内部仍有 continuous batching 队列。两层队列必须协同：入口过于保守会让 GPU 饿死，过于激进又会把排队重新推回后端。

## 扩缩容不能只看 GPU 利用率

GPU utilization 对生成服务并不是单调的健康指标。decode 可能 memory-bound，利用率看起来不满却已达到目标 TPOT；新 Pod 加载数百 GB 权重需要较长时间，扩容生效远晚于普通 Web 服务。

更有意义的信号包括：

- EPP queue depth 与等待时间；
- 每 endpoint running/waiting requests；
- TTFT、ITL/TPOT 与 SLO miss rate；
- KV Cache 使用率、命中率与 eviction rate；
- prefill tokens/s 与 decode tokens/s；
- P/D 两池利用率和 KV transfer backlog；
- 模型加载与 readiness 时间。

llm-d 支持用 HPA/KEDA 消费 EPP 导出的队列等指标，也提供 Workload Variant Autoscaler 的方向来跨 variant 放置副本。无论控制器多复杂，容量模型都应先估算扩容提前量：

$$
T_{ready}=T_{schedule}+T_{image}+T_{weight\ load}+T_{compile/warmup}
$$

若流量尖峰持续时间短于 \(T_{ready}\)，事后扩容很难救场，需要预热副本、预测扩容或入口限流。

缩容同样危险。直接删除持有热门 KV 的 Pod，不仅减少容量，还会让后续请求重算前缀。滚动升级和 scale-down 策略应考虑 cache value 与正在进行的 streaming request，而不只是 replica count。

## Kubernetes 提供生命周期，推理栈要补足语义

生产部署至少要处理这些边界：

### Readiness

容器进程启动并不代表模型可服务。权重加载、distributed group 建立、kernel 编译和 warmup 完成后才能加入 `InferencePool`。探针过早成功会把真实请求发到未就绪实例。

### 成组调度

TP/EP/PP worker 缺一不可时，逐 Pod 随机调度会造成部分 GPU 长期占用却无法运行。需要与底层集群的 gang scheduling、拓扑和 accelerator operator 配合。

### Pod disruption

PDB 可以限制同时中断数量，但不能保证 KV 不丢。维护前应先停止接收、排空 streaming request，再决定是否迁移或舍弃 cache。

### 滚动升级

新旧版本的 KV layout、tokenizer、模型 revision 或 connector 协议可能不兼容。不能因为 Kubernetes label 相同，就把二者当作可共享状态的 endpoint。variant label 应能表达版本边界。

### Namespace 与租户

`InferencePool` 的发现和引用受 namespace 边界约束。多租户还需考虑 prompt 可见性、KV hash 泄露、共享缓存隔离、优先级滥用和网络策略。

## 从单实例到分布式，应逐层增加变量

直接部署完整 WideEP + P/D + tiered cache，很难解释任何结果。更稳妥的实验顺序是让每一步只验证一个假设。

### 单一 Model Server

先确认模型、精度、采样和 tokenizer 正确，建立 TTFT、TPOT、吞吐、显存与质量基线。continuous batching、chunked prefill 和 engine 参数在这一层调好。

### Optimized Baseline

复制多个同构 endpoint，引入 EPP 的 load-aware 与近似 prefix-aware routing。对比 round-robin，观察共享前缀流量下是否减少 recomputed tokens，同时确认热点没有放大 P99。

### 精确 KV 索引

接入 render/tokenize、KVEvents 和 Indexer。故意制造 eviction、Pod restart 和 scale-out，验证索引能否跟上真实 cache 状态，而不只测稳态命中。

### Tiered Cache

先加 CPU 层，再考虑 filesystem。分别测 restore latency、recompute latency、存储容量和命中年龄分布，确认慢层缓存确实节省计算。

### P/D 解耦

在 RDMA 网络上测共置与解耦，对 prompt/output ratio 分桶。拆开 prefill、KV transfer 和 decode 时间，确定 TTFT 变化来自哪里。

### WideEP 与生产控制

最后引入跨节点专家并行、flow control、autoscaling 和 HA。此时单机与数据路径基线已经稳定，才能定位新增 collective、队列或控制面的开销。

well-lit path 提供经过验证的起点和 manifests，但依然要锁定 llm-d、EPP、Gateway、模型服务器、NIXL 与 driver 版本。复制 `main` 分支配置而不记录 revision，会让复现实验随项目更新漂移。

## 一个端到端评测应记录什么

只报告总 tokens/s 会掩盖路由和缓存是否真的工作。建议把指标分成五组。

### 请求体验

```text
TTFT P50/P95/P99
TPOT / ITL P50/P95/P99
end-to-end latency
timeout / cancellation / error rate
SLO attainment / goodput
```

### 路由行为

```text
chosen endpoint and scorer breakdown
queue wait before dispatch
endpoint load at decision time
cache-local choice vs least-loaded choice
fallback / fail-open count
```

### KV 状态

```text
matched prefix tokens
HBM / CPU / filesystem hit rate
KVEvents lag and index staleness
eviction / restore / recompute tokens
bytes moved between tiers
```

### P/D 数据路径

```text
prefill queue + execution
KV metadata latency
KV transfer bytes / latency / bandwidth
decode queue + execution
decoder-only vs disaggregated decision rate
```

### 集群与成本

```text
tokens/s/GPU and goodput/GPU
GPU, HBM, CPU RAM, NIC utilization
replica warm-up time
scale events and disruption recovery
cost per successful SLO-qualified token
```

将这些指标用 request ID 串起来，才能回答“一条慢请求是排队、重算、传输还是 decode 变慢”。只有聚合 dashboard 而没有跨组件 trace，故障往往会在 Proxy、EPP、sidecar 与 engine 之间来回归因。

## 几类典型故障如何判断

| 现象 | 优先检查 | 常见原因 |
| --- | --- | --- |
| cache hit 提升但 P99 更差 | endpoint queue、热门 prefix 分布 | cache affinity 制造单点热点 |
| 索引显示命中，engine 仍重算 | KVEvents lag、block eviction、tokenizer revision | 全局视图陈旧或 token 边界不一致 |
| P/D 的 TTFT 高于共置 | KV transfer、sidecar hop、未缓存长度 | 传输与编排成本大于隔离收益 |
| decode 偶发停顿 | 长 prefill 混入 D、NIC 争用、EP straggler | 角色过滤或网络隔离失效 |
| 扩容成功但请求仍排队 | weight load、warmup、EPP discovery | Pod 存在但未真正 ready |
| rollout 后命中骤降 | revision/label、cache layout | 新旧 variant 不共享 KV |
| EPP 重启造成流量波动 | fail mode、入口队列、索引重建 | 关键状态未恢复或代理 fallback |
| shared storage 命中慢 | IOPS、并发 restore、网络路径 | 拉取 KV 比重新 prefill 更贵 |
| GPU 利用率低且吞吐低 | flow control、双层队列、transfer backlog | admission 过于保守或数据路径饥饿 |

这类问题无法通过简单增加副本统一解决。更多 Pod 可能稀释前缀局部性、增加 cache 冷启动，或者让共享存储与网络更拥塞。

## 如何理解官方性能结果

llm-d 官方页面列出了 prefix-cache routing、predicted latency、P/D、WideEP 和 hierarchical KV offloading 的多个性能案例。这些结果证明对应路径在特定模型、硬件和流量中有效，但每个数字的 baseline 都不同。

阅读时需要对齐：

- 模型、精度、parallelism 和 engine 版本；
- GPU/加速器型号及节点互联；
- shared-prefix 比例与重复间隔；
- input/output length 和并发曲线；
- round-robin baseline 是否已调优；
- 指标是 TTFT、ITL、output throughput 还是 cluster throughput；
- 是否计入 warm-up、缓存预热与失败请求。

例如 prefix-aware routing 的收益高度依赖共享前缀；P/D 的收益依赖 prompt/output 比与 RDMA；WideEP 的收益依赖 MoE 路由和跨节点网络。把其中一个案例外推成“llm-d 固定提升多少”没有意义。

更合适的使用方式，是选择与自己负载最接近的 well-lit path 作为实验起点，再用同一批请求做受控对比。

## 从架构选择回到一条请求

现在重新看开头那条拥有 20K 共享前缀的请求：

1. `InferencePool` 提供健康、版本匹配的 model-server candidates；
2. EPP 在 flow control 后读取负载与 KV index；
3. scorer 比较缓存节省和 endpoint 排队成本；
4. 若 decoder 已有前缀，直接走 decoder-only；
5. 若未缓存后缀足够长且网络合适，再选择 prefiller；
6. sidecar 按执行引擎协议协调 KV transfer；
7. Model Server 完成 decode，Proxy 维持流式返回；
8. KVEvents、指标与 trace 更新下一次决策所需状态。

llm-d 的“云原生”价值，不是把模型塞进 YAML，而是让 Kubernetes 的发现、生命周期和策略机制能够使用 LLM 特有的状态。KV locality、阶段差异和请求不均衡一旦进入路由与运维决策，集群才不再只是若干彼此无知的 GPU Pod。

是否采用每项高级能力，应由请求路径上的真实瓶颈决定：缓存复用不足先改路由，工作集过大再加 tier，长 prefill 干扰 decode 才拆 P/D，超大 MoE 需要跨节点才引入 WideEP。沿着这条因果链逐层构建，复杂度才会换来可解释的性能收益。

## 参考资料

- [llm-d 官方仓库](https://github.com/llm-d/llm-d)
- [llm-d v0.9 Architecture](https://llm-d.ai/docs/architecture)
- [llm-d Router 与 EPP](https://llm-d.ai/docs/architecture/core/router)
- [InferencePool](https://llm-d.ai/docs/architecture/core/inferencepool)
- [Prefix-Cache Aware Routing](https://llm-d.ai/docs/architecture/advanced/kv-management/prefix-cache-aware-routing)
- [KV Cache Management](https://llm-d.ai/docs/architecture/advanced/kv-management)
- [Tiered Prefix Cache](https://llm-d.ai/docs/well-lit-paths/foundations/tiered-prefix-cache)
- [Disaggregated Serving](https://llm-d.ai/docs/architecture/advanced/disaggregation)
- [Multi-Node Wide Expert Parallelism](https://llm-d.ai/docs/well-lit-paths/foundations/wide-expert-parallelism)
