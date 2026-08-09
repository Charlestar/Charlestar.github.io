---
layout: post
title: "vLLM V1 EngineCore：引擎进程与执行核心的解耦"
subtitle: "沿一次请求理解 Client、Scheduler、KV Cache 与 GPU Worker"
date: 2026-05-23 12:00:00 +0800
last_modified_at: 2026-08-09
author: iStar
catalog: true
series: serving-scheduling
series_order: 30
technology_year: 2025
tags: [推理调度, vLLM, LLM推理]
---

一个 OpenAI-compatible 推理服务同时承担两类节奏完全不同的工作。

HTTP 侧要解析 JSON、鉴权、加载多模态输入、tokenize，并以客户端能消费的速度流式返回。GPU 侧则要尽可能连续地形成批次、分配 KV Cache、执行模型并采样。前者受网络和用户行为影响，后者受显存、kernel 与调度影响。

如果两者紧密耦合在同一个控制循环中，任意一侧的长尾都可能干扰另一侧。vLLM V1 用 EngineCore 把“如何运行模型”收拢成清晰边界，再通过不同 client 让它既可以在进程内被同步驱动，也可以在独立进程中持续运行。

理解这套架构，最有效的方式不是从类名列表开始，而是跟随一条请求：

```text
HTTP / LLM.generate
  -> input processing
  -> EngineCoreRequest
  -> EngineCoreClient
  -> Scheduler + KVCacheManager
  -> Executor
  -> GPU Worker / ModelRunner
  -> EngineCoreOutputs
  -> OutputProcessor
  -> text / stream
```

## V1 重构的对象不是只有“一个进程”

V0 已经拥有 PagedAttention、continuous batching 和大量模型支持，但随着 prefix caching、chunked prefill、speculative decoding、多模态和多种并行方式分别演进，调度与执行路径变得越来越难组合。

V1 保留成熟的模型、kernel 和基础设施，同时重构了 scheduler、KV cache manager、worker、sampler 与 API server。官方给出的目标包括模块化、低 CPU overhead、统一关键优化并尽量默认启用合理能力。

“V1 使用多进程”是线上部署的重要事实，但不是全部设计。更根本的变化是职责边界：

- frontend 不直接操纵 scheduler 内部状态；
- EngineCore 不负责 HTTP、字符串 tokenization 和文本流；
- worker 不决定哪些请求应在本轮运行；
- client 将同一组 EngineCore 操作映射到进程内调用或 IPC。

有了这个边界，进程拓扑可以变化，而 request/schedule/execute/output 的语义保持一致。

## 五个层次各自负责什么

### Entrypoint 与 input processor

离线场景通常从 `LLM` 类进入，在线服务从 `vllm serve <model>` 启动的 API server 进入。它们负责把用户输入转成引擎可处理形式，包括：

- 校验 sampling/generation 参数；
- 应用 chat template；
- tokenize 文本；
- 加载和预处理图片、音频等多模态数据；
- 生成 request ID；
- 建立每个请求的异步输出队列。

这些工作可以占用不少 CPU，却不应该进入 GPU 调度热循环。

### EngineCoreClient

Client 是 frontend 与 EngineCore 的协议适配层。当前源码中可看到三种基本形态：

| Client | EngineCore 所在位置 | 调用模型 | 典型角色 |
| --- | --- | --- | --- |
| `InprocClient` | 当前进程 | 直接方法调用、由调用方 step | 进程内/兼容式同步路径 |
| `SyncMPClient` | 后台进程 | ZMQ + 同步输出队列 | 同步 LLM 多进程路径 |
| `AsyncMPClient` | 后台进程 | ZMQ + asyncio output queue | 在线 AsyncLLM 路径 |

Client 暴露的动作不只 `add_request`，还包括 abort、reset prefix cache、profile、sleep/wake 与 utility call。也就是说，它是 EngineCore 的控制协议，而不只是 token 队列。

### EngineCore

EngineCore 持有 scheduler、KV cache manager 的调度状态并协调 executor。多进程形态中的 `EngineCoreProc` 运行 busy loop：不断接收新增/取消请求，只要存在工作就执行下一步，并把紧凑输出送回 client。

它回答三个问题：

1. 本轮哪些请求推进多少 token；
2. 这些 token 使用哪些 KV blocks 与其他缓存资源；
3. worker 返回结果后，请求状态如何更新、哪些输出可以提交。

### Executor 与 GPU worker

Executor 把一次 `SchedulerOutput` 下发到一个或多个 worker。TP/PP 等分布式后端决定进程与通信组织，但 EngineCore 看到的是统一的执行接口。

每个 GPU 通常由独立 worker process 管理。Worker 负责：

- 选择设备并初始化 distributed environment；
- 加载模型权重；
- profiling 后确定可用于 KV Cache 的显存；
- 初始化 cache；
- 执行 collective RPC 与模型前向。

### ModelRunner

每个 worker 内的 ModelRunner 更靠近 tensor 与 kernel。它把 scheduler 给出的 metadata 转成 GPU 输入，维护 attention backend 所需结构，选择 CUDA Graph shape，运行模型并完成 sampling 或返回待采样结果。

因此，“worker 执行模型”和“model runner 准备一次模型调用”是相邻但不同的职责。

## 一条在线请求怎样穿过多进程边界

假设客户端发送：

```json
{
  "model": "example-model",
  "messages": [{"role": "user", "content": "解释 PagedAttention"}],
  "stream": true,
  "max_tokens": 256
}
```

### 1. API server 建立 frontend 状态

Server 校验请求，应用模型对应的 chat template，得到 prompt token IDs，并建立 request ID 与流式输出队列。若包含图片，还会在这里加载与预处理媒体。

### 2. 转换为 `EngineCoreRequest`

Frontend 将 token IDs、sampling params、需要的多模态输入、LoRA/priority 等元数据整理为 core request。EngineCore 不需要知道原始 HTTP connection，也不负责把 token 解码成字符串。

### 3. Client 发送控制消息

在 `AsyncMPClient` 路径，request 被 msgpack 序列化，经 ZMQ input socket 送往 EngineCore。消息带有 request type 与身份信息；ADD、ABORT 与 utility call 使用同一控制边界，但有不同类型。

大模型权重和 KV Cache 不经过这个 socket。它们已经位于 worker/GPU；IPC 传输的是请求与输出 metadata，避免每步复制大张量。

### 4. EngineCore 收到并入队

Core 将传输结构转换成 scheduler request，计算可复用 prefix blocks，随后放入 waiting queue。此时请求不一定立即执行：它仍受 token budget、KV blocks、优先级和多模态 encoder 资源约束。

### 5. Scheduler 形成一轮工作

Scheduler 为 running 与 waiting request 分配 token 数，输出近似这样的逻辑映射：

```python
{
    "req-A": 1,      # decode
    "req-B": 1,      # decode
    "req-C": 1022,   # chunked prefill
}
```

同时给出新/恢复请求信息、KV block 映射、已完成 request IDs、encoder input、speculative token 等 worker 所需状态。

### 6. Executor 运行 worker

Worker 的 ModelRunner 构造 input batch，执行 embedding、各 Transformer 层、attention/MLP 与 logits/sampling。多 GPU 时，TP/PP collective 发生在 worker 拓扑中，而不是 API server。

### 7. Scheduler 提交结果

ModelRunner output 返回 EngineCore。Scheduler 增加请求的 `num_computed_tokens`，接纳实际 sampled token，检查 EOS、长度与停止条件，释放已完成请求的 KV blocks，并生成 `EngineCoreOutputs`。

### 8. Frontend 处理并流式返回

Output socket 的接收任务按 request ID 将结果交给 OutputProcessor。它负责 detokenize、stop string、logprobs 的对外格式以及 finish reason，再唤醒对应的 API stream。

这条链路中，GPU 可以在 frontend 消费上一轮输出时继续形成下一轮批次；网络慢客户端不必直接阻塞 EngineCore busy loop。

## In-process 与 multi-process 到底差在哪里

`InprocClient` 直接持有 `EngineCore`：

```text
caller
  -> add_request(): direct call
  -> get_output(): engine_core.step_fn()
```

调用方拉取输出时顺便驱动一步 core。它没有 ZMQ 序列化、后台 core 进程和 output thread，调用栈更短，适合调试同步语义与某些离线路径。

Multi-process client 则是：

```text
frontend process                     EngineCore process
----------------                     ------------------
add_request -> ZMQ input ----------> request queue
                                       busy loop
output queue <- ZMQ output <--------- EngineCoreOutputs
```

Core 能独立于 frontend 持续调度 GPU，但系统也多出：

- 启动与 ready handshake；
- request/output serialization；
- socket 队列与 backpressure；
- output receiver thread/task；
- process liveness monitor；
- shutdown、timeout 与异常传播。

所以多进程的优势是隔离和重叠，不是“没有成本”。当 CPU 紧张或容器只给很少 core 时，API、EngineCore 与 worker 争抢 CPU，仍会使 TPOT 抖动。

## V1 的进程拓扑怎样计算

官方架构文档给出的线上 V1 典型拓扑是：

- API server：默认数量与 DP size 对齐，也可显式设置；
- EngineCore：每个 data-parallel rank 一个；
- GPU worker：每个 GPU 一个；
- DP coordinator：DP 大于 1 时存在，用于负载均衡及 MoE 同步协调。

用 $A$ 表示 API server 数，$DP$ 表示 data parallel size，$TP$、$PP$ 分别表示 tensor/pipeline parallel size：

$$
N_{GPU\ worker}=DP\times TP\times PP
$$

$$
N_{process}
= A + DP + N_{GPU\ worker}
+ \mathbb{1}[DP>1]
$$

例如单机 4 GPU、TP=4、DP=1 的典型服务包含 1 个 API server、1 个 EngineCore 与 4 个 worker，共 6 个主要进程。

若 8 GPU 配置 TP=2、DP=4，并让 API server 默认对齐 DP，则为 4 API + 4 Core + 8 worker + 1 coordinator，共 17 个主要进程。

这也是 CPU sizing 不能只看“有几张 GPU”的原因。每个 EngineCore 都有调度热循环，API 还可能为媒体加载使用多个线程，worker 则需要准备 batch 与发起 kernel。

## 统一 token 调度是 V1 的核心简化

传统描述常把请求分成 prefill queue 与 decode queue，再为两套对象设计不同状态机。V1 更统一地维护两个进度：

- `num_computed_tokens`：KV/模型已经实际计算到哪里；
- `num_tokens_with_spec`：prompt、已生成输出与当前 speculative tokens 总共要求计算到哪里。

本轮需要追赶的工作量近似为：

$$
num\_new\_tokens
= num\_tokens\_with\_spec
- num\_computed\_tokens
$$

Scheduler 再在全局 token budget 内为各请求分配实际数量。

这个抽象自然覆盖：

- 新 prompt：差值很大，按预算做 chunked prefill；
- 普通 decode：差值通常为 1；
- prefix cache hit：命中的 token 已视作 computed；
- speculative decoding：候选使目标进度一次增加多个 token；
- 未来的 jump decoding：也可表现为进度跳跃。

统一并不等于完全不区分工作成本。Prefill、decode、encoder 和 speculative verify 对 GPU 时间、KV blocks 的需求仍不同；简化的是调度状态表达，而不是硬件成本模型。

## 一个 1024-token budget 的例子

当前有：

- $D_1,D_2$：两个运行中 decode，各需要 1 token；
- $P$：新请求，剩余 1800 prompt token；
- 本轮最大 scheduled token 为 1024。

一种调度结果是：

```text
D1: 1
D2: 1
P : 1022
```

$P$ 剩下 778 prompt token 留给下一轮。这就是 chunked prefill 与 decode 共批次。它避免完整 1800-token prefill 一次占住 GPU，也避免在有 prefill 时让 decode 完全停顿。

但 token budget 不是唯一限制。假设 $P$ 还需要 120 个 KV blocks，当前只有 60 个可用，scheduler 就必须缩小 chunk、抢占其他请求或让它继续等待。多模态请求还受 encoder cache 与 input budget 约束。

## Prefix caching 发生在调度与 cache manager 之间

Automatic Prefix Caching 把完整 KV block 与其 token 内容、父块和相关配置组合成 hash。新请求进入时，KVCacheManager 查找连续命中的 blocks：

```text
prompt blocks: [A][B][C][D]
cache hit:     [A][B]
need compute:        [C][D]
```

命中的 token 增加 `num_computed_tokens`，scheduler 只为未命中部分分配 token budget。多个请求可通过引用计数共享同一物理 block。

这并不等于文本相同就一定命中：

- tokenization 必须一致；
- 复用以完整 block 为单位；
- LoRA、cache salt、多模态 hash 等会进入身份；
- block 可能已经被 eviction；
- 请求若需要 prompt logprobs，当前 V1 行为可能为得到 logprobs 而重算完整 prompt。

监控时除了 request hit rate，还要统计实际跳过的 token 数。一个只命中很短 block 的请求，与复用 100K prompt 的请求，价值完全不同。

## 取消请求如何贯穿边界

客户端断开后，API server 不能只停止发送文本。它必须经 EngineCoreClient 发送 ABORT：

```text
HTTP disconnect
 -> frontend marks request closed
 -> ABORT(request_id)
 -> scheduler removes waiting/running state
 -> KVCacheManager releases blocks
 -> worker receives finished request IDs
 -> frontend ignores any late output
```

这里容易出现竞态：ABORT 可能在 GPU 正执行本轮 batch 时到达，输出也可能已经在 ZMQ 路上。安全实现要让 request ID 生命周期单调，并把迟到结果识别为已取消，而不是错误投递给恰好复用同一业务 ID 的新请求。

被取消请求占用的 KV block 必须最终释放；但当前 GPU kernel 通常不能从批次中途被撤回，本轮已发生的计算只能作为 sunk cost。

## Backpressure 应该放在哪几层

当到达率持续大于服务率，任何无限队列最终都会耗尽资源。多进程只会把积压位置分散开，并不会消除它。

至少要观察三层：

### API admission

限制并发、排队时间与请求大小；过载时尽早返回明确状态，而不是让所有请求在内存里等待到超时。

### Client/IPC queue

Socket high-water mark 与内部 asyncio queue 需要上限。否则 API 进程仍会接受大量 request object，即便 EngineCore 已完全饱和。

### EngineCore scheduler

监控 waiting/running 数、scheduled token、KV cache usage、preemption 与 TTFT/TPOT。调度器有空闲 token budget 不代表有足够 KV block，GPU 利用率高也不代表 SLO 健康。

流式输出端还存在反向 backpressure：慢客户端的文本 buffer 不能无限增长，更不能拖慢共享 OutputProcessor。

## 故障边界更清楚，也更需要协议

多进程后，失败不再只是一次 Python 异常。

### EngineCore 初始化失败

Core 要等待 worker 加载权重、profile cache 并报告 ready。Client 有启动超时并接收 ready response，其中包括校准后的 max model length、KV block 数等。大型模型加载慢时，应调整有依据的 ready timeout，而不是把所有失败都当成超时。

### Worker 或 Core 意外退出

Monitor 检测后台进程 liveness，将 client 标记为 dead，后续调用抛出 `EngineDeadError` 并清理资源。当前正在处理的请求应整体失败；GPU/KV 状态通常不能在另一个新进程中透明接续。

### Frontend 退出

若 API process 崩溃，EngineCore 可能短暂保留孤儿请求。多 API、多 EngineCore 时需要清楚的 client identity、连接关闭处理和请求所有权。

### 优雅关闭

关闭顺序要停止 admission、abort/等待在途请求、停止 core busy loop、关闭 workers/sockets/receiver tasks。直接杀死父进程容易留下子进程和 GPU context。

## 从 V0 或旧 V1 迁移时不要照抄内部类

V1 内部重构速度很快，`EngineCoreRequest`、client constructor 与 scheduler output 都不是稳定的用户 API。迁移应从官方入口和目标版本文档出发：

```bash
vllm serve <model> [supported options]
```

而不是在业务代码里手工实例化 `EngineCoreProc` 或拼 ZMQ message。

升级验证至少包含：

- 目标模型、量化、LoRA、多模态与硬件插件支持；
- chunked prefill 默认行为；
- prefix cache 与 prompt logprobs 组合；
- logprobs 是 raw 还是 processed 的语义差异；
- CUDA Graph 额外显存；
- 已移除的 V0 功能；
- FCFS/priority scheduling 与业务优先级；
- 真实流量下 TTFT、TPOT、吞吐、OOM 和 preemption。

当前官方 V1 guide 已明确 V0 完全 deprecated，因此新部署的合理方向是适配 V1 语义；但每次升级仍需以所锁定版本的 release notes 和 feature matrix 为准。

## 怎样定位一次延迟抖动

清晰分层的另一好处是可以按边界定位：

| 现象 | 优先观察 |
| --- | --- |
| HTTP 接收慢、GPU 空闲 | API CPU、tokenizer、媒体加载、event loop |
| request 已接收但 TTFT 增长 | IPC queue、waiting queue、prefill budget、KV blocks |
| EngineCore step 耗时高 | scheduler CPU、请求数、cache lookup、serialization |
| GPU 执行慢 | ModelRunner shape、kernel、collective、CUDA Graph miss |
| token 已生成但客户端晚收到 | output socket、detokenize、stream backpressure |
| 取消后显存不降 | abort 传播、finished IDs、KV block refcount |

端到端 trace 应携带相同 request ID，并记录 frontend enqueue、core receive、schedule、execute、core output、frontend emit 等时刻。只有总 TTFT 数字，很难区分是 GPU 慢还是请求根本没进入 core。

## 小结

vLLM V1 EngineCore 的意义可以概括为三层解耦：

1. Entrypoint 处理协议、tokenization 与输出，EngineCore 管理生成状态；
2. Scheduler 决定 token 与 KV 资源，Worker/ModelRunner 负责 tensor 和 GPU；
3. EngineCoreClient 将同一语义映射为进程内调用或 ZMQ 多进程通信。

统一的 token-progress 表达又让 chunked prefill、prefix caching 与 speculative decoding 落在同一调度模型里。多进程拓扑为 API 与 GPU 调度提供隔离和重叠，但也要求显式处理 IPC backpressure、取消竞态、进程健康、CPU sizing 与优雅关闭。

真正理解 V1，不是背下“API server + EngineCore + worker”三个框，而是能解释一条 request ID 在每个边界携带什么状态、谁拥有 KV Cache、谁能提交 token，以及任何一层失败后谁负责释放资源。

## 参考资料

- [vLLM Architecture Overview](https://docs.vllm.ai/en/latest/design/arch_overview/)
- [vLLM V1 User Guide](https://docs.vllm.ai/en/latest/usage/v1_guide/)
- [vLLM V1: A Major Upgrade to vLLM's Core Architecture](https://vllm.ai/blog/2025-01-27-v1-alpha-release)
- [EngineCoreClient source](https://github.com/vllm-project/vllm/blob/main/vllm/v1/engine/core_client.py)
- [V1 Scheduler source](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py)
- [EngineCore source](https://github.com/vllm-project/vllm/blob/main/vllm/v1/engine/core.py)
