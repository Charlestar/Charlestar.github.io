---
layout: post
title: "异步连续批处理：CPU/GPU 重叠与正确同步"
subtitle: "沿着相邻两个 Engine Step 拆解调度流水线"
date: 2026-05-16 12:00:00 +0800
last_modified_at: 2026-09-03
author: iStar
catalog: true
series: serving-scheduling
series_order: 60
technology_year: 2026
mathjax: true
tags: [推理调度, GPU优化, LLM推理]
---

LLM serving 的 GPU 每执行完一个 decode step，CPU 都要处理采样结果、更新请求状态、分配 KV slot、组织下一批 metadata，再发起下一轮模型计算。如果这些动作完全串行，哪怕每轮 CPU 只停顿几毫秒，数百个输出 token 累积后也会形成明显的 GPU 空洞。

异步连续批处理试图把这段空洞隐藏起来：GPU 正在执行 step $t$ 时，CPU 和 worker 同时准备 step $t+1$。听起来只是“双缓冲”，实际难点却在于下一步依赖当前采样结果：哪个请求已经 EOS、哪个要继续、哪个 token 要写入哪个 KV 位置，都可能尚未确定。

所以这项优化的核心不是“多开一个 CUDA stream”，而是重新划分哪些状态可以提前准备、哪些必须等待，以及允许多少未结算 token 同时在流水线中飞行。

## 从静态 Batch 到 Continuous Batching

离线静态 batch 通常让一组序列一起生成，直到最长序列结束。假设四个请求分别输出 20、50、100 和 500 token：

```text
request A: #################### done
request B: ################################################## done
request C: ####################################################################...
request D: ####################################################################...
```

A 完成后，如果 batch shape 不变，它的位置会 padding/空转；B、C 也陆续留下空槽。短请求还必须等最长请求结束才能让下一批进入。

Continuous batching 改为每个 engine step 重新组合工作：

```text
step 0: [A, B, C, D]
...
A ends
step n: [E, B, C, D]   # 新请求 E 加入
...
B ends
step m: [E, F, C, D]
```

请求可以在 token 边界加入和退出，GPU batch 更紧凑。但代价是 scheduler 每轮都要维护动态状态。

## 一个同步 Engine Step 做了什么

先看最直接的同步循环：

```text
1. schedule waiting/running requests
2. allocate KV slots and build metadata
3. copy/update GPU inputs
4. launch model forward
5. wait for logits/sampling result
6. read sampled tokens
7. update request state, EOS, stop, lengths
8. stream committed output
9. repeat
```

把每步粗略分为 CPU 准备 $C_t$、GPU 执行 $G_t$ 和 CPU 收尾 $F_t$，同步时间线为：

```text
time ─────────────────────────────────────────────────────►

CPU: [C0]       [F0 C1]       [F1 C2]       [F2 C3]
GPU:     [ G0 ]         [ G1 ]         [ G2 ]
          ↑ idle          ↑ idle          ↑ idle
```

GPU 两轮之间的 idle gap 可能来自 Python scheduler、block table 组装、CPU-GPU copy、`.item()` 读取或 sampling 后处理。模型越小、batch 越轻，固定 CPU 开销占比通常越显眼。

## 异步调度希望得到的时间线

理想情况是：

```text
time ─────────────────────────────────────────────────────►

GPU:     [     G0     ][     G1     ][     G2     ]
CPU: [C0]   [prepare C1] [F0/prepare C2] [F1/prepare C3]
```

step 0 启动后，host 不再等待完整 GPU 完成，而是准备下一批能提前确定的部分。当 G0 结果真正影响 C1 时，再用最小范围的 event/future 建立依赖；其他工作继续重叠。

稳态每步时间从近似

$$
T_{sync}\approx C+G+F
$$

变为理想下的：

$$
T_{async}\approx \max(G,\ C+F)+T_{unhidden}
$$

$T_{unhidden}$ 是无法隐藏的依赖、队列和同步。公式只是直觉：如果 GPU step 远长于全部 host 工作，CPU 开销可被大量隐藏；若 CPU 本身更慢，GPU 仍会等它。

## 下一步究竟依赖当前步的什么

假设 request A、B、C 正在 decode。step $t$ 的目标模型和 sampler 尚未完成时，CPU 已经知道：

- 三个请求当前的上下文长度；
- 每个请求的最大长度与优先级；
- 下一轮最多需要多少 token budget；
- 哪些等待请求可能被接纳；
- 当前 KV block 所有权和剩余容量。

但它还不知道：

- A 是否刚生成 EOS；
- B 的新 token ID 是什么；
- C 是否命中 stop string；
- 新 token 会让 grammar 进入哪个状态；
- 推测解码本轮究竟提交几个 token。

因此，异步实现可以先做容量预留、候选 batch 框架和静态 metadata；依赖新 token 的位置更新、输入 ID、终止判定则必须等待 device-side 结果或在 GPU 上直接消费。

这类设计的目标之一，就是尽量避免把采样结果同步回 CPU 再决定一切。若下一步输入准备、长度更新或 token compact 能在 GPU 上完成，host 可以更早发起后续工作。

## In-flight Batch：已经调度但尚未结算

同步循环同一时刻只有一个 batch 执行。异步流水线至少可能有两个：

```text
batch t:   GPU running, output not settled
batch t+1: scheduled/reserved, waiting or being prepared
```

这会引入 **in-flight state**。对于 request A，可能已经为未来 token 预留 KV slot，但当前 token 是否正式提交还未知。cache manager 不能把这些 slot 同时分配给其他请求，也不能按“已处理长度”过早释放滑动窗口 block。

容量估算因而不再只看一个 `max_num_batched_tokens`。若最多有 $N_f$ 个并发 batch，每批最多 $T_b$ 个 token，运行时需要为未结算工作保留上界：

$$
T_{inflight}\le N_f\times T_b
$$

实际实现可更精细，但原则不变：重叠通过额外 buffer、KV headroom 和状态复杂度换取时间。若缓存接近满载，异步过度接纳会造成抢占/重算，吞掉重叠收益。

## 一步延迟的终止会发生什么

假设 A 在 step $t$ 生成 EOS，但 step $t+1$ 已经将 A 调度进另一个 in-flight batch。系统有几种选择：

1. 在 $t+1$ 启动前拿到终止信号，将 A 从 metadata 中移除；
2. 允许一次可控的多算，但不提交 EOS 后的 token；
3. 在 GPU 上用 active mask 让后续位置失效；
4. 让 scheduling 保守地不超过剩余长度边界。

无论哪种，用户可见输出、KV 正式长度和计费都只能包含 committed token。多算可以是性能权衡，绝不能变成多返回 token 或越过 `max_tokens`。

异步系统最常见的正确性风险正来自“scheduled/computed/committed”三个长度混用。建议把它们明确建模：

```text
num_scheduled >= num_computed >= num_committed
```

只有依赖完成、采样验证和停止条件处理后，computed 才能进入 committed。

## 双缓冲不是简单交换两个 Tensor

最小双缓冲可表示为：

```text
Buffer A: GPU consumes metadata for step t
Buffer B: host/device prepares metadata for step t+1
event A_done -> Buffer A can be rewritten
swap(A, B)
```

一个 buffer 通常包含：

- input token IDs / embeddings；
- positions 与 sequence lengths；
- slot mapping / block table；
- attention 与 sampling metadata；
- request-to-row mapping；
- output/logits/sampled token storage。

所有权必须清楚。CPU 不能在 GPU 尚未读完 Buffer A 时重写；GPU 的准备 stream 也不能在 copy 完成前消费 Buffer B。请求在两步之间完成、取消或被抢占时，B 中对应行与 KV reservation 必须失效。

如果为了 CUDA Graph 使用固定最大 buffer，`active_count` 和映射需要每轮准确更新；残留的旧 request ID 或长度可能让 kernel 访问已经释放的 block。

## CUDA 异步不等于并发执行

CUDA kernel launch 通常对 host 异步：API 返回时，GPU 工作可能尚未开始。**异步**描述调用方不等待，**并发**描述两项工作实际在时间上重叠，两者不是同义词。

### Stream 只表达顺序和可并发机会

同一 CUDA stream 内操作按入队顺序执行。不同 stream 之间没有默认的全局先后关系，可以重叠，但是否真能同时运行取决于：

- kernel 是否占满全部 SM/寄存器/shared memory；
- memory bandwidth 是否已经饱和；
- copy engine 和硬件能力；
- stream priority 与 runtime 调度；
- 是否存在 event、默认 stream 或内存依赖。

把两个大 GEMM 放在两条 stream，不会保证速度翻倍；它们可能争抢相同 Tensor Core 和 HBM，最终串行或互相拖慢。

### Event 表达最小依赖

假设准备 stream 写好下轮 input，compute stream 必须等待：

```python
prep_done = torch.cuda.Event()

with torch.cuda.stream(prep_stream):
    prepare_next_inputs(...)
    prep_done.record(prep_stream)

compute_stream.wait_event(prep_done)
with torch.cuda.stream(compute_stream):
    run_model(...)
```

`wait_event` 让 compute stream 中后续工作等待对应事件，不必让 host 执行全设备 `synchronize()`。事件只保证执行依赖，不理解“request A 已取消”这类业务语义。

### 异步 copy 还有前提

Host↔Device 传输要与计算可靠重叠，host buffer 通常需要 page-locked/pinned memory，硬件还需支持相应 copy engine。调用名带 `non_blocking`/`Async` 不足以证明时间线已经重叠，仍需 profiler 验证。

## 三层异步不要混为一谈

### API 异步

`async def`、事件循环和流式 HTTP 让一个线程可以服务多个连接。它主要改善网络层并发，不会自动改变 GPU 调度。

### Engine 异步

Scheduler/worker 在 GPU 执行 step $t$ 时准备 step $t+1$，使用 batch queue、future 与 in-flight state。这是本文的核心。

### Device 异步

CUDA streams、events、graphs 和 async copy 表达 GPU 内部/CPU-GPU 的执行依赖。

三层可以组合，但每层有独立瓶颈。Python API 完全异步时，scheduler 仍可能每 token `.item()`；device 上有两条 stream 时，host 也可能迟迟不给下一批工作。

## 常见隐式同步点

PyTorch/CUDA 代码中一些操作会迫使 host 等 GPU：

- 对 CUDA tensor 调用 `.item()` 并立即做 Python 分支；
- `.cpu()`、`.numpy()` 或同步 device-to-host copy；
- 打印 CUDA tensor 内容；
- 某些动态 shape/allocator/exception 检查；
- 默认 stream 与自定义 stream 之间的保守依赖；
- 为错误计时而每轮 `torch.cuda.synchronize()`；
- logits processor 必须在 CPU 读取完整数据。

不能机械删除同步。若 CPU 确实需要 sampled token 才能正确推进 grammar，等待是必要依赖；优化方向是把相关逻辑搬到 device、缩小 copy、批量处理，或推迟到更晚边界，而不是读取未完成结果。

## Sampling 为什么是异步化关键

模型 forward 产生 logits 后，还要执行 temperature、top-k/top-p、grammar mask 和随机采样。若 sampling 在 GPU 完成且结果直接写入下步 input buffer，token 可以不经 CPU 往返：

```text
GPU forward -> GPU sampling -> GPU input update -> next forward
                          │
                          └─ small committed output copied for streaming
```

但请求终止、stop string、tool parsing 和业务回调常在 CPU。可以将纯 token 级 EOS/长度检查留在 GPU，把较复杂字符串/协议检查异步放到 host；关键是定义在检查完成前最多允许多少 in-flight 工作，以及如何丢弃未提交结果。

结构化输出还依赖 FSM 状态。如果 FSM 更新仍在 CPU，每步可能产生同步；若搬到 GPU，则要保证状态与已接受 token 一致，推测解码拒绝分支也必须回滚。

## 与推测解码组合为什么更复杂

普通 decode 每请求每步通常提交一个 token。推测解码可能提交 1 到 $k+1$ 个，直到目标验证完成前无法知道准确长度。

异步 scheduler 若提前准备下一轮，需要为变长结果预留：

- 多个可能的 KV slots；
- 候选树/验证 metadata；
- grammar、stop 与 position 的最大推进；
- 被拒 suffix 的释放路径。

这正是 async-first model runner 重构强调 GPU-native preparation 和零 host sync 的原因之一：让 device-side accepted length 直接驱动后续 buffer，而不是先 `.item()` 回 CPU 再重建全部输入。实现支持度仍应以目标引擎版本和 feature matrix 为准。

## 取消、抢占和错误如何进入流水线

### 用户取消

已经在 GPU 执行的 kernel 通常不能低成本只撤销 batch 中一行。系统标记请求 cancelled，不再提交/流式返回结果，并在安全 event 后释放其 KV 和 buffer slot。

### KV 不足与抢占

未来 batch 已预留 blocks 时，cache manager 计算空闲容量必须包含 in-flight reservation。抢占某请求也要确保没有仍在运行的 kernel 访问其 blocks。

### CUDA 异步错误

Kernel 错误可能在后续同步点才被观察到，堆栈看起来落在无关操作。调试时可以临时使用 blocking 模式定位，但不应把全局同步留在生产热路径。Engine 还要决定：失败 batch 中哪些请求可重试，哪些 KV 写入已不可信。

## Backpressure：流水线不能无限提前

若 host 准备速度远快于 GPU，持续把 batch 推入队列会占用越来越多 metadata、KV reservation 和输出 buffer，也让取消与优先级调整变迟钝。

因此 batch queue 应有明确上限：

```text
if in_flight_batches >= capacity:
    scheduler waits / yields
```

常见异步重叠只需要少量并发 batch。更深流水线是否有益取决于 PP stage、CPU/GPU 比例和网络；深度增加会放大 stale scheduling 与内存压力。

`max concurrent batches` 与 `max_num_batched_tokens` 应共同参与容量规划。吞吐测试中如果只提高队列深度、不计入额外 KV 和延迟，容易得到不可上线的数字。

## 如何证明真的重叠了

功能开关生效不等于性能路径生效。使用 Nsight Systems 或等价 profiler 查看：

```text
CPU thread: schedule / prepare / enqueue
CUDA stream: H2D / metadata kernels / model / sampling
events: wait and record points
GPU gaps: no kernels active intervals
```

应回答：

- G$t$ 执行时，CPU 是否真的在准备 $t+1$；
- 下一轮 kernel 是否紧接上一轮，还是仍有空洞；
- host 在哪里等待，等待哪个 event；
- 多 stream 是否真实 overlap，还是资源饱和后串行；
- H2D/D2H copy 是否与计算重叠；
- async 后多用了多少 buffer 和 KV Cache。

计时时，CUDA Event 适合测 device 工作；端到端 wall clock 适合测用户延迟。异步 launch 后立即读取 CPU 时钟会低估 GPU 耗时，基准结束前必须在正确边界同步。

## 正确性测试比吞吐测试更重要

需要系统覆盖跨 step 状态变化：

1. A 在 $t$ 生成 EOS，B 继续，C 在 $t+1$ 加入；
2. B 在 in-flight 时被取消；
3. KV pool 接近满载，D 需要抢占；
4. prefix hit 让某请求跳过部分 prefill；
5. 推测解码分别接受 0、部分和全部候选；
6. grammar/stop string 跨 token 边界命中；
7. CUDA Graph buffer 重用后 request 行发生变化；
8. 任一步注入异常并恢复。

对每个请求核对：输出 token、scheduled/computed/committed length、KV block 所有权、streamed count 和最终释放次数。与同步基线的 greedy token 序列对比，是发现竞态的有效手段。

## 什么时候收益有限

如果一次 GPU forward 很长，CPU 早已在其间完成准备，异步化只能隐藏很小一段；如果模型运行受 GPU 计算完全饱和，多开并行 kernel 也不会增加吞吐。

其他限制包括：

- workload 很低，GPU 大部分时间等待新请求而非 scheduler；
- 高并发 batch 已大，CPU overhead 被摊薄；
- 每步必须执行复杂 CPU logits processor；
- 网络/Tokenizer 才是主要瓶颈；
- 额外 in-flight memory 导致 KV 抢占增加。

收益可能主要表现为减少 GPU idle gap、降低 TPOT 尾延迟，而非峰值 tokens/s 大幅提高。文章或 benchmark 不应给出脱离模型、batch、CPU 和 feature 组合的固定百分比。

## 小结

Continuous batching 解决“哪些请求每轮一起执行”，异步调度解决“相邻两轮的 host 与 device 工作如何重叠”，CUDA streams/events 则表达设备任务的顺序与依赖。三者相关，却不是同一个开关。

正确的异步系统必须显式区分 scheduled、computed 和 committed 状态，为 in-flight batch 保留 KV 与 buffer，使用 event 保护真实数据依赖，并在 EOS、取消、抢占和推测拒绝后只提交合法 token。优化是否成功，最终要在时间线上看到 GPU gap 缩短，同时保持与同步基线一致的请求语义。

## 参考资料

- [CUDA Programming Guide: Asynchronous Execution](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html)
- [PyTorch CUDA Semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html)
- [vLLM Model Runner V2：Async-first 设计](https://vllm.ai/blog/mrv2)
- [vLLM 官方仓库](https://github.com/vllm-project/vllm)
