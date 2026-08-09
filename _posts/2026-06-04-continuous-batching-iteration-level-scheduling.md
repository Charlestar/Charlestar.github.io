---
layout: post
title: "Continuous Batching：为什么 Batch 可以动态变化"
subtitle: "从请求粒度到 Iteration-Level Scheduling，理解生成式推理的批次重组"
date: 2026-06-04 09:00:00 +0800
last_modified_at: 2026-08-09
author: iStar
catalog: true
series: serving-scheduling
series_order: 10
technology_year: 2022
mathjax: true
tags: [推理调度, LLM推理]
---

把多个请求组成 batch，是提高 GPU 利用率最常见的办法。但语言模型的生成请求有一个特殊之处：模型执行一次通常只产生下一个 token，而不同请求要生成的 token 数又不相同。

如果 batch 在开始执行后就不再变化，短请求即使已经生成完毕，也要等最长的请求结束才能离开；刚到达的请求即使遇到空出来的计算位置，也只能继续排队。batch 越大，这种等待越容易被放大。

Continuous Batching 改变的不是矩阵乘法本身，而是调度边界：**引擎不再把“完整生成一个请求”视为不可分割的任务，而是在一次或少数几次模型迭代后重新决定下一轮 batch。**

一次重组看起来只是把完成的请求移走、把等待的请求补进来，背后却需要同时解决四个问题：

1. 每个请求已经计算到哪个 token；
2. 下一轮应该让哪些请求继续运行；
3. 请求的历史 KV Cache 放在哪里；
4. 形状不断变化的输入怎样高效交给 GPU。

本文从固定批次的阻塞问题开始，逐步建立 iteration-level scheduling、请求状态、token budget 与 KV Cache 之间的关系，最后说明应如何衡量 Continuous Batching 是否真的改善了线上服务。

## 生成请求为什么不适合固定批次

先看一个普通分类模型。服务端收集 16 张图片，将它们组成 batch，执行一次模型前向，16 个结果一起返回。虽然图片预处理时间可能不同，但进入模型后，每个样本经过的网络结构与执行次数基本相同。

自回归语言模型则不同。给定 prompt：

```text
解释什么是 KV Cache
```

模型不会在一次前向中直接得到完整段落，而是重复执行：

```text
prompt                  -> “KV”
prompt + “KV”           -> “ Cache”
prompt + “KV Cache”     -> “ 是”
prompt + “KV Cache 是”  -> ...
```

设请求 $i$ 最终需要生成 $O_i$ 个 token。忽略推测解码等一次可提交多个 token 的机制，它至少要经历 $O_i$ 次 decode 迭代。不同请求的 $O_i$ 由停止条件、模型输出和用户参数共同决定，在执行前并不能准确知道。

现在同时到达三个请求：

| 请求 | 输入长度 | 最终输出长度 | 完成所需 decode 轮数 |
| --- | ---: | ---: | ---: |
| A | 128 | 8 | 8 |
| B | 256 | 64 | 64 |
| C | 64 | 20 | 20 |

若执行引擎接收整个 batch 后，一直运行到三条序列全部结束才把控制权交回服务端，那么 A 在第 8 轮已经完成，却还要等 B 再运行 56 轮。第 10 轮到达的请求 D，也必须等 B 完成后才有机会进入 GPU。

这就是固定批次的 **head-of-line blocking**：batch 的生命周期被最长请求决定，提前完成与中途到达都无法改变当前批次。

可以把固定批次的占用时间粗略写成：

$$
T_{static}\approx T_{prefill}+\max_i(O_i)\cdot T_{decode}
$$

这里的 $T_{decode}$ 会随 batch size 和上下文长度变化，因此不是常数；公式只强调一个事实：其他请求是否已经完成，不会缩短这个 batch 的生命周期。

## “动态批处理”为什么还不是答案

很多推理服务在模型执行前会设置一个短暂等待窗口，例如最多等待 5 ms，收集更多请求后一起运行。这通常叫 dynamic batching：batch 在**发射之前**动态形成，发射之后仍然固定。

它适合一次前向就完成的模型，却没有消除生成任务中的最长请求阻塞：

```text
dynamic batching

arrival queue --等待窗口--> [A B C] --完整生成--> 统一结束
                            batch 内不再变化
```

Continuous Batching 关注的是 batch 在**执行期间**也能改变：

```text
continuous batching

iteration 1: [A B C]
iteration 2: [A B C]
...
iteration 8: [A B C]  -> A 完成
iteration 9: [D B C]  -> D 加入
```

两者都包含“动态”，但动态发生的边界不同。Continuous Batching 也常被称为 iteration-level batching 或 in-flight batching；这些术语在具体系统中的功能范围略有差异，核心都是在在途请求尚未全部完成时重新形成后续批次。

## Orca 把调度边界从请求降到迭代

2022 年 OSDI 论文 Orca 系统化提出了 **iteration-level scheduling**。在传统接口中，服务层把一批请求交给执行引擎，引擎负责把每个请求完整生成完；服务层只有在整个 batch 结束时才重新获得调度权。

Orca 改为每次只让执行引擎运行一轮模型。引擎返回后，调度器会：

- 读取本轮产生的 token；
- 判断请求是否遇到 EOS 或长度上限；
- 立即提交已完成请求；
- 接收这段时间内新到达的请求；
- 重新选择下一轮要执行的请求。

于是调度循环从：

```text
schedule requests -> run until every request finishes -> schedule again
```

变成：

```text
schedule one iteration
-> execute
-> update request states
-> schedule next iteration
```

调度粒度更细之后，新请求最多只需等待当前迭代结束，就有机会被下一轮考虑；短请求也可以在自身完成时立即释放位置，而不再陪跑到最长请求结束。

这里的“一轮”是调度语义，不应机械理解成永远只计算一个 token。普通 decode 常为每个请求计算一个新 token；prefill chunk、speculative verification 或某些多步执行会让一次调度包含更多 token。真正不变的是：**执行一段有界工作后，控制权回到调度器，请求集合可以被重新决定。**

## 为什么早期 Orca 还需要 Selective Batching

允许任意请求中途加入后，batch 内请求的历史长度不再整齐：

```text
A: 已有  32 个历史 token
B: 已有 512 个历史 token
C: 已有 127 个历史 token
```

线性层可以把不同请求当前的 token 沿 batch 维拼在一起，共享一次权重读取和矩阵乘法；attention 却要让每个 query 读取各自不同长度的历史 K/V，直接填充到最长长度会浪费大量计算。

Orca 的方案叫 **selective batching**：对 QKV projection、MLP 等适合的算子做批处理，而对论文狭义定义下的 attention 分别处理不同请求。论文指出，attention 本身不包含模型参数，因此拆开它不会失去跨请求复用权重读取的主要收益。

现代运行时更多使用 paged attention、variable-length attention 与 packed input，在 kernel 层直接表达不同序列长度。实现方式已经演进，但问题没有消失：Continuous Batching 形成的是一个逻辑 batch，运行时仍要为不同请求携带 slot mapping、sequence length 与 KV block table，才能避免把所有序列填充到同一长度。

## Prefill 与 Decode 是两种不同的工作

一次请求进入模型后通常经历两个阶段。

### Prefill：把 prompt 变成状态

Prefill 并行处理输入 token，生成每一层的 K/V，并得到第一个采样位置所需的 logits。若 prompt 长度为 $L_p$，这一步会处理约 $L_p$ 个新 token，矩阵乘法规模较大，通常更偏计算密集。

### Decode：利用状态逐步生成

Decode 每轮通常只给每个请求增加一个 token，但 attention 要读取此前累积的 KV Cache。随着上下文变长，读取量持续增加，单轮工作往往更偏内存带宽受限。

因此，“把一个新请求补进 batch”并不总是只增加一行 decode。新请求首先要完成 prefill，可能一次带来数千甚至数万 token 的工作。如果调度器把完整长 prompt 直接塞进正在 decode 的 batch，已有请求的下一 token 会被长时间推迟。

Continuous Batching 回答的是“何时可以重组 batch”，却没有单独回答：

- prefill 与 decode 能否在同一轮混合；
- 一个长 prefill 每次最多处理多少 token；
- 两类工作冲突时谁优先。

后来出现的 chunked prefill 会把长 prompt 切成有界片段，与 decode 共同使用 token budget。它是 Continuous Batching 上的进一步调度能力，而不是二者的同义词。

## 一个请求必须保存哪些状态

固定 batch 可以让执行引擎在内部隐式维护状态；按迭代重组后，状态必须成为调度器能够识别和转移的对象。一个简化请求至少包含：

```python
RequestState(
    request_id="req-C",
    prompt_token_ids=[...],
    output_token_ids=[...],
    num_computed_tokens=384,
    status="RUNNING",
    kv_block_ids=[7, 19, 42, ...],
    sampling_params={...},
    arrival_time=...,
    priority=...,
)
```

这些字段大致分成四类。

### 生成进度

调度器需要知道目标序列中哪些 token 已经完成模型计算，哪些 token 等待计算。只记录“处于 prefill 或 decode”往往不够，因为 prefix cache 命中、chunked prefill 和推测 token 都可能让进度一次移动不同距离。

### 持久化模型状态

历史 token 对应的 KV Cache 必须在本轮结束后继续保留。下一轮即使请求在逻辑 batch 中换了位置，也要能通过 block table 找到相同的 K/V。

### 终止状态

模型产生 EOS、达到最大长度、命中 stop token、客户端取消或执行失败，都会使请求离开后续批次。离开时不仅要改一个状态值，还要释放 KV blocks、清理输出队列并防止迟到结果再次提交。

### 调度属性

到达时间、优先级、SLO、adapter、并行采样分支和多模态资源，都会影响请求能否与其他请求共批次。Continuous Batching 允许动态选择，并不意味着任意请求都能无条件混在一起。

## 请求状态机怎样驱动批次重组

可以用下面的简化状态机理解一次请求：

```text
              admission
                 |
                 v
WAITING ------> RUNNING ------> FINISHED
   ^                |               ^
   |                |               |
   +--- PREEMPTED <-+               +--- cancelled / EOS / limit
```

不同引擎会使用不同命名，关键转移相似。

**WAITING** 表示请求已经被接受，但还没有拿到本轮 token budget 或 KV blocks。

**RUNNING** 表示请求拥有可继续使用的 KV 状态，并被当前或近期迭代推进。它不保证每一轮都一定入选；调度策略可能暂时跳过某些 running request。

**PREEMPTED** 表示显存压力下请求的运行资格或 KV 资源被回收。恢复时可能重算被丢弃的状态，也可能从其他层级重新载入。

**FINISHED** 表示不会再被调度。只有输出提交、资源释放与队列清理都完成，生命周期才真正闭合。

每轮调度的本质就是在这些状态上执行一次事务：先选择候选与预留资源，执行模型，再根据结果提交进度或处理失败。如果“队列里已移除”与“KV block 已释放”不是一致的状态边界，长期运行后就容易出现显存泄漏或重复分配。

## 下一轮 Batch 是怎样形成的

设某一时刻有：

- `running`：已经持有模型状态、可以继续推进的请求；
- `waiting`：新到达或等待恢复的请求；
- `token_budget`：本轮最多允许处理的新 token 数；
- `max_seqs`：本轮最多容纳的序列数；
- `free_kv_blocks`：仍可分配的 KV Cache blocks。

一个只用于解释的调度循环可以写成：

```python
def schedule(running, waiting, token_budget, max_seqs):
    batch = []

    for req in choose_running_requests(running):
        work = next_token_work(req)
        if fits(work, token_budget, max_seqs) and has_kv_capacity(req, work):
            reserve(req, work)
            batch.append(work)
            token_budget -= work.num_tokens

    for req in choose_waiting_requests(waiting):
        work = next_prefill_work(req, token_budget)
        if fits(work, token_budget, max_seqs) and has_kv_capacity(req, work):
            reserve(req, work)
            batch.append(work)
            token_budget -= work.num_tokens

    return batch
```

真实调度器还要考虑 prefix cache、encoder input、LoRA、推测 token、pipeline parallel、优先级和抢占，但约束可以归纳为三种。

### 序列数量约束

`max_seqs` 限制同时维护多少条序列。Decode 请求每条通常只贡献一个新 token，但每条都有独立 metadata、KV block table 与采样状态，因此序列数不能无限增长。

### Token 数量约束

`max_num_batched_tokens` 一类参数限制一轮处理多少新 token。一个 4096-token prefill 和 128 条单 token decode 都占用 token budget，却具有不同的执行特征。

### KV Cache 约束

请求即使满足前两个限制，也可能没有显存保存新增 K/V。调度器必须分配 block；若失败，就需要缩小 prefill chunk、抢占其他请求或暂缓执行。

所以“当前 batch size 是多少”不是一个足够完整的问题。至少要同时说明 request 数、scheduled token 数、上下文长度分布与 KV Cache 占用。

## 用六轮时间线看一次动态变化

假设每轮最多容纳 3 条序列，先忽略长 prefill 的执行时间。初始时 A、B、C 已经完成 prefill：

| 迭代 | 到达/完成事件 | 本轮逻辑 batch | 迭代后状态 |
| ---: | --- | --- | --- |
| 1 | A、B、C 运行 | A B C | 各生成 1 token |
| 2 | D 到达 | A B C | D 在 waiting |
| 3 | A 遇到 EOS | A B C | A 完成并释放 blocks |
| 4 | D 获得资源 | D B C | D 完成 prefill 或首个 chunk |
| 5 | E 到达，C 被取消 | D B C | C 退出，E 等待 |
| 6 | E 获得资源 | D B E | batch 再次重组 |

与固定批次相比，D 不需要等待 B 和 C 全部结束，A 也可以在第 3 轮后立即返回。

不过这里隐藏了一个成本：D 的 prefill 可能远比 A 的一次 decode 重。如果 D 有 8K prompt，第 4 轮可能拉长 B、C 的 token 间隔。因此，能把 D 放进来与应该一次给 D 多少工作，是两个不同问题。

这正是调度技术继续从 Continuous Batching 发展到 chunked prefill、deadline-aware scheduling 与 prefill/decode 分离的原因。

## KV Cache 让请求能够“换座位”

没有 KV Cache，每生成一个 token 都要重新计算完整前缀。对长度为 $L$ 的序列，decode 到下一 token 时需要重复处理此前 $L$ 个 token；Continuous Batching 即使减少了排队，也会被大量重复计算淹没。

KV Cache 保存每层历史 token 的 K/V，使下一轮只需计算新增 token，再让新 query 读取已有 K/V。对请求 $i$，第 $t$ 轮可以抽象为：

$$
q_t = x_tW_Q,
\quad
k_t = x_tW_K,
\quad
v_t = x_tW_V
$$

$$
K_{1:t}=[K_{1:t-1};k_t],
\quad
V_{1:t}=[V_{1:t-1};v_t]
$$

逻辑 batch 位置可以从第 0 行变成第 7 行，但请求的 block table 仍指向原来的 $K_{1:t-1},V_{1:t-1}$。因此 request ID 与 KV 映射才是稳定身份，batch row 只是本轮执行位置。

连续大块分配会带来另一个难题：请求长度未知，预留最大空间浪费显存；按实际长度扩容又容易碎片化。PagedAttention 把 KV Cache 分成固定大小的 blocks，让请求按需增长并在完成后回收。这不是 Continuous Batching 的定义组成，却显著提高了动态请求集合可达到的并发度。

二者的关系可以概括为：

- Continuous Batching 决定**哪些请求**在下一轮运行；
- KV Cache 保存请求跨轮次的**模型状态**；
- Paged KV 管理这些状态占用的**物理空间**；
- attention kernel 根据 block table 读取不同请求的历史。

只实现动态队列而没有可靠的状态和内存管理，batch 很快会被显存容量限制。

## 逻辑 Batch 动态，不等于 GPU 可以接受任意形状

每轮请求集合变化，会让 tensor 的第一维、总 token 数和上下文长度分布变化。GPU kernel 虽然能执行动态 shape，但过度动态会产生额外成本：

- host 端需要重新组装 input IDs、positions 与 slot mapping；
- 不规则 padding 会浪费显存带宽和算力；
- kernel launch 数量过多会放大 CPU 调度开销；
- CUDA Graph 通常需要预先捕获有限的一组 shape；
- 分布式 worker 必须对本轮 batch metadata 保持一致。

现代系统通常通过 packed input 把有效 token 紧密排列，再用 offset、sequence length 和 block table 恢复序列边界。TensorRT-LLM 的官方文档也将 packed input 作为 in-flight batching 的效率前提，因为把单 token decode 填充到最长 prompt 长度会造成严重浪费。

CUDA Graph 场景则可能把实际 batch padding 到最近的已捕获 bucket，例如请求数 13 使用 shape 16。这里的 padding 是运行时为了复用执行图做的受控权衡，与把所有序列 padding 到最长上下文不是同一回事。

因此，一套成熟实现常同时维护两个世界：

```text
调度器：可变长请求集合、精确 token 与 block 状态
运行时：有限 shape bucket、packed tensor、预分配 buffer
```

前者追求灵活，后者追求稳定高效；ModelRunner 的重要职责就是把两者衔接起来。

## Continuous Batching 不保证每轮都接纳新请求

“Continuous” 容易让人误以为请求一到达就会立即加入当前计算。实际上，batch 只能在安全调度点改变，并且仍受资源与策略约束。

新请求可能继续等待，原因包括：

- 本轮 `max_seqs` 已满；
- token budget 已被运行请求用完；
- 没有可分配的 KV blocks；
- 长 prefill 会破坏正在生成请求的延迟目标；
- 请求依赖的 LoRA adapter 或多模态 encoder 资源尚未就绪；
- 优先级策略选择了其他请求；
- pipeline 或 data-parallel rank 尚未到达一致的调度边界。

Continuous Batching 提供的是**每轮重新选择的机会**，不是无条件的即时执行承诺。线上过载时，所有轮次都可以是满的，waiting queue 仍会持续增长。

## 调度策略决定谁得到这个机会

有了逐轮重组能力，调度器还要定义顺序。常见基础策略包括：

### FCFS

按到达顺序推进，请求行为最容易解释，但长 prompt 或长输出可能让后续短请求等待较久。

### Decode 优先

优先让已经开始生成的请求继续推进，通常能改善 inter-token latency；持续到达的 decode 工作也可能压缩新 prefill 的机会。

### Priority

按业务优先级选择请求，适合区分交互流量与离线流量，但需要防止低优先级请求饥饿，并明确抢占成本。

### SLO-aware

根据 TTFT、TPOT deadline 或剩余 slack 决定紧迫度。它更贴近服务目标，却需要可靠的执行时间估计和更复杂的 admission control。

这些不是 Continuous Batching 的替代方案，而是运行在它提供的迭代级决策点之上。本系列后面的 FairBatching 会专门讨论如何把已经取得的生成进度转化为可用的调度余量。

## 取消、流式输出与 Backpressure

逐轮返回控制权还有一个重要收益：系统可以在轮次之间处理外部事件。

### 流式输出

本轮生成的 token 可以立即送入 output processor 并返回客户端，无需等请求完整结束。GPU 同时可以开始调度下一轮，网络发送与模型执行由此形成流水线。

但流式输出不是 Continuous Batching 自动带来的。若 frontend 必须同步完成 detokenize 和网络写入后才允许下一次 engine step，慢客户端仍可能拖住 GPU。引擎需要用有界队列隔离 output path。

### 请求取消

客户端断开后，取消事件应进入调度器。正在执行的 kernel 通常无法从中途移除单个请求，但下一轮可以不再调度它，并释放 KV blocks。

需要处理的竞态是：取消消息与本轮输出可能同时到达。系统必须保证 token 只提交一次、已取消请求不被重新加入、资源最终只释放一次。

### 入口限流

Continuous Batching 提高服务率，却不会让容量变成无限。若到达率 $\lambda$ 长期超过完成率 $\mu$，waiting queue 仍会发散：

$$
\lambda > \mu \quad \Longrightarrow \quad queue\ length \uparrow
$$

因此 API admission、scheduler queue 和 output queue 都需要上限。过载时及时拒绝或排队到有界容量，通常比让所有请求一起超时更可控。

## 它改善了哪些指标

Continuous Batching 最直接消除的是固定批次中的空槽与陪跑等待，但收益会反映到多类指标。

### 吞吐

短请求完成后立即用新请求补位，GPU 更容易维持足够并发。单位时间完成的 request 或 output token 通常会增加。

### TTFT

新请求不必等待整个旧 batch 完成，只需等待可用的后续调度点。但若请求有长 prefill、KV Cache 已满或系统过载，TTFT 仍可能很高。

### TPOT / ITL

更多请求共享一轮执行会提高吞吐，也会增加单轮耗时。batch 扩得过大或混入过重 prefill，已有请求的 token 间隔可能变差。

### GPU 利用率

利用率提高通常是好现象，却不是最终目标。一个始终满载但 P99 TTFT 超出契约的服务，可能只是高效地制造超时。

所以调优时要同时看：

- request throughput 与 output token throughput；
- TTFT 的 P50、P95、P99；
- TPOT/ITL 的 P50、P95、P99；
- 满足 SLO 的 goodput；
- running/waiting request 数；
- scheduled token 数与有效 batch size；
- KV Cache usage、allocation failure 与 preemption；
- scheduler CPU time 和 GPU execution time。

只报告离线总 tok/s，很难判断动态调度是否改善了真实交互体验。

## 一个简单模型解释吞吐与延迟的拉扯

设本轮逻辑 batch 中有 $B$ 条 decode 序列，执行耗时为 $T(B)$。若每条序列提交一个 token，理想 decode 吞吐近似为：

$$
Throughput(B)=\frac{B}{T(B)}
$$

在 GPU 尚未饱和时，增加 $B$ 往往使 $T(B)$ 增长慢于 $B$，所以吞吐提高。但某个请求获得下一 token 的间隔至少包含一轮 $T(B)$；当 $B$ 很大后，$T(B)$ 的增长会直接恶化 TPOT。

Continuous Batching 让系统能够持续把 $B$ 维持在高效区间，却没有告诉系统这个区间的最佳上限。`max_seqs`、token budget、KV capacity 和 SLO 共同决定可接受的工作点。

这也是为什么调参不能只追求最大 batch：

```text
batch 太小  -> GPU 未饱和，吞吐低
batch 合适  -> 吞吐提升，延迟仍在目标内
batch 太大  -> 单轮变慢，排队与尾延迟上升
```

最佳点还会随模型大小、量化方式、GPU、上下文长度和流量分布变化。

## 怎样设计一次可信的对照实验

验证 Continuous Batching 时，最常见的误区是用一组等长 prompt、固定 `max_tokens` 的离线 batch 代替在线流量。等长请求几乎不会暴露提前完成与中途加入的价值。

更合理的实验至少包含：

### 可变到达时间

使用 Poisson、突发或真实 trace 驱动请求，让请求在旧 batch 运行期间到达。否则观察不到 late joining。

### 可变输入和输出长度

混合短问答、长文档与不同输出上限。输出长度差异用于制造 early finish，输入长度差异用于暴露 prefill/decode 干扰。

### 稳态与过载区间

逐步提高 request rate，在相同硬件和模型配置下比较固定批次与连续批次。低负载下两者都可能很快，接近饱和点时差异才明显。

### 一致的终止与缓存条件

确保采样参数、EOS、prefix cache 命中率、KV Cache 容量和 warm-up 一致。否则吞吐差异可能来自输出长度或缓存，而不是调度。

### 时间线观测

为每个 request ID 记录：

```text
arrival
-> admitted
-> first scheduled
-> prefill finished
-> each decode scheduled
-> first token emitted
-> finished / cancelled
```

再把每一轮的 request IDs、scheduled tokens、KV usage 与 GPU duration 对齐。这样才能回答新请求究竟是在排队、等待 KV block，还是已经进入 batch 但被长 prefill 拖慢。

## 常见误解

### “Continuous Batching 就是每来一个请求启动一次模型”

不是。请求通常只能在迭代边界加入，引擎仍会把多个请求组成一次 GPU 执行。过于频繁地启动小 kernel 反而会降低效率。

### “batch size 会一直保持最大值”

不是。请求到达不足、KV Cache 紧张、调度策略限制或 shape bucket 都可能让实际 batch 小于上限。Continuous 描述可重组能力，不描述固定占用率。

### “有了 Continuous Batching 就不需要 PagedAttention”

定义上二者独立，工程上高度互补。前者减少计算槽位浪费，后者减少动态 KV 状态的内存浪费。缺少高效 KV 管理时，可并发请求数会很快触顶。

### “它一定同时改善吞吐和所有延迟”

不是。更大的在途 batch 往往提高吞吐，却可能拉长单轮 decode；prefill 插入也可能恶化 ITL。系统必须根据工作负载在 TTFT、TPOT 与 throughput 之间取舍。

### “Continuous Batching 和 Chunked Prefill 是一回事”

不是。前者允许迭代间改变请求集合；后者把一个 prompt 的 prefill 拆成多段，使它能在 token budget 内与其他工作交错。一个系统可以支持前者而仍一次执行完整 prefill。

### “Streaming 就证明底层使用了 Continuous Batching”

不是。单请求逐 token 返回也可以 streaming；固定 batch 的引擎同样能把中间 token 发给客户端。判断依据应是其他请求能否在在途 batch 未整体结束时加入或离开。

## 从 Orca 到现代推理引擎

Orca 的核心贡献是明确了生成式模型的多迭代特征，并把调度边界从完整请求降到模型迭代。今天的 vLLM 和 TensorRT-LLM 等系统沿用了这种基本思想，但在它周围增加了更完整的基础设施：

- paged KV Cache 按需分配和回收状态；
- packed input 与变长 attention 处理不规则序列；
- token budget 同时限制 prefill 与 decode 工作量；
- chunked prefill 控制长 prompt 对 decode 的阻塞；
- CUDA Graph 和 shape bucket 降低动态执行的 CPU 开销；
- prefix caching、speculative decoding 与多模态输入进入统一调度；
- priority、deadline 与集群路由进一步决定服务公平性。

例如，当前 TensorRT-LLM 文档把 in-flight batching、continuous batching 与 iteration-level batching 作为同类名称，并允许 context 与 generation 序列一起执行；当前 vLLM V1 则以最大序列数和最大 scheduled token 数约束每一轮工作，并让 scheduler 输出每个 request 本轮推进的 token 数。

这些实现细节会继续变化，但识别系统边界的方法相对稳定：

1. 调度器多久重新获得一次控制权；
2. 完成或取消的请求何时真正离开；
3. 新请求何时有资格进入后续执行；
4. KV 状态如何跨 batch 位置保持；
5. GPU runtime 如何承接动态 shape；
6. 过载时由谁限制 admission。

能回答这六个问题，才算真正理解一个引擎所说的 Continuous Batching。

## 小结

Continuous Batching 的起点，是固定批次不适合长度未知、需要多轮执行的自回归生成。

Orca 用 iteration-level scheduling 把一次不可变的长任务拆成连续的调度点：每轮执行后，完成请求可以离开，新请求可以加入，仍在运行的请求依靠 KV Cache 保留历史状态。现代引擎再用 paged KV、packed input、token budget 与 shape bucket，让动态重组在显存和 GPU 执行层面可行。

但可重组只提供机会，并不自动给出最佳策略。Prefill 与 decode 如何共享预算、长请求是否抢占短请求、怎样兑现 TTFT/TPOT，以及流量超过容量时如何 admission，仍需要更高层的调度设计。

因此，理解 Continuous Batching 最重要的不是记住“batch 会变化”，而是看清三个稳定关系：

```text
iteration boundary 决定何时重组
request state 决定什么可以重组
KV/runtime 决定重组能否高效执行
```

本系列接下来会在这个基础上继续讨论 chunked prefill；再向后，EngineCore、异步调度与 FairBatching 会分别展开进程边界、CPU/GPU 重叠和 SLO-aware 批次形成。

## 参考资料

- [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu)
- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- [TensorRT-LLM: Paged Attention, In-flight Batching, and Request Scheduling](https://nvidia.github.io/TensorRT-LLM/features/paged-attention-ifb-scheduler.html)
- [vLLM SchedulerConfig](https://docs.vllm.ai/en/latest/api/vllm/config/scheduler/)
- [vLLM V1 Scheduler](https://docs.vllm.ai/en/latest/api/vllm/v1/core/sched/scheduler/)
- [vLLM Optimization and Tuning: Chunked Prefill](https://docs.vllm.ai/en/latest/configuration/optimization/)
