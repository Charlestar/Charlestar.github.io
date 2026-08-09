---
layout: post
title: "CUDA Graph：把动态 LLM Serving 装进可重放的静态执行图"
subtitle: "从 Kernel Launch 开销到 Batch Bucket、静态地址与 Piecewise Capture"
date: 2026-08-09 17:04:00 +0800
last_modified_at: 2026-08-09
author: iStar
catalog: true
series: gpu-runtime-precision
series_order: 10
technology_year: 2019
mathjax: true
tags: [AI Infra, CUDA Graph, LLM Serving, GPU Runtime, vLLM]
---

LLM Decode 每一步只为每条活跃 sequence 生成一个新 token，却要依次执行归一化、投影、Attention、MLP、collective、logits 与采样等许多 GPU kernels。单个 kernel 可能只运行几十微秒，CPU/Python/C++/driver 为每个 kernel 准备参数、建立依赖并发起 launch 的时间就不再可以忽略。

典型时间线会出现：

```text
CPU:  launch A ─ launch B ─ launch C ─ launch D ─ ...
GPU:       A ─ gap ─ B ─ gap ─ C ─ gap ─ D ─ ...
```

如果 GPU 很快做完 A，却要等 CPU 发出 B，设备就处于 launch-bound，而不是 compute-bound。模型越小、Decode batch 越小、kernel 越碎，这个问题越明显。

CUDA Graph 的思路是先记录一段 GPU 工作和依赖关系，实例化为可执行图，之后用一次 `cudaGraphLaunch` 重放整段工作：

```text
capture once:
  A → B → C → D → ... → Z

replay many times:
  cudaGraphLaunch(graph_exec)
```

它用静态性换取低 launch 开销。LLM Serving 的难点恰恰在于请求、batch、序列长度、KV blocks 和路由都在变化。真正的工程问题不是“怎样调用 CUDA Graph API”，而是**怎样把动态服务状态投影到有限组稳定的 GPU 地址和执行拓扑中**。

## Eager Execution 的成本在哪里

一次 PyTorch eager forward 的 kernel 需要穿过多层软件栈：

```text
Python / scheduler
  → framework operator dispatch
  → C++ / ATen / custom op
  → shape and argument preparation
  → CUDA runtime / driver launch
  → GPU kernel
```

设一轮有 $K$ 个 kernels，平均每次 host launch 与框架准备开销为 $t_l$，GPU 实际执行总时间为 $T_g$。如果 CPU 不能充分 ahead-of-time 提交，粗略 wall time 为：

$$
T_{eager}\approx T_g+K t_l-T_{overlap}
$$

CUDA Graph replay 把多次 launch 合并为一次图 launch：

$$
T_{graph}\approx T_g+t_{graph}+T_{copy/pad}
$$

其中 $T_{copy/pad}$ 是把本轮输入和 metadata 写入静态 buffers、处理 padding 的额外成本。

因此 Graph 收益依赖：

- $K t_l$ 是否显著；
- 图内 GPU 工作本身是否足够短；
- 静态 buffer copy/padding 是否很小；
- capture size 是否贴近实际 batch；
- 图内 kernel 是否和 eager 使用同样高效的实现。

如果一次 Prefill 已有很长的大 GEMM，GPU 计算占主导，省下的 host launch 百分比可能很小。不能因为 Graph replay 更“先进”就假定任何 workload 都会加速。

## Graph 记录的不是计算公式，而是一次具体执行

CUDA Graph 是带依赖的操作节点集合。节点可以包括 kernel、memcpy、memset、event、child graph 和其他受支持操作：

```text
input memcpy
     │
     ├──── kernel A ── kernel B ─┐
     │                            ├── kernel D
     └──── kernel C ─────────────┘
```

创建主要有两种方式：

1. 使用 Graph API 显式添加 nodes 与 edges；
2. 对已有 stream-based 代码做 stream capture。

LLM runtime 通常已有成熟 kernel 调用路径，stream capture 更容易接入：

```cpp
cudaStreamBeginCapture(stream, mode);

kernel_a<<<grid_a, block_a, 0, stream>>>(static_x, static_meta);
kernel_b<<<grid_b, block_b, 0, stream>>>(static_tmp);
library_call(stream, static_tmp, static_out);

cudaStreamEndCapture(stream, &graph);
cudaGraphInstantiate(&graph_exec, graph, ...);
```

重放时不再重新执行这些 CPU 函数，而是提交已实例化的 GPU 节点。Graph 因此不会重新运行 capture 区间里的普通 Python/C++ 控制逻辑。

## 三个核心静态约束

PyTorch 官方 CUDA semantics 将 Graph 的主要限制归纳为静态 shape、静态 control flow 和稳定 memory addresses。三者在 serving 中含义不同。

### Shape 静态

Capture 时 kernel grid、tensor shape 和内部 workspace 路径已确定。若 batch 从 32 变成 47，不能直接让为 32 捕获的图处理 47 rows。

### Control Flow 静态

Capture 记录当时实际执行的分支。如果下一轮 Python 判断改走另一 attention backend、跳过某层或改变 speculative path，replay 不会重新执行 Python `if`。

GPU 数据依赖的选择可以在 kernel 内通过 mask/indices 表达，但节点拓扑本身仍要兼容。

### 地址静态

Graph 中 kernel 参数包含 pointer。每次 replay 都读写同一组 virtual addresses。新 batch 不能简单创建一组新 tensors 再把 Python 引用传给 graph；必须把内容拷入 capture 时的静态 input/metadata buffers。

```text
request batch N dynamic tensors
          │ copy into
          ▼
long-lived static graph buffers
          │ replay uses same pointers
          ▼
long-lived static output buffers
          │ copy/view valid region
          ▼
request batch N result
```

地址稳定不表示内容不变。每轮可以更新同一地址内的 token ids、positions、block table、slot mapping 和 sampling metadata，然后 replay 相同 kernels。

## Capture 前为什么要 Warmup

第一次 forward 常触发许多一次性工作：

- CUDA context 与 library handle 初始化；
- kernel JIT compilation；
- cuBLAS/cuDNN heuristic 或 autotune；
- caching allocator 建立 blocks；
- NCCL communicator lazy init；
- custom op 初始化 workspace；
- 权重 layout 转换和量化 scale 准备。

这些 CPU side effects 不应意外发生在 capture 中。正确流程是：

```text
load model
  → eager warmup with representative shapes
  → synchronize and verify outputs
  → allocate long-lived graph buffers/pools
  → capture each chosen graph shape
  → replay canary and compare with eager
  → mark worker ready
```

vLLM 当前设计也把 eager warmup 与 full graph capture 分开，明确让 Attention backend 在 warmup 时运行，以便捕获时走到已经初始化的稳定路径。

Capture 是启动成本。捕获 shape 越多、模型越大，worker ready 越慢；滚动升级和故障恢复容量规划必须包含这段时间。

## Batch Bucket 把动态 Batch 映射到静态 Shape

在线 Decode 的 active sequence 数每步变化。Serving runtime 通常预先选一组 capture sizes：

```text
capture sizes = [1, 2, 4, 8, 16, 24, 32, 40, ...]
```

实际 batch size 为 $B$ 时，选择最小的 $C\ge B$：

$$
C(B)=\min\{c\in\mathcal C\mid c\ge B\}
$$

把 $B$ 个有效 rows 放入容量 $C$ 的静态 buffer，其余 $C-B$ rows 填充为无效 slots：

```text
actual batch = 19
selected graph = 24

rows  0..18: real sequences
rows 19..23: padding / masked slots
```

若 $B$ 大于最大 capture size，或本轮形状不兼容，则回退 eager/piecewise 路径。

### Padding 的浪费

理想情况下，额外工作比例为：

$$
W_{pad}=\frac{C-B}{C}
$$

但不同 kernels 对无效 rows 的处理不同：

- 有些仍做完整 GEMM，只是结果被 mask；
- 有些使用 valid count 跳过尾部；
- collective 可能仍按 $C$ 发送；
- Attention backend 可能用 metadata 跳过空 sequence。

所以 capture buckets 越密，padding 少但图更多；buckets 越稀，启动/显存更省但浪费计算。

## 怎样选择 Capture Sizes

固定使用等差或 2 的幂只是起点。更合理的方法是用线上 Decode batch histogram 选择有限 buckets。

设 batch size $B$ 的概率为 $p(B)$，使用 graph $C(B)$ 的执行成本为 $T(B,C)$，每个 graph 的捕获/显存成本为 $M(C)$。选择集合 $\mathcal C$ 可以写成：

$$
\min_{\mathcal C}
\sum_B p(B)T(B,C(B))
+\lambda\sum_{c\in\mathcal C}M(c)
$$

工程上不必真的求复杂优化器，可以按步骤：

1. 采集一周或多个流量周期的 batch size；
2. 找出高频 sizes 和 P90/P99 范围；
3. 从稀疏 buckets 开始；
4. 计算每个区间的平均 padding ratio；
5. 在 padding 最严重且流量高的区间增加 bucket；
6. 比较 capture 时间、reserved memory、TPOT 与 goodput；
7. 超大低频 batch 保持 eager fallback。

vLLM 不同版本的默认 capture size 列表和上界曾变化，当前文档也允许显式配置。部署应固定版本并根据 workload 调整，而不是把某一版默认数组当作 CUDA Graph 的普遍规律。

## 静态 Metadata 是 LLM Serving 的关键

模型权重本来就长期固定，真正动态的是 execution metadata：

```text
input token ids
positions
sequence lengths
query lengths
KV block table
slot mapping
request-to-row mapping
sampling parameters
speculative token validity
adapter ids
```

这些数据必须存入稳定地址。以 paged KV 为例，KV blocks 的物理内容可以位于动态分配的 cache pool；Graph kernel 不能每轮换一条 host pointer 链，而是从静态 `block_table` tensor 读取本轮逻辑 block 到物理 block 的映射。

```text
static pointer captured by graph
        │
        ▼
block_table_buffer[row, logical_block]
        │ values updated each replay
        ▼
physical KV block ids
```

这是一种通用技巧：把“动态地址选择”转换成“静态地址中的动态整数索引”。Graph 的 pointer 不变，kernel 仍能访问不同请求的 KV blocks。

必须完整覆盖 padding rows。若上一轮 graph size=32，本轮只有 19 条请求，而 rows 19..31 的旧 block ids 没有清空，kernel 可能访问已经释放或属于其他请求的 KV。Padding metadata 应写入安全 sentinel，并让所有相关 kernels 尊重 valid mask。

## Full Graph、Piecewise Graph 与 Eager

不是所有算子都 graph-safe，也不是所有 batch 都能使用同一完整拓扑。Serving runtime 通常需要三条路径。

### Full CUDA Graph

从模型 forward 入口到输出的大部分 GPU 工作都捕获在一个图里：

```text
static inputs
  → attention
  → MLP/MoE
  → collectives
  → logits/sampling
  → static outputs
```

优点是 host launch 最少，适合形状规则的 uniform Decode。限制是所有关键 backend 都要支持 capture，动态分支和 metadata contract 更严格。

### Piecewise CUDA Graph

把 graph-safe 的编译片段分别捕获，不兼容的 Attention、collective 或动态 op 仍 eager：

```text
graph piece A
  → eager attention
  → graph piece B
  → eager dynamic op
  → graph piece C
```

它保留更多动态性，但片段间仍有 host dispatch，收益低于理想 full graph。好处是某个不兼容 op 不必让整条 forward 放弃 Graph。

### Eager Fallback

用于：

- 超过最大 capture size；
- 未捕获的 shape/feature 组合；
- backend 不支持；
- 调试正确性；
- Graph replay 发生可隔离错误后的降级。

生产系统不能把 fallback 当作异常角落。新模型、Adapter、multimodal shape、MoE 路由和 speculative mode 都可能触发它，必须监控 fallback rate 和原因。

## vLLM 的 Graph Modes 在解决什么

vLLM 当前 CUDA Graph 设计把 Graph mode 与 batch descriptor/runtime dispatcher 分开。文档列出几类模式：

- `NONE`：完全 eager；
- `PIECEWISE`：只捕获可编译片段；
- `FULL`：使用完整图；
- `FULL_DECODE_ONLY`：只为 uniform Decode 使用完整图；
- `FULL_AND_PIECEWISE`：uniform Decode 走 full，其他兼容批次走 piecewise。

这些名称和默认值是版本相关的 vLLM 配置，不是 CUDA API 本身。其背后的通用决策树是：

```text
batch arrives
  │
  ├─ uniform decode + full-compatible backend?
  │      └─ yes → full graph bucket
  │
  ├─ has graph-safe compiled pieces?
  │      └─ yes → piecewise graph
  │
  └─ otherwise → eager
```

P/D 解耦的 Decode pool 可以选择 decode-only capture，避免为几乎不会出现的 Prefill/mixed shapes 支付捕获时间和显存；混合引擎则需要更灵活的 piecewise/fallback。

## Uniform Decode 为什么最适合 Full Graph

Uniform Decode 常满足：

```text
query length per sequence = 1
same model layers and kernel topology
dynamic values live in metadata buffers
batch varies but can be bucketed
```

Prefill/mixed batch 则有不同 query lengths、chunk sizes、Attention kernel paths 和 workspace。即使总 token 数相同：

```text
32 decode sequences × 1 token
```

与：

```text
1 prefill chunk × 32 tokens
```

并不具有相同 Attention 语义或 kernel topology。Graph dispatch key 不能只有 `num_tokens=32`，还要包含 uniform query length、batch composition、attention/backend mode 等真正决定执行图的字段。

## MoE 为什么让 Full Graph 更难

MoE router 每轮产生不同的 `tokens_per_expert`。Dropless dispatch 的 receive shape、grouped GEMM $m$ 分布和通信 counts 都可能变化：

```text
step 1: [8, 0, 3, 12, ...]
step 2: [1, 5, 5,  2, ...]
```

若 kernel 支持预分配最大 buffer、动态 valid counts，并在图内完成 permutation/dispatch，就有机会 capture；若 runtime 按本轮 counts 在 CPU 构造不同 tensor shapes，则不兼容 full static graph。

常见折中是：

- Attention 或 dense 子图做 full/piecewise capture；
- MoE dynamic dispatcher 保持 eager；
- 对固定 capacity + padding 的 MoE capture，但支付额外计算/显存；
- 只在 routing shape 可归一化的 Decode buckets 捕获；
- Graph 内传递 counts，kernel 使用最大 shape 与 valid mask。

Megatron Core 当前文档也指出，dropless MoE 的动态 shape 可能使 MoE 层无法 capture，此时仍可只 capture Attention 部分。关键是让 Graph boundary 与真正的动态边界对齐，而不是强迫整个模型静态化。

## Collective 能不能被 Capture

多 GPU 模型中，TP AllReduce、EP All-to-All 等 collective 也在 forward 路径上。是否 graph-safe 取决于通信库版本、process group 初始化、buffer 地址和调用模式。

接入原则：

1. Capture 前完成 communicator 初始化与 eager collective warmup；
2. 每个 rank 使用一致的 graph shape 与 collective 序列；
3. input/output buffer 地址在 replay 间稳定；
4. 不允许某 rank 走 Graph、另一 rank 同轮走不同 collective 拓扑；
5. Batch padding 后 collective count 必须在所有 ranks 一致解释；
6. 进程组重建后旧 graph 全部失效，需要重新 capture。

Graph replay 不能修复 distributed ordering bug。若 ranks 对同一图的执行次数不一致，仍可能 deadlock。

## 多 Stream Capture 需要闭合依赖

CUDA 支持把跨 streams 的 event dependency 捕获进同一 graph，但所有支线必须从 origin stream 正确 fork，并在结束 capture 前 join 回来：

```text
origin stream: A ─ record e1 ───────── wait e2 ─ D
                           │            ▲
side stream:               wait e1 ─ B ─ record e2
```

如果 side stream 工作没有通过 event 回到 origin，capture 结束时依赖图不完整。Capture 期间也不能让同一进程其他线程随意提交不相关 CUDA work，PyTorch 官方约束对此非常严格。

Serving runtime 中常有 H2D metadata copy stream、compute stream、communication stream。要逐条审计 event，而不是假设多 stream 会自动被捕获。

## 内存为什么会上升

Graph 要维持捕获时使用的稳定地址。PyTorch caching allocator 为 Graph 建立专用/private memory pool，避免 eager 分配复用这些地址。多个 graphs 若不共享合适的 pool，可能各自保留中间 buffers。

总额外显存可以粗略写为：

$$
M_{graphs}
\approx \sum_{c\in\mathcal C}
\left(M_{staticIO,c}+M_{workspace,c}+M_{privatePool,c}ight)
-M_{safeReuse}
$$

捕获更多 sizes、full 与 piecewise 同时启用、模型路径更多，显存和启动时间都会增加。它会直接挤压 KV Cache：

$$
N_{KV\ tokens}
\approx\frac{M_{free}-M_{graphs}-M_{other}}{bytes/token}
$$

因此 CUDA Graph 带来的单步延迟下降，可能以可并发 sequence 数减少为代价。最终要比较 goodput，而不是只测单 batch latency。

CUDA 官方的新 graph memory nodes 能表达图内 stream-ordered allocation/free，并允许物理内存复用，但这不等于任意 PyTorch/runtime 分配都可在 capture 中安全发生。具体 allocator 与框架支持仍要按版本验证。

## RNG 与采样状态

随机采样涉及 CUDA RNG。Graph replay 固定 kernel 和 pointer，但不应每次输出相同随机数。框架需要使用 graph-safe generator 状态管理，让 RNG offset 随 replay 正确推进。

PyTorch 允许 CUDA RNG operations 出现在 Graph 中，但对多 `torch.Generator` 的状态获取/设置有专用 graph-safe API。接入要验证：

- 同一 seed 下 eager 与 graph 的统计/确定性约定；
- 多 requests 的 RNG stream 不串位；
- batch padding 不意外消费有效请求的随机数；
- request migration/retry 能否恢复 sampler state；
- speculative decode 接受/拒绝分支怎样推进 RNG。

Graph 只负责重放 GPU 操作，不自动提供请求级随机语义。

## 一个最小 PyTorch 模式

下面的代码只展示稳定地址与 copy/replay 关系：

```python
static_x = torch.empty((32, hidden), device="cuda")
static_out = torch.empty_like(static_x)

# Warm up on a side stream before capture.
side = torch.cuda.Stream()
side.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(side):
    for _ in range(3):
        static_out.copy_(model(static_x))
torch.cuda.current_stream().wait_stream(side)

graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    static_out.copy_(model(static_x))

def replay(x):
    # x's address may change; static_x's address does not.
    static_x.copy_(x)
    graph.replay()
    return static_out
```

真实 serving 还要管理 metadata、bucket padding、多 streams、collectives、output lifetime 和并发，不能直接把返回的静态 output buffer 暴露给多个异步请求。下一次 replay 覆盖它之前，consumer 必须完成读取或复制。

## 最危险的正确性问题：旧数据仍在静态地址里

Graph replay 很容易产生“没有 crash，但用了上一轮 metadata”的错误。常见来源：

- 本轮没有覆盖完整 block table；
- padding rows 仍保留旧 request ids；
- 异步 H2D copy 尚未完成就 replay；
- output 在 consumer 读取前被下一轮覆盖；
- graph input tensor 被释放，地址由其他 allocation 使用；
- Adapter/weight pointer 更新后仍 replay 旧图；
- Graph bucket 选择漏掉影响 kernel topology 的字段。

建立输入 buffer generation 有助于 debug：

```text
graph_key
batch_generation
model_revision
adapter_set_hash
metadata_checksum
valid_rows
```

Debug 构建可在 graph 前后运行轻量校验 kernel，确认 valid rows、block ids 和 generation；生产则按采样率对比 eager canary。

## Graph Key 应包含什么

一个 graph cache 不能只按 batch size 索引。可能影响节点拓扑和 tensor shape 的字段包括：

```text
model / runner mode
batch bucket
uniform query length
prefill / decode / mixed
speculative token count
attention backend and mode
tensor/expert parallel layout
adapter execution mode
multimodal encoder budget
dtype / quantization path
full / piecewise capture mode
```

Graph key 过粗会错误复用，过细会捕获数量爆炸。可以将无法有限分桶的特征留在 piecewise/eager 路径，而不是组合出几千张低命中图。

## 性能验证怎样做

至少比较三个层次。

### Microbenchmark

对相同静态 shape 比较：

```text
eager forward latency
graph replay latency
input metadata copy latency
padding compute
```

这能确认 launch overhead 是否存在，但不能代表 continuous batching。

### Trace Replay

用真实 batch/time sequence 回放，记录：

- 每个 bucket 命中次数；
- padding ratio；
- full/piecewise/eager fallback；
- capture cache memory；
- Graph dispatch/key lookup 开销；
- P50/P99 iteration time。

### Serving Benchmark

在相同 request trace 和 SLO 下比较：

```text
TTFT / TPOT / E2E
goodput
CPU utilization
GPU idle gaps
KV capacity / max concurrency
worker startup and recovery time
```

Graph 最直接的证据是在 GPU timeline 中 kernel 间 host-induced gaps 减少，同时 CPU launch 线程负载下降。若 GPU 已持续饱和且没有 gaps，收益有限是合理结果。

## Capture 生命周期与部署

一张 graph 绑定多项运行状态。以下变化通常要求失效/重新 capture：

- model weights 或其地址变化；
- quantization/kernel backend 切换；
- process group/communicator 重建；
- parallel layout 改变；
- static buffer pool 重建；
- Graph key contract 或 runtime 版本升级；
- Adapter 执行方式改变且 pointer 进入图参数。

滚动部署顺序可以是：

```text
start worker
  → load exact model revision
  → initialize comm and kernels
  → eager warmup
  → capture selected graph set
  → replay/eager correctness canary
  → report graph memory and coverage
  → readiness=true
```

Graph capture 失败时，可以在策略允许的情况下以 eager/piecewise 模式启动，但 readiness metadata 应暴露降级状态，容量系统也要使用降级后的吞吐，而不是按 Graph 性能分配流量。

## 排查顺序

### Graph 没有变快

1. Timeline 是否真的存在 CPU launch gaps；
2. Graph 是否频繁 fallback；
3. Bucket padding 是否过大；
4. Metadata copy 是否同步 CPU；
5. Full 还是只有很小 piecewise 片段；
6. Capture kernel path 是否与 eager 相同；
7. GPU 是否本来 compute/memory-bound。

### Replay 报 Illegal Memory Access

1. Static input/output 是否仍有长期引用；
2. 所有动态 metadata 是否写在有效范围；
3. Padding block ids/slot ids 是否安全；
4. 前一次 replay/consumer 是否完成；
5. Pointer-owning object 是否被 allocator 回收；
6. Graph key 是否误复用；
7. Communicator/weights 是否在 capture 后重建。

### 输出偶发错误

1. 同 shape eager 对照；
2. 清零全部 static buffers 后重试；
3. 禁用异步 copy 验证 event ordering；
4. 检查 padding 与 RNG；
5. 按模块缩小 capture boundary；
6. 对每个 bucket 分别做多轮 replay；
7. 检查多 stream fork/join。

## CUDA Graph 与 torch.compile 的关系

二者解决不同层次的问题：

- `torch.compile` 分析/改写 operator graph，做 fusion、code generation、消除框架开销；
- CUDA Graph 记录最终 GPU launches 和依赖，以一次 replay 降低 driver/CPU launch 开销。

编译后的子图仍可能包含许多 kernels，适合再被 CUDA Graph capture；CUDA Graph 也可以捕获未由 `torch.compile` 生成的手写/custom kernels。vLLM 当前设计同样把 full Graph 支持与 compilation 视为可正交组合的能力。

编译处理动态 shape 时可以依赖 guards、recompile 或 graph break；CUDA Graph replay 则必须落到具体稳定的 shape/address bucket。下一篇会从这条边界继续解释 `torch.compile` 怎样把 LLM forward 切成可编译片段，又为什么 attention、custom op、collective 和 Python scheduler 会形成 graph breaks。

## 小结

CUDA Graph 通过记录并重放一段 GPU 工作，把大量逐 kernel host launch 压缩成一次图提交。它最适合 kernel 碎、CPU launch-bound、形状可分桶的 LLM Decode。

在 serving 中要抓住八条原则：

1. Graph 固定的是 kernel 拓扑、参数地址和 shape，不是 buffer 内容；
2. 动态请求通过静态 metadata buffers、indices 和 masks 注入；
3. Batch bucket 用 padding 换图复用，capture size 集合要按流量分布选择；
4. Uniform Decode 适合 full graph，Prefill/mixed/MoE 动态路径常需 piecewise 或 eager；
5. Capture 前必须完成 warmup、lazy init、autotune 和 communicator 建立；
6. 多 streams、collectives、RNG 与 output lifetime 都需要显式正确性契约；
7. 更多 graphs 会增加启动时间和显存，可能挤压 KV Cache；
8. 最终用 full/piecewise coverage、TPOT goodput 与恢复时间评估，而不是只看单次 replay。

动态 serving 并没有被 CUDA Graph 消灭；它被重新组织到了图外的 scheduler、bucket dispatcher 和图内的静态 metadata 中。能否明确这条边界，决定了 Graph 是可靠的生产优化，还是一组难以复现的偶发加速与隐蔽错误。

## 参考资料

- [NVIDIA CUDA Programming Guide: CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)
- [PyTorch CUDA Semantics: CUDA Graphs](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-graphs)
- [PyTorch `torch.cuda.graph` API](https://docs.pytorch.org/docs/stable/generated/torch.cuda.graph.html)
- [NVIDIA CUDA Graph Best Practice for PyTorch](https://docs.nvidia.com/dl-cuda-graph/latest/)
- [vLLM: CUDA Graphs Design](https://docs.vllm.ai/en/stable/design/cuda_graphs/)
- [Megatron Core: Mixture of Experts](https://docs.nvidia.com/megatron-core/developer-guide/nightly/user-guide/features/moe.html)
