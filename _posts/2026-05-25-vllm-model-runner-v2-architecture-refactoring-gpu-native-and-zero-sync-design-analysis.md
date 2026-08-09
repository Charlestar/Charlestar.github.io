---
layout: post
title: "vLLM Model Runner V2：GPU-native 与 async-first 的执行核心"
subtitle: "从稳定状态表到无 CPU 同步的 CUDA Stream"
date: 2026-05-25 12:00:00 +0800
last_modified_at: 2026-08-09
author: iStar
catalog: true
mathjax: true
tags: [AI Infra, vLLM, LLM推理, GPU优化]
---

在一次 vLLM engine step 中，Scheduler 只决定“每个请求推进多少 token、使用哪些 KV block”。这些逻辑状态还不能直接交给模型。GPU ModelRunner 必须把它们变成连续 tensor：`input_ids`、`positions`、block table、sequence length、query start offset、sampling 参数与 CUDA Graph 所需的 padded batch。

当模型很大、单步 GPU 计算很长时，几十或几百微秒的 CPU 准备不显眼。GPU 越快、模型越小、batch step 越短，Python 循环、小 tensor copy 和一次无意的 GPU→CPU 同步就越容易变成气泡：

```text
CPU: prepare N       prepare N+1       prepare N+2
GPU:           run N           run N+1           run N+2
               ^ 等 CPU         ^ 等 CPU
```

Model Runner V2（MRV2）是对这条执行热路径的重写。它不改变用户的 OpenAI API，也不是“vLLM V2”；它重新安排请求状态、每步输入、CPU/GPU 所有权与 sampler，使目标时间线变成：

```text
CPU: prepare N+1  prepare N+2  prepare N+3
GPU: run N        run N+1      run N+2
     <---------- overlap ---------->
```

官方用三个词概括设计：modular、GPU-native、async-first。真正理解它们，要从 V1 persistent batch 的一个结构性矛盾开始。

## ModelRunner 位于哪一层

一条 vLLM V1 执行链可以简化为：

```text
Scheduler
  │ SchedulerOutput
  ▼
Executor / GPU Worker
  │
  ▼
GPU ModelRunner
  ├─ update persistent request state
  ├─ prepare per-step input tensors
  ├─ select eager / CUDA Graph path
  ├─ model forward
  ├─ logits processing and sampling
  └─ ModelRunnerOutput
```

Scheduler 负责策略与 KV 资源，ModelRunner 负责把策略落实为 tensor 和 kernel。它必须同时理解：

- 当前有哪些新增、继续、恢复和完成请求；
- prompt/decode/speculative token 在本轮如何排列；
- PagedAttention block table 怎样映射；
- 多模态 embedding 替换哪些 token；
- 每个请求使用什么 temperature、top-k/top-p、penalty、logprobs；
- TP/PP 与 attention backend 需要哪些 metadata；
- 当前 shape 能否 replay 已捕获的 CUDA Graph。

所以 ModelRunner 很容易变成所有特性的交汇点。MRV2 的第一目标并不是某个单独 kernel 更快，而是让这些交汇关系有更稳定的数据模型。

## 为什么 V1 需要 persistent batch

Continuous batching 的相邻两步通常非常相似。假设 step $N$ 中有 128 个 decode 请求，step $N+1$ 只完成 3 个、加入 5 个。若每轮从 Python request object 重建整个 block table 和 sampling tensor，绝大多数工作是在复制未变化状态。

Persistent batch 维护长期 tensor，只写增量：

```text
step N:   [A B C D E F]
step N+1: [A B C   E F G]
                 remove D, add G
```

这比每步全量构造高效，但 V1 让同一批 tensor 同时承担两个职责：

1. 保存活跃请求的持久状态；
2. 直接作为本轮 model/sampler 的有序输入。

“状态表的行”与“本轮执行顺序”被绑定在一起。Attention backend 改变顺序、请求完成或被抢占时，为保持输入连续，runner 可能需要移动其他请求的行；移动时又不能覆盖仍在被异步 GPU 读取的数据，因此还维护 `CachedRequestState` 等备份。

复杂性不是 persistent batch 本身造成的，而是**持久身份与瞬时布局共用一份 tensor**。

## MRV2：稳定 state row 与每步 input 分离

MRV2 预分配固定行数的 request state table。官方设计文档给出的多数平台默认 `max_num_reqs=1024`。一个请求在活跃生命周期内获得稳定 row：

```text
state row 0: request C
state row 1: request A
state row 2: free
state row 3: request F
state row 4: request B
```

Scheduler 本轮可能要求执行顺序 `[A, B, C]`。Runner 不重排 state table，而是在 GPU 上按 row index gather：

```text
step order:        A  B  C
state row index:   1  4  0
                         │
GPU gather -------------┘
                         ▼
per-step input:    [A B C]
```

这形成了两个清晰对象：

- persistent state：按 request identity 稳定存储；
- `InputBatch`：按本轮执行布局临时构造。

请求完成后 row 回收到 free list。MRV2 把 preemption 视作完成：释放当前 row；请求恢复时像一个新请求重新写入状态。这样不需要长期保存一个可能已被复用的物理行身份。

稳定 row 带来的好处包括：

- 新增/删除请求变成局部 row 更新；
- attention backend 可以自由给出每步 order；
- 不再因全表重排维护冗余 backup state；
- GPU gather 天然适合从大表构造连续 batch；
- speculative branch 可用 indirection 映射回 request state。

代价是每步多一次 gather，但它在 GPU 上并行且相对模型前向很小，换来了更简单、可组合的状态所有权。

## GPU-native 不是把 Scheduler 搬进 CUDA

CPU 仍决定 admission、priority、token budget 和 KV block 分配。GPU-native 指的是：**已经确定的调度结果，尽量在设备上批量展开为模型所需 metadata。**

MRV2 使用 Triton kernel 生成：

- `input_ids`；
- `positions`；
- `query_start_loc`；
- `seq_lens`；
- block table 的 step-specific view；
- sampling/logits processing 所需索引。

V1 常见路径是 Python 遍历请求、写 CPU tensor，再做 H2D：

```text
request objects
  -> Python loop
  -> CPU tensor writes
  -> pin/copy to GPU
  -> model input
```

MRV2 更接近：

```text
small CPU diff + GPU-resident state
  -> async H2D of compact diff
  -> Triton gather/expand on GPU
  -> model input
```

这同时减少 Python 开销和传输字节。更重要的是，某些信息本来就只在 GPU 上及时可见，例如 speculative decoding 实际接受了几个 token。若 CPU 为了准备下一步必须读取 accepted count，就会产生同步；设备侧 input prep 可以直接消费它。

## Async-first 的真实约束

异步调度让 CPU 在 GPU 执行 step $N$ 时准备 $N+1$：

```text
time ─────────────────────────────────────>

CPU   schedule N+1 | prepare N+1 | schedule N+2
GPU   execute N    | execute N+1 | execute N+2
```

这要求 CPU 热路径不等待 GPU。同步既可能显式出现：

```python
torch.accelerator.synchronize()
```

也可能隐式出现：

```python
count = gpu_tensor.item()          # D2H wait
gpu = pageable_cpu_tensor.cuda()   # copy may block
if gpu_tensor.any():               # Python needs device result
    ...
```

MRV2 把核心执行模型设为“一条持续排队工作的 CUDA stream”：CPU entrypoint 只向 stream enqueue 操作，尽量不读取本轮 device result。

所谓 zero synchronization 是**受支持热路径的设计目标**，不是整个程序从启动到结束从不等待 GPU。输出 token 最终要返回 CPU，profiling、错误检查、模型重载和不支持特性也会建立同步边界。

## 异步 copy 为什么会产生竞态

Pinned CPU memory 可以用 `non_blocking=True` 异步传到 GPU，但源 buffer 必须在 DMA 完成前保持不变。

下面的共享 pinned state 是危险的：

```python
self.states[req_idx] = new_value
device_states = self.states.to("cuda", non_blocking=True)

# CPU 很快进入下一 step，再次改 self.states
# GPU/DMA 可能还在读上一版 self.states
```

CPU 与 copy engine 同时访问同一页，结果取决于时序。V1 可以在关键区设置 async barrier，等 GPU 不再使用这些 buffer 后才允许 CPU 修改。但 barrier 有三个问题：

- 每添加一种共享 buffer 都可能漏保护；
- 所有 CPU 操作被迫围绕统一屏障组织；
- 屏障过宽会减少重叠，过窄又产生 race。

MRV2 的方向是消除共享写入：持久 CPU state 留在普通内存；每步把要传输的快照/差异放进独立 pinned staging buffer。GPU 读 staging，CPU 可继续修改下一版 persistent state，两者不触碰同一存储。

```text
CPU persistent state ──snapshot──> pinned staging N ──H2D──> GPU
       │
       └──继续修改 N+1───────────> pinned staging N+1
```

异步正确性的关键不是“用了 pinned memory”，而是源 buffer 的生命周期与 stream event 有明确所有权。

## `StagedWriteTensor`：只传 block table 的差异

Block table 可能是 `[max_num_reqs, max_num_blocks]` 的大矩阵。即使只有几个请求加入，整张表每步 H2D 仍然浪费。

MRV2 的 `StagedWriteTensor` 让 base tensor 常驻 GPU，CPU 只暂存 ragged diff：

```text
GPU base state
  row 0: [...]
  row 1: [...]
  row 2: [...]

CPU staged writes
  (row=2, start=3, values=[3,1,2])
  (row=0, start=1, values=[-1,-2,-5])
```

提交时：

1. 将多个 diff 打包为连续的 row/start/length/value buffers；
2. 异步传输紧凑 buffer；
3. 发起一个 GPU kernel 把各段写入 base tensor；
4. 后续 gather/model input 在同一 stream 上自然等待写入完成。

数据量从“整张状态表”缩到“本轮实际变化”。它尤其适合 block table，以及既可能被 CPU 又可能由 GPU 结果推进的 `num_computed_tokens`。

实现上必须处理同一位置的冲突写顺序、staging buffer 复用时机、preemption 后 row generation，以及 diff 超出预分配容量时的 fallback。

## UVA 为什么适合大 prompt token

并非所有状态都值得常驻 GPU。长 prompt 的 token IDs 可能只在 prefill 的对应 chunk 用一次，全部复制会占显存。

Universal Virtual Addressing（UVA）允许 GPU kernel 通过统一地址访问 pinned host memory。MRV2 在部分路径中让 GPU 直接读取 CPU-resident `prefill_token_ids`，避免建立完整 device 副本。

这是一种容量与带宽的交换：

- 访问少量、一次性的 host data，UVA 可省 copy 和显存；
- 重复、高带宽或不连续访问，PCIe/互连延迟可能比预拷贝更差；
- host buffer 仍需 pin，并保持到 kernel 完成；
- 多 NUMA socket 下，分配位置会影响 GPU 访问路径。

因此 UVA 不是“CPU tensor 免费变 GPU tensor”，而是让 runner 能按访问模式选择 staging copy 或设备直接读取。

## Sampler 为什么也要重写

模型 forward 结束后，采样仍在每个 decode step 的关键路径。MRV2 主要用 Triton 重新实现 sampler，以更直接地控制随机数、精度、内存和 speculative mapping。

### Gumbel-max 避免显式概率采样

对 logits $z_i$，categorical sample 可以通过 Gumbel-max 得到：

$$
y=\arg\max_i(z_i+g_i),
\qquad g_i\sim Gumbel(0,1)
$$

而：

$$
g_i=-\log(-\log u_i),qquad u_i\sim Uniform(0,1)
$$

这样不需要显式 materialize softmax 概率再调用 multinomial。Triton kernel 从 seed/counter 生成 stateless random number，叠加 logits 后做 reduction。

数值精度仍很重要：低精度 Gumbel noise 可能改变概率，尤其在大词表与接近 logits 中。当前 release notes 仍持续记录 FP32 Gumbel accuracy 修复，说明“GPU-native”不能以分布偏差换速度。

### Top-k logprobs 不应物化整个词表

若用户只请求 top-5 logprobs，V1 风格路径可能先得到完整 vocabulary logprobs，再做 top-k。这会产生 `[num_tokens, vocab_size]` 大 tensor。

MRV2 先从 logits 找 top-k token，只为选中项计算/保留 logprob 所需结果，从而降低峰值显存。长 prompt 的 prompt logprobs 还支持更细粒度 chunking，避免一次生成巨大的全 prompt × vocab 中间量。

注意：严格 logprob 仍需 softmax normalization 的 log-sum-exp，优化的是不保存每个 token 的完整归一化结果，而不是凭 top-k logits 就省略分母。

### `idx_mapping` 连接 speculative logits 与 request state

普通 decode 中每个请求通常有一个 logits row。Speculative verification 可能让一个请求对应多个候选位置，sampling state 的行数与 logits 行数不再相同。

V1 若复制 temperature、penalty 等 request metadata 以匹配每个 logits row，会增加扩展和同步逻辑。MRV2 在 kernel 内用 `idx_mapping`：

```text
logits rows:   0 1 2 3 4 5
request state: A A A B B C
idx_mapping:   0 0 0 1 1 2
```

每个 logits row 间接读取正确 request state，无需在外部 materialize 重复参数。这就是稳定 state table 与 GPU kernel indirection 组合后的直接收益。

## Modularity 解决的是功能扩散

旧 runner 容易把 M-RoPE、multimodal、LoRA、penalty、spec decode 等分支堆在大型 `gpu_model_runner.py` 中。新增模型时，开发者可能为了一个特殊 position rule 改动公共热路径。

MRV2 将逻辑拆到专门模块，并用 `InputBatch` 聚合模型输入。模块边界应该回答：

- 谁拥有长期 request state；
- 谁将 scheduler delta 应用到 state；
- 谁按本轮 order gather input；
- 模型特有位置/embedding 怎样注入；
- sampler 接收哪些通用 tensor；
- CUDA Graph 捕获哪些静态资源。

Modular 不等于把每十行代码放一个文件，而是让功能通过明确数据结构组合，不直接读写 runner 的任意内部属性。

## 为什么不应借 `dummy_run` 偷渡逻辑

推理系统需要 warm-up、memory profiling 与 CUDA Graph capture。若模型特性把关键初始化副作用塞进 `dummy_run`，正常执行是否正确就依赖某个隐式启动顺序；不同 capture mode 或测试路径可能漏掉它。

MRV2 设计文档强调区分：

- 显式初始化；
- dummy/profiling forward；
- CUDA Graph capture；
- 真实 execute。

每个阶段应有明确输入和资源所有权。这样 feature compatibility 可以在启动时验证，而不是第一次真实请求时才因某个未 warm 的分支失败。

## CUDA Graph 管理为什么必须显式

CUDA Graph 通过重放固定 kernel DAG 减少 launch overhead，但要求输入地址、shape bucket 和控制流满足捕获约束。Persistent state 和 `InputBatch` 分离后，MRV2 可以更清楚地管理：

- graph capture size 与实际 batch padding；
- full graph、piecewise graph 或 eager fallback；
- attention backend 是否支持 capture；
- speculative draft/verify 的不同 graph pool；
- PP 多个 in-flight batch；
- graph replay 时更新输入但保持地址稳定。

“自动 capture 一次 dummy run”很难覆盖这些组合。显式 manager 将 graph 视为有版本、容量和兼容 predicate 的执行资源，而不是隐藏在 model forward 周围的装饰器。

## Zero-sync 需要怎样验证

只看 Nsight 时间线中 GPU 很满，不足以证明没有错误同步。可以按层建立证据。

### 静态搜索

检查热路径中的：

- `.item()`、`.tolist()`、`.cpu()`；
- device tensor 参与 Python `if`；
- 未 pinned 的 H2D；
- `synchronize()`；
- 默认 stream 与 side stream 之间缺失 event；
- pinned staging buffer 在 event 完成前复用。

### 同步检测

使用 vLLM 提供的 GPU sync check/调试环境能力，结合 `CUDA_LAUNCH_BLOCKING=1` 只用于定位，不能拿阻塞模式做性能结论。

### Timeline

在 Nsight Systems 中标出 EngineCore schedule、ModelRunner input prep、H2D、Triton prep、graph replay、sampling 与 D2H output。理想情况是 CPU N+1 与 GPU N 重叠，且没有无解释的长空隙。

### Race test

高频加入/完成/抢占请求，混合不同 prompt 长度与 speculative acceptance；在 sanitizer 和不同 stream 时序下重复运行。异步 race 往往在低并发单测中不出现。

## 怎样公平比较 MRV1 与 MRV2

MRV2 主要优化 bookkeeping、input prep、sampling 与异步重叠，因此收益对 workload 很敏感。

### 固定变量

- 相同 vLLM commit/build；
- 同一模型、dtype、量化、attention backend；
- 相同 TP/PP/DP；
- 相同 CUDA Graph mode/capture sizes；
- 相同 scheduler token budget；
- 相同 prompt/output trace 和 sampling seed；
- 确认两边没有功能 fallback。

### 分解耗时

- scheduler time；
- runner state update；
- CPU staging/H2D；
- GPU input-prep kernels；
- model forward；
- sampler；
- GPU→CPU output；
- 每步 bubble。

### 端到端指标

- TTFT、TPOT 的 P50/P95/P99；
- request throughput 与 output tok/s；
- GPU utilization 与 CPU core usage；
- peak GPU/host pinned memory；
- prefix cache、spec acceptance 与 preemption；
- correctness/logits/logprobs 对照。

小/中型 dense 模型、高并发和短 decode step 更容易暴露 CPU 瓶颈；超大模型若 GPU compute 占绝对主导，优化相同微秒数只会产生较小比例收益。不要把官方图中的最大提升当成所有部署的固定百分比。

## 从实验开关到默认路径的时间线

MRV2 的状态已经发生变化，文章必须把公告时间与当前版本分开：

- 2026-03-24 官方介绍发布时，MRV2 尚未 feature-complete，需要 `VLLM_USE_V2_MODEL_RUNNER=1` 试用；
- 后续版本逐步让 Qwen3、Llama、Mistral、量化模型和部分 MoE 使用 MRV2；
- vLLM v0.25.0 release notes 明确：MRV2 已成为**所有 dense 模型**的默认执行路径；
- MoE、hybrid、特殊 connector 与新功能仍可能根据 oracle/compatibility check 选择 MRV1 或专门路径。

截至本文校订日期，部署时不应机械设置旧试验开关，也不能仅因模型是 dense 就假定任意功能组合一定留在 MRV2。应查看启动日志、目标 release notes 与源码 oracle，并监控 fallback。

设计文档顶部仍保留“not yet feature-complete/not rigorously tested”的早期说明，这反映文档撰写阶段；当前支持状态应以锁定版本的 release notes 为准。文档解释“为什么这样设计”，release notes 解释“这个版本默认覆盖什么”。

## 迁移与回归矩阵

MRV2 声称对外 API 不变，但内部 runner 切换仍可能影响数值、显存和组合能力。至少覆盖：

| 维度 | 代表场景 |
| --- | --- |
| 阶段 | prefill、chunked prefill、decode、pooling |
| 采样 | greedy、temperature、top-k/p、min-p、penalty |
| 输出 | logprobs、prompt logprobs、large top-N |
| 推测 | draft model、EAGLE/MTP、greedy/随机验证 |
| 缓存 | prefix cache、抢占、混合 KV group、offload connector |
| 模型 | dense、MoE、Mamba/hybrid、多模态 |
| 参数 | BF16/FP16、FP8/INT、量化 KV、LoRA |
| 并行 | TP、PP、DP、EP、不同 worker backend |
| 执行 | eager、full/piecewise CUDA Graph、async scheduling |

任何不支持组合都应在启动时失败或明确 fallback；最危险的是静默选择另一 runner，随后把性能结果错误归因于 MRV2。

## 阅读源码时沿数据所有权走

比逐行读一个大 runner 更有效的顺序是：

1. 从 `SchedulerOutput` 找本轮新增、继续与完成请求；
2. 看 request 如何获得/释放 stable row；
3. 看 `StagedWriteTensor` 怎样更新 GPU state；
4. 看 per-step order 与 `InputBatch` 怎样 gather；
5. 看 input-prep Triton kernel 输出哪些 metadata；
6. 看 CUDA Graph manager 如何选择执行路径；
7. 看 sampler 的 seed、`idx_mapping` 与 output D2H；
8. 最后加入 speculative、多模态或 PP 分支。

对每个 tensor 标记四件事：owner、CPU/GPU 位置、有效生命周期、在哪条 stream 上最后写入。所谓 zero-sync 设计最终都能还原成这四个问题。

## 小结

Model Runner V2 的核心不是“用更多 Triton kernel”这么简单，而是重新定义状态与时间：

1. 请求在 stable state table 中拥有固定 row，本轮输入通过 GPU gather 形成；
2. 大状态常驻 GPU，CPU 用 `StagedWriteTensor` 只提交增量；
3. input metadata 与 sampler 尽量在 GPU 内完成，不读取尚未就绪的 device result；
4. pinned staging、UVA 和 stream ownership 消除 CPU/GPU 共享写 race；
5. 模型特有逻辑、sampling 和 CUDA Graph 管理通过明确模块组合。

这些选择让 async scheduling 从附加优化变成执行模型本身，并自然适配 speculative decoding 的多 token、设备侧结果与复杂映射。当前 MRV2 已是 v0.25.0 起 dense 模型的默认路径，但具体部署仍应以锁定版本、功能组合和实际 runner 选择为准。

## 参考资料

- [Model Runner V2 官方设计文档](https://docs.vllm.ai/en/stable/design/model_runner_v2/)
- [Model Runner V2: A Modular and Faster Core for vLLM](https://vllm.ai/blog/2026-03-24-mrv2)
- [vLLM v0.25.0 Release Notes](https://github.com/vllm-project/vllm/releases/tag/v0.25.0)
- [Persistent Batch Redesign RFC](https://github.com/vllm-project/vllm/issues/23446)
- [vLLM Model Runner V2 source](https://github.com/vllm-project/vllm/tree/main/vllm/v1/worker/gpu)
- [vLLM releases](https://github.com/vllm-project/vllm/releases)
