---
layout: post
title: "torch.compile：LLM Serving 从 Python Forward 到融合 GPU Kernel"
subtitle: "沿着 Dynamo、FX、Guards、Inductor 与 Graph Break 理解编译收益和冷启动"
date: 2026-07-31 09:00:00 +0800
last_modified_at: 2026-08-09
author: iStar
catalog: true
series: gpu-runtime-precision
series_order: 20
technology_year: 2022
mathjax: true
tags: [AI Infra, torch.compile, TorchDynamo, TorchInductor, LLM Serving]
---

PyTorch eager mode 每执行一行 tensor 运算，就经过 Python、operator dispatcher、C++ 和 CUDA kernel launch。它让模型代码容易调试和扩展，却把原本可以合并的算子隔成许多独立步骤：

```python
x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
x = x * weight
```

这段 RMSNorm 风格计算可能产生 square、reduce、add、rsqrt、multiply 等多个 kernels，反复读写中间 tensor。GPU 真正需要的不是“按 Python 行执行”，而是把可融合的运算组合成更少的 kernel，在寄存器/共享内存中保留中间值。

`torch.compile` 的目标就是在保留大部分 Python/PyTorch 编程体验的同时，捕获一段 operator graph，再交给编译 backend 生成优化代码。默认链路可以概括为：

```text
Python model/function
  → TorchDynamo traces Python execution
  → FX graph
  → functionalization / AOTAutograd when needed
  → TorchInductor optimization and scheduling
  → Triton/CUDA kernels on GPU
  → compiled artifact cache
```

“加一行 `torch.compile`”只是入口。它是否加速 LLM Serving，取决于 Dynamo 能捕获多大区域、guards 是否稳定、dynamic shapes 是否合理、custom ops 是否暴露优化机会，以及编译成本能否在 worker 生命周期内摊薄。

## 它和 CUDA Graph 优化的不是同一层

上一篇讨论的 CUDA Graph 记录一次具体 GPU launch 序列，用一次 replay 减少 host/driver launch 开销；`torch.compile` 在更上层重写 operator graph，减少 kernel 数、融合内存访问并生成新 kernel。

| 能力 | `torch.compile` | CUDA Graph |
| --- | --- | --- |
| 输入 | Python/PyTorch operator program | 已经形成的 GPU operations |
| 主要收益 | fusion、codegen、减少框架/operator 开销 | 一次 replay 替代多次 kernel launch |
| 动态 shape | guards、symbolic shape、recompile | 最终仍需具体静态 graph/bucket |
| 动态 Python | 可 trace 的路径与 graph breaks | capture 外执行，replay 不重新运行 |
| 生成新 kernel | 可以，Inductor/Triton | 不会，只记录现有 kernels |
| 冷启动 | trace、compile、autotune | warmup、capture、instantiate |

二者可以组合：先由 Inductor 把十几个 elementwise/reduction ops 融合成少量 kernels，再由 CUDA Graph 把整段 compiled kernels 与 custom ops 一起捕获。vLLM 当前编译设计也明确把 TorchInductor 与 CUDA Graph 作为可分别开关的层。

## TorchDynamo 捕获的是一次实际 Python 路径

TorchDynamo 挂接 CPython frame evaluation，符号执行 bytecode，在程序运行时记录 PyTorch operations，生成 FX graph。它不是先解析完整源代码，再证明所有可能分支；它沿给定输入走过的路径进行 trace。

```python
@torch.compile
def f(x, use_residual):
    y = torch.sin(x)
    if use_residual:
        y = y + x
    return y
```

首次传入 `use_residual=True` 时，Dynamo 捕获的是执行过的 `sin + add` 路径。为了保证下次复用仍正确，它会为 `use_residual`、tensor dtype/device/shape 等假设生成 guards。

因此可以把一个 compiled entry 理解为：

```text
if guards(input, globals, module_state, ...):
    run compiled_graph
else:
    trace/compile another valid graph or fall back
```

Dynamo 的价值不只是把 Python 转成 FX，还包括在灵活 Python 语义与可优化静态区域之间建立这套守卫契约。

## FX Graph 是编译器看到的中间程序

FX graph 由 nodes 组成，每个 node 表示 function/module/method call 或 graph input/output：

```text
placeholder x
  → aten.pow
  → aten.mean
  → aten.add
  → aten.rsqrt
  → aten.mul
  → output
```

进入 graph 后，编译器可以跨越原 Python 语句边界做：

- elementwise/reduction fusion；
- 常量传播与 dead code elimination；
- layout、buffer reuse 和 memory planning；
- 根据 shape 选择 tiling；
- 生成 Triton 或其他设备代码；
- 插入/替换特定 pattern 的优化 kernel；
- 对某些 sizes 做专门化。

Graph 越完整，跨算子优化空间越大。但“完整”不是唯一目标。一个超大 graph 若因一个 batch shape 变化频繁重编译，冷启动和抖动可能比多个稳定子图更差。

## AOTAutograd 在推理文章里为什么仍会出现

`torch.compile` 同时支持训练和推理。训练时 AOTAutograd 从 captured forward 生成 backward graph，并可在 forward/backward 之间做 partition，以决定保存或重算哪些 tensors。

纯 inference mode 没有 backward，重点主要是 Dynamo + Inductor；但完整栈、日志和某些 functionalization/decomposition 仍可能显示 AOT 相关层。理解边界有助于避免两种误解：

- `torch.compile` 不只是一套 inference fusion；
- 推理性能问题也不必从 backward partition 开始排查。

如果模型意外在 grad enabled 状态运行，capture 路径和保存的中间状态可能改变。Serving 应明确使用 inference/no-grad 语义，并将其纳入基准条件。

## TorchInductor 怎样产生收益

Inductor 接收 graph 后，建立 loop-level IR、分析依赖与布局、做 fusion/scheduling，再为 GPU 生成 Triton/CUDA 相关代码。最常见收益来自减少 HBM 往返。

假设 eager 中间 tensors 为 $z_1,z_2,z_3$，分别写回/读出 HBM：

$$
B_{eager}\approx
\sum_i (B_{read,i}+B_{write,i})
$$

Fusion 后中间值留在寄存器，外部流量更接近：

$$
B_{fused}\approx B_{input}+B_{output}
$$

对 LayerNorm/RMSNorm、activation、residual、quantization scale 等 memory-bound 小算子，这种差异可能比理论 FLOPS 更重要。

Inductor 也会减少 kernel launch 数，使后续 CUDA Graph 更小。但大型 GEMM/Attention 常已由高度优化 library/custom kernel 完成，强行改写未必更快。编译的价值通常来自“让可融合的胶水算子更紧凑，并保留专业大 kernel”。

## Guard 是正确性条件，不是多余检查

Compiled graph 可能针对以下条件专门化：

```text
tensor dtype/device/rank
shape or symbolic shape range
stride/layout/contiguity
Python scalar and boolean values
module attributes
global state
object identity / aliasing
training/inference mode
```

当输入不满足 guard，旧代码不能安全复用，系统需要寻找另一条 cache entry 或重新编译。这就是 guard failure。

例如 kernel 为 batch=16 选择了固定 tile 和 buffer layout，batch=24 到来时若没有 dynamic dimension，就必须生成另一版本。去掉 guard 并不会让旧 kernel magically 正确，只会破坏 soundness。

线上要监控的是“guards 是否过度专门化”，而不是追求零 guards。PyTorch 提供 `TORCH_LOGS=guards,recompiles` 等日志查看失败原因。

## Recompile 为什么会让 Serving 抖动

JIT 路径通常是：

```text
first unseen input pattern
  → Dynamo trace
  → Inductor code generation
  → Triton/CUDA compile/autotune
  → cache artifact
  → execute
```

编译可能花费秒到分钟，远大于一次请求的 latency budget。若在线请求不断触发新 shape、Adapter 或 Python scalar specialization，会出现 compilation storm：CPU 满载、磁盘 cache 膨胀、请求长尾和 worker readiness 延迟。

PyTorch 对每个 code object 的 compiled entries/recompile 有上限，达到限制后可能回退 eager；具体默认值和行为随版本演进，应以部署版本为准。生产观测至少包括：

- compiled graph/variant 数；
- compile time 与 cache hit；
- guard failure/recompile rate；
- eager fallback/skip code；
- 首请求与稳态 latency；
- worker 启动阶段每个 compile phase。

## Dynamic Shapes 不是“一个 Kernel 支持任意输入”

`torch.compile` 可以用 `SymInt` 表达 batch/token 等动态维度，并为取值范围建立 guards。一个 dynamic kernel 可能接受多种 $N$：

```text
x: [N, hidden]
guard: 1 <= N <= max_tokens
```

但动态性有代价：

- 代码生成不能使用所有静态常量优化；
- 某些 op 仍要求固定 shape；
- data-dependent output shape 更难捕获；
- 约束过宽可能产生较差的通用 kernel；
- 某些分支仍根据 shape 产生不同 graph；
- 最终 CUDA Graph 仍需具体 bucket。

PyTorch 默认 `dynamic=None` 会先按静态假设编译，检测到 shape 变化后尝试更动态的 graph；`dynamic=True/False` 可改变策略。文章不应把某个 mode 当作固定最佳值。

LLM Serving 常使用“通用动态 graph + 少数静态热点 sizes”的混合策略：

```text
general dynamic compiled graph
  ├── handles long-tail token counts
  └── avoids many recompiles

specialized small shapes
  ├── batch/token sizes with high traffic
  └── better kernel choice + CUDA Graph capture
```

是否值得 specialization 要由 batch histogram 与真实 kernel profile 决定。

## Shape Guards 要描述真实约束

有些关系对所有合法请求成立，却未被编译器自动推导。例如：

```text
num_tokens <= max_num_batched_tokens
num_tokens divisible by TP size for sequence-parallel path
block_table_width <= configured maximum
top_k fixed by model
hidden size static
```

可以通过 API/运行时 assert 给 symbolic shape 提供范围与关系，使 compiler 不必为不可能的值保留路径。错误或过宽的范围会降低优化，过窄则触发 recompile/guard failure。

约束必须来自 scheduler/model contract，而不是为了消除日志随意声明。若线上确实可能出现超界，应该有明确 fallback，而非依赖 undefined behavior。

## Graph Break 发生了什么

Dynamo 遇到无法安全 trace 的操作时，可以结束当前 FX graph，回到 Python/eager 执行该段，再从后续可捕获位置开始新 graph：

```text
Python frame
  → compiled graph A
  → eager unsupported operation
  → compiled graph B
```

这就是 graph break。它保留兼容性，但带来：

- 更多 graph launch/框架往返；
- fusion 不能跨越 break；
- 中间 tensor 需要 materialize；
- 每段有自己的 guards/cache；
- CUDA Graph piecewise boundary 变复杂。

`fullgraph=True` 要求整个函数形成单图，遇到 break 直接报错，适合诊断和建立强契约；默认 `fullgraph=False` 更宽容，适合逐步接入。不能为了“没有报错”就忽略几十个 graph breaks。

## LLM Forward 常见的 Graph Break

### Tensor 值回到 Python

```python
n = tensor.sum().item()
if n > 0:
    ...
```

`.item()` 引入 GPU→CPU 同步，后续 Python control flow 依赖运行时数据。可考虑把逻辑改成 tensor operation、`torch.cond` 等受支持高阶算子，或把它明确留在 graph 外。

### Python Logging/Printing

模型 forward 内 `print`、复杂 logging 和 side effects 无法作为纯 tensor graph 优化。调试代码应放到 graph 边界外或受控禁用。

### 动态 Python 容器

根据 token 数据 append 不同数量 tensors、创建形状不定的 list/dict，容易导致 specialization 或 break。Serving metadata 最好预先整理成结构稳定的 tensors。

### Unsupported Custom Op

一个只有 Python 实现、缺少正确 dispatcher schema/meta/fake implementation 的 custom op，Dynamo/Inductor 无法理解 shape 与 side effects。

### Collective 与外部 Runtime

Distributed collective、KV cache manager、scheduler callback、CPU RPC 不应假定可被普通 operator graph完整吸收。可以用 registered custom op 建立显式边界。

### Data-dependent Shape

`nonzero`、动态 Top-k 结果长度、MoE 每 expert token 数等可能让输出 shape 依赖数据。需要 padded/static capacity、symbolic support 或 piecewise boundary。

## Custom Op 是黑盒边界，也是稳定契约

高性能 serving 已有 FlashAttention、PagedAttention、grouped GEMM、KV copy、sampling 等手写 kernels。让 Inductor 展开重写它们不一定合理。Registered custom op 可以告诉 compiler：

```text
this op has known schema and shape behavior
implementation is provided externally
do not decompose its internals unless a decomposition is registered
```

优点：

- 保留针对硬件/形状深度优化的 kernel；
- 提供明确 graph boundary；
- fake/meta implementation 支持 shape tracing；
- 可标记 mutation、alias 与 CUDA Graph safety；
- runtime 可按 backend dispatch。

代价：

- compiler 看不到内部 ops，无法跨边界 fusion；
- 每个 custom op 仍有 dispatch/launch；
- schema 或 fake implementation 错误会产生静默 shape/alias bug；
- 过多黑盒把 graph 切碎。

选择原则不是“custom op 越少越好”，而是把已高度优化、语义复杂的核心 kernel 保留为边界，把周围 elementwise/reshape/scale 等胶水暴露给 Inductor。

## Decomposition 让编译器看到更多

某些高层 PyTorch op 可以分解为更基础的 ATen ops。分解后 Inductor 能做 fusion/codegen；不分解则可能调用现有 library kernel。

```text
high-level op
  ├── keep as external kernel
  └── decompose → primitive ops → fuse/codegen
```

哪条更快取决于 shape 和硬件。大 GEMM 保留 library 实现通常合理；小复合 elementwise op 分解后更易融合。vLLM 当前编译路径还会通过自定义 passes 识别 serving-specific patterns，在不把优化硬编码进每个模型实现的情况下做融合。

所有 pattern pass 都需要版本化正确性测试：匹配条件过宽会对不等价模型结构应用错误重写。

## Piecewise Compilation 为什么适合 Serving

LLM forward 中 Attention/custom collective 可能要求特殊 runtime metadata，其他 dense blocks 又很适合 Inductor。可以在指定 splitting ops 处分割：

```text
compiled piece 0: embedding + norm + qkv projections
custom/eager:      paged attention
compiled piece 1: output projection + residual + norm
custom/eager:      MoE dispatch/grouped GEMM
compiled piece 2: residual + next-layer glue
```

每个 compiled piece 可以处理动态 token 数，并选择性再做 CUDA Graph capture。Full CUDA Graph 则可以在外层捕获整个已编译 forward，只要所有 runtime operations 都 capture-safe。

vLLM 当前实现已经超出“直接调用 stock `torch.compile`”：它用内部 compile APIs 做一次动态 full graph capture，再可选 split/specialize、运行自定义 Inductor passes、保存 compile cache，最后叠加 CUDA Graph。官方调试文档明确提醒 vLLM-compile 是自定义编译系统，不能把它的 mode/行为等同于 PyTorch 原生装饰器。

## Attention 为什么常是分割点

Serving Attention 与训练 dense attention 不同：

- Paged KV block table 来自 runtime；
- Prefill/Decode/mixed 选择不同 kernel；
- query length、context length 与 causal mask 语义变化；
- prefix cache/cascade attention 可能改变路径；
- TP/CP collective 与 backend capability 不同；
- workspace 与 launch 由 metadata 决定。

把这些全部表示成一个通用 Inductor graph 很难，也可能不如 FlashAttention/FlashInfer 等专用 kernel。因此一个实用策略是把 Attention 注册为 shape/alias 明确的 custom op，让 surrounding projections/norm/residual 编译融合。

当 backend 后来变得 compile/CUDA-Graph-safe，可以移动 boundary，而不是重写全部模型。

## MoE 给编译器带来的动态性

Router 产生 `tokens_per_expert`，每个 expert GEMM 的 $m$ 动态：

```text
[m0, m1, ..., m(E-1)]
```

编译器可以捕获 router projection、Top-k 前后的部分 elementwise ops；真正 dispatch、permutation 和 grouped GEMM 常需要 custom op 或最大 capacity + valid counts。

值得区分：

- **batch token 数动态**：一个 `SymInt N` 可能覆盖；
- **每 expert ragged shape 动态**：是长度为 $E$ 的数据依赖向量，难度更高；
- **physical placement 动态**：属于 runtime/EPLB generation，不应隐藏在编译常量中；
- **Top-$k$ 固定**：通常是模型常量，适合 specialization。

如果 compiler 把 expert placement pointer 或 Top-$k$ Python list 专门化，EPLB 重排后可能 guard failure/重新编译，甚至错误复用旧常量。应通过 tensor metadata 或明确 custom op contract 传递动态映射。

## 编译缓存怎样影响冷启动

编译 artifact 可以跨同一进程后续调用复用，serving 系统还会尝试把 cache 保存到磁盘，让新 worker 跳过部分 compile。Cache key 至少需要覆盖：

```text
PyTorch / compiler version
runtime code revision and compile passes
model architecture/revision
dtype and quantization
GPU architecture / target
CUDA/Triton relevant versions
parallel layout
graph/shape specialization
custom op ABI/schema
compiler options
```

Key 不完整会加载不兼容 artifact；过度细化则 cache 命中低。Cache 还要考虑：

- 多 ranks 同时编译造成 thundering herd；
- 网络文件系统 metadata 压力；
- 不完整写入与进程崩溃；
- cache 容量/淘汰；
- artifact 来源与供应链安全；
- 首次加载后仍需 CUDA Graph capture。

可以由 build pipeline 在目标 GPU 架构上预热常见模型/shape，再以内容 hash 发布只读 cache。任何 cache 都必须有 miss 后安全编译或 eager fallback。

## AOT Compilation 的机会与限制

JIT 把编译放在首次请求/worker warmup；Ahead-of-Time 能在部署前完成 trace、Inductor codegen、Triton compile/autotune 并打包 artifact，降低冷启动。

PyTorch 当前也提供实验性的 `torch.compile().aot_compile()` 路径，但文档明确其特性和限制仍在演进，例如 AOT full capture 对 graph breaks 更严格。生产使用应固定版本并验证：

- target GPU/driver compatibility；
- dynamic shape guards；
- custom op/link dependencies；
- artifact 可移植性；
- 编译器更新后的 invalidation。

AOT 不会消除模型加载、communicator 初始化、runtime buffer 建立与 CUDA Graph capture，冷启动预算仍要逐段测量。

## 正确性风险

Compiler 优化必须保持模型语义，但 serving 的 custom metadata/ops 让验证更重要。

### Alias 与 Mutation

KV Cache update、in-place residual、output buffer reuse 都涉及 mutation。如果 custom op schema 没有正确标注写入/alias，编译器可能错误重排或删除操作。

### Fake/Meta Kernel 不一致

Tracing 用 fake/meta implementation 推导 shape/stride，真实 CUDA implementation 若返回不同 layout，会造成后续 kernel 错误。

### Python Side Effect 被省略

Capture 的 graph replay 不会按 eager 次数重新运行被消除的普通 Python side effect。请求计数、cache lease、日志不能偷偷放在 model forward 内依赖执行次数。

### Guard 漏项

自定义 compiler 绕过/删除 guards 时，必须证明变化维度已由 graph 正确表示。盲目“去掉 recompile”可能让旧 graph 处理不兼容 Adapter、placement 或 shape。

### 数值变化

Fusion、reassociation、不同 reduction/tile 会改变浮点误差。验证要覆盖 logits、sampling 分布和生成任务，而不只比较少量 hidden states。

## 建立逐层对照

一个可靠验证矩阵：

```text
eager PyTorch baseline
stock torch.compile + eager backend   (Dynamo capture only)
stock torch.compile + Inductor
runtime custom compile + eager backend
runtime custom compile + Inductor
above + CUDA Graph
```

逐层开关可以回答：

- Dynamo tracing 本身是否改变语义；
- Inductor codegen 是否出错/变慢；
- runtime custom passes 是否有问题；
- CUDA Graph 静态 buffer 是否出错。

测试输入覆盖：

- 多个 batch/token sizes；
- Prefill、Decode、mixed；
- short/long context 与不同 KV block tables；
- TP/EP 多 rank；
- LoRA/Adapter combinations；
- speculative decoding；
- MoE 极端 routing；
- 量化与不同 attention backends。

对确定性路径比较 logits/hidden tolerance；对随机生成固定 RNG contract，并做统计与任务质量验证。

## 诊断工具应该怎样用

### `TORCH_LOGS`

常用类别包括：

```text
graph_breaks
guards
recompiles
dynamic
graph_code
```

先看 recompile/graph break 的具体源位置，不要从一个总“compile 慢”猜原因。

### `tlparse` / `TORCH_TRACE`

适合查看整个任务的 compile timeline、frames、guards 与 graph breaks，并形成可分享的诊断资料。vLLM 官方调试文档也优先建议使用 tlparse。

### `TORCH_COMPILE_DEBUG`

可输出 Inductor IR、fusion 前后信息和生成代码。用于确认目标 ops 是否被融合、是不是生成过多小 kernels。

### GPU Timeline

编译日志说明“捕获/生成了什么”，Nsight 等 timeline 说明“运行时到底执行了什么”。两者要关联：

- kernel 数是否下降；
- HBM traffic 是否下降；
- Graph breaks 间是否有 CPU gaps；
- custom op/collective 是否成新瓶颈；
- CUDA Graph replay 是否覆盖 compiled pieces。

## 性能评测的四笔账

### Steady-state latency

同 shape cache hit 后的 iteration/forward latency。报告 P50/P99，而非只取最好一次。

### Compile cost

$$
T_{compile,total}
=T_{trace}+T_{codegen}+T_{kernel\ compile}+T_{autotune}
$$

### Break-even requests

若 eager 每次耗时 $T_e$，compiled 稳态 $T_c$，一次 compile 成本 $T_{comp}$，粗略摊平需要：

$$
N_{break-even}
=\frac{T_{comp}}{T_e-T_c}
$$

Worker 生命周期或某 graph variant 调用次数低于它，编译可能得不偿失。

### Serving capacity

最终比较：

```text
TTFT / TPOT / E2E goodput
CPU utilization
GPU kernels and HBM traffic
worker startup/readiness
compile cache hit rate
memory/KV capacity
fallback rate
```

编译可能减少 iteration time，却增加 code/cache/Graph memory 或启动时间；容量规划需要全部纳入。

## 一条上线顺序

### 第一阶段：建立 eager 基线

- 固定模型、runtime、GPU 与 request trace；
- 保存正确性输出和 GPU timeline；
- 分解 Attention、MLP、MoE、sampling 与 CPU gaps。

### 第二阶段：只看 Dynamo

- 使用 eager backend 隔离 tracing；
- 记录 graph breaks、guards 和 recompiles；
- 清理 forward 内 Python side effects；
- 为 custom ops 补全 schema/fake/alias contract。

### 第三阶段：开启 Inductor

- 查看生成 graph/IR 与 kernel 数；
- 逐项开启 runtime custom passes；
- 比较 dynamic general graph 与热点 static specialization；
- 验证数值与多 shape。

### 第四阶段：建立 Cache/Warmup

- 固定 cache key/version；
- 在 readiness 前预编译常见路径；
- 防止多 ranks 同时重复编译；
- 记录 cache miss/failure/fallback。

### 第五阶段：叠加 CUDA Graph

- 选 capture buckets；
- full/piecewise/eager 分流；
- 监控静态 buffer 正确性和 Graph memory；
- 比较端到端 goodput 与恢复时间。

每阶段都有独立关闭开关。编译出错时先关闭最小子层，例如仅禁用 Inductor 或特定 pass，不必把 CUDA Graph、custom kernels 和整个 runtime 一起退回最慢模式。

## 哪些情况不应强行 Compile

- 核心时间已在高度优化的大 GEMM/Attention，胶水占比很小；
- shape/feature 组合极多，每个 variant 调用次数少；
- worker 生命周期短，无法摊平编译；
- custom ops 占几乎全部路径，Inductor 看不到可融合区域；
- Python side effects/控制逻辑与 tensor 计算高度交织；
- 编译 cache 无法可靠复用；
- 版本/模型迭代太快，artifact 经常失效；
- 正确性矩阵还没有覆盖生产功能。

此时保持高质量 eager/custom-kernel 路径，或只编译少数稳定子模块，通常比追求 full graph 更稳。

## 小结

`torch.compile` 不是一个单体 JIT，而是一条从 Python trace 到设备 codegen 的编译栈。Dynamo 沿实际 Python 路径生成 FX graph，用 guards 保证专门化假设；Inductor 再做 fusion、scheduling 和 Triton/CUDA codegen。Graph breaks、recompiles 与 dynamic shapes 决定这条链在动态 serving 中能否稳定复用。

可以抓住八条原则：

1. 编译收益主要来自 operator fusion、内存流量和 kernel 数，而 CUDA Graph 负责 launch replay；
2. Guards 是正确性边界，目标是避免无意义专门化，不是删除所有 guards；
3. Dynamic graph 与热点 static specialization 应按流量组合；
4. Graph break 保兼容，却切断 fusion 并增加运行时边界；
5. Custom op 应封装高度优化/语义复杂 kernel，同时正确声明 shape、alias 与 mutation；
6. vLLM-compile 是基于 PyTorch 内部 API 的定制系统，不等同于原生一行装饰器；
7. 编译 cache、预热与 AOT 决定冷启动，稳态 speedup 不能单独评价；
8. 逐层关闭 Dynamo、Inductor、自定义 passes 与 CUDA Graph，才能可靠定位问题。

编译器真正适合做的是把稳定的 tensor 计算变成更紧凑的 GPU 程序；请求调度、KV ownership、EPLB placement 和外部副作用仍应留在清楚的 runtime 边界。把边界画对，才能同时保住 PyTorch 的灵活性与生产级 LLM Serving 的性能。

## 参考资料

- [PyTorch: `torch.compile` API](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)
- [PyTorch: torch.compile Programming Model](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/programming_model.html)
- [PyTorch: Dynamo Deep-Dive](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_dynamo_deepdive.html)
- [PyTorch: Dynamic Shapes Core Concepts](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/dynamic_shapes_core_concepts.html)
- [PyTorch: torch.compile FAQ](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_faq.html)
- [vLLM: Debugging the torch.compile Integration](https://docs.vllm.ai/en/stable/design/debug_vllm_compile/)
- [vLLM: Fusion torch.compile Passes](https://docs.vllm.ai/en/stable/design/fusions/)
- [vLLM: CUDA Graphs Design](https://docs.vllm.ai/en/stable/design/cuda_graphs/)
