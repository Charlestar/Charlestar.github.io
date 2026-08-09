---
layout: post
title: "FlashInfer：面向 LLM Serving 的可组合 GPU Kernel 库"
subtitle: "从动态请求形状到 plan/run、JIT 与后端分派"
date: 2026-05-13
last_modified_at: 2026-08-09
author: iStar
catalog: true
series: gpu-runtime-precision
series_order: 50
technology_year: 2025
mathjax: true
tags: [AI Infra, FlashInfer, CUDA, LLM推理]
---

FlashAttention 证明了改变数据搬运方式可以显著加速精确注意力，但 LLM serving 面对的问题比一组规则的 $Q、K、V$ 更复杂：每个请求长度不同，KV Cache 分散在页中，batch 每轮变化，prefill 与 decode 的 Query 长度也完全不同，还可能同时出现 GQA、MLA、低精度缓存、滑动窗口和自定义 mask。

FlashInfer 的定位，是为这些 serving 形状提供可组合的 GPU kernel 与生成基础设施。它不是完整推理服务器，不负责 HTTP、请求队列和业务路由；上层的 vLLM、SGLang 等 runtime 可以把具体计算交给它。

理解 FlashInfer 的关键不在于记住一串 API 名，而是看清三层边界：runtime 提供什么 metadata，FlashInfer 怎样据此选择/生成 kernel，GPU 又如何执行不同请求的注意力。

## Kernel、Kernel Library 与 Serving Engine

这三个概念经常被混在一起：

| 层次 | 负责什么 | 例子 |
| --- | --- | --- |
| GPU kernel | 完成一次具体并行计算 | attention、sampling、grouped GEMM |
| kernel library | 提供多类 kernel、生成和分派 | FlashInfer |
| serving engine | 管理请求、缓存、调度、API 和 worker | vLLM、SGLang |

可以类比餐厅：kernel 是某位厨师完成一道工序，kernel library 是后厨的菜谱和工位集合，serving engine 则负责接单、排队、备料、组合工序和上菜。

安装了 FlashInfer，不代表每次 attention 都会调用它；上层引擎可能根据形状选择 FlashAttention、cuDNN、Triton、TensorRT-LLM kernel 或其他 backend。反过来，单独调用 FlashInfer 也不会自动获得 continuous batching 或 prefix cache，因为那些属于 runtime 状态管理。

## 为什么 Serving 需要很多 Attention Kernel

训练中的 batch 常被 padding 成规则矩形，每个样本序列长度接近。在线服务则可能在同一轮出现：

```text
request A: prefill 2048 new tokens, KV length 0
request B: decode 1 new token,    KV length 8192
request C: chunked prefill 256,   KV length 4096
request D: speculative verify 5,  KV length 1200
```

即使数学上都计算 $\operatorname{softmax}(QK^T)V$，性能最优的并行方式也不同：

- 长 Query 的 prefill 有较大矩阵，容易利用 Tensor Core；
- 单 token decode 要读取长 KV，更偏内存带宽；
- GQA 的 Query heads 多于 KV heads，读取映射不同；
- paged KV 需要间接访问 page table；
- ragged batch 中每条请求的有效范围不同；
- MLA 的 latent cache 与普通 MHA/GQA 布局不同；
- FP8/NVFP4 KV 还要读取 scale 并反量化。

不存在一个固定 tile、固定 launch 形状的 kernel 能在所有组合上都最优。FlashInfer 的“可定制”就是在这片组合空间中生成、规划或分派更合适的实现。

## Paged KV Cache 的输入长什么样

假设 page size 为 4，物理 KV pool 中有许多 pages。三个请求使用的页表可能是：

```text
request 0 -> page [7,  2], last_page_len = 3
request 1 -> page [9],     last_page_len = 4
request 2 -> page [1, 5, 8], last_page_len = 1
```

为了把不等长列表放进张量，常用 CSR 风格的 `indptr + indices`：

```text
paged_kv_indptr = [0, 2, 3, 6]
paged_kv_indices = [7, 2, 9, 1, 5, 8]
last_page_len = [3, 4, 1]
```

`indptr[i]:indptr[i+1]` 给出第 $i$ 个请求的 page IDs。于是：

```text
request 0 pages = indices[0:2] = [7, 2]
request 1 pages = indices[2:3] = [9]
request 2 pages = indices[3:6] = [1, 5, 8]
```

Query 也可以用 `qo_indptr` 描述每个请求有多少新 token。decode 请求范围长度为 1，chunked prefill 可能为数百。kernel 读取这些 metadata 后，才能为每个请求找到正确 Query 和 KV 范围。

KV pool 的常见布局包括：

```text
NHD: [num_pages, page_size, num_kv_heads, head_dim]
HND: [num_pages, num_kv_heads, page_size, head_dim]
```

NHD 更接近 token/page 内连续，HND 把 head 放在更外层。最佳布局取决于 kernel 的访存方式；调用方必须让实际 tensor 与 wrapper 的 `kv_layout` 完全一致，否则轻则结果错误，重则越界访问。

## Ragged KV 与 Paged KV 的差别

Ragged 表示把各请求的有效 K/V 紧凑拼接：

```text
K/V storage: [request0 tokens][request1 tokens][request2 tokens]
kv_indptr:    [0, len0, len0+len1, len0+len1+len2]
```

它适合已经有连续拼接数据、一次性 prefill 等场景。Paged 表示则通过 page IDs 指向长期 KV pool，更适合请求动态增长和释放的 serving。

两者都避免把 batch pad 到统一最大长度，但地址结构不同。runtime 决定 cache 的物理管理，FlashInfer wrapper 根据相应 metadata 执行 attention；不能把 ragged/paged 只当作同一个 API 的不同名字。

## 为什么接口分成 plan 和 run

执行一个动态 batch 前，需要完成不少不属于核心矩阵乘的工作：解析 indptr、估计每条序列的工作量、把任务分配给 thread blocks、准备临时 buffer 和选择 kernel variant。

FlashInfer 的许多 wrapper 将过程分成两步：

```text
qo_indptr + page table + lengths + heads + dtype
                      │
                      ▼
                    plan()
          scheduling metadata / workspace
                      │
               can be reused when valid
                      ▼
             run(query, kv_cache)
                      │
                      ▼
                 attention output
```

`plan()` 处理问题规格与辅助数据结构，`run()` 留在热路径执行 GPU 计算。如果同一布局在多个 step 可复用计划，就能减少重复调度开销。

“可复用”不等于 metadata 永远不变。请求加入、退出、page table 改变或 Query 长度变化时，上层 runtime 必须按 API 契约重新 plan 或更新固定 buffer。拿旧计划运行新布局，会导致错误地址或任务划分。

官方文档也明确区分了 CUDA Graph/`torch.compile` 捕获边界：plan 本身通常不放进 graph，run 则使用预分配、稳定地址的 buffer 进入捕获路径。

## 用工作量规划解决长短不一

固定地让“一个请求对应一个 CTA/thread block”很容易负载不均：一个请求 KV 长度 128，另一个 128k，前者很快结束，后者独占工作。

更合理的 scheduler 会根据 Query 数、KV 长度和 head 数把长请求拆成多个 tile，让更多 SM 参与；之后再合并部分 attention 结果。短请求则可能合并调度，减少 launch 浪费。

这也是 serving attention 与规则训练 attention 的差别：GPU 工作划分不仅取决于 tensor 总尺寸，还取决于每条请求的长度分布。FlashInfer 论文讨论的 load-balanced scheduling，目标是在动态 workload 下保持并行度，同时兼容需要静态资源配置的 CUDA Graph。

## Attention state 为什么可以合并

对一个 Query，将 KV 集合拆成互不重叠的两段 $A、B$。每段 attention 可以保存三个状态：

- 局部输出累积；
- 局部 log-sum-exp 或等价归一化统计量；
- 对应的数值尺度。

与在线 Softmax 相同，只要正确重标定，就能把两段状态合并成对 $A\cup B$ 的精确 attention。抽象写成：

$$
S(A\cup B)=\operatorname{merge}(S(A),S(B))
$$

这个递归性质带来多种 serving 优化：

- 长 KV 可由多个 CTA 分段计算，再合并；
- 共享前缀和每请求私有 suffix 可以分开 attention；
- 多级存储中的 KV 分层计算后再合并；
- cascade attention 复用公共状态，减少共享部分的重复访问。

它不是简单平均两个输出。Softmax 的分母跨越全部 KV，合并必须使用每段的归一化统计量进行缩放，否则数学结果会改变。

## Cascade Attention 解决什么

假设一批请求共享同一长前缀 $K_s,V_s$，但拥有不同 suffix $K_i,V_i$。朴素做法为每个请求都重新读取共享前缀：

```text
Q0 attends [shared prefix + suffix0]
Q1 attends [shared prefix + suffix1]
Q2 attends [shared prefix + suffix2]
```

Cascade attention 可以先计算/组织对共享前缀的 attention state，再与每个请求的私有 state 合并。这与 RadixAttention 的职责互补：runtime 判断哪些 token/KV 真正共享，kernel 层利用共享结构减少计算或内存访问。

收益取决于共享前缀长度、batch、Query 是否相同/相关以及具体 wrapper；并不是打开某个 API 后所有 prefix cache hit 都自动获得相同比例的 kernel 加速。

## JIT 为什么适合 Attention 变体

一个 attention variant 可能在 score 上加入：

- causal/sliding-window/custom mask；
- RoPE 或其他位置处理；
- logits soft cap；
- KV scale 与反量化；
- 自定义 score transform。

如果为每种 head dimension、dtype、page size、mask 和 GPU 架构预编译全部组合，包体、编译时间和维护成本都会膨胀。JIT（Just-In-Time compilation）在首次遇到具体规格时生成或编译专用 kernel，把静态信息固化：

```text
variant = hash(
  GPU architecture,
  CUDA/toolchain,
  head dimensions,
  q/kv/output dtype,
  page size,
  mask/position variant,
  compile options
)
```

编译器由此可以消除不需要的分支、选择 tile 和生成对应指令。但 JIT 把成本移到了首次使用：冷启动可能出现明显延迟，多进程同时编译还会争抢资源。

生产环境应：

- 枚举允许上线的模型与 variant，提前预热；
- 持久化并验证编译缓存；
- 将缓存键与 GPU、驱动、CUDA、PyTorch 和 FlashInfer 版本绑定；
- 限制用户输入直接制造无限 variant；
- 编译失败时记录 backend 回退，而不是静默忽略。

## Workspace 与 CUDA Graph

许多 wrapper 需要调用方或内部持有 workspace。预分配的价值是避免每个 decode step 动态申请显存，并保持 CUDA Graph 所需的地址稳定。

动态 serving 与静态 graph 的矛盾通常通过固定容量 buffer 解决：

```text
buffer capacity: max_batch / max_pages / max_tokens
actual workload: indptr + lengths + valid counts
```

每轮更新实际 metadata，但 tensor 地址和最大形状保持。workspace 太小会无法规划目标 workload；设置过大又会挤压 KV Cache，降低并发容量。它应该与模型权重、KV pool 和其他 graph buffer 一起做显存预算。

使用 CUDA Graph 也不等于所有 CPU 开销消失。请求调度、page 分配、tokenization 和 `plan()` 仍可能在 graph 之外，实际 profile 要同时观察 CPU timeline 与 GPU timeline。

## FlashInfer 不只包含 Attention

项目当前还覆盖 sampling、speculative sampling、RoPE/page 操作、量化、grouped GEMM 与 MoE 等 serving kernel。它们共享同一设计逻辑：针对 LLM runtime 中频繁出现但形状多变的算子，提供优化实现和组合接口。

这不意味着上层 engine 会统一采用整套 FlashInfer。一个系统可能用 FlashInfer attention、另一套 MoE communication、PyTorch sampling；backend dispatch 应逐算子、逐阶段观察。

## 如何确认实际调用了什么

“依赖列表中有 FlashInfer”不能证明热点走了 FlashInfer。确认路径可按以下层次：

1. 查看引擎启动日志中的 attention/sampling/MoE backend；
2. 在已知不支持的形状上强制 backend，观察是否明确报错；
3. 用 PyTorch profiler/Nsight Systems 查看实际 kernel 名和时间线；
4. 使用 FlashInfer 提供的调用日志/转储能力定位 wrapper；
5. 将目标 backend 与 reference/backend-off 基线做 A/B。

常见回退原因包括 GPU 架构、head dimension、dtype、mask、KV layout、量化 scale 或功能组合不支持。回退不是一定错误，静默回退却会让性能结论失真。

## 正确性测试应该覆盖哪些边界

先用小尺寸 FP32/PyTorch attention 建立 reference，再测试实际低精度路径。矩阵至少包括：

- prefill、decode、append 与 speculative verify；
- causal、non-causal、sliding window、自定义 mask；
- MHA、GQA、不同 QK/VO head dimension；
- ragged 与 paged KV、NHD 与 HND；
- 空/极短序列、非整页尾部、极长序列；
- BF16/FP16/FP8/NVFP4 及 scale 边界；
- CUDA Graph 下多轮 page table 更新；
- 多 GPU 架构与软件版本。

低精度和不同归约顺序不要求逐 bit 相等，但必须满足预先定义的误差预算，并检查 NaN/Inf、全 mask 行和越界 page index。

## Microbenchmark 怎样避免误导

单个规则 shape 的 kernel latency 只是起点。Serving 评测应扫描：

```text
batch size
query length distribution
context length distribution
num_qo_heads / num_kv_heads
head dimensions
page size and last-page utilization
dtype / quantization
prefill-decode mixture
shared-prefix ratio
```

先预热 JIT 与 allocator，再用 CUDA Event 或官方 benchmark 测 GPU 时间；记录计划时间和首次编译时间，但不要混入稳态 kernel latency。随后按线上 shape 频率加权，最后回到 TTFT、TPOT、吞吐、峰值显存和 SLO goodput。

如果 profiler 显示主要时间在 tokenizer、MoE all-to-all、CPU scheduler 或网络流式输出，更换 attention backend 不会产生显著端到端收益。Kernel benchmark 的第一名也不保证整个 engine 第一名。

## 小结

FlashInfer 位于 serving runtime 与 GPU 硬件之间：runtime 把动态请求压成 indptr、page table、长度和 layout，FlashInfer 用 plan 建立工作划分，再通过专用或 JIT kernel 执行 attention、sampling 与其他算子。

它的技术主线是处理异构：不同 Query/KV 长度、不同缓存布局、不同注意力变体和不同硬件。掌握 paged/ragged metadata、attention state 合并、plan/run 边界和 backend dispatch 后，就能判断一次性能变化究竟来自 kernel、本轮形状，还是上层调度与缓存。

## 参考资料

- [FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving](https://arxiv.org/abs/2501.01005)
- [FlashInfer 官方文档](https://docs.flashinfer.ai/)
- [FlashInfer Attention API](https://docs.flashinfer.ai/api/attention.html)
- [FlashInfer KV Cache Layout 教程](https://docs.flashinfer.ai/tutorials/kv_layout.html)
- [FlashInfer 官方仓库](https://github.com/flashinfer-ai/flashinfer)
