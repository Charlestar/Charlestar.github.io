---
layout: post
title: "Multi-LoRA Serving：一份 Base Model 怎样服务不同 Adapter"
subtitle: "从低秩增量、异构批处理到 Adapter Cache，理解多租户推理的计算与隔离边界"
date: 2026-08-21 09:00:00 +0800
last_modified_at: 2026-09-03
author: iStar
catalog: true
series: model-serving-agents
series_order: 30
technology_year: 2023
mathjax: true
tags: [LLM推理, 推理调度, GPU优化]
---

LoRA 让同一个 Base Model 可以低成本适配代码、客服、检索、行业问答等不同任务。训练侧只需保存一组小型低秩矩阵，但到了 Serving 侧，问题不再只是“怎样加载 LoRA”：同一个 Continuous Batch 里可能同时出现多个 Adapter，请求到达顺序不稳定，GPU 放不下全部 Adapter，Prefix/KV Cache 还必须区分它们产生的状态。

最浪费的部署方式是为每个 Adapter 启动一份完整 Base Model。若 Base Model 有数十 GB，而每个 Adapter 只有数十到数百 MB，复制 Base Model 会让显存和集群碎片远大于真正的定制参数。

Multi-LoRA Serving 的目标是：

```text
one shared base model
  + many independently versioned LoRA adapters
  + per-request adapter selection
  + heterogeneous batching on the same GPU
```

要做到这一点，Runtime 必须同时解决低秩计算、Adapter 驻留、请求调度、缓存身份和多租户隔离。

## LoRA 在一层 Linear 上增加了什么

设 Base Linear weight：

$$
W\in\mathbb{R}^{d_{out}\times d_{in}}
$$

对输入：

$$
X\in\mathbb{R}^{n\times d_{in}}
$$

普通 Linear 为：

$$
Y=XW^T
$$

LoRA 不直接训练完整 $W$，而是学习：

$$
A\in\mathbb{R}^{r\times d_{in}},
\qquad
B\in\mathbb{R}^{d_{out}\times r}
$$

其中 $r$ 远小于 $d_{in},d_{out}$。加入缩放 $s=\alpha/r$ 后：

$$
Y
=
XW^T
+
s(XA^T)B^T
$$

完整增量矩阵是：

$$
\Delta W=sBA
$$

但推理时通常不需要显式 materialize $\Delta W$。两次低秩 GEMM 的参数量为：

$$
P_{LoRA}=r(d_{in}+d_{out})
$$

相对完整矩阵：

$$
P_W=d_{in}d_{out}
$$

当 $r\ll d$ 时，Adapter 很小。这种参数效率正是共享 Base Model 的基础。

## 为什么单 Adapter 可以 Merge，多 Adapter Serving 却不适合

如果一个部署实例永远只服务 Adapter $a$，可以预先计算：

$$
W'_a=W+s_aB_aA_a
$$

之后推理仍执行普通 Linear。这消除了每 Token 的 LoRA 额外算子，适合固定模型导出。

Multi-LoRA 请求却可能是：

```text
request 0 → adapter sql-v7
request 1 → adapter support-zh-v3
request 2 → base model only
request 3 → adapter code-review-v5
```

同一个 batch 中不存在唯一的 $W'$。若每切换一组请求就把 Adapter merge/unmerge 到 Base weight：

- 需要反复读写大矩阵；
- 很难并发处理不同 Adapter；
- merge 期间必须保护 Base weight 不被其他请求使用；
- 量化 Base weight 与 LoRA dtype/layout 可能不允许直接融合；
- CUDA Graph 和 weight pointer 稳定性被破坏。

因此 Online Multi-LoRA 通常保持 Base weight 不变，在运行时按请求附加低秩分支。

## 一个混合 Batch 的数学形式

设 batch 按 Adapter 分成若干 token segments。属于 Adapter $a$ 的输入为 $X_a$：

$$
Y_a
=
X_aW^T
+
s_a(X_aA_a^T)B_a^T
$$

Base GEMM 可以把所有 tokens 合并：

$$
Y_{base}=XW^T
$$

因为所有请求共享 $W$。LoRA 分支则要知道每行 token 属于哪个 Adapter：

```text
rows [0, 1, 2, 3]      → adapter A
rows [4, 5]            → base only
rows [6, 7, 8, 9, 10]  → adapter B
row  [11]               → adapter C
```

最直接的方法是逐 Adapter 循环调用小 GEMM，但 Adapter 数量增加后会产生大量小 kernel launches，Tensor Core tile 利用率也很差。Punica、S-LoRA 一类系统的关键贡献之一，就是让不同 Adapter 的低秩计算能够以 segmented/batched kernel 形式共同执行。

计算没有变：每个 token 仍只能使用自己的 $A_a,B_a$。优化的是如何把许多不同 pointers、不同 segment lengths 和可能不同 ranks 组织成少量高效 GPU kernels。

## Heterogeneous Batching 为什么比普通 Batching 难

普通 Continuous Batching 主要面对不同 sequence lengths：每个 Decode step 为活跃序列各取一个 token，Base Model weights 完全相同。Multi-LoRA 又增加了 Adapter identity 这一维。

Runtime 需要维护：

```text
token row
  → request id
  → sequence state
  → adapter id + immutable revision
  → adapter slot / device pointer
  → local LoRA rank and target modules
```

如果先按 Adapter 分批，Kernel 形状更整齐，却会破坏 Continuous Batching 的低等待时间：冷门 Adapter 可能要等很久才凑够 batch。若完全按到达顺序混批，Adapter segments 很碎，低秩 kernel 与 metadata dispatch 开销上升。

调度器真正要权衡的是：

- request latency 与 Adapter locality；
- Base GEMM batch size 与 LoRA kernel fragmentation；
- Adapter load/eviction cost；
- Prefill tokens 和 Decode tokens 的计算差异；
- 不同租户的优先级与公平性。

不能只按“相同 Adapter 请求数”评估调度效果。

## Prefill 与 Decode 的 LoRA 形状不同

Prefill 时，一个请求可贡献数百到数万 input tokens，$X_a$ 的 $n$ 较大，两次 LoRA GEMM 更接近正常矩阵乘。Decode 时每个活跃请求通常只贡献一个新 token，单 Adapter 可能只有一两行，接近许多小 GEMV。

因此同一个 Multi-LoRA Kernel 要覆盖两类 workload：

| 阶段 | 每请求 Token Rows | 主要问题 |
| --- | ---: | --- |
| Prefill | 多 | Adapter groups 不均衡、与 Base GEMM/Attention 配合 |
| Decode | 通常 1 | 大量小 segments、launch 与 pointer indirection |

把 Prefill benchmark 的收益直接外推到 Decode，或只测固定 batch size 的 Decode，都可能误判线上性能。

## Segmented Kernel 需要哪些 Metadata

一种常见组织方式是先按 Adapter slot 对 token rows 做逻辑分组，再提供：

```text
segment_offsets: [0, 4, 6, 11, 12]
adapter_slots:   [slot_A, base, slot_B, slot_C]
token_indices:   permutation or row mapping
ranks:           [16, 0, 32, 8]
scales:          [s_A, 0, s_B, s_C]
```

Kernel 根据 segment 读取对应 $A,B$ pointers，完成：

$$
H_a=X_aA_a^T
$$

以及：

$$
\Delta Y_a=s_aH_aB_a^T
$$

最后把结果 scatter/add 回原 token rows。实现可以使用 grouped GEMM、segmented GEMV、fused gather/scatter 或专用 CUDA kernel；无论名字如何，必须保证 token→Adapter 映射在重排前后完全一致。

错误映射非常危险：shape、dtype 和 kernel 都可能合法，输出却混入另一租户的 Adapter 增量。

## 不同 LoRA Rank 会怎样影响 Kernel

Adapters 可能使用 $r=8,16,32,64$。一种简单实现按系统允许的最大 rank 分配统一 buffer，并把较小 rank padding 到 $r_{max}$。这使 shape 静态、便于 kernel 和 CUDA Graph，却浪费显存与计算：

$$
M_{allocated}
\propto
N_{slots}\times r_{max}
$$

即使大多数 Adapter 只有 $r=8$，配置成 256 仍会抬高 buffer budget。vLLM 的 LoRA 配置也显式要求给出可接受的最大 rank，这不是纯校验参数，而会影响资源规划。

另一种做法按 rank buckets 编译/选择 kernel，例如把 8/16 放一组、32/64 放一组。它减少 padding，却增加 kernel variants、batch fragmentation 和 graph captures。应根据实际 rank 分布选择，而不是只追求“支持任意 rank”。

## Adapter 不是一块连续 Weight

一个 Adapter 通常覆盖多个 Transformer layers 和 target modules，例如：

```text
q_proj, k_proj, v_proj, o_proj
gate_proj, up_proj, down_proj
lm_head or embedding (optional)
```

每个 target 都有自己的 $A,B$。Runtime 需要把 Adapter checkpoint 规范化成与 Base Model execution layout 一致的 slots：融合 QKV 的模型可能要重排独立 `q_proj/k_proj/v_proj`；gated MLP 可能把 gate/up stack 在同一 weight；MoE 还要区分 shared、per-expert 或 stacked adapter layout。

加载成功不等于 layout 正确。错误的 QKV 顺序或 MoE stacked dimension 可能不会立即报错，却会产生稳定的错误输出。

Adapter manifest 至少应包含：

- Base Model id 与不可变 revision/hash；
- architecture 和 target module mapping；
- rank、alpha/scaling、dtype；
- tensor names、global shapes 与 layout version；
- tokenizer/vocabulary compatibility；
- Adapter 自身的 immutable revision 与 checksum。

## Adapter Cache 为什么需要分层

所有 Adapter 都放在 GPU 上最简单，但显存有限。S-LoRA 的设计说明了常见方向：大量 Adapter 保存在 host memory，当前请求需要的部分再进入 GPU；GPU 侧使用统一或分页式的内存管理降低不同 rank/size 带来的碎片。

可以把驻留分为：

```text
remote/object storage
  → local disk cache
  → host memory adapter cache
  → pinned staging
  → GPU adapter slots
```

请求延迟取决于命中层级：

$$
T_{request}
=
T_{queue}
+T_{adapter\ resolve}
+T_{adapter\ load}
+T_{prefill}
+T_{decode}
$$

若只报告已驻留 Adapter 的 TTFT，会完全隐藏冷加载成本。容量规划应分别给出 GPU hit、host hit、disk/remote miss 的 latency 分布。

## Eviction 不能只看 LRU

简单 LRU 会逐出最久未使用 Adapter，但 Serving 还要考虑：

- Adapter 是否仍被 active requests 引用；
- 是否有排队请求即将使用；
- 加载成本和大小；
- 租户优先级与保底驻留；
- 热度预测和突发流量；
- 同 Adapter 的不同 revisions 是否可共存；
- GPU slot 是否被 CUDA Graph 或 kernel metadata 引用。

一个 Adapter slot 至少需要 reference count 或 generation/version。不能在 request 仍 Decode 时复用 slot 给另一个 Adapter，否则下一 Token 会静默切换模型。

安全替换流程类似：

```text
mark old slot draining
  → stop assigning new requests
  → wait for active references to reach zero
  → invalidate related graphs/cache entries
  → load and verify new immutable revision
  → atomically publish new slot mapping
```

## 调度器要不要等待相同 Adapter

假设 Adapter A 每毫秒都有请求，Adapter B 每秒只有一个请求。若只追求 Adapter locality，B 会长期等待；若每次都立即把 B 插入 batch，低秩 kernel segments 会变碎。

可以给等待设置上限：

- 在短窗口内聚合同 Adapter 请求；
- 超过 deadline 后允许异构混批；
- 为高优租户预留 token budget，而不是永久 GPU slot；
- Prefill 和 Decode 分别设置 batching 策略；
- 把 Adapter load 纳入 admission control，避免接收后才发现没有可用 slot。

衡量调度器时需要同时看：

- per-Adapter/tenant TTFT 与 TPOT；
- deadline miss 和 starvation；
- Base batch occupancy；
- 每步 Adapter segment 数；
- Adapter GPU/host cache hit ratio；
- load/eviction bytes 与 stalls。

Aggregate throughput 可能很好，却把长尾 Adapter 饿死。

## Prefix Cache 与 KV Cache 必须包含 Adapter 身份

即使两个请求拥有完全相同的 prompt tokens，只要 Adapter 不同，它们经过 Linear/Attention 后的隐藏状态通常也不同。由 Adapter A 产生的 KV Cache 不能直接给 Adapter B 使用。

Prefix cache key 至少应包含：

```text
base_model_revision
adapter_id + immutable_revision
token_ids / parent_block_hash
position and attention configuration
multimodal input identity if present
```

只使用可变名称 `customer-support/latest` 不安全：当 alias 指向新 revision 时，旧 KV blocks 仍可能命中。请求进入系统时应把 alias 解析成 immutable Adapter revision，并让整个请求生命周期保持不变。

如果 Adapter 只作用于非常靠后的模块，理论上某些更早层的中间结果可能复用，但这需要 layer-aware cache contract；普通 KV Cache 包含各 Attention layer 的状态，默认应把 Adapter revision 纳入身份。

## Tensor Parallel 下 LoRA 怎样切

LoRA 必须跟随 Base Linear 的 logical sharding。以：

$$
\Delta W=BA
$$

为例：

- Column-parallel Base weight 沿 output dimension 切分时，$B$ 的 output rows 可随各 TP ranks 分片，$A$ 可复制或按具体 kernel contract 处理；
- Row-parallel Base weight 沿 input dimension 切分时，$A$ 的 input columns 跟随输入 shard，产生的 LoRA output 仍可能需要与 Base partial output 一起归约；
- fused QKV、GQA、vocab parallel 和 MoE experts 各有不同 shard axes；
- Adapter checkpoint 若以未分片 PEFT layout 保存，加载器要按目标 TP layout 转换。

关键不是背固定规则，而是让 $A,B$ 的乘积与本 rank 的 Base $W_r$ 表示同一段 logical $\Delta W_r$，并在相同边界完成必要 collective。

从 TP=1 验证通过，不代表 TP=4 的 Adapter layout 正确。必须聚合 global output 与 reference 对比。

## Quantized Base Model 与 LoRA 的 Dtype 边界

Base weight 可以是 INT4/INT8/FP8，LoRA 常为 FP16/BF16。计算可能是：

```text
quantized base GEMM → accumulation/output dtype
BF16/FP16 LoRA GEMMs
→ scaled add
```

要明确：

- Base GEMM accumulation dtype；
- LoRA input 是否复用量化前 activation；
- $A,B$ 的 dtype 和 scale；
- 两条路径在哪里相加；
- Tensor Parallel reduction 在相加前还是相加后；
- CUDA Graph 是否捕获稳定的 Adapter pointers。

直接把 LoRA merge 进量化 weight 会触发重新量化，误差与 scale 也会变化；Online low-rank branch 更灵活，但要承担额外 kernel 和 memory reads。

## 动态加载是一个安全边界

允许 API 动态指定任意 Adapter path，会把文件系统和模型加载权限暴露给请求方。生产环境应区分控制面与数据面：

```text
control plane:
  authenticate → fetch allow-listed artifact → verify manifest/checksum
  → scan/convert → register immutable revision

data plane:
  request references registered adapter id/revision only
```

不要让普通生成请求携带服务器本地路径或任意远程 URL。Adapter artifact 应使用安全 tensor 格式，限制 tensor shape/size、target modules、rank 和总显存预算；加载失败要保持旧 revision 可用，而不是留下半写入 slot。

多租户系统还要防止某租户通过大量冷 Adapter 请求反复冲刷 GPU cache，影响其他租户。Admission control 应限制每租户的注册数量、冷加载带宽、并发和驻留预算。

## 正确性测试怎样做

应先建立逐 Adapter reference：

1. Base Model 单独输出；
2. 将 Adapter merge 到 FP reference weight 后输出；
3. Online LoRA branch 输出；
4. 混合 Adapter batch 中每个请求的输出；
5. Prefill 与逐 Token Decode；
6. TP=1 与目标 TP layout；
7. Adapter eviction/reload 前后；
8. Prefix/KV Cache hit 与 miss；
9. dynamic revision 切换时的 in-flight request。

比较不能只看最终自然语言是否相似。至少对 selected layer outputs、logits、next-token ids 和多步 Decode 做数值或确定性验证。

构造混批时要覆盖：

- base-only 与多个 Adapters 同时存在；
- 相同名称不同 revisions；
- 不同 ranks 和 target modules；
- Adapter segment 长度为 1；
- Prefill 长短差异极大；
- GPU cache miss 与并发 eviction；
- 请求取消后 reference count 回收。

## 性能 Benchmark 应怎样设计

至少分开报告：

### Kernel 层

- $n$、$d_{in}$、$d_{out}$、rank $r$；
- Adapter segment 数与长度分布；
- Prefill/Decode；
- Base GEMM 与 LoRA 两阶段耗时；
- gather/scatter、metadata 与 launch 开销；
- dtype、quantization、TP size。

### Engine 层

- tokens/s、TTFT、TPOT、P99；
- 活跃 Adapter 数和请求 Zipf/均匀分布；
- GPU/host cache hit ratio；
- 冷加载、eviction 和 H2D bytes；
- 每 Decode step Adapter segments；
- CUDA Graph hit ratio；
- 每租户 fairness 与 deadline miss。

### 容量层

- Base Model 与 KV Cache 占用；
- 每个 Adapter 的实际/分配 bytes；
- `max rank` padding 浪费；
- GPU slots、host cache 和 remote storage 容量；
- workload 热度变化后的 thrashing 临界点。

只用 4 个常驻 Adapter、均匀请求做测试，会掩盖真实多租户 workload 的长尾和冷加载问题。

## 一条可执行的落地路径

1. **固定 Base revision 与 Adapter schema**：明确 target modules、rank、scale 和 layout；
2. **实现单 Adapter Online Branch**：与 merge 后 reference 比较 logits；
3. **加入 Base-only + 双 Adapter 混批**：验证 token→Adapter 映射；
4. **分别优化 Prefill 与 Decode**：不要假设同一 kernel 最优；
5. **建立 immutable Adapter registry**：请求只引用已验证 revision；
6. **加入 Host/GPU Cache**：先做 reference count，再做 eviction policy；
7. **让 Prefix/KV Cache Key 包含 Adapter revision**；
8. **适配 TP 与量化路径**：逐 tensor 检查 logical shard；
9. **为调度器加入 Adapter locality 与 deadline**：防止长尾 starvation；
10. **做故障与安全测试**：坏 artifact、超大 rank、取消、热更新和 cache thrash；
11. **用真实热度分布验收**：同时报告吞吐、P99、命中率与租户公平性。

## 常见误区

### “LoRA 很小，所以 Serving 开销可以忽略”

参数量小不代表运行时免费。Decode 中许多 rank 很小、segment 很短的 GEMM 容易被 launch、gather/scatter 和 metadata 开销主导。

### “相同 Prompt 可以跨 Adapter 复用 KV Cache”

Adapter 改变层输出，KV 状态通常也改变。缓存身份必须包含 Base 与 Adapter 的不可变 revisions。

### “把请求按 Adapter 分组就能获得最佳性能”

过度等待相同 Adapter 会牺牲长尾延迟与公平性。调度器要在 locality 和 deadline 之间折中。

### “Adapter 加载成功就说明 Layout 正确”

QKV、gated MLP、MoE 和 TP layout 错误可能完全不报 shape error，却产生错误 logits。

### “动态加载接口只是运维便利功能”

它同时是 artifact、文件系统、显存和多租户资源的控制面，必须鉴权、验证、限额并使用 immutable revision。

## 小结

Multi-LoRA Serving 把一份共享 Base Model 变成许多逻辑模型，但系统必须在每个 Token 上保持 Adapter 身份：

1. Base GEMM 可跨所有请求共享，LoRA 的两次低秩计算按 Adapter segments 执行；
2. Prefill 是较大的 segmented GEMM，Decode 更像大量小 GEMV，两者优化重点不同；
3. Grouped/segmented kernel 减少逐 Adapter launch，但要求严格的 token→Adapter mapping；
4. 不同 ranks、target modules、fused weights、TP 和量化会改变 Adapter layout；
5. Host/GPU 分层 cache 解决容量，reference count 与 immutable revision 保证热更新安全；
6. 调度器要在 Adapter locality、Continuous Batching、deadline 和公平性之间取舍；
7. Prefix/KV Cache、CUDA Graph 与所有运行时缓存都要包含 Adapter identity；
8. 动态加载属于控制面安全边界，不能让普通请求直接加载任意路径；
9. 验收必须覆盖混批、冷加载、eviction、TP 与多步 Decode，而不只是单 Adapter demo。

下一篇会讨论结构化生成：JSON Schema、Regex 与 Context-Free Grammar 怎样被编译成逐 Token 状态机，为什么“保证合法格式”不等于“保证内容正确”，以及 Grammar Mask 如何进入 GPU Sampling 的关键路径。

## 参考资料

- [Punica: Multi-Tenant LoRA Serving](https://arxiv.org/abs/2310.18547)
- [S-LoRA: Serving Thousands of Concurrent LoRA Adapters](https://arxiv.org/abs/2311.03285)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [vLLM: LoRA Adapters](https://docs.vllm.ai/en/latest/features/lora/)
- [vLLM: Automatic Prefix Caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)
