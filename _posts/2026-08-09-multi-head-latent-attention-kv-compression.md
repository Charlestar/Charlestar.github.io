---
layout: post
title: "MLA：为什么一份 Latent 可以代替多头 KV Cache"
subtitle: "从联合低秩压缩、矩阵吸收到 Decoupled RoPE 与 FlashMLA 执行模式"
date: 2026-08-09 17:58:00 +0800
last_modified_at: 2026-08-09
author: iStar
catalog: true
series: attention-long-context
series_order: 50
technology_year: 2024
mathjax: true
tags: [AI Infra, MLA, DeepSeek, KV Cache, FlashMLA, LLM Serving]
---

自回归生成每次只产生一个新 token，却要让它关注此前所有 token。为了避免重复计算历史 Key 和 Value，推理引擎会为每层保存 KV Cache。上下文越长、并发越高，这份缓存越容易成为显存容量和 Decode 带宽的主要成本。

GQA 与 MQA 通过减少 KV heads 来缩小缓存：多个 Query heads 共享一组 K/V。Multi-head Latent Attention（MLA）走了另一条路线：不直接缓存每个 head 的完整 K/V，而是先把 Key 与 Value 联合压缩到每 token 一份低维 latent，推理时通过矩阵吸收直接在 latent 表示上完成 attention 的主要计算。

MLA 的执行逻辑可以先概括为：

```text
hidden state h_t
  ├─ down projection → joint KV latent c_t^KV ───────┐
  └─ small positional projection → RoPE key k_t^R ──┤→ cache
                                                     │
new query → absorbed query × cached latent/RoPE key ┘
          → attention probabilities
          → weighted latent sum
          → absorbed output projection
```

真正被缓存的是 `c_t^KV + k_t^R`，不是先压缩、使用时再无条件还原出的完整多头 K/V。理解矩阵吸收为何成立、RoPE 为何要单独拆出，才算理解 MLA 的系统价值。

## 先计算标准 MHA 的 KV 成本

设 attention 有 $n_h$ 个 heads，每个 Key/Value head 的维度为 $d_h$，层数为 $L$，上下文长度为 $T$，batch 中有 $B$ 条序列，缓存 dtype 占 $b$ bytes。

标准 MHA 每 token、每层要保存：

$$
N_{MHA}=2n_hd_h
$$

其中系数 2 对应 Key 与 Value。整个 batch 的理论主体为：

$$
M_{KV}=B\cdot T\cdot L\cdot2n_hd_h\cdot b
$$

例如 $n_h=32$、$d_h=128$、$L=32$、BF16 时，每 token 跨全模型需要：

$$
32\cdot2\cdot32\cdot128\cdot2
=524{,}288\ \text{bytes}
$$

也就是约 512 KiB。单条 32K 上下文仅 KV 主体便约 16 GiB；paged allocator、block metadata、padding 和碎片还会增加实际占用。

这个例子不是某个 DeepSeek 模型的配置，只用于展示为什么“每 token 少存一些元素”会直接改变可承载并发。

## MQA 与 GQA 压缩的是 Head 数量

若只有 $n_{kv}$ 个 KV heads，而 Query 仍有 $n_h$ 个 heads，则：

$$
N_{GQA}=2n_{kv}d_h
$$

当 $n_{kv}=1$ 时就是 MQA。它们把多个 Query heads 映射到相同 K/V heads：

```text
MHA:  one K/V head per Query head
GQA:  one K/V head per Query group
MQA:  one K/V head shared by all Query heads
```

好处是缓存 layout 与普通 attention 相近，很多 kernel 容易扩展支持。代价是减少了独立 KV 表示的数量，模型容量与质量可能受到影响。已有 MHA checkpoint 也不能只改一个 `num_key_value_heads` 就无损转换。

MLA 保留多头 Query 的表达能力，但把每个 token 的 K/V 信息先放进一个共享的低秩 latent，再由各 head 的投影解释这份 latent。它压缩的是 **joint representation rank**，不是简单减少 KV heads。

## Joint KV Compression 保存什么

设当前层输入为：

$$
h_t\in\mathbb{R}^{d}
$$

MLA 先用 down-projection 得到低维 KV latent：

$$
c_t^{KV}=W^{DKV}h_t,
\qquad
c_t^{KV}\in\mathbb{R}^{d_c}
$$

其中 $d_c$ 远小于把所有 heads 的 K/V 展开后的总维度。训练的直接表达中，可以再用两个 up-projections 得到 content Key 与 Value：

$$
k_t^C=W^{UK}c_t^{KV}
$$

$$
v_t^C=W^{UV}c_t^{KV}
$$

$W^{UK}$ 与 $W^{UV}$ 的不同切片仍可为不同 heads 产生不同 Key/Value，因此这不是所有 heads 共享同一个完整 K/V vector 的 MQA。

关键是 $k_t^C$ 和 $v_t^C$ 不一定要写入 KV Cache。只要保留 $c_t^{KV}$，它们在数学上可以恢复；更进一步，优化推理甚至不需要显式恢复它们。

## Query 也可以经过低秩投影

MLA 还对 Query 使用单独的低秩路径：

$$
c_t^Q=W^{DQ}h_t
$$

$$
q_t^C=W^{UQ}c_t^Q
$$

Query compression 主要降低 Query 投影相关的 activation/计算成本，并不是 KV Cache 节省的直接来源。历史 Query 不会被后续 token 重读，缓存关注的仍是 KV 路径。

这两个 latent 也不能混为一谈：

- $c_t^{KV}$ 属于历史状态，会进入每层 KV Cache；
- $c_t^Q$ 只服务当前 token/query 的计算，生命周期更短；
- 它们使用不同的 down/up projection 与维度。

## 第一次矩阵吸收：不还原 Content Key

对第 $i$ 个 head，设其 content Key up-projection 为 $W_i^{UK}$。若先还原 Key，attention score 的 content 部分是：

$$
(q_{t,i}^C)^T k_{j,i}^C
=
(q_{t,i}^C)^T W_i^{UK}c_j^{KV}
$$

利用矩阵乘法结合律：

$$
(q_{t,i}^C)^T W_i^{UK}c_j^{KV}
=
\left((W_i^{UK})^Tq_{t,i}^C\right)^Tc_j^{KV}
$$

定义吸收后的 Query：

$$
\tilde q_{t,i}^C=(W_i^{UK})^Tq_{t,i}^C
$$

于是：

$$
(q_{t,i}^C)^Tk_{j,i}^C
=
(\tilde q_{t,i}^C)^Tc_j^{KV}
$$

运行时只需把当前 Query 投影一次，再与所有历史 latent 做 dot product。无需为每个历史 token、每个 head materialize $k_{j,i}^C$。

这会把 Decode 中持续读取的 Key 数据从“多头展开表示”改为“一份共享 latent”。若 runtime 仍对每步历史 cache 执行 `latent → full K`，就重新引入了大量计算和临时数据，失去矩阵吸收的主要价值。

## 第二次矩阵吸收：延后 Value Up-projection

设 attention probability 为 $p_{t,j,i}$。传统写法先还原每个历史 Value，再加权：

$$
o_{t,i}
=
\sum_jp_{t,j,i}v_{j,i}^C
=
\sum_jp_{t,j,i}W_i^{UV}c_j^{KV}
$$

因为 $W_i^{UV}$ 与历史位置 $j$ 无关，可以移到求和外：

$$
o_{t,i}
=
W_i^{UV}
\left(
\sum_jp_{t,j,i}c_j^{KV}
\right)
$$

也就是说，attention kernel 先在 latent space 中对历史 $c_j^{KV}$ 做 weighted sum，只对聚合后的一个结果执行 Value up-projection。再将各 heads 的 $W_i^{UV}$ 与最终 output projection $W^O$ 组合，就能进一步减少中间 materialization。

两次吸收共同改变了 Decode 的数据流：

```text
naive:
  read latent history
  → expand full per-head K/V for every position
  → attention

absorbed:
  transform current Q once
  → attend directly over latent history
  → transform aggregated result once
```

MLA 的低秩不是只为减小 checkpoint 参数，而是为这条推理数据流设计。

## RoPE 为什么破坏普通矩阵吸收

Rotary Position Embedding 对 Query/Key 应用随 token position 变化的旋转矩阵 $R_t$。若直接对 content Key 使用 RoPE：

$$
k_{j,i}=R_jW_i^{UK}c_j^{KV}
$$

score 变成：

$$
q_{t,i}^TR_t^TR_jW_i^{UK}c_j^{KV}
$$

$R_j$ 随历史位置 $j$ 改变，不能把 $W_i^{UK}$ 作为一个与位置无关的常量完全吸收到 Query 侧。若强行吸收，就会丢失正确的相对位置信息；若为每个 $j$ 重新组合矩阵，则不再是一次 Query 投影。

这不是代码实现上的小障碍，而是“线性权重固定、位置变换随 token 改变”造成的代数边界。

## Decoupled RoPE 把 Content 与 Position 分开

MLA 的解决方式是只让一小段专门的 Query/Key 维度承载 RoPE，把可吸收的 content path 保持为无位置旋转的线性映射。

可以写成：

$$
q_{t,i}=[q_{t,i}^C;q_{t,i}^R]
$$

$$
k_{j,i}=[k_{j,i}^C;k_j^R]
$$

其中：

$$
q_{t,i}^R=\operatorname{RoPE}_t(W_i^{QR}c_t^Q)
$$

$$
k_j^R=\operatorname{RoPE}_j(W^{KR}h_j)
$$

位置 Key $k_j^R$ 可以在 heads 间共享，而 content Key $k_{j,i}^C$ 仍由 joint latent 解释。拼接向量的 dot product自然分解为：

$$
q_{t,i}^Tk_{j,i}
=
(q_{t,i}^C)^Tk_{j,i}^C
+
(q_{t,i}^R)^Tk_j^R
$$

第一项使用前述矩阵吸收，在 latent space 计算；第二项保留 RoPE 的相对位置语义，直接与缓存的小维度 positional Key 计算。

所以 MLA 每 token 的核心缓存不是只有 $c_j^{KV}$，而是：

$$
\operatorname{cache}_j=[c_j^{KV};k_j^R]
$$

省略 $k_j^R$ 会让位置语义不完整；把 $k_j^R$ 误认为传统全维 Key，又会高估缓存大小。

## 每 Token 到底缓存多少元素

忽略 scale、对齐与 page padding，MLA 每 token、每层缓存的元素数为：

$$
N_{MLA}=d_c+d_R
$$

其中 $d_c$ 是 joint KV latent dimension，$d_R$ 是 decoupled RoPE Key dimension。与 MHA/GQA 对比：

| Attention | 每 token、每层 KV 元素数 |
| --- | ---: |
| MHA | $2n_hd_h$ |
| GQA | $2n_{kv}d_h$ |
| MQA | $2d_h$ |
| MLA | $d_c+d_R$ |

DeepSeek-V2 论文把其 MLA 缓存量描述为约等于 2.25 个 GQA groups，这是模型具体维度下的比较。官方同时报告相对 DeepSeek 67B 的整体 KV Cache 减少 93.3%；该数字还受两代模型配置影响，不应当成所有 MLA 相对任意 MHA 的固定压缩比。

部署自己的模型时，应直接从 checkpoint config 读取 $d_c$、$d_R$、层数和 dtype，再加上：

- page/block padding；
- cache scale 与量化 metadata；
- block table；
- speculative decoding 的额外 tokens；
- prefix cache 的引用与淘汰开销；
- tensor/context parallel 的复制或分片。

## MHA Mode 与 MQA Mode 指的是执行形态

MLA 在数学上可以用不同方式执行。工程文档中的 `MHA mode` 与 `MQA mode` 往往描述 kernel 如何组织 operands，不表示 checkpoint 临时变成了传统 MHA 或 MQA。

### MHA Mode

显式或局部展开 per-head K/V，再调用接近标准 multi-head attention 的 kernel。Prefill 中 Query 数量大，矩阵化展开与高吞吐 attention kernel 可能更合适。

### MQA Mode

使用吸收后的 Query，直接把共享 latent cache 视为一个 KV head；不同 Query heads 对同一 latent history 计算。Decode 中历史长而 Query 很短，这能避免反复展开完整 K/V。

FlashMLA 当前官方支持矩阵也明确区分 dense decoding 的 MQA mode，以及不同硬件上的 dense/sparse prefill 形态。具体支持会随版本变化，部署时应查目标版本而不是从概念推导。

一个 runtime 可以在 Prefill 使用 MHA mode、Decode 使用 MQA mode，只要两条路径与同一 MLA 数学语义一致。

## Prefill 为什么未必使用与 Decode 相同的吸收方式

Prefill 一次处理许多 Query tokens，计算量大，通常更接近 compute-bound。显式生成某些 K/V tiles 的成本可以被大矩阵乘摊薄，并能复用成熟的 FlashAttention 风格 kernel。

Decode 每序列通常只有一个 Query，却要读取很长历史。若展开历史多头 K/V：

```text
small current query
  → read latent history
  → expand every historical position
  → run attention
```

扩展工作与历史长度一起增长，极不划算。吸收后只转换当前 Query，并读取紧凑 latent cache，更符合 Decode 的 memory-bound 特性。

因此 MLA kernel selection 不应只有一个静态开关。需要至少考虑：

- Query length；
- KV length；
- batch 中变长序列分布；
- Query head 数量；
- latent 与 RoPE 维度；
- speculative decoding 一次验证的 token 数；
- GPU architecture 与可用 kernel。

## FlashMLA 解决的不是模型公式本身

MLA 定义“算什么”，FlashMLA 一类 kernel 库负责“怎样在具体 GPU 上高效算”。Dense Decode 的输入通常包括：

```text
absorbed query
paged latent/RoPE KV cache
cache sequence lengths
block table
tile scheduler metadata
```

变长 batch 中，每条序列的 KV 长度不同。若静态地给每个 request 分配相同 tile 数，短序列浪费计算，长序列又可能成为 straggler。FlashMLA 的接口会预先生成 tile scheduler metadata 和 split 信息，再让 kernel 对 cache pages 做调度。

它仍需要 FlashAttention 的 online Softmax 思路：不同 KV tiles 分别产生局部最大值、归一化和与输出，最后稳定合并。Paged cache、split-KV 和 latent head dimensions 会影响 tile 设计，但不会改变 attention 的精确语义。

安装 FlashMLA 并不会把任意 MHA 模型转换成 MLA。模型必须具有对应 projection weights、decoupled RoPE 和训练语义；kernel 只消费这类 checkpoint 已经定义的计算。

## Paged KV Cache 的 Block 里放什么

对 MLA，page 不再保存传统 `[K heads, V heads, head_dim]`，而是保存每 token 的 joint latent 与 RoPE Key。逻辑上可以表示为：

```text
page
  token 0: [c_KV | k_R]
  token 1: [c_KV | k_R]
  ...
```

具体 layout 可能交错、分区、对齐或量化。Runtime 必须让以下组件对同一 layout 达成一致：

- cache writer；
- block allocator / block table；
- prefix cache hashing；
- attention kernel；
- KV transfer connector；
- KV quantization/dequantization；
- checkpoint restore 或 fault recovery。

若 P/D 分离，传输的是 MLA latent cache 与 RoPE segment，而不是 full MHA K/V。NIXL/Mooncake 一类数据面只负责移动 bytes，并不会自动理解模型的 cache schema；schema、dtype、layer/shard ownership 与 checksum 必须由 runtime contract 提供。

## Tensor Parallel 下矩阵应该怎样放

传统 MHA/GQA 常按 heads 对 Q/K/V 投影与 cache 分片。MLA 的 latent 被多个 Query heads 共享，分片策略多了一层选择：

- 每个 TP rank 复制完整 $c^{KV}$，本地持有一部分 Query heads；
- 按 latent dimension 分片，在 score/value 聚合中增加 collective；
- 在某些阶段使用不同的并行方式；
- Decode 使用 data parallel，避免短 Query 上的 TP collective。

复制 latent 会增加总集群 KV bytes，却可能避免每一步跨 rank 交换 latent；分片节省副本，但会引入 reduction 或 gather。最优选择取决于 latent 大小、TP size、互联和 batch。

还要区分 weight sharding 与 cache sharding。`W^{UK}` 被吸收到 Query 路径后，导出的 absorbed weights 必须与 Query-head ownership 一致；`W^{UV}` 与 output projection 的组合也要遵守相同 rank 布局。只把原 checkpoint 按传统 MHA 轴切分，可能在数值上或通信上都不成立。

## MLA 与 KV Cache 量化可以叠加

MLA 减少每 token 保存的元素数，KV quantization 减少每个元素的 bytes：

$$
M_{cache}
\propto
(d_c+d_R)\times\text{bytes per element}
$$

两者是正交维度，可以组合，但 latent 和 RoPE segment 的分布可能不同，未必适合共用一个 scale。量化时至少要明确：

- $c^{KV}$ 与 $k^R$ 是否分别量化；
- scale 是 per-token、per-block 还是更粗；
- cache write 时何时计算 scale；
- attention kernel 是否直接消费低精度 cache；
- dequantization 是否融合进 load/main loop；
- 长上下文检索和位置敏感任务是否退化。

FlashMLA 的不同 dense/sparse kernel 对 BF16/FP8 cache 的支持并不完全相同。不能因为模型是 MLA 就默认 FP8 KV 一定可用，也不能把某个 sparse kernel 的格式用于 dense kernel。

## MLA 改变了容量规划中的哪些项

使用传统 KV 公式估算 MLA 模型会严重高估缓存；只用 `d_c` 又会漏掉 RoPE segment 和 metadata。容量模型应按实际 cache schema 计算：

$$
M_{request}
=
T\cdot L\cdot
\left(
d_c b_c+d_R b_R+m_{scale}+m_{padding}
\right)
$$

然后再加入 runtime 固定成本：

```text
weights
+ MLA cache pages
+ block tables
+ attention workspaces
+ graph pools
+ MoE dispatch buffers
+ communication buffers
+ fragmentation reserve
```

MLA 让单请求 KV 更小，但 DeepSeek 类 MoE 模型可能有很大的总权重、expert buffers 和通信开销。缓存省下的显存是否能全部转化为并发，要看其他部分是否成为新的容量上限。

## 不能直接把 MHA Checkpoint 改成 MLA

MLA 的 down/up projections、joint latent rank、decoupled RoPE 和多头表达都是训练得到的模型架构。将已有 MHA 的 K/V 权重做一次低秩分解，通常只能得到近似初始化：

- 截断 SVD 会丢失信息；
- K 与 V 需要联合考虑，而不是分别压缩；
- 原模型的全维 RoPE 与 decoupled RoPE 不同；
- output projection 与 absorbed Value path 需要重新适配；
- 不同 layers 的可压缩程度不同。

可以研究转换与继续训练方法，但这属于 model surgery 与训练任务，不是 serving runtime 的无损优化开关。推理引擎只能忠实执行 checkpoint 已定义的 MLA。

## 正确性验证应该从两种实现互证

MLA 很适合建立一个清晰但较慢的 reference：显式还原 content K/V，拼接 RoPE segment，再用普通 attention 计算。优化实现则使用 absorbed MQA mode 与 paged latent cache。

验证链可以分为：

### 1. Projection

比较 joint latent、显式 $k^C/v^C$、Query content 与 RoPE segments，确认维度、转置和 head reshape。

### 2. Absorption

对同一组 Q/latent 验证：

$$
(q^C)^TW^{UK}c
\approx
((W^{UK})^Tq^C)^Tc
$$

并比较显式 Value 聚合与延后 up-projection 的结果。

### 3. Position

覆盖不同绝对位置、长上下文、prefill/decode 边界和 RoPE scaling。交换 interleaved/non-interleaved layout 可能让代码正常运行但位置语义错误。

### 4. Cache

逐 token 比较 contiguous reference cache 与 paged cache，覆盖 block 边界、prefix reuse、eviction、fork 和 P/D transfer。

### 5. Kernel

比较 MHA mode、absorbed MQA mode 和 FlashMLA 等目标 kernel 的 outputs 与 log-sum-exp，覆盖变长 batch、不同 Query/KV lengths 和 speculative tokens。

### 6. End to End

使用 greedy decoding 比较 token/logits，再覆盖长上下文检索、代码、数学、多语言与结构化输出。若加入 KV quantization，要单独测它的增量误差。

## 性能验证要防止“缓存小但算得更多”

MLA artifact 加载成功并不代表执行高效。Profiler 中至少要回答：

1. Decode 是否直接读取 latent cache，还是 materialize full K/V？
2. Query/Value absorption 是否在 engine build 时完成，还是每步重复转换 weights？
3. Prefill 和 Decode 分别选择了哪种 MLA mode？
4. Paged cache load 是否合并，RoPE/latent layout 是否导致额外 transpose？
5. 变长 batch 的 tile scheduler 是否均衡？
6. Split-KV 合并、online Softmax 与 output projection 是否融合合理？
7. TP collective 或 latent replication 是否抵消缓存收益？
8. 低精度 cache 的 scale/dequant 是否进入关键路径？

Serving 指标则要同时记录：

- KV bytes/token 与最大 resident tokens；
- TTFT、TPOT 和 tokens/s；
- 相同 SLO 下的 goodput；
- Prefill/Decode 各阶段 attention latency；
- prefix cache hit 后的复用成本；
- P/D KV transfer bytes 与时间；
- 实际 GPU 显存，而不仅是理论 tensor 大小。

官方论文的压缩比和吞吐倍数来自特定模型与系统对比。自己的部署应把 MHA/GQA/MLA 模型质量差异与 runtime 效率分开，不能用架构变化后的端到端结果替代 kernel 消融。

## 一条可执行的 Runtime 集成路径

1. **解析 checkpoint contract**：读取 $d_c$、$d_R$、Query heads、projection 与 RoPE 配置；
2. **实现显式 reference**：先还原 K/V，验证模型语义与已有 logits；
3. **实现 latent cache schema**：定义 page layout、dtype、scale、padding 与 block table；
4. **离线构造 absorbed weights**：按目标 TP/DP placement 生成并版本化；
5. **实现 Decode MQA mode**：直接对 latent/RoPE cache 做精确 attention；
6. **实现 Prefill 路径**：按 shape 选择 MHA mode 或匹配的优化实现；
7. **逐层互证**：显式、absorbed、paged 与目标 kernel 四条路径对齐；
8. **接入 scheduler**：支持变长 batch、split-KV、speculative tokens 与 graph capture；
9. **接入 prefix/P-D 数据面**：让缓存复用、传输与恢复理解 MLA schema；
10. **再加入 KV 低精度**：一次只改变一个 segment 或 dtype；
11. **建立 shape benchmark**：按真实流量分布比较 kernel selection；
12. **发布监控与回退**：记录 backend coverage、cache bytes、质量和 SLO。

## 常见误区

### “MLA 就是把 K 和 V 各自做低秩分解”

MLA 使用联合 KV latent；K/V 共享被缓存的低维表示，再由不同 up-projections 解释。

### “每一步仍要恢复所有历史 K/V”

优化推理通过 Query-side 和 output-side 矩阵吸收直接在 latent 上计算。完整恢复只是易懂的 reference，不是理想 Decode 路径。

### “MLA Cache 里只有一个 latent”

还要保存 decoupled RoPE Key segment，位置相关部分不能被普通静态权重吸收。

### “MLA 等价于 MQA”

MLA 的某种执行形式称为 MQA mode，但模型仍通过多头 projections 从 joint latent 获得多头表达；它不是传统单 KV head 架构的同义词。

### “MLA 是推理引擎可以给任意模型打开的选项”

它改变 checkpoint 架构与训练语义。Kernel 只能加速已有 MLA 模型，不能无损改写普通 MHA/GQA 模型。

### “缓存压缩 93.3% 是固定公式”

这是 DeepSeek-V2 官方相对其指定基线的模型级报告。通用压缩比必须用双方的 heads、dimensions、dtype 和 metadata 重新计算。

## 小结

MLA 的核心不是“低秩”三个字，而是一条围绕自回归 Decode 设计的代数与系统闭环：

1. K/V 先联合压缩为每 token 一份 $c^{KV}$；
2. content Key projection 吸收到当前 Query 侧，历史 Key 无需展开；
3. Value projection移到 attention 聚合之后，并可与 output projection 组合；
4. RoPE 因位置相关无法普通吸收，所以拆成小维度 $k^R$ 单独缓存；
5. 实际 cache 是 $[c^{KV};k^R]$，而不是完整多头 K/V；
6. Prefill 可以偏向 MHA mode，Decode 更适合 absorbed MQA mode；
7. Paged layout、tile scheduling、TP placement 和低精度决定理论压缩能否转化为性能；
8. MLA 是模型架构，不能由 runtime 给任意 checkpoint 无损开启。

MLA 先减少“每个历史 token 存多少、读多少”。下一篇 FlashAttention-3 会回到单次 dense attention kernel：在 Hopper 上用 TMA、WGMMA 和异步 warp pipeline 提高这些数据进入 Tensor Core 后的执行效率；再往后，DeepSeek Sparse Attention 会继续减少“究竟访问多少历史 token”。

## 参考资料

- [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434)
- [DeepSeek-V2 Official Repository](https://github.com/deepseek-ai/DeepSeek-V2)
- [FlashMLA: Efficient Multi-head Latent Attention Kernels](https://github.com/deepseek-ai/FlashMLA)
- [FlashMLA New Kernel Deep Dive](https://github.com/deepseek-ai/FlashMLA/blob/main/docs/20250422-new-kernel-deep-dive.md)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- [DeepSeek-V3.2 Technical Report: MLA Execution Modes and Sparse Attention](https://arxiv.org/abs/2512.02556)
