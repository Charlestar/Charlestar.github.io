---
layout: post
title: "KV Cache：自回归推理为什么必须保存历史状态"
subtitle: "从重复计算、每 Token 显存到 MHA/GQA/MQA 与缓存生命周期"
date: 2026-06-26 09:00:00 +0800
last_modified_at: 2026-09-03
author: iStar
catalog: true
series: kv-cache-memory
series_order: 10
technology_year: 2017
mathjax: true
tags: [KV Cache, LLM推理]
---

大语言模型生成一个 token 后，会把它追加到已有序列，再运行一次模型生成下一个 token。如果每一轮都重新计算整个前缀，已经处理过的 token 会被反复投影成 Key 和 Value，解码长度越长，浪费越严重。

KV Cache 保存每一层历史 token 的 Key/Value。下一轮只为新增 token 计算一组 Q/K/V，再让新 Query 读取缓存中的所有历史 K/V：

```text
without cache:
  step 1 -> compute token 1..L
  step 2 -> compute token 1..L+1 again
  step 3 -> compute token 1..L+2 again

with KV cache:
  prefill -> compute and store K/V for token 1..L
  step 1  -> compute new token, append one K/V
  step 2  -> compute new token, append one K/V
```

它把大量重复算术换成持续增长的显存状态。推理因此从“不断重算前缀”变成“每轮读取越来越长的缓存”。这项转换决定了现代 LLM Serving 的许多特征：decode 常受显存带宽限制，并发上限常由 KV Cache 而不是模型权重决定，调度器必须跟踪每个请求的 block，prefix caching 和 P/D 分离还要移动或共享这些状态。

本文从一层 attention 的公式开始，推导 KV Cache 为什么正确、每个 token 占多少显存、MHA/GQA/MQA 如何改变容量，再沿请求的完整生命周期讨论位置、batch 重排、分页、复用、量化和回收。

## 自回归生成究竟重复了什么

设 prompt token 为：

$$
x_1,x_2,\ldots,x_L
$$

模型第一次前向得到下一个 token $x_{L+1}$。为了生成 $x_{L+2}$，输入变成：

$$
x_1,x_2,\ldots,x_L,x_{L+1}
$$

在 decoder-only Transformer 的第 $\ell$ 层，hidden states 经过投影：

$$
Q^{(\ell)}=X^{(\ell)}W_Q^{(\ell)}
$$

$$
K^{(\ell)}=X^{(\ell)}W_K^{(\ell)},\qquad
V^{(\ell)}=X^{(\ell)}W_V^{(\ell)}
$$

由于 causal mask，历史位置 $1\ldots L$ 的 hidden state 不会依赖未来新增 token。只要模型权重和前缀内容不变，它们在下一轮得到的 K/V 也不变。

真正新增的只有位置 $L+1$ 对应的：

$$
k_{L+1}^{(\ell)},v_{L+1}^{(\ell)}
$$

因此没有必要重新计算：

$$
K_{1:L}^{(\ell)},V_{1:L}^{(\ell)}
$$

这就是可以缓存的依据。Query 不同：普通增量 decode 只需要新位置的 $q_{L+1}$ 来预测下一个 token，历史 Query 不会在未来步骤再次使用，所以通常不保存 Q Cache。

“KV Cache”这个名字不是随意省略 Q；它反映了自回归 attention 的真实复用方向。

## 一轮 Decode 如何使用缓存

对单个 attention head，当前新 token 的 query 为：

$$
q_t\in\mathbb{R}^{1\times d}
$$

历史加当前的缓存为：

$$
K_{1:t}\in\mathbb{R}^{t\times d},\qquad
V_{1:t}\in\mathbb{R}^{t\times d}
$$

本轮 attention：

$$
s_t=\frac{q_tK_{1:t}^T}{\sqrt d}
$$

$$
p_t=\operatorname{softmax}(s_t)
$$

$$
o_t=p_tV_{1:t}
$$

执行前后缓存变化是：

```text
before: K[1:t-1], V[1:t-1]
append: k[t], v[t]
read  : K[1:t],   V[1:t]
after : same cache becomes input state for next step
```

每个 Transformer layer 都有自己的 K/V，因为每层 hidden state 和投影权重不同。某一层的 cache 不能直接被另一层使用。

对于 $L_{layers}$ 层模型，一个请求实际上维护的是：

```text
layer 0: K/V for all cached positions
layer 1: K/V for all cached positions
...
layer L-1: K/V for all cached positions
```

这也是为什么序列只有几万个 token，KV Cache 总量却可能达到数 GB。

## Prefill 如何一次建立 Cache

Prompt 阶段不需要逐 token 串行执行。Causal attention 允许所有 prompt positions 的 Q/K/V 并行计算，只在 attention mask 中阻止未来信息。

Prefill 对每层执行：

1. 为全部 prompt tokens 计算 Q/K/V；
2. 运行 causal attention；
3. 将每个位置的 K/V 写入 cache；
4. 将输出送入下一层；
5. 最后用末位置 logits 采样第一个输出 token。

所以 prefill 同时产生两类结果：

- 用于采样第一个生成 token 的 logits；
- 面向后续 decode 的完整层级状态。

Prefill 计算量大、token 并行度高，通常偏计算密集；decode 每轮只增加一个 token，却反复扫描长 KV，通常偏显存带宽受限。KV Cache 没有让 attention 对历史的读取消失，它消除的是历史 K/V projection 和前面层输出的重复计算。

## 不使用 Cache 时，复杂度怎样增长

假设 prompt 长度 $L$，需要生成 $T$ 个 token。把产生第一个生成 token 的 prompt forward 记为 $t=0$，此时输入长度为 $L$；产生后续 token 时，输入长度依次为 $L+1,\ldots,L+T-1$。

仅从线性投影和 MLP 的 token 工作量看，总处理 token 数约为：

$$
\sum_{t=0}^{T-1}(L+t)
=TL+\frac{T(T-1)}{2}
$$

有 cache 后：

$$
L+T-1
$$

个 token 的 projection/MLP 只需各计算一次：prompt 的 $L$ 个 token 在 prefill 计算，前 $T-1$ 个已生成 token 在后续 decode 中各计算一次；最后返回的 token 不必再送入模型才能完成这 $T$ 个 token 的生成。

Dense attention 的历史读取仍随当前上下文长度增长，但从“每轮为所有 query rows 重算完整 causal matrix”变成“每轮只为新 query row 读取历史 K/V”。按常用的平方 attention 工作量计数（省略 causal 三角带来的常数因子），无缓存路径为：

$$
\sum_{t=0}^{T-1}(L+t)^2
=TL^2+LT(T-1)+\frac{T(T-1)(2T-1)}{6}
$$

缓存路径包含一次 prompt prefill，以及后续 $T-1$ 个单 query row：

$$
L^2+\sum_{t=1}^{T-1}(L+t)
=L^2+(T-1)L+\frac{T(T-1)}{2}
$$

因而更准确的端到端量级比较是：

```text
recompute:  O(TL² + LT² + T³)
cached:     O(L² + TL + T²)
```

只有在 $L$ 与 $T$ 都随总序列长度同比增长时，前者才可简写成累计三次量级、后者连同 prefill 简写成二次量级；当 $T\ll L$ 时，保留 $L,T$ 的展开式更有解释力。这里不包含不同 kernel、prefill 并行和 batch。KV Cache 让增量生成从不可接受的重复前缀计算，降为每 token 仍随上下文线性增长的 attention read。

## 每个 Token 的 KV Cache 占多少显存

设：

- 层数为 $L_{layers}$；
- KV head 数为 $H_{kv}$；
- 每个 head 维度为 $d$；
- cache dtype 每元素字节数为 $b$；
- token 数为 $N$。

每层每 token 有 K 和 V 两份，因此：

$$
Bytes_{token}
=2\cdot L_{layers}\cdot H_{kv}\cdot d\cdot b
$$

总 KV Cache：

$$
Bytes_{KV}
=N\cdot2L_{layers}H_{kv}db
$$

如果 batch 内有多个请求 $r$，长度各为 $N_r$：

$$
Bytes_{batch}
=\sum_r N_r\cdot2L_{layers}H_{kv}db
$$

这还没有加入 block 对齐浪费、allocator metadata、scale、临时 workspace 和 prefix sharing。它是容量规划的理论基线。

## 用一个 32 层模型算一遍

假设：

```text
layers       = 32
query heads  = 32
KV heads     = 8
head dim     = 128
cache dtype  = BF16 = 2 bytes
```

每 token：

$$
2\times32\times8\times128\times2
=131072\text{ bytes}
=128\text{ KiB}
$$

一条 8K-token 序列约为：

$$
8192\times128\text{ KiB}=1\text{ GiB}
$$

32 条平均 8K 的并发请求，理论 KV 就约 32 GiB。模型权重、CUDA Graph pools 和临时激活尚未计入。

如果同模型使用 32 个 KV heads 的 MHA，每 token 变为 512 KiB，8K 序列约 4 GiB。GQA 从 32 个 KV heads 降到 8 个，容量恰好降到四分之一。

这个例子说明“70B 模型权重能放入 GPU”并不代表服务能承载目标并发。权重是静态项，KV Cache 随请求数和上下文持续增长。

## MHA、GQA、MQA 如何改变 Cache

设 Query head 数为 $H_q$。

### Multi-Head Attention

MHA 通常：

$$
H_{kv}=H_q
$$

每个 Q head 拥有独立 K/V head，表达能力强，cache 和 decode 带宽也最大。

### Multi-Query Attention

MQA 让所有 Q heads 共享一组 K/V：

$$
H_{kv}=1
$$

2019 年的 MQA 工作强调，增量 decode 的主要成本之一是反复读取大型 K/V tensors；共享 K/V 能显著减少 cache 和显存带宽。

### Grouped-Query Attention

GQA 取中间值：

$$
1<H_{kv}<H_q
$$

每组 Query heads 共享一个 KV head。例如 $H_q=32,H_{kv}=8$，每 4 个 Q heads 共用 K/V。

容量相对 MHA 的比例为：

$$
\frac{Bytes_{GQA}}{Bytes_{MHA}}
=\frac{H_{kv}}{H_q}
$$

GQA 的目标是在 MHA 质量和 MQA 推理效率之间折中。博客已有一篇文章专门讨论其训练与模型质量；这里最需要记住的是：**KV head 数直接进入每 token 显存和每轮读取字节数。**

## Decode 为什么常受显存带宽限制

普通 decode 每个请求每轮只有一个新 query。线性层可以把多个请求 batch 成矩阵乘，但 attention 必须为每条请求读取其历史 K/V。

对当前总缓存 $M_{KV,read}$，若有效显存带宽为 $BW$，只看读取下界：

$$
T_{attention}\gtrsim\frac{M_{KV,read}}{BW}
$$

随着 context 变长，FLOPs 和 bytes 都增长，但单 query 的矩阵形状较窄，Tensor Core 不容易像 prefill 那样充分利用。读取 K/V、解量化、地址映射和 reduction 会成为主导。

这解释了几项常见优化为什么直接瞄准 KV：

- MQA/GQA 减少 KV heads；
- FP8/FP4 KV 量化减少每元素 bytes；
- sliding-window attention 只保留/读取有限历史；
- sparse attention 只选择部分 blocks；
- prefix caching 避免重复写入相同前缀；
- P/D 分离让 cache 在不同阶段和设备间转移；
- paged allocation 提高可用于 cache 的实际显存比例。

模型权重量化能释放容量，却不会直接减少每轮 KV read；KV 量化则同时影响容量和 decode 带宽。

## Cache 的 Tensor Layout 长什么样

概念上，一层 cache 可以写成：

```text
K: [request, kv_head, sequence, head_dim]
V: [request, kv_head, sequence, head_dim]
```

实际 kernel 可能采用：

- sequence/head 维交换；
- vectorized head_dim packing；
- K 与 V 分离或交错；
- blocks/pages 在物理内存中不连续；
- 每 block/layer 独立 scale；
- TP rank 只保存本地 KV heads；
- MLA 保存压缩 latent 与位置相关分量。

Layout 的选择服务于读写路径。Prefill 偏向连续写入大量 token，decode 偏向为一个 query 读取很多历史 blocks；两者的最佳 tile 与访问顺序不完全相同。

因此从一个引擎导出的 KV Cache 往往不能直接交给另一个引擎。即使模型相同，还要匹配：

- dtype 与 scale granularity；
- layer/head sharding；
- RoPE 是否已经应用；
- block size 和物理排列；
- token position 与有效长度；
- attention backend 的 vectorization。

KV transfer connector 的重要工作正是定义或转换这些布局，而不只是“把一段 GPU memory 复制过去”。

## RoPE 在写 Cache 前还是读 Cache 时应用

Rotary Position Embedding 通常作用于 Q/K，而不作用于 V。常见路径是：

```text
project q, k, v
-> apply RoPE to q and k for absolute position t
-> write rotated k and raw v into cache
```

下一轮直接读取已经按原位置旋转的 K。这样避免每轮对整个历史 K 重做 RoPE。

但某些 backend 或模型变体可能缓存未旋转 K，在 attention kernel 内根据 position 处理。无论选择哪种，cache identity 必须包含相关语义：同一 token IDs 在不同 RoPE scaling、position offset 或模型配置下，不一定可以共享同一 K。

Chunked prefill、prefix reuse 和 P/D transfer 都要保持全局 position 连续。若接收端把 transferred cache 当作从 position 0 开始，数值不会因 shape 错误而立刻崩溃，却会产生错误 attention。

## Cache Position 为什么不能等同于数组下标

最简单单请求 cache 可以用 `cache_position=t` 写入第 $t$ 行。但在线引擎中：

- batch 中请求长度不同；
- 已完成请求会离开，新请求进入同一 batch row；
- paged blocks 物理地址不连续；
- sliding window 会循环覆盖旧位置；
- prefix cache 让多个请求共享 blocks；
- beam/parallel sampling 会产生共享后再分叉。

稳定身份应是：

```text
request/sequence ID
logical token position
layer/KV group
physical block ID + block offset
```

Batch row 只是本轮 kernel 中的临时位置。若 cache 直接以当前 batch index 作为长期 owner，Continuous Batching 一重排就可能让请求读到别人的历史。

PagedAttention 的 block table 正是把逻辑位置映射到物理 blocks；本系列下一篇会专门解释这层虚拟内存式映射。

## 为什么预分配最大长度会浪费显存

一种连续布局是为每个请求预留：

$$
[L_{layers},2,H_{kv},N_{max},d]
$$

如果 `max_model_len=128K`，一个只生成到 2K 的请求也占用 128K 容量。在线请求的输出长度又无法提前知道，按最大长度预留会产生巨大内部浪费。

另一种做法按当前长度连续扩容，类似动态数组。但 GPU 上许多不同大小、不同生命周期的请求会造成外部碎片；扩容还可能需要移动已存在的 cache，破坏稳定地址和 kernel 并行。

分页将 cache 切成固定 token 数的 blocks：

```text
logical tokens 0..15   -> physical block 42
logical tokens 16..31  -> physical block 7
logical tokens 32..47  -> physical block 105
```

请求按需领取 blocks，完成后归还 pool。最后一个 block 仍可能有少量内部浪费，但不再为整条最大序列预留连续空间。

KV Cache 是状态本身；PagedAttention 是管理状态物理空间的方法，二者不能互换概念。

## Block Size 是容量与 Kernel 的共同参数

若每 block 容纳 $B_t$ 个 token，请求长度 $N$ 需要：

$$
N_{blocks}=\left\lceil\frac{N}{B_t}\right\rceil
$$

尾块内部浪费最多：

$$
B_t-1
$$

个 token slot。Block 越小，平均尾部浪费越少，prefix reuse/eviction 粒度也更细；但 block table 更长，allocator metadata 更多，attention kernel 要处理更多地址片段。

Block 越大，寻址和 metadata 更简单、连续访问更长，却可能增加尾部浪费，并让一个很短 prefix 也占完整大块。

因此 block size 不是只由内存管理器决定。Attention kernel 的 page size 支持、vectorization、TP layout 和量化 scale granularity都会约束它。

生产参数必须用目标 backend 验证，不能看到“更小碎片更少”就无限缩小。

## Prefix Caching 与普通 KV Cache 的区别

普通 KV Cache 是**同一个请求跨 decode steps**复用自身历史。Prefix caching 是**不同请求或不同时刻的请求**复用相同前缀已经计算好的 K/V。

```text
request A: [system prompt][user A]
request B: [system prompt][user B]
                 ^
          shared prefix blocks
```

要安全共享，至少需要确认：

- token IDs 完全一致；
- 模型权重/revision 相同；
- adapter/LoRA 身份相同；
- position 与 RoPE 配置相同；
- 多模态输入内容/hash 相同；
- cache dtype/scale/layout 相同；
- 影响 hidden state 的其他条件相同。

引擎通常以 block token 内容、父 block hash 和额外 identity 生成 cache key。只按原始文本字符串比较不够，因为 tokenizer/chat template 可能不同；只按 token IDs 比较在带 adapter 或多模态条件时也不够。

Prefix reuse 减少 prefill 计算和 TTFT，不减少后续 decode 对这些历史 K/V 的读取量。RadixAttention 会把不同长度的共享 prefix 组织成 radix tree，本系列已有文章单独展开这套数据结构。

## Copy-on-Write 为什么会出现在 KV Cache 中

Beam search、parallel sampling 或共享 prefix 的请求开始时可以引用同一组 blocks：

```text
prefix blocks: [A][B][C]
branch 1 -> shares A/B/C
branch 2 -> shares A/B/C
```

当两个分支追加不同 token，完整 blocks 仍可共享，但可写尾块可能需要复制：

```text
branch 1: [A][B][C1]
branch 2: [A][B][C2]
```

如果直接让两个请求写同一物理尾块，会互相覆盖。Copy-on-write 在首次分叉写入时复制必要 block，并更新各自 block table；引用计数保证共享 blocks 在所有 owner 完成前不被回收。

这引入类似文件系统/虚拟内存的并发问题：refcount、hash、block table 和 GPU 写入必须有一致提交边界。一个迟到的 kernel 不能在 block 已重新分配给其他请求后继续写入。

## 请求取消时必须释放哪些状态

客户端取消不是只停止输出。请求可能持有：

- 私有 KV blocks；
- 共享 prefix blocks 的引用；
- 正在分配但尚未提交的 blocks；
- offload/transfer 中的 blocks；
- CUDA Graph input slots；
- encoder cache 或 speculative state。

安全生命周期近似：

```text
ACTIVE
-> cancellation requested
-> stop future scheduling
-> wait/mark current GPU step result as discardable
-> release private blocks
-> decrement shared refs
-> cancel or drain transfers
-> FINISHED
```

正在执行的 GPU kernel 通常无法移除单个请求，本轮写入可能仍会发生。Allocator 不能在确认相关 work 完成前立刻把同一 block 给新请求，否则会发生 use-after-free。

压力测试应在 prefill 中间、decode 中间、block 边界和 transfer 中分别取消，检查 cache usage 最终回落。

## KV Cache 满了以后有哪些选择

显存 pool 用尽时，调度器通常在以下策略间选择。

### 等待

不接纳新请求，等 running 请求完成释放 blocks。最安全，但 TTFT 上升。

### Preemption + Recompute

回收某个请求的 cache，将其移回 waiting；恢复时重新计算被丢弃前缀。省去 CPU transfer，却重复 GPU 算术。

### Swap/Offload

把 KV 移到 host memory、其他 GPU、远端内存或存储层，恢复时传回。避免重算，但受 PCIe/网络带宽和延迟影响。

### Evict Reusable Prefix

对已经没有 active request 引用、仅为未来命中保留的 prefix blocks 做 eviction。不会影响当前请求正确性，只降低未来 hit rate。

### 限制窗口或压缩

模型支持 sliding window 时可淘汰窗口外状态；量化可让相同 pool 容纳更多 token。若模型本身需要全局 dense attention，随意删除历史 K/V 会改变输出。

选择应考虑 block 的 recompute cost、复用概率、传输成本、请求 SLO 和优先级，而不只是最近最少使用。

## KV Cache 量化省下了什么，又付出什么

若从 BF16 2 bytes 降到 FP8 1 byte，理论 cache 容量减半；降到 4 bit，payload 理论再减半。

但真实大小还包含 scales 和对齐：

$$
Bytes_{quantized}
=N_{elements}\cdot\frac{bits}{8}
+N_{scales}\cdot bytes_{scale}
+padding
$$

量化收益有两类：

1. 相同显存容纳更多 tokens/requests；
2. decode attention 从 HBM 读取更少 bytes。

成本包括：

- 写入时 quantize 与 scale 统计；
- 读取时 dequantize 或低精度 kernel；
- 数值误差随层数和上下文传播；
- per-tensor/per-head/per-block scale 的精度与 metadata 取舍；
- kernel/backend 对 dtype 和 page layout 的限制。

如果 attention kernel 先把 4-bit cache 全量解压到大临时 buffer，再计算，显存峰值和带宽收益可能下降。真正有效的路径应在读取 tile 时融合解量化，或直接使用低精度计算。

NVFP4 文章会继续讨论 Blackwell 上的 4-bit 格式和硬件路径；这里先把它放在 KV 容量公式中的 `b` 项理解。

## Sliding Window 为什么可以限制 Cache 长度

若模型每层只允许当前 token 关注最近 $W$ 个位置，那么更旧 K/V 对该层未来输出不再可见，可以使用 cyclic buffer：

$$
slot=t\bmod W
$$

每层最大缓存从 $N$ 限制为 $W$。但现代模型可能混合：

- 部分层 full attention；
- 部分层 sliding window；
- 不同层使用不同 window；
- attention sink/global tokens 永久保留。

Cache manager 因此可能需要多个 pools，按 KV heads、head dim、window 和 dtype 分组。不能因为某几层 local attention 就全局删除旧 blocks；full-attention layers 仍需要它们。

此外，训练时定义的 mask 决定哪些状态可安全淘汰。服务端自行把 full attention 改成窗口 attention 属于模型近似，不是无损内存优化。

## Tensor Parallel 下 Cache 如何分片

在 head-wise Tensor Parallel 中，Q/K/V heads 分给不同 ranks。若 $H_{kv}$ 能被 TP size 整除，每张卡只保存本地 KV heads：

$$
Bytes_{KV,rank}\approx\frac{Bytes_{KV,total}}{TP}
$$

但 GQA/MQA 会出现 $H_{kv}<TP$。此时可能：

- 复制某些 KV heads 到多个 ranks；
- 使用不同 attention parallel layout；
- 限制 TP size；
- 在运行时增加 collective；
- 将 Q heads packing 到持有相应 KV 的 rank。

所以简单用总 KV 除以 TP 不总是正确。容量估算必须读取目标模型和引擎的 KV sharding 规则。

P/D 分离时，发送端与接收端 TP/PP 拓扑若不同，还可能需要 KV layout transformation。传输字节之外，reshape/all-to-all 也会进入 TTFT。

## Context Parallel 下 Cache 又怎样变化

Context Parallel 将同一条序列沿 token 维分到多卡，每卡只保存部分 positions。Prefill 可用 Ring Attention 等方法完成全局 attention。

Decode 时新 Query 需要读取分布在多个 ranks 的 KV：

- 广播 Q 到 KV shards；
- 每卡计算局部 score、max、sum 和 output partial；
- 跨卡合并 online Softmax statistics；
- 将最终 attention output 交给后续层。

Cache 容量随 CP 分摊，单 token latency却增加 collective/P2P。长上下文是否值得 CP，取决于单卡容量、KV read 时间和网络延迟。

这与 Ring Attention 文章形成连接：训练/prefill 的长 Q 可以用大量本地计算隐藏 K/V 通信，decode 的 Q 很短，通信更容易暴露。

## Prefix Cache、Session Cache 与模型状态不要混用

工程系统里“缓存”可能指：

- tokenizer/chat template 结果；
- prompt embedding；
- encoder output；
- KV Cache；
- prefix block hash index；
- response/result cache；
- Agent 会话历史文本。

只有 KV Cache 是每层 attention 的数值状态。保存对话文本后重新发送，可以重新计算出 KV，但不等于保存了 KV；response cache 命中则可能完全跳过模型，语义和失效策略又不同。

监控和 API 命名应明确层级。例如 `cache_hit` 如果没有说明是 tokenizer、prefix KV 还是完整响应，无法判断它节省了多少 GPU 工作。

## 如何估算一个部署能容纳多少 Token

先从显存预算分解：

$$
M_{GPU}
=M_{weights}
+M_{KVpool}
+M_{activations}
+M_{runtime}
+M_{reserve}
$$

得到可分配 KV pool：

$$
M_{KVpool}
=M_{GPU}-M_{other}
$$

理论 token capacity：

$$
Capacity_{tokens}
=\left\lfloor
\frac{M_{KVpool}}
{2L_{layers}H_{kv}db}
\right\rfloor
$$

再按实际引擎修正：

- TP/CP sharding 或 replication；
- block padding；
- hybrid layer pools；
- quantization scale；
- speculative lookahead slots；
- prefix blocks 的引用共享；
- encoder/cross-attention reserve；
- graph capture 和 temporary workspace。

最后用 runtime 报告的 `num_gpu_blocks × tokens_per_block` 校准。公式用于解释，初始化后的实际 blocks 才是调度器能分配的容量。

## 请求并发不能只用总 Token 相除

若 KV pool 可容纳 1M token，平均请求长度 8K，简单估算是 125 条。但线上长度不是常数：

- prompt/output 有长尾；
- 请求在生成中持续增长；
- prefix cache 可能共享部分 blocks；
- partial prefills 同时占用半成品状态；
- 调度器可能为最大增长保留 watermark；
- 请求结束时间不同，blocks 动态回收。

容量规划应使用长度分布和到达过程做仿真或 trace replay，观察：

- active tokens；
- allocated slots 与实际有效 tokens；
- KV utilization 分位数；
- allocation failure；
- preemption/recompute；
- waiting queue 与 TTFT；
- prefix hit saved tokens；
- 不同租户的 block 占用。

平均长度只能估一个中心点，P99 长请求可能在短时间吞掉大量 blocks，触发全局抖动。

## 监控 KV Cache 应看哪些指标

### 容量

- total/free/used blocks；
- effective token capacity；
- active、reusable、offloaded blocks；
- tail-block internal waste；
- 每类 KV pool 的 usage。

### 生命周期

- allocation/free rate；
- refcount 和 copy-on-write 次数；
- cancellation 后释放延迟；
- orphan/leaked block 检测；
- preemption、recompute、swap 次数与 bytes。

### 复用

- request hit rate；
- matched token rate；
- 实际跳过的 prefill tokens；
- cache lookup time；
- eviction 后的 missed reuse value。

### 性能

- prefill cache write bandwidth；
- decode KV read bandwidth；
- attention kernel duration vs context length；
- dequantize/transfer 时间；
- block-table preparation 和 scheduler CPU 时间。

一个命中 1 个 16-token block 的请求和命中 100K token 的请求都算一次 hit，所以 request hit rate 不能单独衡量收益。

## 常见正确性错误

### Position Off-by-One

把当前 token 写入 slot $t+1$，或 attention 只读到 $t-1$，会造成输出偏差。需要用逐 token reference 比较每层 K/V 和 logits。

### Batch Row 串线

Continuous Batching 重排后，block table 未同步更新，请求 A 读取请求 B 的 cache。错误可能只在某个请求提前结束时出现。

### Cache 未包含模型身份

相同 tokens 在不同 LoRA、RoPE scaling 或 multimodal input 下错误共享，返回数值看似合理却属于另一个模型状态。

### Tail Block 复用错误

未填满的 block 被当作完整 prefix 共享，后续写入发生覆盖。需要 copy-on-write 或只 hash/reuse 已确认稳定的范围。

### 量化 Scale 错配

Block 数据复用后使用了新请求或新 layer 的 scale；结果不会 OOM，却会产生严重数值错误。

### 异步释放

GPU kernel 或 KV transfer 尚未完成，allocator 已把 block 分给其他请求。压力下出现偶发输出污染或非法访问。

### Sliding Window 物理位置混淆

循环 slot 已覆盖，但逻辑 position/mask 仍当作旧 token，导致 RoPE 和 attention window 错误。

## 一套可执行的验证方法

### 单请求逐步对照

使用小模型和 greedy decoding，分别运行：

- 每轮完整重算；
- 连续 KV Cache；
- paged KV Cache；
- 量化 KV Cache。

逐 step 对比 logits、selected token 和各层 cache，先定位第一个发生差异的位置。

### 变长批次

让请求在不同 step 结束和加入：

```text
A: short output
B: long output
C: arrives after A finishes
D: cancelled mid-step
```

检查 batch compaction 后 request ID、logical position 和 blocks 仍一致。

### 容量边界

- 正好填满一个 block；
- 多一个 token 触发新 block；
- pool 只剩一个 block；
- allocation 失败和 preemption；
- 释放后立即复用；
- 大量 partial prefills。

### 复用边界

- 完整相同 prefix；
- 只差最后一个 token；
- 同 token 不同 adapter；
- 不同 cache salt；
- 多模态 hash 不同；
- 分叉后 copy-on-write。

### 长时间压力

持续混合 add、decode、finish、abort、evict 和 offload，验证 used blocks 最终回到基线，并对每个 physical block 检查唯一写 owner 与合法 refcount。

## 怎样判断优化是在省计算还是省带宽

不同 KV 技术解决的问题不同，可以按资源路径分类：

| 技术 | 主要减少 Prefill 计算 | 主要减少 KV 容量 | 主要减少 Decode 读取 | 引入的主要成本 |
| --- | --- | --- | --- | --- |
| 普通 KV Cache | 不减少首次 prefill；避免 decode 时重算历史前缀 | 否，反而保存状态 | 相对重算大幅更合理，但仍读全历史 | 显存容量 |
| MQA/GQA | 间接 | 是 | 是 | 模型结构/质量折中 |
| PagedAttention | 否 | 减少预留和碎片浪费 | 读取有效 blocks | block table 寻址 |
| Prefix caching | 是 | 共享时减少重复物理块 | 通常仍读共享历史 | hash/refcount/eviction |
| KV 量化 | 否 | 是 | 是 | 量化误差与转换 |
| Offload | 否 | 降低 GPU 常驻 | 可能增加传输 | PCIe/网络延迟 |
| Sparse/window attention | 是或部分 | 是 | 是 | 模型约束或近似 |

明确资源路径后，才能选正确指标。例如 PagedAttention 的主要收益是提升可用 token capacity 和并发，不必期待单请求短上下文 kernel 一定更快。

## 小结

KV Cache 来自 causal Transformer 的一个稳定性质：历史 token 的 K/V 不会因未来 token 到来而改变。Prefill 一次建立每层缓存，decode 每轮只追加新 token 的 K/V，并用新 Q 读取整个历史。

它把重复计算问题转换成显存与带宽问题：

$$
Bytes_{token}
=2L_{layers}H_{kv}d\cdot bytes_{dtype}
$$

这个公式连接了模型结构和 Serving 容量。层数、KV heads、head dim、cache dtype 任一项变化，都会改变可承载 token 数；decode 每轮又要读取随上下文增长的状态，所以 GQA/MQA 和 KV 量化同时影响容量与速度。

工程上，cache 还必须拥有比 batch row 更稳定的身份：request ID、逻辑 position、layer/KV group 和物理 block mapping。取消、分叉、prefix sharing、offload 和异步 kernel 都要求 allocator 在正确时刻提交或释放引用。

有了这些基础，后续文章的关系会更清楚：

```text
PagedAttention -> KV blocks 如何按需分配和映射
RadixAttention -> 不同请求的公共 prefix 如何组织与复用
Mooncake/NIXL -> KV blocks 如何跨设备和存储层流动
NVFP4 KV     -> 每个 cache element 如何进一步压缩
```

KV Cache 并不是 attention 的附属数组，而是 LLM Serving 中最重要的动态状态。调度、内存、网络和 kernel 的许多设计，最终都在回答同一个问题：下一 token 所需的历史状态放在哪里、由谁拥有、何时读取、何时可以安全回收。

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150)
- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)
- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- [TensorRT-LLM KV Cache System](https://nvidia.github.io/TensorRT-LLM/features/kvcache.html)
- [TensorRT-LLM Attention](https://nvidia.github.io/TensorRT-LLM/latest/torch/attention.html)
