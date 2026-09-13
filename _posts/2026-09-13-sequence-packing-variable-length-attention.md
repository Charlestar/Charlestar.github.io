---
layout: post
title: "序列打包与变长 Attention：少算 Padding 之后，怎样保持训练目标不变"
subtitle: "从 cu_seqlens 和段内可见性追踪位置、shifted labels 与有效 target 分母，区分布局优化和训练语义变化"
date: 2026-09-13
last_modified_at: 2026-09-13
author: iStar
catalog: true
series: distributed-training
series_order: 75
technology_year: 2021
mathjax: true
tags: [分布式训练, 注意力机制]
---

同一批训练数据里，短样本经常要等长样本：张量被补齐成规则矩形，短样本后面的 padding 也占据输入位置，有时还进入投影、MLP 和 Attention 的计算。把短样本紧凑排列，能够减少这些空位置，却引出另一个问题：物理上相邻的 token，是否因此变成了同一个训练样本？

这里必须先区分两种任务。有意把文档串成连续预训练流，是一种数据与训练目标选择；把原本独立的样本打包，只改变存储和执行布局，则应保留原来的上下文边界。前者不必遵守后者的隔离要求，但不能把二者混称为同一种“无损 packing”。

[Efficient Sequence Packing without Cross-contamination](https://arxiv.org/html/2107.02027v2)讨论了打包后的 Attention、位置与样本 loss 隔离。本文技术年份取其预印本首次公开的 2021 年，实际核对 v2 的 §3 和相关附录；这不是 packing 的发明年份。论文主体是 BERT 等任务，下面的 causal decoder、FlashAttention 接口和 shifted labels 则另据固定版本实现推导，不把论文的硬件实验直接当作本文性能结论。

讨论等价时，先固定同一批逻辑样本、token 顺序与更新窗口，采用 token-local norm/MLP 的 dense decoder，并关闭 dropout 或保持同一随机路径。位置策略、loss 权重以及可能的跨样本耦合都需要另行保持一致；这些前提让下面的段内 Attention 等价能够继续传递到模型输出。

## 九个 token 放在一起，仍然保留三份样本身份

先取三个已经 tokenization 的独立样本，不额外假设有 BOS：

```text
A = [a0, a1, EOS]          长度 3
B = [b0, EOS]              长度 2
C = [c0, c1, c2, EOS]      长度 4

flat      = [a0, a1, EOS, b0, EOS, c0, c1, c2, EOS]
segment   = [ 0,  0,   0,  1,   1,  2,  2,  2,   2]
position  = [ 0,  1,   2,  0,   1,  0,  1,  2,   3]
cu_seqlens = [0, 3, 5, 9]
```

`flat[3]`（下标从 0 开始）是 B 的首 token，而它在 B 内的位置是 0。物理下标、样本身份与样本内位置，是三个相关但不同的量；后端不能仅凭这个 token 恰好接在 EOS 后面，就自动理解这些关系。

`cu_seqlens` 是长度的前缀和。第 $$r$$ 段占据半开区间 $$[cu_r,cu_{r+1})$$；对属于该段的物理位置 $$i$$，局部位置为：

$$
p_i=i-cu_r
$$

本例有 3 个逻辑样本，前缀数组长度为 4，末项 9 是总 token 数；`max_seqlen` 是各段最大长度 4，不是总长度 9，也不是某个装箱容量。

本文限定各段长度为正，因此调用方应保证 `cu[0] == 0`、前缀严格递增、`cu[-1] == total_tokens`，并让 `max_seqlen` 与差分中的最大值一致。传入后文 int32 接口时，前缀值与总 token 数也必须能用该类型表示。正确的 dtype 和 shape 不能证明这些内容正确：把 `[0, 3, 5, 9]` 错写成 `[0, 4, 5, 9]`，数组仍然很像前缀和，却把 b0 分到了 A 的上下文。

装箱过程也不一定只产生一条巨大 tensor。可以有多个 physical pack，但每个 pack 内若还放了多个独立样本，下游仍需知道每个逻辑段的边界。只记录装箱容器的边缘，会丢掉真正决定隔离语义的信息。

## EOS 是内容，真正的隔离发生在 softmax 之前

对有效 token $$i,j$$，设样本编号分别为 $$s_i,s_j$$，样本内位置为 $$p_i,p_j$$。独立 causal self-attention 的可见条件是：

$$
\operatorname{allow}(i,j)
=(s_i=s_j)\ \land\ (p_j\le p_i)
$$

相应 additive mask 在允许处为 0，其余处为 $$-\infty$$，并在 softmax 之前加入 score：

$$
O=\operatorname{softmax}\left(\frac{QK^T}{\sqrt d}+M\right)V
$$

对三条样本来说，允许区域是三块分别下三角的块对角结构，不是覆盖整条 `flat` 的单一下三角。后一种 mask 会允许 B 看 A，也允许 C 看 A、B。

一个很小的反例就能看见差异。让所有 QK score 都为 0，A 的三个 V 都为 1，B 首 token 的 V 为 10。正确隔离时，B 的首 token 只能看自己，输出是 10；整条 flat causal 时，它会均匀关注 A 的三个 token 和自己，输出：

$$
\frac{1+1+1+10}{4}=3.25
$$

A 的最后一个 token 即使名为 EOS，也没有被 softmax 自动删除。EOS 能作为模型学习到的内容信号，却不是 GPU kernel 的分段指令。把不允许的 Attention 概率在 softmax 后简单置零也不够，因为原来的分母已经包含了其他段；应在归一化前施加可见性，或采用严格等价的后端实现。

如果仅为验证构造布尔 mask，还要核对 API 的布尔语义。[PyTorch v2.5.1 SDPA](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/nn/functional.py)中 `True` 表示允许关注，不能不加说明地照搬其他 Attention 接口中 `True` 表示屏蔽的 mask。本例去掉了 padding，每个有效 query 至少可见自己；若参考实现还保留全 mask 的 padding query，必须另外处理空行，不能让全 $$-\infty$$ softmax 产生 NaN 后继续参与训练。

## Varlen 接口让逻辑段直接成为计算范围

显式构造一张 9×9 block-causal mask，可以定义正确结果，却不保证底层没有计算跨段 score。语义上的“不可见”和执行上的“不计算这些块”，是两个层次。

FlashAttention 的变长接口用前缀数组描述每条逻辑序列，使同一次调用能够处理不同长度的段。下面只示意 self-attention，且每段 Q、K 等长，Q/K 已按约定应用 RoPE：

```python
import torch
from flash_attn import flash_attn_varlen_func

# q、k、v: [9, H, 64]，同一 CUDA 设备、相同 FP16/BF16 dtype。
# 本例同头数，最后一维连续；具体 GPU 支持由所安装版本决定。
cu = torch.tensor([0, 3, 5, 9], dtype=torch.int32, device=q.device)

out = flash_attn_varlen_func(
    q, k, v,
    cu_seqlens_q=cu,
    cu_seqlens_k=cu,
    max_seqlen_q=4,
    max_seqlen_k=4,
    dropout_p=0.0,
    causal=True,
)
```

接口按 [FlashAttention v2.7.4.post1](https://github.com/Dao-AILab/flash-attention/blob/v2.7.4.post1/flash_attn/flash_attn_interface.py)及对应 [CUDA 入口](https://github.com/Dao-AILab/flash-attention/blob/v2.7.4.post1/csrc/flash_attn/flash_api.cpp)核对：该路径要求 CUDA 上连续的 int32 前缀数组，并对 Q/K/V 的 dtype、最后维 stride 和 head dimension 有约束。示例不是完整模型，也未在本文环境执行 GPU kernel。

如果传入 `[0, 9]`，就相当于告诉后端这里只有一条长度 9 的序列；`causal=True` 只会施加这条序列内部的因果关系。只重置 `position_ids`，但不确认框架确实将其转换为分段信息，也不能保证隔离。某些集成有这种适配逻辑，不代表所有模型与后端都默认具备。

变长接口负责 Attention 的逻辑范围，不负责从文本猜测文档边界，也不自动生成位置与 labels。非方形 Q/K 的 causal 对齐还有额外版本约定，不能把本例等长 self-attention 的直观下三角规则直接推广到 decode 场景。

## 位置通常在段内重置，但纯 RoPE 的理由需要讲准确

最清楚的数据合同，是让每条独立样本沿用独立处理时的 position：A 从 0 开始，B 也从 0 开始。绝对位置 embedding 会直接改变输入内容，这时把物理下标当作逻辑 position，自然不再等价。

但对于固定频率的纯 RoPE，不能简单宣称“只要不重置就一定错”。对同一段的 Q、K 都增加相同位置偏移 $$c$$，有：

$$
\big(R(m+c)q\big)^T\big(R(n+c)k\big)
=q^TR(n-m)k
$$

整体偏移在内积中抵消。这来自 [RoFormer §3.2](https://arxiv.org/html/2104.09864v5#S3.SS2)的旋转构造，详细推导见 [RoPE]({% post_url 2026-09-09-rope-relative-position-context-extension %})。前提是频率与缩放固定、同段采用同一偏移、没有其他绝对位置机制，而且段间可见性已经正确隔离。

真实实现还可能有三角函数舍入和动态频率策略。例如 [Transformers v4.46.3 的 Llama RoPE](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/llama/modeling_llama.py)在动态路径中会依据 `max(position_ids) + 1` 更新频率。若用整条 pack 的物理下标，最大位置改变，前面固定旋转频率的前提就未必成立。

反过来，仅重置 position 也不是所有动态策略的等价性证明：独立逐样本执行和合成一个 batch，可能使用不同的最大长度来触发 recipe。做数值对照时，应固定频率或明确统一动态有效长度规则，而不是认为 position 数组看起来从零开始，就已经恢复了全部语义。

## 上下文隔离之后，还有一个跨边界预测对

Causal LM 的监督存在一步偏移。本文采用 [Transformers v4.46.3 `ForCausalLMLoss`](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/loss/loss_utils.py)的约定：`logits[..., :-1, :]` 对齐 `labels[..., 1:]`。也就是位置 i 的 logit 预测位置 i+1 的 label。

如果把三个样本直接拼好，仍使用原始 token 作为全部 labels，那么 A 的 EOS 位置会被要求预测 b0，B 的 EOS 位置会被要求预测 c0。Attention 已经没有跨段访问，但 loss 又加入了原任务不存在的预测对。

所以，对于这种内部 shift 约定，要把每段首 token 对应的 label 设为忽略：

```text
input  = [  a0, a1, EOS,   b0, EOS,   c0, c1, c2, EOS]
labels = [-100, a1, EOS, -100, EOS, -100, c1, c2, EOS]
```

第一段首 label 本来也会被整条 shift 切掉，显式忽略可以统一规则。真正留下来的预测对是：

```text
A：a0 → a1，a1 → EOS
B：b0 → EOS
C：c0 → c1，c1 → c2，c2 → EOS
```

总输入有 9 个 token，有效监督只有 6 个 target，即 $$\sum_r(L_r-1)$$。这个计数基于本例没有额外 BOS、没有 prompt 忽略的前提；若给每段加 BOS，就应让它预测该段首内容 token，并重新按实际 labels 统计。

不能为了消除边界而把所有 EOS label 都屏蔽。`a1 → EOS`、`b0 → EOS` 和 `c2 → EOS` 都是原样本内合法的终止监督；需要去掉的是 EOS 位置的 logit 对下一段首 token 的预测。若采用自己提前生成 next-token labels、模型不再内部 shift 的另一种合同，忽略位置才应落在每段末 logit 上。两种写法可以等价，但不能同时 shift 两次或混用 mask 的坐标。

SFT 还需要区分 loss mask 与 Attention mask。Prompt target 可以不计 loss，但 response 通常仍需要看同一段的 prompt。标签为 `-100` 只表示这项监督被忽略，不表示对应 token 从上下文中消失。若 padding 与 EOS 使用相同 token ID，也不能按 ID 过滤全部 EOS；应根据真实 padding 位置与段结构构造 labels。

下面从已经得到的 `[T, vocab_size]` logits 显式计算两项统计，不再把这些预移位数据传给内部会再次 shift 的 loss。这里 `input_ids` 是同一份 flat token 的 `torch.long` 张量，`logits`、`input_ids` 和前面构造的 `cu` 位于同一设备；它们均已由调用方准备好，并非独立可运行的完整模型：

```python
import torch.nn.functional as F

labels = input_ids.clone()            # 本例 input_ids 的形状为 [9]
labels[cu[:-1].long()] = -100         # 每段起点；任务的其他忽略位置另行设置

shift_logits = logits[:-1].float()
shift_labels = labels[1:]
loss_sum = F.cross_entropy(
    shift_logits, shift_labels, ignore_index=-100, reduction="sum"
)
target_count = (shift_labels != -100).sum()
```

先保留和与计数，而不是立即求每个 pack 的平均，是为了让下一层训练循环按真正目标选择分母。本文没有 class weights 和 label smoothing；改变这些选项后，不能不核对 [`CrossEntropyLoss` 的归约定义](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/nn/modules/loss.py)就继续套用同一个计数规则。

## Packing 不应该替你决定长样本有多大权重

设第 $$r$$ 条样本有 $$n_r>0$$ 个有效 target，loss 和为 $$S_r$$。至少有两种不同目标：

$$
\mathcal L_{token}=\frac{\sum_r S_r}{\sum_r n_r}
$$

$$
\mathcal L_{sample}=\frac1B\sum_r\frac{S_r}{n_r}
$$

前者让每个有效 token 同权，因此长样本可能贡献更大的总权重；后者让每条样本同权，因此短样本的单个 token 权重更大。两者都可以是合理任务定义，问题在于是否被布局变化悄悄互换。

另取两个样本，A 有 2 个有效 target，每个 loss 为 1；B 有 6 个，每个 loss 为 3。Token 平均是 $$(2+18)/8=2.5$$，样本平均是 $$(1+3)/2=2$$。如果它们落在两个不同 micro-batch，而训练循环直接平均两个 micro-batch 的 mean，就会得到后者。

改变装箱方式后，即使 token 集合、每个 logit 和单项 loss 都相同，mean of means 仍可能改变样本权重。[Hugging Face 对梯度累积问题的说明](https://huggingface.co/blog/gradient_accumulation)正是通过 loss sum 与有效 target 总数区分这两件事；这里采用的不是任意 batch 均值，而是提前定义好的 token 平均目标。

若某条样本所有 labels 都被忽略，$$n_r=0$$，样本均值没有定义，需要明确过滤或零权重政策。整个更新窗口若没有任何有效 target，则不能执行除零或假装得到一次有意义更新；分布式训练中必须让相关 ranks 对这项判断达成一致。

## 分布式训练的分母属于整个更新窗口

假设有 $$W$$ 个 rank，使用默认梯度平均的 DDP，不启用改变归约语义的通信 hook 或不均匀输入 join。第 $$r$$ 个 rank 在本次更新窗口内的 loss 和为 $$S_r$$，有效 target 数为 $$N_r$$；全局计数为：

$$
N=\sum_{r=1}^{W}N_r>0,
\qquad
\mathcal L=\frac{\sum_r S_r}{N}
$$

如果每个 rank 都对自己的均值 $$S_r/N_r$$ 反向，DDP 平均后是：

$$
\frac1W\sum_r\frac{\nabla S_r}{N_r}
$$

只有在分母满足相应等权条件时，它才等于全局 token 平均的梯度。更直接的做法，是每个 rank 对下面的局部量反向：

$$
\mathcal L_r^{backward}=\frac{W S_r}{N}
$$

因为默认 DDP 随后会再除以 $$W$$：

$$
\frac1W\sum_r\nabla\left(\frac{W S_r}{N}\right)
=\frac1N\sum_r\nabla S_r
$$

这里 N 是不依赖参数的数据计数。前因子 W 只用于抵消本段假定的 DDP 平均，绝不是任何分布式框架都要额外乘的通用规则。[PyTorch v2.5.1 DDP](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/nn/parallel/distributed.py)明确讨论了平均梯度与 sum loss 的关系；若框架已有 token 校正、使用 sum 归约或改变有效 world size，需要重新对齐整条公式，避免重复归一化。

若一个 rank 的窗口包含 G 个 micro-batch，令 $$S_r=\sum_g S_{rg}$$、$$N_r=\sum_g N_{rg}$$。先从已经准备好的 target masks 得到全窗口的 N，然后每个 micro-batch 对 $$W S_{rg}/N$$ 反向即可，不能再额外除一次 G。

工程上可以在前向前统计本地整个窗口的 int64 count，再只归约这个计数，不需要逐 token 通信。使用 DDP `no_sync` 时，窗口前 G−1 个 micro-batch 的前向和反向都应进入对应上下文，最后一次正常前后向同步已累积梯度；不能只把 backward 包住就假设同步已被正确推迟。

如果某个 rank 本地没有有效 target、全局却有，它不能单独跳过其他 ranks 正在等待的同步。应保持一致的参与路径，让本地贡献符合零监督语义。这里与“全局 N 为零，大家一起跳过”是两个不同分支。

与 [混合精度训练]({% post_url 2026-09-13-mixed-precision-training-loss-scaling %})结合时，全局 target 分母与 loss scale 是两个独立量：前者定义目标权重，后者移动反向数值范围。整个窗口 scale 保持不变，梯度完成所需累积、同步和解除缩放之后，再做 global-norm clipping 与更新。Packing 不应悄悄改变每次更新包含多少有效监督，也不应使不同 rank 各自决定前进到不同参数版本。

## 少了多少空位、少了多少 Attention 工作，是两本账

对于长度为 $$L_r$$ 的 B 条样本，令 $$T=\sum_rL_r$$，最大长度为 $$L_{max}$$。补齐布局有 $$BL_{max}$$ 个 token 位置，紧凑布局只有 T 个。投影、norm 与 MLP 等逐 token 工作能否相应减少，取决于这些层是否也处理紧凑布局；只换 Attention kernel，不保证其他层同时去掉 padding。

Attention 的规模要看长度平方，而不只是 token 总数：补齐的稠密 score 槽位为 $$BL_{max}^2$$，逐段处理是 $$\sum_rL_r^2$$，把全部 token 错当一条长序列则变成 $$T^2$$。若只数合法 causal pair，又是另一套三角计数。

对于前面的 `[3, 2, 4]`：

| 计数对象 | 对齐到最长样本 | 按独立逻辑段处理 | 误当单条长度 9 序列 |
| --- | ---: | ---: | ---: |
| token 位置 | 12 | 9 | 9 |
| 稠密 score 槽位 | 48 | 29 | 81 |
| 合法段内 causal pair | 需排除 padding 后为 19 | 19 | 单条 causal 允许 45 对 |

最后一列的 45 对中，多了 26 个跨样本 pair。它既改变计算范围，也改变任务。相反，正确 varlen 去掉跨段计算，并没有消除单条长度 L 样本自身 dense Attention 的 $$O(L^2)$$ 工作；它不是通过 packing 把长上下文 Attention 变成线性复杂度。

这些都是数学槽位或有效 pair 的计数，不是 GPU 测量，也不能直接将 48/29 换算成实际加速倍数。Kernel 还有 tile 尾部、softmax、归约、访存和调度成本；大量短段可能让块利用率不理想，长度 bucketing 也可能是足够有效且更简单的方案。

跨 rank 分配同样不能只按 T 平衡。两个 rank 的 token 数相同，一个主要是许多短段，另一个有单条长段，$$\sum L_r^2$$ 可能差很多，Attention 时间也就可能失衡。进一步加入 [Context Parallel]({% post_url 2026-08-14-sequence-context-parallel-long-context-training %})时，还要让分片、通信与全局段边界保持一致。

完整数据流水线还支付 tokenization、排序装箱、元数据构造、搬运和重排成本，动态形状也可能影响编译或执行图复用。因此应同时记录有效 input tokens/s、监督 targets/s、每次更新样本数和 target 分母，而不只是“物理 pack 填得多满”。把 token budget 填满，却增加了每步监督数量，是训练配置变化，不能全部归功于 kernel 优化。

## 最有针对性的验证：改 A，不该改变 B

先关闭 dropout，固定位置 recipe，选没有跨样本统计的 dense decoder。分别独立执行 A/B/C，再使用 packed 路径，比较各段 logits、有效 target loss 和参数梯度。用合理浮点容差接受不同 kernel 与归约顺序，不承诺逐位一致。

然后只替换 A 的 token，保持 B/C、它们的位置与 task masks 不变。若 B/C 的 logits 改变，应沿 Attention 边界、位置策略和其他跨样本耦合寻找原因。这个测试直接检查样本隔离，比只观察总 loss 接近更能暴露泄漏。

Loss 还要单独验：枚举实际 next-token 预测对，确认没有 EOS→下一段开头，没有误删 EOS target，统计 shift 后未忽略项，重新装箱或改变 micro-batch 划分后应保持相同目标。分布式版本再检查不等 target 数时的权重，不能只用各 rank 长度相同的样本。

等价性也有范围。BatchNorm 等跨样本统计、对比损失、MoE 容量竞争，以及未映射到同一 token 的 dropout 随机路径，都可能让布局变化影响结果；动态 RoPE 也已展示了类似边界。Attention block mask 正确，并不能独自证明整个模型在任意 packing 下等价。

本文的配套 CPU 复算可在仓库根目录运行：

```sh
node scripts/reviews/sep13-sequence-packing-examples.mjs
```

它只使用 Node.js 标准库，检查独立与 block-causal Attention 的数值等价、Q/K/V 梯度的有限差分、跨段污染、前缀边界、shifted labels、有效分母与本文的计数。它不执行上述 Python 片段，不调用 FlashAttention 或 DDP，也不验证完整模型的参数梯度、GPU 性能与训练收敛。

本文的计数、标签对齐与污染反例是数学或固定接口层面的说明，未声称完成 FlashAttention GPU 实测或大模型收敛验证。保持独立样本的 packing，本质上是在保留三件事的同时改变布局：每个 token 原本能读到的上下文、它原本使用的位置规则，以及它原本在训练目标中的权重。少算 padding 的收益，只有在这三者没有被交换掉时，才是同一任务上的执行优化。
