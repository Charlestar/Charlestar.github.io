---
layout: post
title: "DeepSeek V3.2 稀疏注意力：Lightning Indexer 与 Top-k 选择"
subtitle: "从 MLA 全量访问到内容相关的细粒度稀疏化"
date: 2026-05-18 12:00:00 +0800
last_modified_at: 2026-08-09
author: iStar
catalog: true
mathjax: true
tags: [AI Infra, LLM推理, 稀疏注意力]
---

长上下文让模型能够阅读完整代码库、长文档和多轮代理轨迹，也让 attention 成为越来越显眼的计算瓶颈。对长度为 $L$ 的序列，dense causal attention 需要考虑约 $L(L+1)/2$ 个 query-key 关系；序列翻倍，关系数量接近四倍。

DeepSeek V3.2 的思路不是把注意力限制在固定窗口，而是先问一个更有针对性的问题：**对当前 query 来说，历史上究竟哪些 token 值得进入正式 attention？**

DeepSeek Sparse Attention（DSA）用 Lightning Indexer 扫描历史位置，再只取 index score 最高的 $k$ 个 KV 条目执行主注意力。它把昂贵计算从“全部历史”缩到“内容相关的少量候选”，但也引入了索引训练、top-k、离散访存和专用 kernel 等新成本。

## 先明确 DSA 改了什么

DeepSeek V3.2 并不是从一套全新的 Transformer 重新训练。官方技术报告说明，它从已经扩展到 128K 上下文的 DeepSeek-V3.1-Terminus checkpoint 出发，通过继续训练引入 DSA；相较 V3.1-Terminus，架构上的变化集中在 attention。

这点很重要，因为 DSA 同时继承了两层背景：

1. DeepSeek 原有的 **Multi-head Latent Attention（MLA）** 负责压缩 KV 表示；
2. DSA 再从历史 latent KV 中选择当前 query 真正访问的子集。

MLA 解决“每个历史 token 保存多少 KV”，DSA 解决“当前计算读取多少历史 token”。一个偏向存储表示，另一个偏向访问范围，不能把二者当成同一优化。

## Dense attention 的成本从哪里来

标准 attention 可以写为：

$$
Attention(Q,K,V)
= Softmax\left(\frac{QK^T}{\sqrt{d}} + M\right)V
$$

其中因果 mask $M$ 禁止 token 看到未来位置。若序列长度是 $L$，prefill 阶段的 $QK^T$ 与概率乘 $V$ 都包含 $O(L^2d)$ 量级的工作。

decode 阶段每次只有一个新 query，不会在单步形成完整的 $L\times L$ 矩阵，但它要扫描长度为 $L$ 的 KV Cache。生成很多 token 后，累计成本仍随上下文和输出长度增长，而且通常受显存带宽影响。

稀疏注意力想减少的是这次“全量扫描”。但采用哪种稀疏模式，会决定模型还能访问哪些信息。

### 固定窗口不是内容选择

滑动窗口只保留最近的 $w$ 个 token，复杂度清楚、内存连续、kernel 也容易优化。然而早期的函数定义、系统提示或检索证据可能落在窗口之外。

块稀疏可以额外保留全局 token 或固定跨块连接，但模式仍主要由位置决定。

DSA 的选择则随 query 内容变化。查询“第一章给出的假设”时可以选择很远的开头，查询“上一行变量值”时又可能集中在局部。这种细粒度带来表达能力，也让选择过程本身成为需要计算和学习的模块。

## DSA 的两段式数据流

对位置 $t$ 的 query，DSA 先让 Lightning Indexer 为每个历史位置 $s\le t$ 打分：

```text
hidden state h_t
  │
  ├─> Lightning Indexer ─> score I_{t,0...t}
  │                            │
  │                         top-k positions
  │                            │
  └─> MLA query ───────────────┼─> gather selected latent KV
                               │
                               └─> sparse MLA ─> output u_t
```

用论文的形式表示：

$$
I_{t,s}
= \sum_{j=1}^{H^I}
w^I_{t,j}\cdot
ReLU\left(\mathbf q^I_{t,j}\cdot\mathbf k^I_s\right)
$$

其中：

- $\mathbf q^I_{t,j}$ 是 query token 派生出的第 $j$ 个 indexer query；
- $\mathbf k^I_s$ 是历史 token 的 index key；
- $w^I_{t,j}$ 是当前 query 为不同 indexer head 产生的权重；
- $H^I$ 是 indexer head 数量；
- 多个 head 的 ReLU 点积被加权汇总为每个历史位置的单一分数。

然后取 $I_{t,:}$ 的 top-k 位置集合 $\mathcal S_t$：

$$
\mathcal S_t
= \{s\mid I_{t,s}\in TopK(I_{t,:})\}
$$

主 attention 只读取这些位置：

$$
\mathbf u_t
= Attn\left(
\mathbf h_t,
\{\mathbf c_s\mid s\in\mathcal S_t\}
\right)
$$

$\mathbf c_s$ 是 MLA 保存的 latent KV 条目。Indexer score 只负责“选谁”，并不是最终的 attention probability；top-k 之后仍要用主模型的 query 与选中的 KV 重新计算正式 attention。

## 为什么叫 Lightning Indexer

如果索引器和主 attention 一样昂贵，先筛选再计算不会有收益。DSA 让 indexer 采用更小的表示与较少的头，并允许以 FP8 实现。论文选择 ReLU 也明确包含吞吐方面的考虑。

所以两段工作的角色不同：

| 阶段 | 扫描范围 | 单个位置的计算 | 输出 |
| --- | --- | --- | --- |
| Lightning Indexer | 全部历史 token | 小维度、少量头、可用 FP8 | 每个位置一个选择分数 |
| Sparse MLA | top-k 历史 token | 完整的模型 attention 计算 | attention 输出 |

核心交换是：保留一次便宜的全局检索，用它换掉绝大多数昂贵的主 attention 计算。

“Lightning”描述的是相对于 MLA 的轻量性，并不表示索引过程没有复杂度。它仍要为每个 query 检查历史位置，这会成为进一步扩展上下文时的新瓶颈。

## DSA 怎样嵌入 MLA

MLA 不直接缓存传统 MHA 中每个 head 的完整 K/V，而是缓存压缩后的 latent 表示，并在计算中与 query 侧权重组合。它显著降低 KV Cache 体积，但 sparse kernel 还面临一个约束：同一个被选中的 KV 条目最好能被多个 query head 共享，否则每个 head 选择不同位置会让 gather 和矩阵运算碎片化。

DeepSeek V3.2 因此在 MLA 的 MQA 模式下实例化 DSA：一个 query token 的所有 query head 共享同一组 top-k latent KV 位置。可以把过程理解成：

```text
一个 token 的多组 query head
        │
        ├─ indexer 汇总出一份位置排名
        │
        └─ 所有主 attention heads 共享 top-k latent KV 集合
```

共享选择集合牺牲了“每个主 head 都有完全不同稀疏模式”的自由度，却换来了更适合 GPU 的数据复用和 kernel 组织方式。

官方演示代码也清楚展示了这一边界：先用 indexer 得到 `topk_indices`，构造只允许这些位置通过的 mask，再对选定范围做主 attention。生产 kernel 会避免真的物化巨大 dense mask，但语义相同。

## 复杂度降低到哪里

若主 attention 对每个 query 只选固定 $k$ 个历史 token，它的核心复杂度由：

$$
O(L^2d)
$$

下降为近似：

$$
O(Lkd)
$$

在 DeepSeek V3.2 的稀疏训练阶段，官方报告使用 $k=2048$。当 $L=128K$ 时，$k/L$ 约为 $1.56\%$，主 attention 访问范围大幅缩小。

但完整成本还包含：

$$
T_{DSA}
= T_{index}
+ T_{topk}
+ T_{gather}
+ T_{sparse\_attn}
$$

Indexer 对每个 query 仍扫描历史，渐进复杂度仍可写成 $O(L^2d_I)$，只不过 $d_I$、头数和精度都显著小于主 attention。官方技术报告也明确没有把它省略，而是将 DSA 的收益归因于“低成本 indexer + 优化实现 + 主 attention 的 $O(Lk)$”。

因此不能用 $L/k$ 直接推导端到端加速倍数。随着主 attention 被压缩，indexer、top-k 和内存访问在总时间中的比例会越来越高。

## 模型必须学会如何选择

把一个 dense checkpoint 的 attention 权重直接裁成 top-k，往往会丢失模型从未学会压缩的信息。DeepSeek 的继续训练分成两个有明确职责的阶段。

### Dense warm-up：先训练索引器模仿主注意力

这一阶段仍执行 dense attention，冻结除 Lightning Indexer 外的模型参数。主 attention 在所有 head 上聚合后的分数，经 L1 归一化得到目标分布 $p_{t,:}$；indexer 则输出自己的位置分布。

训练目标是 KL divergence：

$$
\mathcal L^I
= \sum_t D_{KL}\left(
p_{t,:}\ \|\ Softmax(I_{t,:})
\right)
$$

它让 indexer 先学会近似回答：“dense attention 认为哪些位置重要？”官方报告记录的 warm-up 为 1000 步、共约 2.1B token。这个阶段的目的不是让语言模型重新获得能力，而是初始化一个可用的检索器。

### Sparse training：让整个模型适应可见集合

完成 warm-up 后，训练真正启用 top-k，只让主 attention 访问 $\mathcal S_t$，同时优化全部模型参数。Indexer 继续通过自己的 KL 目标学习，主模型则通过语言建模 loss 适应稀疏访问。

在该阶段，indexer 的输入会从主模型计算图中 detach：

- indexer 只接收 $\mathcal L^I$ 的训练信号；
- 主模型只按 language modeling loss 更新；
- top-k 的离散选择不需要把梯度穿过主模型与 indexer 纠缠在一起。

官方报告给出的稀疏阶段为 15000 步、约 943.7B token，并为每个 query 选择 2048 个 KV token。这一训练规模说明，DSA 不是推理时随手打开的剪枝开关，而是 checkpoint 架构与训练配方的一部分。

## Prefill 与 decode 中发生了什么

### Prefill

一段长度为 $L$ 的 prompt 同时产生很多 query。Indexer 需要为这些 query 与因果范围内的历史 token 计算分数，再完成每行 top-k。若直接物化完整 $L\times L$ score 矩阵，显存峰值会抵消稀疏化收益。

高效实现通常需要：

1. 按 query/key tile 计算 index score；
2. 每个 tile 只保留局部候选；
3. 分层归并成最终 top-k；
4. 把选择结果直接交给 sparse MLA kernel；
5. 尽量不把完整 score 矩阵写回 HBM。

短 prompt 是另一个边界。当 $L\le k$ 时，没有历史 token 可以裁掉；即使 $L$ 稍大于 $k$，top-k 的管理成本也可能高于 dense kernel。官方报告提到，短序列 prefill 使用专门的 masked MHA 模式模拟 DSA，以获得更合适的效率。也就是说，稀疏语义不要求所有长度都走同一套 kernel。

### Decode

每个新 token 只产生少量 query，但历史 KV 已经很长。此时 indexer 是一次对长前缀的低维扫描，随后 sparse MLA 读取离散的 top-k latent KV。

decode 的关键不只是 FLOPs，还包括：

- index key 与 MLA latent KV 的缓存体积；
- top-k position 到物理 KV page 的映射；
- 离散位置是否能形成合并访存；
- continuous batching 中各请求不同的上下文长度；
- tensor/data parallel 下选择结果和 KV 所在设备的对应关系。

理论上少读很多 token，如果选中的位置遍布大量物理 page，实际带宽收益仍可能低于计算量降幅。

## Top-k 是算法与 kernel 的交界面

Top-k 看似只是一个排序操作，长序列下却不能用通用 `sort` 粗暴解决。完整排序需要处理所有位置，而 DSA 只关心最大的 $k$ 个；合理 kernel 会采用分块选择与归并，尽量在寄存器或共享内存中保留候选。

实现还必须定义好：

- causal mask 后哪些位置有效；
- 分数相同时如何稳定选择；
- padding 与 ragged sequence 如何排除；
- paged KV 下逻辑位置怎样变成 block/offset；
- tensor parallel 各分片是先局部 top-k 再全局合并，还是复制索引信息；
- FP8 score 的缩放和数值误差是否会改变临界位置排序。

DeepSeek 已把高性能 indexer logit kernel（包括 paged 版本）放在 DeepGEMM，把 sparse attention kernel 放在 FlashMLA；TileLang 版本则更便于阅读和研究。这些组件的拆分也反映了 DSA 的真实数据流，而不是一个单独的 `sparse_attention()` 调用。

## 位置编码是容易被低估的正确性边界

2025 年 11 月，DeepSeek 官方仓库修正过演示代码中 indexer 的 RoPE 布局问题：indexer 输入需要 non-interleaved layout，而 MLA 模块期望 interleaved layout。两者如果复用错误的旋转方式，代码仍能运行，却可能让 index score 的语义位置错位并降低模型表现。

这个修复很能说明稀疏 attention 的验证难点：

- 最终输出没有 shape error，不代表选中的位置正确；
- indexer 与主 attention 可能使用不同的数据布局；
- 短文本回归测试未必能暴露长距离检索退化；
- kernel 优化必须同时检查 top-k 集合与最终 logits。

实现兼容层时，应以官方最新代码为语义基准，并对 index score、top-k index、主 attention 输出逐层建立数值对照。

## 怎样评估质量与性能

“公开榜单基本持平”不能推出每个任务都严格无损。DeepSeek 官方为了隔离 DSA 的影响，将 V3.2-Exp 与 V3.1-Terminus 的训练配置尽量对齐，并报告了综合能力与长上下文评测的总体相近表现；各单项分数仍有升有降。

一套完整验证应分四层进行。

### 选择层

- indexer 分布与 dense attention 聚合分布的 KL；
- top-k 对 dense 高权重 token 的 recall；
- 相邻层、相邻 query 的选择重叠度；
- 不同位置距离和内容类型下的召回。

### 模型层

- 相同输入下 logits/KL 与参考实现的差异；
- perplexity 与生成稳定性；
- 长距离 needle retrieval；
- 长文档问答、代码依赖与多轮代理轨迹。

### Kernel 层

- prefill/decode 分开测量；
- 不同 $L$、batch size、并行配置与 page size；
- indexer、top-k、gather、sparse MLA 的独立耗时；
- HBM 读写量、workspace 峰值和 graph capture 兼容性。

### 服务层

- TTFT、TPOT、吞吐和 goodput；
- 短长请求混合下的调度行为；
- prefix cache 与 continuous batching 的影响；
- dense/masked/sparse 路径切换阈值。

只有端到端加速而没有选择正确性，可能是在用模型质量换速度；只有理论 FLOPs 降低而 TTFT/TPOT 不变，则说明瓶颈已经转移到 indexer、通信或访存。

## DSA 真正改变的瓶颈

DSA 的贡献不是让长上下文成本凭空消失，而是把最昂贵的主 attention 变成内容相关的固定规模访问。完成这一步后，系统关注点也随之变化：

```text
dense MLA
  └─ 主要成本：对全部历史 KV 做正式 attention

DSA
  ├─ 便宜但全量的 index score
  ├─ 长序列 top-k
  ├─ 离散 KV gather
  └─ 只对 2048 个候选做 sparse MLA
```

当上下文继续增长，Indexer 仍有 $O(L^2)$ 的 prefill 关系数和 $O(L)$ 的单步 decode 扫描。后续研究自然会走向层间复用、分层检索或块级粗筛，但这些属于对 DSA indexer 的进一步改进，不应反过来写成 DeepSeek V3.2 原始机制的一部分。

## 小结

理解 DeepSeek Sparse Attention，可以抓住四个彼此连接的事实：

1. 它建立在 MLA 之上，先压缩 KV，再稀疏访问 KV；
2. Lightning Indexer 对全部历史做低成本内容检索，主 attention 只读取 top-k；
3. Indexer 先在 dense attention 上蒸馏 warm-up，随后模型经过大规模 sparse continued training；
4. 主 attention 降到 $O(Lk)$ 后，indexer、top-k、paged gather 与数据布局成为新的系统瓶颈。

因此，DSA 既是模型架构，也是训练方法和 kernel 协同设计。只复制公式或只写一个 top-k mask，都不足以复现它的效率与质量。

## 参考资料

- [DeepSeek-V3.2 技术报告](https://arxiv.org/html/2512.02556)
- [DeepSeek-V3.2-Exp 官方仓库与更新说明](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp)
- [DeepSeek-V3.2-Exp 官方推理实现](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py)
- [DeepSeek-V2：Multi-head Latent Attention](https://arxiv.org/abs/2405.04434)
- [DeepGEMM：Indexer Logit Kernels](https://github.com/deepseek-ai/DeepGEMM)
- [FlashMLA：Sparse MLA Kernels](https://github.com/deepseek-ai/FlashMLA)
