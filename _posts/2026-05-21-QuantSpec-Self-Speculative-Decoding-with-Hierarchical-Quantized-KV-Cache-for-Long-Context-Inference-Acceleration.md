---
layout: post
title: "QuantSpec：分层量化 KV Cache 的自推测解码"
subtitle: "用同一份位平面同时服务 INT4 草稿与 INT8 验证"
date: 2026-05-21 12:00:00 +0800
last_modified_at: 2026-09-02
author: iStar
catalog: true
series: speculative-decoding
series_order: 50
technology_year: 2026
mathjax: true
tags: [推测解码, KV Cache, 模型量化]
---

推测解码通常让一个小模型连续提出多个 token，再由目标模型一次并行验证。只要草稿便宜、接受率又足够高，一次昂贵的目标模型前向就能提交多个 token。

长上下文改变了这笔账。此时 decode 不仅要加载模型权重，还要在每一层读取一条很长的 KV Cache。独立草稿模型也需要自己的 KV Cache；它即使参数更少，额外缓存仍会迅速消耗显存。更麻烦的是，小模型对长文档的理解常常弱于目标模型，候选 token 更容易被拒绝。

QuantSpec 给出的方案很有针对性：**草稿与目标使用同一模型架构，但读取同一份 KV Cache 的不同精度视图。草稿只读上 4 bit，验证同时读上、下两个 4-bit 位平面以恢复 INT8。** 最近尚未稳定提交的 token 则留在双 full-precision buffer 中，避免频繁量化和回滚。

这不是简单地“把旧 KV 量化、新 KV 保持高精度”，而是一套为推测解码事务设计的共享缓存格式。

## 为什么长上下文需要不同的草稿模型

先看普通 autoregressive decode。每生成一个 token，模型都要执行所有层，并在 attention 中读取此前的 K/V：

$$
q_tK_{0:t}^{T}\quad\text{和}\quad softmax(\cdot)V_{0:t}
$$

当上下文很短时，模型权重往往是主要的 HBM 流量；当上下文增长到数万甚至十几万 token，KV Cache 的读取占比持续上升。论文用 arithmetic intensity 分析说明，长上下文 decode 的 attention 接近典型 memory-bound 工作：计算不一定很多，搬运 KV 的字节却很多。

传统大小模型组合面对两个矛盾：

- 小草稿模型仍要维护与上下文长度成正比的缓存；
- 小模型可能无法理解完整长上下文，导致接受率下降。

一些 self-speculative 方法让同一个目标模型只读取稀疏 KV 来生成草稿。这样不需要另一个模型，但删除历史 token 可能恰好丢掉摘要、长距离引用或检索证据。

QuantSpec 不删除 KV 条目，而是用较少 bit 近似它们。草稿仍能看到完整上下文位置，只是数值精度更低。

## 推测解码的收益由什么决定

设草稿一次提出 $\gamma$ 个 token：

$$
g_1,g_2,\ldots,g_\gamma
$$

目标模型对整段候选做一次前向，得到各位置的分布 $p_1,\ldots,p_\gamma$。草稿生成时的分布记为 $q_1,\ldots,q_\gamma$。经典随机采样验证会以：

$$
\min\left(1,\frac{p_i(g_i)}{q_i(g_i)}\right)
$$

的概率接受 $g_i$；首次拒绝时从修正分布采样，并丢弃后续候选。若全部接受，还可以从目标模型额外得到一个 token。

于是每轮时间可以粗略写成：

$$
T_{round}
= \gamma T_{draft}
+ T_{verify}(\gamma)
+ T_{bookkeeping}
$$

而收益取决于每轮平均提交多少 token。草稿很快但候选大量被拒绝，或接受率很高但草稿与目标一样昂贵，都不会带来理想加速。

QuantSpec 同时在两个方向调整这笔账：

1. 用 INT4 权重与 INT4 KV 视图降低 $T_{draft}$；
2. 草稿与目标共享架构、保留全部上下文位置，以维持较高接受率。

## “自推测”不是再加载一个模型

QuantSpec 的 draft path 与 target path 使用相同架构和参数语义，但执行精度不同：

| 路径 | 权重/计算角色 | 历史 KV 视图 | 目的 |
| --- | --- | --- | --- |
| Draft | 4-bit 量化权重的快速近似路径 | upper INT4 | 连续提出候选 |
| Target | 目标验证路径 | upper + lower，恢复 INT8 | 计算接受/修正分布 |
| Recent buffer | 两条路径都可访问 | full precision | 保存近期与未提交 KV |

“同一个模型”带来的是参数和长上下文能力对齐，不代表 draft 与 target 的 logits 完全相等。量化正是二者速度差和分布差的来源。

## 一份 KV 怎样同时表示 INT4 与 INT8

QuantSpec 的核心是 hierarchical quantization。对已经量化的历史 KV，不分别保存一份 draft INT4 cache 和一份 target INT8 cache，而是拆成两个 4-bit 分量：

- $C_U^{INT4}$：上 4-bit，也是草稿直接使用的粗粒度表示；
- $C_L^{INT4}$：对上 4-bit 量化残差再次量化得到的下 4-bit；
- 目标路径把二者组合，得到更细的 INT8 表示。

论文写作：

$$
C^{INT8}
= 2^4 C_U^{INT4} + C_L^{INT4}
$$

若沿用 QuantSpec 论文的记号，把 INT8 反量化的 scale 和浮点域加性偏置分别记为 $S^{INT8}$ 与 $Z^{INT8}$（论文将后者称为 zero point）：

$$
C^{FP}
= C^{INT8}S^{INT8}+Z^{INT8}
$$

这里的加号不是反量化符号写错，而是 zero point 的参数化约定不同。若实现采用更常见的**整数域** zero point $z$，反量化会写成：

$$
C^{FP}=(C^{INT8}-z)S^{INT8}
$$

两式在 $Z^{INT8}=-zS^{INT8}$ 时完全等价。换言之，不能只把加号改成减号而仍让 $Z$ 保持原定义；对接 kernel、checkpoint 或量化工具时，必须先确认 metadata 保存的是整数 zero point $z$，还是已经乘过 scale 的浮点偏置 $Z$。

代入后，draft 只加载 $C_U$，target 加载 $C_U$ 和 $C_L$。两条路径共享 bit plane，不需要在每轮开始时把 INT8 重新量化成另一份 INT4 cache。

可以把同一个缓存条目想成分层图像：

```text
目标视图： [ upper 4-bit | lower 4-bit ]  -> INT8 reconstruction
草稿视图： [ upper 4-bit ]                -> INT4 approximation
```

这里的“hierarchical”描述的是**精度位平面的嵌套**，不是仅按 token 年龄设置不同 bit width。

### 为什么上、下 4-bit 的量化方式不同

论文先用 asymmetric、round-to-nearest 得到 $C_U$。剩余误差大致以 0 为中心，因此 $C_L$ 使用 symmetric、round-to-nearest 更适合残差分布。

这和随意截取一个 INT8 整数的高半字节并不完全相同：它从量化误差出发构造第二层，使上层本身就是可用的 INT4 近似，下层再补偿它遗漏的信息。

## K 与 V 为什么采用不同量化轴

Key 和 Value 的异常值结构并不相同。QuantSpec 延续 KV 量化研究中的常见观察：

- K 沿 channel 方向量化更有利于控制误差；
- V 沿 token 方向量化更合适；
- 两者都采用 per-group asymmetric quantization；
- 论文实验把 group size $G$ 设为 head dimension，以平衡误差与 metadata/执行开销。

一个普通对称量化公式可以帮助理解 scale：

$$
s=\frac{\max |x|}{2^{b-1}-1},\qquad
q=clip\left(round\left(\frac{x}{s}\right)\right)
$$

但 QuantSpec 的实际方案包含 asymmetric upper quantization、symmetric residual quantization、按 K/V 不同轴分组，不能用上式替代实现细节。

量化轴还影响缓存布局。若 scale 所属维度与 attention kernel 的读取方向不一致，反量化会产生额外转置、非连续访问或大量 metadata load，理论上的字节节省就未必能兑现。

## 为什么最近的 KV 不能立刻量化

推测解码具有“先写候选、后决定是否提交”的事务特征。假设草稿提出四个 token：

```text
confirmed prefix | g1 | g2 | g3 | g4
                         ^ target rejects here
```

只有 $g_1$ 被接受；$g_2$ 需要替换为修正 token，$g_3,g_4$ 必须丢弃。如果每生成一个候选就把 KV 合并进分组量化 cache，拒绝时就要：

1. 找到候选所在量化 group；
2. 解包或反量化整组；
3. 删除被拒绝位置；
4. 写入修正 token；
5. 重新计算 scale 并量化。

这些操作发生在 decode 热路径，且拒绝越频繁，重复工作越多。

另一方面，如果只为凑满 channel-wise group 临时保留一小段 recent KV，但一满就立即量化，最靠近当前 query 的 token 会很快失去 full precision，也可能放大量化引起的分布差异，降低 draft/target 接受率。

QuantSpec 用 double full-precision buffer 同时解决这两个问题。

## 双缓冲如何维护可提交边界

Full-precision 区总容量为 $2G$，分成两个长度为 $G$ 的部分：

$$
[C_{F_1}, C_{F_2}]
$$

它们的职责不同：

- $C_{F_1}$ 保存已经确认、但暂未量化的最近一组 token；
- $C_{F_2}$ 接收本轮及后续 decode 新产生的 token，其中可能包含未验证候选。

Prefill 完成后，大部分前缀被转换为 $C_U,C_L$，最近至少 $G$ 个 token 留在 $C_{F_1}$。Decode 时新 KV 只追加到 $C_{F_2}$。

一次推测轮次的状态变化可以写成：

```text
1. draft(g1...gγ)
   quantized history | F1 confirmed | F2 tentative candidates

2. target verify
   找到接受前缀，产生修正/额外 token

3. rollback
   只截断 F2 中被拒绝及其后的 KV

4. commit
   接受 token 留在 F2，成为 confirmed

5. F2 接近容量边界时
   quantize(F1) -> append to upper/lower cache
   move confirmed part of F2 -> F1
```

因为未确定状态只存在 full-precision $C_{F_2}$，回滚是一次按逻辑长度截断，不需要修改已经打包的量化历史。量化也从“每个 decode step”摊销为“大约每 $G$ 个已确认 token 一次”。

至少 $G$ 个最近 token 保持 full precision，还减小了草稿与目标在局部上下文上的差异。对自然语言生成而言，近期 token 往往对下一 token 有很强影响；但应把它视为论文设计与实验观察，而不是所有任务中“远端 KV 都不重要”的结论，因为远端 token 仍被保留在量化 cache 中。

## 一轮 QuantSpec 的完整流程

将各组件放到一起，流程如下：

### 1. Prefill

目标模型处理完整 prompt，生成初始 KV。除 recent buffer 外的前缀被量化为 upper/lower 两个 INT4 位平面，最近 $G$ 至 $2G$ 个条目保留 full precision。

### 2. Draft

量化权重路径连续运行 $\gamma$ 次。Attention 对旧历史只加载 $C_U$，对近期部分读取 $C_{F_1},C_{F_2}$。每次生成的候选 KV 追加到 $C_{F_2}$，并保存草稿概率 $q_i$。

### 3. Verify

目标路径一次处理候选序列。旧历史同时读取 $C_U$ 与 $C_L$ 恢复 INT8，近期仍用 full precision buffer，得到目标分布 $p_i$。

### 4. Accept 或 correct

从第一个候选开始按推测采样规则检查。连续接受的 token 被提交；首次拒绝时采样修正 token，并停止接受后续候选。若全部候选通过，可提交目标路径额外给出的 token。

### 5. KV rollback

把 $C_{F_2}$ 的逻辑长度截到已接受前缀，并写入正确 token 对应的 KV。量化历史不受影响。

### 6. Buffer rotation

当 recent 区达到容量条件且本轮验证已经结束，量化稳定的 $C_{F_1}$ 并追加到 $C_U,C_L$；再把 $C_{F_2}$ 中已确认部分向前移动，为下轮候选腾出空间。

这套顺序体现了一个关键不变量：**任何进入不可逆量化历史的 KV，都必须已经由 target path 确认。**

## 分布正确性应该怎样表述

经典 speculative sampling 在接受、拒绝与修正公式都正确时，可以保持**目标路径定义的分布**。QuantSpec 的 target path 使用 INT8 hierarchical KV，而不是 FP16 KV baseline；因此需要分开两层准确性：

1. 推测算法是否严格采样自 QuantSpec target distribution；
2. INT8 KV target distribution 与原 FP16 模型有多大差异。

第一层由验证概率、修正采样和 KV rollback 保证。第二层属于量化误差，需要 perplexity、logits 和下游任务评测。论文在 Llama-2-7B 上报告 INT8 KV 的 WikiText-2/C4 perplexity 与 FP16 很接近，但这不等于对所有模型和长上下文任务都数学等价。

若工程实现为了速度又近似 target logits、跳过修正分布或让被拒绝 KV 泄漏到下一轮，就不能再声称保持目标分布。

## 为什么 custom kernel 不可缺少

把 4-bit KV 存进显存，只解决容量问题。若 attention 前先用独立 kernel 把完整 KV 解压成 FP16 workspace，HBM 又会写入并读取一份大缓存，带宽优势会大幅缩水。

有效实现要把以下操作融合或流式化：

- 读取 packed upper/lower nibbles；
- 加载 group scale 与 zero point；
- 在寄存器中反量化；
- 计算 query-key score；
- 在线 softmax；
- 与 value 累积；
- 合并 full-precision recent buffer 对应的 attention 分块。

论文将量化历史视作多个 Flash Decoding chunk，再把最长 $2G$ 的 full-precision 区作为额外 chunk，最后用 log-sum-exp 语义合并各块结果。这让双缓冲不会破坏 Flash Decoding 的分块并行方式。

论文报告自定义 kernel 在其测试配置下，相对 FP16 FlashAttention kernel 有显著加速；这是 kernel-level 对比，不能直接当成端到端生成倍数。端到端还包括权重层、draft 多轮、target verify、sampling 和 buffer 管理。

## 上下文长度改变最有效的优化

QuantSpec 的消融提供了一个比“量化总能加速”更细的结论：

- 短上下文 decode 时，加载模型权重占比高，weight quantization 更重要；
- 中等上下文中，权重与 KV 量化都能贡献收益；
- 长上下文中，KV Cache 流量成为主导，hierarchical KV 的价值更大。

这也解释了 self-speculation 为什么在长上下文特别合理。同一套量化草稿权重在各长度都能减少 weight traffic，而共享的 INT4/INT8 KV 视图避免了上下文越长、两份 cache 惩罚越大的问题。

## 如何阅读论文里的 2.49 倍

论文在 RTX A6000 节点上评测 Llama-2-7B-32K-Instruct 与 LWM-Text-Chat-128k，batch size 为 1，覆盖 PG19、Multi-LexSum 和长上下文摘要数据。报告的最高约 $2.49\times$ 是 128K 特定设置相对 autoregressive target 的结果，同时 sparse-KV baselines 在该设置 OOM。

这个数字成立于论文配置，不能直接外推到：

- 更大的 batch 或 continuous batching；
- GQA/MQA 与不同 head dimension；
- H100/Blackwell 等不同带宽和低精度指令；
- 已经使用 paged KV、prefix cache 的服务系统；
- 不同 weight quantization kernel；
- 低接受率的开放式采样或结构化输出。

更有意义的是复现完整分解，而不只复现最终 tok/s。

## 一套可定位问题的复现顺序

### 阶段一：只实现 hierarchical KV

关闭推测解码，分别运行 FP16 KV、upper+lower INT8 target view 和 upper INT4 draft view。检查：

- K/V 重构误差及异常值分布；
- 单层 attention output；
- 最终 logits KL、perplexity 与长距离检索；
- 每 token 实际 KV 字节，包括 scale、zero point 和对齐 padding。

### 阶段二：验证量化 kernel

与 PyTorch/reference dequantize 实现逐 shape 比对，覆盖不同序列长度、head dimension、group 尾部和 batch。随后记录 INT4/INT8 kernel latency、HBM 吞吐和 workspace。

### 阶段三：加入 self-draft

先不做随机接受，只记录相同 prefix 下 draft 与 target 的 top-1 一致率、概率 KL 和接受长度分布。调整 $\gamma$ 时同时观察 draft 成本与 wasted verify work。

### 阶段四：加入事务式 KV

为这些情况建立测试：

- 第一个候选就拒绝；
- 中间候选拒绝；
- 全部接受并追加 extra token；
- $C_{F_2}$ 恰好跨过容量边界；
- 客户端取消发生在 draft 与 verify 之间；
- batch 中不同请求接受长度不同。

每轮之后，都应比较“增量 cache 继续生成”与“从已确认 token 重新 prefill”的 logits。二者不一致，通常意味着长度、position id 或回滚边界出错。

### 阶段五：端到端服务评测

报告：

- acceptance rate 与 accepted tokens/target step；
- draft、verify、sampling、rollback、quantize/rotate 各阶段耗时；
- TTFT、TPOT、tok/s 和 goodput；
- 峰值显存及 batch capacity；
- 从 4K 到目标最大长度的性能曲线；
- greedy、低温和高温采样的差异。

## 落地到现有推理框架的边界

QuantSpec 论文描述的是研究系统，不应把示意性的 `QuantSpecConfig` 写成 vLLM 或 SGLang 已存在的稳定接口。真正接入 paged serving engine，需要扩展的不只有 quantization config：

- 一个 logical KV block 要关联 upper、lower 与 recent buffer 状态；
- block allocator 必须知道两种 packed plane 的物理地址；
- scheduler 要为 draft/verify 预留 lookahead slots；
- batch 中每个请求有独立的 speculative frontier；
- CUDA graph 要覆盖不同 $\gamma$ 和 rollback 长度，或建立有限 shape bucket；
- prefix cache 命中条目需要兼容量化 scale 与模型配置；
- tensor parallel 下 cache plane 与验证结果必须一致提交。

若上游框架没有原生 hierarchical KV backend 和事务式 buffer，这仍然是自定义 model runner、attention backend 与 scheduler 的联合工程，而不是一个命令行参数。

## 小结

QuantSpec 的设计可以归结为三个相互依赖的选择：

1. 同一模型用 4-bit 权重路径生成草稿，避免小模型长上下文能力不足；
2. KV 以 upper/lower 两个 INT4 位平面存储，draft 读 upper，target 读两者恢复 INT8；
3. 最近与未确认 token 留在双 full-precision buffer，验证后才量化进入稳定历史。

Hierarchical cache 节省了 draft/target 重复存储，double buffer 则为接受、拒绝、修正和量化提供了清晰的提交边界。二者缺一，QuantSpec 都容易退化成“省了 cache，却在反量化、回滚或低接受率上把时间花回来”的系统。

## 参考资料

- [QuantSpec: Self-Speculative Decoding with Hierarchical Quantized KV Cache](https://arxiv.org/html/2502.10424)
- [Fast Inference from Transformers via Speculative Decoding](https://proceedings.mlr.press/v202/leviathan23a.html)
- [Speculative Decoding with Big Little Decoder](https://arxiv.org/abs/2302.07863)
- [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache](https://arxiv.org/abs/2402.02750)
- [Flash-Decoding for Long-Context Inference](https://crfm.stanford.edu/2023/10/12/flashdecoding.html)
