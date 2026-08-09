---
layout: post
title: "DeepSeek V3.2 稀疏注意力：Lightning Indexer 与 Top-k 选择"
subtitle: "复杂度、实现代价与质量边界"
date: 2026-05-18 12:00:00 +0800
last_modified_at: 2026-08-09
author: iStar
catalog: true
mathjax: true
tags: [AI Infra, LLM推理, 稀疏注意力]
---

> **校订说明**：原文对 Claude/Gemini 的内部注意力结构作了无来源断言，并包含自制硬件基准、质量表和成本数据，现已删除。

## 1. DSA 的结构

DeepSeek Sparse Attention（DSA）由两个阶段组成：

1. **Lightning Indexer** 为当前 query 与历史 token 计算轻量相关性分数；
2. **Fine-grained token selection** 选择 top-k KV 条目执行正式注意力。

当 $k$ 固定且 $k\ll n$ 时，正式注意力部分由 $O(n^2)$ 降为约 $O(nk)$。但索引、top-k、稀疏 gather 和缓存布局也有成本，所以不能直接把渐进复杂度等同于端到端加速。

## 2. Lightning Indexer

官方 V3.2-Exp 说明中，indexer 使用少量头，并可采用 FP8。概念上可写为：

$$I_{t,s}=\sum_j w_{t,j}^{I}\,\operatorname{ReLU}(q_{t,j}^{I}\cdot k_s^{I})$$

随后根据 $I_{t,:}$ 选择 top-k 历史位置。这里的 index 分数服务于候选选择，不等同于最终 attention probability。

## 3. 工程难点

- top-k 本身需要高效 kernel，长序列下不能大量物化中间张量；
- 稀疏索引要映射到分页 KV Cache 的物理块；
- gather 后的访问模式可能不连续，理论 FLOPs 下降不保证带宽效率同步提高；
- continuous batching 中不同请求长度和选择位置不同，需要 ragged metadata；
- 短上下文可能没有足够收益，应允许 dense fallback。

## 4. 质量与评测

稀疏选择会改变模型能访问的上下文。评测应覆盖 needle retrieval、长文档问答、长代码和长链推理，并与 dense attention 在相同权重/采样条件下比较。不能把某个平均分差异概括为所有任务“质量无损”。

## 5. Indexer 与正式注意力的数据流

```text
hidden state h_t
 -> lightweight query/index weights
 -> score against historical index keys
 -> top-k positions
 -> gather compressed/full KV for selected positions
 -> sparse attention
 -> output projection
```

Indexer key 可以比正式 K/V 更小，并使用更低精度，从而让扫描历史的成本低于完整 attention。最终 attention 仍使用模型定义的表示计算输出。

## 6. 复杂度拆解

令序列长度为 $n$、选择数量为 $k$。正式 attention 约为 $O(nk)$，但 index score 若对所有历史位置计算，仍有 $O(n^2d_I)$ 项，只是 $d_I$ 和头数较小、精度更低。工程收益来自“便宜的全局筛选 + 昂贵计算只做 top-k”，不能简单忽略 indexer 成本。

## 7. Top-k Kernel

长序列 top-k 需要分块选择和归并。直接生成完整 score 矩阵会破坏内存优势；更合理的是每个 tile 保留局部候选，再层级归并全局 top-k。并行实现还要稳定处理相同分数、mask 和 causal 范围。

## 8. 与 Paged KV 的结合

逻辑 token position 经 block table 转成物理 page/offset，稀疏选择结果可能非常离散。实现需要批量 gather 并尽量合并相邻位置，减少随机访存。IndexCache 一类结构可缓存 indexer 所需状态，但不等同于正式 KV Cache。

## 9. Dense fallback

当上下文不超过 $k$、top-k 管理成本高于 dense kernel，或某种 backend 不支持稀疏路径时，使用 dense attention 更合理。阈值应由 microbenchmark 决定，并按 GPU 架构调整。

## 参考资料

- [DeepSeek-V3.2-Exp 官方仓库](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp)
- [vLLM 官方仓库](https://github.com/vllm-project/vllm)
- [FlashInfer 官方仓库](https://github.com/flashinfer-ai/flashinfer)
