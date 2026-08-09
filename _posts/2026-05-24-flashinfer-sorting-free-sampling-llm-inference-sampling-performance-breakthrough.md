---
layout: post
title: "FlashInfer Sorting-Free Sampling：无需显式排序的 GPU 采样"
subtitle: "Top-k、Top-p 与拒绝采样的正确性边界"
date: 2026-05-24 12:00:00 +0800
last_modified_at: 2026-08-09
author: iStar
catalog: true
tags: [AI Infra, FlashInfer, LLM推理, Sampling]
---

> **校订说明**：原文自制了客服、翻译和代码生成的性能表，并使用了并不存在的统一配置 API。本文改为解释官方算子的统计语义与集成边界。

## 1. 为什么避免排序

Top-k/Top-p 采样的直观实现会对整个词表排序。词表很大、batch 较小时，排序和中间张量会成为可见开销。FlashInfer 提供基于 GPU 拒绝采样的算子，在不显式全排序的情况下实现 top-k、top-p 及组合过滤。

关键点是 **分布等价**，不是固定随机种子下逐 token 与另一实现完全相同。不同实现消耗随机数的顺序可能不同，因此输出样本可不同，但应服从同一目标分布。

## 2. 常用接口

当前官方 API 包括：

- `top_k_sampling_from_probs`
- `top_p_sampling_from_probs`
- `top_k_top_p_sampling_from_logits`
- `min_p_sampling_from_probs`

参数名、输入是 logits 还是 probabilities、以及 top-k/top-p 的应用顺序会随具体接口而不同，应以安装版本的文档为准。

## 3. 正确性验证

1. 检查被过滤 token 的经验概率接近零。
2. 在小词表上与可枚举的参考分布做统计检验。
3. 覆盖 $k=1$、$k=V$、$p\to0$、$p=1$ 和混合 batch 参数。
4. 分别验证 deterministic 选项和每请求 generator/seed 的行为。
5. 不把“统计等价”误写成“位级输出一致”。

## 4. 性能验证

单独测 kernel 延迟后，还要在真实 serving workload 中测端到端 TPOT。采样通常只是解码路径的一部分；当模型前向占主导时，即使采样 kernel 大幅加速，端到端收益也可能有限。

vLLM 已包含调用 FlashInfer sampling 的实现，但是否启用取决于版本、安装和采样配置。不要依赖博客中的虚构 `sampling_backend` 配置，应查看目标版本源码与文档。

## 5. Top-k 与 Top-p 的语义

Top-k 只保留概率最高的 $k$ 个 token；top-p 保留按概率从高到低累积达到阈值 $p$ 的最小集合，再归一化采样。两者组合时，先后顺序可能影响候选集合，因此 API 提供的 `filter_apply_order` 需要明确记录。

Min-p 则按当前最大概率的比例设置动态阈值，常写成保留 $p_i\ge p_{min}\max_j p_j$ 的 token。它与 top-p 的累计概率语义不同。

## 6. 拒绝采样直觉

无需排序的方法可以从原分布提出候选，再检查候选是否位于 top-k/top-p 允许集合；不满足则重试。GPU 上可并行生成随机量、估计/确定阈值并进行接受判断。最坏情况下可能多轮重试，因此实现通常设置最大轮数和 fallback。

## 7. Deterministic 参数

FlashInfer 文档中的 deterministic 通常描述算法执行/随机数使用的特定模式，不应直接理解为跨 GPU、跨版本和跨实现逐 bit 相同。需要可重放时，还要固定 generator/seed、offset、输入 logits 和软件栈。

## 8. 统计测试

构造小词表和已知概率，采样足够多次，比较经验频率与 reference 分布。除了卡方/总变差距离，还应检查被过滤 token 从未出现、概率和为 1、每请求不同 $k/p$ 以及 NaN 输入行为。

## 9. Serving 集成

动态 batch 中每个请求可有不同 top-k/top-p/seed。实现需避免为每个请求启动独立 kernel，并正确维护 generator 状态。返回 logprobs 时，过滤前还是过滤后的概率也要遵循 API 契约。

## 参考资料

- [FlashInfer sampling API](https://docs.flashinfer.ai/api/sampling.html)
- [FlashInfer 官方仓库](https://github.com/flashinfer-ai/flashinfer)
- [vLLM FlashInfer sampler implementation](https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/ops/topk_topp_sampler.py)
