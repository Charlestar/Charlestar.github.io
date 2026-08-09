---
layout: post
title: "SpecForge：面向 SGLang 的推测解码模型训练框架"
subtitle: "EAGLE 系列草稿模型的训练与交付边界"
date: 2026-05-27 12:00:00 +0800
last_modified_at: 2026-08-09
author: iStar
catalog: true
tags: [AI Infra, SGLang, SpecForge, 推测解码]
---

> **校订说明**：原文包含 GPT-4/Claude 草稿模型、跨硬件吞吐表和成本/碳排放数字，官方资料并不支持这些内容，现已删除。

## 1. SpecForge 解决什么问题

SpecForge 是 SGLang 团队维护的推测解码模型训练框架。它的价值主要在工程链路：支持在线/离线数据、张量并行和 FSDP 训练，并让训练产物更顺畅地接入 SGLang serving。

推测解码包含两个不同问题：

1. **训练草稿模型**：SpecForge 主要覆盖这一环节。
2. **运行时验证与采样**：由 SGLang 等推理引擎完成。

因此“训练完成”不等于必然获得固定加速比。端到端收益取决于草稿成本、平均接受长度、目标模型验证效率、batch 大小和请求分布。

## 2. EAGLE-3 的关键思路

EAGLE 系列不是简单地用一个小语言模型独立生成 token，而是利用目标模型的特征来预测后续候选。EAGLE-3 进一步融合多层特征，旨在提高草稿对目标分布的拟合能力。

“接受率”也不能单独代表性能：更大的草稿树可能提高每轮接受 token 数，却同时增加草稿计算、验证宽度和显存占用。应同时报告：

- accepted tokens per verification step；
- draft/verify 时间占比；
- TPOT、吞吐与峰值显存；
- greedy 与 sampling 两种模式下的正确性。

## 3. 实践建议

- 从官方已发布的 checkpoint 和配置开始，不要把概念性 Python 类当作真实 API。
- 训练数据要贴近线上目标模型的输出分布和采样策略。
- checkpoint 必须与目标模型、tokenizer、特征层和 serving 版本匹配。
- 先验证输出分布/greedy 等价性，再做性能对比。
- 在真实 prompt 长度、输出长度和并发分布上评测，不外推单条样例结果。

## 4. 训练数据流水线

```text
prompts
 -> target model rollout / hidden-state capture
 -> tokenize + align target features
 -> training shards
 -> EAGLE drafter training
 -> checkpoint conversion
 -> SGLang serving validation
```

离线数据易复现，但可能与最新 drafter 分布不一致；在线/on-policy 数据更贴近当前模型行为，成本和系统复杂度更高。实际训练可混合使用，并记录生成参数与目标模型 revision。

## 5. Feature alignment

草稿训练样本中的 token position 必须和目标 hidden feature 一一对应。BOS/EOS、chat template、padding、截断和 packed sequence 都可能造成 off-by-one。最先应做的小测试，是在单条短序列上打印 token IDs、labels 和 feature indices，确认每个预测目标对齐。

## 6. 并行训练

FSDP 主要切分参数/优化器状态，Tensor Parallel 切分层内计算。训练 drafter 时还要考虑目标特征数据的读取吞吐，避免 GPU 等待大量 hidden-state 文件。shard 应按样本边界组织并支持断点续训。

## 7. Checkpoint 交付契约

产物至少要记录 base/target model、tokenizer、feature layers、drafter architecture、dtype、训练配置和导出格式。Serving 启动时应验证这些字段，而不是等到生成错误后才发现模型不匹配。

## 8. 从离线 loss 到线上收益

更低训练 loss 不一定带来更高 accepted length。最终模型选择应在代表性 prompt 上测 acceptance、draft latency、verify latency 和端到端 TPOT，并监控不同领域/语言的差异。

## 参考资料

- [SpecForge 官方仓库](https://github.com/sgl-project/SpecForge)
- [SpecForge 文档](https://docs.sglang.ai/SpecForge/)
- [EAGLE-3 论文](https://arxiv.org/abs/2503.01840)
- [SGLang 官方仓库](https://github.com/sgl-project/sglang)
