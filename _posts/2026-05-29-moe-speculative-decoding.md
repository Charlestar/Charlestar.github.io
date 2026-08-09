---
layout: post
title: "MoE 与推测解码：计算、通信和接受率的联合优化"
subtitle: "为什么目标模型更贵不等于一定更容易加速"
date: 2026-05-29 12:00:00 +0800
last_modified_at: 2026-08-09
author: iStar
catalog: true
mathjax: true
tags: [AI Infra, MoE, 推测解码, LLM推理]
---

> **校订说明**：原文重复了整段“最佳实践”，并把多个研究构想、vLLM/SGLang 配置和精确收益写成已验证生产能力。以下版本删除这些断言。

## 1. 为什么这个组合值得研究

推测解码减少目标模型串行执行的轮数。MoE 目标模型的每轮验证还包含路由、专家 GEMM 和 all-to-all，因此一次成功验证多个 token 可能摊薄权重读取与通信启动成本。

但“MoE 更贵”不自动推出“加速比更高”：草稿 token 可能触发不同专家，验证宽度会改变 grouped GEMM shape 和通信量，低接受率还会浪费这些工作。

## 2. 成本模型

可用一个简化模型判断是否值得开启：

$$T_{spec}=T_{draft}(k)+T_{verify}(k)+T_{rollback}$$

只有当每轮平均接受 token 数带来的串行步数减少，超过草稿、宽验证和回滚成本时，端到端才会加速。MoE 场景还应把 dispatch/combine 通信计入 $T_{verify}$。

## 3. 可优化的环节

- 根据近期接受长度动态调整 draft length；
- 让验证 batch 的 token 排布更适合 grouped GEMM；
- 重叠专家通信与可独立执行的计算；
- 对草稿路径使用轻量模型、MTP 头或特征预测器；
- 正确回滚 KV Cache、路由 metadata 和随机采样状态。

“限制验证时激活的专家”可能改变目标模型分布，除非方法给出严格修正或明确接受近似，否则不能与 lossless speculative sampling 混为一谈。

## 4. 评测清单

- accepted tokens/step 与 draft/verify 时间；
- 专家负载分布、all-to-all 时间和通信字节；
- greedy 一致性或 sampling 分布正确性；
- TTFT、TPOT、吞吐、显存和 goodput；
- batch、输入/输出长度及硬件拓扑的敏感性。

## 5. 验证树如何影响专家负载

一次验证多个候选 token 会把更多 token 同时送入 router。优点是每个 expert 的局部 batch 可能变大，grouped GEMM 更高效；缺点是候选分支可能分散到更多 experts，增加 dispatch 和 all-to-all。收益取决于草稿树形状和路由分布。

## 6. Target efficiency

可把目标模型有效工作定义为“最终保留 token / 目标验证成本”。只看 accepted token 数会忽略验证了多少最终被丢弃的分支。MoE 场景还可记录每个保留 token 对应的 expert token executions 和通信字节。

## 7. 通信重叠机会

若多个验证 microbatch 独立，可尝试在一组 experts 计算时 dispatch 下一组。但 dependency、buffer ownership 和 collective 顺序必须一致，尤其在多 rank 上不能让不同 rank 以不同顺序发起 collective。

## 8. 近似专家预算的风险

为了省成本而少激活目标模型原本选中的 expert，会改变 logits，经典接受/拒绝证明不再直接成立。此类方法应标为 approximate decoding，并通过质量指标与分布差异评估；不能与严格 lossless 方法混在同一表中。

## 9. 实验分解

先在单 GPU/单节点上验证解码正确性和接受率，再启用 EP 测通信；随后比较无推测、密集/小模型草稿、EAGLE/MTP 等方案。把 kernel、通信和调度时间分别 profile，才能判断瓶颈是否真的被摊薄。

## 参考资料

- [Speculative Decoding paper](https://arxiv.org/abs/2211.17192)
- [EAGLE paper](https://arxiv.org/abs/2401.15077)
- [DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3)
- [vLLM speculative decoding documentation](https://docs.vllm.ai/en/latest/features/spec_decode/)
