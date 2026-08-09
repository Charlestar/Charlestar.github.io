---
layout: post
title: "FairBatching：面向 LLM 推理的公平批次形成"
subtitle: "Prefill/Decode 竞争、SLO 与调度边界"
date: 2026-05-17 12:00:00 +0800
last_modified_at: 2026-08-09
author: iStar
catalog: true
mathjax: true
tags: [AI Infra, LLM推理, 调度]
---

> **校订说明**：原文把未注明出处的延迟表、p 值和 vLLM 伪实现写成论文结果。现改为可核验的论文机制概述；具体数字请直接查看论文实验表。

## 1. 问题：两个阶段争夺批次预算

Prefill 通常计算密集，并直接影响 TTFT；decode 每轮 token 少但需要持续被调度，影响 TPOT。只强调 decode 的“无停顿”可能让新请求等待过久，只强调 prefill 又会破坏在途请求的流式体验。

FairBatching 将问题定义为 batch formation：在有限 token/计算预算下选择本轮 prefill 与 decode 工作，同时考虑请求等待时间和服务目标，而不是只最大化瞬时 GPU 利用率。

## 2. 核心思想

- 持续跟踪请求距离其延迟目标还有多少余量；
- 根据当前队列和 SLO 风险确定可容纳的工作量；
- 在批次形成时显式平衡 prefill 与 decode，而不是依赖固定优先级。

“公平”并不只等于平均分配 GPU 时间。不同请求长度、到达时间和 SLO 不同，调度器需要定义可解释的公平目标。

## 3. 集成到推理引擎时

论文级伪代码不能直接粘贴进 vLLM scheduler。真实集成还要处理：

- chunked prefill 与 token budget；
- KV Cache block 分配、抢占与恢复；
- speculative decode 的 lookahead slots；
- 多租户优先级和取消请求；
- 调度计算自身的 CPU 开销。

## 4. 评测方法

至少同时报告 TTFT/TPOT 的 P50、P95、P99，SLO 违规率、goodput、吞吐与 GPU 利用率。公平性改善可能牺牲少量峰值吞吐；是否值得由业务 SLO 决定，不能只比较一张平均延迟表。

## 5. 一个调度例子

队列中有一个等待很久的长 prompt 和多个正在 decode 的请求。decode-first 会持续给后者分配 token，长 prompt 的 TTFT 风险不断增大；一次性执行完整 prefill 又可能让所有 decode 的 TPOT 抖动。折中方案是把 prefill 切成 chunk，并根据两类请求距离 SLO 的余量动态决定 chunk 大小。

## 6. Envelope/SLO 跟踪的直觉

对请求 $r$，可维护随等待/服务进度变化的 slack：

$$slack_r = deadline_r - predicted\_finish_r$$

slack 越小，越需要被调度。实际论文算法比这个表达更细致，但直觉是把“已经等了多久”和“继续等待会否违规”显式进入 batch formation，而不是固定优先级。

## 7. Goodput 比吞吐更适合 SLO

吞吐统计所有完成 token；goodput 只统计满足服务目标的请求/输出。一个系统可能 tok/s 很高，却因大量请求超时而 goodput 很低。公平调度的价值应通过 goodput、违规率和尾延迟体现。

## 8. 实现开销

预测完成时间、维护优先队列和尝试多个 batch 组合都会消耗 CPU。调度算法必须限制搜索复杂度，并验证 scheduler time 没有抵消 GPU 侧收益。预测模型失准时还需安全退化到简单策略。

## 参考资料

- [FairBatching paper](https://arxiv.org/abs/2510.14392)
- [Sarathi-Serve paper](https://arxiv.org/abs/2403.02310)
- [vLLM scheduler source](https://github.com/vllm-project/vllm/tree/main/vllm/v1/core)
