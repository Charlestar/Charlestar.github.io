---
layout: post
title: "vLLM Model Runner V2：GPU-native 与 async-first 的执行核心"
subtitle: "设计目标、迁移状态与性能验证"
date: 2026-05-25 12:00:00 +0800
last_modified_at: 2026-08-09
author: iStar
catalog: true
tags: [AI Infra, vLLM, LLM推理, GPU优化]
---

> **校订说明**：原文虚构了多组模型吞吐、内存碎片率和匿名电商案例，并把实验性状态写成完全兼容，现已删除。本文按 vLLM 官方公告与发布说明更新。

## 1. 为什么重写 Model Runner

Model Runner V2（MRV2）是 vLLM 对 GPU 执行路径的重新实现，不是 “vLLM V2”。官方总结的三项原则是：

- **modular**：分离模型特有逻辑与公共执行路径；
- **GPU-native**：把适合的 bookkeeping 和输入准备移向 GPU；
- **async-first**：从设计上支持 CPU/GPU 重叠，而不是事后叠加异步逻辑。

它还重新梳理 persistent batch 的状态所有权，降低请求插入、删除和重排时的耦合。

## 2. “零同步”应如何理解

目标是减少热路径上不必要的 GPU→CPU 同步点，让调度、输入准备、采样与下一轮执行尽可能重叠；它不意味着整个系统永远没有同步。日志、返回 token、动态控制流和不支持的算子仍可能形成同步边界。

## 3. 当前迁移状态

MRV2 最初通过 `VLLM_USE_V2_MODEL_RUNNER=1` 试用，之后逐步覆盖更多模型与功能。到 vLLM v0.25.0，官方发布说明称它已成为所有 dense 模型的默认路径；MoE、混合架构或特殊功能仍可能使用不同路径或回退策略。

因此部署文档不要固定写“v0.20+ 一律手动开启”或“API 完全兼容”。正确做法是：

1. 查看目标版本 release notes 和支持矩阵；
2. 记录实际选中的 runner，而不是只看环境变量；
3. 使用同一模型、同一 workload 比较 TTFT、TPOT、吞吐和显存；
4. 对 speculative decoding、LoRA、多模态、量化和分布式组合单独回归。

## 4. 性能数字如何阅读

官方文章展示了部分平台和工作负载上的提升，但不能把某张图的最大值复制成所有模型的固定 “56%”。MRV2 主要减少 CPU bookkeeping 和同步开销，因此小模型、高并发、短步长 workload 往往更敏感；大模型计算占主导时，收益比例可能不同。

## 5. Persistent batch 为什么复杂

相邻 decode step 的 batch 大部分请求不变，重建所有输入浪费 CPU 时间，因此 runner 会维护持久状态，只增量处理新增、完成和重排请求。状态通常包括 token IDs、positions、slot mapping、block table、sampling metadata 等。

V1 的困难在于这些结构既充当长期状态又直接充当某些 kernel 输入，布局变化容易牵动多个功能。MRV2 用更明确的状态层和输入准备层隔离这种耦合。

## 6. GPU-native input preparation

“GPU-native”不是把所有 Python 逻辑机械搬进 CUDA，而是识别适合批量执行的索引、拷贝和 metadata 更新，用 GPU/Triton kernel 处理，减少大量小 CPU 操作和 H2D 传输。复杂控制面仍可留在 CPU。

## 7. Async-first 的依赖图

若 step $t+1$ 的准备只依赖 step $t$ 的少量采样结果，就可以让 GPU 上的后处理产生紧凑结果，再通过 event/stream 串联下一步，而不把完整 tensor 同步回 CPU。遇到动态停止、logprobs 或 unsupported feature 时可能需要不同路径。

## 8. 回归矩阵

除了性能，MRV2 迁移应验证 greedy/sampling、logprobs、beam/speculative decoding、prefix cache、LoRA、量化、多模态、TP/PP 和 CUDA Graph。任何 fallback 都应出现在日志/指标中，避免把 MRV1 结果误认为 MRV2。

## 参考资料

- [vLLM 官方：Model Runner V2](https://github.com/vllm-project/vllm-project.github.io/blob/main/_posts/2026-03-24-mrv2.md)
- [vLLM releases](https://github.com/vllm-project/vllm/releases)
- [vLLM 官方仓库](https://github.com/vllm-project/vllm)
