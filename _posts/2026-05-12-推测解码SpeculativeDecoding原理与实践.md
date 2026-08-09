---
layout: post
title: "推测解码：从拒绝采样到工程实践"
subtitle: "何时保持目标分布，何时真正获得加速"
date: 2026-05-12
last_modified_at: 2026-08-09
author: iStar
catalog: true
mathjax: true
tags: [AI Infra, 推测解码, LLM推理]
---

## 1. 基本流程

草稿模型 $q$ 先提出 $k$ 个 token，目标模型 $p$ 用一次并行前向计算这些位置的概率。对第 $i$ 个候选 $x_i$，经典随机采样版本按

$$a_i=\min\left(1,\frac{p(x_i)}{q(x_i)}\right)$$

接受；首次拒绝时从归一化的 $\max(0,p-q)$ 修正分布采样，然后停止本轮验证。若全部接受，还可从目标模型的下一个位置再采一个 token。

这个修正步骤是“保持目标分布”的关键。省略它、只比较 argmax，或者让验证路径使用近似概率，都不能直接声称与目标模型采样分布完全一致。

## 2. 加速条件

推测解码不会减少目标模型参数量。它把多个位置合并到一次验证中，利用 GPU 并行性摊薄串行解码开销。是否加速取决于：

- 草稿生成成本；
- 平均接受长度；
- 目标模型验证多个位置的效率；
- batch、上下文、量化与采样参数；
- 额外 KV Cache 和回滚成本。

因此不存在通用 “2–6 倍”。低接受率、高并发或草稿过大时甚至可能变慢。

## 3. 方法家族

- 独立小模型：概念清晰，但需额外权重与 KV Cache。
- Medusa/MTP：从目标模型特征预测多个候选。
- EAGLE：以目标模型特征作为草稿输入。
- n-gram / prompt lookup：重复文本场景成本低，但适用范围窄。

这些方法的训练要求、正确性保证和运行时支持不同，不能只按论文最大加速比排序。

## 4. vLLM 实践

vLLM 的配置项随版本演进，应以目标版本 `speculative_config` 文档为准。上线前验证：greedy token 一致性或 sampling 统计正确性、acceptance 指标、TTFT/TPOT、显存、prefix caching 组合以及 structured output/tool call 边界。

## 5. 修正分布为什么成立

对候选 token $x$，草稿分布为 $q(x)$、目标分布为 $p(x)$。接受候选产生的概率质量是：

$$q(x)\min(1,p(x)/q(x))=\min(q(x),p(x))$$

目标分布尚缺的质量为 $p(x)-\min(q(x),p(x))=\max(0,p(x)-q(x))$。首次拒绝后从归一化的正残差采样，正好补齐缺失质量。这也是为什么不能简单在拒绝后直接从 $p$ 重采样：那会重复计算已经由接受路径覆盖的概率质量。

## 6. Greedy 验证

温度为 0 时通常采用 greedy speculative decoding：只要草稿 token 等于目标模型在对应位置的 argmax 就继续接受，首次不同时使用目标 argmax。此时目标是与未加速 greedy 解码 token 级一致，不需要随机拒绝采样。

## 7. 简化成本模型

设每轮草稿 $k$ 个 token，平均接受 $a$ 个，草稿时间 $T_d(k)$，验证时间 $T_v(k)$。忽略额外 token 时，每个已接受 token 的时间近似：

$$\operatorname{TPOT}_{spec}\approx\frac{T_d(k)+T_v(k)}{a}$$

当增加 $k$ 导致 $a$ 增长变慢、验证变宽或草稿成本过高时，最优 $k$ 会下降。线上自适应应依据实测的 $a,T_d,T_v$，而非只看 acceptance rate。

## 8. KV Cache 状态机

目标模型一次为候选序列产生多个 KV slots。接受前缀后，已接受 slots 变为正式状态；拒绝位置及其后的 slots 必须释放/覆盖。草稿模型也要回到一致位置。停止 token、最大长度和 grammar 状态必须按接受结果推进，否则可能出现多生成或缓存错位。

## 9. 可复现实验

在固定 checkpoint 上选择代码、对话、摘要三类 prompt，分别测试 greedy 与 temperature sampling。扫描 draft length，报告平均接受长度、draft/verify 时间、TPOT、吞吐和显存，并与完全相同参数的无推测基线比较。

## 参考资料

- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
- [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318)
- [vLLM speculative decoding documentation](https://docs.vllm.ai/en/latest/features/spec_decode/)
- [EAGLE](https://arxiv.org/abs/2401.15077)
