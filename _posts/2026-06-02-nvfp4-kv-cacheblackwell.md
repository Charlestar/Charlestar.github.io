---
layout: post
title: "NVFP4 KV Cache：Blackwell 上的 4-bit 缓存量化"
subtitle: "格式、收益边界与部署校验"
date: 2026-06-02 12:00:00 +0800
last_modified_at: 2026-08-09
author: iStar
catalog: true
mathjax: true
tags: [AI Infra, LLM推理, KV Cache, 量化]
---

> **校订说明**：原文包含无法追溯的公司案例、成本数字和自制性能表，现已删除。本文只保留 NVIDIA 与 vLLM 一手资料能够支持的结论；所有收益都必须在具体模型、上下文分布和硬件上复测。

## 1. 为什么量化 KV Cache

自回归解码会为历史 token 保存每层的 Key 和 Value。忽略对齐和元数据时，缓存大小可近似写成：

$$M \approx 2 \times L \times N \times H_{kv} \times D \times B$$

其中 $L$ 是层数，$N$ 是已缓存 token 数，$H_{kv}$ 是 KV 头数，$D$ 是头维度，$B$ 是每个元素的字节数。量化降低 $B$，因此主要改善容量与内存带宽压力；它不会自动让所有算子获得同等比例的加速。

## 2. NVFP4 到底是什么

NVFP4 使用 E2M1 的 4-bit 数据值，并采用两级缩放：每 16 个值共享一个 E4M3 FP8 微块缩放因子，张量再共享一个 FP32 缩放因子。计入缩放开销后，它不是“严格每元素 4 bit”，但仍明显小于 FP8。

NVIDIA 公布的 KV Cache 方案把缓存从 FP8 压缩到 NVFP4，官方测试中缓存占用约减少 50%，并在所测代码、知识与长上下文基准上报告了小于 1% 的精度差异。这里的比较基线是 **FP8 KV Cache**，不能误写成相对 FP16 只减少 50%。

## 3. 收益边界

- **容量**：相同缓存预算可容纳更多 token 或请求，但实际倍数受页大小、元数据和模型结构影响。
- **带宽**：decode 阶段读取的缓存更小，可能改善 memory-bound 工作负载。
- **计算**：缓存通常需要在注意力计算前反量化；端到端收益取决于内核实现和缓存命中率。
- **硬件**：NVFP4 的主要目标是 NVIDIA Blackwell。其他 GPU 不应默认具有相同支持和收益。
- **质量**：官方结果不是对所有模型与任务的保证，特别是长链推理、代码和检索任务必须单独验证。

## 4. 部署检查表

1. 先固定模型版本、数据集、采样参数和随机种子，记录 BF16/FP8 基线。
2. 同时比较 TTFT、TPOT、吞吐、峰值显存和任务质量，不只比较缓存字节数。
3. 分开测试短上下文、长上下文、低并发和高并发；量化收益通常随负载变化。
4. 确认推理框架、attention backend 和模型结构都支持 NVFP4；不支持时应显式失败而非静默回退。
5. 灰度发布并保留回退到 FP8/BF16 KV Cache 的能力。

## 5. 两级缩放展开

对每个 16-value 微块，先选择 FP8 E4M3 scale，把值映射到 E2M1 可表示范围；整个 tensor 还有 FP32 scale 负责更大范围校准。概念上：

```text
x
 -> divide by tensor_scale
 -> divide by block_scale (one per 16 values)
 -> round/clip to E2M1
 -> pack two 4-bit values per byte
```

读取时反向应用 scales，并通常转换到 attention kernel 使用的更高精度。两级 scale 改善局部动态范围，但引入 metadata 和量化/反量化工作。

## 6. 为什么不是 75% 实际节省

从 16-bit 数值到裸 4-bit 看似减少 75%，但 NVFP4 还需每 16 个值一个 FP8 scale 和每 tensor 一个 FP32 scale，并有对齐/页 metadata。相对 FP8 的官方 KV 方案约节省 50% 更符合实际对比。相对 BF16 的比例需按具体布局计算，不能只用 4/16。

## 7. Prefill 与 Decode 的不同影响

Prefill 计算密集，KV 写入只是一部分；decode 每步反复读取历史 KV，更容易受缓存带宽影响。NVFP4 可能在长上下文 decode 和高缓存压力下更有价值，但反量化 kernel、缓存命中和 batch shape 会决定实际 TPOT。

## 8. 与 GQA/MLA 叠加

GQA 通过减少 KV heads 降低 token cache，MLA 用低维 latent 表示压缩 KV，NVFP4 再降低存储精度。这些优化可以叠加，但基线已经很小时，量化 metadata/反量化成本占比会提高；质量敏感性也依表示而变。

## 9. 质量校准

校准集应覆盖不同层、head、上下文位置和输入领域。除平均误差外，观察 scale saturation、异常值比例和 attention/logit 偏差。部署验证使用与业务相同的长上下文任务，而不是只跑短问答。

## 10. 容量计算

先根据模型配置计算未量化 KV bytes/token，再乘目标并发和上下文分布；随后加入 block 对齐、scale metadata、prefix cache、CUDA Graph 和权重占用。只有剩余 HBM 才是可分配 KV 池，不能用 GPU 总显存直接除。

## 参考资料

- [NVIDIA：Optimizing Inference for Long Context and Large Batch Sizes with NVFP4 KV Cache](https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache/)
- [NVIDIA：Introducing NVFP4 for Efficient and Accurate Low-Precision Inference](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
- [vLLM KV cache interface](https://docs.vllm.ai/en/latest/api/vllm/v1/kv_cache_interface/)
