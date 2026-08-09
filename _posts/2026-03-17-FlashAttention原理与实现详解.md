---
layout: post
title: "FlashAttention：IO-aware 的精确注意力"
subtitle: "分块、在线 Softmax 与复杂度边界"
date: 2026-03-17
last_modified_at: 2026-08-09
author: iStar
catalog: true
mathjax: true
tags: [AI Infra, LLM推理, 注意力机制]
---

## 1. 优化目标

标准注意力为

$$O=\operatorname{softmax}(QK^T/\sqrt d)V$$

朴素实现会把 $N\times N$ 的 score/probability 矩阵写入 HBM。FlashAttention 的关键不是近似或稀疏化，而是通过 tiling 与重计算减少 HBM↔片上存储的读写，同时得到数学上精确的注意力结果（有限精度舍入次序可能不同）。

## 2. 在线 Softmax

对每行分块处理时，需要维护当前最大值 $m$ 和指数和 $\ell$。新块最大值为 $m_b$、指数和为 $\ell_b$ 时：

$$m'=\max(m,m_b)$$

$$\ell'=e^{m-m'}\ell+e^{m_b-m'}\ell_b$$

旧的部分输出也按 $e^{m-m'}$ 重标定。这样无需保存完整注意力矩阵，就能稳定地合并多个块。

## 3. 复杂度不要混淆

对 dense attention，FlashAttention 的算术复杂度仍为 $O(N^2d)$；它降低的是额外内存占用与 IO 复杂度。标题或总结不能写成把 dense attention 变成线性时间。

## 4. FlashAttention-2/3 的方向

后续版本通过更好的 work partition、减少非矩阵乘法开销、适配 Hopper 异步能力与低精度路径继续优化。可用版本受 GPU 架构、CUDA、PyTorch、dtype、head dimension 和 mask 类型限制。

## 5. 使用建议

- 优先使用 PyTorch SDPA 或上层框架的 backend dispatch，再按需直接调用包 API。
- 检查实际选择的 kernel；不支持的 shape 可能回退。
- 分别测 prefill 与 decode。长序列 prefill 更容易体现 IO 优势，单 token decode 的瓶颈不同。
- 比较输出误差、峰值显存和端到端时间，不只测一个 kernel。

## 6. 分块计算过程

设 $Q$ 按行分块、$K,V$ 按列分块。对一个 query block，算法遍历所有 KV blocks，并维护每行的最大值 $m_i$、归一化因子 $\ell_i$ 与未归一化输出累积 $o_i$：

```text
for each Q block:
    initialize m = -inf, l = 0, o = 0
    for each K/V block:
        s = Q_block @ K_block.T * scale
        m_new = max(m, rowmax(s))
        p = exp(s - m_new)
        l_new = exp(m - m_new) * l + rowsum(p)
        o = exp(m - m_new) * o + p @ V_block
        m, l = m_new, l_new
    O_block = o / l
```

真实 kernel 会融合 mask、dropout、位置偏置和数据搬运，并根据共享内存/寄存器容量选择 tile。伪代码的价值是说明：只要保存每行少量统计量，就能逐块合并 softmax。

## 7. 反向传播为什么省显存

普通实现为反向传播保存完整 attention probability。FlashAttention 保存输出和 softmax 归一化统计量，在 backward 中重新计算局部 score/probability。这样用额外计算换取更少 HBM 写入和更低峰值显存。这里的 recomputation 是设计的一部分，不是实现缺陷。

## 8. Prefill 与 Decode 的差异

Prefill 同时有多个 query token，矩阵形状较大，更容易发挥 Tensor Core 和 tiling 优势。Decode 通常每个序列只有一个新 query，却要读取全部历史 KV；瓶颈更偏向缓存带宽、分页布局和 batch 组织。因此 serving 框架可能为 prefill 与 decode 选择不同 kernel/backend。

## 9. 数值与正确性检查

在小尺寸上以 FP32 朴素 attention 为 reference，分别测试 causal/non-causal、不同长度、head dimension、GQA 和 mask。比较最大绝对误差与相对误差，并对全 mask 行、极大 logits、非对齐长度和 dropout 做边界测试。低精度结果不应要求逐 bit 相等，但必须满足设定的误差预算。

## 参考资料

- [FlashAttention paper](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2 paper](https://arxiv.org/abs/2307.08691)
- [FlashAttention official repository](https://github.com/Dao-AILab/flash-attention)
