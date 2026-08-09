---
layout: post
title: "PagedAttention 与 vLLM KV Cache 管理"
subtitle: "逻辑块、物理块与调度器协同"
date: 2026-03-17
last_modified_at: 2026-08-09
author: iStar
catalog: true
mathjax: true
tags: [AI Infra, LLM推理, KV Cache, vLLM]
---

## 1. 连续缓存的问题

请求长度事先未知。若为最大长度预留连续 KV Cache，会产生内部浪费；频繁申请不同长度连续区间又会导致外部碎片和搬移。PagedAttention 把每个序列的逻辑 KV blocks 映射到可不连续的物理 blocks，思路类似虚拟内存分页。

```text
logical blocks:  [0] [1] [2]
block table:      7  19   3
physical pool:  ...[3]...[7].........[19]...
```

Attention kernel 通过 block table 找到历史 K/V。最后一个块仍可能有少量内部空闲，因此准确说法是“显著降低浪费”，不是绝对零浪费。

## 2. 调度器与缓存管理器

调度器决定本轮处理哪些 token，KV Cache manager 分配/释放 blocks，并在容量不足时触发等待、抢占或重计算策略。只实现分页布局而没有调度协同，无法得到 vLLM 的整体吞吐收益。

## 3. Prefix Caching 与 Copy-on-Write

多个请求可共享相同完整前缀 blocks。共享块应只读；分支写入时使用新的物理块或 copy-on-write 语义。缓存键必须包含影响 KV 的 token 与模型上下文，避免错误复用。

## 4. 性能结论的边界

PagedAttention 论文报告了特定模型、硬件和 workload 下的吞吐收益，但“2–4×”不是对所有版本的保证。现代 vLLM 的性能还来自 continuous batching、优化 kernels、编译、prefix caching 与分布式执行。

## 5. 实践建议

- 用官方 CLI/API，避免依赖内部 block manager 类；
- 监控 GPU KV cache usage、preemption、queue、TTFT 与 TPOT；
- 调整最大上下文和显存利用率时保留 OOM 安全余量；
- prefix-heavy 与无共享前缀流量分开评测；
- 升级 vLLM 时阅读 release notes，因为 cache manager 与 backend 会持续演进。

## 6. Block 分配生命周期

请求到达时，调度器先为 prompt 所需的逻辑块分配物理块。生成 token 填满当前块后再申请新块；请求结束、取消或被完全回收时归还块。一个简化流程是：

```text
schedule(request, token_budget)
  -> reserve blocks
  -> build slot mapping / attention metadata
  -> model forward writes new K/V into selected slots
  -> update sequence length
  -> free blocks on finish or eviction
```

block size 过大时最后一块浪费增加，过小时 block table、metadata 和 kernel 地址计算开销增加，因此它是工程折中而非越小越好。

## 7. 抢占与重计算

物理块不足时，系统可以让请求等待、抢占低优先请求，或丢弃部分 KV 后在恢复时重新 prefill。swap/offload 还能把缓存移到 CPU 或其他层级，但受 PCIe/网络带宽限制。调度策略应比较“保留缓存的容量成本”和“重新计算的时间成本”。

## 8. Prefix Cache 的键

前缀缓存通常按完整 token block 的哈希建立索引。为了正确性，键还要区分影响 KV 的模型/adapter、位置与多模态输入等信息。只有完整块容易安全共享；末尾不完整块仍可能属于单个请求。

## 9. 小型实验

可用同一模型构造两组流量：一组共享长 system prompt，一组 prompt 完全随机。逐步提高并发，记录 KV cache usage、prefix hit、preemption、TTFT 和 TPOT。这个实验能把分页分配的容量收益与前缀复用收益分开观察。

## 参考资料

- [vLLM/PagedAttention paper](https://arxiv.org/abs/2309.06180)
- [vLLM documentation](https://docs.vllm.ai/)
- [vLLM official repository](https://github.com/vllm-project/vllm)
