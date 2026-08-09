---
layout: post
title: "异步连续批处理：CPU/GPU 重叠与正确同步"
subtitle: "从调度流水线到 CUDA Stream"
date: 2026-05-16 12:00:00 +0800
last_modified_at: 2026-08-09
author: iStar
catalog: true
tags: [AI Infra, LLM推理, CUDA, 调度]
---

> **校订说明**：原文包含不存在的 vLLM 配置项和自制的固定 “20–30%” 收益。异步执行的收益依 CPU 开销、模型大小、batch 和同步点而变，本文不再给出无来源通用数字。

## 1. Continuous batching 与 async scheduling

Continuous batching 允许请求在生成过程中动态加入或离开 batch，减少静态 padding。Async scheduling 进一步把第 $t+1$ 步的 CPU 调度/metadata 准备与第 $t$ 步 GPU 执行重叠：

```text
time ->
GPU:   [step t compute] [step t+1 compute]
CPU:        [prepare t+1]    [prepare t+2]
```

二者是相关但不同的优化：启用 continuous batching 不代表 CPU 与 GPU 已完全重叠。

## 2. CUDA Stream 不等于自动并行

把操作放到不同 stream 只是允许并发；是否真正重叠还取决于数据依赖、GPU 资源占用、内存拷贝类型与事件同步。主机到设备的异步拷贝通常还需要 pinned memory。

正确做法是用 CUDA event 表达最小依赖，并避免热路径中的 `.item()`、隐式张量打印或同步内存拷贝。错误地省略依赖会读到尚未完成的数据，过度 `synchronize()` 又会抹掉重叠收益。

## 3. 双缓冲的边界

双缓冲可让一组输入执行时准备下一组，但会增加显存并使状态所有权更复杂。动态 batch 中请求会插入、完成或被抢占，不能简单交换两个固定 Tensor 就宣称支持完整调度语义。

## 4. 如何验证

- 用 Nsight Systems 确认时间线上存在真实重叠；
- 比较同步点数量、CPU step time、GPU idle gap、TTFT 和 TPOT；
- 覆盖请求加入/退出、取消、OOM、prefix cache 和 speculative decode；
- 在低并发与高并发分别测试，避免只报告最有利的 workload。

## 5. 三层异步不要混在一起

1. **API 异步**：网络线程不阻塞等待整段输出。
2. **调度异步**：CPU 准备下一步时 GPU 执行当前步。
3. **设备异步**：不同 CUDA streams 上的 kernel/copy 在依赖允许时并发。

只实现 `async def` 不会自动带来 GPU 重叠；只创建两个 CUDA stream 也不能解决 scheduler 的 Python 开销。

## 6. Event 驱动的双缓冲

```text
buffer A: GPU executing step t
buffer B: CPU/GPU preparing metadata for step t+1
event t_done -> B may consume sampled tokens / freed slots
swap(A, B)
```

每个 buffer 的所有权必须清楚。若请求在两步之间完成，下一 buffer 中要移除其 slot；若有新请求加入，要初始化 block table、position 和 sampling state。event 只保证执行顺序，不负责业务状态正确性。

## 7. 常见隐式同步

- 在 CPU 分支中读取 GPU tensor 的 `.item()`；
- 对 CUDA tensor 调用会触发 host copy 的操作；
- 默认 stream 与自定义 stream 缺少/多余依赖；
- allocator 或日志路径中的同步；
- 测时时忘记用 CUDA event 或显式同步，得到错误时间。

## 8. 什么时候收益小

当单步模型计算远大于 CPU 准备、batch 已足够大、或 workload 经常触发必须回 CPU 的动态功能时，重叠能隐藏的时间有限。异步设计的主要价值也可能表现为降低 GPU idle gap 和尾延迟，而非峰值吞吐大幅增长。

## 参考资料

- [CUDA C++ Programming Guide: Asynchronous Concurrent Execution](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-concurrent-execution)
- [PyTorch CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html)
- [vLLM 官方仓库](https://github.com/vllm-project/vllm)
