---
layout: post
title: "llm-d：Kubernetes 原生分布式 LLM 推理栈"
subtitle: "路由、KV Cache 与 Prefill/Decode 解耦"
date: 2026-06-01 12:00:00 +0800
last_modified_at: 2026-08-09
author: iStar
catalog: true
tags: [AI Infra, Kubernetes, llm-d, 分布式推理]
---

> **校订说明**：原文把概念性 YAML、Redis 任务队列和匿名金融公司数据写成 llm-d 的既成实现，缺少来源且会误导部署，现已移除。本文依据当前官方架构重新整理。

## 1. llm-d 的定位

llm-d 不是替代 vLLM 的单机推理引擎，而是位于模型服务器之上的 Kubernetes 分布式服务栈。它组合 vLLM、Kubernetes Gateway API/Inference Gateway 及 KV 传输组件，解决大规模服务中的路由、缓存、解耦部署和运维问题。

官方当前将能力概括为：

- 前缀缓存与负载感知的智能路由；
- 分层 KV Cache 管理与全局缓存索引；
- Prefill/Decode 解耦和宽专家并行；
- 流量控制、SLO 感知扩缩容与批处理。

## 2. 核心数据路径

```text
client
  -> Gateway / llm-d Router
  -> selected model-server endpoint (vLLM)
  -> optional disaggregated prefill and decode workers
  -> KV transfer/offload layer when the chosen recipe enables it
```

路由器可以利用队列、负载和前缀缓存信息选择端点。Prefill/Decode 解耦适合两个阶段资源特征不同的工作负载，但会引入 KV 传输、故障恢复和容量规划成本，并非默认一定更快。

## 3. 不应混淆的概念

- **Prefix cache** 是模型服务器可复用的计算状态；它不等同于 Redis 中的普通业务对象。
- **KV 传输** 需要面向大张量的高性能数据路径；官方方案会组合 NIXL 等组件，不能用示例消息队列替代。
- **Kubernetes 调度** 决定 Pod 放置，llm-d Router 负责请求级路由；两者职责不同。
- **“多硬件支持”** 代表项目提供多种 recipe 和集成，不代表任意模型、任意加速器无需验证即可互换。

## 4. 上线前的验证顺序

1. 先用单一 vLLM 实例建立质量、TTFT、TPOT 和吞吐基线。
2. 再测试 cache-aware routing，记录缓存命中率和尾延迟。
3. 只有 prefill/decode 比例和网络条件合适时再引入解耦。
4. 使用官方 Helm chart 与 well-lit-path 指南，不复制版本不明的概念 YAML。
5. 对路由器、模型服务器和 KV 数据路径分别设置容量与故障演练。

官方展示的性能结果都绑定具体模型、硬件和流量分布，不能直接外推为通用的“成本降低 35%”或“QPS 提升 54%”。

## 5. Router 的决策信息

普通 round-robin 不知道请求长度、队列、KV 命中或各副本 GPU 状态。llm-d Router 的 Endpoint Picker 可以利用推理相关信号给候选 endpoint 打分，再由 proxy 转发。评分仍需防止热点：所有共享同一热门前缀的请求都去一个副本，可能让缓存收益被排队延迟抵消。

## 6. Prefill/Decode 解耦数据流

```text
request -> prefill worker -> KV blocks
                       \-> transfer metadata
                           -> decode worker -> tokens
```

解耦允许两个池分别扩缩容和选择并行策略。代价是 KV 传输延迟、路由一致性和失败恢复。只有当阶段资源特征、网络带宽和请求长度分布适合时，解耦才优于共置。

## 7. 分层 KV Cache

GPU HBM 最快但最小，CPU 内存/本地盘/远端存储更大但更慢。分层缓存的目标是把热门前缀留在快层，把较冷数据下沉，并用全局索引判断数据位置。命中慢层是否值得，取决于传输成本与重新 prefill 成本的比较。

## 8. Kubernetes 资源模型

生产 recipe 还需处理 GPU 拓扑、Gang scheduling、健康探针、PodDisruptionBudget、滚动升级和多租户限流。HPA 仅看 CPU 利用率通常不足，应结合队列、TTFT/TPOT、KV 容量与模型加载时间设计扩缩容信号。

## 9. 最小实验路径

先对同一批 prompt 比较 round-robin 与 prefix-aware routing；再增加请求率观察热点和尾延迟；最后才引入 P/D 解耦并测 KV transfer。每次只改变一个维度，更容易确定收益来自路由、缓存还是资源池拆分。

## 参考资料

- [llm-d 官方仓库](https://github.com/llm-d/llm-d)
- [llm-d 官方架构文档](https://llm-d.ai/docs/architecture/)
- [llm-d founding proposal](https://github.com/llm-d/llm-d/blob/main/docs/proposals/llm-d.md)
