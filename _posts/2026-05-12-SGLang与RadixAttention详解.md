---
layout: post
title: "SGLang 与 RadixAttention：跨请求复用 KV Cache"
subtitle: "Radix tree、调度与缓存失效"
date: 2026-05-12 15:00:00 +0800
last_modified_at: 2026-08-09
author: iStar
catalog: true
mathjax: true
tags: [AI Infra, SGLang, RadixAttention, KV Cache]
---

## 1. 要解决的浪费

系统提示词、few-shot 示例和文档前缀常被多个请求重复使用。普通请求级 KV Cache 会在请求结束后释放这些状态；RadixAttention 把 token 序列组织在 radix tree 中，使不同请求共享最长公共前缀对应的 KV blocks。

```text
[system prompt]
  ├─ [document A] ─ question 1
  │              └ question 2
  └─ [document B] ─ question 3
```

命中共享前缀可以减少 prefill 计算，但不会减少每个请求自己的 decode 工作。

## 2. 缓存与调度必须协同

缓存条目占用有限 HBM，调度器需要知道哪些请求可以复用哪些前缀，并在内存压力下驱逐未引用条目。常见策略会综合可复用前缀长度、引用计数和最近使用时间；具体实现以当前 SGLang 源码为准。

## 3. 与 PagedAttention 的关系

- PagedAttention 解决单个及批量请求的 KV Cache 分块存储和内存碎片。
- RadixAttention 重点解决请求之间按 token 前缀匹配和复用。

两者关注层次不同，可以组合，不能简单写成互相替代。

## 4. 收益边界

- prompt 必须在 token 级完全共享；空格、模板或 tokenizer 差异都会降低命中。
- 低重复流量可能只增加管理开销。
- cache hit 主要改善 TTFT/prefill，长输出 workload 的端到端收益会被 decode 稀释。
- LoRA、模型参数、位置编码或其他影响 KV 的配置必须纳入缓存隔离键。

## 5. 实践

使用当前 `sglang` server CLI 和 OpenAI-compatible API，先记录 prefix cache hit、eviction、TTFT 和显存，再比较关闭缓存的基线。不要把旧版 SGLang DSL 示例当成长期稳定接口。

## 6. Radix tree 操作过程

一个新请求进入时，runtime 对 token 序列执行 longest-prefix match：

1. 已匹配节点对应的 KV blocks 增加引用；
2. 只对未命中的 suffix 执行 prefill；
3. 新产生的 token 路径插入树；
4. 请求结束后释放引用，但可缓存节点仍保留到被驱逐。

树的边可以保存一段 token，而非每个 token 一个节点，从而压缩只有单一分支的路径。节点拆分发生在新请求与已有边只共享部分 token 时。

## 7. Cache-aware 调度

若多个请求等待执行，优先选择能复用更多前缀的请求可能提高吞吐，但也可能让其他请求等待更久。因此调度器要在 cache locality 与公平/SLO 之间权衡。多副本部署时，请求路由也应考虑每个副本拥有的前缀，而不是只看队列长度。

## 8. Structured generation 的关系

结构化生成通过 grammar/regex/JSON schema 限制下一 token 集合；RadixAttention 负责 KV 复用，两者是独立能力。在 agent workload 中它们常同时出现：共享工具定义带来 prefix reuse，结构化采样保证 tool call 语法。不能把结构化输出的收益都归因于 RadixAttention。

## 9. 调试命中率

如果预期共享前缀却没有命中，依次检查 chat template、system prompt 空白符、tokenizer revision、动态时间戳/请求 ID、LoRA 和多模态占位符。应比较 token IDs，而不是只肉眼比较字符串。

## 参考资料

- [SGLang paper](https://arxiv.org/abs/2312.07104)
- [SGLang official repository](https://github.com/sgl-project/sglang)
- [SGLang documentation](https://docs.sglang.ai/)
