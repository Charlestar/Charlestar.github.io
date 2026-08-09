---
layout: post
title: "SGLang 与 RadixAttention：跨请求复用 KV Cache"
subtitle: "从最长前缀匹配到缓存感知调度"
date: 2026-05-12 15:00:00 +0800
last_modified_at: 2026-08-09
author: iStar
catalog: true
series: kv-cache-memory
series_order: 30
technology_year: 2023
mathjax: true
tags: [AI Infra, SGLang, RadixAttention, KV Cache]
---

许多 LLM 应用会反复计算相同的 prompt 前缀。客服请求都带着同一套服务规则；RAG 用户围绕同一篇文档连续提问；Agent 的每次模型调用都可能附带相同的工具定义。对模型来说，这些前缀一旦 token 完全相同，对应的 KV Cache 也相同，重复 prefill 是可以避免的。

SGLang 的 RadixAttention 把这一现象提升为运行时的核心数据结构：用 radix tree 组织 token 序列，让请求查找最长已计算前缀，只对剩余 suffix 执行 prefill；调度器也可以利用缓存位置来安排请求。

它的价值不只是“有缓存”。普通 KV Cache 解决单条请求在 decode 中复用自己的历史，RadixAttention 关注的是 **不同请求、不同程序分支之间复用相同前缀**。

## 重复计算发生在哪里

考虑两次对同一产品手册的提问：

```text
Request A = [system rules][product manual][question A]
Request B = [system rules][product manual][question B]
```

模型处理 A 时，会为整个输入执行 prefill，并为每层生成 K/V。若 A 结束后这些状态全部释放，B 又要重新计算 `[system rules][product manual]`。

假设共同前缀有 16k token，两个问题各 100 token。没有前缀复用时，两次 prefill 约处理：

$$
(16000+100)\times2=32200\ \text{tokens}
$$

若第二次命中完整的 16k 前缀，需要新计算的输入约为：

$$
16000+100+100=16200\ \text{tokens}
$$

这不是说端到端时间必然减半。第一次请求仍要完整计算，decode 也不会因为 prompt 命中而消失；但对长输入、短输出、重复率高的服务，prefill 的节省会显著改善 TTFT 和可用算力。

## 为什么使用 radix tree

前缀缓存需要回答：给定一串 token，已经计算过的最长前缀有多长？Trie 可以逐 token 保存路径，但单分支很长时会产生大量节点。Radix tree（压缩前缀树）允许一条边保存一段 token，将没有分叉的路径压在一起。

假设已有两个请求：

```text
[SYS, DOC_A, Q1]
[SYS, DOC_A, Q2]
```

树可以表示为：

```text
root
 └─ [SYS, DOC_A]
       ├─ [Q1]
       └─ [Q2]
```

再插入 `[SYS, DOC_B, Q3]` 时，共享部分只到 `SYS`，原边需要拆分：

```text
root
 └─ [SYS]
       ├─ [DOC_A]
       │    ├─ [Q1]
       │    └─ [Q2]
       └─ [DOC_B, Q3]
```

真实边内容是 token IDs 或其块级表示，不是图中的自然语言片段。节点/边关联已计算 KV Cache 的位置、引用状态和淘汰信息。

## 一次 longest-prefix match

新请求到达后，runtime 从根开始比较 token：

1. 若当前边全部匹配，沿子节点继续；
2. 若只匹配边的一部分，最长前缀停在分歧处；
3. 已匹配部分直接引用缓存中的 KV；
4. 未匹配 suffix 进入 prefill；
5. 新计算的路径插入树，供后续请求复用。

用 token 字母表示，已有缓存：

```text
[A B C D E F]
```

新请求是：

```text
[A B C D X Y]
```

最长公共前缀是 `[A B C D]`。运行时不能复用 `E F`，也不能因为 `Y` 在别处出现过就单独复用；自回归 Transformer 的 K/V 依赖此前上下文和位置，通常只有从开头连续一致的前缀才具有相同状态。

## “文字相同”不等于“token 相同”

RadixAttention 比较的是 token IDs。下面这些细节都可能改变 tokenization：

- 多一个空格或换行；
- Unicode 形式不同；
- chat template 或 special token 不同；
- tokenizer revision 不同；
- 工具定义的顺序或 JSON 序列化不同。

肉眼看起来接近的两段文本，不一定能命中：

```text
"Hello world"
"Hello  world"
```

反过来，即使字符串来源不同，只要在相同模型上下文下产生完全相同的前缀 token IDs，就具备复用条件。

调试命中问题时，应直接比较 token IDs：

```python
ids_a = tokenizer(prompt_a, add_special_tokens=True).input_ids
ids_b = tokenizer(prompt_b, add_special_tokens=True).input_ids

prefix = 0
for a, b in zip(ids_a, ids_b):
    if a != b:
        break
    prefix += 1

print("shared prefix tokens:", prefix)
```

## Prompt 布局会决定缓存价值

考虑两种模板：

```text
Template A:
[request_id][current_time][stable system rules][tool schemas][user message]

Template B:
[stable system rules][tool schemas][current_time][request_id][user message]
```

在 A 中，请求 ID 和时间一开始就不同，最长公共前缀几乎为零；后面再长的固定内容也无法复用。B 把稳定内容放在前面，动态字段放在靠后位置，缓存命中会高得多。

因此，前缀缓存优化常从 prompt 工程开始：

- 固定 system prompt 的空格与换行；
- 对工具 schema 使用确定性排序和序列化；
- 将用户、时间和 trace ID 等动态内容后移；
- 不在固定模板中插入无意义随机数；
- 保证同一模型副本使用同一 tokenizer/chat template revision。

这不是为了迎合某个特定引擎，而是让应用中本来相同的语义结构也表现为相同 token 前缀。

## 缓存节点的生命周期

KV Cache 有限，radix tree 不能无限增长。一个路径通常会经历以下状态：

### 命中与引用

请求命中前缀后，关联缓存的引用计数增加。只要仍有运行中请求使用，就不能覆盖或淘汰这段 KV。

### 插入

未命中 suffix 完成 prefill 后，新 KV 被接到已有路径。若新旧路径只部分相同，树边会发生拆分。

### 释放引用

请求结束只意味着它不再持有节点，不一定立即删除缓存。保留已计算但无人引用的路径，未来请求才可能再次命中。

### 淘汰

当需要给新请求分配显存，runtime 从未被引用的缓存中选择牺牲者。最近是否使用、缓存长度与树结构都会影响复用价值；具体策略应以对应 SGLang 版本的实现为准。

这揭示了缓存系统的基本冲突：为未来命中保留更多路径，会占用本可接纳新请求的 KV 空间；过早淘汰又会让 prefill 重复发生。

## RadixAttention 与 PagedAttention 的关系

两者经常同时出现，却解决不同问题。

**Paged KV Cache / PagedAttention** 关注物理存储：把每条序列的 KV 分成固定块，用映射避免为最大长度预留连续空间。

**RadixAttention** 关注逻辑复用：根据 token 前缀找到不同请求可以共享的已计算 KV，并组织其生命周期。

可以把二者放在两层理解：

```text
radix tree:
  哪些 token 前缀表示同一份计算结果？
                         │
                         ▼
paged KV allocator:
  这份结果实际位于哪些物理 blocks？
```

因此，它们可以组合；也不能因为一个引擎支持分页，就推断它一定采用 radix tree 做前缀匹配。

## Cache-aware 调度为什么重要

假设 GPU A 缓存了文档 X，GPU B 队列更短。新的文档 X 请求该发到哪里？

- 发往 A，可以跳过长 prefill，但可能多排队；
- 发往 B，立刻开始，却要重新计算前缀。

单实例内部也有类似问题：优先调度缓存命中较长的请求，可以提高短期吞吐和 locality；若总是如此，没有命中的请求可能长期等待。

所以调度器需要在几项成本间权衡：

$$
\text{estimated completion cost}
=\text{queue cost}
+\text{uncached prefill cost}
+\text{decode cost}
+\text{cache pressure}
$$

这不是一个固定公式，而是理解决策所需的四类信息。实际生产还要加入优先级与 SLO，避免缓存命中率成为唯一目标。

多副本部署时，普通 round-robin 或最短队列路由看不到各实例的 KV 状态，会破坏跨请求复用。prompt-aware router 需要知道前缀亲和性，同时处理实例故障、扩缩容和热点文档倾斜。

## Structured generation 是另一条优化路径

SGLang 论文同时讨论了 structured output 的压缩有限状态机。它与 RadixAttention 经常在 Agent workload 中共同出现，但作用不同：

- RadixAttention 复用已经计算的历史 K/V；
- grammar/regex/JSON schema 限制下一 token 的合法集合；
- compressed FSM 减少结构化解码过程中的状态处理开销。

例如工具定义作为固定前缀可被缓存，工具调用结果又通过 JSON grammar 保证语法。前者主要降低 prefill，后者影响 sampling/decode 的合法 token 选择。评测时应分别开关，避免把所有收益都归因于“RadixAttention”。

## 哪些场景收益有限

### 前缀重复率低

完全随机或高度个性化的 prompt 很少命中，树维护和哈希只有额外开销。通常这部分开销可控，但不应预期显著加速。

### 输出远长于输入

命中只省 prefill。若请求输入 100 token、输出 4000 token，端到端时间主要花在 decode，前缀缓存对总时长的影响有限。

### 动态内容位于开头

时间戳、用户 ID 或每次变化的检索结果如果出现在固定模板之前，会截断后续全部共享机会。

### 模型状态不同

LoRA adapter、模型权重、位置、某些多模态输入或影响 K/V 的配置不同，缓存不能混用。只比较 token 而忽略计算上下文，会产生错误结果甚至租户间数据风险。

## 设计一组能解释结果的实验

SGLang 官方 benchmark 提供共享前缀合成数据。更接近业务的实验可以分三组：

1. **随机组**：输入长度相同，但 token 随机，作为零复用基线；
2. **共享组**：若干请求共享固定长前缀，问题部分不同；
3. **模板扰动组**：语义相同，但在前部加入时间戳、调整 schema 顺序或空白符。

对每组扫描请求率，并记录：

- 命中的 prefix token 数与总输入 token 数之比；
- 实际执行的 prefill tokens；
- TTFT P50/P95/P99；
- 吞吐、队列时间和 KV Cache usage；
- eviction 与 preemption；
- 每个副本的命中分布。

只报告“cache hit request ratio”可能误导：命中 1 个 token 和命中 16k token 都算一次命中。更有解释力的是被复用 token 的数量、节省的 prefill 工作以及最终 SLO goodput。

实验还应比较关闭 radix cache 的基线：

```powershell
# 参数名称以当前 SGLang 文档为准
python -m sglang.launch_server `
  --model-path <model> `
  --disable-radix-cache
```

固定模型、精度、并发与 prompt 后，开启/关闭缓存的差值才能说明 RadixAttention 自身带来的收益。

## 小结

RadixAttention 把 LLM 应用中的公共前缀变成可以被 runtime 观察和管理的结构。radix tree 负责最长前缀匹配与路径压缩，KV allocator 保存实际状态，调度器则决定何时利用 locality、何时照顾公平性。

它最适合长输入、短到中等输出、重复前缀明显的 workload。要获得这类收益，应用必须先产生稳定的 token 前缀，多副本路由也必须保留缓存亲和性。理解了“token 匹配—KV 引用—suffix prefill—路径插入—引用释放—淘汰”这条链路，命中率和 TTFT 的变化就不再是黑盒。

## 参考资料

- [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104)
- [SGLang 官方仓库](https://github.com/sgl-project/sglang)
- [SGLang 官方文档](https://docs.sglang.ai/)
- [SGLang Bench Serving Guide](https://docs.sglang.ai/developer_guide/bench_serving.html)
