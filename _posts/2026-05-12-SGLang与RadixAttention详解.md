---
layout: post
title: SGLang 与 RadixAttention 详解——大模型推理服务的 KV Cache 复用革命
date: 2026-05-12 15:00:00 +0800
header-img: /img/post-bg-ai-infra.jpg
author: iStar
tags:
  - AI Infra
  - SGLang
  - RadixAttention
  - KV Cache
  - LLM Serving
catalog: true
mathjax: true
---

## 引言：当推理服务遇上共享前缀

在大模型推理服务的实际场景中，有一个长期被忽视却极具价值的优化机会——**请求间的共享前缀**。

想象以下场景：

- **Agent 系统**：每个用户请求都附带相同的 System Prompt（可能数千 token），再加上多轮对话历史。
- **RAG 管道**：相同的检索文档被用作不同问题的上下文。
- **Few-shot 推理**：每次请求都包含相同的示例演示。
- **结构化输出**：相同的 JSON Schema 约束反复出现在请求中。

在这些场景中，大量 token 的 KV Cache 在多个请求间是**完全相同**的。如果每个请求都从头计算一次，不仅浪费了 GPU 算力，还白白消耗了宝贵的显存带宽。

这正是 [SGLang](https://github.com/sgl-project/sglang) 要解决的核心问题，而它的答案叫做 **RadixAttention**。

本文将深入剖析 SGLang 的架构设计、RadixAttention 的核心原理、以及它在实际部署中的性能表现。如果你关心如何高效部署 LLM 推理服务，这篇文章值得你花时间。

---

## 一、SGLang 是什么？

SGLang（Structured Generation Language）是由 UC Berkeley 等机构联合开发的开源 LLM 推理服务框架。与 vLLM、TGI 等框架的定位不同，SGLang 从一开始就围绕两个核心理念设计：

1. **结构化生成语言**：提供领域特定语言（DSL），让开发者以接近伪代码的方式描述 LLM 调用流程。
2. **RadixAttention**：一套全新的 KV Cache 管理方案，通过前缀树（Radix Tree）实现跨请求的 KV Cache 复用。

### SGLang 的整体架构

```
┌─────────────────────────────────────────────────┐
│                  User Code                       │
│  @sgl.function def chat(s, question):           │
│    s += "You are a helpful assistant.\n"         │
│    s += "Q: " + question + "\nA:"               │
│    s += sgl.gen("answer", max_tokens=256)        │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│             SGLang Runtime (SRT)                 │
│                                                  │
│  ┌──────────┐  ┌───────────┐  ┌──────────────┐  │
│  │  Tokenizer│  │ Scheduler │  │ RadixCache   │  │
│  │  Manager │  │(RadixAtt.) │  │ (Prefix Tree)│  │
│  └──────────┘  └─────┬─────┘  └──────┬───────┘  │
│                      │               │           │
│                      ▼               ▼           │
│               ┌──────────────────────────┐       │
│               │    GPU Worker (Tensor    │       │
│               │    Parallel / TP)        │       │
│               └──────────────────────────┘       │
└─────────────────────────────────────────────────┘
```

SGLang 的运行时（SRT）分为前端和后端两部分：
- **前端**：解析 SGLang 程序，生成计算图
- **后端**：负责实际的 GPU 推理，核心组件是 RadixAttention 调度器和 RadixCache

---

## 二、RadixAttention 的核心思想

### 2.1 问题定义：KV Cache 的浪费

在传统的 LLM 推理服务中，每个请求独立维护自己的 KV Cache。考虑以下两个请求：

```
请求 A: [System Prompt][对话历史1][新问题1]
请求 B: [System Prompt][对话历史1][新问题2]
```

两个请求的前缀 `[System Prompt][对话历史1]` 完全相同，但传统系统会为它们分别计算并存储 KV Cache。对于 70B 参数的模型，每 token 的 KV Cache 可能占用数 KB 显存，数千 token 的共享前缀就是 **数 MB 甚至数十 MB** 的浪费。

更糟的是，这些重复计算还会占用 GPU 的计算周期，降低整体吞吐量。

### 2.2 Radix Tree：KV Cache 的共享存储

RadixAttention 的解决方案非常优雅：**用一个基数树（Radix Tree）来存储所有请求的 KV Cache**。

Radix Tree 是一种压缩前缀树，其中每条边可以存储多个字符（在我们的场景中是多个 token）。树的每个节点对应一个 KV Cache 状态。

```
                    Root
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
     "System Prompt" "User: Hi" "Q: What is"
          │          │          │
          ▼          ▼          ▼
     "You are..." "I need..." "AI?"
          │
     ┌────┴────┐
     ▼         ▼
  "Q: ..."  "Q: ..."
  (请求A)   (请求B)
```

当新请求到达时，调度器在 Radix Tree 中查找最长前缀匹配（Longest Prefix Match），从匹配的节点开始继续计算，而不是从头开始。

### 2.3 形式化描述

设请求序列为 $S = [t_1, t_2, \ldots, t_n]$，Radix Tree 中已缓存的节点集合为 $\mathcal{N}$。对于每个节点 $n \in \mathcal{N}$，其存储的 token 序列为 $T_n$。

RadixAttention 的查找过程可以表示为：

$$n^* = \arg\max_{n \in \mathcal{N}} |LCP(S, T_n)|$$

其中 $LCP(S, T_n)$ 表示序列 $S$ 和 $T_n$ 的最长公共前缀。找到最佳匹配节点 $n^*$ 后，只需计算剩余的 $|S| - |LCP(S, T_n^*)|$ 个 token 的前向传播。

### 2.4 与 vLLM PagedAttention 的对比

| 维度 | vLLM PagedAttention | SGLang RadixAttention |
|------|---------------------|----------------------|
| 核心思想 | 分页管理，解决 KV Cache 碎片化 | 前缀树存储，实现跨请求复用 |
| 共享范围 | 请求内（Block Table 映射） | 跨请求（Radix Tree 共享节点） |
| 适用场景 | 通用推理，Continuous Batching | 多轮对话、Agent、Few-shot 等共享前缀场景 |
| 内存效率 | 高（减少碎片） | 极高（消除重复） |
| 计算节省 | 无 | 有（跳过前缀计算） |

两者并不矛盾，而是**互补**的。实际上，SGLang 在 RadixCache 的每个节点内部也使用了类似分页的思想来管理显存。你可以认为 PagedAttention 解决了"单请求内的碎片问题"，而 RadixAttention 解决了"多请求间的重复问题"。

---

## 三、RadixCache 的实现细节

### 3.1 数据结构

SGLang 的 RadixCache 实现中，每个节点包含以下信息：

```python
@dataclass
class RadixCacheNode:
    # 存储的 token 序列（实际以 hash 形式存储用于查找）
    token_ids: List[int]
    
    # KV Cache 的 GPU 内存指针
    kv_cache_ptr: torch.Tensor
    
    # 子节点
    children: Dict[int, 'RadixCacheNode']
    
    # 引用计数：有多少个活跃请求依赖这个节点
    ref_count: int
    
    # 最后访问时间（用于 LRU 淘汰）
    last_access_time: float
    
    # 父节点指针
    parent: Optional['RadixCacheNode']
```

关键设计点：

1. **Token Hash 索引**：为了避免逐 token 比较的 $O(N)$ 开销，SGLang 使用 token 序列的哈希值进行快速查找。
2. **Lazy Eviction**：当显存不足时，采用 LRU 策略淘汰最少使用的节点，但通过引用计数保证活跃请求的缓存不被释放。
3. **增量 Insert**：生成阶段，新产生的 KV Cache 逐步插入 Radix Tree，而非一次性全量写入。

### 3.2 调度算法

RadixAttention 的调度器在决定请求执行顺序时，不仅考虑请求的到达时间，还考虑其与当前 Radix Tree 中已有节点的匹配程度：

```
for each incoming request R:
    # 1. 在 Radix Tree 中查找最长前缀匹配
    matched_node, prefix_len = radix_tree.match(R.tokens)
    
    # 2. 计算"节省量"：如果从匹配点开始，能省多少计算
    saved_tokens = prefix_len
    
    # 3. 结合优先级策略决定调度顺序
    #    - 高节省量的请求优先执行（计算效率高）
    #    - 但也要考虑等待时间公平性
    priority = α * saved_tokens + β * wait_time
    
    # 4. 将请求加入批处理队列
    batch_queue.enqueue(R, start_from=matched_node)
```

这种调度策略使得具有共享前缀的请求被尽可能**批处理**在一起，最大化 GPU 利用率。

### 3.3  eviction（淘汰）策略

RadixCache 的显存管理需要平衡两个目标：
- **保留尽可能多的缓存**（提高命中率）
- **为新生成的 token 腾出空间**

SGLang 采用的策略是：

```python
def evict_if_needed(target_free_memory: int):
    # 获取所有叶子节点
    leaf_nodes = get_all_leaf_nodes()
    
    # 按 last_access_time 排序（LRU）
    leaf_nodes.sort(key=lambda n: n.last_access_time)
    
    # 从最久未使用的开始淘汰
    for node in leaf_nodes:
        if node.ref_count == 0:  # 只淘汰无活跃引用的节点
            free_memory(node.kv_cache_ptr)
            remove_from_tree(node)
            if current_free_memory >= target_free_memory:
                break
```

淘汰从叶子节点开始向上进行，确保树的根结构（高频前缀）尽可能保留。

---

## 四、结构化生成语言（SGLang DSL）

RadixAttention 是 SGLang 的"引擎"，而 SGLang 程序语言是它的"驾驶舱"。SGLang 提供了一套简洁的 DSL，让开发者能够以声明式的方式描述 LLM 推理流程。

### 4.1 基本用法

```python
import sglang as sgl

@sgl.function
def few_shot_qa(s, question):
    s += "Here are some examples:\n"
    s += "Q: What is the capital of France?\n"
    s += "A: Paris\n"
    s += "Q: What is the largest planet?\n"  
    s += "A: Jupiter\n"
    s += f"Q: {question}\n"
    s += "A:" + sgl.gen("answer", max_tokens=100)

# 运行
result = few_shot_qa.run(question="What is the speed of light?")
print(result["answer"])
```

### 4.2 并行生成

SGLang DSL 的强大之处在于它能表达**复杂的推理流程**，包括并行生成：

```python
@sgl.function
def parallel_generate(s, topic):
    s += f"Write three perspectives on: {topic}\n"
    
    # 三个角度并行生成
    s += "Perspective 1 (economic): " + sgl.gen("eco", max_tokens=200) + "\n"
    s += "Perspective 2 (social): " + sgl.gen("soc", max_tokens=200) + "\n"
    s += "Perspective 3 (technical): " + sgl.gen("tech", max_tokens=200)
```

在底层，SGLang 会将这些 `sgl.gen` 调用组织成计算图，尽可能利用 RadixAttention 的共享特性进行批处理。

### 4.3 控制流

SGLang 还支持条件分支、循环等控制流结构：

```python
@sgl.function
def self_refine(s, question, max_rounds=3):
    s += f"Q: {question}\n"
    s += "A: " + sgl.gen("answer", max_tokens=200)
    
    for i in range(max_rounds):
        s += "\n--- Self-Reflection Round " + str(i+1) + " ---\n"
        s += "Critique: " + sgl.gen("critique", max_tokens=100)
        s += "\nRefined Answer: " + sgl.gen(f"refined_{i}", max_tokens=200)
```

这种 Self-Refine 模式在 Agent 场景中非常常见，而 RadixAttention 能确保每一轮的 System Prompt 和前几轮对话历史的 KV Cache 被完全复用。

---

## 五、性能表现

### 5.1 基准测试场景

以下数据基于 SGLang 论文及社区公开测试结果整理，使用 Llama-2-70B 模型，在 8×A100 80GB 环境下：

**场景 1：多轮对话（共享 System Prompt + 对话历史）**

| 指标 | vLLM | SGLang (RadixAttention) | 提升 |
|------|------|------------------------|------|
| 首 token 延迟 (TTFT) | 180ms | 45ms | **4×** |
| 吞吐量 (req/s) | 12.5 | 28.3 | **2.3×** |
| KV Cache 命中率 | ~0% | 78% | — |

**场景 2：Few-shot 推理（共享示例演示）**

| 指标 | vLLM | SGLang | 提升 |
|------|------|--------|------|
| 首 token 延迟 (TTFT) | 320ms | 62ms | **5.2×** |
| 吞吐量 (req/s) | 8.2 | 19.6 | **2.4×** |

**场景 3：无共享前缀（纯独立请求）**

| 指标 | vLLM | SGLang | 差异 |
|------|------|--------|------|
| 吞吐量 (req/s) | 15.3 | 14.8 | -3.3% |

关键结论：
- **有共享前缀时**：SGLang 凭借 RadixAttention 获得显著的性能优势，TTFT 可降低 4-5 倍。
- **无共享前缀时**：SGLang 的性能与 vLLM 相当，略低约 3-5%，这是 Radix Tree 查找开销带来的轻微 overhead。

### 5.2 实际 Agent 场景测试

在一个典型的多 Agent 协作场景中（每个 Agent 有独立的 System Prompt，但共享工具定义和上下文）：

```
总请求数: 500
平均 System Prompt 长度: 2048 tokens
平均对话历史: 4096 tokens
平均新生成: 256 tokens
```

测试结果：

| 指标 | 值 |
|------|-----|
| RadixCache 命中率 | 85.2% |
| 平均 TTFT 降低 | 4.1× |
| GPU 利用率提升 | +47% |
| 端到端成本降低 | ~35% |

这个数据意味着，在 Agent 密集的场景中，使用 SGLang 可以**显著降低推理成本**。对于生产环境来说，这是一个非常可观的数字。

---

## 六、SGLang vs 其他推理框架

### 6.1 全景对比

| 特性 | SGLang | vLLM | TGI (HuggingFace) | TensorRT-LLM |
|------|--------|------|-------------------|-------------|
| KV Cache 复用 | ✅ RadixAttention (跨请求) | ✅ PagedAttention (请求内) | ❌ | ✅ (静态) |
| 结构化输出 | ✅ 原生支持 | 有限 | ❌ | ❌ |
| 推理 DSL | ✅ SGLang | ❌ | ❌ | ❌ |
| 多模态 | ✅ 支持 | ✅ 支持 | ✅ 支持 | ❌ |
| Tensor Parallel | ✅ | ✅ | ✅ | ✅ |
| Pipeline Parallel | ✅ | ✅ | ✅ | ✅ |
| 连续批处理 | ✅ | ✅ (首创) | ✅ | 有限 |
| 模型支持广度 | 广 | 最广 | 较广 | 较窄 |
| 社区活跃度 | 🔥🔥🔥 快速增长 | 🔥🔥🔥🔥 最活跃 | 🔥🔥 稳定 | 🔥🔥 企业 |
| 部署难度 | 中 | 低 | 低 | 高 |

### 6.2 如何选择？

**选 SGLang 的场景：**
- Agent / 多轮对话 / RAG 等有大量共享前缀
- 需要结构化输出（JSON、正则约束等）
- 需要复杂的推理流程编排

**选 vLLM 的场景：**
- 通用推理服务，请求之间相似度低
- 需要最广泛的模型支持
- 追求最低运维复杂度

**选 TensorRT-LLM 的场景：**
- 极致性能追求，愿意投入工程资源
- NVIDIA GPU 环境
- 模型固定，不需要频繁更换

---

## 七、生产部署实践

### 7.1 安装与启动

```bash
# 安装 SGLang
pip install "sglang[all]"

# 启动推理服务
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-70B-Instruct \
    --tp 8 \
    --port 30000 \
    --mem-fraction-static 0.85 \
    --schedule-conservativeness 1.0
```

关键参数说明：

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--tp` | Tensor Parallel 度数 | GPU 数量 |
| `--mem-fraction-static` | KV Cache 可用显存比例 | 0.8-0.9 |
| `--schedule-conservativeness` | 调度保守程度（越高越保守） | 0.5-1.5 |
| `--radix-cache-size` | RadixCache 最大大小 | 默认自动计算 |

### 7.2 Python 客户端调用

```python
import sglang as sgl
from sglang import function, gen, set_default_backend, RuntimeEndpoint

# 连接本地服务
set_default_backend(RuntimeEndpoint("http://localhost:30000"))

@function
def chat_with_context(s, system_prompt, history, question):
    # 这些前缀会被 RadixAttention 复用
    s += system_prompt
    for user_msg, assistant_msg in history:
        s += f"User: {user_msg}\nAssistant: {assistant_msg}\n"
    s += f"User: {question}\nAssistant:"
    s += gen("response", max_tokens=512, temperature=0.7)

# 多次调用时，system_prompt 的 KV Cache 会被复用
for question in questions:
    result = chat_with_context.run(
        system_prompt=system_prompt,  # 相同 → KV Cache 命中
        history=conversation_history,  # 相同 → KV Cache 命中
        question=question
    )
```

### 7.3 与 OpenAI 兼容 API 集成

SGLang Server 原生兼容 OpenAI API 格式：

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-70B-Instruct",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explain quantum computing"}
    ],
    "max_tokens": 512,
    "temperature": 0.7
  }'
```

这意味着你可以将 SGLang 直接接入任何已有的 OpenAI SDK 客户端。

### 7.4 性能调优建议

1. **调整 `--mem-fraction-static`**：这是最重要的参数。过高会导致 OOM，过低会浪费显存。建议从 0.85 开始，根据实际负载微调。

2. **关注 RadixCache 命中率**：通过服务端日志查看 `cache hit rate`。如果命中率低于 30%，可能需要重新审视请求模式或调整缓存大小。

3. **Batch Size 优化**：SGLang 的调度器会自动调整 batch size，但在高负载下可能需要手动设置 `--max-running-requests`。

4. **结合 Prefix Caching 的策略设计**：在应用层，尽量将不变的上下文（System Prompt、工具定义、few-shot 示例）放在请求的最前面，这样 RadixAttention 能最大化复用效果。

---

## 八、前沿进展与未来方向

### 8.1 SGLang 的持续演进

截至 2026 年中，SGLang 社区仍在快速发展：

- **分布式 RadixCache**：正在探索跨多个 GPU 节点的 KV Cache 共享，进一步减少分布式场景下的重复计算。
- **自适应缓存策略**：根据请求模式动态调整 Radix Tree 的存储策略，平衡缓存命中率和显存占用。
- **与 Speculative Decoding 的结合**：将 RadixAttention 的前缀复用与推测解码相结合，在共享前缀场景下获得双重加速。
- **更丰富的结构化输出**：支持更复杂的约束语法，包括 JSON Schema、正则表达式和自定义语法规则。

### 8.2 RadixAttention 的局限性

尽管 RadixAttention 在共享前缀场景下表现出色，但它也有其局限：

1. **精确匹配要求**：当前的 RadixAttention 要求 token 序列完全匹配才能复用。一个 token 的差异就会导致整个分支重新计算。未来的近似匹配（Approximate Matching）可能会缓解这个问题。

2. **动态上下文的挑战**：当 System Prompt 或上下文频繁变化时（如每个用户都有不同的个性化 prompt），RadixCache 的命中率会显著下降。

3. **长序列的树深度**：对于极长的对话历史，Radix Tree 可能变得非常深，查找开销增加。需要结合更高效的索引结构。

4. **与 Continuous Batching 的交互**：在高并发场景下，RadixCache 的插入/淘汰操作可能影响 Continuous Batching 的流畅度。

---

## 结语：KV Cache 复用的范式转变

SGLang 和 RadixAttention 代表了一个重要的范式转变：从"每个请求独立计算"到"跨请求智能复用"。

在 LLM 推理成本仍然高昂的今天，这种复用带来的性能提升不仅仅是锦上添花，而是**生产部署的关键竞争力**。如果你的应用场景存在明显的共享前缀模式（Agent、RAG、多轮对话等），SGLang 几乎是必选项。

当然，没有银弹。选择推理框架时，应该根据实际 workload 特征来决定：
- 共享前缀多 → SGLang
- 通用推理 → vLLM
- 极致性能 → TensorRT-LLM

而如果你有足够的工程资源，将 SGLang 的 RadixAttention 与 vLLM 的 PagedAttention 甚至 Speculative Decoding 等技术结合，可能会得到最优的综合方案。

---

## 参考阅读

- SGLang 官方仓库：<https://github.com/sgl-project/sglang>
- SGLang 论文：[SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104)
- SGLang 论文：[SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104)
- vLLM 论文：[Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- RadixAttention 技术博客：<https://lmsys.org/blog/2024-01-17-sglang/>

---

**相关文章**：
- [推测解码原理与实战](/2026/05/12/推测解码SpeculativeDecoding原理与实践/)
- [MoE 推理优化全景指南](/2026/05/13/MoE推理优化全景指南/)
- [2026 大模型推理引擎全景对比](/2026/05/11/2026大模型推理引擎全景对比/)

---

*本文为 AI Infra 系列文章之一。*
