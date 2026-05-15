---
layout: post
title: 推测解码（Speculative Decoding）原理与实战
subtitle: 从理论证明到 EAGLE-3、SSD，让 LLM 推理速度翻倍的系统指南
date: 2026-05-12
author: iStar
header-img: img/post-bg-speculative-decoding.png
catalog: true
mathjax: true
tags:
  - AI Infra
  - 推测解码
  - LLM推理
  - 推理优化
---

> **摘要**：大语言模型的推理生成是一个典型的 **memory-bound** 过程——每次生成一个 token 都需要把整个模型权重从 HBM 中读取一遍，而计算量相对极少。推测解码（Speculative Decoding）通过"小模型先猜、大模型一次验证多个"的策略，在保证输出分布完全一致的前提下，将推理速度提升 2-6 倍。本文将从数学原理、技术演进、性能对比到实战代码，全面剖析这一 2025-2026 年最重要的推理优化技术。

---

## 一、为什么 LLM 推理这么慢？

要理解推测解码的价值，首先要明白 LLM 推理的瓶颈在哪里。

### 1.1 Memory-Bound 的本质

假设我们有一个 70B 参数的模型，使用 BF16 精度：

```
模型权重大小 = 70 × 10⁹ × 2 bytes ≈ 140 GB
```

每次生成一个 token，模型需要：
1. 从 HBM 中读取约 140 GB 的权重数据
2. 进行一次前向传播计算（FLOPs 相对很少）
3. 输出下一个 token 的概率分布

在 A100 80GB 上，HBM 带宽约为 2 TB/s。理论上，仅权重读取就需要：

```
140 GB / 2 TB/s = 70 ms
```

而实际的 GPU 计算时间可能只有 5-10 ms。**超过 85% 的时间花在了数据传输上**，这就是 memory-bound 的典型特征。

### 1.2 传统自回归生成的困境

```
Prompt → [Forward Pass] → Token₁ → [Forward Pass] → Token₂ → [Forward Pass] → Token₃ → ...
```

每个 token 都要经过一次完整的模型前向传播。如果生成 100 个 token，就要执行 100 次完整的 Forward Pass。GPU 的计算能力被严重浪费，大部分时间都在等待数据从内存中加载。

**核心矛盾**：GPU 的算力远超需求，但内存带宽成了瓶颈。

---

## 二、推测解码的核心思想

推测解码的灵感很简单：既然验证 N 个 token 的成本和验证 1 个 token 几乎相同（因为权重都要读一遍），那我们为什么不一次性验证多个 token 呢？

### 2.1 基本流程

```
┌──────────────────────────────────────────────────────────────┐
│                   推测解码流程                                │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Step 1: 草稿阶段（Draft）                                   │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐                  │
│  │ Draft   │───▶│ Draft   │───▶│ Draft   │                  │
│  │ Token₁  │    │ Token₂  │    │ Token₃  │                  │
│  └─────────┘    └─────────┘    └─────────┘                  │
│       │              │              │                        │
│  (小模型快速生成 γ 个候选 token)                             │
│                                                              │
│  Step 2: 验证阶段（Verify）                                  │
│  ┌─────────────────────────────────────────┐                │
│  │         Target Model (一次 Forward)      │               │
│  │                                         │                │
│  │  Token₁ ✓   Token₂ ✓   Token₃ ✗        │                │
│  │    ↓          ↓          ↓              │                │
│  │  接受       接受      拒绝→重新采样       │                │
│  └─────────────────────────────────────────┘                │
│                                                              │
│  最终输出：Target_Token₁, Target_Token₂, Corrected_Token₃   │
└──────────────────────────────────────────────────────────────┘
```

### 2.2 关键洞察

在 Google TPU v5p 等高端硬件上，研究者发现了一个重要现象——**K-Flat 特性**：验证 1024 个 token 的成本几乎与验证 16 个 token 相同！这是因为：

- 模型权重的读取是固定成本（与验证的 token 数量无关）
- 额外的计算量在 memory-bound 场景下几乎可以忽略

这意味着：**生成更长的草稿序列几乎是"免费"的**。瓶颈从验证成本转移到了草稿质量上。

---

## 三、数学原理：为什么输出分布完全一致？

推测解码最精妙的地方在于——**它能保证输出分布与目标模型完全一致**。不是"近似"，而是**数学意义上的精确匹配**。

### 3.1 拒绝采样（Rejection Sampling）

设目标模型对 token $x$ 的概率为 $P(x)$，草稿模型的概率为 $Q(x)$。

对于每个草稿 token $x_i$，接受概率为：

$$\alpha_i = \min\left(1, \frac{P(x_i)}{Q(x_i)}\right)$$

**直觉理解**：
- 如果 $P(x_i) \geq Q(x_i)$（目标模型认为更可能），**总是接受**
- 如果 $P(x_i) < Q(x_i)$（目标模型认为不太可能），**按比例接受**

### 3.2 拒绝后的修正采样

当某个草稿 token 被拒绝时，我们不能简单地跳到下一个草稿 token——那样会破坏分布。正确的做法是从**残差分布**中重新采样：

$$P_{\text{residual}}(x) = \frac{\max(0, P(x) - Q(x))}{1 - \sum_{x'} \min(P(x'), Q(x'))}$$

这个残差分布恰好补偿了被拒绝的概率质量，确保整体分布与 $P(x)$ 完全一致。

### 3.3 严格证明

对于单个位置，采样过程产生的边缘分布为：

$$\begin{align*}
P_{\text{output}}(x) &= Q(x) \cdot \min\left(1, \frac{P(x)}{Q(x)}\right) + Q(x) \cdot \left(1 - \min\left(1, \frac{P(x)}{Q(x)}\right)\right) \cdot P_{\text{residual}}(x) \\
&= \min(P(x), Q(x)) + \max(0, P(x) - Q(x)) \\
&= P(x)
\end{align*}$$

**证毕。** 无论草稿模型多差，只要验证阶段正确执行拒绝采样，最终输出的分布就严格等于目标模型的分布。

### 3.4 加速比分析

设草稿模型的接受率为 $\alpha$（即每个 token 被目标模型接受的概率），每次草稿 $\gamma$ 个 token，则期望的加速比为：

$$\text{Speedup} = \frac{\gamma + 1}{(1 - \alpha^{\gamma+1}) / (1 - \alpha)}$$

当 $\alpha \to 1$（完美草稿）时，加速比趋近于 $\gamma + 1$。

| 接受率 α | γ=3 | γ=5 | γ=10 |
|---------|-----|-----|------|
| 0.5 | 1.60x | 1.78x | 1.82x |
| 0.7 | 2.11x | 2.48x | 2.59x |
| 0.8 | 2.44x | 2.93x | 3.17x |
| 0.9 | 2.78x | 3.43x | 3.86x |

**核心结论**：接受率 $\alpha$ 是决定加速效果的最关键因素。这就是为什么 EAGLE 系列追求更高的接受率，而不仅仅是更多的草稿 token。

---

## 四、技术演进：从经典到前沿

### 4.1 经典方案：独立草稿模型

最早期的推测解码使用一个独立的小模型作为草稿生成器。

```python
# 伪代码：经典推测解码
def speculative_decode(target_model, draft_model, prompt, gamma=5):
    tokens = tokenize(prompt)
    
    while not is_eos(tokens):
        # Step 1: 草稿模型生成 γ 个 token
        draft_tokens = []
        draft_input = tokens
        for _ in range(gamma):
            next_token = draft_model.generate(draft_input)
            draft_tokens.append(next_token)
            draft_input = concat(draft_input, next_token)
        
        # Step 2: 目标模型一次性验证所有草稿 token
        target_probs = target_model.forward(tokens + draft_tokens)
        
        # Step 3: 接受/拒绝
        accepted = 0
        for i, draft_tok in enumerate(draft_tokens):
            p_target = target_probs[i][draft_tok]
            p_draft = draft_model.get_prob(draft_tok)
            if random() < min(1, p_target / p_draft):
                accepted += 1
            else:
                # 从残差分布采样
                tokens.append(sample_residual(target_probs[i], draft_model))
                break
        
        if accepted == gamma:
            # 全部接受，目标模型再多生成一个
            tokens.append(sample(target_probs[gamma]))
        
        tokens.extend(draft_tokens[:accepted])
    
    return tokens
```

**问题**：需要训练和维护一个独立的草稿模型，且要确保草稿模型与目标模型的词表和对齐方式兼容。

### 4.2 Medusa：多头解码

Medusa 的核心创新是**不再使用独立草稿模型**，而是在目标模型的顶部添加多个额外的解码头。

```
┌───────────────────────────────────────────┐
│           Target Model Body               │
│         (共享的 Transformer 层)            │
└────┬────┬────┬────┬────┬────┬────────────┘
     │    │    │    │    │    │
  Head₀ Head₁Head₂Head₃Head₄Head₅
   (原始) (t+1) (t+2) (t+3) (t+4) (t+5)
```

每个头预测不同位置的 token：
- `Head₀`：预测当前 token（原始 LM head）
- `Head₁`：预测下一个 token（t+1）
- `Head₂`：预测下下个 token（t+2）
- 以此类推...

**训练方式**：通过自蒸馏（self-distillation），让每个头学习目标模型在对应位置上的输出分布。

**性能数据**：
- Medusa-1：~2x 加速
- Medusa-2：2.3-2.8x 加速，代码任务可达 3.29x，数据提取任务可达 3.62x
- 草稿准确率约 0.6

**代价**：以 Mistral-7B 为例，添加 5 个 Medusa 头会增加约 7.5 亿参数。

### 4.3 EAGLE：特征级外推

EAGLE（Extrapolation Algorithm for Greater Language-model Efficiency）采取了更聪明的策略。

**关键洞察**：草稿模型不需要从头开始预测 token——它可以直接利用目标模型已经计算出的内部特征（hidden states）。

```
EAGLE 架构：

Prompt → [Target Model Layers] → Hidden State h
                                    │
                              ┌─────▼─────┐
                              │  EAGLE     │
                              │  Draft     │
                              │  Head      │
                              └─────┬─────┘
                                    │
                              Draft Tokens
                                    │
                              (反馈给 Target Model 验证)
```

EAGLE 的训练数据来源于目标模型推理过程中的 hidden states。它学习的是：**给定目标模型的内部表征，下一个 token 应该是什么**。

**为什么这样更好？**
1. **信息更丰富**：hidden states 包含了目标模型对这个输入的完整理解
2. **分布更对齐**：因为直接基于目标模型的特征训练，草稿分布天然更接近目标分布
3. **参数极少**：只需添加一个轻量级的自回归头（以 LLaMA-8B 为例，仅增加 0.25B 参数）

**性能对比**：

| 方法 | LLaMA-2-70B 加速比 | 草稿准确率 |
|------|-------------------|-----------|
| 经典草稿模型 | ~1.5-2x | 变化大 |
| Medusa-2 | ~2x | ~0.6 |
| EAGLE-1 | 2.7-3.5x | ~0.75 |
| EAGLE-2 | ~4x | ~0.78 |
| EAGLE-3 | 3.0-6.5x | ~0.8 |

### 4.4 EAGLE-3：直接 token 预测与多层融合

2025 年末发布的 EAGLE-3 做了两个关键改进：

1. **放弃特征预测，改为直接 token 预测**：不再预测 hidden states，而是直接预测下一个 token 的概率分布
2. **多层信息融合**：同时利用目标模型多个层的输出，而非单一层

这两个改动让 EAGLE-3 在保持输出一致性的同时，进一步提升了草稿质量。

**实际生产环境数据**（NVIDIA 测试）：
- Batch size = 2：吞吐量提升 1.81x
- Batch size = 64：吞吐量仍保持 1.38x 提升
- 跨多种模型稳定在 3.0-6.5x 延迟加速

### 4.5 P-EAGLE：并行草稿生成

EAGLE 系列的一个瓶颈是草稿生成本身是自回归的。P-EAGLE（Parallel EAGLE）打破了这个限制：

**核心改进**：将所有草稿 token 的生成合并为**单次前向传播**，从自回归草稿变为并行草稿。

**效果**（在 NVIDIA B200 上测试）：
- 相比 EAGLE-3 额外提升 1.69x
- 低并发时吞吐量提升 55-69%
- 高并发时仍保持 5-25% 提升

### 4.6 推测的推测解码（SSD/Saguaro）：2026 年的新突破

这是 2026 年初提出的最新范式，核心思想是**连"草稿-验证"这个串行流水线也要并行化**。

```
传统推测解码：
  [Draft] ──▶ [Verify] ──▶ [Draft] ──▶ [Verify] ──▶ ...
  (串行依赖，验证完才能开始下一轮草稿)

SSD 并行推测：
  [Draft₀] ──▶ [Verify₀] ──▶ [Output₀]
      │             │
      ▼             ▼
  [Draft₁]     [Speculation Cache]
  [Draft₂]         │
  (预生成)         └──▶ 命中 → 零延迟获取
                   └──▶ 未命中 → 回退策略
```

**Saguaro 算法**的三个核心创新：

1. **几何扇出缓存构建**（Geometric Fan-Out）：高效构建推测缓存，存储多种可能验证结果对应的预计算草稿
2. **新型采样方案**：提高缓存命中率
3. **Batch Size 感知回退策略**：在预测失败时根据当前 batch 大小自适应调整策略

**性能**：相比优化过的推测解码基线再提速 2x，相比原始自回归解码提速最高 5x。

### 4.7 DFlash：扩散风格的推测解码

Google 在 TPU v5p 上提出的另一种思路：**用类似扩散模型的方式来"涂绘"token 块**。

不同于自回归逐个生成草稿 token，DFlash 以并行的方式同时预测整个 token 块，类似于扩散模型的去噪过程。

**效果**：在 TPU v5p 上平均加速 3.13x，复杂任务峰值接近 6x，在与 EAGLE-3 的直接对比中表现更优。

---

## 五、各方案全景对比

| 维度 | 独立草稿模型 | Medusa-2 | EAGLE-3 | P-EAGLE | SSD/Saguaro |
|------|------------|----------|---------|---------|-------------|
| **架构** | 两个独立模型 | 目标模型+多头 | 目标模型+轻量头 | 并行草稿头 | 推测缓存+异步草稿 |
| **典型加速** | 1.5-2.5x | 2-3.6x | 3-6.5x | EAGLE-3 × 1.69 | 5x (vs AR) |
| **草稿准确率** | 不稳定 | ~0.6 | ~0.8 | ~0.8 | 依赖命中率 |
| **输出一致性** | ✅ 严格一致 | ⚠️ 非贪婪可能不一致 | ✅ 严格一致 | ✅ 严格一致 | ✅ 严格一致 |
| **额外参数** | 整个小模型 | ~7.5亿(7B模型) | ~0.25亿(8B模型) | 类似EAGLE-3 | 缓存内存 |
| **训练成本** | 需要独立训练 | 自蒸馏，低成本 | 需训练草稿头 | 需训练 | 需训练 |
| **生产就绪** | ✅ | ✅ | ✅ | ✅ | 早期 |

---

## 六、实战：在 vLLM 中使用推测解码

### 6.1 使用独立草稿模型

```bash
# 启动 vLLM 服务器，使用独立草稿模型
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --speculative_model meta-llama/Llama-3.1-8B-Instruct \
    --num_speculative_tokens 5 \
    --gpu_memory_utilization 0.9 \
    --tensor_parallel_size 4
```

关键参数说明：
- `--speculative_model`：草稿模型，通常选择与目标模型同系列的小模型
- `--num_speculative_tokens`：每次草稿生成的 token 数量（即 $\gamma$）
- 草稿模型和目标模型必须**词表相同、架构兼容**

### 6.2 使用 Medusa

```bash
# 需要先训练/加载 Medusa 头
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --medusa_num_heads 5 \
    --medusa_max_seq_length 4096 \
    --gpu_memory_utilization 0.9
```

Medusa 头可以通过自蒸馏训练：

```python
# Medusa 头训练伪代码
import torch
from medusa.model import MedusaModel

# 加载目标模型
model = MedusaModel.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    medusa_num_heads=5,
    medusa_num_layers=1,
)

# 自蒸馏训练：用目标模型自己的输出作为标签
for batch in train_data:
    # 目标模型的标准输出（head 0）
    standard_output = model(batch, head=0)
    
    # 训练其他头
    for head_idx in range(1, 6):
        head_output = model(batch, head=head_idx)
        # 用 head 0 在对应位置的输出作为训练目标
        target = standard_output.logits[:, head_idx-1:, :]
        loss = cross_entropy(head_output.logits, target)
        loss.backward()
    
    optimizer.step()
```

### 6.3 在 OpenAI 兼容 API 中调用

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-abc123"
)

response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-70B-Instruct",
    messages=[
        {"role": "system", "content": "你是一个技术助手。"},
        {"role": "user", "content": "解释一下推测解码的原理。"}
    ],
    max_tokens=500,
    temperature=0.7,
    # 注意：推测解码需在服务端启动时配置
    # 客户端无需额外参数
)

print(response.choices[0].message.content)
```

---

## 七、生产环境最佳实践

### 7.1 如何选择方案？

```
你的场景是什么？
│
├── 有同系列小模型可用？
│   └── 是 → 经典推测解码（最简单）
│   └── 否 ↓
│
├── 追求部署简单、任务结构化程度高？
│   └── 是 → Medusa
│   └── 否 ↓
│
├── 追求极致加速比和输出一致性？
│   └── 是 → EAGLE-3（推荐）
│   └── 否 ↓
│
├── 在 B200/H200 等新硬件上？
│   └── 是 → P-EAGLE
│   └── 否 ↓
│
└── 愿意尝试最新方案？
    └── SSD/Saguaro
```

### 7.2 调参指南

**$\gamma$（草稿 token 数量）的选择**：

- 代码生成、技术文档：$\gamma = 5-8$（结构可预测性强，接受率高）
- 创意写作、对话：$\gamma = 3-5$（多样性高，接受率较低）
- 数学推理：$\gamma = 2-4$（精确性要求高，草稿容易出错）

**接受率监控**：

```python
# 在 vLLM 中监控接受率
import requests

stats = requests.get("http://localhost:8000/stats").json()
acceptance_rate = stats["spec_decode_acceptance_rate"]
draft_tokens = stats["spec_decode_draft_len"]

print(f"接受率: {acceptance_rate:.2%}")
print(f"平均草稿长度: {draft_tokens:.1f}")

# 如果接受率 < 0.5，考虑减少 γ 或换一个草稿模型
# 如果接受率 > 0.9，考虑增加 γ
```

### 7.3 常见陷阱

1. **词表不匹配**：草稿模型和目标模型的词表必须一致，否则需要额外的映射层，增加延迟
2. **KV Cache 共享**：草稿和验证阶段要共享 KV Cache，否则显存消耗翻倍
3. **batch size 影响**：推测解码在低并发时加速效果最明显（延迟敏感场景），高并发时收益递减但吞吐仍有提升
4. **采样温度**：温度越高，草稿模型的接受率越低，加速效果越差。推测解码对 greedy 或低 temperature 场景最有效

---

## 八、未来展望

2025-2026 年的推测解码领域呈现出几个明确趋势：

1. **从独立模型到内嵌头**：Medusa 和 EAGLE 的成功证明了"不需要独立草稿模型"的方向是正确的
2. **从自回归到并行**：P-EAGLE 和 DFlash 都在打破草稿生成的串行瓶颈
3. **从两级到多级**：SSD 的"推测的推测"开启了新的加速维度
4. **硬件协同设计**：K-Flat 特性的发现说明算法和硬件需要一起优化
5. **通用化**：Intel + Weizmann 的工作使得任意小模型可以加速任意大模型（即使词表不同）

一个值得关注的预测是：**未来的 LLM 可能在架构设计时就内建推测解码友好性**，比如 DeepSeek V3 的多 token 预测（MTP）就是一个信号——模型不再仅仅为自回归训练，而是从一开始就设计为能够一次性输出多个 token。

---

## 九、总结

推测解码是 LLM 推理优化中**唯一能在不改变模型、不损失质量的前提下带来 2-6x 加速**的技术。它的核心魅力在于：

- **数学上优雅**：拒绝采样保证分布精确一致
- **工程上实用**：vLLM、TensorRT-LLM、SGLang 均已生产就绪
- **演进上活跃**：从 EAGLE-3 到 SSD，每年都有重大突破

如果你在生产环境中部署 LLM 推理服务，推测解码应该是你最先考虑的优化手段之一——因为它几乎只有收益，没有代价。

---

**参考资料**：

1. Leviathan et al., "Fast Inference from Transformers via Speculative Decoding", ICML 2023
2. Cai et al., "Medusa: Simple LLM Inference Acceleration Framework", 2024
3. Li et al., "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty", 2024
4. EAGLE-3 Technical Report, 2025
5. Saguaro: Speculative Speculative Decoding, arXiv:2603.03251
6. Google Research, "Supercharging LLM Inference on TPUs with Diffusion-Style Speculative Decoding", 2026
7. NVIDIA Developer Blog, "An Introduction to Speculative Decoding", 2025
8. AWS ML Blog, "P-EAGLE: Faster LLM Inference with Parallel Speculative Decoding in vLLM", 2026

**相关文章**：
- [SGLang 与 RadixAttention 详解](/2026/05/12/SGLang与RadixAttention详解/)
- [MoE 推理优化全景指南](/2026/05/13/MoE推理优化全景指南/)
- [FlashInfer 深度解析](/2026/05/13/flashinfer-deep-dive/)
- [2026 大模型推理引擎全景对比](/2026/05/11/2026大模型推理引擎全景对比/)
