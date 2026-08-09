---
layout: post
title: "SpecForge：把 EAGLE3 从训练样本交付到 SGLang"
subtitle: "理解特征对齐、Training-Time Test 与草稿模型的部署契约"
date: 2026-05-27 12:00:00 +0800
last_modified_at: 2026-08-09
author: iStar
catalog: true
series: speculative-decoding
series_order: 40
technology_year: 2025
tags: [推测解码, EAGLE, SGLang]
---

推测解码常被概括为“用小模型猜、用大模型验”。这个说法解释了运行时，却没有说明草稿模型如何得到。

对于 EAGLE3，草稿模型不是一个可以随意替换的小语言模型。它读取目标模型指定层的隐藏状态，沿着与服务阶段相似的多步路径产生候选，再把候选交给目标模型验证。因此，一个可部署的 EAGLE3 checkpoint 实际上与以下对象共同构成一份契约：

- 目标模型及其权重版本；
- tokenizer、词表映射和 chat template；
- 被抽取的隐藏层以及特征融合方式；
- 训练时的数据与 loss mask；
- SGLang 中的草稿树、验证和采样实现。

SpecForge 的作用，就是把这些分散的约束组织成一条可复现的训练链路，并让产物能够接入 SGLang。理解这条链路，比记住一组启动命令更重要。

## 先看 checkpoint 最终要参与什么计算

设目标模型为 \(p\)，EAGLE3 草稿模型为 \(q\)。一次生成循环可以简化为：

```text
目标模型 prefill / 上一轮验证
        │
        ├── 产生目标分布 p
        └── 暴露低层、中层、高层隐藏状态
                     │
                     ▼
              EAGLE3 草稿模型
                     │
             构造多步候选树 q
                     │
                     ▼
          目标模型一次并行验证整棵树
                     │
          接受连续前缀，拒绝处进行校正
                     │
                     └── 进入下一轮
```

草稿 token 不会未经验证直接成为结果。目标模型会为候选树计算真实概率，并按照推测采样的接受/拒绝规则决定保留哪些 token。实现正确时，最终采样分布仍然等价于只使用目标模型；草稿模型影响的是完成一次目标模型前向后，平均能够推进多少 token。

因此，评价 drafter 不能只看训练 loss。至少要区分四类量：

| 层次 | 典型指标 | 回答的问题 |
| --- | --- | --- |
| 预测质量 | token loss、top-1 命中、KL divergence | 草稿分布是否接近目标分布 |
| 接受行为 | 每层接受率、平均接受长度、每轮接受 token 数 | 候选树是否真正命中目标路径 |
| 局部开销 | draft latency、verify latency、候选节点数 | 为多接受 token 付出了多少计算 |
| 端到端效果 | TTFT、TPOT、吞吐、P50/P99、显存 | 在真实负载下是否值得启用 |

同一个 checkpoint 在 batch size 1 上可能显著降低延迟，在高并发下却因为目标模型本身已经被批处理充分利用，只获得较小收益。部署结论必须落到具体服务负载上。

## 为什么不直接找一个更小的同系列模型

传统推测采样可以用独立小模型生成候选，例如让 8B 模型为 70B 模型起草。它的优点是结构直观，但有两个限制：

1. 模型家族未必发布了词表、训练分布都合适的小型号；
2. 小模型只看到 token，不知道目标模型在当前上下文中已经形成了什么内部表示。

EAGLE 改变了第二点。它复用目标模型 LM head 之前的顶层特征，学习预测下一时刻的特征，再通过目标模型的 LM head 得到 token 分布。因为输入携带了目标模型的信息，一个很浅的 drafter 也可能给出质量较高的候选。

但原始 EAGLE 与 EAGLE2 同时优化特征预测和 token 预测。特征损失帮助模型获得多步生成能力，却也要求预测向量接近目标模型的真实下一步特征。论文发现，这一约束限制了模型从更多训练数据中继续获益。

EAGLE3 的变化不是简单地“再加几个隐藏层”，而是同时完成三件事：

- 去掉必须还原下一步真实特征的约束，直接以 token 预测为目标；
- 融合目标模型低、中、高层的隐藏状态；
- 在训练阶段模拟推理时的多步自回归路径，即 Training-Time Test。

这三点互相依赖。只删除特征损失，drafter 在第一步可能更自由；若训练时始终读取真实目标特征，到了第二步却突然读取自己的输出，误差会迅速累积。

## 多层特征融合到底融合了什么

假设目标模型隐藏维度为 \(d\)，从低层、中层、高层分别抽取同一 token 位置的向量：

\[
h_t^{low},\quad h_t^{mid},\quad h_t^{high}\in\mathbb{R}^{d}
\]

先连接三个向量，再用全连接层投影回 drafter 的工作维度：

\[
g_t = W_f [h_t^{low};h_t^{mid};h_t^{high}]
\]

不同深度的表征承担的功能并没有严格边界，但可以用一个直观视角理解：较低层保留更多局部词法和句法信息，中间层逐渐组织上下文，高层更直接地服务于下一个 token 的预测。融合并不是把信息简单平均，而是让训练学出在不同场景下如何组合这些视角。

token embedding 仍然不可缺少。drafter 需要知道已经选择了哪个候选 token，同时读取融合特征，才能继续向前预测。概念上可写成：

\[
a_{t+1}=D(e(x_t), g_t)
\]

其中 \(D\) 是很浅的 drafter，\(e(x_t)\) 是 token embedding，\(a_{t+1}\) 是它的输出表示。输出经过 LM head 后得到下一 token 的草稿分布。

真正困难的地方出现在第二个草稿步。

## 第二步为什么没有“正确隐藏状态”可用

在目标模型刚完成的位置 \(t\)，低、中、高层真实特征都已经存在，可以形成 \(g_t\)。drafter 由此生成候选 \(\hat{x}_{t+1}\)。

如果继续生成 \(\hat{x}_{t+2}\)，目标模型此时还没有验证 \(\hat{x}_{t+1}\)，自然也没有它对应的真实多层特征 \(g_{t+1}\)。如果为了取得这个特征先运行一次目标模型，推测解码就失去了减少目标模型串行前向的意义。

EAGLE3 在后续草稿步中使用前一步 drafter 输出 \(a_{t+1}\) 代替不存在的 \(g_{t+1}\)：

```text
第 1 步：真实目标特征 g_t       + token x_t       -> a_(t+1) -> x̂_(t+1)
第 2 步：drafter 输出 a_(t+1)   + token x̂_(t+1)  -> a_(t+2) -> x̂_(t+2)
第 3 步：drafter 输出 a_(t+2)   + token x̂_(t+2)  -> a_(t+3) -> x̂_(t+3)
```

这解释了为何普通 teacher forcing 不够。若训练只覆盖第一行，模型看到的始终是真实目标特征和真实 token；推理时第二行开始，输入却来自模型自己的输出。训练分布与推理分布发生偏移，越深的候选越容易失真。

## Training-Time Test 如何把推理路径搬进训练

Training-Time Test 的核心不是在训练结束后多跑一次测试，而是在训练过程中展开若干草稿步，让 drafter 接触自己在前一步产生的表示。

一条训练样本不再只构造“一步输入、一步标签”，而是模拟：

1. 从目标模型的真实融合特征出发；
2. 由 drafter 产生第一步输出表示；
3. 把该表示送回 drafter，形成第二步的输入状态；
4. 重复若干步，并在各步对真实后续 token 计算监督信号。

为了并行训练这些展开路径，EAGLE3 使用专门的 attention mask。不同模拟步既要看到共享的历史上下文，又不能偷看本应在未来才出现的信息。可以把 mask 想成一棵在 batch 内展开的树：同一路径上的节点按因果顺序相互可见，不相关分支与未来标签被遮蔽。

它解决的是 exposure bias：模型不是只在理想输入上学会第一步，而是在自己可能制造的输入上学习怎样继续生成。也正因如此，单看第一步准确率不足以判断 TTT 是否有效；接受率需要按树深度分别统计。

## 从序列到动态草稿树

若每一步只保留一个 token，任何一次猜错都会终止连续接受。EAGLE3 沿用 EAGLE2 的 dynamic draft tree：在每个深度保留若干高概率分支，并根据候选置信度决定有限节点预算应放在哪些路径上。

例如，固定 8 个候选节点不一定排成长度为 8 的链，也可以成为：

```text
root
├── A
│   ├── C
│   │   └── F
│   └── D
├── B
│   ├── E
│   └── G
└── H
```

目标模型借助 tree attention 一次验证这些节点。每个节点只能关注它的祖先路径，因此一次前向可以覆盖多条互斥候选，又不会把其他分支的信息泄漏进当前分支。

树越宽，命中正确路径的机会通常越大，但验证 token 数、tree attention 成本和临时显存也随之上升。树越深，成功时一次推进更多 token，但深层候选更容易受到累积误差影响。服务阶段的树参数应与 checkpoint 的深度接受曲线一起调，而不是固定追求更大的候选数。

## 训练数据不是通用语料，而是目标分布的样本

训练一个普通语言模型时，文本质量决定模型学到什么；训练 drafter 时还多了一层要求：数据中的 assistant 输出要接近目标模型在线上会产生的分布。

假设目标模型在线使用某个 chat template、temperature、reasoning 模式和系统提示词，而训练集保留的是另一个模型的回答。即使回答内容正确，token 选择和表达路径仍可能不同。drafter 学到的分布与目标模型不一致，目标验证时就会更频繁拒绝。

SpecForge 支持两类文本输入：

```json
{"id":"case-001","conversations":[
  {"role":"user","content":"解释什么是 KV Cache"},
  {"role":"assistant","content":"KV Cache 保存注意力层中……"}
]}
```

或者已经应用 chat template 的文本：

```json
{"id":"case-001","text":"<已按目标模板格式化的完整对话>"}
```

第二种格式并不意味着可以省略模板配置。SpecForge 仍需知道使用了什么 chat template，才能识别 assistant span 并建立 loss mask。

若追求线上接受率，更稳妥的做法是保留输入问题，用目标模型重新生成 assistant 回答，并记录生成参数。这样得到的不是“更高质量答案”的保证，而是更忠实的目标分布样本。

## 推理模型的数据还要处理隐藏思维边界

reasoning 模型可能把可见回答与结构化的 `reasoning_content` 分开返回。如果服务阶段启用了推理，而数据再生成时关闭了推理，或者反过来，drafter 看到的 token 序列就与真实请求不一致。

多轮对话还有一个容易忽略的监督问题。假设一条记录包含三个 assistant 回合，而配置只对最后一个 assistant span 计算 loss，那么前两个回答只是上下文，不是训练目标。若希望每个回合都成为目标，应把对话展开为三个 generation event：

```text
事件 1：历史截至 assistant_1，并监督 assistant_1
事件 2：历史截至 assistant_2，并监督 assistant_2
事件 3：历史截至 assistant_3，并监督 assistant_3
```

历史回合中不应重新暴露当时的隐藏推理，只保留服务边界上实际可见的内容；当前目标回合则按所选 reasoning 契约保存需要监督的字段。这个规则不是数据清洗细节，而是在定义 drafter 推理时究竟能看到什么。

处理完数据后，至少应检查：

- 每个输入 `id` 都落入 success、error 或 skipped 之一，没有静默丢失；
- 对话角色顺序合法，assistant 目标非空；
- chat template 与目标服务配置一致；
- reasoning 开关与解析器一致，不残留未解析的 `<think>` 标记；
- 截断后仍然存在有效的监督 token；
- train/eval 按会话或问题划分，避免同源回答泄漏。

## 在线、离线与解耦训练是在交换不同成本

EAGLE3 训练既需要 token，也需要目标模型隐藏状态。隐藏状态在哪里生成，决定了整条系统的形态。

| 模式 | 隐藏状态来源 | 优点 | 主要代价 |
| --- | --- | --- | --- |
| 离线 | 预先运行目标模型，写入特征分片 | 训练可复现；训练阶段不必常驻目标模型 | 特征文件很大；配置变化后可能需要重做 |
| 在线 | SGLang 在训练期间生成并发布特征 | 无需保存完整特征集；更容易更新样本 | 目标推理资源必须持续可用；生产与消费速度要匹配 |
| 解耦 | 独立 producer/consumer 通过传输层交换数据 | 训练和目标推理可分别扩缩容 | 多服务编排、背压与故障恢复更复杂 |

这里的“在线”不等于把目标模型塞进每个 trainer 进程。当前 SpecForge 架构中，patched SGLang server 负责目标模型推理与其并行方式，trainer consumer 读取发布的特征并进行数据并行训练。两侧之间若使用 Mooncake 等传输后端，还要监控队列深度、特征传输吞吐和过期数据。

选择模式时可以先问三个问题：

1. 目标模型是否大到必须独占一组推理节点？
2. 隐藏状态能否在可接受的存储预算内长期保存？
3. 数据、模板或特征层是否会频繁变化？

固定数据和稳定配置更适合离线复现实验；特征规模过大或样本持续更新时，在线/解耦模式更有吸引力。

## 离线特征文件是一份严格 schema

当前 SpecForge 为 EAGLE3 离线记录定义的核心张量包括：

```text
input_ids
loss_mask
hidden_state
aux_hidden_state
```

`input_ids` 定义 token 序列，`loss_mask` 定义哪些 token 参与训练，`hidden_state` 与 `aux_hidden_state` 保存策略所需的目标特征。名字看似简单，真正的约束在位置和版本上：

- BOS/EOS 的插入方式必须一致；
- padding side、截断长度和 packed sequence 边界必须一致；
- 每个特征位置必须对应正确 token；
- feature capture 使用的 strategy、draft config 和 layer IDs 必须与训练一致；
- 不能把其他草稿策略的特征目录交给 EAGLE3 reader。

最有价值的对齐检查不是直接启动多机训练，而是打印一条短样本：

```text
position | token_id | decoded token | loss_mask | feature row
0        | ...      | ...           | 0         | 0
1        | ...      | ...           | 0         | 1
2        | ...      | ...           | 1         | 2
```

逐行确认 assistant 起点、要预测的下一个 token 和特征索引。off-by-one 不一定导致 shape error，却会让模型稳定地学习错误对应关系，是最隐蔽也最昂贵的数据问题之一。

## 一份 YAML 为什么能成为运行契约

SpecForge 当前将训练入口统一为：

```bash
specforge train --config path/to/run.yaml
```

公开接口不再是每种方法各自的一组 `train_*.py`。YAML 以 typed schema 组织 `model`、`data`、`training`、`tracking`、`profiling`、`runtime` 和 `deployment`，另有 `run_id` 与 `output_dir`。未知字段会报错，避免拼写错误被静默忽略。

下面不是可直接复制到任意模型的万能配置，而是展示一份运行记录需要表达哪些关系：

```yaml
model:
  target_model_path: Qwen/Qwen3-8B
  draft_model_config: configs/qwen3-8b-eagle3.json
  target_backend: sglang
  vocab_mapping_path: cache/vocab_mapping/qwen3-8b.pt
  torch_dtype: bfloat16

data:
  train_data_path: cache/dataset/target_regenerated.jsonl
  max_length: 4096
  chat_template: qwen

training:
  strategy: eagle3
  batch_size: 1
  learning_rate: 1.0e-4
  max_steps: 10000
  save_interval: 1000

run_id: qwen3-8b-eagle3-exp01
output_dir: outputs/qwen3-8b-eagle3-exp01

deployment:
  mode: disaggregated
  trainer:
    nnodes: 1
    nproc_per_node: 8
```

具体字段应以当前版本的示例配置和 schema 为准。这里重要的是：目标模型、数据、优化器与部署拓扑被记录在同一份配置中。实验差异可以通过配置 diff 审查，而不是从终端历史里猜测。

从旧训练脚本迁移时还要警惕默认值变化。即使命令行参数名称看起来相似，epoch、learning rate、warmup、最大长度、保存间隔与 buffer dtype 都可能不同。复现实验应显式写出关键值，并保存 SpecForge、SGLang 和目标权重的 revision。

## 并行拓扑要围绕瓶颈设计

训练侧常见的并行方式解决不同问题：

- Data Parallel 让不同 rank 处理不同样本；
- FSDP 切分参数、梯度和优化器状态；
- Tensor/Sequence Parallel 在受支持的模式中切分层内计算或序列维度；
- 目标模型自己的 TP/DP 则由在线 SGLang server 管理。

不能看到更多 GPU 就同时打开所有并行维度。EAGLE3 drafter 通常远小于目标模型，训练计算可能不是瓶颈；离线模式更可能卡在特征文件读取和解压，在线模式则可能卡在目标 rollout 或网络传输。

可用一条简单的生产者—消费者关系定位问题：

\[
T_{step}\approx\max(T_{feature\ producer},T_{transfer},T_{trainer})
\]

如果 trainer GPU 经常等待数据，继续增加 trainer rank 只会扩大饥饿。应先看特征生产速率、磁盘吞吐、网络带宽、队列占用和每步有效 token 数，再决定扩哪一侧。

## 在大规模训练前建立一条可证伪的检查链

一条可靠训练链路应能逐层回答“错在数据、模型还是系统”：

### 单样本对齐

选一条很短的对话，打印格式化文本、token、assistant mask、各层 feature shape 和预测标签。确认截断、特殊 token 和位置偏移。

### 极小数据过拟合

用少量样本训练到 loss 明显下降。若完全无法过拟合，优先检查 mask、feature alignment、vocab mapping 和参数是否真的进入 optimizer，不要直接扩大数据。

### 多步行为

分别统计第 1、2、3……层的 token 命中与接受率。第一层正常、后续层陡降，往往指向 TTT 路径、tree mask 或训练/推理实现不一致。

### 断点续训

验证恢复后 `global_step`、学习率、优化器状态、随机数状态和数据游标是否连续。只成功加载权重不等于恢复了同一次实验。

### 小规模 serving

在与训练完全一致的目标模型上加载 checkpoint，先跑 greedy case，再跑 sampling case。对比不开推测解码时的输出或分布，排除 tokenizer、词表映射和采样校正错误。

这组检查的意义在于让失败尽早发生。多机跑数小时后才发现 assistant mask 全为零，任何吞吐优化都没有价值。

## checkpoint 交付的不只是权重文件

一个能够交给 SGLang 的 drafter 目录，至少应附带以下元数据：

```text
target_model_id + immutable revision
tokenizer revision + chat template
reasoning mode / parser contract
draft architecture config
feature layer IDs + fusion dimensions
vocab mapping + special token IDs
training strategy + dtype
SpecForge / SGLang / PyTorch revisions
data generation parameters
evaluation report
```

加载阶段应主动验证维度、词表大小、特殊 token 和目标模型标识。若只依赖张量 shape，两个词表大小相同但 token 排列不同的模型仍可能“成功加载”，随后生成不可解释的候选。

部署验收可以按三层推进：

1. **兼容性**：checkpoint 能加载，配置、词表和 feature layer 匹配；
2. **正确性**：greedy 路径一致，sampling 分布通过统计检查，无非法 token；
3. **性能**：在目标请求分布上测延迟、吞吐、显存和尾延迟。

只有第三层变快且前两层成立，checkpoint 才算完成交付。

## 为什么更低的 token loss 不保证更快

训练 loss 通常对所有受监督位置取平均，而 serving 收益受候选树路径影响。两者之间至少隔着三层差异：

1. 一个位置的概率稍有改善，未必改变采样或 top-k 候选；
2. 树节点预算会决定哪些概率质量真正得到验证；
3. 深层 token 依赖之前的草稿输出，误差不是独立的。

两个 checkpoint 的平均 loss 接近，可能一个第一层更准，另一个在第三、第四层保持得更好。若树深较大，后者反而可能带来更长接受前缀。

因此模型选择最好使用同一份 held-out prompt 和同一套 serving 参数，输出至少包括：

```text
draft token loss / top-k coverage
acceptance rate by depth
accepted length distribution
drafted nodes per round
draft / verify / sampling latency
TTFT / TPOT / request throughput
peak memory and P99 latency
```

还应按领域、语言、输入长度、输出长度和 reasoning 状态分桶。全局平均值会掩盖 drafter 在某些流量上有效、另一些流量上频繁被拒绝的事实。

## 怎样理解论文中的加速数字

EAGLE3 论文报告过最高 6.5 倍的加速比，并在 SGLang、batch size 64 的实验中报告 1.38 倍吞吐提升。这些数字证明方法在论文设置中能够产生显著收益，但不是任何模型与硬件上的固定倍率。

复现实验时需要同时对齐：

- 目标模型和 drafter checkpoint；
- GPU 型号、精度和并行度；
- prompt/response 长度与数据集；
- temperature、候选树和最大 draft token；
- batch size、并发与调度策略；
- 统计口径是纯 decode、TPOT 还是请求吞吐。

尤其不能把 batch-1 延迟倍率直接用于高并发容量规划。高并发时，草稿计算和树验证会与 continuous batching、内存带宽及调度器交互，收益结构已经不同。

## 常见异常如何沿链路定位

| 现象 | 优先检查 | 原因 |
| --- | --- | --- |
| loss 不下降 | loss mask、feature/token 对齐、optimizer 参数 | 训练信号可能为空或错位 |
| 第一步接受率正常，深层迅速归零 | TTT 展开、tree mask、drafter 回馈路径 | 训练没有覆盖真实多步输入 |
| 离线训练正常，在线 serving 很差 | 目标 revision、chat template、reasoning 模式 | 训练与服务分布不一致 |
| checkpoint 能加载但 token 异常 | vocab mapping、special token IDs、tokenizer revision | shape 相同不代表语义对应 |
| trainer GPU 利用率低 | producer 速率、磁盘/网络、consumer 队列 | 特征供应成为瓶颈 |
| 接受长度增加但 TPOT 变差 | 树宽、draft 时间、verify token 数 | 为候选付出的成本超过收益 |
| greedy 正常、sampling 偏离 | 接受/拒绝概率与校正采样 | 确定性路径没有覆盖概率错误 |
| 只在长上下文退化 | 截断分布、位置编码、特征传输和树成本 | 训练长度或系统开销不匹配 |

这张表也说明为什么 SpecForge 不只是“训练一个小模型”的封装。大多数失败都发生在目标模型、数据、特征、训练和 serving 的边界上。

## 把整条链路收束成一个工程判断

训练 EAGLE3 的正确起点不是选择 learning rate，而是先定义线上契约：目标模型输出什么序列、SGLang 从哪些层取特征、drafter 要构造什么树、最终用什么负载评估。

确定契约后，数据生成与 loss mask 才有明确含义；在线或离线模式才有选择依据；checkpoint 的兼容性也能在加载阶段验证。SpecForge 的价值正是在这些环节之间建立统一配置和可交付路径。

最终是否上线，则应由端到端结果决定：在正确性不变的前提下，接受更多 token 是否真的抵消了草稿、传输和验证成本。能回答这个问题的实验，才完成了从 EAGLE3 原理到 SGLang 生产部署的闭环。

## 参考资料

- [SpecForge 官方仓库](https://github.com/sgl-project/SpecForge)
- [SpecForge 文档](https://sgl-project.github.io/SpecForge/)
- [SpecForge：Data Preparation](https://sgl-project.github.io/SpecForge/basic_usage/data_preparation.html)
- [SpecForge：Training](https://sgl-project.github.io/SpecForge/basic_usage/training.html)
- [EAGLE-3 论文](https://arxiv.org/abs/2503.01840)
- [EAGLE-2 论文](https://arxiv.org/abs/2406.16858)
- [SGLang 官方仓库](https://github.com/sgl-project/sglang)
