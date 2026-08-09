---
layout: post
title: "PagedAttention 与 vLLM KV Cache 管理"
subtitle: "一条请求的分块、映射、共享与回收"
date: 2026-03-17
last_modified_at: 2026-08-09
author: iStar
catalog: true
mathjax: true
tags: [AI Infra, LLM推理, KV Cache, vLLM]
---

LLM 服务很难提前知道一次请求最终会占用多少 KV Cache。用户可能只生成一句话，也可能持续输出几千个 token；不同请求又在不同时间到达和结束。如果为每个请求一次性预留最大长度，显存会被大量空槽占据；如果按实际长度反复寻找连续空间，则会出现碎片、搬移和扩容问题。

PagedAttention 的出发点不是改变注意力公式，而是改变 KV Cache 的地址组织方式：序列在逻辑上仍然连续，物理上却可以分散在固定大小的 block 中。attention kernel 通过 block table 找到这些 block，就像程序使用连续虚拟地址、操作系统再把它映射到不连续物理页。

分页只是基础。vLLM 的实际吞吐还依赖调度器、continuous batching、专用 kernel 和缓存复用。本文沿着一条请求的生命周期，解释这些部分如何协同。

## KV Cache 从哪里来

自回归生成通常分成两个阶段：

1. **Prefill**：一次处理 prompt 的多个 token，计算它们的隐藏状态与 K/V；
2. **Decode**：每轮处理一个或少量新 token，新 Query 读取全部历史 K/V。

如果没有 KV Cache，生成第 $t$ 个 token 时就要重新为前 $t-1$ 个 token 计算 K/V。缓存避免了这部分重复工作，但需要在每层保留历史状态。

设层数为 $L$、序列长度为 $S$、KV head 数为 $H_{kv}$、head dimension 为 $d_h$，每个元素占 $b$ 字节，单序列缓存主体约为：

$$
M_{KV}=2LSH_{kv}d_hb
$$

缓存随序列长度线性增长。服务端还要同时保存许多请求，因此真正受限的往往不是某条序列能否放下，而是在目标延迟下能并发容纳多少 token。

## 连续分配为什么浪费显存

假设模型最大上下文为 8192 token，请求 A 实际使用 300 token，请求 B 使用 5000 token。

一种简单做法是为每个请求预留 8192 个位置：

```text
Request A: [used 300][             reserved 7892             ]
Request B: [used 5000][       reserved 3192       ]
```

这样地址计算简单，却把尚未使用、甚至永远不会使用的空间锁住了。

另一种做法是只分配当前需要的连续区间，并在序列增长时扩容。问题类似动态数组和堆内存：相邻位置可能已被别的请求占用，系统只能搬移已有缓存或寻找更大的空洞。请求持续到达和结束后，总空闲量可能足够，却找不到足够大的连续区间，这就是外部碎片。

LLM 请求有三个特征，让这个问题格外突出：

- 长度事先未知；
- 每个 decode step 都可能增长；
- 请求结束时间不同，分配和释放非常频繁。

## 从 token 序列到固定大小 block

PagedAttention 把逻辑序列切成固定 token 数的块。假设每个 block 容纳 4 个 token，请求包含 10 个 token：

```text
logical block 0: token 0  1  2  3
logical block 1: token 4  5  6  7
logical block 2: token 8  9  _  _
```

这三个逻辑块可以映射到任意空闲物理块：

```text
logical blocks:  [0] [1] [2]
block table:      7  19   3

physical pool:  ...[3]...[7].........[19]...
```

当请求继续生成第 11、12 个 token，只需填满物理块 3 的剩余槽位；生成第 13 个 token 时才申请一个新块。请求不必在开始时知道最终长度，也不要求下一块与当前块物理相邻。

最后一个块仍可能有空槽，所以分页不是绝对零浪费。若 block size 为 $B_s$，单条序列尾部最多浪费 $B_s-1$ 个 token slot；相较为最大长度预留，浪费被限制在一个 block 内。

## Block 中实际保存什么

“每块 4 个 token”只是逻辑说法。物理 KV block 还包含所有相关层或某组层、K/V、KV heads 和 head dimension 对应的数据。不同引擎版本和 attention 类型可能采用不同布局。

可以把地址查找概括为：

```text
(request, logical token position)
              │
              ├─► logical_block = position // block_size
              ├─► offset        = position %  block_size
              ├─► physical_block = block_table[logical_block]
              └─► KV address = layout(layer, K_or_V,
                                      physical_block, offset,
                                      kv_head, head_dim)
```

PagedAttention kernel 需要按 block table 收集 K/V。与连续张量相比，这增加了间接寻址和元数据处理；但换来了更高的有效缓存容量和动态分配能力。kernel 的任务就是让这层间接性不会抵消内存管理收益。

## 一条请求的完整生命周期

### 1. 到达与 token 化

请求先被 tokenizer 转成 token IDs。调度器知道 prompt token 数、最大输出长度、优先级等信息，但不知道请求会在何时遇到停止词，也不知道模型实际生成多长。

### 2. 查询已计算前缀

若启用 automatic prefix caching，KV Cache manager 会按完整 token block 计算哈希，查找是否已有相同前缀。命中部分无需重新 prefill。

### 3. 预留写入槽位

调度器确定本轮允许计算多少 token；cache manager 检查需要多少新块，从 free block queue 中分配物理块，并生成 slot mapping 或 attention metadata。

### 4. 模型执行

worker 对本轮 token 做 forward，将新 K/V 写入分配好的 slot。PagedAttention kernel 按每个请求的 block table 读取历史 K/V。

### 5. 更新状态

执行完成后，系统更新已计算 token 数、采样结果和 block 状态。部分块填满后可以进入前缀缓存索引；未填满的尾块继续属于当前请求。

### 6. 继续、抢占或结束

若请求继续生成，下一轮复用现有 blocks，必要时再申请新块。若资源不足，调度器可能让请求等待或抢占；请求完成、取消且 block 不再被其他请求引用后，blocks 返回空闲队列。

简化的数据流是：

```text
request arrives
      │
      ▼
lookup computed prefix ──► reserve blocks / slots
      │                              │
      └──────────────────────────────┘
                                     ▼
                           schedule model forward
                                     │
                                     ▼
                         write new K/V into slots
                                     │
                 ┌───────────────────┴───────────────────┐
                 ▼                                       ▼
          request continues                       request finishes
          append/allocate                         decrease ref count
                 │                                free or keep cached
                 └──────────── next step ────────────────┘
```

## 分页与 continuous batching 的协同

Continuous batching 允许每个调度 step 移除已完成请求，再加入等待队列中的新请求。若缓存必须连续分配，batch 频繁变化会带来昂贵的重排；分页后，请求只需携带自己的 block table，物理块可以独立分配和回收。

调度器和缓存管理器因此必须共同决定本轮工作：

- 调度器有 token budget，决定多少 prefill/decode token 可以执行；
- cache manager 有 block budget，判断这些 token 是否有写入空间；
- worker 需要两者生成的 metadata，才能读写正确地址。

只有 PagedAttention kernel，没有与之配套的动态调度，并不能自动得到 vLLM 的整体吞吐；反过来，调度器若忽略缓存容量，也可能过量接纳请求并触发频繁抢占。

## Automatic Prefix Caching 如何复用 block

很多服务请求共享长 system prompt、工具定义或同一文档。相同前缀的 K/V 与后续采样参数无关，可以被复用，从而省掉重复 prefill。

vLLM 的哈希式前缀缓存可以把第 $i$ 个 block 的身份理解为：

$$
H_i=\operatorname{hash}(H_{i-1},\ tokens_i,\ extra)
$$

父 block 哈希把此前完整前缀纳入身份，`tokens_i` 是当前 block 的 token，`extra` 则应包含会影响 K/V 的其他信息，如 LoRA adapter、多模态输入哈希或缓存隔离 salt。

只有完整 block 容易安全命中。设 block size 为 4：

```text
Request A: [A B C D] [E F G H] [I J _ _]
Request B: [A B C D] [E F G X] [...]
```

两者只能共享第一块。第二块最后一个 token 不同，从这里开始后续 K/V 都处在不同上下文中；A 的不完整尾块也不作为完整前缀共享。

共享块带有引用计数。多个请求引用时，物理块不能被释放或覆盖；引用归零后，它可以暂时留在缓存索引中等待未来命中，也可以在需要空间时被淘汰。

### 共享不应突破租户边界

跨请求前缀命中会造成时间差：命中者的 prefill 更快。多租户服务若无隔离，攻击者可能利用延迟推测某段前缀是否已缓存。官方设计提供 cache salt，将首块哈希与指定信任域绑定。缓存正确性不仅是“token 相同”，还包括模型状态一致与安全边界一致。

## Copy-on-Write 与分支解码

在 beam search 或并行采样中，多个候选序列起初共享相同前缀。物理复制全部 KV 会造成浪费，因此论文设计允许逻辑 block tables 指向相同物理块，并用引用计数管理。

当分支要修改共享尾块时，系统需要分配新块并复制必要内容，即 copy-on-write；已经填满且只读的历史块则可继续共享。具体实现会随引擎版本和解码路径演进，但不变量是：任何写入都不能破坏其他序列看到的历史 K/V。

## Block size 是一组折中

block 越大：

- block table 更短，地址与元数据开销更小；
- kernel 访问可能更规整；
- 但每条请求最后一块的内部浪费更大；
- 前缀只有达到更粗粒度边界才能命中。

block 越小：

- 尾部浪费与前缀匹配粒度更小；
- 但 block 数、哈希、引用计数、调度 metadata 和间接寻址更多。

因此 block size 不是一个应盲目调到最小的旋钮。它还可能受具体 kernel、dtype、模型 head dimension 与硬件对齐约束。使用公开 serving 接口时，通常让引擎选择经过支持的配置比依赖内部类更稳妥。

## 缓存不足时会发生什么

当 free blocks 不足，系统不能继续给所有请求分配 slot。可选策略包括：

- 让新请求继续排队；
- 抢占正在运行的请求并释放其 blocks；
- 恢复时重新执行部分 prefill；
- 将缓存 offload 到 CPU 或外部层级，之后再取回；
- 在分布式 prefill/decode 架构中传输 KV。

这些方案是在容量、计算与传输之间交换成本。重计算浪费 GPU FLOPs，但 PCIe 或网络较慢时，可能比 swap 更合适；offload 保留计算结果，却可能增加尾延迟。监控中若只看 GPU 利用率，很难分辨请求是在有效 decode，还是因反复 preemption 重算 prompt。

## 如何验证内存管理是否真的有效

可以设计三类流量，而不是只跑一个吞吐数字：

### 长短请求混合

混合短 prompt/短输出与长 prompt/长输出，逐步提高并发，记录：

- GPU KV cache usage；
- waiting/running request 数；
- preemption 或 recomputation 次数；
- TTFT（首 token 延迟）与 TPOT（输出 token 间延迟）；
- 吞吐和满足延迟目标的 goodput。

这组实验主要观察动态分配和调度。

### 共享前缀与随机前缀

一组请求复用相同长 system prompt，另一组使用相同长度但随机 token 的 prompt。两组分开测，才能把“分页提高可用容量”与“prefix caching 省掉 prefill”区分开。

### 压力与取消

在接近缓存上限时插入超长请求，并随机取消部分请求。检查 block 是否及时回收、队列是否恢复，以及持续运行后是否出现容量泄漏或异常碎片。

论文中的 2—4 倍吞吐来自当时特定模型、硬件、请求分布和对比系统，不能作为任意现代版本的保证。今天的 vLLM 还包含新的调度器、编译、不同 attention backend、混合 KV 类型与分布式缓存连接器；任何性能结论都应附带版本、配置和 workload。

## 小结

PagedAttention 的核心价值，是把“序列连续”与“显存连续”解耦。固定大小 block 限制了尾部浪费，block table 让请求动态增长，引用计数和哈希又让相同前缀能够安全复用。

但分页不是独立魔法。请求能否高效运行，取决于调度器是否按 token 和 block 两种预算接纳工作，kernel 是否高效完成间接寻址，以及缓存不足时采用怎样的抢占、重算或传输策略。沿着“分配 slot—写入 K/V—更新 block table—下一轮读取—结束回收”这条路径，就能定位大多数 KV Cache 管理问题。

## 参考资料

- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- [vLLM 官方文档](https://docs.vllm.ai/)
- [vLLM Automatic Prefix Caching 设计](https://docs.vllm.ai/en/latest/design/prefix_caching/)
- [vLLM 官方仓库](https://github.com/vllm-project/vllm)
