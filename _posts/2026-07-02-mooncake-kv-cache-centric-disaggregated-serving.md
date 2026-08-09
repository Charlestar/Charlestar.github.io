---
layout: post
title: "Mooncake：让 KV Cache 跨 Prefill 与 Decode 流动"
subtitle: "从分离式推理、分布式缓存池到缓存感知调度与高速传输"
date: 2026-07-02 09:00:00 +0800
last_modified_at: 2026-08-09
author: iStar
catalog: true
series: kv-cache-memory
series_order: 40
technology_year: 2024
mathjax: true
tags: [AI Infra, Mooncake, KV Cache, 分离式推理, LLM推理]
---

KV Cache 最初看起来只是单个推理进程中的一块显存：prefill 计算 prompt 的 K/V，decode 在生成每个 token 时继续读取并追加它。但当服务扩展到多机、多副本和长上下文后，KV Cache 不再只是 attention kernel 的输入，它还决定请求应该发到哪里、是否值得复用、怎样跨机器移动，以及 GPU 之外的大量内存和存储能否参与推理。

Mooncake 把这个变化推到了系统架构层。它将 prefill 与 decode 放入不同资源池，并利用 GPU 集群中的 CPU、DRAM、SSD 与高速网络构建分布式 KV Cache；全局调度器不再只比较哪台 GPU 的队列最短，而是同时考虑缓存位置、可复用前缀、传输开销、prefill 负载与 TTFT/TBT SLO。

因此，Mooncake 不能只概括成“P/D 分离”或“用 RDMA 传 KV”。这两项都是手段，真正的设计中心是：**把 KV Cache 视为可以寻址、复用、搬运、复制和淘汰的分布式数据，再围绕它安排计算。**

## 一条请求为什么会分成两种负载

Decoder-only Transformer 的一次请求包含两个计算形态差异很大的阶段。

### Prefill

Prefill 一次处理整个输入序列。对长度为 $S$ 的 prompt，它要为所有位置构造表示并写出每层 K/V。矩阵较大，并行度高，通常更偏 compute-bound；长上下文 attention 的计算和中间状态还会显著增加 TTFT。

### Decode

Decode 每轮只为每个请求生成一个新 token，却要读取大量模型权重和历史 KV Cache。低到中等 batch 下，它更容易受 HBM 带宽限制。用户持续感知的 token 间隔取决于这个阶段。

把两者放进同一 worker 的好处是数据天然本地：prefill 写出的 KV 已经在当前 GPU 上，decode 可以立刻使用。问题在于两种 workload 会相互干扰：

```text
同一 GPU 时间线

decode step ─ decode step ─ long prefill ─ decode step ─ decode step
                            ▲
                            └─ 已在生成的请求等待，TBT 出现尖峰
```

Continuous batching 与 chunked prefill 能缓解阻塞，却不能让一组 GPU 同时针对两种计算形态独立选取并行策略、批大小和资源比例。

Mooncake 把它们拆成两个池：

```text
incoming request
       │
       ▼
  Prefill Pool                     Decode Pool
┌────────────────┐   KV Cache   ┌────────────────┐
│ compute prompt │ ───────────► │ generate token │
│ optimize TTFT  │              │ optimize TBT   │
└────────────────┘              └────────────────┘
```

这样 prefill worker 可以面向长序列计算，decode worker 可以维持稳定的 continuous batch。代价也立刻出现：原来在同一张卡上的 KV Cache，现在必须从 P 端送到 D 端。

## P/D 分离的账首先是一笔数据搬运账

设模型有 $L$ 层，KV head 数为 $H_{kv}$，每个 head 维度为 $D$，缓存元素字节数为 $B$。每个 token 的 KV 大小近似为：

$$
M_{token}=2LH_{kv}DB
$$

前面的 2 分别代表 K 与 V。长度为 $S$ 的 prompt 产生：

$$
M_{prompt}=2SLH_{kv}DB
$$

若传输有效带宽为 $BW_{eff}$，完全串行的数据移动下界为：

$$
T_{transfer}\ge
\frac{M_{prompt}}{BW_{eff}}
$$

这还没有计算注册内存、控制消息、排队、跨 NUMA、GPU/CPU copy 和尾部小传输。长上下文与 MHA 模型会让 KV 很大，P/D 分离并不会自动获利。

它成立通常依赖三个条件：

1. 高速数据通路能让传输明显快于重复 prefill；
2. 传输可与逐层 prefill、GPU 计算或异步加载重叠；
3. 分离后更稳定的 decode batching 和资源隔离足以覆盖通信成本。

因此，判断标准不是“网络标称 400 Gbps”，而是端到端关键路径中还剩多少不可隐藏的 KV 传输时间。

## 从 GPU 本地 Cache 到分布式 KV Cache 池

如果 KV 只从一个 prefill worker 点对点发送给一个 decode worker，系统解决了阶段交接，却没有解决跨请求复用。多轮对话、公共 system prompt、长文档问答和共享 few-shot 示例都可能重复使用相同前缀；当下一次请求落在另一台机器时，本地缓存就无法命中。

Mooncake 将 GPU 节点中经常未被充分使用的 CPU、DRAM、SSD 与 NIC 组织成分布式缓存池：

```text
                     Global metadata / Conductor
                              │
       ┌──────────────────────┼──────────────────────┐
       ▼                      ▼                      ▼
Node A                    Node B                 Node C
GPU HBM                   GPU HBM                GPU HBM
CPU DRAM                  CPU DRAM               CPU DRAM
local SSD                 local SSD              local SSD
       └──────────── high-speed fabric ──────────────┘
```

HBM 仍是 attention 直接消费 KV 的位置，但不再是唯一保存位置。冷一些或暂时不用的缓存可以留在 CPU DRAM/SSD，其他推理实例需要时再取回；同一个热门前缀还能在多个节点保留副本，避免所有请求争用一个数据源。

这是一种“以更多存储换更少重复计算”的选择。它并不让 prefill 计算消失，而是让已经付出过的 prefill 成本更有机会跨请求、跨 worker 和跨时间复用。

## KV 块为什么必须连同前缀一起哈希

KV Cache 的值不仅由当前 token block 决定，还由它之前的所有 token 决定。字符串相同的一块 token，出现在不同上下文后，其 contextual K/V 通常不同。

假设两个序列都含有 token block `B`：

```text
sequence 1: A → B
sequence 2: X → B
```

由于 causal attention 的历史不同，`KV(B | A)` 不能当作 `KV(B | X)` 复用。因此，缓存 key 不能只计算当前块内容：

$$
k_i=H(k_{i-1},\operatorname{tokens}(block_i),\text{model identity})
$$

递归 prefix hash 让第 $i$ 块的 key 同时承诺当前 token 与前面完整 block 链。真实实现通常还需要把以下信息放进 namespace 或兼容身份：

- 模型与精确权重 revision；
- tokenizer 与 chat template 形成的实际 token 序列；
- adapter/LoRA 身份；
- 影响 hidden state 的模型配置和位置编码条件；
- KV 数据类型、布局与并行分片方式。

如果只比较自然语言文本，忽略模板插入的 special token，缓存可能看似命中却对应不同 token 前缀。正确性要求复用以模型真正消费的 token 和执行身份为准。

论文/FAST 版本中的 Mooncake Store 把 KV 组织为 paged blocks，block key 包含自己的内容和 prefix 信息，并允许同一 key 有多个副本。块大小要兼顾命中粒度、元数据规模和网络传输效率：块越小，部分前缀命中更精细；块越大，批量传输更有效但尾部浪费更明显。

## 一条请求在 Mooncake 中怎样流动

Mooncake 的全局调度器名为 Conductor。请求完成 tokenization 后，它选择 prefill instance/group 与 decode instance，并协调四个阶段。

```text
                   ┌──────────────────────────┐
                   │        Conductor         │
                   │ cache + load + SLO view  │
                   └────────────┬─────────────┘
                                │
      1. reuse prefix           │        select P and D
                 ┌──────────────▼──────────────┐
KV cache pool ─► │        Prefill Worker       │
                 │ 2. incremental prefill      │
                 └──────────────┬──────────────┘
                                │ 3. stream new KV
                                ▼
                 ┌─────────────────────────────┐
                 │         Decode Worker       │
                 │ 4. continuous-batch decode │
                 └─────────────────────────────┘
```

### 1. 复用已有前缀

Conductor 找到请求可复用的最长 KV 前缀，以及这些块当前位于哪些节点。选中的 prefill worker 把相应块载入 GPU，用它们作为增量 prefill 的起点。

命中最多的节点不一定就是最佳节点。如果它的队列很长，等待缓存本地性可能比把 KV 传到空闲节点或直接重算更慢。调度器需要比较：

$$
T_{TTFT}
\approx T_{queue}+T_{cache\ load}
+T_{uncached\ prefill}+T_{handoff}
$$

### 2. 只计算未命中的输入

Prefill worker 跳过已复用前缀，对剩余 token 做 incremental prefill。长输入可以切成 chunk；更长或单节点难以满足 TTFT 的请求，还可以使用多节点 prefill 组织方式。

新生成的 KV 不只服务当前请求，也可以写入分布式缓存池，成为后续请求的前缀缓存。

### 3. 把新 KV 流向 Decode

如果等所有层的 prefill 都完成后再一次性发送，网络传输会完整落在 TTFT 关键路径上。Mooncake 采用 layer-wise 的加载/存储与流式传输：某层 KV 产生后即可向目标 decode 节点移动，同时 prefill GPU 继续计算后续层。

```text
time ─────────────────────────────────────────►

prefill GPU:  layer 1 | layer 2 | layer 3 | layer 4
network:              send L1 | send L2 | send L3 | send L4
decode DRAM:          recv L1 | recv L2 | recv L3 | recv L4
```

如果网络和计算能够充分重叠，不可隐藏延迟接近 pipeline 的尾部，而不是所有层传输时间之和。

### 4. 加入 Decode continuous batch

完整 KV 到达 decode 节点 CPU 内存后，再异步加载进 HBM。请求进入 decode worker 的 continuous batch，开始逐 token 生成。

Conductor 会提前按 decode 负载与 TBT SLO 选择节点，本地 scheduler 仍需在真正入队时复核，因为 prefill 期间队列状态可能已变化。这体现了分离式系统的一项难题：P 调度与 D 入队之间有时间差，早先预测可能过期。

## PagedAttention、RadixAttention 与 Mooncake 在不同层工作

三者经常一起出现，但不能互相替代。

### PagedAttention：一台引擎内怎样放置请求 KV

PagedAttention 将请求的逻辑 KV block 映射到非连续物理显存块，减少预留和外部碎片，并支持 block 级共享/回收。它主要解决单个 serving engine 的 HBM 管理与 attention 访问。

### RadixAttention：哪些 token 前缀可以跨请求复用

RadixAttention 用 radix tree 索引请求前缀，进行最长前缀匹配与缓存感知调度。它表达的是 token prefix 的逻辑共享关系。

### Mooncake：缓存怎样跨实例、跨存储层移动和调度

Mooncake 把 KV block 放到分布式内存/存储中，通过高速数据通路移动，并让全局 scheduler 根据位置和负载安排请求。当前开源生态中，Mooncake Store 还可以作为 SGLang HiCache 的外部存储后端，将 RadixAttention 的复用扩展到设备、主机和远端层级。

可以把它们叠起来看：

```text
logical prefix identity      Radix tree / prefix hash
             │
             ▼
local GPU block placement    paged KV manager
             │
             ▼
host / remote persistence    Mooncake Store
             │
             ▼
cross-node data movement     Transfer Engine
             │
             ▼
global request placement     Conductor / serving scheduler
```

实际产品的模块边界会因引擎版本而不同，但这组分层能避免把“支持 block cache”误解为“已经拥有分布式缓存系统”。

## Mooncake Store 管理的不只是一个地址表

一个分布式 KV Cache 至少要处理对象身份、物理位置、生命周期和并发访问。FAST 论文中的 Store 提供 `put`、`get`、`change_replica` 一类对象级操作，并在底层使用批量内存传输接口。

### 副本与热点

系统 prompt 等热门前缀可能被大量请求同时读取。即使数据已缓存，若只有一个副本，源节点 NIC 和 DRAM 带宽仍会成为热点。

多副本把读取分散到多个节点：

```text
              hot block K
            /      |      \
       replica A replica B replica C
          │          │          │
      reader set  reader set  reader set
```

副本不是越多越好。每个副本都占容量和复制带宽，还需要在热度下降时回收。Mooncake 的缓存负载均衡策略会迁移或复制热点块，让调度器更容易同时获得高命中与低排队。

### 淘汰与正在使用的块

缓存池满时可以使用 LRU 等策略淘汰，但正在被请求读取的 block 不能突然覆盖。系统需要 pin/reference 状态，区分：

- 可淘汰的冷 block；
- 传输中的 block；
- 正在被 prefill/decode 使用的 block；
- 新写入但元数据尚未对外可见的 block。

数据写入与目录发布也要有顺序。若 metadata 先宣告对象可用，远端 reader 可能读到尚未完成的 KV；若节点失败，未完成对象必须可识别并清理。

### DRAM 与 SSD 不是 HBM 的透明替代品

低层缓存容量更大，但延迟和带宽更差。命中远端 SSD 的前缀不一定比 GPU 重算快，特别是前缀很短或网络拥塞时。调度决策应比较：

$$
T_{reuse}=T_{lookup}+T_{transfer}+T_{load}
$$

与：

$$
T_{recompute}=T_{queue}+T_{prefill}
$$

只有 $T_{reuse}<T_{recompute}$ 且不破坏其他请求 SLO，复用才是好选择。

## Transfer Engine 要解决哪些数据面问题

论文中的高速传输层在后续开源项目中形成 Mooncake Transfer Engine。它不是为 KV 语义做匹配的 scheduler，而是负责让大块 tensor 在异构设备和网络之间高效移动。

### Zero-copy 与内存注册

传统路径可能经历多次复制：

```text
GPU → application buffer → kernel/network buffer
    → remote host buffer → application buffer → GPU
```

RDMA/GPUDirect 路径尽量让 NIC 直接访问已注册的 host/GPU memory，减少 CPU copy 和上下文切换。但注册内存本身有成本，地址生命周期、权限和设备支持也必须管理；不能把 zero-copy 理解为完全没有任何数据移动。

### 多 NIC 带宽聚合

长上下文 KV 是大对象，单条链路可能成为瓶颈。Transfer Engine 可以把对象切分并在多个 NIC/path 上并行发送。理想总带宽接近各路径之和，实际受以下因素限制：

- 源/目标 DRAM 或 HBM 带宽；
- PCIe/NVLink 与 NUMA 拓扑；
- 每条网络路径拥塞；
- 分片大小和尾部不均衡；
- endpoint 数、queue pair 与 CPU polling 开销。

### 拓扑感知路径

某块 GPU 与某张 NIC 可能位于不同 NUMA socket，跨 socket 搬运会消耗互连带宽。拓扑感知会优先选择更接近源和目标 memory 的设备路径，而不是把所有 NIC 当作等价端口。

### 批量与异步完成

KV 通常由许多 layer/block tensor 构成。逐个发起小请求会让控制面和网络提交开销占比过高。批量接口聚合操作，异步 completion 则让 layer-wise prefill、network 和 decode load 形成 pipeline。

### 故障与重试

缓存是可重建数据，传输失败不应悄悄返回损坏结果。数据面需要报告明确状态，并在可行时改走备用路径；上层可以选择其他副本、重传或回退到 prefill 重算。校验、超时和幂等对象身份共同保证“失败最多损失性能，不损失生成正确性”。

## Cache-aware 调度不是“命中最长者优先”

假设请求在节点 A 可复用 20k token，在节点 B 只能复用 12k token：

```text
Node A: cache hit 20k, queued prefill 8 s
Node B: cache hit 12k, queued prefill 0.5 s
```

只按最长命中会选 A，但 B 也许更早完成。Mooncake 的思路是估算每个候选 instance 上的 prefix match、未命中 prefill 时间和排队时间，选择预计 TTFT 更短、又满足约束的位置。

可以写成简化目标：

$$
i^*=\arg\min_i
\left(
\widehat T_{queue,i}
+\widehat T_{load,i}
+\widehat T_{prefill}(S-P_i)
\right)
$$

其中 $P_i$ 是该节点可复用的前缀长度。论文使用离线 profile 拟合 prefill 执行时间；传输时间更难估计，因为它受实时网络拥塞和源节点热点影响。

Decode 侧目标又不同：希望组成足够大的 batch 提高吞吐，同时受 TBT SLO 与 HBM 可容纳 KV 总量约束。于是 Conductor 实际在协调两个不同优化问题：

```text
Prefill placement:
maximize useful cache reuse
subject to TTFT / compute / DRAM constraints

Decode placement:
maximize sustainable batching and throughput
subject to TBT / HBM constraints
```

P/D ratio 也不能写死。输入很长、输出很短的流量需要更多 prefill 能力；短输入、长输出会占用 decode worker 更久。资源池比例应该由真实 token 分布和 SLO profile 推导，并随负载变化重新评估。

## 过载时为什么“晚拒绝”会浪费更多

大量 serving 论文默认所有到达请求最终都会执行。商业服务遇到持续过载时，这个假设不成立：若排队会让请求必然违反 SLO，继续接受只会让更多请求一起变慢。

P/D 分离又带来新的时间差。请求可能已经花费昂贵 prefill，等 KV 送到 D 池才发现 decode 已无容量。此时拒绝意味着 prefill 计算和数据传输全部浪费。

```text
admit request → expensive prefill → transfer KV → D overloaded → reject
                    └──────── wasted work ────────────────┘
```

Mooncake 研究了 early rejection，并进一步预测请求完成 prefill 后的未来 decode load，在 P 阶段开始前判断能否满足后续约束。

然而，简单早拒绝也可能引起负载振荡：P 池看到 D 忙而大量拒绝；一段 prefill 延迟后 D 变空，但此时没有足够请求接上；系统又转为大量接收，稍后 D 再次拥塞。控制器因此要面对延迟反馈，而不是只比较瞬时队列长度。

预测输入至少包括：

- 当前 P/D 队列与 active batch；
- 已接收请求的预计 prefill 完成时间；
- decode 输出长度或剩余时间估计；
- KV 容量与可用 block；
- SLO deadline 和服务等级。

输出长度不可精确预知，所以 admission policy 还需要保守裕量、在线校准和降级策略。早拒绝的目标不是追求更高“总接收数”，而是避免把资源消耗在已经不可能按约完成的请求上。

## Prefix Cache 命中率本身也不够

分布式缓存常用 hit ratio 作为核心指标，但“命中一个 block”和“省下一次 100k-token prefill”的价值完全不同。至少还要看：

### Token-weighted hit ratio

$$
H_{token}=
\frac{\sum_r \text{reused tokens}_r}
{\sum_r \text{input tokens}_r}
$$

它比 request hit ratio 更接近节省的计算量。

### Recompute time saved

不同长度位置的 prefill 成本不完全线性，模型和并行配置也不同。用离线 profile 估算被复用 token 实际省下的 GPU 时间，比单纯 token 数更准确。

### Reuse net benefit

$$
G_{reuse}=T_{recompute\ saved}
-(T_{lookup}+T_{move}+T_{load}+T_{contention})
$$

命中了但远端读取更慢，净收益可能为负。

### Effective capacity / SLO goodput

最终应统计在 TTFT 与 TBT 都满足约束时完成的请求或 token。缓存系统可能提高平均吞吐，却因为热点或尾延迟让更多请求违反 SLO；这种情况下不能只展示总 tokens/s。

## 一套能够定位瓶颈的观测面

### 请求阶段

```text
tokenize
global schedule / queue
prefix lookup
remote cache read
incremental prefill
P→D transfer
D-side HBM load
decode queue
decode
```

每段都需要 trace span，并记录重叠关系。简单把耗时相加会重复计算被隐藏的 transfer。

### Cache

- request hit ratio 与 token-weighted hit ratio；
- 命中层级：HBM、local DRAM、remote DRAM、SSD；
- block 热度、副本数与热点源节点；
- eviction、pin、写入失败和 stale metadata；
- 每个 model/adapter namespace 的容量占用；
- 复用后节省的估计 prefill GPU 时间。

### Transfer

- payload bytes 与 goodput，而非只看链路标称带宽；
- 注册、排队、发送、完成各段延迟；
- NIC/NUMA/path 分布与重试；
- P→D、cache read、cache write 分开统计；
- 与 prefill 重叠的字节比例和不可隐藏尾部时间。

### Scheduler

- 因 cache locality、queue 或 SLO 各自做出的 placement 数；
- 预测 TTFT/TBT 与实际值误差；
- P/D pool 利用率与资源比例；
- admission、early rejection 与晚拒绝；
- 因 HBM KV 容量而无法加入 batch 的请求。

没有这些维度，TTFT 变差时无法判断是缓存没命中、命中位置太远、网络热点、P 队列拥塞，还是 D 端 HBM 已满。

## 哪些场景更适合 Mooncake

收益更明显的 workload 通常具有：

- 很长的输入和相对短的输出；
- 多轮对话或共享 system/document prefix；
- prefill 会干扰 decode TBT；
- 集群拥有可用的 host memory、SSD 与高速网络；
- 流量规模足以让全局缓存与独立 P/D 池摊薄复杂度；
- TTFT/TBT SLO 明确，需要按 goodput 而非裸吞吐调度。

不一定适合直接引入完整分布式架构的情况包括：

- 单机或小集群，本地 KV 已能覆盖大部分请求；
- 输入短且几乎无前缀复用，查找与搬运不如重算；
- 网络带宽低或竞争严重，KV 交接占据关键路径；
- 模型 KV 已因 MQA/GQA/MLA 极度压缩，传输不是主要问题；
- 请求隐私或租户隔离不允许跨边界共享缓存；
- 运维系统无法可靠维护全局 metadata、RDMA 与多层缓存一致性。

“KV 是可重建数据”降低了持久性要求，却不降低在线正确性要求。错误复用会直接污染模型输出，比普通 cache miss 更危险；设计应优先保证 identity、生命周期和失败回退，再追求命中率。

## 论文系统与当前开源组件不要混写

Mooncake 论文描述的是 Kimi 的 KVCache-centric serving architecture，包括 P/D pools、Conductor、分布式缓存、长上下文 prefill 和过载调度。开源仓库此后继续演进，包含：

- **Mooncake Transfer Engine**：跨 DRAM/VRAM/NVMe、不同网络与设备的数据移动框架；
- **Mooncake Store**：基于传输引擎的分布式 tensor/KV Cache 存储与管理；
- 与 vLLM、SGLang/HiCache 等 serving runtime 的集成；
- 之后扩展出的训练、MoE 通信和其他 tensor infrastructure 组件。

因此，部署一套 Transfer Engine demo 不等于复现了论文中的完整 Conductor 与过载策略；把 Mooncake Store 接入某个引擎，也不表示自动获得论文的全部 P/D scheduling。评估和写配置时应注明具体组件、版本、connector 与 control plane。

同样，论文中的性能结果属于特定版本、硬件、模型和 trace。早期技术报告与 FAST 2025 版本的实验设置和数字也有更新。引用“最高提升”时必须带上来源版本和 workload，不能把模拟场景中的峰值当成任意线上系统的预期。

## 如何验证自己的 P/D + KV Store 设计

### 先测传输是否值得

对实际模型生成不同 prompt length 的 KV，测量：

```text
HBM → local DRAM
local DRAM → remote DRAM
remote DRAM → HBM
end-to-end P → D handoff
```

分别扫描并发流、块大小、NIC 数与 NUMA 绑定，并与重新 prefill 的延迟比较。需要的是 payload goodput 和不可隐藏时间，而不是 microbenchmark 的单向峰值。

### 再测缓存正确性

构造以下对照：

- token 块相同、前缀不同，必须 miss；
- 文本相同、chat template/tokenizer 不同，必须隔离；
- model revision 或 adapter 不同，必须隔离；
- 命中部分 block 后的 logits 与完整重算对比；
- block 在传输中被淘汰、节点宕机和 metadata 过期；
- 副本读取失败后切换其他副本或重算。

数值比较要允许后端布局和浮点 kernel 的合理误差，但生成语义不能因错误 block 身份而变化。

### 最后回放真实到达过程

保留请求到达时间、输入/输出长度、prefix relation 和租户边界，比较：

```text
colocated baseline
P/D only without cross-request store
P/D + local host cache
P/D + distributed KV store + cache-aware scheduler
```

观察不同 QPS 下 TTFT、TBT、P99、SLO goodput、网络、HBM/DRAM 容量和拒绝率。逐层加组件能够回答收益究竟来自 P/D 隔离、cache reuse，还是 scheduler placement，而不是把所有提升都归因于“Mooncake”。

## 小结

Mooncake 将 KV Cache 从进程内部的 attention 状态提升为集群级的数据对象。Prefill 与 decode 分离后，KV 成为两类计算资源之间的交接物；多轮和共享前缀又要求它能跨请求保存与复用；分布式缓存池、Store 和 Transfer Engine 负责放置与移动，Conductor 则用缓存位置、负载预测和 TTFT/TBT SLO 决定计算应该发生在哪里。

这套架构的核心取舍是“用存储与网络换重复 prefill 计算”。它是否成立取决于有效传输带宽、计算/通信重叠、前缀复用价值和调度质量。缓存命中只是起点，真正有意义的是扣除查找、搬运、加载和热点竞争后，节省了多少 prefill 时间，又让多少请求同时满足 TTFT 与 TBT。

沿着 KV Cache 系列看，PagedAttention 解决 GPU 内块的分配，RadixAttention 组织可复用前缀，Mooncake 再把这些块带到跨实例、跨存储层和全局调度的范围。下一层问题不再是“KV 放不放得下”，而是“整个集群中的 KV 在哪里，以及计算是否应该去找它”。

## 参考资料

- [Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving](https://arxiv.org/abs/2407.00079)
- [Mooncake: Trading More Storage for Less Computation（FAST 2025）](https://www.usenix.org/conference/fast25/presentation/qin)
- [Mooncake 官方仓库](https://github.com/kvcache-ai/Mooncake)
- [Mooncake Transfer Engine 文档](https://kvcache-ai.github.io/Mooncake/transfer-engine/)
- [Mooncake Store 文档](https://kvcache-ai.github.io/Mooncake/mooncake-store/)
