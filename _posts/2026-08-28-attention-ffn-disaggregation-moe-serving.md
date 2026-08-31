---
layout: post
title: "Attention–FFN 解耦：一层 Transformer 为什么要跨两类 GPU Pool"
subtitle: "从 KV 状态、MoE Expert 权重到逐层 Activation 传输，理解 AF Disaggregation 的容量与通信边界"
date: 2026-08-28 09:00:00 +0800
last_modified_at: 2026-09-01
author: iStar
catalog: true
series: moe-communication
series_order: 60
technology_year: 2025
mathjax: true
tags: [MoE, 专家并行, 分布式推理]
---

大模型推理的资源解耦正在向 Transformer layer 内部推进。P/D Disaggregation 把 Prefill 和 Decode 放到不同 GPU pool，是按请求生命周期拆阶段；Attention–FFN Disaggregation（AFD）则在一次 decode step 的每一层中，把有状态的 Attention 与无状态的 FFN 或 MoE experts 放到不同设备，是按算子和状态所有权拆组件。

它试图解决一个真实矛盾：decode attention 持有不断增长的 KV Cache，通常受显存容量与带宽限制；MoE FFN 持有庞大的 expert weights，只有汇聚足够多 token 才能形成高算术强度。把两者绑在同一组 GPU 上，副本数、batch size 和扩容比例只能一起变化，很难同时适合两类资源需求。

但 AFD 不是“再做一次 P/D”那么简单。P/D 通常只在阶段切换时传一次 KV；AFD 需要在每个 decode step、每个 MoE layer 传 activation，随后把 expert 输出送回 Attention 侧。网络、同步 barrier 与负载抖动由辅助开销变成了模型前向本身的一部分。

这篇文章从一层 Transformer 的数据流开始，说明 AFD 为什么可能提高 FFN 利用率、Attention/FFN 比例该怎样估算，以及为什么标准集群上会出现通信 dead zone。结论不是“越解耦越先进”，而是给出判断它是否值得部署的可测条件。

## 先看一个普通 MoE Transformer layer

忽略 LayerNorm、residual 和细节算子，一个 decoder layer 可以写成：

$$
u_l=x_l+\mathrm{Attention}_l(\mathrm{Norm}(x_l),KV_l)
$$

$$
x_{l+1}=u_l+\mathrm{MoE}_l(\mathrm{Norm}(u_l))
$$

Attention 与 MoE FFN 看起来只是顺序相邻的两个模块，资源属性却不同：

| 组件 | 长期驻留状态 | Decode 主要工作 | 常见瓶颈 |
| --- | --- | --- | --- |
| Attention | 每个请求、每层的 KV Cache | 读取历史 KV，计算当前 query | HBM 容量与带宽 |
| Routed FFN | expert weights | token dispatch、grouped GEMM、combine | 权重读取、计算与互联 |

对 decode 中的一个 token，Attention 要读取该请求已有上下文。上下文越长，读取量越大；FFN 不依赖历史 token 状态，只处理当前 activation。FFN 的计算形状取决于同一时刻有多少 token 路由到各 expert，而不是这些请求已经生成了多长。

在 collocated 部署中，两者位于同一组 ranks：

```text
GPU group:
  [Attention + KV] -> [Router] -> [Local/Remote Experts] -> [Combine]
```

AFD 则把所有权拆开：

```text
Attention pool                        FFN pool
[Attention + KV + Router] --dispatch--> [Routed Experts]
        ^                                  |
        └──────────── combine ─────────────┘
```

Attention 侧保留请求槽位、KV、dense layers 和调度状态；FFN 侧主要保存 routed expert weights，接收当前层 token，计算后返回同形状 activation。shared expert 放哪一侧并没有普遍唯一答案，要看算术强度、融合机会和通信布局。

## P/D 与 A/F 是两条正交切分轴

两种 disaggregation 经常同时出现在一张架构图里，却不能混为一谈。

### P/D：阶段级切分

```text
请求 ──> Prefill pool ──KV handoff──> Decode pool ──> tokens
```

一个请求通常在 prefill 完成后转移一次 KV。优化重点是 TTFT/TPOT 隔离、KV transfer、P/D 容量比例与 cache-aware routing。

### A/F：层内切分

```text
Decode step:
  Layer 1: A -> F -> A
  Layer 2: A -> F -> A
  ...
  Layer L: A -> F -> A
```

它在每个 step 重复 $L$ 次 dispatch/combine。优化重点是 activation 通信、microbatch pipeline、A/F 比例、barrier 与 expert load balance。

组合后可以有 Prefill-A、Prefill-F、Decode-A、Decode-F 四类角色，也可以只在 decode 侧启用 AFD。是否拆到四池必须从数据流与 SLO 推导，不能因为控制面支持更多 role 就默认全部拆开。

## AFD 真正想释放的三个约束

### KV 容量不再决定 expert 副本数

collocated 实例为了容纳更多长上下文，常常需要增加实例数。每增加一份实例，也会被迫复制一份或一组 expert weights。AFD 允许 Attention pool 横向扩容 KV 容量，而 FFN pool 保持更少、更集中的副本。

### FFN 可以聚合多个 Attention 实例的 token

设 $r$ 个 Attention workers 各提供 microbatch $B$，一个 FFN worker 接收约 $rB$ 个 token。更大的 FFN batch 可以提高每个 expert 获得的 token 数，让 grouped GEMM 从反复读取权重的 memory-bound 区域向 compute-bound 区域移动。

对于简化矩阵乘 $C=AB$，若忽略 activation 流量，单 expert 的算术强度随 token 数 $m$ 增长：

$$
I\approx\frac{2mkn}{kn}=2m
$$

这解释了 AFD 的直觉：不是 FFN 天生 compute-bound，而是聚合足够大的 $m$ 后才可能 compute-bound。

### Attention 与 FFN 可以使用不同硬件和比例

Attention 更看重可用 HBM、显存带宽和 KV 访问效率；FFN 更看重矩阵算力、expert 权重容量与高速互联。解耦后理论上可以让两类 pool 独立选择并行度甚至硬件类型，但异构部署还要满足数值格式、collective、拓扑和故障恢复兼容性，不能只按峰值 TFLOPS 拼接设备。

## 每层究竟要传多少数据

设当前 microbatch 有 $B$ 个 token，hidden size 为 $H$，activation 每元素 $s$ bytes。一次 A→F dispatch 的原始 hidden-state 下界约为：

$$
V_{A\to F}=B\cdot H\cdot s
$$

FFN 输出返回的 F→A combine 近似同量，所以每个 MoE layer 至少是：

$$
V_{layer}\approx 2B\cdot H\cdot s
$$

若 Top-$k$ 路由在 Attention 侧完成，dispatch 还会把 token 复制或分发到 $k$ 个 experts，网络上的逻辑载荷可能接近：

$$
V_{dispatch}\approx kB\cdot H\cdot s
$$

实际字节还包括 routing index、scale、padding、alignment 与 collective metadata。$L_{moe}$ 个 MoE layers、每秒 $R$ 个 decode steps 的流量量级为：

$$
BW_{AF}\gtrsim R\cdot L_{moe}\cdot(V_{dispatch}+V_{combine})
$$

与 P/D 传输相比，AFD 单次传的是 activation，不是完整历史 KV；但它频率极高、处于每个 token 的关键路径上。平均带宽够用仍不代表可用，单次消息延迟、tail jitter 与 incast 都会直接进入 TPOT。

## 为什么 AFD 通信不是普通 All-to-All

传统 Expert Parallel 中，各 rank 同时拥有 Attention 与一部分 experts，dispatch/combine 常表现为对称 All-to-All：所有 rank 都可能发送和接收。

AFD 把角色分离后，通信更像有方向的二部图：

```text
A ranks  --token dispatch-->  F ranks
A ranks  <--expert combine--  F ranks
```

A、F 两侧 rank 数可以不同，路由是 many-to-many，发送与接收行为不对称。通信库需要知道：

- `request_id / slot_id / layer_id / token_pos`，用于结果归位；
- logical expert 到 physical F rank 的映射版本；
- Top-$k$ 权重与 combine 顺序；
- microbatch epoch，防止迟到包写入下一轮；
- cancellation 与 timeout 状态，避免已释放 slot 被旧结果污染。

直接把为对称 EP 调优的 All-to-All 搬过来，往往既浪费连接，也难以表达 A/F 不同规模。实现需要针对 M-to-N dispatch/combine、RDMA buffer ownership 与 credits 重新设计。

## 一次 decode step 怎样流水化

若每个 microbatch 严格串行执行：

```text
A compute -> dispatch -> F compute -> combine -> next layer
```

网络与另一侧 GPU 都会产生大量空洞。AFD 通常把 batch 切成多个 microbatches，让 Attention、Communication、FFN 重叠：

```text
time --->
A:     A(m0)  A(m1)  A(m2)  A(m3)
Net:          C(m0)  C(m1)  C(m2)
F:                   F(m0)  F(m1)
```

在理想稳态中，cycle time 由最慢 stage 决定：

$$
T_{cycle}\approx\max(T_A,T_C,T_F)
$$

但 microbatch 数增加并非免费：

- batch 太小，FFN grouped GEMM 又退回 memory-bound；
- 消息变碎，collective latency 与 launch overhead 占比升高；
- buffer、event 与 stream 依赖更复杂；
- pipeline 填充和排空会增加单请求延迟。

针对现代 MoE AFD 的分析指出，为同时覆盖 A、通信与 F 三个阶段，往往需要至少 three-batch overlap 的思路。它不是一个固定 API 名称，而是提醒系统必须准备足够多独立 microbatch，使三段能稳定并行。

## Attention 时间会增长，FFN 时间却相对稳定

连续批处理中，一个 slot 每个 step 生成一个 token，KV 长度随时间增长；请求结束后，又被新请求补入。对 Attention worker $j$，可把当前总历史 token load 写成：

$$
T_j=\sum_{b=1}^{B}(P_{j,b}+A_{j,b})
$$

$P$ 是 prompt 长度，$A$ 是当前已生成长度。Attention 时间可近似拟合为：

$$
t_A(T_j)=\alpha_AT_j+\beta_A
$$

FFN 每步只看当前 token batch，时间更接近：

$$
t_F(rB)=\alpha_F(rB)+\beta_F
$$

通信则可写成：

$$
t_C(rB)=\alpha_C(rB)+\beta_C
$$

这意味着即使系统在某一时刻让三段恰好平衡，decode 继续推进后 $t_A$ 也会变长，原本隐藏的通信或 FFN 可能暴露成 pipeline bubble。静态 benchmark 的最佳比例不会自动适配真实长短请求混合。

## A/F 比例不是按平均长度相除

设一个 bundle 由 $r$ 个 Attention workers 和 1 个 FFN worker 组成，每个 A worker 有 $B$ 个 slots。若所有 A workers 需要同步，cycle 由最慢的 worker 决定：

$$
W_{B,r}=\max_{1\le j\le r}T_j
$$

于是：

$$
\tau(B,r)=\max\left\{
\alpha_AW_{B,r}+\beta_A,
t_C(rB),
t_F(rB)
\right\}
$$

按实例数归一化的吞吐可以写成：

$$
Throughput_{inst}=\frac{1}{r+1}\cdot\frac{rB}{E[\tau(B,r)]}
$$

这个表达式说明 $r$ 太小与太大都会浪费：

- $r$ 太小，F pool 收不到足够 token，expert GEMM 饥饿；
- $r$ 太大，通信或 F 成为瓶颈，A workers 在 barrier 前等待；
- $r$ 增大还会提高遇到长上下文 straggler 的概率。

因此容量规划应从生产 trace 估计 prompt、decode、并发与长度方差，再拟合 $t_A,t_C,t_F$。只代入平均 context length 会低估 length-biased effect：长输出在随机观察时更容易仍留在 batch 中，Attention 实际看到的驻留负载高于“每个请求平均长度”所暗示的值。

## 三个运行区间

把上一节的 cycle time 拆开，可以得到三个容易解释的区间。

### Attention-bound

$$
t_A>t_C,\quad t_A>t_F
$$

表现为 A pool HBM bandwidth 高、F 与网络等待。应优先改善 KV 布局、MLA kernel、Attention batch balance，或增加 A 侧资源；盲目增加 F 节点没有收益。

### Communication-bound

$$
t_C>t_A,\quad t_C>t_F
$$

表现为链路接近饱和、collective P99 上升、A/F 两侧都有空洞。需要优化拓扑放置、message fusion、协议、microbatch 或减少跨 scale-out 边界。继续堆计算卡只会扩大等待。

### FFN-bound

$$
t_F>t_A,\quad t_F>t_C
$$

表现为 F stream 长时间忙碌、A 侧 credits 耗尽。可以增加 F 容量、调整 expert placement、减少热点或改善 grouped GEMM。

真正的目标不是让某个 kernel HFU 最大，而是在 SLO 下让三段的有效 cycle 接近平衡，并把实例成本算进去。

## 为什么标准集群上会出现 AFD dead zone

AFD 的吸引力来自聚合 token 后提升 FFN 算术强度。但增加 F 规模并不保证更多 token 能及时送达。若 scale-out 网络先到瓶颈，FFN 的输入速率被通信上限锁死：增加 FFN ranks 只会把固定 token 进一步摊薄，单个 expert 的活跃时间下降。

可以把 FFN 在一个 latency budget $t_B$ 内的硬件利用率拆成：

$$
HFU=OFU\cdot S_t
$$

其中 OFU 描述 grouped GEMM 活跃期间的效率，$S_t=t_G/t_B$ 是该算子在整个预算内的时间占比。更大 batch 可能提高 OFU；可若网络限制使 $t_G$ 在预算中的占比继续下降，最终 HFU 仍很低。

这就是通信 dead zone：看起来 expert GEMM 单算子很快，整台 F GPU 却大部分时间在等数据。具有高带宽 scale-up fabric 的 Superpod 更容易跨过这个区间；普通多机网络上，AFD 可能不如已经高度优化的 Wide EP。

## fine-grained experts 对 AFD 更苛刻

设模型有 $E$ 个 routed experts，每个 token 激活 $k$ 个，聚合输入 token 数为 $N$。均匀时单 expert 平均接收：

$$
\bar n_e=\frac{kN}{E}
$$

$E$ 越大、expert 越细，单 expert 的 $\bar n_e$ 越小。为了形成高效 GEMM，需要聚合更多 Attention workers；但更多 A workers 又会增加网络 fan-in、barrier 和负载不均衡。

所以“MoE 越稀疏越适合 AFD”并不成立。已有分析认为，expert 粒度较粗、稀疏度较低的模型更容易从 AFD 获益；fine-grained MoE 可能需要模型—系统共同设计、expert replication 或更强互联才能跨过利用率门槛。

## 负载不均衡为何比普通 EP 更危险

AFD 中有两类波动：

### DP 侧的上下文不均衡

不同 A worker 上的请求长度不同，Attention latency 由最慢 worker 决定。可以通过按剩余长度、KV load 或 predicted work 路由请求降低方差，但在线输出长度未知，无法完全消除。

### EP 侧的 expert 不均衡

Router 产生的 expert 热点会让部分 F ranks 慢于其他 ranks。普通 collocated EP 有时还能在同一 latency budget 中通过调整 batch 或 overlap 吸收一部分波动；AFD 的 A 与 F 是离散节点级资源，某一侧稍慢就把抖动传到另一侧，比例调整也不能细到任意小数。

因此控制面不应只看平均 `attention_ms` 与 `ffn_ms`，还要记录：

- 每个 A worker 的 active KV tokens 与 stage P95/P99；
- 每层、每 expert、每 F rank 的 token histogram；
- dispatch/combine bytes、queue time 与 collective tail；
- barrier wait 分解：`wait_for_A`、`wait_for_net`、`wait_for_F`；
- pipeline bubble ratio 与 microbatch occupancy。

没有这些分解指标，HFU 下降只能被模糊地归因于“通信慢”。

## Router、shared expert 和 dense layer 放在哪里

组件放置决定了传输语义。

### Router 通常放在 Attention 侧

这样 A→F dispatch 可以直接按目标 expert 排列，F 侧成为接收并计算的执行池。但 router weights、Top-$k$、capacity policy 与 expert map version 必须跟 F pool 一致。

### Routed experts 放在 FFN 侧

这是 AFD 的核心。F pool 可以独立做 expert parallel、replication 和热点迁移，并将 combine 结果送回原 A slot。

### Shared expert 需按 profile 决定

shared expert 对所有 token 都执行，batch 通常更大，可能与 dense FFN 一样具有较高算术强度。留在 A 侧可与 residual 或 norm 融合、减少网络；放到 F 侧可统一权重池并汇聚 batch。不能只按“名字里有 expert”决定。

### Norm、residual 与 dense layers 多留在 A 侧

这样 F 服务接口可以保持为“输入 hidden state 与 routing metadata，返回加权 expert output”。状态边界清楚，也减少跨池往返类型。但具体模型的 fused kernel 可能要求调整切点，任何调整都要做数值等价验证。

## 状态所有权必须是协议的一部分

F pool 常被称为 stateless，是指它不持有每个请求随序列增长的 KV Cache，并不表示请求可以没有身份。一次 activation 往返至少要关联：

```text
request_id
request_epoch
decode_step
layer_id
microbatch_id
token_slot
expert_map_version
deadline
```

如果请求已取消，迟到的 F result 必须被丢弃；如果 A worker 重试，同一 `(epoch, step, layer)` 不能重复提交 residual；如果 expert map 正在滚动升级，dispatch 和 combine 必须使用同一版本。推荐让每层调用具有幂等键，并让 A 侧作为 commit owner：只有收到完整、版本匹配的 combine 结果才推进 layer state。

F worker 失败时是否能安全重试取决于算子确定性和随机状态。MoE FFN 本身通常是纯函数，但量化 kernel、通信归约和故障后的不同 expert replica 可能产生数值差异。恢复策略要明确“容许近似重算”还是“要求 bitwise/within-tolerance 等价”。

## Backpressure 要从 F 侧传回准入层

当 F pool 饱和时，继续让 A workers 产生更多 activation 只会堆积网络 buffer，并把显存从 KV 挤给 pending tensors。应建立端到端 credits：

```text
F free slots / bytes
        ↓
AF transport credits
        ↓
A microbatch scheduler
        ↓
request admission
```

每个 A worker 只有拿到 F credits 才能启动相应 layer 的 microbatch。credits 既要限制 token 数，也要限制 bytes 与 in-flight layer 数。高水位时可以减小 A batch、暂停低优先级租户或拒绝新请求；不能等待 OOM 后再靠全局重启恢复。

优先级也不能只存在 API gateway。一个高优先级请求若在 F queue 中与大量离线请求同级排队，Attention 侧再精细的调度都无法守住 TPOT。deadline、tenant class 与 cancellation 必须随 dispatch metadata 传播到 F scheduler。

## AFD 与 Experts-as-a-Service 的边界

AFD 通常让一个固定 A bundle 与固定 F bundle 紧耦合，以 microbatch pipeline 满足几十毫秒级 TPOT。更进一步，可以把 F pool 做成跨模型或跨 A group 共享的 Experts-as-a-Service（EaaS）：

```text
many A pools -> shared expert service -> many A pools
```

共享范围越大，越容易聚合 token、减少冷门 expert 副本；但 queueing、路由版本、模型隔离和尾延迟问题也更严重。为了攒够一个 expert batch 而等待更多 token，本质上是以 latency 换 throughput。

对拥有大量 fine-grained experts 的模型，完全异步 EaaS 还可能要求巨大物理规模，才能让任意层的任意 expert 都常驻并持续收到工作。因此可以把 EaaS 看作 AFD 的更松耦合延伸，而不是自然免费的下一步。

## 怎样判断 AFD 是否值得

先建立 collocated/Wide EP 基线，再回答以下问题。

### 模型条件

- decode Attention 是否明显受 KV bandwidth/capacity 限制？
- FFN 汇聚 batch 后能否显著跨过 roofline ridge？
- expert 是否足够粗，Top-$k$ 与 $E$ 是否能形成合理单 expert batch？
- shared expert、dense layer 和 MTP 会怎样改变切分点？

### 硬件条件

- A↔F 是否能尽量留在 scale-up fabric，而不是频繁跨 ToR？
- 单消息 P99、双向带宽与 incast 行为能否满足每层 budget？
- 是否能让 transport 与 A/F compute 真正 overlap？
- 异构 GPU 的 dtype、kernel、collective 和容错是否一致？

### 工作负载条件

- context 长度与输出长度的分布及方差是多少？
- Prefix Cache 会把多少 prefill 工作移走，并改变 A 侧负载？
- TPOT SLO 是否允许更多 microbatch pipeline depth？
- expert 热点、租户混合和优先级会造成多大 jitter？

### 经济条件

最终比较的是满足 SLO 的单位成本：

$$
Cost_{Mtoken}=
\frac{GPU\_hours\cdot price+network+host}{accepted\ output\ tokens/10^6}
$$

若 AFD 提高了 FFN operator HFU，却需要更多空闲 A nodes、昂贵互联或更高尾延迟，它仍可能是成本负优化。

## 一个可执行的评测矩阵

### 单算子与单链路

- Attention：按 active KV tokens 拟合 $\alpha_A,\beta_A$；
- FFN：按每 expert token 数测试 grouped GEMM roofline；
- 通信：按 message size、M:N 比例和并发测 dispatch/combine P50/P99；
- overlap：验证 compute stream 与 communication stream 是否真实重叠。

### 单 bundle

- 扫描 $rA:1F$ 与不同 microbatch 数；
- 分别注入固定长度、长短混合、heavy-tail decode；
- 记录三阶段 latency、barrier 和 bubble；
- 注入 DP/EP imbalance，观察尾延迟放大；
- 测试取消、F timeout、A 重试和 expert map 升级。

### 集群与 SLO

- 与同预算 collocated EP、Wide EP、P/D-only 对比；
- 同时报告 TTFT、TPOT、E2E、throughput 与 cost；
- 测 scale-up 内、跨节点、跨 ToR 三种放置；
- 混入高优先级和长输出请求，检查 starvation；
- 执行 worker 滚动升级、链路降速和单卡故障。

AFD 的关键证据不是某个平均吞吐点，而是“在目标 SLO、生产长度分布和固定预算下，优于最佳 collocated 基线”。

## 上线时的安全边界

1. **从固定 topology 开始。** 先稳定一个 A/F bundle 与 expert map，不要同时启用动态扩缩和在线迁移。
2. **所有请求带 epoch。** 取消、重试与滚动升级必须能拒绝迟到结果。
3. **credits fail closed。** F 状态未知或 transport 异常时停止发新 microbatch，不无限堆积。
4. **比例调整要慢于请求调度。** A/F 节点级扩缩是离散、昂贵动作；毫秒级波动交给 batching 和 routing。
5. **保留 collocated fallback。** 模型、backend 或互联不满足 AFD 条件时，能够回到已验证路径。
6. **按版本观察。** kernel、expert map、量化配置、transport protocol 进入同一 trace，避免把版本回归误判为 workload 抖动。

## 总结

Attention–FFN Disaggregation 把 Transformer layer 里的资源差异变成物理资源池：Attention pool 拥有请求和 KV 状态，FFN pool 拥有 routed expert weights，二者通过每层 activation dispatch/combine 连接。它允许 KV 容量与 expert 算力独立扩展，也能聚合多个 Attention 实例的 token，提高 FFN 的算术强度。

代价是更严格的系统耦合。通信每层发生，A/F 比例受随机上下文负载影响，任何 Attention straggler、expert hotspot 或网络 jitter 都会穿过 barrier 放大。标准集群还可能进入通信 dead zone：单个 GEMM 很高效，整池 GPU 却因等不到 token 而空闲。

因此 AFD 应被视作针对特定模型、硬件和 workload 的优化，而非默认架构。先测出 Attention、Communication、FFN 三条延迟曲线，再用生产 trace 选择比例；同时保留状态协议、credits、故障回退和 collocated 基线。只有当这些条件共同成立，“A/F 解耦”才真正意味着更低成本，而不是把原来的 GPU bubble 搬到网络另一端。

## 参考资料

- [Analytical Provisioning for Attention–FFN Disaggregated LLM Serving under Stochastic Workloads](https://arxiv.org/abs/2601.21351)
- [Revealing the Challenges of Attention-FFN Disaggregation for Modern MoE Models and Hardware Systems](https://arxiv.org/abs/2602.09721)
- [How Far Can Disaggregation Go? A Design-Space Exploration of Attention-FFN Disaggregation for Efficient MoE LLM Serving](https://arxiv.org/abs/2605.28302)
- [Toward Cost-Efficient Serving of Mixture-of-Experts with Asynchrony](https://arxiv.org/abs/2505.08944)
- [Expert-as-a-Service: Towards Efficient, Scalable, and Robust Large-scale MoE Serving](https://arxiv.org/abs/2509.17863)
- [快手万擎大模型推理成本和性能优化实践](https://zhuanlan.zhihu.com/p/2067652898524345525)
