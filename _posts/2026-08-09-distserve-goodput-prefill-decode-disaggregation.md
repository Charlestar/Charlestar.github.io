---
layout: post
title: "DistServe：为什么 Prefill 与 Decode 要分开配置"
subtitle: "从阶段干扰、每 GPU Goodput 到带宽感知 Placement"
date: 2026-08-09 16:20:00 +0800
last_modified_at: 2026-08-09
author: iStar
catalog: true
series: distributed-inference
series_order: 10
technology_year: 2024
mathjax: true
tags: [AI Infra, DistServe, 分离式推理, Goodput, LLM推理]
---

同一个 Decoder-only Transformer，在 prefill 和 decode 阶段表现得像两类不同服务。Prefill 一次处理整段 prompt，矩阵大、并行度高，主要决定首 token 何时出现；decode 每轮只推进一个位置，反复读取模型权重与历史 KV Cache，主要决定后续 token 是否平滑输出。

早期 LLM serving 系统把两个阶段放在同一组 GPU 中，通过 continuous batching 提高总体利用率。这种设计省去了跨 worker KV Cache 交接，也只需保存一份模型权重，但它把两种 SLO、两种计算形态和两套资源配置绑在了一起。

DistServe 的贡献不是最早说出“可以把 Prefill 与 Decode 分开”，而是把这件事写成一个可优化的部署问题：给定模型、请求长度分布、到达过程、TTFT/TPOT 目标和集群网络，分别选择两个阶段的并行策略与实例数，再决定它们应该放在哪些 GPU 上，使每块 GPU 能承载的合格请求率最大。

理解 DistServe，关键不是记住一次实验的倍数，而是理解它改变了 serving 系统的目标函数：从追求裸吞吐，转向追求同时满足两类延迟 SLO 的 **per-GPU goodput**。

## 一次生成为什么有两个延迟目标

设请求到达时间为 $t_0$，第一个输出 token 返回时间为 $t_1$，后续 token 时间为 $t_2,t_3,\ldots$。

Time To First Token 可以写成：

$$
\operatorname{TTFT}=t_1-t_0
$$

它包含排队、prompt prefill、可能的 KV 读取/传输、采样与网络返回。用户发送问题后等待系统“开始回答”，主要感知这个量。

Time Per Output Token 常用后续 token 间隔的平均值表示：

$$
\operatorname{TPOT}
=\frac{t_N-t_1}{N-1}
$$

也有系统记录逐 token Inter-Token Latency 并观察 P90/P99。无论具体口径，目标都是约束 decode 流畅度。

两个阶段对应关系大致是：

```text
request arrives
      │ queue + prefill + first sampling
      ├────────────────────────────────► first token
      │                                     │
      │                            decode steps / TPOT
      │                                     ▼
      └────────────────────────────────► remaining tokens
             TTFT
```

不同应用对两者的容忍度不同。对话希望快速开始响应；批量摘要可能更关心整段生成完成速度；代码补全对首 token 和持续生成都可能敏感。系统不能用一个 tokens/s 数字表达所有服务质量。

## Colocated Serving 的干扰从哪里来

Continuous batching 以 iteration 为单位重组 active requests。假设 GPU 正在为一组请求做 decode，此时新请求需要 prefill：

```text
iteration 1   decode 64 requests
iteration 2   decode 64 + prefill one long prompt
iteration 3   decode 64 requests
```

第二轮的工作量远大于普通 decode iteration。已有请求必须等待长 prompt 的计算，TPOT 出现尖峰；新 prompt 又与 decode token 竞争算力和 memory bandwidth，TTFT 也可能变长。

### Chunked Prefill 能缓解但没有消除共享

把长 prompt 切成固定 token chunk，可以限制单轮 prefill 的最大工作量：

```text
long prefill = chunk 1 + chunk 2 + chunk 3 + ...

batch step = decode tokens + one prefill chunk
```

这能显著改善调度，是现代 colocated engine 的重要方案。但 chunk size 仍是一种折中：

- chunk 大，prefill 效率更好，decode stall 更长；
- chunk 小，TBT 更平滑，prefill 要经历更多轮、更多调度和较差的矩阵形状；
- decode load 变化时，固定 chunk 不一定始终合适。

DistServe 的观点是，只要两类工作仍在同一 GPU iteration 中竞争，就需要在 TTFT 与 TPOT 之间分配预算。物理分离让这类直接干扰从执行资源层消失，但会引入 KV 传输和权重复制。

## 第二个问题是 Resource Coupling

即使 scheduler 能完美切分时间，共用 GPU 也要求 prefill 与 decode 使用同一套模型实例和并行配置。

### Prefill 倾向

Prefill 单请求已经有很多 token 可并行计算。严格 TTFT 下，增加 tensor parallel 可能缩短一次请求执行时间；高到达率下，更多 pipeline stage 或 replica 可能降低排队、提高 phase capacity。最佳选择取决于 prompt length、并行效率和到达率。

### Decode 倾向

Decode 单请求每轮工作很小，需要较大 continuous batch 才能提高 GPU 利用率。严格 TPOT 下，tensor parallel 能降低单步 latency，但通信会带来递减收益；若单实例已经满足 TPOT，复制 decode instance 往往能更接近线性地增加 request capacity。

### 同一个配置很难同时最优

```text
colocated model instance
    ├─ TP/PP chosen for prefill latency?
    ├─ TP/PP chosen for decode latency?
    ├─ memory reserved for prefill burst?
    └─ memory reserved for many decode KV sequences?
```

DistServe 将模型权重分别部署到 P 与 D instance，使两个阶段可以拥有不同的 TP/PP、batch 策略与 replica 数。额外权重副本是成本，但解耦后的配置空间也因此打开。

## Throughput 与 Goodput 不是同一个目标

### Throughput

裸吞吐通常统计：

$$
\text{throughput}
=\frac{\text{completed requests or tokens}}
{\text{wall-clock time}}
$$

一个系统可以通过大 batch 获得很高总吞吐，同时让大量请求等待很久。只要最终完成，它们仍计入 throughput。

### SLO Attainment

设请求 $r$ 的两个约束为 $S_{TTFT}$ 与 $S_{TPOT}$，可以定义：

$$
I_r=\mathbf 1[
TTFT_r\le S_{TTFT}
\land
TPOT_r\le S_{TPOT}
]
$$

请求集合上的 attainment 为：

$$
A=\frac{1}{|R|}\sum_{r\in R}I_r
$$

如果产品要求至少 90% 的请求同时满足两项 SLO，就需要 $A\ge0.9$。

### Per-GPU Goodput

DistServe 将 per-GPU goodput 定义为：在给定 SLO attainment 目标下，系统能够承载的最大请求到达率，再除以使用的 GPU 数：

$$
G=\frac{\lambda_{max}(A\ge A_{target})}
{N_{GPU}}
$$

它把成本、负载和用户体验放进同一个指标。更高 $G$ 表示同样 SLO 下每块 GPU 能服务更多请求，或同样流量下需要更少 GPU。

Goodput 仍不是模型质量指标。请求生成了不正确答案，即使延迟达标也会计入上述系统指标；质量、安全和业务成功率需要另一组 guardrail。

## 一个算例说明为什么 P/D 比例不是 1:1

假设 profile 得到：

```text
one Prefill GPU group capacity: 6 req/s under TTFT SLO
one Decode GPU group capacity: 11 req/s under TPOT SLO
target traffic:                20 req/s
```

至少需要：

$$
N_P=\left\lceil\frac{20}{6}\right\rceil=4
$$

$$
N_D=\left\lceil\frac{20}{11}\right\rceil=2
$$

整体容量受较小的一侧限制：

$$
\lambda_{system}
\le\min(N_PG_P,N_DG_D)
$$

如果盲目部署 3P+3D，D 端有富余，P 端却只能提供 18 req/s；额外 D GPU 没有变成可交付的请求。

真实计算还要考虑每个 instance 使用多少 GPU、KV handoff、负载波动和 tail SLO。这个例子只说明：输入/输出长度分布决定两个阶段的 service demand，P/D ratio 不应该由架构图对称性决定。

## DistServe 的离线输入是什么

Placement optimizer 不是只读一个模型名称。它需要：

- 模型层数、hidden size、权重与 KV 内存；
- 可选 TP/PP 与每种配置的执行 profile；
- GPU/node 数与显存；
- 节点内和节点间带宽拓扑；
- 请求到达过程；
- input/output length 分布；
- TTFT、TPOT 与 attainment target；
- 目标 traffic rate。

输出 placement 包含：

1. Prefill instance 的并行配置；
2. Decode instance 的并行配置；
3. 两类 instance/replica 的数量；
4. 每个 stage/instance 在物理节点上的位置。

所以它更接近 capacity planner + topology-aware deployer，而不是在线每 token scheduler。

## 为什么用模拟器搜索而不是一个闭式公式

均匀 prompt 和 Poisson 到达下，单个 prefill server 可近似为 M/D/1 queue。设单请求服务时间为 $D$、到达率为 $R$ 且 $RD<1$，平均响应时间为：

$$
E[T]
=D+\frac{RD^2}{2(1-RD)}
$$

第二项是排队延迟。当利用率接近 1，分母变小，延迟会迅速上升。这解释了为什么复制 instance 有时比单纯缩短执行时间更能改善高负载排队。

但真实请求长度不同、到达有 burst、decode 持续时间由输出长度决定，TP/PP 又改变 service curve。SLO attainment 是分位/比例目标，不能仅靠平均值推导。

DistServe 因此：

1. 从历史 trace 拟合较长时间尺度的 workload 分布；
2. 为模型与并行配置建立 latency model；
3. 重新采样到达、输入和输出长度；
4. 在 simulator 中回放 scheduler；
5. 二分搜索满足 attainment 的最大 request rate；
6. 枚举可行 placement，比较每 GPU goodput。

论文报告其测试设置中 simulator 与真实运行的 SLO attainment 误差较小，但这种准确度依赖 profile 和 workload 是否仍代表当前系统。线上模型、kernel、请求域或网络改变后必须重新校准。

## P 与 D 可以分别选择怎样的并行策略

论文使用 intra-op/inter-op 的分析框架，实际结果以 TP/PP 配置呈现。用今天常见术语理解：

### Tensor Parallel

把同一层矩阵与 attention heads 分到多 GPU。优点是缩短单请求/单 step 计算，代价是每层 collective communication 和分片后利用率下降。

严格 TTFT 的 prefill 或严格 TPOT 的 decode 可能需要更高 TP；但超过某点后，通信让速度不再近线性提升。

### Pipeline Parallel

把模型层分成多个 stage。单个 microbatch 要串过所有 stage，端到端 latency 不会像理想 TP 那样直接缩短；多个请求/微批在流水线中并行后，吞吐可以提高。请求长度不均会造成 stage bubble。

### Replication

复制完整 instance，按请求分流。若模型能放入更少 GPU 且单 instance 已满足 SLO，replica 常能近线性增加 rate capacity，并降低单队列到达率；代价是重复保存权重。

DistServe 不预设 P 一定 TP、D 一定复制，而是让 workload 与 SLO 决定。低流量、严格延迟和高流量、较宽松延迟可能得到完全不同 placement。

## KV Cache 交接怎样进入成本模型

对 prompt 长度 $S$，KV payload 近似：

$$
M_{KV}=2SLH_{kv}DB
$$

如果请求率为 $\lambda$，平均每秒 P→D 数据量约为：

$$
BW_{required}\approx
\lambda E[M_{KV}]
$$

论文用 OPT-66B、512-token 请求举例，单请求 KV 约 1.13 GB；10 req/s 已需要约 11.3 GB/s，也就是约 90 Gbps 的有效数据率。现代 GQA/MQA 模型会显著减小 KV，但长上下文和更高 RPS 仍可能把传输推成瓶颈。

需要区别：

```text
link bandwidth       端口理论能力
effective bandwidth  扣除协议、路径、并发后的 payload rate
visible latency      没有被计算/排队隐藏、真正增加请求延迟的部分
```

Placement 必须看后两者，而不是只看网卡标签。

## 高跨节点带宽时怎样 Placement

如果节点间 fabric 足够快，P 与 D instance 可以跨节点自由组合。DistServe 的搜索可分两层：

1. 分别找到 P 与 D 阶段 per-GPU goodput 最优的并行配置；
2. 按目标 traffic rate 复制两种 instance，匹配整体容量。

概念上：

```text
search P configs ─► best phase goodput Gp ─► Np replicas

search D configs ─► best phase goodput Gd ─► Nd replicas

physical placement has few P↔D locality constraints
```

这里的“传输可忽略”是经过相应硬件和 workload 测量后对 placement 的简化，不表示 KV 真的没有通信。若平均 payload、到达率或其他作业的网络竞争改变，这个假设需要重新验证。

## 低跨节点带宽时为什么要对齐 Model Stage

如果跨节点只有较慢网络，最简单的想法是把完整 P instance 和完整 D instance 放进同一节点，通过 NVLink 交接。但大模型的两份权重可能放不下：例如单个实例已经占满多张 GPU，再复制一份没有空间。

DistServe 利用 KV 只需要在对应模型层之间交接这一事实。若 P 与 D 都采用 pipeline stage：

```text
Prefill pipeline:  P-stage-0 → P-stage-1 → P-stage-2
                       │           │           │
                      KV0         KV1         KV2
                       │           │           │
Decode pipeline:   D-stage-0 → D-stage-1 → D-stage-2
```

将相同 stage 的 P segment 与 D segment 放在同一节点：

```text
Node 0: P-stage-0 + D-stage-0  → local high-bandwidth KV transfer
Node 1: P-stage-1 + D-stage-1  → local high-bandwidth KV transfer
Node 2: P-stage-2 + D-stage-2  → local high-bandwidth KV transfer
```

这样大模型仍跨节点分 stage，但每层产生的 KV 不需要走慢速 cross-node path。代价是 P/D placement 与 PP degree 相互约束，搜索空间不再能分开优化。

今天的系统可以用 RDMA/NIXL/Mooncake 等更强数据面放宽部分约束，但拓扑原则没有过时：对应 KV shard 应尽量沿有效带宽高、竞争少、NUMA 合理的路径交接。

## 在线调度还要处理 Burst

即使平均 P/D capacity 匹配，短时 burst 也可能让 P 在一段时间内完成大量请求，将大批 KV 同时推向 D。D 的 HBM 既要容纳 active decode KV，也要接收新请求，push storm 可能使其瞬间过载。

DistServe 采用 pull-style KV transfer：P 完成后暂时保留 KV，D 有能力接收时再主动拉取。

```text
Prefill completed queue (KV stays at P)
        [r1][r2][r3][r4][r5]
                    │
                    │ D has slots, pulls r1/r2
                    ▼
Decode active set [r1][r2] ...
```

P 的 GPU memory 在这里临时充当交接队列。它能对 D 施加 backpressure，却也引出新的约束：

- 等待中的 KV 占用 P HBM；
- P block 满后不能继续无限接收 prompt；
- 请求取消或 D failure 必须释放源 KV；
- D 选择哪个 completed request 会影响公平性和 deadline；
- transfer lease/timeout 必须防止 block 永久 pin。

现代 KV Connector 中的 block ownership、pull/read、lease 和完成回执，正是在工程上继续解决这类生命周期问题。

## 一次 Request 的完整路径

```text
1. Router admits request
        │
        ▼
2. Select Prefill instance / queue
        │
        ▼
3. Run prompt, produce first token + KV Cache
        │
        ▼
4. Retain KV in completed-prefill buffer
        │
        ▼
5. Decode scheduler reserves capacity and pulls KV
        │
        ▼
6. Decode instance joins continuous batch
        │
        ▼
7. Stream remaining tokens
```

TTFT 到底截止在第 3 步还是第 5/6 步，要看 API 是否允许 P 先返回首 token、以及 D handoff 是否在首 token 前完成。做 benchmark 时必须统一定义，否则不同实现的 TTFT 不可比较。

第一 token、sampling state、position、stop/grammar 状态也要随请求从 P 交给 D。只传 KV bytes 而漏掉逻辑状态，D 的下一步生成仍可能错误。

## DistServe 与 Mooncake 的关注点不同

两者都采用 P/D disaggregation，但文章不应把它们写成同一个系统的不同名字。

### DistServe

重点是 prefill/decode 干扰、per-GPU goodput，以及怎样联合选择两阶段资源、TP/PP、replica 和 bandwidth-aware placement。它将 KV 主要视为阶段交接时必须传输的中间状态。

### Mooncake

进一步把 KV Cache 当成可跨请求复用、可在 CPU/DRAM/SSD 中保存、可复制与淘汰的分布式缓存对象，并让全局调度器围绕缓存位置和负载做决策。

### NIXL/KV Connector

解决 P/D 或 KV Store 选定源、目标后，serving engine 怎样注册内存、交换 descriptors、异步搬运并安全管理 block 生命周期。

可以这样排列：

```text
DistServe:      为什么分离，以及怎样按 goodput 配资源
Mooncake:       分离后怎样围绕全局 KV Cache 复用与调度
NIXL/Connector: 选定路径后怎样执行可靠高效的数据交接
```

## DistServe 与 llm-d 的关系

DistServe 的 placement optimizer 假设掌握模型、workload、SLO 与 cluster profile，输出一套阶段部署方案。llm-d 位于更现代的 Kubernetes serving control plane：用 Gateway/调度器、模型服务 Pod、KV-aware routing 和 P/D 组件组织线上集群。

二者不是直接的代码继承关系，但 DistServe 提出的几个问题会在 llm-d 一类平台中重新出现：

- Prefill 与 Decode deployment 分别扩多少 replica；
- route request 时同时看 queue 与 KV locality；
- 网络 topology 是否支持所选 P→D pairing；
- 何时触发扩缩容，避免只扩一侧；
- 怎样用 TTFT/ITL goodput 而不是 GPU utilization 判断容量。

如果没有阶段 capacity model，Kubernetes HPA 即使能把 Pod 数调高，也可能扩错资源池。

## 论文结果应该怎样引用

OSDI 2024 论文在多种 OPT 模型、应用 trace、SLO 与 32 张 A100 的实验条件下，报告相对当时基线最高可承载约 $7.4\times$ 请求率，或满足约 $12.6\times$ 更严格 SLO，并以超过 90% 请求达标为主要目标。论文的某些逐 workload 对比区间小于这些最大值。

这些数字不能直接迁移到现代 GQA/MoE 模型、H100/Blackwell、不同网络与最新 vLLM/SGLang：

- baseline 已加入 chunked prefill、异步 scheduling 与更强 kernel；
- KV size、attention 算法和模型并行方式改变；
- 长 reasoning 输出让 D demand 占比上升；
- 高速 RDMA 降低 handoff，低成本 Ethernet 可能放大它；
- prefix cache 命中会减少 P 计算并改变 P/D ratio。

应该引用它所证明的系统规律和实验条件，而不是把峰值倍数写进容量预算。

## 一套从 Colocated 到 Disaggregated 的实验

### 1. 建立三个 Baseline

```text
A. colocated continuous batching
B. colocated + tuned chunked prefill
C. P/D disaggregated + KV transfer
```

如果 C 只对比未调优 A，就无法判断收益来自物理分离，还是 baseline 缺少 chunked scheduling。

### 2. 固定 Workload Trace

保留：

- 到达时间与 burst；
- input/output token length；
- sampling/stop 配置；
- prefix reuse relation；
- service tier 与 deadline。

平均长度相同的两个 trace 可能有完全不同 tail 和排队行为，不能只生成固定长度请求。

### 3. 扫描 P/D 配置

对每组 P/D replica、TP、PP 与 placement，测：

- phase execution/queue latency；
- KV payload、effective bandwidth 与 visible handoff；
- TTFT、TPOT/ITL、E2E 的 P50/P90/P99；
- 两项 SLO 同时满足的 attainment；
- 每 GPU goodput；
- HBM KV/weight 占用与网络峰值。

### 4. 检查稳定性边界

逐步提高 request rate。接近饱和点时，排队会非线性上升。需要找：

```text
largest λ such that attainment >= target
```

而不是在过载后仍统计所有最终完成请求的平均 tokens/s。

### 5. 故意制造偏斜

分别回放：

- long-input / short-output；
- short-input / long-output；
- reasoning 长输出；
- 高 prefix hit；
- 网络限速和拥塞；
- P 或 D 单侧 worker failure。

观察固定 placement 何时失配，以及 controller 多久才能调整 P/D capacity。

## 上线后的容量控制不应只看 GPU Utilization

Prefill GPU 可能在低到达率下利用率不高，却是满足严格 TTFT 必需的 headroom；decode GPU 利用率不满，也可能因为 TPOT 已接近 deadline 而不能继续加 batch。

更合适的扩缩容信号包括：

### Prefill Pool

- queue delay 与 predicted TTFT；
- uncached prefill tokens/s；
- prompt length 分布；
- completed-KV buffer 占用；
- P stage SLO attainment。

### Decode Pool

- active sequences 与 KV blocks；
- batch size、step latency 和 ITL tail；
- remaining output tokens 估计；
- waiting-to-transfer requests；
- D stage SLO attainment。

### Handoff

- KV bytes/s 与 transfer queue；
- P block pin time；
- D load wait；
- topology/path saturation；
- retry 与 timeout。

扩容 D worker 通常还要加载完整模型并预热 kernel/Graph，不能等 TPOT 已经持续违约才开始。预测性扩容和预留容量是分离式系统真正上线时的重要补充。

## 哪些条件下分离可能不值得

DistServe 揭示的是一个强有力的配置空间，不是一条无条件规则。下列情形需要认真比较 colocated baseline：

- 单机/少量 GPU，复制两份权重明显挤压 KV capacity；
- 流量低，几乎没有 prefill/decode 同时执行；
- prompt 很短，chunked prefill 已能把干扰控制在 SLO 内；
- 网络慢或不稳定，KV handoff 占 TTFT 大头；
- 模型太大，同节点无法对齐 P/D stage，跨节点又没有高速 fabric；
- workload 快速变化，固定 P/D pool 经常单侧闲置；
- 多模型服务导致每个模型的独立 P/D replica 进一步碎片化资源。

也可以采用 hybrid 策略：基础流量 colocated，只有长 prompt 或严格 tail ITL 请求走 disaggregated path；或者让节点具备差异化能力，在负载变化时切换角色。选择标准仍应是同一 workload 和 SLO 下的 goodput。

## 小结

DistServe 把 Prefill/Decode 分离从一个架构技巧变成了可量化的资源配置问题。Colocated engine 中，长 prefill 会拖慢正在生成的 decode，decode batching 也会影响 TTFT；更深层的限制是两个阶段被迫共享同一 TP/PP、replica 数和 HBM 预算。

分离后，Prefill Pool 可以围绕 TTFT 和 prompt 分布配置，Decode Pool 可以围绕 TPOT、batch 与 KV capacity 配置。DistServe 用 simulator 在 workload、SLO 和 cluster profile 上搜索阶段并行策略、实例数与 physical placement，并用 per-GPU goodput 衡量“每块 GPU 能交付多少按时完成的请求”。

代价是模型权重复制、KV handoff 与更复杂的跨阶段状态。高带宽集群可以更自由地放置 P/D；低跨节点带宽则需要将对应 model stage 对齐到节点内高速互连。在线 pull 又把 decode backpressure 传回 prefill KV buffer。

这套思路构成后续分布式推理平台的容量基础：Mooncake 把 KV 扩展成全局缓存，NIXL/Connector 把交接实现成可靠数据面，llm-d 一类控制平面把路由和 Pod 编排带到 Kubernetes。无论工具如何变化，判断 P/D 是否该分离的起点仍是同一个：在真实请求分布下，哪种配置能以更少 GPU 同时满足 TTFT 与 TPOT。

## 参考资料

- [DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin)
- [DistServe OSDI 2024 论文 PDF](https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf)
- [DistServe 官方实现](https://github.com/LLMServe/DistServe)
- [DistServe 预印本](https://arxiv.org/abs/2401.09670)
- [Sarathi-Serve: Taming Throughput-Latency Tradeoff with Chunked Prefills](https://www.usenix.org/conference/osdi24/presentation/agrawal)
