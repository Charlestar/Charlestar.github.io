---
layout: post
title: "LLM 推理容量规划：从请求 Trace 推到 GPU 数量"
subtitle: "把 ISL、OSL、KV Cache、P/D Goodput 与扩容提前量放进同一张账"
date: 2026-07-11 09:00:00 +0800
last_modified_at: 2026-09-03
author: iStar
catalog: true
series: distributed-inference
series_order: 50
technology_year: 2025
mathjax: true
tags: [分布式推理, 推理调度, LLM推理]
---

“一张 GPU 每秒能生成多少 token”不能直接回答线上需要多少 GPU。相同 tokens/s 的两个 workload，可能一个由短 prompt、长输出构成，另一个由超长 prompt、短回答构成；前者持续占用 decode KV 和 memory bandwidth，后者集中消耗 prefill compute。再加入 prefix cache、请求 burst、P99 SLO、模型并行与 worker 冷启动，单个峰值吞吐很快失去解释力。

容量规划真正要回答的是：在给定模型、硬件、流量分布与服务目标下，哪套并行和资源配置能够让足够比例的请求同时满足 TTFT、ITL/TPOT 与可用性约束，并留出应对波动和故障的余量。

因此，规划单位不应是“理论 TFLOPS”或“平均 tokens/s”，而应是 **qualified requests per second per GPU**，也就是带 SLO 的 goodput。计算过程则要把请求拆成 prefill work、decode work、KV memory 和跨阶段 transfer 四笔账。

## 先定义要交付的服务

没有 SLO，容量只有“跑到崩溃前的最大值”，无法决定何时算足够。

一份最小服务目标可以包含：

```text
TTFT P90/P99       <= target
ITL or TPOT P90/P99 <= target
E2E deadline        <= target (optional)
SLO attainment      >= 99%
availability        >= target
max queue time      <= target
```

### TTFT

$$
TTFT=t_{first\ token}-t_{arrival}
$$

包括 gateway、排队、prefix lookup/load、prefill、P→D handoff 和首 token 返回。它不是纯 GPU prefill kernel 时间。

### ITL 与 TPOT

第 $j$ 与 $j+1$ 个 token 的间隔：

$$
ITL_j=t_{j+1}-t_j
$$

请求级 TPOT 常写为：

$$
TPOT=\frac{t_N-t_1}{N-1}
$$

平均 TPOT 会掩盖单个严重 stall，交互式服务还要观察 ITL tail。

### Goodput

设请求 $r$ 同时满足所有目标时 $I_r=1$，否则为 0。给定请求率 $\lambda$：

$$
A(\lambda)=\frac{1}{N}\sum_{r=1}^{N}I_r
$$

若要求 $A\ge A_{target}$，则容量是最大可行 $\lambda$，而不是系统过载后最终仍能吐出的 token 数。

vLLM 的 serving benchmark 已支持 TTFT、TPOT、ITL、E2E percentiles 与 `goodput` SLO 参数；这些工具提供测量能力，业务仍需自己定义目标值和 attainment 口径。

## 请求 Trace 至少要保留什么

容量模型的输入不是三个平均数。对每条请求 $r$，建议保留：

$$
r=(t_{arrival}, ISL, OSL,
prefix, model, adapter,
sampling, tenant, deadline)
$$

### Arrival Time

平均 RPS 相同，均匀到达与 1 秒 burst 的 queue tail 完全不同。应保留原始时间戳，或至少拟合 burstiness 和小时/星期季节性。

### Input Sequence Length（ISL）

记录实际 tokenized length，不是字符数。Chat template、tool schema、图片 token 和历史对话都会改变 prefill work。

### Output Sequence Length（OSL）

真实生成长度受 EOS、reasoning、tool call 与 `max_tokens` 影响。只使用客户端上限会高估平均 decode demand，但忽略上限又会低估 tail reservation。

### Prefix Relation

不仅记录“是否命中”，还要记录复用 token 数、命中层级与模板/模型 namespace：

$$
ISL_{uncached}=ISL-prefix\_hit\_tokens
$$

Prefill compute 主要由未命中部分决定，cache load/transfer 则由复用部分及其位置决定。

### Model、Adapter 与 Tenant

不同模型不能共享权重副本；LoRA 可能影响 KV compatibility、batch grouping 和 adapter cache；租户隔离可能禁止 prefix 跨边界复用。把它们混成总 RPS 会隐藏碎片化。

### Sampling 与功能开关

Structured output、logprobs、beam、speculative decode、multimodal encoder 和自定义 logits processor 都可能改变 kernel、CPU 或显存开销。容量 trace 要能按功能分桶。

## 平均 ISL/OSL 为什么会误导

考虑两个集合：

```text
Trace A: 50% (ISL=100, OSL=100)
         50% (ISL=3900, OSL=100)

Trace B: 100% (ISL=2000, OSL=100)
```

二者平均 ISL 都是 2000，Prefill latency 和 batching 却不同：A 的长 prompt 会制造更明显 queue head-of-line 和显存峰值，B 的工作更均匀。

再看相关性：

```text
long input often short output
short chat often long reasoning output
```

若分别采样 ISL 与 OSL 的边缘分布，会产生现实中不存在的“超长输入 + 超长输出”比例，或漏掉真实相关性。应保留 joint distribution 或直接 replay 请求对。

## 第一笔账：模型权重与固定显存

模型参数量为 $P$，权重平均字节数为 $B_w$，仅权重近似：

$$
M_{weight}=PB_w
$$

量化模型还可能包含 scale、zero point、未量化层和 padding，不能只用 `70B × 1 byte` 当作实际显存。

单 rank 的可用于 KV 的显存不是：

$$
M_{HBM}-M_{weight}
$$

而更接近：

$$
M_{KV,budget}=
M_{HBM}
-M_{weight,rank}
-M_{runtime}
-M_{graph}
-M_{workspace}
-M_{safety}
$$

其中：

- runtime：CUDA context、通信库与 allocator；
- graph：不同 batch/shape 的 CUDA Graph pool；
- workspace：attention、MoE、sampling 等临时 buffer；
- safety：碎片、版本变化和瞬时峰值余量。

必须以目标 engine 启动日志和实测峰值校准，不能把所有 free HBM 都承诺给 KV。

## 第二笔账：每个 Token 的 KV Cache

对 $L$ 层、$H_{kv}$ 个 KV heads、head dimension $D$、元素字节数 $B_{kv}$：

$$
M_{KV/token,total}=2LH_{kv}DB_{kv}
$$

K 与 V 对应前面的 2。一个长度 $S$ 的 active sequence 需要：

$$
M_{KV/request}\approx
S\cdot M_{KV/token,total}
$$

还要向上对齐 block size，并加入 allocator metadata。

### 并行后的单 Rank 大小

若 TP 按 KV heads 分片且均匀：

$$
M_{KV/token,rank}\approx
\frac{M_{KV/token,total}}{TP}
$$

若 PP 将层分到不同 rank，每个 rank 只保存自己 stage 的层。MQA 的 KV heads 可能小于 TP degree，复制/分组策略需要以 backend 布局为准。

整个 instance 的 KV 总 bytes 不一定因 TP 下降，主要是分散到更多 GPU；每张卡可容纳 token 增加，但 GPU 成本也增加。

### Active Tokens 才是 Decode 内存单位

令当前请求集合为 $R_{active}$：

$$
T_{active}=\sum_{r\in R_{active}}
(ISL_r+generated_r)
$$

Decode KV 利用率大致随 $T_{active}$ 增长，而不是只随 request count。10 条 100k 上下文与 100 条 1k 上下文，active requests 数更少，KV 压力反而更大。

## 从 KV Budget 推最大并发只是上界

若每 rank KV budget 为 $M_{KV,budget}$：

$$
T_{max,rank}\approx
\left\lfloor
\frac{M_{KV,budget}}
{M_{KV/token,rank}}
\right\rfloor
$$

但不能直接把它换算为 `max_num_seqs`。还要考虑：

- token block 向上取整；
- prefix cache 占用但暂不 active 的 block；
- speculative/tree verification 临时 token；
- preemption/recompute headroom；
- 请求长度 tail；
- backend 对 Graph shape 和 batch 的限制。

接近 100% KV usage 时，新请求分配失败或频繁 preempt，TTFT/ITL 会先于 OOM 恶化。容量目标应在压力测试中找安全 operating region，而不是宣称理论 token 上限都可用。

## 第三笔账：Prefill Compute Demand

最粗略可以统计每秒未缓存 prompt tokens：

$$
D_P=\lambda\cdot E[ISL_{uncached}]
$$

但 prefill time 不是 token 数的全局常数乘法。Attention、MLP、kernel shape、batch 和长上下文算法让 cost curve 随长度变化：

$$
T_P=f(ISL_{cached},ISL_{uncached},batch,
TP,PP,backend,hardware)
$$

正确做法是按长度 bucket profile，例如：

```text
ISL: 128, 512, 2k, 8k, 32k, 128k
batch/num_tokens: multiple levels
cache hit: 0%, partial, high hit
TP/PP candidates
```

得到每种配置的执行时间和 TTFT service curve，再用真实 arrival replay 计算 queue。

Prefix hit 也不等于零成本：external cache 要 lookup、transfer、onboard；本地 HBM hit 才接近直接跳过计算。Profile 要按命中层级分开。

## 第四笔账：Decode Compute 与 Occupancy

每秒需要生成的 token 数为：

$$
D_D=\lambda\cdot E[OSL]
$$

它是 decode work 的起点，却不包含 active duration 和 batch shape。由 Little's Law，在稳定系统中平均并发近似：

$$
N_{active}\approx
\lambda\cdot E[T_{decode\ residence}]
$$

若平均输出 1000 token、平均 ITL 30 ms，仅生成时间约 30 秒；即使 RPS 不高，也会累积大量 active sequences 与 KV。

Decode profile 应扫描：

```text
batch size / active sequences
total active KV tokens
context length distribution
TP/PP/DP/EP
KV dtype and layout
speculative decoding mode
```

相同 batch size 下，8k 与 128k context 的 attention KV read 不同；只测固定短 context 会高估线上 decode capacity。

## Open-loop 与 Closed-loop 基准回答不同问题

### Closed-loop Concurrency

固定 32 个客户端，每个请求完成后再发下一个：

```text
client completion ─► next request
```

系统变慢时，发起速率自动下降，形成背压。适合测某并发下体验，却会掩盖服务过载时真实到达队列。

### Open-loop Arrival

请求按外部时间过程到达，不等待前一请求完成：

```text
arrival clock independent of server latency
```

它能测最大可承载 RPS、queue 增长与 SLO goodput。容量规划应以 open-loop 为主，再用 closed-loop 描述产品实际并发。

基准工具中的“unlimited request rate”更像瞬时 burst，不是线上稳定吞吐。需要扫描有限 RPS，并保留 warm-up、steady-state 和 drain 三段。

## 先做 Engine Micro-profile，再做 Request Replay

### Micro-profile

目标是构造性能面：

$$
Perf(model,hardware,backend,parallelism,
ISL,OSL,batch,KV\ load)
$$

它用于快速筛除不可能的配置，并估算候选 TP/PP 与 P/D ratio。

### Replay

把真实 arrival 和 ISL/OSL/prefix relation 放回完整 serving stack，包含 gateway、router、connector、network 和 cache。Replay 负责捕获 queue、burst、head-of-line、transfer 与 tail。

分析模型不能取代 replay；单次 replay 也无法穷举所有候选。两者应形成：

```text
analytical/perf model → shortlist configs
                       → live replay validates
                       → observed error recalibrates model
```

Dynamo AIConfigurator/Profiler 与 Planner 体现了同样分层：性能模型枚举/选取配置，real-GPU sweep 或 live ForwardPassMetrics 再校正。官方也明确，分析工具不会替你模拟完整 request-by-request scheduler 和 KV 行为。

## Aggregated 配置怎样找容量

对每个候选 TP/PP/replica：

1. 确认模型和目标 max context 能放下；
2. 选择 KV budget 与 scheduler limits；
3. 调整 chunked prefill/token budget；
4. open-loop 扫描 RPS；
5. 计算两项 SLO 同时满足的 attainment；
6. 找到最大可行 $\lambda$；
7. 除以 GPU 数得到 per-GPU goodput。

候选表类似：

| Config | GPUs/replica | Max feasible RPS | Attainment | RPS/GPU |
| --- | ---: | ---: | ---: | ---: |
| TP2 × 2 replicas | 4 | measured | target met | calculated |
| TP4 × 1 replica | 4 | measured | target met | calculated |
| TP4 × 2 replicas | 8 | measured | target met | calculated |

不要假设更高 TP 一定更快。它可能降低单 step latency，却复制通信并减少独立 replicas，排队反而更差。

## P/D 配置怎样分别算

从 profile/simulator 得到：

- 一个 Prefill engine 在 TTFT phase target 下的 capacity $G_P$；
- 一个 Decode engine 在 ITL/TPOT target 下的 capacity $G_D$。

目标请求率 $\lambda$ 的初始 replica 数：

$$
N_P=\left\lceil
\frac{\lambda\cdot H_P}{G_P}
\right\rceil
$$

$$
N_D=\left\lceil
\frac{\lambda\cdot H_D}{G_D}
\right\rceil
$$

$H_P,H_D>1$ 是根据 burst、预测误差和故障目标设置的 headroom factor。两个 factor 不必相同：P burst 可能很尖，D residence 更长但变化较平滑。

若每个 P/D engine 分别使用 $g_P,g_D$ 张 GPU，总 GPU：

$$
N_{GPU}=N_Pg_P+N_Dg_D
$$

然后必须完整 replay，因为 P queue、handoff 与 D active KV 是耦合的，分别达标不保证端到端达标。

## 用一个假设 Profile 演示计算

假设目标是 12 req/s，测得：

```text
P config: TP2, 2 GPUs/replica, phase capacity 4 req/s
D config: TP4, 4 GPUs/replica, phase capacity 7 req/s
headroom: HP=1.25, HD=1.20
```

则：

$$
N_P=\left\lceil\frac{12\times1.25}{4}\right\rceil=4
$$

$$
N_D=\left\lceil\frac{12\times1.20}{7}\right\rceil=3
$$

GPU 预算：

$$
N_{GPU}=4\times2+3\times4=20
$$

这只是候选起点，不是最终答案。若 replay 发现 D KV usage 长期只有 35%，可能降低 D replica 或使用更少 TP；若 P99 TTFT 因 burst 违约，可能增加 P、按 request class 分池，或提高 prefix reuse。

Profile 数字必须来自目标模型与硬件，不能把这个例子代入生产。

## 第五笔账：P→D 与 External KV 带宽

每请求 P→D payload 近似：

$$
M_{handoff}=2L H_{kv}D B_{kv}\cdot ISL_{effective}
$$

平均 payload rate：

$$
BW_{avg}=\lambda E[M_{handoff}]
$$

网络规划不能只满足平均：

$$
BW_{provisioned}\ge
\max(BW_{burst},BW_{concurrent\ flows})
\times overhead
$$

还要与 TP/PP shard mapping、多个 NIC、NUMA、其他 collective traffic 共同评估。

记录三种时间：

```text
post time       提交 transfer 的同步 CPU 开销
xfer time       backend 从提交到完成
visible wait    请求/attention 真正因 KV 未到而等待
```

只有 visible wait 直接进入 SLO，xfer time 决定 pipeline 余量，payload goodput 决定是否接近饱和。

## Prefix Cache 怎样进入容量模型

不要简单乘一个平均 hit ratio。更合适的是按请求计算：

$$
saved\_prefill_r
=f(ISL_r)-f(ISL_r-hit_r)
$$

再减去读取成本：

$$
net\_gain_r=
saved\_prefill_r
-lookup_r-load_r-onboard_r
$$

不同 prefix 位置的计算价值不同，命中层级也不同。HBM local hit、remote DRAM hit 与 SSD hit 不能合并。

容量 planning 还要考虑 cache warm-up：新 replica 刚启动时 hit rate 低，如果离线模型用稳态高命中，扩容后的 TTFT 会被低估。可以采用：

- sticky/KV-aware routing；
- hot prefix prewarm；
- shared external cache；
- scale-down 时迁移热门 block；
- performance model 区分 cold/warm worker。

## MoE 模型要再加一笔 Expert 通信

Dense 模型的 TP/PP profile 不能直接用于 MoE。每个 token 只激活部分 experts，但 expert parallel 会产生 dispatch/combine all-to-all；负载不均还会让某些 rank 成为 straggler。

容量矩阵要加入：

- activated experts/token；
- EP/TP/DP 组合；
- all-to-all payload 与 fabric；
- expert load imbalance；
- shared expert 与 routed expert kernel；
- prefill/decode 不同 token density 下的通信效率。

平均 FLOPs 稀疏不等于网络和权重读取也按同样比例下降。

## Headroom 应该覆盖什么

统一写“预留 30%”很方便，但不同风险需要不同容量：

### Traffic Headroom

覆盖正常预测误差和 burst。由历史 peak-to-forecast error 分布推导。

### Failure Headroom

要求失去一个 node/worker 后仍满足哪一级 SLO。N+1 不是简单多买一台：若一个 node 承载 8-GPU TP instance，就要能在其他 topology 中重建完整 instance。

### Deployment Headroom

滚动升级时新旧版本短暂并存，或先拉起新副本再 drain 旧副本。

### Memory Headroom

吸收 KV length tail、临时 workspace、Graph 和 allocator 波动。

### Control Headroom

扩容需要 $T_{ready}$ 分钟，预测窗口必须覆盖这段 demand 增长。

分别建模后，可以知道成本花在什么风险上，也能在紧急时选择降级哪一项。

## Autoscaling 的容量 Floor 与 Burst Layer

长周期 planner 根据预测流量设置 capacity floor：

$$
N_{floor}=CapacityModel(\widehat W_{t:t+T})
$$

短周期 controller 根据 queue/KV/estimated SLO 添加 burst replicas：

$$
N_{target}=\max(N_{floor},N_{reactive})
$$

这比单一 reactive HPA 稳定：

- floor 提前覆盖 worker cold start；
- reactive layer 处理预测未捕获的 burst；
- scale-down 受 cooldown 和 minimum residency 限制；
- P/D target 受总 GPU budget 共同约束。

Dynamo Planner 当前文档也区分较长周期的 throughput/prediction 与较短周期的 load-based adjustment；具体默认间隔和算法会变化，架构上仍是“预测给下界，实时负载做修正”。

## Worker Ready Time 必须实测

扩容延迟可以分解：

$$
T_{ready}=
T_{schedule}+T_{image}+T_{weight}
+T_{init}+T_{compile/graph}+T_{health}
$$

每段优化方法不同：

- image：node image cache；
- weight：local cache、peer-to-peer weight streaming；
- init：并行 communicator 和 memory registration；
- graph：预编译或预捕获关键 shape；
- health：区分进程存活与真正可接流量。

容量系统应观察 target replica、Pod running、model ready、Router admitted 四个时间点。只看 Kubernetes `Running` 会把尚未加载完成的 GPU 算进容量。

## 一张容量表应该长什么样

### Workload

```text
RPS P50/P90/P99 by 1m/5m window
ISL/OSL joint percentiles
prefix hit tokens by cache tier
model/adapter/tenant mix
cancel/error/rate-limit
```

### Candidate Engine

```text
backend/version/commit
model revision and quantization
GPU SKU / topology / driver
TP/PP/DP/EP
max model length / block size / KV dtype
graph and scheduler settings
```

### Per-config Result

```text
max feasible open-loop RPS
TTFT/ITL/E2E P50-P99
SLO attainment / request goodput
prompt/output tokens per second
active/waiting/preemptions
KV usage and max active tokens
network payload and visible wait
GPU count / GPU-hours / cost
```

### Operational Margin

```text
traffic headroom
failure scenario
upgrade overlap
cold-start time
min/max replicas
scale thresholds and cooldown
```

版本信息必须随结果保存。engine、kernel 或 driver 升级后，旧 profile 不能自动视为仍有效。

## 运行时指标怎样反哺模型

以 vLLM 为例，当前版本提供 running/waiting requests、KV cache usage、prefix query/hit、prompt/generation tokens、请求时延与长度等指标。容量控制可以形成以下诊断：

```text
waiting ↑ + KV low     → compute/queue bottleneck
waiting ↑ + KV high    → memory/concurrency bottleneck
preemption ↑           → scheduler/KV operating point too aggressive
prefix hit ↓           → router/cache warm-up changed P demand
ITL ↑ + network high   → collective/KV transfer contention
```

关联不等于因果，仍需 trace 和 profile。指标名也会随 engine 版本变化，dashboard 应以部署版本的官方 metrics schema 为准。

### 在线校正

记录每个 bucket 的预测与实际：

$$
e_{TTFT}=TTFT_{observed}-TTFT_{predicted}
$$

$$
e_{ITL}=ITL_{observed}-ITL_{predicted}
$$

当误差持续偏移，重新 profile 或更新 correction factor；若只在某模型/长度/租户出现，就拆分 bucket，而不是全局增加 GPU 掩盖问题。

## 常见规划错误

### 用离线 Throughput 当在线容量

Offline batch 能提前知道所有请求并构造满 batch，没有到达、排队和 streaming SLO，通常远高于 online goodput。

### 只测一个 ISL/OSL

会漏掉长上下文 KV 与 burst head-of-line。至少使用多个 bucket，最好 replay joint trace。

### 用 Max Tokens 当实际 OSL

过度保守会浪费大量 D 容量；只用平均 OSL 又会忽略 reasoning tail。使用分布、按产品策略设置 hard cap，并为极端请求单独 pool/queue。

### 把 GPU Utilization 当 SLO

低利用率可能是严格 tail latency 必需的 headroom；高利用率也可能伴随严重排队。利用率是解释变量，不是交付结果。

### 忽略 Cache Cold Start

稳态 hit rate 不能代表新 replica。滚动升级和大规模扩容时尤其明显。

### 把所有请求塞进一个 Pool

128k prompt、短 chat 与长 reasoning 对资源需求完全不同。若 workload 多峰明显，按 request class 建立不同 topology/policy 的 pool，再用 global router 分流，可能比一个折中配置更省。

### 不测 Failure Capacity

正常态有 20% headroom 不代表失去一个 8-GPU 节点仍可运行，因为剩余 GPU 拓扑和 gang placement 可能无法组成新 instance。

## 从规划到上线的闭环

### 1. Trace 清洗

重建实际 token、ISL/OSL、prefix relation、到达和租户边界，过滤测试流量但保留 burst。

### 2. Candidate Enumeration

列出硬件可行的 quantization、TP/PP/EP、aggregated/P/D 与 KV dtype。先用内存公式淘汰放不下的组合。

### 3. Performance Profiling

覆盖长度、batch、KV load 与 cache tier。保存完整环境版本。

### 4. Simulator/Analytical Sizing

计算 P/D demand、network、replica 和 headroom，筛出少量候选。

### 5. Open-loop Replay

从低 RPS 扫到 SLO knee，比较 per-GPU goodput 与 failure scenario。

### 6. Shadow Deployment

在真实镜像流量上只观测不返回结果，验证 cache hit、路由、tail 和资源预测。

### 7. Gradual Traffic

小比例真实流量，设置 SLO、KV、queue 和 error 自动回退门槛。

### 8. Model Reconciliation

把实际 profile 和 forecast error 写回 planner，定期在 engine/模型升级后重新生成容量表。

## 小结

LLM 推理容量规划不是用 GPU 峰值 tokens/s 除以业务 QPS，而是从请求 trace 推导四种约束：未命中输入决定 prefill compute，输出长度和驻留时间决定 decode work，完整 active context 决定 KV memory，P/D 或 external cache 决定数据移动。

正确流程先定义 TTFT、ITL/TPOT、E2E 与 attainment，再保存 arrival、ISL/OSL joint distribution、prefix、模型和功能身份。权重与 runtime 固定显存决定 KV budget；每 token KV 公式给出 active token 上界；长度/batch profile 给出 P/D service curves；open-loop replay 最终找到满足 SLO 的最大 request rate。

分离式部署中，P 与 D 要分别选择并行配置和 replica，再通过 KV transfer、active memory 与 queue 做端到端验证。容量 headroom 也应分为流量、故障、升级、内存和冷启动，而不是用一个模糊百分比覆盖所有风险。

最后，规划表必须回到线上：waiting、KV usage、prefix hit、preemption、transfer 与 TTFT/ITL 持续校正模型。只有“预测—部署—观测—修正”闭环成立，GPU 数量才不是一次 benchmark 的截图，而是可以解释、复算和随流量变化更新的工程决策。

## 参考资料

- [DistServe：Per-GPU Goodput 与 P/D Placement](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin)
- [vLLM Serving Benchmark CLI](https://github.com/vllm-project/vllm/blob/main/docs/benchmarking/cli.md)
- [vLLM Metrics](https://docs.vllm.ai/en/latest/design/metrics/)
- [Dynamo Profiler Guide](https://docs.nvidia.com/dynamo/knowledge-base/modular-components/profiler/profiler-guide)
- [Dynamo Planner Guide](https://docs.nvidia.com/dynamo/knowledge-base/modular-components/planner/planner-guide)
- [AIConfigurator 官方仓库](https://github.com/ai-dynamo/aiconfigurator)
