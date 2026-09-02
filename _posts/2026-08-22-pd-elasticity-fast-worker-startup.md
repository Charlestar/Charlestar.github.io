---
layout: post
title: "P/D 弹性：扩容决定发出后，GPU 为什么还不能接请求"
subtitle: "从 SLO 控制环、权重分发与编译缓存到 Worker Ready Gate，拆解分离式推理的容量生效时间"
date: 2026-08-22 09:00:00 +0800
last_modified_at: 2026-09-03
author: iStar
catalog: true
series: distributed-inference
series_order: 30
technology_year: 2024
mathjax: true
tags: [分布式推理, 分离式推理, Kubernetes]
---

Prefill/Decode 分离之后，两个阶段终于可以独立扩容：输入突然变长就增加 Prefill Worker，输出变长或并发升高就增加 Decode Worker。这个控制思路很自然，却隐藏了一个决定线上效果的时间差。

控制器在时刻 \(t\) 把副本数从 4 改成 6，并不代表时刻 \(t\) 已经多出两份服务能力。新实例还要获得 GPU、拉取镜像、装载数十到数百 GB 权重、建立并行通信组、编译 Kernel、捕获 CUDA Graph、规划 KV Cache，最后通过正确性与健康检查，才可以进入 Router 的候选集合。

如果整段流程要几分钟，而一次流量峰值只持续几十秒，扩容可能在拥塞结束后才生效。此时 Autoscaler 的指标与公式即使完全正确，也无法挽回已经发生的 TTFT/TPOT 违约。

因此 P/D 弹性不是一个简单的副本数控制问题，而是两个互相约束的系统：

```text
容量控制面：何时需要多少 P / D capacity
启动数据面：怎样让目标 capacity 尽快变为 ready capacity
```

本文从固定 P/D 组的局限出发，逐步拆解 SLO 控制环、Worker 启动关键路径、分层权重分发、编译缓存、显存布局复用、Ready Gate 与异构硬件。重点不是复述某个平台的配置，而是建立一套能测量、验证和安全降级的弹性模型。

## 1. 先区分三种“容量”

Kubernetes Deployment 的 `replicas: 8` 只说明控制面期望有 8 个 Pod。对 LLM Serving，更有意义的是三种容量：

```text
desired capacity：控制器希望得到的容量
provisioned capacity：Pod 已获得 CPU、内存、GPU 和网络
ready capacity：模型已加载并能满足当前接口与 SLO 的容量
```

三者之间存在两个延迟：

$$
T_{capacity}
=T_{provision}
+T_{worker\_ready}
$$

其中 `provision` 包含调度、GPU 分配、镜像与容器启动；`worker_ready` 又包含权重、运行时与验证过程。

如果只监控 Deployment 副本数，系统可能显示“扩容完成”，Router 却仍只有原来的 Worker。容量模型必须以 Ready Endpoint 或可实际完成请求的服务率为准。

## 2. 为什么固定 xPyD 组会浪费资源

早期 P/D 部署常把 \(x\) 个 Prefill 与 \(y\) 个 Decode Worker 绑定成一个服务组：

```text
group 0: P0, P1 → D0, D1, D2, D3
group 1: P2, P3 → D4, D5, D6, D7
```

这种拓扑有清晰的故障域，KV 传输目标也比较固定。但流量形态不会总按相同比例变化：

- ISL 增长主要提高 Prefill Compute 与 TTFT 压力；
- OSL 增长会延长 Decode 驻留时间，扩大 KV 占用；
- Prefix Cache 命中会减少 Prefill 工作，却不减少后续 Decode；
- 推理模型的长输出可能让 Decode 突然成为瓶颈；
- 长文档摘要则可能主要压住 Prefill。

若 P 侧排队而 D 侧空闲，扩一个完整 `2P4D` 组会同时增加 2 个需要的 P 和 4 个暂时不需要的 D。资源调节粒度被部署拓扑锁死。

全局资源池把它改成：

```text
             ┌──────── Prefill Pool ────────┐
request ───► │ P0  P1  P2  ...             │
             └──────────┬───────────────────┘
                        │ KV handoff
             ┌──────────▼───────────────────┐
             │ D0  D1  D2  D3  ...         │
             └──────── Decode Pool ─────────┘
```

每个请求在运行时选择 P 和 D，两个池也能独立改变规模。代价是 Router、KV 位置索引、传输数据面和故障处理都从组内问题变成全局问题。

## 3. P 与 D 不能共用一个负载指标

GPU Utilization 不能直接回答用户是否快要遇到 SLO 违约。Prefill 侧和 Decode 侧的用户体验指标不同：

- Prefill Queue、输入长度和 Prefill Service Time 主要影响 TTFT；
- Decode Active Sequences、KV 水位和每轮迭代时间主要影响 TPOT/ITL。

可以分别构造归一化压力：

$$
Load_P=
\frac{\max(TTFT_{pred},TTFT_{actual})}
{TTFT_{target}}
$$

$$
Load_D=
\frac{\max(TPOT_{pred},TPOT_{actual})}
{TPOT_{target}}
$$

这类公式的价值在于把两种量纲变成共同的 SLO 距离：`Load=1` 表示到达目标边界，超过 1 表示服务体验已经或预计将越界。

但它不是通用标准。`max(predicted, actual)` 是一种偏保守策略，是否适合取决于预测误差、指标窗口和业务对成本/违约的偏好。生产控制器还需要置信区间、最小样本数与异常值处理，不能把一个公式当成无需校准的开关。

## 4. 为什么实际指标和预测指标都需要

实际 TTFT/TPOT 能反映真实体验，但它们天然滞后：请求先排队、执行、完成部分阶段，系统才看见指标恶化。

预测信号可以更早看到压力：

```text
P predicted work
  = queued uncached prompt tokens
  × profiled prefill cost curve

D predicted work
  = active + waiting sequences
  × expected remaining output
  × profiled decode iteration cost
```

预测也有风险。用户没有声明可靠的 OSL，模型可能提前 EOS，也可能在推理任务中输出很久；Prefix Cache 命中会随路由变化；不同 Batch Shape 的服务率并非常数。

因此控制环应同时保存：

- 真实完成速率与延迟分位数；
- 队列中的 Token Work，而不是只有 Request Count；
- ISL/OSL 联合分布；
- Prefix Cache 命中与传输位置；
- Worker 冷/热状态；
- 模型预测误差与最近校正因子。

预测负责提前量，真实指标负责纠偏。两者缺一都会让控制器在 Burst 或分布漂移时失真。

## 5. 控制器真正需要预测多远

若新 Worker 从创建到 Ready 需要 \(T_{ready}\)，控制器至少要预测这个时间范围内的容量：

$$
C_{desired}(t)
=f\bigl(W[t,t+T_{ready}], SLO, headroom\bigr)
$$

当 \(T_{ready}=180s\) 时，只看最近 10 秒 Queue Length 是不够的。扩容决策必须更早发出，或者长期维持足以覆盖这 180 秒的 Warm Capacity。

降低 `T_ready` 会直接改善弹性：

- 预测窗口更短，误差通常更小；
- 常驻 Headroom 可以更少；
- 故障替换与滚动升级更快；
- P/D 比例能更及时跟随 ISL/OSL 变化。

这就是 Worker 快速启动为什么不是部署体验优化，而是 Autoscaling 算法的组成部分。

## 6. Worker Ready Time 应怎样分解

完整启动路径可以写成：

```text
T_ready
= T_schedule
+ T_image
+ T_artifact_discovery
+ T_weight_read
+ T_weight_deserialize
+ T_host_to_device
+ T_parallel_init
+ T_compile
+ T_cuda_graph
+ T_kv_profile
+ T_warmup
+ T_verify
```

不同环境的主项完全不同：

- 镜像未预热时，容器层下载可能最慢；
- 共享存储拥塞时，所有 Worker 会争抢同一个 Checkpoint；
- 权重在本地 NVMe，但逐 Tensor Python 反序列化仍可能拖慢；
- 权重已在 GPU，编译和 CUDA Graph Capture 又成为主项；
- 多节点 MoE 还要等待所有 Rank、RDMA 与 Communicator 就绪。

只有给每段建立 Trace，才能决定该做本地模型缓存、P2P 权重传输、AOT 编译还是减少 Graph Shape。笼统记录一个 `pod_start_seconds` 无法定位瓶颈。

## 7. 权重加载首先是一条带宽流水线

模型权重从持久化位置到 GPU 至少经过：

```text
object store / remote filesystem
        ↓
node local disk or page cache
        ↓
host memory
        ↓ PCIe / NVLink-C2C
GPU HBM
```

若模型大小为 \(M\)，路径中可持续带宽最小值为 \(B_{min}\)，忽略其他开销时有下界：

$$
T_{weight}\ge\frac{M}{B_{min}}
$$

真实系统还要支付文件元数据、Shard 打开、校验、反序列化、Layout 转换和并发争用。十个 Worker 同时从同一对象存储下载，不会自动得到十倍总带宽，反而可能制造 Thundering Herd。

因此权重分发要回答两个问题：

1. 与目标 Worker 拓扑最近的兼容副本在哪里；
2. 怎样把 Read、Decode/Deserialize 与 H2D 尽量流水化。

## 8. 多层 Checkpoint Cache 怎样缩短冷启动

ServerlessLLM 的关键观察是，推理服务器通常拥有可利用的近 GPU 存储与主机内存。它把 Checkpoint 放进多层本地存储，并采用面向加载优化的格式，让调度器优先选择已经拥有模型副本的节点。

可以把模型位置分成：

```text
L0: 另一张 GPU 的 HBM
L1: 本机 CPU DRAM / page cache
L2: 本机 NVMe
L3: 机架共享存储
L4: 远端对象存储
```

这里的 L0-L4 只是概念层级，不是统一 API。不同平台可能跳过其中某层。

调度器不应只问“哪台 GPU 空闲”，还要比较：

$$
T_{start}(worker,model)
=T_{queue}+T_{artifact\_to\_GPU}+T_{runtime\_init}
$$

一个稍晚空闲、但模型已在本地 NVMe 的节点，可能比立刻空闲却要跨区域下载的节点更早 Ready。

## 9. 从存储读权重与从现有 Worker 复制权重

当集群已有 Worker 正在服务相同模型时，它本身就是一个权重源。NVIDIA Dynamo 的 ModelExpress 文档描述了这种路径：Source Worker 发布兼容 Tensor 的可用性，新 Worker 通过 NIXL/RDMA 拉取权重；没有兼容源时才回退到存储。它也可以配合 ModelStreamer，从对象存储把 Safetensors 流式送入 GPU。

```text
serving GPU A ── P2P RDMA ──► new GPU B
       │                         │
       └─ keep serving           └─ assemble model
```

它与“共享 PVC 已缓存文件”解决的问题不同：

- 共享文件仍要经历文件系统读取与 H2D；
- GPU-to-GPU 路径可以避免重复从远端存储取完整模型；
- P2P Source 必须保证不会影响前台 Serving SLO；
- 没有 RDMA 或兼容 Source 时仍需可靠回退。

不能看到 `RDMA` 就默认启动一定更快。若 Source GPU 正在高负载 Decode、网络与 KV Transfer 共用链路，权重复制可能干扰线上请求。必须限速、分片、观测并允许 Router 选择其他 Source。

## 10. “兼容权重副本”需要强身份

两个 Worker 的模型名称相同，并不保证 Tensor 可以直接复用。兼容身份至少应包括：

```text
checkpoint content digest
tensor names / shapes / dtypes
quantization format and scales
weight layout / packing version
tensor-parallel and expert placement contract
runtime loader version
target GPU capability where relevant
```

例如 AWQ、NVFP4 或某个融合 Kernel 需要特定 Packed Layout；把原始 Safetensors 与已经转换的 Engine Artifact 混为一谈，会在加载后得到 Shape 正确、语义错误的权重。

Source 发布的不是“我有 Llama”，而应是一个不可变 Artifact Manifest。Target 在变为 Ready 前重新校验分片覆盖、Checksum 和 Rank Ownership。

## 11. 流式加载要避免第二份峰值内存

最直观的加载方式是先把完整权重读入 Host Memory，再整体复制到 GPU。这可能同时保留：

```text
serialized checkpoint
+ deserialized CPU tensors
+ GPU tensors
+ layout conversion workspace
```

对于大模型，多一份完整副本就足以让主机 OOM。更好的加载器以 Tensor 或 Chunk 为单位流水：

```text
read shard i
  → validate
  → optional convert
  → copy to final GPU address
  → release staging buffer
```

双缓冲可以让读取第 \(i+1\) 块与第 \(i\) 块 H2D 重叠，但 Buffer 数量和并发必须受内存预算控制。流式加载优化的是峰值与重叠，不会绕过路径带宽下界。

## 12. 编译缓存与权重缓存必须分开管理

模型加载完成后，Runtime 可能还要执行 `torch.compile`、Triton 编译、Autotune 或 TensorRT Engine 反序列化。它们产生的是另一类 Artifact：

```text
weight artifact：决定模型参数
compile artifact：决定某组代码/shape/hardware 如何执行
```

vLLM 官方调优文档说明，重复启动相同 `(model, config, hardware)` 组合时可以复用编译缓存；但模型、配置、环境变量、PyTorch Build 或 GPU 型号变化都可能使缓存失效。

因此 Compile Cache Key 需要包含：

- 模型结构与 Runtime 配置；
- Framework、Compiler、CUDA 与 Driver 版本；
- GPU Compute Capability；
- Dtype、量化与 Attention Backend；
- 并行规模与捕获 Shape；
- 影响 Codegen 的环境变量。

命中错误缓存比未命中更危险。发布流程应支持“缓存不兼容时明确回退编译”，并记录 Miss 原因，不能为了启动快而静默加载不匹配 Binary。

## 13. AOT、JIT 与 Eager 是启动和稳态的三角关系

可以在三个时间点支付优化成本：

| 路径 | 启动成本 | 稳态性能 | 适合场景 |
| --- | --- | --- | --- |
| Eager | 最低 | 通常较低 | 调试、极短生命周期 |
| JIT + Cache | 首次高，后续低 | 较高 | 配置相对稳定的弹性池 |
| AOT Artifact | 构建阶段高 | 启动稳定 | 硬件与配置可枚举的生产发布 |

vLLM 也允许关闭 CUDA Graph、跳过部分优化来换取更快启动，但这可能降低 Decode 稳态吞吐。Autoscaler 不应只最小化 `T_ready`，而忽略 Worker 进入 Ready 后的服务率。

真正要最小化的是峰值期间的总违约或成本：

$$
Cost=
Cost_{startup}
+Cost_{steady\_state}
+Penalty_{SLO}
$$

短暂 Burst 可以启用轻量启动配置，长期扩容则值得支付更多编译成本；但两类 Worker 的能力必须被 Router 和 Capacity Model 区分。

## 14. CUDA Graph Capture 也属于容量成本

CUDA Graph 可以减少 Decode 的 CPU Launch Overhead，却要在启动时为多个 Batch Shape 捕获执行图，并保留 Graph Memory Pool。捕获形状越多：

- Worker Ready 越慢；
- Graph Artifact/内存越大；
- 留给 KV Cache 的显存可能越少；
- 新配置第一次启动越容易抖动。

可以采用分级策略：

```text
phase 1: eager / limited graph shapes → pass readiness
phase 2: background warm additional shapes
phase 3: advertise upgraded capacity profile
```

但阶段升级不能在未知请求上静默改变数值或产生长时间 Stall。Router 需要知道 Worker 当前支持哪些稳定路径，性能模型也要按阶段更新。

## 15. 显存布局为什么可能成为启动瓶颈

启动过程中，Runtime 要为权重、KV Pool、Workspace 和 CUDA Graph 分配大量显存。反复分配、清零和 Profile 会产生额外时间。

CUDA Virtual Memory Management 提供保留虚拟地址区间、创建物理分配并进行 Map/Unmap 的能力。基于它可以设计稳定的虚拟布局，让某些生命周期切换只重新映射物理页，而不重新构造全部地址关系。

但 VMM 不是“保存上一次进程显存”的通用快照：

- 物理 Allocation、访问权限和进程生命周期仍需管理；
- Weight、KV 与 Graph 对地址稳定性的要求不同；
- Driver/GPU/Runtime 兼容边界必须验证；
- 错误复用可能让旧数据被新 Worker 读取。

把 VMM 用于快速启动时，需要把 Address Plan、Allocation Ownership、Zeroing 和失败回滚写成明确协议，而不是只记录 `cudaMalloc` 次数下降。

## 16. KV Cache Profiling 可以跳过吗

推理引擎常在启动时 Profile 峰值 Activation，再用剩余显存决定 KV Cache 大小。这个过程确保在当前模型、GPU 与运行时配置下不会过度分配。

重复启动完全相同环境时，可以把已验证的 KV Bytes 作为输入跳过 Profile。vLLM 官方文档也提供显式 KV Cache Memory 配置来复用上次结果。

风险是“相同环境”经常不成立：

- 同卡还有其他进程；
- Driver/Kernel Workspace 改变；
- Graph Shape 或并行规模变化；
- Multi-LoRA、Encoder 或新 Backend 引入额外显存；
- MIG/容器限制不同。

保守值会降低并发，激进值会启动 OOM。系统需要在 Manifest 中绑定硬件和配置指纹，失败时自动回到完整 Profile，而不是无限重启同一个错误 Pod。

## 17. 并行 Worker 的 Ready 是一个 Barrier

TP、PP 或 EP 实例不是若干独立 Pod 的简单集合。只有全部 Rank 满足同一 Generation 的条件，整个实例才可服务：

```text
all ranks scheduled
→ identical artifact manifest verified
→ communicator established
→ rank placement agreed
→ warmup/canary passed on all ranks
→ group epoch published
→ router endpoint becomes ready
```

任何一个 Rank 失败，都不能让其余 Rank 以“部分 Ready”接收请求。否则请求可能在 Collective 中永久等待，或者不同 Rank 使用不同权重版本。

控制面应发布 Group-level Readiness，而不是把多个 Pod 的 HTTP `/healthz` 简单相加。滚动升级也应按完整 Parallel Group 切换 Generation。

## 18. Ready Probe 不能只检查端口

端口开始监听只证明进程启动。一个可接流量的 LLM Worker 至少要通过：

- 模型与 Tokenizer Revision 一致性；
- 全部 Tensor Checksum/分片覆盖；
- Collective 与 KV Connector 连通；
- 最小 Prefill + Decode Canary；
- 量化、Sampling、EOS 与结构化输出基础路径；
- KV Pool 容量和 Free Block 基线；
- 本 Worker 的性能 Profile 已注册；
- 目标 Generation/Lease 仍有效。

Canary 不必跑完整评测集，但要能发现“权重加载完成却 Layout 错了”“某 Rank 版本不同”“Connector 只单向可达”等启动类故障。

只有通过这组 Gate，Worker 才应加入 Router。`desired replicas` 和 `ready service rate` 的差值也应作为一等指标暴露。

## 19. P/D 全局池怎样选择一次请求的两端

P/D 都 Ready 后，请求还要动态配对：

```text
choose P_i by:
  uncached prefill work
  + queue delay
  + prefix locality

choose D_j by:
  KV transfer time from P_i
  + decode queue
  + available KV blocks
  + expected output residency
```

不能先独立选“最空 P”和“最空 D”，再忽略二者之间的传输拓扑。一个 D 虽然 Queue 最短，却可能跨慢链路并且 KV 已接近高水位。

Router 需要把 Cache Location、拓扑与负载放进同一成本函数，同时保留最大等待时间，防止 Prefix Locality 让某些请求长期黏在热门节点。

## 20. 异构 P/D 的容量不能直接按 GPU 数相加

Prefill 更偏 Compute，Decode 更依赖 HBM Capacity/Bandwidth，这使不同硬件分别承载 P 与 D 具有吸引力。但异构集群不能说“2 张 A 卡等于 2 张 B 卡”。

每个 Worker Class 都需要独立 Profile：

```text
P class:
  prefill_ms(ISL, cached_prefix, batch_tokens)

D class:
  iteration_ms(active_sequences, kv_lengths)
  kv_capacity_tokens

link class:
  transfer_bw / setup_latency / contention
```

此外还要验证：

- P 写出的 KV Dtype/Layout 能被 D 原样消费；
- Position、Shard Ownership 与 Quantization Metadata 一致；
- 跨厂商设备是否有可用的直接传输路径；
- 不支持时是否经 Host Staging，以及成本多大；
- 两端 Kernel 数值差异是否在接受范围。

异构 P/D 是一个联合兼容矩阵，不只是 Scheduler 给 GPU 打不同标签。

## 21. 扩缩容控制必须防止振荡

若 `Load_P` 刚超过 1 就扩 P，刚低于 1 就缩 P，启动延迟和观测噪声会让系统不断来回变化。稳定控制器通常需要：

- Scale-up 与 Scale-down 使用不同阈值；
- 规定最短 Ready/Idle 时间；
- 加入 Cooldown 与每轮最大变化量；
- 统计目标容量与实际 Ready 容量差；
- 扩容期间仍保留 SLO 降级策略；
- 缩容前先 Drain，不能直接切断 Active Decode。

P 与 D 还存在耦合。只扩 P 会更快地产生 KV Handoff，可能把本已紧张的 D 压垮；只扩 D 又可能没有足够 Prefill 输出可消费。Planner 要检查下一状态的端到端可行性。

## 22. 缩容比扩容更需要状态语义

空闲 P Worker 通常可以较快退出；D Worker 可能持有大量 Active Sequence 与 Prefix KV。缩容顺序应是：

```text
mark draining
→ router stops new assignments
→ migrate / finish / explicitly abort active work
→ publish KV ownership changes
→ revoke lease and endpoint
→ release GPU
```

直接删除 D Pod 会把正常缩容变成故障恢复，并可能造成客户端流中断。若支持请求迁移，还要传输已提交 Token、KV、Sampling RNG 与约束状态；否则只能等待请求自然结束，并把 Drain Time 纳入 Scale-down 成本。

权重 Source Worker 也不能在仍被 ModelExpress Target 使用时突然消失。Source Selection、传输 Lease 与取消需要明确协议。

## 23. 模型化调优与强化学习应该放在哪一层

未来控制器可以用时序模型或强化学习选择 P/D 比例、扩容阈值和 Cache Policy，但学习策略不应直接绕过硬安全边界。

较稳妥的分层是：

```text
hard constraints:
  GPU / KV capacity
  compatibility matrix
  min replicas / fault headroom
  rollout and tenant policy

optimizer:
  choose feasible P/D target
  choose worker class and cache placement

runtime guard:
  reject unsafe action
  detect drift
  fall back to conservative controller
```

训练目标也不能只有平均 GPU Utilization。至少要包含 TTFT/TPOT 违约、单位有效 Token 成本、启动/迁移开销、尾延迟和失败率。

把优化器限制在可行域内，才能让“自适应调优”成为改进建议，而不是拥有无限权限的线上实验。

## 24. 应该观测哪些时间和状态

一套可诊断的弹性面板至少包含：

```text
desired / provisioned / ready P replicas
desired / provisioned / ready D replicas
P queue tokens / predicted TTFT / actual TTFT
D active sequences / KV usage / predicted TPOT / actual TPOT
schedule_to_gpu_seconds
image_pull_seconds
weight_source_tier
weight_read / transfer / deserialize seconds
compile_cache_hit / compile_seconds
cuda_graph_capture_seconds
kv_profile_seconds
parallel_group_barrier_seconds
canary_seconds
router_registration_seconds
```

每次扩容还应带一个 `scale_decision_id`，贯穿 Planner、Kubernetes Event、Artifact Loader、Worker Trace 和 Router Registration。否则很难回答“这次扩容为什么三分钟才生效”。

## 25. 如何验证弹性确实改善了 SLO

单 Worker 冷启动计时不等于完整验证。需要回放动态流量：

1. **ISL Burst**：只提高输入长度，观察是否主要扩 P；
2. **OSL Burst**：增加长输出，观察 D/KV 是否先升压；
3. **Cache Shift**：突然降低 Prefix 命中，验证 P 预测修正；
4. **Cold Cluster**：没有本地模型与编译缓存；
5. **Warm Peer**：已有兼容 Worker 可做 P2P Source；
6. **Source Failure**：权重传一半时 Source 退出；
7. **Mixed Hardware**：不同 P/D Class 与跨设备链路；
8. **Rolling Upgrade**：新旧 Artifact Generation 并存；
9. **Scale Oscillation**：周期性负载验证 Hysteresis；
10. **Drain with Long Decode**：缩容时仍有长输出。

报告应同时给出容量生效时间、SLO 违约面积、峰值成本、缓存命中、前台流量受干扰程度和回退次数。只报告“启动从 10 分钟降到 10 秒”，无法说明前台 Decode 是否因权重传输变慢。

## 26. 一条可落地的实施顺序

可以按下面顺序逐步建立能力。

### 第一步：分开 desired 与 ready capacity

让 Planner 只使用已通过 Gate 的服务率计算可用容量，建立完整 Worker 启动 Trace。

### 第二步：建立启动关键路径基线

分别测量冷/热镜像、远端/本地权重、编译命中/未命中、Graph Capture 与 KV Profile，不提前假设瓶颈。

### 第三步：先做安全的 Artifact Cache

固定权重与编译 Manifest、Checksum、容量和逐出策略；再加入 P2P/RDMA Source，保留存储回退。

### 第四步：将 Ready Time 放入 Autoscaler

按 P/D 分别使用 Token Work、SLO 反馈和启动提前量，加入 Headroom、Hysteresis 与 Cooldown。

### 第五步：解除固定组绑定

建立全局 P/D Pool、KV/拓扑感知配对和 Group-level Readiness，再逐步独立扩缩两侧。

### 第六步：最后引入异构与学习优化器

先有每类硬件的性能曲线、KV 兼容矩阵和保守回退，再扩大优化空间。

## 27. 结语

P/D 分离解决了“Prefill 和 Decode 能否使用不同资源”，弹性系统还要解决“需要的资源何时真正可用”。副本数变化只是控制命令，权重、编译、显存、并行组与正确性验证全部完成后，容量才真正生效。

缩短 Worker Ready Time 会同时改善扩容、故障恢复和滚动升级，但每一种捷径都有边界：P2P 权重传输可能干扰前台网络，编译缓存可能版本不兼容，跳过 KV Profile 可能 OOM，过早 Ready 可能把错误权重带入服务。

因此一套可信的 P/D 弹性系统必须把容量控制面与启动数据面连起来：Planner 预测 TTFT/TPOT 压力，Artifact Loader 选择最快兼容来源，Runtime 复用可验证缓存，Parallel Group 通过统一 Ready Gate，Router 最终只接纳真正可服务的容量。

当 `desired → provisioned → ready` 的每一步都可观察、可校验、可回退时，扩缩容才不再是“希望几分钟后有 GPU”，而是一个可以纳入 SLO 预算的确定性系统过程。

## 参考资料

- [快手万擎大模型推理成本和性能优化实践](https://zhuanlan.zhihu.com/p/2067652898524345525)
- [DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin)
- [ServerlessLLM: Low-Latency Serverless Inference for Large Language Models](https://www.usenix.org/conference/osdi24/presentation/fu)
- [NVIDIA Dynamo ModelExpress](https://docs.nvidia.com/dynamo/knowledge-base/kubernetes/model-loading/model-express)
- [NVIDIA Dynamo Planner](https://docs.nvidia.com/dynamo/knowledge-base/modular-components/planner/overview)
- [vLLM Optimization and Tuning](https://docs.vllm.ai/en/latest/configuration/optimization/)
- [CUDA Driver API: Virtual Memory Management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VA.html)
