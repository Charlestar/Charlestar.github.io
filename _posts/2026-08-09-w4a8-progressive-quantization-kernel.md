---
layout: post
title: "W4A8：4-bit Weight 怎样喂给 8-bit Tensor Core"
subtitle: "从 Progressive Quantization、寄存器解包到 QServe 的系统协同设计"
date: 2026-08-09 17:49:00 +0800
last_modified_at: 2026-08-09
author: iStar
catalog: true
series: gpu-runtime-precision
series_order: 60
technology_year: 2024
mathjax: true
tags: [AI Infra, W4A8, QServe, INT4, INT8, LLM Serving]
---

W4A16 与 W8A8 各自只解决了一半问题。W4A16 把 weight 压到 4-bit，适合降低小 batch Decode 的权重带宽，却要在计算前恢复到 FP16/BF16；W8A8 可以直接利用 INT8 Tensor Core，但 weight 容量和带宽仍是 8-bit。W4A8 看起来正好取两者之长：4-bit weight storage，加上 8-bit activation 与高吞吐矩阵乘。

真正实现时，4-bit weight 与 8-bit activation 不能直接因为名字相邻就相乘。以 QServe 为代表的整数 W4A8 路径，会先把 packed UINT4 weight 在寄存器中恢复成一个 INT8 中间表示，再执行 INT8 × INT8 Tensor Core GEMM。weight 解包、group scale、zero point 与 layout conversion 都落在 GEMM 热循环附近，任何低效步骤都可能吃掉 4-bit 带来的带宽收益。

所以 W4A8 的核心问题不是“能不能把模型保存成 4-bit”，而是：

```text
4-bit storage format
  → load and unpack efficiently
  → reconstruct an 8-bit compute operand
  → keep correction work off the critical main loop
  → execute Tensor Core GEMM
  → apply scales and epilogue
```

这是一项 quantization algorithm、packed layout 与 GPU kernel 的共同设计。

## W4A8 不是唯一一种格式

`W4A8` 只表示 weight 使用 4-bit、activation 使用 8-bit，并没有说明整数还是浮点：

| 名称 | Weight storage | Activation | 可能的计算路径 |
| --- | --- | --- | --- |
| Integer W4A8 | INT4/UINT4 | INT8 | weight 展开为 INT8，执行 INT8 Tensor Core GEMM |
| FP8 W4A8 | INT4/FP4 或带 scale 的 4-bit weight | FP8 | 转为 backend 支持的 FP8/低精度 operand 后计算 |
| W4A16 | INT4/UINT4 | FP16/BF16 | 在线反量化 weight，执行高精度 GEMM |

QServe 论文中的 QoQ 是 `W4A8KV4`：4-bit weight、INT8 activation、4-bit KV Cache。当前 TensorRT-LLM / Model Optimizer 中也存在名为 W4A8 AWQ 的 recipe，其 activation 可以是 FP8。两者不能只凭 `W4A8` 字符串互换 artifact 或推断 kernel。

阅读配置时至少要继续问：

- 4-bit 是 signed、unsigned、integer 还是 floating encoding？
- activation 的 8-bit 是 INT8、E4M3 还是其他格式？
- 计算 operand 是多少 bit，accumulator 是什么 dtype？
- weight 在 main loop 中怎样解包、反量化或重编码？
- scale 是 per-channel、per-group、per-token 还是 block-wise？

位宽只是契约的开头，不是契约本身。

## 用 Roofline 理解为什么需要 W4A8

线性层可以写成：

$$
Y_{M\times N}=X_{M\times K}W_{K\times N}
$$

在 LLM Serving 中，$K$、$N$ 通常很大，$M$ 随阶段和 batch 改变：

- 小 batch Decode 的 $M$ 很小，GEMM 常受 weight memory bandwidth 限制；
- Prefill 或高并发 Decode 的 $M$ 变大，weight 被更多 token 复用，计算吞吐更重要。

这解释了两类传统路径的优势区间：

```text
W4A16:
  weight traffic 更小
  → 小 M 时有利
  → 大 M 时 FP16/BF16 compute 和反量化可能限制吞吐

W8A8:
  weight traffic 比 W4 多
  → 小 M 时带宽优势有限
  → 大 M 时可利用 INT8 Tensor Core 吞吐
```

理想 W4A8 同时减少 weight traffic，并保留 8-bit Tensor Core compute。但 Roofline 只计算字节与峰值算力，没有自动计入 unpack、zero-point correction、scale multiplication、register pressure 和额外指令。W4A8 能否接近这条理想曲线，取决于 kernel 是否把转换成本压到足够低。

QServe 论文在其硬件、模型和 batch 假设下给出了具体交叉点与吞吐结果；这些数字用于说明设计动机，不应直接套到其他 GPU、shape 或 runtime 版本。

## Storage Precision 不等于 Compute Precision

一个 UINT4 weight 只占半个 byte，通常两个值打包进一个 byte：

```text
byte: [ high_nibble | low_nibble ]
       weight_1       weight_0
```

但 Ampere/Hopper 上的经典 QServe W4A8 路径并不是让 tensor core 直接接受“一个 UINT4 operand 和一个 INT8 operand”。kernel 要先把 nibble 解包为 8-bit values，再用 INT8 MMA 指令计算。

因此需要分别描述：

```text
storage dtype: UINT4 / INT4
intermediate compute operand: INT8
activation operand: INT8
dot-product accumulator: wider integer
epilogue/output: typically FP16 in QServe
```

这与 AWQ 常见的 W4A16 类似：checkpoint 是 4-bit，只说明 weight storage；若运行时先恢复成 FP16 再 GEMM，乘法精度仍是 FP16。评估低精度系统时必须把 storage、operand、accumulator 与 output 四个层次分开。

## 为什么 Naive Group Dequantization 很慢

GPU GEMM 通常把输出 tile 固定在寄存器或 accumulator 中，沿 reduction 维 $K$ 反复迭代。这段迭代就是 main loop：

```text
for each K tile:
  load activation tile
  load weight tile
  transform operands if needed
  tensor-core MMA

epilogue:
  apply scale / bias / activation
  store output
```

main loop 会执行很多次，里面每增加一条依赖链都会被重复支付。W8A8 可以把 INT8 operands 直接送入 tensor core，大部分 scale 工作放在 epilogue。Naive W4A16 则需要在每个 K tile 中把 4-bit weight 恢复为 FP16；Naive per-group W4A8 也要解包、减 zero point、乘 group scale，再形成 INT8 operand。

Tensor Core 的峰值吞吐很高，而普通 CUDA Core 上的标量/向量转换、地址运算和整数处理相对容易成为瓶颈。最终可能出现：

```text
Tensor Core 等待 operand
→ theoretical low-bit TOPS 很高
→ 实际 kernel 被 dequantization main loop 限制
```

这就是为什么单纯把 group size 调小以提高精度，可能同时让 scale/zero metadata 与 dequantization 工作变多。

## Progressive Group Quantization 的两级表示

QServe 的 QoQ 不直接从浮点 weight 得到“任意 group INT4 再恢复 FP16”，而是建立一个适合 INT8 compute 的两级表示。

第一层先把浮点 weight 按较粗粒度量化为 INT8 中间值：

$$
W\approx S^{(0)}Q^{(0)}_{s8}
$$

其中 $S^{(0)}$ 是第一层浮点 scale，$Q^{(0)}_{s8}$ 是希望在计算时恢复出的 signed INT8 weight。

第二层再把这个 INT8 中间值按 group 压成 UINT4：

$$
Q^{(0)}_{s8}
\approx
S^{(1)}_{u8}
\left(Q^{(1)}_{u4}-Z^{(1)}_{u4}\right)
$$

组合后：

$$
W\approx
S^{(0)}
S^{(1)}
\left(Q^{(1)}_{u4}-Z^{(1)}\right)
$$

这里两级 scale 的职责不同：

- $S^{(0)}$ 把最终 INT8 dot-product 对应回浮点 weight 范围，适合延后到 epilogue；
- $S^{(1)}$ 把每个 UINT4 group 恢复成 INT8 中间 operand，必须在使用该 group 时处理；
- $Z^{(1)}$ 描述非对称 UINT4 group 的 zero point。

运行时无需把 weight 完整恢复成 FP16。它只需高效地把 packed UINT4 重建为 INT8 operand，随后执行：

$$
Q_X^{int8}Q_W^{int8}
$$

这就是 progressive quantization 的系统价值：第二级反量化的目标不是原始浮点 weight，而是 tensor core 可以直接消费的 INT8 weight。

## 为什么需要保护 INT8 范围

两级量化有一个容易忽视的边界：UINT4 值经 scale 与 zero point 恢复后，必须仍落在 signed INT8 范围。

例如，某组 INT8 中间 weight 的范围近似为 $[-113,120]$。将它非对称压到 $[0,15]$ 后，整数化的 group scale 可能为 16；最大 UINT4 code 反量化后可能得到 128，已经超过 INT8 最大值 127。

可以在运行时 saturation，但 saturation 指令和误差都不是免费的。QServe 选择在第一级量化时把对称 INT8 范围从常见的 $[-127,127]$ 收紧到保护范围 $[-119,119]$，以约束第二级恢复后的溢出风险。

这个细节说明：

```text
两个单独看起来合法的量化层级
≠ 组合后一定得到合法的 compute operand
```

多级量化必须验证所有 code、scale、zero point 的最坏情况，而不只是随机抽样后的平均误差。

## 解包需要为 Tensor Core 的消费顺序服务

普通 checkpoint 常按逻辑矩阵顺序保存 weight，但 GPU threads 获取 MMA fragment 的顺序并不等同于连续矩阵顺序。4-bit storage 与 8-bit compute 还带来一个额外问题：线程按相同 byte 数取数时，拿到的 4-bit element 数量是 INT8 的两倍，通用 load/permute 路径未必能直接得到每个线程需要的 fragment。

QServe 采用 compute-aware weight reordering：离线把 weight 按 kernel 中线程和 MMA tile 的实际消费顺序重排。运行时每个线程可以读取连续的 packed word，用少量逻辑指令并行展开多个 nibbles，而不是在 main loop 中进行复杂 pointer arithmetic 和跨线程重排。

这类 layout 有三个含义：

1. packed artifact 与 kernel 强绑定，不能当成通用 row-major INT4；
2. 重排应在导出或 engine build 阶段完成，不能放到请求热路径；
3. kernel version、tile shape、GPU architecture 变化时，cache key 也要变化。

一个数学上正确的 INT4 checkpoint，若 layout 与 kernel contract 不一致，会得到完全错误的输出；若运行时临时转 layout，则会增加启动时间、峰值内存和部署复杂度。

## Zero Point 为什么最好离开 Main Loop

非对称 group quantization 通常需要：

$$
q_w-z_w
$$

若对每个 weight 在 main loop 里逐个减 zero point，指令数与依赖都会增加。对 per-channel 情形，可以利用代数展开把修正搬到 epilogue。设某个 output channel 的 zero point 为 $z_w$：

$$
\sum_k x_k(q_{w,k}-z_w)
=
\sum_k x_kq_{w,k}
-z_w\sum_kx_k
$$

第一项走 tensor core dot product；第二项由 activation 的 row sum 和 zero point 构成，可以独立计算并在 epilogue 修正。

per-group zero point 会随 $K$ group 改变，不能简单地只做一次全局 epilogue 修正。此时 kernel 需要在组边界处理 correction，并尽量使用寄存器级向量指令并行完成。优化的关键不是“完全没有 dequantization”，而是减少 main loop 中低吞吐指令的占比并打破串行依赖。

## Activation 的 INT8 路径从哪里来

W4A8 的另一半是 activation quantization。若 activation 仍是 BF16，kernel 就变成 W4A16，而不是整数 W4A8。

QServe 的 block 边界保持 FP16：线性层输入先量化为 INT8，W4A8 GEMM 使用 INT8 Tensor Core，输出经 epilogue 恢复为 FP16；attention 等路径仍可在 FP16 中计算。这种设计限制了低精度边界，便于与非 GEMM 算子集成。

Activation INT8 仍会遇到 outliers，因此 W4A8 recipe 常组合前文介绍的思路：

- smoothing 把稳定的 channel outlier 迁移到 weight；
- rotation/reordering 改善 group 内部分布；
- per-token 或 per-tensor scale 处理请求间变化；
- 少量敏感模块保留更高精度。

但 4-bit weight 比 INT8 weight 更脆弱。把 activation outlier 迁到 weight 时，迁移过多会扩大某些 weight group 的范围，使 UINT4 rounding error 上升。因此 smoothing 参数要同时优化 activation INT8 与 weight INT4，不能直接复制 W8A8 的 alpha。

## W4A8 与 KV4 是三条不同路径

QServe 的完整名称是 W4A8KV4，但三种 dtype 解决不同资源：

```text
W4: linear weight capacity and bandwidth
A8: GEMM operand and compute throughput
KV4: attention cache capacity and read bandwidth
```

KV Cache 不参与线性层 W4A8 GEMM。Decode attention 通常要持续读取历史 K/V，算术强度低，长上下文或高并发时容易 memory-bound，所以 KV4 有独立价值。但它也需要单独的 quantization、scale metadata、fused attention kernel 和质量验证。

若只是部署 W4A8 weights/activations，不能宣称已经获得 KV4 容量；若单独打开 KV4，也不会让 MLP GEMM 自动变快。

QServe 为 KV4 引入 SmoothAttention 并针对 attention kernel 做系统优化。这说明 weight、activation 与 KV 的低精度必须逐条闭环，而不是由一个总开关隐式完成。

## 当前 FP8 W4A8 是另一条执行路线

在支持 FP8 Tensor Core 的 GPU 上，一些现代 recipe 把 4-bit weight 与 FP8 activation 组合，也称 W4A8。它的动机仍然是兼顾 weight compression 与 8-bit compute，但数值和 kernel 契约不同：

- activation 要选择 FP8 format 与 scaling recipe；
- 4-bit weight 需要转换为 kernel 所需的 FP8 或其他 compute representation；
- accumulator、output 和 epilogue 遵循浮点低精度规则，而不是 INT32 dot-product 规则；
- backend 的支持矩阵随 GPU architecture、模型和版本变化。

因此看到 `w4a8_awq` 之类配置时，应以该版本官方文档、导出工具与 profiler 为准。不要拿整数 QServe 的两级 UINT4→INT8 公式解释所有 FP8 W4A8，也不要把某个 GPU 上的支持推断到其他架构。

## W4A8、W4A16、W8A8 怎样选择

可以从瓶颈和实现成熟度出发：

| 条件 | 更值得优先验证的路径 | 原因 |
| --- | --- | --- |
| 小 batch、weight bandwidth 主导 | W4A16 或 W4A8 | 4-bit weight traffic 更小 |
| 大 Prefill / 高并发、GEMM compute 主导 | W8A8 或成熟 W4A8 | 8-bit Tensor Core 吞吐更重要 |
| activation outlier 难以控制 | W4A16 / FP8，或更强校准 | W8A8/W4A8 的 activation 精度风险更高 |
| runtime 没有优化 W4A8 kernel | W4A16 或 W8A8 | naive dequantization 可能抵消理论收益 |
| 显存主要被 KV Cache 占用 | 独立评估 KV quantization | weight 从 8-bit 降到 4-bit 不解决主要容量项 |

这不是静态规则。模型 shape、batch scheduler、TP shard、GPU bandwidth、Tensor Core 代际和 attention 占比都会移动交叉点。最终选择必须来自目标硬件上的真实 traffic benchmark。

## 容量计算要把 Metadata 加回来

4-bit weight 主体为每个参数 0.5 byte，但 group quantization 还需要 scale、zero point、padding 和 alignment。假设每 $G$ 个 weights 保存一个 16-bit scale 与 4-bit zero point，忽略对齐时，平均额外成本近似为：

$$
\frac{2+0.5}{G}\ \text{bytes/weight}
$$

若 $G=128$，约为 $0.0195$ bytes/weight；若 $G=32$，约为 $0.0781$ bytes/weight。更小 group 往往提高精度，却增加 metadata traffic，并让 kernel 更频繁地切换 scale。

Progressive quantization 还包含第一级 scales，实际 artifact 大小应从文件与运行时 resident memory 测量。不能简单用参数量乘 0.5 推导进程显存，更不能据此推导最大并发，因为 KV、workspace、graph pool 和 allocator 仍然存在。

## 量化算法怎样保护 4-bit Weight

4-bit 只有 16 个 code，weight 质量比 W8A8 更敏感。一个完整 recipe 通常会组合多种技术。

### Group Quantization

缩小每个 scale 覆盖的范围，让不同 weight groups 不必共享同一动态范围。代价是 metadata 和 dequantization 频率上升。

### Activation-aware Reordering

用 calibration activation 判断 channel salience，再把相近通道放进同一组。这样极重要通道不会随机与分布完全不同的通道共享 group scale。重排必须同步修改 activation 或相邻权重，保持模型函数一致。

### Weight Clipping

牺牲少量极值，换取普通 weight 更细的 quantization grid。clip ratio 应以 layer output error 或任务质量选择，而不是只最小化 weight 自身 MSE。

### Smoothing / Rotation

降低 activation outliers，使 A8 更稳定；但要同步观察迁移后 W4 group 的误差。对输入模块、输出模块和 attention projection，最佳策略可能不同。

### Mixed Precision Exclusion

Embedding、LM head、某些 projection 或异常敏感层可以保留更高精度，但排除列表应来自消融实验。过多 fallback 会破坏 kernel coverage 和性能可预测性。

## 正确性验证要覆盖五道边界

W4A8 的系统链比单级 weight-only quantization更长，建议按以下顺序验证。

### 1. Mathematical Transform

关闭量化，确认 smoothing、rotation、channel reorder 与 folding 前后输出等价。先排除 graph rewrite 错误。

### 2. Quantization Reference

在高精度中模拟两级 quantize/dequantize，检查每层误差、饱和率、zero rate 和最坏 group。明确 rounding 与 clamp 规则。

### 3. Packed Decode

对 packed UINT4 做 CPU 或简单 GPU reference decode，逐元素比较期望的 INT8 中间 weight，覆盖 nibble 顺序、sign、padding 和尾部 group。

### 4. Real GEMM

比较目标 W4A8 kernel 与 reference GEMM，覆盖不同 $M/N/K$、非对齐维度、bias、epilogue、TP shards 和多个 GPU architecture。

### 5. Serving Output

比较 logits、greedy token、任务成功率、长上下文质量、TTFT、TPOT、goodput、显存与功耗。采样输出本身有随机性，不能只肉眼比较几段文本。

每一层验证只跨一个新边界，才能区分算法误差、packing bug、kernel bug 和系统调度问题。

## Profiler 中应该看什么

一次可信的性能报告至少回答：

- W4A8 kernel 覆盖了多少 linear layers 和请求 buckets？
- weight unpack / dequant 占 main loop 多少周期？
- Tensor Core pipe 是否在等待 CUDA Core 或 memory dependency？
- global/shared memory load 是否合并，bank conflict 是否异常？
- register pressure 是否降低 occupancy？
- activation quantization 是否与上游/当前 GEMM 融合？
- epilogue 是否融合 scale、zero correction、bias 和 activation？
- unsupported shape 是否 fallback 到 W4A16/BF16？

Microbenchmark 要比较相同 $M/N/K$、相同 output dtype 与相同 epilogue。Serving benchmark 则要固定 scheduler、KV dtype、并行策略、上下文分布和 SLO。否则观察到的差异可能来自 KV4、batching 或更大并发，而不是 W4A8 GEMM 本身。

## Artifact Manifest 必须比 “int4” 更详细

建议至少保存：

```yaml
format: w4a8
recipe: <qoq / awq / gptq / custom>
base_model_revision: <immutable revision>
weight:
  storage_dtype: <uint4 / int4 / fp4>
  intermediate_dtype: <int8 / fp8 / other>
  group_size: <n>
  symmetric: <true / false>
  scale_levels: <n>
  protected_range: <optional>
activation:
  dtype: <int8 / fp8 format>
  granularity: <per-tensor / per-token / block>
  dynamic: <true / false>
compute:
  mma_dtype: <int8 / fp8 / other>
  accumulator_dtype: <dtype>
  output_dtype: <dtype>
packing:
  backend: <name and version>
  gpu_arch: <target>
  tile_layout: <layout id>
parallelism:
  tp_size: <n>
calibration:
  dataset_revision: <revision or checksum>
  method: <recipe details>
```

同一模型即使量化参数一致，只要 tile layout 或 GPU target 不同，也可能需要不同 artifact revision。运行时加载时应验证 contract，而不是失败后静默 fallback。

## 一条可执行的部署路径

1. **建立 BF16、W4A16、W8A8 三条基线**：明确目标 workload 的带宽与计算区间；
2. **锁定 W4A8 backend**：确认 activation 格式、compute operand、GPU support 与 kernel shape；
3. **构造 calibration set**：覆盖生产任务、长度、语言、长尾和并行配置；
4. **选择量化 recipe**：联合搜索 smoothing/reorder、group size、clipping 与敏感层；
5. **验证两级数值范围**：枚举 protected range、scale、zero point 的边界；
6. **离线 pack 到目标 layout**：连同 scales、zeros、shard metadata 生成不可变 artifact；
7. **逐元素验证 unpack**：确认每个 nibble 到 INT8/FP8 operand 的映射；
8. **逐 shape 验证 real kernel**：覆盖 Prefill、Decode、padding 与 fallback；
9. **分别衡量 W4、A8、KV4**：一次只改变一条低精度路径做消融；
10. **运行真实 traffic benchmark**：比较 TTFT、TPOT、goodput、容量和成本；
11. **发布 kernel coverage 指标**：把 fallback 与 quantization overhead 纳入监控；
12. **灰度与回滚**：artifact、engine、runtime、driver 与 GPU architecture 一起版本化。

## 常见误区

### “Weight 是 4-bit，所以一定执行 INT4 乘法”

4-bit 常是 storage precision。QServe 的整数 W4A8 路径把它恢复为 INT8 operand，再执行 INT8 Tensor Core GEMM。

### “W4A8 一定比 W4A16 和 W8A8 都快”

理论上兼顾带宽和算力，实际还受 unpack、scale、zero point、shape、fusion 与 fallback 限制。

### “Group 越小质量越好，所以应一直缩小”

metadata、scale load、main-loop correction 与 layout 成本会增加，性能可能下降。

### “W4A8KV4 是一个不可拆分的开关”

Weight、activation、KV 属于不同 kernel 和容量路径，必须分别验证。

### “所有 W4A8 都可以加载同一个 checkpoint”

INT8 activation 与 FP8 activation、两级整数表示与 FP8 recipe、不同 packed layouts 都不兼容。

## 小结

W4A8 的难点不在把两个 weights 塞进一个 byte，而在让 4-bit storage 持续、低成本地供应 8-bit Tensor Core main loop。

可以抓住九点：

1. W4A8 只描述位宽，必须继续明确 INT8 还是 FP8 activation；
2. Storage、compute operand、accumulator 和 output precision 是四层契约；
3. W4A8 试图结合 W4A16 的权重带宽与 W8A8 的计算吞吐；
4. main loop 中的 unpack/dequantization 会让理论 Roofline 失效；
5. Progressive Quantization 先定义 INT8 中间 weight，再把它按 group 压成 UINT4；
6. 两级 scale 与 protected range 要保证恢复结果合法且可高效计算；
7. Compute-aware reorder 和寄存器级并行是 packed artifact 的一部分；
8. W4、A8 与 KV4 解决不同瓶颈，不能互相代替；
9. 只有 real kernel、coverage、质量和 SLO goodput 同时通过，W4A8 才算落地。

至此，量化路径已经从 FP8、SmoothQuant W8A8、AWQ W4A16 延伸到 W4A8。下一篇将回到低精度 kernel 的通用支撑层：FlashInfer 怎样把不同 attention、GEMM、sampling 与量化实现组织成可组合的 Serving primitives。

## 参考资料

- [QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving](https://arxiv.org/abs/2405.04532)
- [MIT HAN Lab: QServe Official Repository](https://github.com/mit-han-lab/qserve)
- [NVIDIA TensorRT-LLM: Quantization](https://nvidia.github.io/TensorRT-LLM/1.1.0/features/quantization.html)
- [NVIDIA TensorRT Model Optimizer: LLM PTQ Examples](https://github.com/NVIDIA/TensorRT-Model-Optimizer/tree/main/examples/llm_ptq)
- [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438)
- [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978)
