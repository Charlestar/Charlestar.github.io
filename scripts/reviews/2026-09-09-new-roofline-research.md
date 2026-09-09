# Roofline：调研与写作交接

状态：调研完成；2026-09-09。全部数值示例为公式计算或显式构造，不包含 GPU 实测、性能分析复现或硬件成绩。

## 选题与系列定位

- 固定文件：`_posts/2026-09-09-roofline-gpu-performance-model.md`。
- 建议标题：`Roofline：从计算量、数据流量到可信的 GPU 性能判断`。
- `series: gpu-runtime-precision`、`series_order: 5`、`technology_year: 2009`、`mathjax: true`、`tags: [GPU优化, LLM推理]`。5 在现有 CUDA Graph=10 之前，不改旧文。
- 现有 FP8、W4A8、编译器和注意力优化文章会用到算术强度、带宽限制、吞吐，但没有独立的模型推导与测量口径文章。本篇补足这些文章的前置判断工具，不重复写量化内核或 CUDA Graph API 教程。
- 2009 对应 CACM 代表性论文发表年份；Berkeley 同名技术报告 UCB/EECS-2008-134 是 2008-10-17，故不能说 Roofline 于 2009 才首次出现或首次发明。

## 实际查阅的一手来源与阅读边界

1. [Williams、Waterman、Patterson 原文档案，第 1 页](https://digital.library.unt.edu/ark:/67531/metadc934195/m1/1/)：这是原作者论文的 UNT 政府文档镜像；实际读取可访问的第 1 页 OCR，其中 §3 开头解释 operational intensity 是操作数除以穿过 cache 与 DRAM 边界的流量，并强调 sustainable bandwidth，不是芯片引脚带宽。原 CACM 页面 403，Berkeley PDF 跳转受限，第 2/3 页读取失败，因此本调研不宣称完整读过原始论文。原报告元数据可查 [Berkeley 技术报告](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.html)；该页搜索返回了作者、报告号、2008 日期，直接打开未成功。
2. [NVIDIA Matrix Multiplication Background](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)：实际读取 §1–3.2，尤其 GEMM 的 `2MNK`、算术强度、tile/wave quantization、复用与并行度的权衡。该页 2023-02-01 更新；只引用原理，避免把旧示例设备或版本要求写成当前所有 GPU 的定律。
3. [NVIDIA GPU Performance Background](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html)：实际读取 Understanding Performance 相关模型与 kernel 行为说明。用于区分 math/bandwidth/latency 约束，避免将低算术强度直接解释成带宽已饱和。
4. [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)：2026-09-09 查阅的页面版本为 13.3；实际读取 §2.2.1、§2.2.3、§2.2.6、§2.6.1–2.6.3、§2.9 及 §2.10 相关内容。重点是 sets/sections、replay/开销、序列化、时钟控制、默认 replay 前清缓存、Roofline 图和 memory chart。最新版在线文档未来可能变化，文章应注明所查页面版本而非宣称跨版本固定行为。
5. [NERSC Roofline Performance Model](https://docs.nersc.gov/tools/performance/roofline/)：实际读取模型定义、hierarchical Roofline、ERT 与 profiling 段落。分层模型使用各存储层自己的流量；实测/经验峰值应报告条件。该页含旧 V100/nvprof 教程，不应复制其命令作为现代统一操作流程，也不要把旧时某工具的存储层支持边界当成当前限制。
6. [CUDA C++ Best Practices Guide 12.8.0](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-best-practices-guide/index.html)：实际读取 §3.1、§6.1/6.3、§8.1.1–8.1.2、§8.2.1–8.2.3。固定版本来源支撑正确性比较、异步执行下的 CPU/GPU 计时、CUDA events、有效带宽与实际流量、GB/GiB 一致口径。
7. [Time-Based Roofline for Deep Learning Performance Analysis, v1](https://arxiv.org/html/2009.04598v1)：实际读取 §I–II.C 与 §III 实验设置。该文以原始研究强调：更高 FLOP/s 可以同时对应更多工作与更长时间；launch 开销和总工作量不能被传统吞吐图掩盖。只借其分析观点，不复述 V100 特定 launch 时间为通用常数，不声称复现论文实验。

正文建议以官方文档和上述可读原文为就近引用来源；不需要把原论文访问失败过程写入技术正文，也不能写“通读原论文”。

## 必须先锁定的定义与推导

令一次明确计时范围执行 `F` 个浮点运算，在存储层级 `l` 传输 `Q_l` 字节，耗时 `t` 秒：

\[
P=\frac{F}{t},\qquad I_l=\frac{F}{Q_l}.
\]

`F` 的单位是 FLOP（操作计数），`P` 是 FLOP/s（速率），`I_l` 是 FLOP/byte。令 `C` 为匹配当前数值格式、指令路径和统计口径的计算上限，`B_l` 为同一层级的带宽上限，单位 byte/s。理想资源下界与 Roofline 上界是：

\[
t\ge\max\!\left(\frac{F}{C},\frac{Q_l}{B_l}\right),\qquad
P\le\min(C,B_lI_l),\qquad I_l^*=\frac{C}{B_l}.
\]

- `max` 对应理想情况下计算和数据搬运可充分重叠的资源下界，不保证真实实现达到，也不是两段耗时一定完全重叠的断言。额外串行依赖、低并行度、指令发射、同步和启动成本会使实测更慢。
- 点在 ridge 左侧：带宽线是这两个潜在上界中较低的一条；这不证明真实 HBM 已跑满。右侧同样不证明 Tensor Core 已饱和。必须结合实际流量、利用率、指令路径和时间线。
- 多层近似约束可写 `P <= min(C, B_HBM I_HBM, B_L2 I_L2, ...)`，每个 `I_l` 都必须使用相同层级的 `Q_l`。不能拿 HBM 算术强度乘 L2 带宽。
- 峰值必须匹配 FP32、FP16/BF16 Tensor Core、FP8、稀疏/稠密、累加方式和计数；不把 INT8 TOPS 当 FP32 TFLOP/s。一个 FMA 通常按 2 FLOP 计数，全文保持一致。
- 用“有用算法 FLOP”还是“实际执行 FLOP”都必须说清；padding、重计算、反量化可以改变实际工作。一个融合/算法改写降低总 FLOP 后，FLOP/s 下降仍可能更快，优化目标首先是正确结果下的时间与资源预算。

## GEMM 可复算例：batch 如何移动潜在瓶颈

矩阵 `A[M,K]` 与 `B[K,N]` 相乘写 `C[M,N]`，三者 FP16、每元素 2 byte，`beta=0`，忽略低阶系数处理。为隔离模型，假设输入在计时前不驻留更近缓存，且每个输入只从外存读取一次、输出只写一次：

\[
F=2MKN,\qquad Q_{\mathrm{model}}=2(MK+KN+MN).
\]

这只是冷缓存、理想复用的张量流量估算，不能无条件叫作真实 DRAM 最小流量：暖 L2 可减少 HBM 流量，重复读取/写回/transaction 效率可增加流量。`beta != 0` 时需要读旧 C，公式相应增加 `2MN` 字节；其他 dtype、epilogue、workspace 必须重算。

选 `K=N=4096`，虚构设备 `C=120 TFLOP/s`、`B_HBM=2 TB/s`（十进制），不是任何具体型号的规格或实测。ridge 为 `60 FLOP/byte`。此时：

\[
F=33\,554\,432M,\quad
Q_{\mathrm{model}}=33\,554\,432+16\,384M,\quad
I_{\mathrm{model}}=\frac{2048M}{2048+M}.
\]

以下数值已用本地 JavaScript 实际复算；时间是资源下界分量，不是 benchmark：

| M | F（FLOP） | Q_model（byte） | I（FLOP/byte） | F/C（微秒） | Q/B（微秒） |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 33,554,432 | 33,570,816 | 0.999512 | 0.279620 | 16.785408 |
| 32 | 1,073,741,824 | 34,078,720 | 31.507692 | 8.947849 | 17.039360 |
| 128 | 4,294,967,296 | 35,651,584 | 120.470588 | 35.791394 | 17.825792 |
| 512 | 17,179,869,184 | 41,943,040 | 409.600000 | 143.165577 | 20.971520 |

正文可以选前三行，重点解释权重在更多行之间复用；越过 ridge 只是理想模型的主导上界变化。M 是参与这次 GEMM 的行数，不必等于服务请求 batch。prefill 和 decode 包含多个算子/通信/访存阶段，不能从一张 GEMM 表宣布整个 prefill 必定 compute-bound、decode 必定 memory-bound。

## 第二个可选例：汇总 Roofline 会丢失顺序结构

同一虚构设备上，两个必须严格串行执行的阶段：

- A：`F_A=0.12 GFLOP`、`Q_A=200 MB`，计算下界 1 微秒、访存下界 100 微秒。
- B：`F_B=12 GFLOP`、`Q_B=2 MB`，计算下界 100 微秒、访存下界 1 微秒。

分阶段下界之和为 `100+100=200` 微秒；把两阶段粗略汇总成一块，只得到 `max(12.12 GFLOP / 120 TFLOP/s, 202 MB / 2 TB/s)=101` 微秒。后者仍是合法但更松的下界，不能当作可达到的端到端时间。需明确严格串行假设；不要从此推断所有 kernel 均不能重叠。

可用该例收束到关键路径；若篇幅紧，替代例是 Amdahl：占总时长 20% 的 kernel 加速 2 倍，端到端最多 `1/(0.8+0.2/2)=1.111...` 倍，假设其他部分不变。两例不必都展开。

## 建议递进大纲：十节各自承担不同问题

1. **为什么相同矩阵乘法会有不同速度？** 从 M 变化与权重复用提出问题；先定义优化对象是一次 kernel、一个依赖阶段还是端到端请求，交代本文是模型推导与测量方法，没有实测成绩。
2. **两个资源账本导出一条屋顶。** 先介绍 F、Q、t，再推出下界和上界，最后 ridge。读者应能独立检查所有单位；解释模型忽略什么，而不先堆硬件术语。
3. **给一段 GEMM 逐项记账。** 从读 A/B、写 C 到 `2MKN` 和字节公式，再用三行表解释 M。保留 beta、冷/暖缓存、复用假设，连接批量化而不泛化整个 LLM 阶段。
4. **左边、右边、屋顶下面分别说明什么？** 区分潜在限制和真实饱和；低于屋顶可能来自 occupancy、依赖、指令路径、transaction 效率、启动开销。GPU utilization 的 busy 百分比不是带宽/计算峰值利用率，也不等于 occupancy。
5. **缓存让“搬了多少字节”变成层级问题。** 用 HBM/L2 的不同 Q 解释 hierarchical Roofline；算法张量账本、请求字节和物理流量不是同一个数。串联 tiling、fusion 和 FlashAttention，但不要给没有测量的数据移动量结论。
6. **FLOP/s 不是优化目标本身。** 正确选 dtype/稀疏/指令峰值；区分有用与执行工作；量化/融合可能改动工作量与流量。提醒同样结果、同样精度、同样 shape 才能建立公平对比。
7. **先获得可信的时间。** 正确性检查、预热、预分配、同一 stream 的 CUDA events、重复样本与分位数；CPU 端到端时间单独度量。简短代码可选，若给代码应明确依赖版本与计时范围，不声称运行过 GPU。
8. **再用分析工具补齐 F 和 Q。** 时间线定位 → 选代表性 kernel/范围 → 明确 metric 定义与层级 → 用 F、Q、t 重算点。用发现 sets/sections 的方式而非硬编码跨架构 counter 名；指出 replay、flush、串行化、时钟控制改变测量条件。
9. **从单点返回端到端关键路径。** 用严格串行两阶段的 200/101 微秒例说明汇总丢失依赖，或 Amdahl 小例；再连接 CUDA Graph 的主机提交成本和融合的边界改变。
10. **收束为可执行判断。** 固定工作与正确性、选择存储边界和峰值、检查模型与实测差距、提出假设、验证时间收益。不要用十几项泛化清单替代本文推导。

## 测量流程的关键限定

- CPU 发起 CUDA kernel 通常异步；只围住 launch 的普通计时主要测提交，不自动等于 GPU 完成时间。若用 CPU 测完整 GPU 工作，必须显式定义并同步测量边界。
- CUDA event 记录在 stream 中。简单示例只覆盖一个 stream：记录 start、提交范围、记录 stop、等待 stop 完成，再读取 elapsed；单位通常是毫秒。其他 stream 的工作除非显式依赖汇合，不自动纳入该范围。
- events 时间也可以包含 stream 在两事件间等待依赖或等主机继续提交的空闲间隙；不能一律称“纯 kernel 执行时间”。预热包括库初始化、编译、autotune；若任务本来就是衡量冷启动，则应保留这些开销并单独命名指标。
- 检查数值误差时选择与 dtype/算法相符的容差，随机输入覆盖应服务于被验证性质；通过一个随机张量不等于证明全域正确。
- 报告 shape、dtype、布局、device/software 版本、是否稀疏、缓存策略、预热/样本、计时范围以及是否包含传输/分配。GB=10^9，GiB=2^30，同一比值不要混用。
- Nsight Compute 收集多组指标可能 replay；kernel replay 会保存/恢复有关设备内存，application replay 重启应用，均有额外成本。进程墙钟总耗时不是未分析运行的 latency。
- profiler 的序列化可能消除原本并发；默认 replay 前清缓存改变暖缓存重用；时钟控制和温度限制也影响上限。关闭 cache flush 不是自动更正确，需要匹配问题、报告条件并检查 replay 可重复性。
- 实际 API/metric 以本机 `ncu --list-sets`、`ncu --list-sections`、查询支持指标与所装文档为准；不要承诺一个架构的 counter 字段适用所有 GPU。文章可以给流程，不强行附一条“放之四海皆准”的长命令。
- 要关联 unprofiled 时间与 profiler 计数，必须核对相同 shape、工作与运行条件；不能把冷缓存计数与暖缓存时间随手拼成一个物理性能点。

## 审核重点与非目标

- 所有不等式、单位、FMA 计数、三个表格行与 200/101 微秒例可直接复算。
- 无具体硬件峰值承诺；假想 120 TFLOP/s、2 TB/s 必须紧邻标识。文章没有“我们测得”“实验表明”措辞。
- 不宣称超过一条经验 roof 必然不可能：经验 ceiling 是特定测量条件下的参考；超过可能来自条件/数据口径不同或更有效内核，先查证。理论 peak 也必须匹配硬件状态和指令定义。
- 不将点在带宽区域等价于 HBM 饱和；不将“GPU 利用率高”当成“接近计算峰值”；不把 occupancy 最大化当作优化目标本身。
- 不把 logical bytes 当 DRAM 实测；不跨存储层混用强度；不混合有用 FLOP、执行 FLOP 与不同精度峰值。
- 不把 `max` 当实测延时或将多段时间简单无条件相加；不把单 kernel 提升直接等同端到端吞吐/延时提升。
- 可以建议一张简洁概念图或仅用公式与计算表；不做伪 benchmark 图。无需增加硬件 SKU 比较或泛化 GPU 微架构百科。
