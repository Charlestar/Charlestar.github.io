# Triton GEMM：research 交接

日期：2026-09-13。状态：一手资料与固定版本源码核对完成，可交 writer。未运行 Triton/GPU，不声称跑赢 cuBLAS 或通读全部编译器代码。

## 覆盖与编排

- 已检查现有正文。torch.compile 文讲图断点/编译边界，FlashInfer Bench 文讲生产验证，Roofline 文已推整体 GEMM 2MNK 与外存流量；都没有从 program tile 到 stride/mask/K累积的独立 Triton GEMM 推导。
- root确认 `series: gpu-runtime-precision`、`series_order: 15`、`technology_year: 2019`、`tags: [GPU优化, 工程实践]`。2019 对应 MAPL 原论文，不说 Python 前端在2019就与今天相同。承接 Roofline5 与 CUDA Graph10，后接 FP8 20。
- 建议题目：`Triton GEMM：从一个输出 Tile 到复用、边界与调优`。只解决同一段dense GEMM，不扩展成语言/微架构百科。明确指 GPU 编程语言，不是同名 NVIDIA Triton Inference Server。

## 实际阅读范围与一手链接

1. [IBM 原作者论文页](https://research.ibm.com/publications/triton-an-intermediate-language-and-compiler-for-tiled-neural-network-computations)：实际读取标题、摘要、作者与2019-06-22/MAPL元数据。
2. [作者站原论文 PDF](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf)：通过网页文本实际查读 §3.1–3.3 的 tile、broadcast、SPMD 与 §5.2.1–5.2.3 hierarchical tiling/coalescing/shared allocation。没有完整审阅IR、所有passes或实验。旧 Triton-C 语法不是本文API，引用只支持抽象与历史。
3. [Triton 官方 Programming Guide Introduction](https://triton-lang.org/main/programming-guide/chapter-1/introduction.html)：实际读取 blocked-program 模型与 CUDA 对照；该在线页2026-09-13访问，可变。正文主体API以固定v3.1.0为准。
4. [v3.1.0 官方 matrix multiplication tutorial](https://github.com/triton-lang/triton/blob/v3.1.0/python/tutorials/03-matrix-multiplication.py)：实际读取 blocked算法、pointer arithmetic、group ordering、kernel、wrapper、unit test与benchmark口径。核到 M/N `%` 包回、K masked zero、FP32 accumulator、FP16 output、grouped pid、autotune shape key。本文可做更直观的显式M/N/K load mask版本，必须说明是解释性简化，不是逐字复现官方最快实现。
5. [v3.1.0 language/core.py](https://github.com/triton-lang/triton/blob/v3.1.0/python/triton/language/core.py)：实际读取 `program_id`、`arange`、`dot`、`load`、`store`、`where`相关接口。`tl.where`两分支都会求值，不能把未mask的tl.load包在where里防越界。dot对FP32输入的input_precision与FP32 output/accumulator不是同一概念，本文FP16主路径不展开TF32百科。
6. [v3.1.0 language/semantic.py](https://github.com/triton-lang/triton/blob/v3.1.0/python/triton/language/semantic.py)：实际读取 `dot`形状/dtype核验和输出选择：匹配inner dimension，非batch块维度>=16；BF16输入选择FP32返回；BF16不能直接作为dot out_dtype而应cast。这里只以FP16输入/FP32acc/FP16输出写代码，不把手算2x2例当合法Tensor Core tile。
7. [v3.1.0 runtime/autotuner.py](https://github.com/triton-lang/triton/blob/v3.1.0/python/triton/runtime/autotuner.py)：实际读取Config文档、run缓存key、autotune文档和reset/restore逻辑。num_warps/num_stages是编译调度选项，不是BM/BN/BK别名；v3.1.0 run会在指定key之外附tensor dtype，因此不能说“key=['M','N','K']完全忽略dtype”，但stride/layout并不自动都是调优key；JIT specialization不等价于autotune key。
8. [NVIDIA Matrix Multiplication Background](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)：实际读取§1、§2.1–2.3、§3.1–3.2。借用tile复用vs并行度、tile/wave quantization，不复制该页旧GPU固定倍数为现代预测。K内循环分块和split-K并行必须区分。

## 锁定算子合同

本文只实现 C=A B，A[M,K]、B[K,N]、C[M,N]，alpha=1、beta=0，无bias/activation、无批维/稀疏/量化、同GPU二维FP16输入、FP32累积、FP16输出。M,N,K>0；输入是有效非重叠tensor view，允许普通正stride转置/切片；输出新分配连续且不alias输入。FP32输出或BF16扩展需要改wrapper dtype和最终cast，不能只把一处类型改掉。

数学 C_mn=Σ_(k=0)^(K-1) A_mk B_kn。FMA记2FLOP，主要有用工作F=2MNK。并行划分M/N，因为不同输出位置互不依赖；每个program持有BM×BN累加tile，沿K以BK推进。普通这段实现每个program由GPU上多个线程协作，不是每个program一个CUDA thread，也不是一元素一个SM。正文只讲普通num_ctas=1配置，避免用一条规则泛化clusters/persistent kernels。

## 由地址到代码的最小推导

矩阵元素地址（typed pointer的元素单位，不是bytes）：

- A[m,k]：a_ptr + m*s_am + k*s_ak。
- B[k,n]：b_ptr + k*s_bk + n*s_bn。
- C[m,n]：c_ptr + m*s_cm + n*s_cn。

一维grid：gridM=ceil(M/BM)，gridN=ceil(N/BN)，pid∈[0,gridM*gridN)，pid_m=pid//gridN，pid_n=pid%gridN。rm=pid_m*BM+arange(BM)，rn=pid_n*BN+arange(BN)，rk=q*BK+arange(BK)。rm[:,None]、rk[None,:]广播成BM×BK的A指针；rk[:,None]、rn[None,:]广播成BK×BN的B指针。

每次q：

- a = load(Aptrs, mask=(rm<M)&(rk<K), other=0)
- b = load(Bptrs, mask=(rk<K)&(rn<N), other=0)
- acc = dot(a,b,acc)
- 完成ceil(K/BK)轮后store时mask=(rm<M)&(rn<N)。

如果q每次直接进入地址，则不要再把指针额外累计推进一次，防止K偏移算两次。或者采用起始地址+每步加BK*stride二者择一。last K tile必须填0；mask只遮store不能防止之前load越界。`tl.where(mask, tl.load(ptr), 0)`不是安全替代，load仍执行。

官方tutorial的M/N `%M/%N`允许padding位置重复加载有效行/列，最后store丢弃这些无效输出；K维不能照搬取模，因为每个k贡献会累到有效输出。

### 地址反例

存储为2×3的矩阵X=[[10,11,12],[20,21,22]]，其转置B=X.T形状3×2，stride=(1,3)。B[2,1]应为22，offset=2*1+1*3=5。若误用连续3×2的stride=(2,1)，offset=5恰好巧合仍对，因此不要拿这个元素做唯一验证！使用B[1,0]应为11，正确offset1，错误offset2会读12；B[0,1]应为20，正确offset3，错误offset1会读11。建议正文取B[1,0]说明shape不能决定stride。

### K取模反例

手算1×3 dot：[1,2,3]·[4,5,6]=32。概念BK=2，第2块应[3,0]·[6,0]；若K索引[2,3]%3变[2,0]，多加1×4，得36。这是数学小例，非可直接传tl.dot的合法tile参数。

## 可复算形状、计数与性能模型

### Ragged三个维度都要覆盖

M=130,N=70,K=65，BM=64,BN=32,BK=32：gridM=3、gridN=3，总9program；每program3轮K。最后输出tile `(pid_m,pid_n)=(2,2)`实际只有2行×6列=12个有效输出；末K块只有索引64有效。

有用F=2*130*70*65=1,183,000 FLOP。若所有边缘tile仍执行完整块dot，规则块工作模型F_tile=2*(ceil(M/BM)BM)*(ceil(N/BN)BN)*(ceil(K/BK)BK)=3,538,944 FLOP；有效比例≈33.428%。这是规则tile的模型上计数，不是本机instruction counter，也不能据此严格推耗时3倍。mask通常不把Tensor Core运算自动压缩成任意稀疏shape。

### 为什么增加BM/BN能改善复用，但增加BK不自动改善同一比值

只看一个满tile的一轮K，输入每元素b bytes，输入加载模型Q_stage=b(BM*BK+BK*BN)，F_stage=2BM*BN*BK，因此

I_stage=2BM*BN/[b(BM+BN)]。

这是program逻辑输入载荷与该轮FLOP比值，忽略输出、元数据、cache、transaction；不是完整GEMM HBM算术强度，不能直接贴到Roofline HBM横轴。b=2且BM=BN=T时I_stage=T/2 FLOP/byte。64×64=32，128×128=64；BK代数相消，所以BK翻倍不在此模型中增加复用，却可减loop次数、改变Tensor Core调度/流水/资源开销。

FP32 accumulator逻辑数据量4BM BN：64×64=16KiB，128×128=64KiB。它不是每线程寄存器量，也不承诺全部简单铺在寄存器上；实际layout、live range、编译器分配/溢出另核。大tile提高复用却减少program数量、增加资源压力和边缘浪费。

每program完整K逻辑输入载荷模型约bK(BM+BN)，但矩阵边缘masked应按有效输入计；不同program可能重复需要A/B，L2可以复用，因此不能把所有program load字节总和宣称HBM实测。

### 调度顺序不是执行先后保证

grouped pid把靠近的一组program映射到有共享A/B数据的输出tile，以提高L2复用机会。不能保证GPU严格按pid执行、相邻program一定在同SM、cache永不被驱逐，不能用group order实现跨program依赖。小文解释reuse动机即可，不必把完整复杂group映射再贴一遍；若贴必须处理末group较小。

split-K是让多个program并行算同一输出的K分段，随后合并partial结果；与单program内部BK循环不同。会有workspace/归约/原子与精度顺序成本，所以不要把“增BK”写成“增K并行度”。只用一段说明边界，不做第二套kernel。

以上算术已本轮用JavaScript复算。未实际编译或跑GPU，所有性能数值均为模型，不作测量结果。

## 建议正文大纲

1. **从Roofline的理想复用到一段具体kernel。** 为什么同样2MNK差很多？锁定C=AB合同/版本/无GPU成绩。
2. **把输出切开，K维留在同一份累加器里。** 标量式→tile式→program grid，解释为什么M/N并行独立、K归约不能随意交换。
3. **shape描述数学，stride决定地址。** 写三条地址式、broadcast维度，转置反例，让读者能手查每个指针。
4. **三个尾块，三种必须做对的边界。** 130×70×65例贯穿，K零padding反例、M/N输出mask、where非惰性。
5. **把这些约束装进一个短kernel。** 显式mask、FP32 accumulator、FP16 store；wrapper验证dtype/device/dim/nonalias/正shape；固定BM64/BN32/BK32例如4warps3stages只是讲解参数，不宣称最快。
6. **为什么tile越大不一定越快。** I_stage推导、64/128例、BK相消、acc资源、grid并行度与tile/wave尾部，指出这不是HBM实测。
7. **从单tile复用走向相邻program的L2复用。** grouped mapping原则、调度非保证、K分块vs split-K；图可用2×2输出tile的共享A/B标签，不画虚构性能曲线。
8. **调的是资源和形状的共同约束。** BM/BN/BK、warps/stages、shape/stride/dtype key、编译与autotune冷启动成本、候选非法配置与重复副作用；不照搬官方某设备最佳组。
9. **怎样证明优化没有只让错误更快。** 非整除/转置/切片/小K/幅值、FP32/FP64 CPU数学参考与低精度舍入误差分离、sanitizer可选、warmup之后固定outputbuffer计时，明确编译/autotune/分配/转换/传输是否包含。用有用2MNK报吞吐并同时报latency，性能取决于匹配库基线。
10. **回到用途。** 定制epilogue/fusion为何有价值但改变合同；通用GEMM不自动胜过vendor library，说明精度和验证边界。

## 审核关卡与非目标

- wrapper应拒绝空shape或显式正确处理，示例不用modulo从而没有%0，但launch空grid和K0不在合同内；正stride支持不等于所有重叠as_strided输入都能泛化。
- torch新输出可连续，stride可传可省；不要声称本教学wrapper支持任意dtype/所有backend。FP16 CUDA路径只锁接口，环境未GPU测试。
- 暂不涉及TMA、warp specialization、persistent GEMM、Blackwell TMEM、Gluon、IR pass全景、完整autograd backward、自定义梯度。若能不用这些词解释主线，就不增加术语负担。
- 明确dtype三分：FP16输入、FP32累加、FP16输出；FP32累加并不等于FP32输入精度，转换回FP16还会舍入/溢出。
- CPU手算模拟通过不能证明Triton编译、GPU访存安全、真实性能与完整floating semantics正确；未运行GPU请醒目标注。
- benchmarks保持输入、layout、output dtype/precision、epilogue完全可比，异步计时边界和warm cache条件可链接Roofline，不复制一长段通用工具清单。
