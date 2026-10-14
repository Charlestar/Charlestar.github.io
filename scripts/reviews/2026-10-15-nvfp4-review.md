# NVFP4 / MXFP4 独立审核记录

日期：2026-10-15。对象：`_posts/2026-10-15-nvfp4-mxfp4-format.md`。收到 writer 静止稿后独立阅读全文和三个 Python 块，结合实际一手来源审查概念、公式、数值合同、分组和预算。写作前预读不作为未写正文的“已通过”结论；本记录对应最终稿的真实执行。

## 一、一手来源与实际阅读范围

- [OCP Microscaling Formats Specification v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)：实际读取 §4 Scope、§5.1–5.2、§5.3.3 FP4、§5.4.1 E8M0、§6.1–6.3 及相关表的可提取文本；版本表 2023-09-07，封面 Sep2023。本代理的截图请求没有取得可检视图像，不能说本代理完成了视觉表格核查。root 另行做过 PDF 视觉检查并在 research §8 记录，二者证据不混用。
- [NVIDIA NVFP4 官方介绍，2025-06-24](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)：读格式表、E2M1、16 元素微块、E4M3 与张量 FP32 scale 的正文；不依赖页面 AI 生成摘要，也没有移植其模型精度或性能数字作为本机结果。不把“更灵活的 scale”强化为任意块最小 MSE 的定理。
- ModelOpt 0.35.0 固定提交 `c359cb774db4daab623cddde6c7e0399a49cf7d2`：完整读 [`nvfp4_tensor.py`](https://github.com/NVIDIA/Model-Optimizer/blob/c359cb774db4daab623cddde6c7e0399a49cf7d2/modelopt/torch/quantization/qtensor/nvfp4_tensor.py)、[`mxfp4_tensor.py`](https://github.com/NVIDIA/Model-Optimizer/blob/c359cb774db4daab623cddde6c7e0399a49cf7d2/modelopt/torch/quantization/qtensor/mxfp4_tensor.py)、[`backends/nvfp4_gemm.py`](https://github.com/NVIDIA/Model-Optimizer/blob/c359cb774db4daab623cddde6c7e0399a49cf7d2/modelopt/torch/quantization/backends/nvfp4_gemm.py)、[`triton/fp4_kernel.py`](https://github.com/NVIDIA/Model-Optimizer/blob/c359cb774db4daab623cddde6c7e0399a49cf7d2/modelopt/torch/quantization/triton/fp4_kernel.py)。定点读 `utils.py` 的 block reduction/padding、`BaseQuantizedTensor.__init__` 及 `config.py` 对应默认 recipe。未读全部 CUTLASS/CUDA 实现、未运行这些官方路径。

## 二、主要概念、公式与实现合同

1. E2M1 bias1；非零指数含隐含前导1，零指数不含。16 种位模式含 ±0，仅15个不同实数；正码本 `[0,.5,1,1.5,2,3,4,6]`，最小 normal1、唯一正 subnormal.5、最大6，无 NaN/Inf。没有把全1指数当保留区。最近偶数的“偶”由编码低位决定，七个 midpoint 的结果正确。
2. finite-input RNE/saturation/signed-zero 是文章参考合同，不等于所有实现路径的转换政策。固定 NV 以 `x<0` 提符号，不保留输入负零；MX strict-`>` midpoint 与 RNE 的三个上舍 tie 不同。正文没有把参考 encoder 冒称这两条源码的逐位复现。
3. MX32/E8M0 与 NV16/E4M3FN+FP32 global 的层级正确。E8M0 codes0…254 是 `2^(c-127)`，code255 NaN，code0非零；E4M3FN 有限正候选0…126，max448/min positive2^-9，不是 FNUZ。scale 可有限而与 payload 的乘积超 Float32 范围；OCP §5.1 对这种情况保留实现定义，非所有组合皆有有限 FP32 重建。
4. NV global `a=A/2688` 为反量化方向；先编码 block `b=Q_E4(A_g/(6a))`，再量化 payload、重建 `a*b*p`。GEMM 的 gamma 为倒数，alpha 为 `1/(gammaX*gammaW)`。独立代码只有 a 真的舍入到 FP32，其余运算是 Python double；未声称按官方 dtype 逐步模拟。
5. 零/tiny 分支明确是参考补充：全零 a1/b1，非零张量中的零块可用 b1；非零 a 舍到0、非零 block b 舍到0则拒绝。普通 packed/Triton/TRT 的保护和布局不被合并成一个行为合同。没有声称“有限输入”本身就证明所有中间尺度安全。
6. 默认 max-based scale 与 MSE 最优不同。另一块确定 a1 后，pattern 的 .625/1 scale 误差和理想 .6 对照正确；`[.4]*15+[3.6]` 的 .625 MSE879/102400，大于另一合法 .875 的199/102400，反例有效但不称后者全局最优，也不作完整 MX/NV 模型性能比较。
7. OCP §6.3 推荐 floor-power-of-two/4，不是唯一 encoder；amax7可取scale1并饱和到6。所引固定 MX 源码 ceil(amax/6) 的规则不同，本文独立参考沿此数学覆盖思路，但并不复制其 strict midpoint/source dtype 行为。
8. 行式 M/K/N 轴正确。只有每个共同 K-group 内可提出对应局部 scale；NV 两全局scale才可统一外提。两边分组不同需共同细分，不能强行共用组号。精确实数表达不保证不同 GPU 累加顺序/精度逐位一致。
9. 真正32×32转置反例成立：`[0,1]` 在先量化路径的原行无6，用scale1/16而得到.25；先转置后与6同组，用scale1而得到.5。输入矩阵本身虽对称，量化分组不对称；不能由输入对称断言两过程相等。N×K 存储最后维分组与数学 K×N 的列 K 分组一致；flat view 只验证总数可整除不能证明逐行分组。
10. format、nibble顺序、scale swizzle、shape/padding、kernel支持各是不同合同。固定 NV入口检查 exact recipe、RealQuantLinear、GPU/import、block16及 input.shape[1]%64/weight.shape[0]%32；没有扩成所有Blackwell硬件的通用限制。保存原shape不自动让reshape裁掉padding。
11. 完整块 MX4.25bit/value；NV4.5+32/T。1024元素544/580B，孤立NV16为13B/6.5bit。逐行padding的块数是 R·ceil(K/G)，2×17的NV假设布局为4块、40B；不是 flatten ceil=3，也不是参考程序已经实现了 exporter。均不等于模型进程显存。

## 三、本轮修正

- 修复实际临界点数值错误：原 `ceil(log2(amax/6))` 在 `nextafter(6*2^100,+inf)` 上会得到100，而数学上需要101；在 `6*2^127` 上侧也未正确拒绝 E8越界。参考实现改为 frexp 初选后比较精确可表示阈值 `6*2^e`，保持同一数学 ceil规则与指数下限。正文解释此选择不是官方导出器逐位模拟。
- unpack4 只接受bytes/bytearray，拒绝任意 iterable 的假字节；d4严格0…15整数、positive_e4m3严格0…126整数，均拒绝bool等非整码。防止[256]经移位得到越界高nibble而无异常。
- transpose拒绝空/不规则矩阵，避免zip静默丢维；storage_bytes明确正整数完整16/32块和非负整数字节，拒绝bool、零组、浮点等非法参数。没有把教学helper扩成通用artifact库。
- reconstruct拒绝非有限重建；mse检查等长非空有限向量，以及平方、求和/均值的溢出和非零误差意外下溢。NV重建增加有限检查。正常量化到零仍是允许的量化结果，不把所有小数误认为算术错误。
- 明确只global a转换FP32，其他Python标量为double；补充逐行ceil/padding的具体账单。不修改date、文件名、作者、metadata或其他文章。

## 四、真实执行与独立 oracle

专项：`scripts/reviews/oct15-nvfp4-examples.mjs`。直接提取正文三个唯一 marker，在独立namespace执行，而非另写同名示例后自测。

```powershell
$env:BLOG_PYTHON = 'C:\Users\iStar\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
node scripts/reviews/oct15-nvfp4-examples.mjs
node scripts/audit-posts.mjs
```

最终8组通过：

1. 三个真实正文块全部执行，原assert与打印发生；读回实际pack已知bytes与544/580B结果。
2. 用 binade/significand 构造独立有理 E2M1 码本，不复用正文解码位移；全部16码、±0、七midpoint及正负邻点、-16…16的1/64网格及极大/极小有限值。独立物理输入覆盖256byte，核low-first与已知`10 32 54 76 98 ba dc fe`，不只decode自身encoder输出。
3. 分指数区间构造127个有限正E4值，精确Fraction距离核全部126个midpoint及两侧邻点、零/normal/subnormal边界。构造全部255个有限E8scale，独立算出最大scale乘6超FP32max并真正触发FP32 pack溢出；code255/NaN是规范事实，不假装正文有一个未提供的E8 decoder。另核固定MX strict-tie/NV sign政策的源码算术差异，没有运行Torch。
4. 独立从有理数指数/FP32量化单位与相邻整数候选推导a的舍入，不把第二次struct往返当oracle。覆盖normal、subnormal、偶/奇中点、最大FP32 scale；零和半min-subnormal拒绝/上邻值接受。用十进制Fraction重算pattern及max非MSE最优反例、倒数方向、tiny/shape/overflow合同。
5. 对每个E8指数的 `6*s` 阈值及上下nextafter，搜索所有合法scale并作精确Fraction覆盖比较，不复用正文frexp或log2。包含旧log2误选的真实反例、最大指数上侧拒绝、最小scale夹限与零；另核OCP推荐与本ceil encoder在amax7不同。
6. 独立按行/块用Fraction重建所有32×32位置，核两过程62个不同项；另用3×64非方阵、两组/行检查实际mx_rows；flat总数可整除但17列不可逐行分组，以及ragged拒绝。不是仅复用同一个transpose/encoder往返。
7. 3×64×2的有理dense GEMM与四个K-group恢复逐元素一致；另用不同小组边界的common refinement验证分组含义；正文-120与错误-8也执行读回。小组代数例不是硬件native分组配置。
8. 显式枚举完整块payload与scale容器字节，核544/580/13B及假设逐行padding4块/40B；测试真实失败的类型/字节/shape/非有限/中间溢出/均值下溢合同。

初次重建溢出负例选 `maxfloat/5`，归一化5恰为tie并RNE到4，不应溢出；该测试预期错误已改为 `maxfloat/5.5`，实际选择6后才构成重建溢出。之后完整8组重跑通过。没有用宽泛吞异常或“没有抛错也算成功”的检查掩盖负例。

站点静态审核通过97篇、10series、24tags。其结论仅覆盖元数据、taxonomy、系列顺序、fences/控制字符、图片与残留标记，不替代生产HTML/MathJax/浏览器检查。

## 五、执行边界与移交

只有标准库CPU码本/精确有理数、分组矩阵、局部bytes和容量计算。未运行官方ModelOpt/Torch、FP8/FP4转换kernel、TRT/CUDA、完整checkpoint导出/重载、模型校准或任务质量、GPU指令路径/性能/显存实验；不能由这些例子声称任意后端bitwise parity或模型可靠性已认证。

正文冻结后已告root并行构建/浏览器；其后只完善专项和本记录，没有再改post。现正式移交post、专项与本记录，审核代理停止写入。root最终发布QA另补；本代理未提交或推送。

## 六、主代理最终检查

主代理阅读全文与全部专项，独立运行8组专项、全部31个JavaScript套件、既有Python套件、97篇元数据及历史coverage，全部通过。生产构建生成112个HTML页面、137个文件；90篇数学文章的1372个行内和1022个独立公式通过结构检查。浏览器本地读到66个MathJax容器、0个错误，无美元残留和整页横向溢出；scale误差表与推导实际截图可读。待单篇推送后另记匹配CI和线上证据，不将源码/CPU/构建检查扩为官方后端或GPU结果。
