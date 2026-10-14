# NVFP4 / MXFP4：微缩放格式篇研究交接

日期：2026-10-15。系列第 11 篇，新文；以 **2025 NVFP4 专题**为锚，不声称本文找到 NVFP4 历史首次发布日。MX 前置历史必须标 2023，不能因本篇顺序把 MXFP4 写成 2025 出现。

建议标题：**NVFP4 与 MXFP4：4-bit 浮点如何依靠微块缩放表达真实张量**。

边界：已有 `_posts/2026-06-02-nvfp4-kv-cacheblackwell.md` 是 NVFP4 KV 部署专题。本篇回答“标量编码、scale 编码、分组轴和 GEMM 如何组成一种格式”，不重复 NVFP4 KV 页管理、cache hit、P/D 等系统清单。

## 一、实际读到的来源

1. [OCP Microscaling Formats Specification v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)：实际读 §4 Scope、§5.1–5.2、§5.3.3 FP4、§5.4.1 E8M0、§6.1–6.3；版本表写 9/7/2023，封面 Sep2023。PDF 表格文本已核对，尝试截图被工具拒绝，**没有宣称做过视觉逐页核验**。
2. [OCP 官方公告 2023-10-17](https://www.opencompute.org/blog/amd-arm-intel-meta-microsoft-nvidia-and-qualcomm-standardize-next-generation-narrow-precision-data-formats-for-ai)：读到发布 v1.0 与 MXFP4 族的正文。可说“2023规范、10月官方公告”，不要把两个日期混成技术首次提出。
3. [NVIDIA NVFP4 介绍 2025-06-24](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)：实读正文格式表、What is NVFP4、High-precision scaling 与 Micro-block 段；不以页面 AI 摘要作事实依据，不引用其性能/精度数字来保证本文实验。
4. ModelOpt0.35.0 固定 [`c359cb774db4daab623cddde6c7e0399a49cf7d2`](https://github.com/NVIDIA/Model-Optimizer/tree/c359cb774db4daab623cddde6c7e0399a49cf7d2)，2025-09-04：完整读 [`nvfp4_tensor.py`](https://github.com/NVIDIA/Model-Optimizer/blob/c359cb774db4daab623cddde6c7e0399a49cf7d2/modelopt/torch/quantization/qtensor/nvfp4_tensor.py)、[`mxfp4_tensor.py`](https://github.com/NVIDIA/Model-Optimizer/blob/c359cb774db4daab623cddde6c7e0399a49cf7d2/modelopt/torch/quantization/qtensor/mxfp4_tensor.py)、[`backends/nvfp4_gemm.py`](https://github.com/NVIDIA/Model-Optimizer/blob/c359cb774db4daab623cddde6c7e0399a49cf7d2/modelopt/torch/quantization/backends/nvfp4_gemm.py)；读 `config.py` 的 MXFP4/NVFP4 recipe 和 `triton/fp4_kernel.py` 的 fakequant 与 dequantize。未读取所有 CUDA/CUTLASS 内核，未编译运行。

## 二、核心事实与需避免的误解

### E2M1 是 payload，不是整个 NVFP4

1sign+2exponent+1fraction，bias1；正号 magnitude codes 0..7 对应 `[0,0.5,1,1.5,2,3,4,6]`。负号重复这些幅值，保留 ±0：16种位模式、15个不同实数值。最小 normal1、唯一正 subnormal0.5、最大6；无 Inf/NaN payload 编码。不能套用 IEEE“全1指数保留”的惯例，否则会误删4和6。

最近偶数 tie 按**保留编码低位**，不是把最终十进制实数取偶数：阈值 `[0.25,0.75,1.25,1.75,2.5,3.5,5]` 对应 `[0,1,1,2,2,4,4]`。超范围转换须明确饱和等策略；NaN 输入的处理不能从“payload无NaN”自动推断。

### 两个 block 格式

| 项目 | MXFP4 (OCPv1) | NVFP4 (本文官方介绍/ModelOpt版本) |
| --- | --- | --- |
| payload | E2M1 | E2M1 |
| 每块元素 | 32 | 16 |
| block scale | E8M0，2的整数次幂 | E4M3，允许非2幂 |
| 全局 tensor scale | 该 concrete 格式不要求 FP32 global scale | 另有 FP32 tensor scale |
| 理想主体成本 | 4+8/32=4.25bit/value | 4+8/16=4.5bit/value，另加 global |

E8M0 是 unsigned exponent，code0..254 表示 2^(code-127)；code255 NaN，无零和 Inf。code0不是scale0，而是2^-127。全零块可以用任意合法正scale配全零payload表达。E4M3 scale 有更细有效位但动态范围远小于 E8M0，因此 NV 另用 tensor scale 扩展张量级范围。**NV不是“每个值变FP8”**，FP8只编码局部 scale。

更小 block 与更灵活 scale 是表示能力选择，不是所有模型/任务固定误差优势定理。仍可能有 payload rounding、block scale rounding、global scale 校准/舍入、underflow、clipping 与不良分组。

### 固定一个可核查 NV encoder 示例

沿 ModelOpt0.35 的 weight 路径，先限定全局/块尺度在实际编码后均有限且严格大于零，并满足该路径的形状约束；不能仅凭原张量“非零且有限”就保证所有中间尺度合法：

```
a = max_abs(W) / (6*448)                     # global dequant scale
b_g_ideal = max_abs(W_g) / (6*a)
b_g = Q_E4M3(b_g_ideal)                     # block scale also rounds
p_i = Q_E2M1(W_i / (a*b_g))
W_hat_i = a*b_g*p_i
```

必须明说 a 是 **dequantization scale**；某些接口把 global_scale 命名成倒数 `448*6/amax`，它们数值相反但不矛盾。源码 `nvfp4_gemm.py` 调 TRT op 用倒数，epilogue `alpha=1/(global_scale_W*global_scale_X)`。

这条 max-based encoder **不是在所有E4M3 scale中穷举最优MSE**。NVIDIA介绍中的直观“更好匹配”不能被强化成每块自动全局最优量化定理。

全零张量、全零块和 scale 下溢必须分支处理。教学实现建议显式全零张量用 a=1,b=1,p=0；非零小scale被量化为0时需声明fallback/最小scale策略，不能直接除0。官方不同路径的 zero/tiny guard不完全相同，本文不能用简化代码冒称逐位复现全部 backend。

### MX 规范不等于唯一 encoder

OCP §4 不规定详细 scale 算法，§6.3 给应支持的转换语义，并允许其他算法。其示例取 `floor_pow2(amax)/largest_pow2_element`，E2M1 后者是4；它可能把接近下一幂的最大值裁到6。ModelOpt0.35的 MXFP4QTensor 使用 `2^ceil(log2(amax/6))`，偏向避免 range overflow。两者不同，所以不要写“MX scale 必须/总是 ceil(amax/6)”或“MX永不clipping”。

ModelOpt该版本 MX 的纯 Python packed cast 在 midpoint 用 strict `>0` 计数、选择下侧，而 NV cast/Triton fakequant显式处理 nearest-even；不要把所有实现路径说成 bitwise一致，教学实现应按声明的RNE独立实现。NV源注释 odd_bounds中的文字有笔误，实际indices1/3/5是0.75、1.75、3.5，按实际数组与格式核查，不照抄2.5。

## 三、矩阵语义与物理布局

行式 `Y=XW`，若 X 在每行K维分16/32组，W在每列K维分组，同一 reduction group 内 scale可提出该组：

`Y_hat[m,n] = aX*aW * sum_g bX[m,g]*bW[g,n] * sum_{k in g} pX[m,k]*pW[k,n]`。

但不同g的scale不同，不能将所有scale合并成一个输出外积再对整个K求一次dot。若两边分组边界/轴不同，应取共同细分区域，不能强套上式。

OCP §5.1 **不规定内存物理布局**，§6.1 内部累加精度/操作顺序实现定义。NV encoder 的相邻偶数index放low nibble，奇数index放high nibble，是该reference实现的选择；TRT生成的block-scale buffer还需专用布局转换。不能称任意相同E2M1字节都能交给任意FP4kernel。

转置可能改变分组成员，`quantize(W)^T` 通常不等于 `quantize(W^T)`；不能只对payload转置而不处理scale轴。FP4 Tensor Core 的存在不证明某个 PyTorch tensor 运算已走FP4 kernel；weights/activations/KV/multimodal层各自支持不同。

ModelOpt0.35 NV入口检查imports、GPU、RealQuantLinear、exactrecipe、block16，且该版本代码有input.shape[1]%64和weight.shape[0]%32限制。这是**入口检查的源码事实**，不要扩成所有Blackwell/所有矩阵的通用硬件约束。本文不需要放“当前全架构支持表”。

## 四、CPU可执行例子（已作独立算术检查）

Node标准算术、枚举E2M1码本并以编码低位处理ties；无Torch/GPU执行：

令global dequant scale a=1（可由同张量另一个amax2688的块确定），示例局部输入 `[0,.3,.6,.9,1.2,1.8,2.4,3.6]` 重复两次组成16元素。理想局部scale0.6：

| block scale | 重构8元素（另8重复） | MSE |
| --- | --- | ---: |
| 理想FP32 0.6 | 原值，浮点尾差约1e-16 | 3.24e-32 |
| E4M3 0.625 | [0,.3125,.625,.9375,1.25,1.875,2.5,3.75] | 0.0053515625 |
| E8M0 1 | [0,.5,.5,1,1,2,2,4] | 0.0575 |

这是固定尺度候选下的教学比较，不是实际完整MX32 vs NV16公平模型性能实验。它说明 scale也会有误差，不建立NV对每个张量必胜的定理。

容量：NV含单个FP32global scale、T元素、完整block时为 `4.5+32/T` bits/value；只有16元素的孤立tensor是6.5bit/value，不是4.5。MX是4.25bit/value。padding、scale swizzle、headers、未量化层、workspace另算。

审核建议：枚举16code解码、7个midpoint、±0、NaN/Inf输入策略、全零块/张量、极小scale、odd/tail padding、low/highnibble往返、scale倒数不混、转置分组反例、groupdot公式。教学软件编码不宣称产出可部署checkpoint。

## 五、本篇的详细结构

1. 从INT4/NF4/FP4都4bit但码本不同进入，链接NF4篇；说明本篇补全“格式是什么”。
2. 亲手解出E2M1的16模式，解释非均匀间隔与±0。
3. 为什么[-6,6]可表示真实张量：引入blockscale，而不是误说E2M1自身巨大范围。
4. 2023 MX规范：E8M0、32元素、4.25bit；日期与“机制+类型”层次。
5. 2025 NV专题：16元素、E4M3与FP32global，完整前向Q/DQ公式。
6. 运行上述局部scale例子，包含理想scale与实际scale误差；不只展示收益。
7. 两级scale的方向、下溢与零块策略，说明校准不等同格式。
8. block沿哪个轴：groupdot、转置、payload/scale-layout必须一起移动。
9. 从referencepack到真实kernel的边界；与前篇INT4storage→INT8compute对照。
10. 算清effectivebits，不用参数数*0.5bytes猜进程显存。
11. 本篇只提供格式/CPU检查，真实质量与GPU验收交终篇；KV部署链接旧文，不扩写第13篇。

尚未做：运行官方ModelOpt、Torch FP4/GPU内核、模型校准、任务精度、吞吐/显存测试。不得宣传已部署、固定加速或任意任务质量等价。

## 六、独立源码边界补充：负零、全零张量与 padding（已冻结）

补核日期：2026-10-15。仍固定 ModelOpt0.35.0 commit `c359cb774db4daab623cddde6c7e0399a49cf7d2`，不据此推断后续版本行为。本次重读 `nvfp4_tensor.py` 全文件；另外实际读取 [`utils.py` 的 reduce_block_amax / reduce_block_padding / reduce_amax](https://github.com/NVIDIA/Model-Optimizer/blob/c359cb774db4daab623cddde6c7e0399a49cf7d2/modelopt/torch/quantization/utils.py#L44) 和 [`BaseQuantizedTensor.__init__`](https://github.com/NVIDIA/Model-Optimizer/blob/c359cb774db4daab623cddde6c7e0399a49cf7d2/modelopt/torch/quantization/qtensor/base_qtensor.py#L54)。下述为源码与元素数推导，不是运行Torch后的报错或数值实测。

### 1. 格式具有双零，不等于该 encoder 保留输入负零

[`_cast_fp4` L122–138](https://github.com/NVIDIA/Model-Optimizer/blob/c359cb774db4daab623cddde6c7e0399a49cf7d2/modelopt/torch/quantization/qtensor/nvfp4_tensor.py#L122) 以 `(weight < 0)` 提取sign，而非读取signbit。IEEE负零比较 `-0 < 0` 为false，因此输入负零不能据此生成负零code8；不能将教学实现保留±0称为此encoder的逐位复现。

这不意味着该encoder完全不能产生code8：负的非零小数若幅值量化为0，仍可组合sign1与magnitude0。另一个独立边界是其 `e2m1_values` 解码表在index0/8都写普通0；本文的格式枚举与保留signbit的教学decoder，也不应被说成这份lookup decoder的逐位行为。

写作措辞：**E2M1规范有两个零编码；具体软件转换是否保留负零，要逐路径核查。** 有限值重构相同不代表zero sign相同。

### 2. 全零tensor的默认scale路径未由 `==0` guard兜住

在不传自定义positive全局scale、走默认非TRT路径时，源码顺序为：

1. `get_weights_scaling_factor_2` L93–95：`a=amax/(6*448)`；全零张量给a0。
2. `get_weights_scaling_factor` L81–86：blockamax0，计算 `b=0/(6*0)`，这一步是NaN而非0。
3. 随后的 `per_block_scale[per_block_scale == 0] = 1` 只匹配0，不匹配NaN。

因此该处代码不能支持“全零张量会自动取scale1并安全输出零”的表述。先前建议的教学全零分支 a1/b1/p0，是**显式补充的参考约定**，不是此源码默认已实现的分支；调用者预先提供合法scale或其他路径是否另有保护，要另行核查。

应区分：非零tensor中单个全零块，在全局a>0时原始b0可以被此guard替换；全零tensor的0/0则不能。对极小非零a、块scale转换后下溢到0的情况，本节不声称已执行后续backend行为。

### 3. 有 padding utility 不等于任意shape可 roundtrip

[`reduce_block_padding` L93–126](https://github.com/NVIDIA/Model-Optimizer/blob/c359cb774db4daab623cddde6c7e0399a49cf7d2/modelopt/torch/quantization/utils.py#L93) 逐指定维度补右侧零：padding量为 `(block-size%block)%block`；本次NV调用只指定最后一维。它返回补齐tensor，**不返回裁剪描述或自动还原视图**。`reduce_block_amax`本身L70还要求所归约维可整除block，故不能假设直接调用它也自动padding。

NV `quantize`在L163保存padding前 `input_shape`，L167补齐最后维；L210之后按补齐shape编码，L227把padding前shape传给 `BaseQuantizedTensor`。后者的构造器仅把原shape/dtype存metadata，原样保存packed数据，不裁剪payload。

slow `dequantize` L237–249按packed最后维乘2展开，L292–295乘blockscale后直接 `.reshape(self.metadata["shape"])`，这里没有沿每行最后维裁掉padding。具体元素数反例：输入shape(2,17)、block16→pad成(2,32)，packed每行16bytes，unpack共有64元素；原shape需要34元素，不能靠reshape丢掉多出的30个零。

所以只能说**该quantize函数包含最后维补齐步骤**，不能说“源码已证明任意shape的完整Q/DQ roundtrip都受支持”。教学实现若支持tail，须明确保存logical shape、padded shape，并在按padded layout解码之后逐轴裁剪；这属于教学实现的补充处理。更保守的源码对照例使用最后维可整除16、无需padding的二维输入。不要由这条条件再推断真实GEMM必支持——前述backend还有更强的shape/recipe检查。

### 4. 本次执行边界

仅执行Node标量/元素数断言：`Number(-0<0)=0`；`a=0/(6*448)=0`、`b=0/(6*a)`为NaN且`b===0`为false；(2,17)→(2,32)元素数34→64。没有导入/运行Torch、ModelOpt、TRT或GPU，也没有实测searchsorted/float8转换/真实decoder。上述三项源码限制已纳入写作边界；此研究文件本轮补充后冻结，交root协调，不自行提交。

## 七、审核代理写作前独立预读范围与补注

2026-10-15：旋转篇静止稿到达，NV 预读在此停止，本节不是未完成正文的审核记录。已读 OCP v1.0 PDF 的 §4、§5.1–5.2、§5.3.3、§5.4.1、§6.1–6.3 及相关格式表的可提取文本；截图请求仅返回引用，未取得可检视图像，不宣称完成视觉表格核验。另读 NVIDIA 2025 NVFP4 介绍的格式和缩放正文；完整读同一 ModelOpt 固定版 `nvfp4_tensor.py`、`mxfp4_tensor.py`、`nvfp4_gemm.py`、`triton/fp4_kernel.py`，定点读 `utils.py` 的 block reduction/padding、`BaseQuantizedTensor.__init__` 与 `config.py` 的具体 NVFP4/MXFP4 默认 recipe。未读全部 CUDA/CUTLASS 或执行 Torch/GPU。

- OCP §5.1 还限定：若 block scale 与 payload 的乘积超出 Float32 最大有限值，表示行为是 implementation-defined。E8M0 最大 scale `2^127` 本身有限，不表示 `6*2^127` 在 Float32 中也有限；不能把全部 payload/scale 组合称为无溢出的 FP32 重建。
- MX 此版本的 raw tensor 方法 `view(-1, block_size)` 按展平顺序分组，不自动验证每行最后维整除；若要解释成每行 K-group，参考输入需单独限定最后维可整除 32。其 sign 表达式 `(2-torch.sign(x))//2` 对输入零得到 sign1，不等于 NV 的 `x<0` 提取符号。odd-last-dimension helper 的注释提到 padding，但实际只 clone 较短的 odd slice 后相加，不是完整的 padding/crop 合同；不能由注释宣称任意 odd/tail 受支持。这些是源码阅读边界，不是运行后的报错复现。
- NV Triton fakequant 还有 effective block factor `<1e-5` 时替换为1的分支，不同于 non-TRT packed weight 路径仅在转换前 block scale `==0` 时改1。已核对二者 RNE 阈值，并不因此宣称两路径在零/tiny/所有形状上逐位等价。
- 用 Python Fraction 在候选 E2M1 重建值间比较距离，独立构造 max-based NV scale 非最优 MSE 例：全局 a=1（由另一块 amax2688 决定），本块 `[0.4]*15+[3.6]`。ideal b=.6 经 E4M3 编为.625，MSE=`879/102400≈.008583984375`；另一合法 b=.875 给 `199/102400≈.001943359375`。它反驳“最接近 max-based scale 就必然最小化块 MSE”，不反驳该默认编码器的具体规则，也不是 NV/MX 模型质量对比。

后续正式独审计划用 binade 公式与独立码表核16个 E2M1 编码/双零，Fraction 距离 oracle 检查7个midpoint及邻点，穷举 E8M0/有限正 E4M3 scale，另按明确合同检查全零、tiny、逻辑shape/paddedshape、nibble顺序、转置分组与group-dot。上述仅是待最终正文移交后的测试设计，尚未建立本篇专项或宣布通过。

## 八、Root 对 OCP 原 PDF 的实际视觉复核

Root 从上述 OCP 官方 URL 获取同一 v1.0 PDF，SHA256 为 `d195d6a36dd4a0c89064af0c479bcaad5c0fe29d63f628502ea6d7c4b4279421`，共 16 页。用本机 bundled Poppler 将第 9–15 页渲染为 PNG，实际查看完整第 9、10、12、13、14、15 页，而非仅取得截图引用。表 1 的 MXFP4/E2M1/32/E8M0/8 bits，表 5 的双零、normal/subnormal 与无 NaN/Inf，以及表 7 的指数 bias/range/NaN/no-zero 均与本研究一致；同时视觉读取 §5.1 的物理布局/Float32 越界边界，以及 §6.1–6.3 的累加顺序和 encoder 可选规则。未将规范推荐转换流程强化为唯一 encoder。

这是 root 自己的额外视觉证据，不把此前研究者/审核者失败的截图请求改写为成功。未修改 PDF，也未运行真实 MX/NV 后端。临时来源与渲染图放在本轮专用、忽略的构建临时目录，完结时清理，不复制进博客资源树。
