# W4A8 / QServe：旧文逐项研究交接

日期：2026-10-15；系列第 10 篇，原 URL 不变：`_posts/2026-08-04-w4a8-progressive-quantization-kernel.md`。已读取全文（末尾另行补读，避免输出截断）。标题已有明确问题，可以保留。此次不把原文简单迁移元数据，应将泛化部署清单收束为 QServe 两级数值表示与真实内核契约。

## 实际读到的来源

1. [QServe v3](https://arxiv.org/html/2405.04532v3)，首次公开锚点 2024-05-07：实读 §2.2、§3.1–3.2、§4.1–4.3、§5.1–5.3。方法/系统段足够支撑本文；没有复现实验，不移植吞吐倍数。
2. 官方库旧名 `mit-han-lab/qserve` 现重定向 `mit-han-lab/omniserve`，固定 [`02b2925aa6fa3b92b06316a1524b7f38922cd9c8`](https://github.com/mit-han-lab/omniserve/tree/02b2925aa6fa3b92b06316a1524b7f38922cd9c8)，2025-02-21：完整读取 [`w4a8_linear.py`](https://github.com/mit-han-lab/omniserve/blob/02b2925aa6fa3b92b06316a1524b7f38922cd9c8/omniserve/modeling/layers/quantized_linear/w4a8_linear.py)，以及 checkpoint converter 的 `quant_utils.py`。后者只是 generic pseudo quantizer，**不能把它误作完整 QoQ 参数搜索器**。
3. 逐段读官方 [`per_group/gemm_cuda.cu`](https://github.com/mit-han-lab/omniserve/blob/02b2925aa6fa3b92b06316a1524b7f38922cd9c8/kernels/csrc/qgemm/w4a8_per_group/gemm_cuda.cu) 的 MMA 指令、`share_to_reg_one_stage_B` 解包/乘法/`__vadd4`、epilogue 和 dispatch；同样读 [`per_chn/gemm_cuda.cu`](https://github.com/mit-han-lab/omniserve/blob/02b2925aa6fa3b92b06316a1524b7f38922cd9c8/kernels/csrc/qgemm/w4a8_per_chn/gemm_cuda.cu) 的对应段。没有声称逐行审完共享内存 pipeline/launch 宏，更未编译运行。
4. [TensorRT-LLM 1.1.0 Quantization](https://nvidia.github.io/TensorRT-LLM/1.1.0/features/quantization.html)：已读 recipe 列表、模型与硬件矩阵；页面注明生成 commit `48b7b5d`、更新 2025-12-12。**版本号页面不应表述成“今天所有最新 backend”的普遍支持**。
5. ModelOpt 0.35.0 固定 [`c359cb774db4daab623cddde6c7e0399a49cf7d2`](https://github.com/NVIDIA/Model-Optimizer/blob/c359cb774db4daab623cddde6c7e0399a49cf7d2/modelopt/torch/quantization/config.py)，2025-09-04：实读 `W4A8_AWQ_BETA_CFG` 配置及邻近注释；weight 是 INT4 block128 后接 FP8，activation 是 FP8 `(4,3)`，不是整数 QServe 的同一条执行路线。没有读取整个 ModelOpt 算法或执行导出。

## 结论与具体修改点

### 1. 原有两级公式基本正确，不能照搬论文等号和布局笔误

统一 `Y_(M,N)=X_(M,K)W_(K,N)`：

`W_hat[k,n] = s0[n] * s1[g(k),n] * (u4[k,n]-z[g(k),n])`。

第一阶段目标 INT8 code 只能近似由第二阶段重构，论文 Eq5 的等号不意味着对初次 INT8 codes 无损；原文已经用近似号，保留。`s0` 是 FP16 per-output-channel，`s1` 是整数 per-K-group/output；不能把任何一个随 K 变化的 s1 全部提到单个 GEMM 外面。

### 2. 必须加入源码中实际存的是 scaled zero，而非 raw UINT4 z

`w4a8_linear.py`：per-group `s2_scales` 和 `s2_zeros` 都用 1 byte 容器，后者保存 **-z*s1 的二补码字节**；weight nibble 另有 32×32 tile 的 reshape/permute。数学 raw zero 只有 4 bit 不代表 artifact 中也只占 4 bit。

CUDA 实际执行：先 `u4*s1`，再按 byte lane 加 `(-z*s1)`，最后 MMA 指令为 `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32`。因此 storage UINT4、compute INT8、accumulator INT32、epilogue FP16 四层应明确。

注意：乘法中间的 unsigned byte 可大于 127。例如 15*16=240 仍可保存在单个 byte 中；“每一步中间数都必须小于 signed INT8 的 127”不准确。需要避免的是 packed 32-bit 乘法跨 byte carry，以及最后 signed operand 越界，不能把论文文字的宽泛“INT8 range”照抄成更强约束。

### 3. protected [-119,119] 有原文依据，但论证应自己闭合

原文范围可保留。论文 §4.1 中从 17/2 到 119.5 的展示不宜原样复制（127-8.5 实为 118.5）。更稳妥的教学证明：若第一阶段 code 范围在 [-119,119]，组跨度最多 238，采用本例 nearest-even 整数量化组 scale 时 `s1=max(1,RNE((max-min)/15))<=16`。未饱和的再量化误差至多 s1/2<=8，重构上界 <=127、下界 >=-127；unsigned code clamp 的重构区间包含零，不会把值向更远的同号极值推。**这只证明所声明的简化规则，不能宣称覆盖所有 exporter**。

旧文 [-113,120] 例目前明确先 RNE(120/16) 再加 z7，结果15、重构128，是成立的。若写 RNE(120/16+7)，nearest-even 会得到14、重构112。必须保留 rounding 顺序提醒；不要把论文不明确的 nearest-round 符号当作 IEEE RNE 全等代码。

### 4. per-channel correction 的代数恒等与官方近似不要混淆

泛化公式 `sum x(q-z)=sum xq-z sum x` 是正确恒等。QServe §5.2.2 在后半段额外用未量化 X 的 row sum 替代 X_hat 的 row sum；这一步不是代数恒等。若详细解释官方 per-channel kernel，应标出这个选择，不能将 QServe Eq12→13 整条称为精确变换。

per-group zero/s1 随 K 改变，无法只做一次全 K 的 epilogue correction。实际 per-group kernel 在主循环按组恢复 INT8，只有 `s0[n]*sX[m]` 延迟到 epilogue。

### 5. 补一段真正解释 SmoothAttention，而非只提名字

`Q'=QΛ, K'=KΛ^-1` 保持 scores；QServe attention 中 query 不量化，并不表示全部 projection 输入都不量化。为把 scaling 融合进 RoPE 前的 Q/K weights，RoPE 每一二维旋转对必须共享 λ；论文 Llama 的 split-half 配对是 i 与 i+D/2，其他实现布局可能是相邻配对，应以实际 RoPE 定义为准。任意独立 per-channel λ 通常不与 RoPE 交换。这个点与前篇“Hadamard 必须放 RoPE 后”的对比很有教学价值。

### 6. 元数据不能用通用假设代替本 kernel

原文“假设 FP16 scale+4bit zero”的 2.5/G bytes 不是 QServe 此版本 per-group artifact。忽略 padding/bias，实际核心为 `0.5 + 2/G + 2/K` bytes/weight：一 byte s1、一 byte scaled-zero 每组，FP16 s0 每输出通道。示例 G128,K4096 得 0.51611328125 bytes/weight。先列通用公式再算本实现，比只写假想成本清楚。

### 7. 版本支持不能由 Python 构造器泛化

该 C++ per-group dispatch 明写 `constexpr int G=128`。即使 Python 类有 group_size 参数，也不能因此宣称所有 group size 都有真实内核。pack 路径还隐含 K/N 的 tile 整除要求；shape tail 能否正确必须测试，不能把构造器前几个 assert 当作完整支持矩阵。

### 8. 建议削减/合并的内容

原文后半重复多套“部署路径/五道边界/profiler/九点总结”，可压成一节“把这份 artifact 交给 kernel 前验证什么”，把全栈质量和 SLO 内容交第12篇。保留 Roofline 的定性 M 讨论，不传播特定 GPU 交叉 batch 或倍速。W4A8 FP8 的区别保留一个有版本来源的小框即可，不另写泛化浮点 W4A8 教材。

## 针对性提纲

1. 同一层的四种 dtype，引出 W4A8 名字没有说明算什么。
2. M/K/N 与带宽/计算动机，不先大篇部署清单。
3. 朴素 per-group 反量化为什么卡在 K-loop。
4. progressive 两级表示，明确每个参数的轴。
5. 一组实际整数算例：初次 INT8→UINT4→恢复的 signed INT8，并计算输出。
6. protected range 与 rounding 反例；范围证明独立于性能承诺。
7. compute-aware packing：nibble、32×32 tile、scaled zero；教育 pack 不冒充官方 layout。
8. 从代码读出 MMA 和 epilogue；per-channel 与 per-group correction 对比。
9. A8 与 KV4 各走什么路径，简明推导 SmoothAttention/RoPE 对条件。
10. ModelOpt0.35 FP8 W4A8 的名称碰撞，回到 artifact contract。
11. CPU reference → pack decode → 可选目标 GPU shape 验证的交接清单；引用终篇。

## 已执行检查与待审查

Node 标准库枚举所有整数端点 -119<=min<=max<=119、组内所有整数 q，采用 `s=max(1,RNE((max-min)/15))`，z clamp0..15，u clamp0..15，重构 `(u-z)*s`：最大 s16，观测重构范围 [-121,121]，越出 signed INT8 的次数0。这个有限枚举不等于实测全部 QoQ 参数搜索或 CUDA 内核。

已执行 RNE 次序对比：RNE(7.5)+7=15，而 RNE(14.5)=14。审核应在 Python 独立复跑，加入 scaled-zero bytes、packing roundtrip 和 per-group epilogue 不能全局提 scale 的反例。

本次未运行 GPU、官方 exporter、实际 QServe 模型、质量评测或服务 benchmark。文章不得宣称实测吞吐、显存/质量提升、所有形状支持，亦不得拿 CPU 教学 pack 当作 kernel-compatible checkpoint。

## 独立来源预读补注（审核代理，2026-10-15）

本节只是为后续静止稿审核做准备，不是对未完成校订稿的审核通过。本轮重新完整读取当前旧文及本交接，实际阅读 QServe v3 §2.2、§3.1–3.2、§4.1–4.3、§5.1–5.3；另完整读固定版 `w4a8_linear.py` 与 converter `quant_utils.py`，定点读两种 CUDA GEMM 的 MMA、解包、epilogue、dispatch。GitHub commit API 独立核实 OmniServe 提交为 2025-02-21、ModelOpt 提交为 2025-09-04；实读 ModelOpt 的具体 W4A8 beta 配置和 TensorRT-LLM 1.1.0 的 recipe/模型/硬件表，未把版本页面当成当前所有后端支持。

### 保护范围证明的严谨写法

对本交接声明的简化整数合同，第一阶段 q∈[-119,119]，`s=max(1,RNE((hi-lo)/15))`，故 1≤s≤16。先令 `a=RNE(q/s)*s`，有 `|a-q|≤s/2≤8`，所以 a∈[-127,127]。实际 `u=clamp(RNE(q/s)+z,0,15)`，0≤z≤15；重建 `(u-z)*s` 等于把 a 投影到区间 `[-zs,(15-z)s]`。该区间包含 0，投影只能把越界的 a 向 0 推，不会把它推到更远的同号极值，因此保护范围仍成立。这比只列“不饱和时误差≤半步”完整，也不依赖把 zero 移进 RNE。该证明仍只覆盖所声明规则，不覆盖未读取的全部参数搜索器。

审核代理已用整数 `divmod` 实现独立 ties-to-even oracle，遍历全部整数端点及组内整数值，共 2,303,960 组端点/值情况；重构范围为 [-121,121]，没有越出 signed INT8。也逐值核对 lane 模 256 加法后有符号解释等于数学重建。它是 CPU 有限枚举，不是模型实验或真实 CUDA kernel 运行。未保护示例的两个次序实际得到 15 与 14，需保留顺序而非照抄论文原句。

### byte lane 与实际参数恢复接口

- [CUDA 12.8 SIMD 文档 `__vadd4`](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SIMD.html) 明确逐 byte 相加、忽略各 byte 溢出，不是饱和加法，也不是让进位传向相邻 lane 的普通 32-bit 加法。四个 UINT4 展开到 byte 后，每个乘 s≤16 的结果≤240<256，所以整 word 乘相同标量没有跨 byte carry；随后加入 `(-z*s) mod 256`，只有最终值在 signed INT8 合法域时才能正确作为 s8 MMA operand。`u=15,s=16,z=8` 的局部算术是 240 加 byte128，再模256得到112；这个 lane 示例不声称这些极值一定在简化 exporter 的同一实际 group 中同时出现。
- 两个 kernel 均实际使用 `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32`。group 路径读取一 byte scale 和一 byte scaled-zero，最终 FP16 s0 和 per-token activation scale 在 epilogue 应用；channel 路径 main loop 只展开正的 0…15 nibble，scaled zero 与 input sum 在 epilogue 修正。Python 的 `s1_scales` 对应本文数学 s0，`s2_scales` 对应数学 s1，不能因命名相近而错一层。
- 另读 [`checkpoint_converter.py`](https://github.com/mit-han-lab/omniserve/blob/02b2925aa6fa3b92b06316a1524b7f38922cd9c8/scripts/ckpt_converter/checkpoint_converter.py) 的输入及转换调用：它读取 LMQuant 预先导出的 fake-quant `model.pt` 和 `scale.pt`，再传入 `from_linear`。该方法不是从任意原始 FP weight 自动搜索并实现本文全部两级量化的入口；其 group 第二步是用既有 scales 恢复 codes、随后重排/整型转换，没有再次完整执行教学 RNE+clamp。不可把教学参考称为此 converter 在任意原始 W 上逐位等价的实现。

### 额外近似、分组与位置编码的独立算术

- 精确 per-channel correction 用 `sum(Xhat)`；QServe §5.2.2 额外替换为 `sum(X)`。简例 `X=[1/10,1/5]`、`Xhat=[1/8,1/4]`、u=[2,3]、z=1、sW=1/2：精确量化输入输出为 5/16，原始输入行和修正得到 7/20，差 3/80。另读固定 [`layernorm_kernels.cu`](https://github.com/mit-han-lab/omniserve/blob/02b2925aa6fa3b92b06316a1524b7f38922cd9c8/kernels/csrc/layernorm_kernels.cu) 的 `generalLayerNorm_fuse_sum`：sum 在 INT8 转换之前的归一化输出上累计，并非 INT8 code 的行和；这仍不是浮点逐位精确原始行和保证。
- 不同 K-group 不能统一提 scale：X=[1,2]、(u-z)=[1,1]、s=[2,3] 的结果为8，错误地提第一组 scale 得6。该例只是解释 reduction 轴，不能用来宣称 G2 为固定 dispatch 支持。
- 精确四分之一周 RoPE 对反例：q=[1,0]、k=[0,1]，两位置分别不旋转/旋转90度，原 score=-1；预先以 Λ=diag(2,3) 对 Q/K 互逆缩放会得 -2/3，配对等值 diag(2,2) 才仍为 -1。这说明融合前的 pair 条件，而非否定 post-RoPE 后 QΛ/KΛ^-1 的一般恒等式。所有 λ 还须为正且非零；全零统计的 epsilon/特殊分支要在教学合同中声明，不假设 λ=0 可以求逆。

尚未运行官方 converter、LMQuant、Torch/CUDA kernels 或模型；这些预读与构造不替代之后从最终正文提取代码的专项审核。
