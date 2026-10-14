# AWQ 独立审核记录

审核日期：2026-10-15。正文：`_posts/2026-08-03-awq-activation-aware-weight-quantization.md`。

writer 静止后独立全文审核概念、公式、维度、搜索代理、数值算例、实际代码、格式与性能边界。本轮保留原 date `2026-08-03 09:00:00 +0800`、作者、文件名/URL 和系列 metadata；只修改本篇及配套专项/审核记录。

## 实际独立阅读的一手来源

- [AWQ arXiv v5](https://arxiv.org/html/2306.00978v5)：实际阅读 §3.1/表 1 的 INT3-g128 显著通道实验、§3.2/表 2 的误差与组内 step 分析、输出目标和候选族；读 §4 的 TinyChat、动态反量化、平台 packing 和系统边界，核对相关实验设置。没有复现模型 PPL、任务质量或速度。v5 发布于 2024-07-18，与 2023 首发年份分开。
- [固定 auto_scale.py](https://github.com/mit-han-lab/llm-awq/blob/d6e797a42b9ef7778de8ee2352116e0f48a78d61/awq/quantize/auto_scale.py)：独立读 meanabs、20 比例搜索、pre-clamp/common normalization、模块前向输出 MSE、状态恢复、norm/linear/nonlinear 参数改写以及相关模型分支。该版本搜索只用 act meanabs 的幂，不擅自换成其他实现的 activation/weight 双统计公式。
- [固定 auto_clip.py](https://github.com/mit-han-lab/llm-awq/blob/d6e797a42b9ef7778de8ee2352116e0f48a78d61/awq/quantize/auto_clip.py)：独立读完整分组部分点积、采样 token MSE、10 个候选、模块名字排除和 apply_clip。它没有直接比较所有 groups 相加后的整层 MSE。
- [固定 quantizer.py](https://github.com/mit-han-lab/llm-awq/blob/d6e797a42b9ef7778de8ee2352116e0f48a78d61/awq/quantize/quantizer.py)：独立核对 min/max、epsilon、先 round 后加整数 zero point、浮点重建，以及 real 入口要求 zero_point=True。zero_point=False 的断言引用本分支未赋值 min_val，正常断言开启时不能当可直接运行路径。
- [固定 qmodule.py](https://github.com/mit-han-lab/llm-awq/blob/d6e797a42b9ef7778de8ee2352116e0f48a78d61/awq/quantize/qmodule.py)：额外独立阅读 ScaledActivation、完整 pack_intweight、WQLinear 构造/from_linear/forward。明确检查 packed qweight 的 int16、输入重排/输出 interleave、metadata padding、scales dtype、scaled_zeros 的浮点偏移和 GEMV/GEMM 选择，不沿用通用 int32/nibble 预设。
- 固定 commit `d6e797a42b9ef7778de8ee2352116e0f48a78d61` 的 [GitHub API](https://api.github.com/repos/mit-han-lab/llm-awq/commits/d6e797a42b9ef7778de8ee2352116e0f48a78d61) 已独立核到 2025-07-17；它不是 2023 初版全部实现的同义词。

未使用未经固定验证的 vLLM stable/TensorRT latest 页面作当前兼容性承诺；读源码不等于运行源码或 kernel。

## 审核结论

1. `E_W=W-W_hat` 对应 `Y-Y_hat=XE_W`，Frobenius SSE 与 `tr(E_WᵀXᵀXE_W)` 一致；meanabs 仅是候选生成代理，不能表达全部通道相关性/误差抵消。
2. X[M,K]、W[K,N] 下 S 沿 K，`(XS^-1)(SW)` 在实数中等价；量化后保护界依赖 `Delta_group'/s_j`，不能无条件说放大某通道就必然降低其实际误差。教学 `[-7,7]` 只含 15 值，明确不是官方全部 16 个 code 或 artifact 声明。
3. 独立高精度计算确认 20 个比例 0–.95 中本例 .85 最好，但不包含相对比例 10 的手选精确解，不是所有正 scale 的 global optimum。clamp 在归一化前，最终 S 不保证 >=1e-4。
4. 不做反向训练/GPTQ 式连续补偿，不等于不用 reconstruction/output error 作前向评价。该实现比较原模块与候选的输出 MSE，再恢复状态；Attention/MLP 也可能是检查对象。
5. clipping 默认 10 个阈值比例 1–.55，不含 .5；按名字跳过 q_/k_/query/key/Wqkv。逐组目标忽略组间交叉项，构造中局部误差改善但整层变差的推导和符号已核对；它不是固定官方非对称路径的模型实验。
6. LayerNorm gamma/beta 和共享 consumers、前序可配对 Linear 的输出/bias、非线性后的显式除法必须按图结构区分。`ScaledActivation` 是 `act(x)/S`，不是 `act(x/S)`；代数等价不保证所有位置都零额外成本。
7. 固定 pseudo 的 unsigned q∈[0,15]、整数 zero point、minmax step/epsilon 和 round 顺序正确。它返回浮点 Q/DQ；real 入口才换 WQLinear，WQLinear 只支持 4-bit。源码的对称分支问题未被本篇教学函数掩盖或误称已修复。
8. qweight 确为 int16，不是简单“8 个 code 顺排 int32”。scales 来自传入 scale/weight dtype；scaled_zeros 保存 `-(Delta*z)`，经相应 dtype 转换，而不是原整数 zero tensor。本文 4.25 bit/weight 仅是 G128、scale/offset 均 FP16、忽略 padding 的条件账单。
9. W4A16 不是自动使用原生 INT4 MMA，不压缩 KV；TinyChat 历史系统结果不能直接推成任意 runtime/shape 的加速或生成质量保证。

## 本轮代码实修与数值范围

- 原 `scaled_forward` 会接受所有原始有限输入，却在变换中静默丢失可恢复结果。例如 `x=1e-300,w=1e100,s=1e100`，`x/s` 先下溢为零，但原乘积 1e-200 可表示；反方向也可能把 x/s 溢出为 Inf。现在对变换、关键乘积的非有限值/非零下溢为零、重建和输出的非有限值作显式拒绝。
- clipping 原来把非零 peak×factor 下溢为零当成全零组。现在先区分原 peak=0 的合法分支，再拒绝不可表示的 threshold，并检查重建有限性。
- 正文旁已说明这是受限 Python 浮点参考，即使另一运算顺序可能挽救结果也不处理，不是通用量化库或 FP16/BF16 逐 bit 模拟。guard 不消除普通浮点舍入、病态相消或未归零的精度损失；正常三组算例和搜索结论没有改变。

## 已执行的专项验证

`node scripts/reviews/oct15-awq-examples.mjs` 通过 `BLOG_PYTHON` 调用本机 Python，直接执行最终正文三块代码。7 个不同语义组，不将候选、坐标或格式元素计成独立组数。

| 验证组 | 独立 oracle/边界 |
| --- | --- |
| 三块真实正文 | 唯一 marker 提取、独立执行、自身断言和真实输出；没有遗漏的 Python 块。 |
| 教学码域和范围 | Fraction 枚举重建 level、距离/偶数优先级，核 signed ties、前后浮点、不同二进制 step、全零；实际调用函数核非法输入/shape、step 下溢、变换与乘积溢出/下溢、输出累加溢出。 |
| 矩阵与组内耦合 | 独立有理数 rectangular GEMM 核 K 轴、完整输出误差/trace 二次型、abs 统计相同但相关性不同；核手工 scale1/10/100、公共缩放不改善理想相对分辨率，以及 `Delta'/2s` 上界而非逐元素单调性。 |
| 20 候选前向搜索 | 60 位 Decimal 用未归一化的相对 scale 族和独立 round-half-even 计算，逐候选对照实际 normalized binary64 结果；唯一最佳比例 .85，误差仍高于手选精确解。核教学函数不累积改写输入/权重，以及最终 factor 可小于 pre-clamp。 |
| clipping 目标 | 有理数独立枚举两组各 10 候选，再枚举全部 100 联合组合；得到局部选择 .95/.8、每组误差 -2/175，而 baseline +1/70 与 -1/70 抵消。核局部 MSE、整层 MSE 和交叉项；覆盖常量零与 threshold 下溢拒绝。 |
| 非线性折叠 | 用 erfc 形式 GELU 与独立重建 GEMM 检查正文 erf/linear；多行、非方阵上核非线性后缩放恒等式，另检查 gate 的线性 up 支路与错误穿过 SiLU 的区别。不是实际模型图的认证。 |
| 固定数值合同/字节账单 | 通过有理数核 q 的 RNE 后再加 odd z（与把 z 移进 round 不同）、全零 epsilon 和 `Delta*q+scaled_zero=Delta(q-z)`；从实际源码形状数 int16 qweight、两个浮点 metadata buffer 与 padding。K1024/N8/G128 得 4.25 bit/weight，K128/N8/G128 含 padding 实为 6 bit/weight，另覆盖 G64/G32 和四字节 metadata。这里没有执行官方 packing。 |

## 未验证与移交

最终专项 7 组再次通过；`node scripts/audit-posts.mjs` 通过 94 篇、10 个系列、24 个标签，目标文件 `git diff --check` 无空白错误。结构通过不代表其他文章完成本轮内容复审。

未运行官方 AWQ calibration、Torch state 恢复、真实 packing/CUDA、TinyChat、其他 runtime、真实模型质量或 serving 性能。数值参考、源码布局检查和真实设备执行是不同证据层级。

正文已经静止；交 root 继续生产构建、公式/浏览器检查、单篇提交推送、对应 commit 的工作流和线上核验，并据实补记。此处不预先宣称已发布。

## Root 本地验收

最终三块实际代码及 7 组专项复跑通过；全部 27 个 JavaScript 审核套件、既有 Python、94 篇元数据和历史 coverage 检查通过，`git diff --check` 无空白错误。实际生产构建生成 109 个 HTML 页面、134 个文件；87 篇文章中的 1002 个原生行内公式和 963 个独立公式通过结构检查。

浏览器实际打开原本地 URL：125 个 MathJax 容器、0 个渲染错误、无单双美元残留，1265 像素视口无整页横向溢出，clipping 定义与分组误差公式截图正常。此项没有执行真实模型或 GPU，发布结果另记。
