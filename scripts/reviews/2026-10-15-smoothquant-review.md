# SmoothQuant 独立审核记录

审核日期：2026-10-15。既有正文：`_posts/2026-08-02-smoothquant-w8a8-activation-outliers.md`。

在 writer 静止后独立全文核查概念、公式、矩阵轴、算例、代码、共享参数改写、来源和实验边界。本轮保留既有发布日期 `2026-08-02 09:00:00 +0800`、文件名/URL、作者和系列 metadata；未修改其他文章、计划或配置。

## 实际独立阅读的一手资料

- [SmoothQuant arXiv v7](https://arxiv.org/html/2211.10438v7)：实际阅读 §2–4 的整数表示、外维 scale、迁移恒等式和 alpha，§5.1/表 2 的 O1–O3 与 Implementation；核对 §5.2 的后续模型配置，以及 §5.3–5.5 的 A100、Prefill/Decode、batch/seq、质量/速度边界。v7 是 2024 修订，不将其全部实验与新增模型归于 2022 首发。
- [固定 smooth.py](https://github.com/mit-han-lab/smoothquant/blob/c61476d728e42ae0d8a35e7e78494edcac3237b5/smoothquant/smooth.py)：独立读完整 folding 和模型分支。`smooth_ln_fcs` 沿物理 W[N,K] 的 K 维，从全部共享 Linear consumers 取 weight absmax 的并集最大值，先 clamp weight statistics，再 clamp 最终因子；LayerNorm 的 gamma/beta 同除 S，各 Linear weight 乘 S，当前 Linear bias 不变。RMSNorm helper 只改 gamma。核对 OPT Q/K/V、Llama gate/up、Falcon 相应共享 norm 与 Mixtral router/experts 的消费者集合。
- [固定 fake_quant.py](https://github.com/mit-han-lab/smoothquant/blob/c61476d728e42ae0d8a35e7e78494edcac3237b5/smoothquant/fake_quant.py)：独立读 Q/DQ functions、`W8A8Linear` 的浮点 buffer、`F.linear`、output quant 与相关转换器。from_float 或不同模型 wrapper 的默认参数不等于论文表 2；输出 Q/DQ 也不证明真实 INT8 BMM。
- [固定 calibration.py](https://github.com/mit-han-lab/smoothquant/blob/c61476d728e42ae0d8a35e7e78494edcac3237b5/smoothquant/calibration.py)：独立读 `get_act_scales` 的 Linear input hook、展平非 hidden 维、逐通道 absmax 与跨样本 max；另读 static input/output scalar 收集和 `/127` step。平滑统计和静态量化 step 不是同一个对象。
- [GitHub commit API](https://api.github.com/repos/mit-han-lab/smoothquant/commits/c61476d728e42ae0d8a35e7e78494edcac3237b5)：独立读回 commit 与 2024-07-12 日期，正文链接指向该固定版本。

未使用未经阅读的 `latest` backend 页面支撑具体当前支持表。源码阅读不是执行这些库或模型的证据。

## 审核结论及实修

1. 系列布局 X[M,K]、W[K,N] 下，正对角 S 沿输入 K 维，`(XS^-1)(SW)=XW`；这不等于 INT8 编码后的计算仍严格相等。smoothing factor 与 Q/DQ step 分开；外维 scale 可以移到整数归约后，沿 K 改变的 factor 不能无条件如此处理。
2. a、b 均正且没有 clamp 干预时，`a'= (ab)^(1-alpha)`、`b'=(ab)^alpha`，alpha=0 不等于 identity，alpha=.5 只平衡这两个 range；ab<1 时 S 随 alpha 增大而减小。本文 64×0.01 的 100/80/64 例正确。
3. 修正过强的“量化后误差不可能凭空消失”为“不能保证消除”。正文自己的两通道构造恰好变成可精确重建的 `[1,1]`，输出误差确实为零，不应被前面的措辞否定。
4. 代码不需要扩成通用模型库。补充明确：`smoothing_factor` 只是采用相应 clamp 顺序的 Python 浮点参考；官方先将统计和因子转换到 weight dtype，FP16/BF16 不保证逐 bit 相同。有限正 S 仍可能让后续参数/activation 运算溢出或下溢。本文全零 Q/DQ 的 scale 1 是自己的合法约定，不复制 fake_quant.py 在 absmax 上加最小 clamp 的数值合同。
5. LayerNorm gamma/beta 必须一起改；当前 Linear bias 不改；同一 norm 的所有 Q/K/V 等消费者使用同一 S。用不同 S 反复改同一 norm、遗漏一条分支、忽略 residual，或者任意把缩放穿过 SiLU/GELU，都不由单层线性恒等式保证正确。
6. 表 2 的 O1/O2/O3 权重均为 per-tensor；activation 分别 per-token dynamic、per-tensor dynamic、per-tensor static。它们是不同 recipe，不是依次执行的三个阶段。v7 表 7 后续模型使用 per-channel weight 不改变表 2 的定义。静态 step 省在线统计，不省 activation 编成整数本身。
7. 固定 fake_quant.py 是浮点 Q/DQ + 浮点 linear，但论文 §5.1 的 PyTorch/Hugging Face 与 FasterTransformer 性能实现都使用 CUTLASS INT8 Linear/BMM。两条事实同时成立，不能用文件名或框架名代替真实 kernel 证据。
8. 权重 payload 约减半不推出总服务显存减半；W8A8 不自动得到物理 INT8 KV Cache；Prefill/Decode、TP、scale 方向、融合和完整 serving 质量仍需独立验证。正文未把历史测量变成所有模型/硬件保证。

## 已执行的正文专项

`node scripts/reviews/oct15-smoothquant-examples.mjs` 从最终正文直接提取两块 Python，用 `BLOG_PYTHON` 指定本机 Python。只执行标准库数值验证，7 个不同语义组；不按 code、矩阵元素或参数组合扩充组数。

| 验证组 | 独立证据 |
| --- | --- |
| 两块真实代码 | 唯一 marker 提取、独立执行、运行自身断言和解析真实输出；两通道原输出 2、Q/DQ 前 260/127、平滑后 2。确认没有遗漏的 Python 块。 |
| Q/DQ 码表 | Fraction 搜索 255 个重建 level 的距离并用偶数 parity 打破 tie，核不同二进制精确 step、正负半步、相邻浮点与全零；不是抄 `round` 作为 oracle。 |
| Alpha 与 clamp | 使用 2 的幂在指数域建立独立 oracle，核范围恒等式、alpha 端点、ab>1/ab<1/ab=1；零统计和 epsilon 触发时不再套用严格正值的范围平衡式。 |
| 矩形 GEMM/K 轴 | 精确有理数核 M×K、K×N 的变换恒等式与错误轴方向；实际 Q/DQ 提供逐行/逐输出列重建后，独立 GEMM 与整数部分和外维 scale 交叉检查。两个相同整数和但不同 K 方向系数的构造，拒绝无条件移出归约。 |
| 共享 folding/graph | 对真实正文 affine/linear 结果做独立有理数核对；再测试多行和三个不同输出宽度的投影，gamma/beta、当前 bias、漏消费者、残差支路以及 SiLU/LayerNorm 输入改写反例。只核代数，不运行真实 norm/Transformer graph。 |
| 不保证误差单调 | 另构造 `X=[1,1]`、`W=[127,1]^T`：原 Q/DQ 和 alpha=0/1 都为精确 128，alpha=.5 却约 16250/127，即 127.952756。范围平衡不保证输出误差最优；与正文另一个恰好误差为零的构造并不矛盾。 |
| 有限数值/dtype | 真正调用正文函数，核非法统计/alpha、空向量、非有限值与最小 subnormal 使 step 下溢的拒绝；另明确展示有限 S 可使后续乘法溢出/除法下溢，以及全零统计 alpha=0 的 Python 因子约 1e5 超出 binary16 有限范围。`struct.pack` 只核格式可表示性，未执行 FP16 kernel。 |

## 边界及移交

未执行官方 SmoothQuant/PyTorch/CUTLASS、真实模型 calibration、INT8 kernel、GPU、TP 或 serving 实验；未复现论文质量、显存与速度。公式参考、源码契约、实际 GPU 路径和端到端效果是四类不同证据。

正文已经静止；root 接手生产构建、数学渲染/页面布局、逐篇提交推送、对应 commit 的工作流与线上核验，并按实际结果补记。本记录不提前宣称已经发布。

## Root 本地验收

全文、最终源码限定、完整专项与审核记录已读回；7 组专项在最后补充断言后复跑通过，全部 25 个 JavaScript 套件、既有 Python、93 篇元数据和历史 coverage 检查通过。实际生产构建生成 108 个 HTML 页面、133 个文件，页面检查与 86 篇中 853 个行内、946 个独立公式检查通过。浏览器从原 URL 读到最终限定、folding 示例与 O1–O3 表：92 个 MathJax 容器、0 个错误、无美元残留，1265 像素视口无整页横向溢出；实际截图核对等价变换及维度公式排版。以上不证明真实 CUTLASS、GPU 或模型实验。
