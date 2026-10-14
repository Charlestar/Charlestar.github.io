# GPTQ 独立审核记录

审核日期：2026-10-15。正文：`_posts/2026-10-15-gptq-second-order-reconstruction.md`。

writer 静止后，独立全文审阅并修订概念、逻辑、维度、公式、数值推导、正文代码、来源和实验边界；未修改其他文章、系列计划或配置。

## 实际独立阅读的一手来源

- [GPTQ arXiv v2](https://arxiv.org/html/2210.17323v2)：实际逐段阅读 §3 的层重建和 OBQ 式 1–3、§4 的共享顺序/lazy batch/Cholesky 与完整 Algorithm 1；阅读 §5 的 calibration 设置、分组和 Practical Speedups，以及 §6 限制。本文固定网格反例不来自论文实验，未完整复读全部附录，也未复现论文质量、速度或显存。
- [固定 gptq.py](https://github.com/IST-DASLab/gptq/blob/2d65066eeb06a5c9ff5184d8cebdf33662c67faf/gptq.py)：独立跟读 `add_batch`、dead 坐标、damping、逆矩阵到 upper Cholesky、动态/静态 groups、actorder、block 内和 tail 更新、逆排列及浮点权重写回。注意 L98–104 的 `Hinv` 已经是 upper factor，不是原始逆。
- [固定 quant.py](https://github.com/IST-DASLab/gptq/blob/2d65066eeb06a5c9ff5184d8cebdf33662c67faf/quant.py)：独立阅读量化参数与浮点重建函数、`Quant3Linear.pack` 的整数装位，以及 `forward` 的单 token 限制和 CUDA 调用；没有执行这些框架/打包/kernel 代码。
- [同版本 README](https://github.com/IST-DASLab/gptq/blob/2d65066eeb06a5c9ff5184d8cebdf33662c67faf/README.md)：核对 camera-ready 的 act-order/true-sequential 与 2023 年 7 月 static-groups 增强，不倒写为初版算法定义。
- [GitHub commit API](https://api.github.com/repos/IST-DASLab/gptq/commits/2d65066eeb06a5c9ff5184d8cebdf33662c67faf)：独立读回 SHA 与 2023-07-11 提交日期；采用不可变源码，不以当前 main 的行为替代该版本。

源码阅读支持固定版本的描述，不证明当前设备执行了对应分支。

## 主要核对结论

1. 系列方向为 `X[M,K] W[K,N]`，论文/`nn.Linear.weight` 为 `W[N,K]`。层 proxy 的精确 Hessian 是 `H=2XᵀX`，不是模型任务 loss 的 Hessian。所有输出通道共享输入统计，但 Gram 本身仍有 K² 个元素。
2. 当前自由集为 F，必须使用 `P_F=(H[F,F])^-1`。固定已经量化的坐标，不等于保留完整逆的同名主子块。残差 `r=w_current[q]-Q(w_current[q])`、约束 `delta_q=-r`、更新 `delta=-r P_F[:,q]/P_F[q,q]` 与最小新增损失 `r²/(2P_F[q,q])` 的符号和因子一致。因为此前已经完成剩余连续变量的最优补偿，自由方向梯度为零，当前增量目标没有漏掉一次项。
3. `H^-1=UᵀU` 的 upper factor，其 trailing factor C 满足 `P_F=CᵀC`；因此当前首行比例为 `P_F[q,j]/P_F[q,q]=U[q,j]/U[q,q]`。Schur 更新取在逆矩阵上，不混用 H 自己的 Schur complement。
4. K×N 布局的 block tail 更新为 `U[B,tail]ᵀ E[B,:]`；维度为 `(tail×B)(B×N)`。`blocksize` 组织更新、`groupsize` 定义量化参数共享，是不同契约。专项仅核固定网格的调度与排列，不声称认证动态 grid/exporter。
5. `H+lambda I` 对应 `L+lambda/2 ||Delta w||²`。源码默认按对角均值的 1% 加 damping；整体缩放不变性只在相应矩阵/目标被一致缩放时成立，不把 dead 对角被设置为 1 的额外处理也称为纯归一化。
6. 固定三维反例在同一 16 值网格内依次得到 SSE 0.08、14/75、0.84；RTN 0.44，全部 4096 候选的唯一最优为 `[-1,0,0]`、SSE 0.24。它只否定无条件的离散最优/最终优于 RTN 保证，不否定论文对应模型实验。
7. 浮点 Q/DQ、实际 packing、运行时算术是不同步骤。`gptq.py` 写回浮点原 dtype；`quant.py` 中 `quantize` 返回浮点重建值，3-bit `pack` 才建立低位 buffer。论文的 batch-one 路径动态反量化、减少权重流量，不等于所有乘法使用原生 INT4 MMA。

## 本轮实修

- 二维代码原本只检查非有限最终补偿，可能漏掉可恢复结果的中间下溢。例如 `H=[[1e-150,0.5e-150],[0.5e-150,1e-150]]`、`w=[1e-300,1e-300]`：正确第一步连续补偿为正 `5e-301`，但 `(w-q)/Uqq` 先变成 0，原程序会静默漏掉。现在对 determinant 乘积、inverse/factor 的关键非零量、归一化残差和补偿乘积下溢为零，以及非有限值作显式拒绝。代码旁说明动态范围有限，不尝试挽救所有数学上可表示的结果。
- 这仍不是病态矩阵的稳定数值库：binary64 舍入、近消去、未下溢为零的精度损失不会因几个 guard 就消失；测试只证明列明样例和边界，不能推出所有有限 SPD 输入都有高精度解。
- 澄清 `add_batch` 的样本计数先于 token 展平，不直接等于本文 X 的 token 行数 M。补明 K² 统计成本，以及论文用已量化 block 的输出为下一 block 生成 calibration 输入，避免把其流程误写成始终使用完整高精度中间状态。
- 校正 Cholesky/damping、dead 处理和 `Quant3Linear.forward` 的源码行锚点。
- root 的首次构建发现当天 `09:00 +0800` 尚在未来，Jekyll 默认跳过本文。按 root 指令把 `date` 改为 `2026-10-15`，不带时分；文件名、日期对应 URL、作者和系列顺序不变。这是发布过滤修复，不是算法内容变更。

## 已执行的正文专项

命令：`node scripts/reviews/oct15-gptq-examples.mjs`，`BLOG_PYTHON` 指向本机 Python。只使用 Node/Python 标准库，8 个不同语义组；不把矩阵元素、111 组参数 fixture 或 4096 个候选分别灌成测试组数。

| 验证组 | 真实执行与独立 oracle |
| --- | --- |
| 两块正文代码 | 按唯一 marker 提取实际 Python，各自独立执行、运行自身断言并解析打印结果；确认无其他遗漏的 Python 块。另核对二维 weight MSE 与层 SSE 是不同目标。 |
| 固定网格 | 用 Fraction 对所有重建 level 的距离/偶数优先级搜索，核两个真实量化器在正负 tie、相邻浮点、饱和区间的行为，非复制 `round`。 |
| Hessian/归一化/damping | 用标量重建 loss 的精确有理数中心有限差分核对 Hessian；核 lambda/2、奇异输入拒绝及加正对角后的演示。独立计算对应 `add_batch` 的 Gram 递推与 batch 计数，未运行官方 PyTorch。 |
| 连续约束最小 | 111 组二维 SPD/权重 fixture，直接对剩余变量求导得到最小值，以正负扰动核最小性，交叉检查实际轨迹、P/U 和正比例缩放不变性。 |
| 当前自由集/Schur | 每一步用 Fraction Gauss-Jordan 重新求当前 Hessian 主子矩阵的逆，核 trailing upper factor 和 Schur；另用不同 SPD 构造证明复用原始 P 会把最终 code 从 `[0,1,0]` 错改成 `[0,1,1]`。 |
| Lazy block 与布局 | K=5、N=3 的非方阵，blocksize 1/2/3/128（含不完整尾块），与逐输出列、逐自由集精确求逆的路径比较；核排列前后矩阵乘与恢复，不宣称处理顺序之间量化结果不变。 |
| 4096 候选精确穷举 | 独立用整数残差分子评分，不使用浮点 `isclose` 来决定 global tie；逐个核对正文浮点 loss，精确得到唯一 minimizer 和 RTN/顺序补偿大小关系。 |
| 数值范围契约 | 实际调用正文二维函数，覆盖形状、非对称、非有限、非 SPD、determinant 上/下溢、inverse/factor 不可形成、极近奇异时 factor 消去、归一化残差/补偿乘积下溢、误差/乘积/最终加减溢出；保留可正常处理的零残差与尺度差异很大的对角样例。 |

本轮 `node scripts/audit-posts.mjs` 通过 93 篇、10 个系列、24 个标签；`git diff --check` 无空白错误。没有因这些静态检查就声称其他文章完成本轮内容复审。

## 验证边界与移交

未运行官方 GPTQ/PyTorch、真实模型 calibration、实际 packing/加载、CUDA kernel 或模型 PPL/任务质量/性能实验；没有把教学函数称为完整 GPTQ，也没有把这个固定网格穷举推广到任意 exporter 或模型。

正文已经静止；root 接手进行生产构建、页面公式/布局、逐篇提交推送和 commit-specific 部署/线上核验，并据实补记。本审核不预先宣称已发布。

## Root 实际本地验收

全文、最终专项脚本及 review 已读回。重跑 8 组专项、全部 24 个 JavaScript 套件、既有 Python、93 篇 metadata 和历史 coverage 检查通过；专项在审核最后补充断言后再次通过。修正未来时刻后实际生产构建生成 108 个 HTML 页面、133 个文件，站点检查通过，86 篇中的 776 个行内和 944 个独立公式通过数学检查。浏览器从本篇实际路由读到 91 个 MathJax 容器、0 个错误、无美元残留、1265 像素视口无横向溢出，最终 normalized-error guard 在页面中；实际截图检查了约束补偿公式的排版。以上仍不代表真实 GPTQ/LLM/GPU 实验。
