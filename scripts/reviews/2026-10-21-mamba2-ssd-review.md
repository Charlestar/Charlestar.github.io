# Mamba-2 与 SSD 独立全文审核

日期：2026-10-21。对象：`2026-10-21-mamba2-ssd-state-chunk-matrix.md` 全文、调研交接与 `mamba2-ssd-examples.mjs`。在写作代理冻结交接后接管修改，不与写作代理并发编辑。

## 独立阅读范围

实际读取 arXiv:2405.21060v1 的 §§2、4–7 中模型定义、双重形式、标量单位阵结构、分块矩阵、四步计算、复杂度及 head/block 设计段落；§3 的 semiseparable 表述结合显式跨分割因子自行核验。另读作者 Part I、Part III 中模型/算法、最小实现的 K² 警告、segment sum 稳定性及参数化讨论。未声称精读全部实验表、附录或复现论文评测。

源码逐行对照最终稿固定的 v2.2.2 commit `8ffd905c91d207f5c0cc84fc2a2fb748655094f0`：完整读取 `ssd_minimal.py` 与 `mamba2.py` 的初始化、forward、step 和 inference cache 分配。预检曾查看其他 tag，但审核结论不以其他 tag 偷换正文固定版本。作者博客末尾离散化公式的网页记号不用于覆盖固定 step 实现。

## 概念与公式逐段核验

- 状态为 N×P、输入 P、写入/读取 N，外积和读出形状一致；这是官方 P×N 缓存方向的转置表达，不按布局直接混用。
- 新写入不乘当前 a；矩阵项的积从 s+1 到 t，空乘积为 1。非零初态另含从 0 到 t 的积；进入 chunk 的状态则含从块首 l 到当前 t 的积，首项包含 a_l。
- 固定系数、零初态时核心对 X 线性；固定非零初态时是仿射；系数依赖输入的完整选择性模型不因此成为线性网络。SSD 对应未归一化 structured masked attention，不是 softmax 权重，也不是即插即用替换已训练 Transformer。
- 跨分割因子只需 N 个中间维，因此对应 off-diagonal 子块 rank≤N；mask 自身 rank 结构为 1。整个下三角 M 可满秩，不能把 semiseparable 混称为整体低秩。
- local 输出、零初态块末 summary、边界 affine scan、初态读出四项与展开式一致；复合满足结合律，不满足交换律。
- 令 K=ceil(T/Q) 后，边界顺序 scan 的 O(KNP) 覆盖尾块和 Q>T；固定 Q/N/P 时核心对 T 线性。最小官方函数第 3 步仍 materialize K×K，不能拿理想 scan 的线性复杂度替它背书。
- CPU chunked 采用逆向累乘，不在每个 pair 重做一整段乘积，主要工作与所述 pair/state 计数相符；它是顺序 Python 程序，不是性能对照或并行 kernel。
- 固定 step 中 dt 经 softplus，dA=exp(dt*A_cont)，写入为 dt*B*x。教学 b 吸收 dt 与最小例 X 吸收 dt 是两种等价放置，不能重复吸收；最小函数参数 A 是 log decay。
- 完整 block 另含卷积、SiLU、skip、gate/norm 和投影；共享 B/C group 不等于合并各 head 状态。教学算例没有冒称完整层或其梯度已验证。
- 固定 inference cache 确有 SSM 和短卷积两份状态。声明配置下 16 MiB 与 0.5625 MiB、合计 16.5625 MiB 算术正确；不是总峰值，更不是同质量 Transformer 的性能结论。回滚/前缀复用需恢复相关状态，混合 Attention 层仍需自己的 KV。

## 审核修正

1. 明确零初态的线性条件与固定非零初态的仿射项。
2. 将边界 scan 复杂度先写为 K=ceil(T/Q) 的 O(KNP)，再在 1≤Q≤T 范围内化简，覆盖教学代码允许的尾块和 Q>T。
3. 作者算法链接统一为审核与主代理实际读取的 `tridao.me` 官方交叉发布页；研究记录保持读过的 Part I 来源与最终 Part III 链接。
4. 浮点对照明确还要求中间结果处于有限范围；输入检查有限不代表乘加不会溢出。a=0/1 是教学代数边界，未宣称有限连续参数在实数运算中必生成这些端点。

## 独立补充测试

保留写作阶段的 exact Fraction tagged-write oracle、81 组 gate 组合、81 组浮点形状、非零初态/空序列/尾块/长小衰减和输入拒绝测试。独审新增：

- 用真实正文 recurrence 的单位脉冲输入恢复 temporal mixing matrix，再用 exact Fraction 高斯消元计算秩，不调用正文 dense 的矩阵构造作为 rank 参考。
- N=1、T=6 的整个下三角矩阵 rank=6，反证整体低秩误解。另对 N=1/2/3 各 40 组系数，共 120 个长度 7 算子、720 个切分子矩阵，核对跨分割因子与 rank≤N；遇到切分后首个 gate 为 0 时，该跨分割子块 rank=0。
- 在正文例子的每个 split 处保存完整状态再续算，拼接输出与 final state 均等于整段执行，包括空前缀/空后缀；不冒充完整官方 block 的卷积状态恢复测试。
- 零初态时改变 a_0 不改变输出，非零初态时可以改变，检查“当前写入不乘本步衰减”的边界。
- 只扰动未来输入，早期输出不变；有符号权重 `[1,-1]` 的行和为 0，不伪装成 softmax 概率。
- 对固定非零初态检验 `F(2X)=2F(X)-F(0)`，同时比较输出和最终状态，以仿射关系独立检查初态项。

实际运行 `node scripts/reviews/mamba2-ssd-examples.mjs`、104 篇 `node scripts/audit-posts.mjs` 和 `git diff --check`，均通过。没有导入/执行 Torch、官方 CUDA/Triton kernel、完整模型、反向梯度、GPU 性能或语言模型质量实验。

未发现剩余阻止发表的概念、逻辑、公式或教学代码合同问题。完整构建、数学渲染、提交推送、匹配提交的 CI 与线上正文由主代理另行验证。

## 主代理发布前整体验证

主代理全文复核后重新运行最终专项及全部 38 个 JavaScript 专项，均通过；full-agent-examples.py 通过。104 篇元数据、10 个系列、24 个既有标签检查通过。历史 coverage 工具仅验证 2026-09-09 的 77 篇记录，不将其解释为后来文章的语义审核覆盖。

生产 Jekyll 构建、站点及数学标记验证通过：120 个 HTML 页面、145 个生成文件；97 篇数学文章含 1669 个原生行内、1073 个块级公式。最终本地浏览器中本篇有 124 个 MathJax 容器，0 个公式错误，无裸露 `$$` 或页面横向溢出；另查看分块输出推导截图。三篇均检查过系列页，技术时间线位置正确，未增加标签或改变旧文章路由。

本批前两篇已分别提交推送为 `b184a339c0c36fbfa727efb6d2ec5bf9ca391fd9`、`e4560d2d7ff50f168a2d19f32d4e625c4f7ca059`；对应 Actions run `39476607027`、`39477252445` 的 build/deploy 均成功，实际线上页面含审核后的最终正文，公式无错误。第三篇的 CI 与线上验证必须等待本次提交推送之后，以上本地结果不预先声称其已部署。未执行 Torch、CUDA、官方模型、梯度或性能实验。
