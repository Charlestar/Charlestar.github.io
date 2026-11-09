# 融合 LM Head 与交叉熵独立全文审核

- 审核日期：2026-11-10。
- 正文：`_posts/2026-11-10-fused-linear-cross-entropy.md`。
- 顺序：研究 → 作者完成正文与专项并冻结 → 独审全文、来源与代码，补修 → 冻结交主审。

## 全文覆盖

逐节检查了形状与字节账本、softmax/交叉熵、在线归一化、巨大公共偏置的相消、mean/sum 梯度、忽略行与上下文的区别、token 分块、提前计算局部梯度与上游缩放、全局有效计数、next-token 对齐、临时量寿命、GEMM 计数、两段真实 Python、CCE 词表 tile/过滤、共享权重与词表并行接口和全部引用。不是只看 diff。

## 实际独立阅读的来源

- [Liger 论文 v1 §3.2](https://arxiv.org/html/2410.10989v1#S3)：交叉熵、融合线性交叉熵和 chunk 梯度缩放的 Remark。具体 ignore 计数与接口以固定实现为准。
- [Liger v0.5.2 FLCE](https://github.com/linkedin/Liger-Kernel/blob/v0.5.2/src/liger_kernel/ops/fused_linear_cross_entropy.py)：完整 forward/backward 与 Function；chunk 选择、有效计数、额外 scaled gradient 块、zeros_like dtype、局部梯度 detach、标量上游原地缩放。
- 同版 [CE kernel](https://github.com/linkedin/Liger-Kernel/blob/v0.5.2/src/liger_kernel/ops/cross_entropy.py)、[utils](https://github.com/linkedin/Liger-Kernel/blob/v0.5.2/src/liger_kernel/ops/utils.py)、[FLCE module](https://github.com/linkedin/Liger-Kernel/blob/v0.5.2/src/liger_kernel/transformers/fused_linear_cross_entropy.py)：忽略行早返回、在线统计、梯度原地写入、连续性转换、上游乘法和 reduction 入口。module 接受 `none` 不等于底层实现返回逐行损失，正文没有扩展此承诺。
- [Online Normalizer v2 §2、§3、算法 3 与 §3.1](https://arxiv.org/html/1805.02867v2#S3)：在线最大值/指数和及二元合并。实数恒等式不代表任意浮点归约顺序逐位一致。
- [CCE v2 §3、§4.1–4.3、§5.2–5.3，附录 B/D/E](https://arxiv.org/html/2411.09009v2)：词表 tile、重算梯度、过滤近似及预训练数值边界。没有重复其关于 BF16 最小可表示值的不准确措辞，也没有把微调实验推广为任意训练的无损保证。
- [Transformers v4.46.3 loss_utils.py](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/loss/loss_utils.py)：先在样本维内部 shift，再 flatten；fixed loss 的归约入口。
- [PyTorch v2.8.0 CrossEntropyLoss](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/loss.py) 的 hard-label 定义及 [LossNLL.cpp](https://github.com/pytorch/pytorch/blob/v2.8.0/aten/src/ATen/native/LossNLL.cpp) 的有效计数、空输入/忽略 target 分支和 backward：CPU reference 的全忽略 mean 主动拒绝，是显式策略，不声称与所有框架返回值一致。

以上是实际查看的方法与函数范围，不宣称精读全部论文、所有后端或复现模型实验。

## 发现与修正

1. 开头“交叉熵只返回标量”补上 mean/sum 归约前提；不能覆盖 `none`。
2. 原 CPU 示例验证了有限输入，却未拒绝乘加、损失或梯度的非有限中间/输出。补有限实数转换、巨大整数错误归一化和有限求和检查；覆盖投影、目标差、dX、dW 累加与上游缩放。
3. reference 对 ignored 行不形成 head 投影，使全忽略 sum 在有限但投影本会溢出的输入上也返回零。它不是从模型上下文中移除 token，也不声明 Liger 使用同一跳过路径。
4. 清理 `dz`、`scores` 对上一行临时量的引用；明确 Python 列表示例不对应严格的 GPU buffer/peak 计数。
5. 保留原文正确的负 log-likelihood 梯度符号、有效 token 加权、ignore_index 不移除其他行词表列、2 GiB/64 MiB 局部比较、额外 scaled chunk、detach/原地缩放的一阶单次边界和 CCE 过滤近似。
6. 加入可执行专项命令与未实测边界，润色空格硬切的中文标题。

## 实际执行

`node scripts/reviews/fused-linear-ce-examples.mjs` 从正文两个唯一 marker 提取并执行真实 Python，最终通过。专项包括：

- 四组小形状，两种归约，四种 chunk 大小，与独立的完整矩阵、未平移 exp/log oracle 对照。
- 对 X、W 每个元素做中心差分，覆盖上游 0、0.3、-1.7；输入、权重和标签未被改写。
- 具体 loss 0.5626709415555415 与全部梯度；忽略前缀、空有效块、尾块、全忽略与空输入；ignore_index 位于词表内仍保留其他行的该类别。
- 在线统计与独立 max-shift 归一化对照；三段词表统计合并；公共 logit 偏置与 1e16 相消反例。
- 非法形状、标签、归约、块大小、ignore 类型、非有限输入、巨型整数，以及有限输入导致投影/损失/dX/dW/上游缩放溢出的拒绝。
- 先 flatten 再 shift 的跨样本监督反例；保存梯度连续原地乘 2、3 后不等于重新乘 3 的重复 backward 边界示例。后者是源码语义的纯标量演示，不是实际 autograd 运行。
- 2/4 GiB、64 MiB、1 GiB 的字节算例。

`git diff --check` 通过；未暂存的新文件不在此命令的完整覆盖范围。站点构建、浏览器和发布由主审另行验证。

## 结论与未实测范围

在上述范围内，全文审核及补修已完成，可交主审进行最终站点验证与单篇发布。未执行 PyTorch/Triton/CUDA 的实际 FLCE、GPU 显存/吞吐、BF16 累加误差、共享参数真实训练或分布式 collective；CPU 双精度 oracle 与固定源码阅读不能替代这些证据。有限测试不证明所有可表示输入或所有模型质量不变。

## 主审发布前验证

主审复核最终修订并重跑 CPU 专项、106 篇元数据检查、Jekyll 生产构建、122 个 HTML 页面和 147 个生成文件检查。数学标记检查通过；无界面浏览器打开本地最终稿，确认正文及 65 个 MathJax 结果，未见公式错误。线上部署在推送后另行检查。
