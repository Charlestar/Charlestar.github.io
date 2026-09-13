# 混合精度训练新文独立全文审核

- 审核日期：2026-09-13。
- 对象：`_posts/2026-09-13-mixed-precision-training-loss-scaling.md`。
- 顺序：research → writer 完稿并停止修改 → 独立 auditor 通读、核验和补修。主审另行做发布与页面验证。

## 实际阅读与来源范围

完整阅读本篇正文及代码，不仅检查 diff。覆盖格式表、正常数/subnormal 与舍入、更新吸收、GEMM 累积精度、两种 16P 状态布局与 18P 变体、loss scaling 推导、autocast 分工、有效窗口分母、裁剪非线性、Python 骨架、nonfinite 跳步、scheduler/checkpoint 时钟和性能验证边界。

独立打开并核对以下一手资料的指定部分：

- [Mixed Precision Training v3](https://arxiv.org/html/1710.03740v3)：§1 与 §3.1–3.3；[摘要与提交历史](https://arxiv.org/abs/1710.03740)核对 2017 首次提交和 ICLR 2018。论文的旧式“低于 2^-24 归零”描述未被原样当成最近偶数舍入阈值。
- [PyTorch v2.5.1 AMP 文档源码](https://github.com/pytorch/pytorch/blob/v2.5.1/docs/source/amp.rst)：Gradient Scaling、Op Eligibility、CUDA Op-Specific Behavior。
- [v2.5.1 autocast_mode.py](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/amp/autocast_mode.py)：autocast 类文档、CUDA 示例与 forward/backward 边界。
- [v2.5.1 AMP examples](https://github.com/pytorch/pytorch/blob/v2.5.1/docs/source/notes/amp_examples.rst)：Typical Mixed Precision Training、Gradient clipping、Gradient accumulation。
- [v2.5.1 grad_scaler.py](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/amp/grad_scaler.py)：`scale`、`_unscale_grads_`、`unscale_`、`_maybe_opt_step`、`step`、`update`、`get_scale`、`state_dict`。核对状态机、found_inf 记录复用、普通与 fused/custom 路径、返回值和缩放状态恢复。
- [v2.5.1 AdamW](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/optim/adamw.py)：构造函数、`_init_group` 与 `step`，尤其 moments 的 `zeros_like(p)` 初始化。
- [Google BF16 说明](https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus)：Bfloat16 semantics、Choosing bfloat16、Mixed-precision training、Choosing values to represent。其 TPU subnormal 行为没有被推广成 GPU 通式。

预审还打开了 PyTorch 2.8 的 AMP 文档；最终正文核验已重新对齐 v2.5.1 固定源码，未用较新版本代替文章声明的版本。没有宣称全文精读所有被引用资料，也未重审正文链接到的历史博客全文。

## 发现与补修

1. 主要数学结论和数值无须改正；保留了最小 subnormal 以下仍可向上舍入的 `0.75*2^-24` 反例、原生 autocast 不额外自动创建完整 master、finite 梯度不保证 finite optimizer 更新等必要限定。
2. 补齐 Python 骨架的使用前提：`model`/`loss_fn` 已定义，模型已在目标 CUDA 设备、FP32 参数和训练模式，各输入/标签在同一设备。明确它不是自包含训练程序。
3. 示例显式设置 `fused=False`，使其与所解释的普通 AdamW/GradScaler 路径一致，避免读者将 fused optimizer 的内部处理与普通路径混为一谈。
4. 增加仓库内标准库复算入口，明确没有执行 PyTorch 骨架、GPU FTZ、分布式更新或收敛试验。
5. 为正文新增的逐 micro-batch 裁剪反例加入检查，并复算两种 16P 主体与 18P 变体。

## 已执行检查

`node scripts/reviews/sep13-mixed-precision-examples.mjs`：9 组通过。

- FP16/BF16 的格式上界、下界、局部分辨率，以及最近偶数的零/最小 subnormal 边界。
- `g=2^-26, S=4096` 的放大保留反例与前向 Infinity 不可由 unscale 修复。
- `(3,4), S=128, tau=1` 的正确/错误 clipping 结果与零梯度。
- 先累计 `3 + (-2)` 再 clip 得 1，先分别 clip 再加得 0。
- 固定 scale 的窗口累积及 `2*8+4*16=80` 错解码为 5 而非 6。
- 不等 micro-batch 有效分母，以及标准 DDP 平均下 world-size/global-count 的代数参考。
- 非有限梯度跳过的简化状态反例；它不模拟真实 GradScaler 内部实现。
- 低精度更新吸收与保留 FP32 更新的对比，含正文 `w=1.5, delta=2^-12` 与内存主体账本。

## 未实测与交付边界

没有安装或执行 PyTorch、CUDA、真实 GradScaler/AdamW kernel，没有 GPU 时间/显存、DDP/FSDP、完整训练轨迹或模型收敛对照。JavaScript 脚本是双精度 CPU 数学模型，局部采用 `Math.fround` 模拟 FP32，不能证明任意 GPU 后端的浮点等价。

主审已另行报告在本机 NumPy 上验证 FP16 的若干转换边界；这是主审执行的 CPU 交叉检查，不冒称 auditor 运行，也不是 GPU kernel 验证。Jekyll 构建、原生公式保留、真实浏览器与发布结果由主审追加或单独记录。

## 主审发布前复核

主审再次通读正文，核对修订后的更新骨架，并执行本篇 9 组复算及全仓 13 个 JavaScript 示例套件、Python agent 示例、文章 metadata/taxonomy 检查；全部通过。历史 coverage 脚本仍只声明原 77 篇快照，新文审核范围由本记录提供，不将历史记录扩称为当前全站语义证明。

生产模式 Jekyll 构建、生成页面/本地链接检查、原生公式保留检查和 `git diff --check` 通过；正文一个 Python 代码块通过 AST 语法解析，未导入或运行 PyTorch。主审的 NumPy CPU 转换交叉检查得到 `2^-26 → 0`、`2^-25 → 0`、`0.75*2^-24 → 2^-24`、`1.5-2^-12 → 1.5`，与本文最近偶数舍入例子一致。

使用 computer-use 检查本篇真实浏览器页面：52 个 MathJax 容器，未发现 MathJax 错误节点或页面横向溢出；目视检查标题、浮点格式表与训练代码高亮。此为默认桌面视口检查，不声称覆盖所有设备。上述构建针对当时工作区（含本批其他草稿），提交仅包含本篇正文、research、review 与复算脚本；部署状态另行在线核验。
