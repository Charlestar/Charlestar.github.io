# AdamW 独立全文审核

- 审核日期：2026-11-10。
- 正文：`_posts/2026-11-10-adamw-optimizer-state-weight-decay.md`。
- 顺序：研究交接 → 作者完成正文与专项并冻结 → 独立审核全文、补修正文与专项 → 冻结交主审。

## 完整覆盖

逐节审读了 EMA 展开、初始化偏差、epsilon 重排、SGD/L2 与 AdamW 的区别、原论文和 PyTorch 衰减参数映射、两坐标一步例与第二步零梯度例、错误更新顺序、None/零梯度/零学习率、逐参数步数、冻结参数、累计收缩、两段真实 Python、显存与流量账本、foreach/fused、参数组与恢复，以及全部引用。不是仅审查差异。

## 独立核对的原始来源

- [Adam v8 §2、§2.1、§3](https://arxiv.org/html/1412.6980v8)：算法 1、EMA/偏差修正、整段梯度正比例缩放的理想性质。未援引其收敛定理，也没有复现论文实验。
- [AdamW v3 §2、算法 1/2、附录 A](https://arxiv.org/html/1711.05101v3)：普通 SGD 的系数映射、非标量预条件器的不等价、论文基础学习率与 schedule multiplier 的参数化。
- [PyTorch v2.8.0 adamw.py](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/adamw.py)：构造、解耦开关、state 恢复兼容与伪码。
- [PyTorch v2.8.0 adam.py](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/adam.py)：`_init_group` 的梯度筛选与 lazy state、`_single_tensor_adam` 的 decay/moments/bias correction、fused 路径的 found_inf/step 处理与分派。没有执行 GPU 内核。
- [PyTorch v2.8.0 optimizer.py](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/optimizer.py)：`state_dict`、`load_state_dict` 的参数顺序映射、param_names 说明、dtype 转换规则、foreach/fused 文档。
- [PyTorch v2.8.0 grad_scaler.py](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/amp/grad_scaler.py)：普通与自定义 AMP optimizer 的跳步入口；[lr_scheduler.py](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/lr_scheduler.py)：初始化与 `step` 的调度计数。用于复核外层调用数不应混成每参数有效更新数的边界。

只声明这些方法、章节与函数被实际查看，不宣称完整精读所有论文或所有后端实现。

## 发现与修正

1. 原稿将“衰减”笼统列为梯度缩放不变性的破坏因素。修为未同比缩放的耦合 L2，并说明在同一初始参数/日程、零初始 moments、epsilon=0、非零分母等条件下，AdamW 解耦衰减本身不破坏整段梯度统一正比例缩放的不变性。
2. 原标量代码只检查最终 theta/m/v，可能漏掉有限 v 经偏差修正后变成无限大的 v_hat；这会令归一化方向错误变成零而最终参数仍有限。新增 moments、修正 moments 与分母的有限检查。
3. 巨大 Python 整数可能使 `math.isfinite` 抛 OverflowError。代码显式限制可转换为有限双精度的实数输入，统一 ValueError，并在更新前转为浮点数，避免两个大整数的乘积随后转换时绕过该处理。None 路径仍返回原状态对象。
4. 增加正文专项命令与证据边界；把几处用空格硬切的标题调整为自然中文。
5. 补充两个参数不同参与历史/参数组的独立 oracle、错误全局步数反例、理想梯度缩放不变性及论文衰减系数映射检查。
6. 一步 `(0.89, 1.88)`、coupled `(0.9, 1.9)`、二步 `0.813941985271064`、错误顺序交叉项、16P/8P/28P 账本原文均正确，复核保留。没有把字节估算当成真实显存峰值或 HBM 流量。

## 实际执行

运行 `node scripts/reviews/adamw-examples.mjs`，从正文两个唯一 marker 提取并执行真实 Python 代码，最终通过。

- 126 组固定随机种子的梯度轨迹，与直接展开全部参与历史的加权和 oracle 对照；包括 None、零梯度、零学习率、不同 beta 和不同长度。
- 一步/二步结果、L2 与 decay 的 moments 分歧、None 对象不变、lr=0 仍更新状态。
- 两个参数在 4 次外层调用后分别计数 3/1；不同参数组与独立 oracle 一致；错误地使用全局计数产生不同后续结果。
- 每个切分点序列化完整状态后的继续执行，与连续执行一致；只恢复参数清零 moments 的反例。
- 常梯度偏差修正、纯衰减乘积、错误收缩顺序、epsilon 内外位置及重排。
- 5 个历史前缀下，正比例缩放梯度与解耦 decay 并存时的理想不变性；原论文系数到 PyTorch 系数的独立代数映射。
- 非法类型、非有限数、巨大整数、moment/修正 moment 溢出拒绝；极大但有限 epsilon 的可接受例。
- 字节模型 16 GB、8 GB、28 GB（十进制）。

审核新增的一条极大 epsilon 测试最初误以为 `sqrt(v_hat)+max_float` 必然溢出；实际 sqrt 项远小于该量级的 ULP，会舍入回 max_float。已修正为接受且结果符合纯衰减的检查，最终专项通过。未掩盖这次测试假设修正。

`git diff --check` 通过；新增文件尚未进入索引时此命令不构成对其内容的完整验证。站点构建与浏览器渲染待主审执行，未在本记录中预先宣称通过。

## 边界与交接结论

本轮运行的是标准库 CPU 标量与状态检查，没有执行 PyTorch optimizer、CUDA/fused/foreach 后端、AMP 实际跳步、分布式恢复、GPU 内存/性能或真实模型训练。固定源码阅读与 CPU oracle 是不同证据，不能互相替代。代码也不证明全部可表示输入、所有浮点精度和训练轨迹的正确性。

在上述范围内，全文审核与修订已完成，可交主审进行站点验证和单篇发布。

## 主审发布前验证

独审冻结稿与生产构建前后正文的 Git blob 一致。主审重跑最终 CPU 专项，105 篇元数据检查、Jekyll 生产构建、121 个 HTML 页面与 146 个生成文件检查均通过。97 篇已有数学文章外，本篇新增 72 个行内与 14 个块级原生公式；无界面浏览器打开本地页面确认正文与全部 86 个 MathJax 结果，未发现公式错误。线上部署在提交推送后另行核实。
