# AdamW 研究与作者交接记录

- 日期：2026-11-10。
- 正文：`_posts/2026-11-10-adamw-optimizer-state-weight-decay.md`。
- 研究由 `article_research` 逐项交接，作者完成正文与从正文提取的 CPU 专项，独立审核另记。
- 范围：实数 dense gradient、minimize、固定 beta、无 AMSGrad；框架语义固定 PyTorch v2.8.0。

## 一手来源与实际核对范围

1. [Adam v8](https://arxiv.org/html/1412.6980v8)：研究核对 §2 算法1及 §3 初始化偏差；作者再打开核对算法与论文版本。v9 HTML 不完整，未混引。
2. [AdamW v3](https://arxiv.org/html/1711.05101v3)：研究核对 §2、算法2；作者再核对衰减和学习率参数化。2017 为预印本首发，2019 为 ICLR。
3. [AdamW v2.8.0](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/adamw.py)：研究核对入口与伪码，AdamW 选择 `decoupled_weight_decay=True`。
4. [Adam v2.8.0](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/adam.py)：研究核对 `_init_group`、single/multi/fused 路径与分派；作者再查固定源码。关键为 grad None 跳过、逐 Parameter step、lazy state、moments dtype 和 fused overflow 语义。
5. [Optimizer v2.8.0](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/optimizer.py)：研究核对 state_dict/load 映射；不能把 param_names 误认为自动按名恢复。

## 写作组织与相邻内容

从梯度历史开始，展开 EMA、初始化修正、L2 与 decay 的不同、两步数例、None/zero/lr0，再进入 CPU 状态实现、存储、执行和恢复。与既有 AMP、FSDP、Roofline、Distributed Checkpoint 交叉链接，不重写其梯度累积或发布教程。

原论文算法2的 schedule multiplier 与 base learning rate 独立；正文采用 PyTorch `lr*weight_decay` 形式，并明确原论文系数须除 base alpha 才能映射。收缩应用于更新后的参数会多出 eta²lambda*u 交叉项。

## 数值与 CPU 专项

- `adamw-examples.mjs` 从正文 `adamw-scalar` 与 `adamw-state-example` 两个唯一 marker 提取实际 Python 代码。
- 独立 oracle 直接展开梯度历史的加权和，而非复制正文 moment 递推；覆盖可变 lr、None、显式0与 beta 边界。
- 固定例：theta=(1,2)、g=(1,1)、beta=(0.9,0.99)、lr=0.1、decay=0.1；eps=0 的纸面结果为 (0.89,1.88)，coupled Adam 为 (0.9,1.9)。第二次第一个坐标零梯度得到约0.813941985271064。
- 正文代码要求 eps>0（教学合同），并不是声称框架拒绝 eps=0。专项用小正 eps 与容差对照手算。
- 增加 eps位置、bias correction重排、纯衰减乘积、错误收缩顺序、save/resume、清零moments改变轨迹、gradNone与lr0等反例。
- 16P、8P、28P 是假设dtype和访问次数下的字节账本，不是硬件实测。

## 证据边界

CPU 专项验证文章实际示例与特定代数/状态反例；不执行 Torch/CUDA optimizer、不证明模型收敛、吞吐或全部参数域。完整站点、独立审核、提交与发布由主审处理。
