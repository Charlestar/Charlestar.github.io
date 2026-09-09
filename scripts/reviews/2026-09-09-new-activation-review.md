# Activation Checkpointing 新文独立全文审核

- 审核日期：2026-09-09。
- 对象：`_posts/2026-09-09-activation-checkpointing-rematerialization.md`。
- 顺序：research → writer 完稿移交 → 独立 auditor 全文核对并修改。

## 实际覆盖与来源

完整审读两层 MLP 梯度、分段执行图、平方根内存模型及非对称推广、F/B/R 成本、Transformer/FlashAttention 保存量、PyTorch 包装、RNG 与副作用、SAC、并行通信和测量方法，包含参考资料。未只查看改动部分。

独立核对以下原始章节/函数：

- [Chen 等 2016 v2 §3、§4.1–4.4、§5.1](https://arxiv.org/html/1604.06174v2)：生命周期、逐段重建、平方根模型、递归边界，及 feature-map 估算不包含参数与临时内存的定义。
- [Reducing Activation Recomputation v1 §4、§5](https://arxiv.org/html/2205.05198v1)：GeLU Transformer 与 dropout 的保存量前提、selective 重算对象、并行保存量及额外 all-gather 条件。
- [PyTorch v2.8.0 `checkpoint.py`](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/utils/checkpoint.py)：`checkpoint`、`CheckpointFunction`、`_checkpoint_without_reentrant_generator`、RNG 保存/恢复、`_default_meta_extractor`、early-stop hook、SAC context/cache。
- [PyTorch 2.8 checkpoint 文档](https://docs.pytorch.org/docs/2.8/checkpoint.html)：reentrant/non-reentrant 差异、编译模式 preserve_rng_state、默认 determinism_check 的实际检查范围。
- [PyTorch AC/SAC/Memory Budget 官方说明](https://pytorch.org/blog/activation-checkpointing-techniques/)：联合图分割、算子保存偏好与实验接口边界。
- [Megatron Bridge Activation Recomputation](https://docs.nvidia.com/nemo/megatron-bridge/latest/training/activation-recomputation.html)：full/block/uniform/selective、Flash Attention Integration、parallel/feature interactions；动态页于审核日查阅。

只声明上述章节/函数被实际核对，不宣称全文精读全部来源，也没有执行其框架实现。

## 发现与修正

1. 给“工作量增加 33%、吞吐下降 25%”增加必要条件：只有实际耗时也与算术量同比例变化时，后一吞吐结论才成立。单由 `F+B+R` 不能推出 wall-clock 性能。
2. 补充已核对版本在 `torch.compile` 下不会以 `preserve_rng_state=False` 关闭 RNG 保存；同时仍保留 Python/NumPy/自定义 generator 不属于通用进程快照的边界。
3. 正文新增标准库复算命令，明确它没有执行 PyTorch 包装，不能据此宣称分布式训练或 GPU 实测通过。
4. 两层 MLP 的矩阵维度与所有梯度式、`n/k+k` 中 k 为段长、非对称最优段长、MiB/GiB 数值、dropout 的 loss=2/grad=4 反例原文均正确，经独立复算保留。
5. 保留了 FlashAttention 不重复扣减不存在的 S×S 保存矩阵、early-stop 不是严格执行两遍、SAC 峰值收益不独立相加、FSDP/TP/CP 可能重放通信等限定。旧文献的固定保存量系数未被当成现代后端通式。

## 已执行验证

`node scripts/reviews/new-activation-examples.mjs`：8 组检查通过。

- 64 层等大小代理模型的最优段长与端点。
- 2048/640/512/640 MiB 表、非对称边界最优值、32 MiB hidden 与 1 GiB attention。
- 两层 tanh MLP 的输入、W1、W2 全元素梯度与中心有限差分一致。
- 7 层纯函数链在段长 1/2/3/7 时，loss、输入/权重梯度与保存全部状态的实现一致，且与有限差分一致。
- 与正文相同的 dropout 错误重放梯度反例。
- shape 相同但权重版本改变时 Jacobian 不同。
- 工作量倍率与吞吐倒数的百分比计算。
- 两时刻内存峰值移动导致区域节省量不具有独立可加性。

## 未实测边界

本轮没有安装或执行 PyTorch，没有真实 dropout kernel、GPU 显存/时间、编译、SAC、FSDP/TP/CP/PP 或完整训练轨迹对照。数值脚本采用 JavaScript 双精度与小矩阵；它验证本文特定算例，不证明所有后端/浮点精度的等价性。

## 主审发布前验证

- 主审完整阅读正文、审核记录及 8 组配套检查；独立复跑 `new-activation-examples.mjs` 通过。
- 在全文审核结束后重新进行生产构建，页面检查通过 92 个 HTML / 117 个生成文件，公式保留检查通过 269 个原生行内式与 804 个展示式。当时构建含 79 篇；这不是未来新增文章的验证承诺。
- 真实浏览器打开本篇页面，确认最新的“工作量与耗时同比例增长”限定已显示；63 个 MathJax 公式容器，无 `merror` 或 `data-mjx-error`。
- 目视检查 MLP 梯度与矩阵维度、分段显存对照表和相邻推导，未发现排版异常。
- 本轮此前运行的全站内容检查、历史范围检查、全部 JavaScript 示例套件与 Python 示例也已通过；历史 77 篇审查没有被改写成覆盖新文的声明。
