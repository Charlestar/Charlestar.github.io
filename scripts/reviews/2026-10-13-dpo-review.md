# DPO 独立审阅记录

日期：2026-10-13。审阅对象为 `_posts/2026-10-13-dpo-preference-optimization.md`。数学范围是原始 sigmoid DPO、固定 Reference、无 label smoothing 和混合损失；实现对照固定 TRL v0.23.1，不称其为当日最新版本。

## 独立一手阅读范围

审核者直接阅读相应论文正文和源码，而非只依据调研交接笔记。

- [论文版本元数据](https://arxiv.org/abs/2305.18290)：首次公开时间为 2023-05-29；所读正文为 v3，而非把后续版本日期当作技术首次出现时间。
- [DPO v3 正文](https://arxiv.org/html/2305.18290v3)：§3、§4 的 Eq.3–7、主要梯度与数据流程，§5.1 的奖励等价类，Appendix A.1、A.2、A.4 和 B 中的相关推导/代码。没有声称通读全部实验或复现 PPO 对比结果。
- [作者 `trainers.py`](https://github.com/eric-mitchell/direct-preference-optimization/blob/main/trainers.py)：通过 raw view 阅读 `preference_loss`、`_get_batch_logps` 和相关输入拼接。此作者链接为读取当日的 main，不作为固定软件兼容配方。
- [TRL v0.23.1 文档](https://huggingface.co/docs/trl/v0.23.1/en/dpo_trainer)：数据合同、sigmoid/default 与其他变体的区分、logged metrics、reference 预计算与相关配置。
- [TRL v0.23.1 数据格式](https://huggingface.co/docs/trl/v0.23.1/en/dataset_formats)：标准/对话偏好记录、explicit/implicit prompt、模板转换。
- [TRL v0.23.1 Trainer 源码](https://github.com/huggingface/trl/blob/v0.23.1/trl/trainer/dpo_trainer.py)：通过完整 raw view 阅读 `_prepare_dataset`、`tokenize_row`、普通 `concatenated_forward` 的拼接、截断、mask、shift 和 sum；`dpo_loss` 的原始 sigmoid 路径；`compute_ref_log_probs`、reference 缓存列与 Adapter 上下文；初始化的 dropout 处理。没有据此宣称审计了全部 Liger、视觉、多损失和分布式实现。
- [TRL v0.23.1 Config](https://github.com/huggingface/trl/blob/v0.23.1/trl/trainer/dpo_config.py)：beta、reference_free、label_smoothing、长度与截断、预计算、disable_dropout、IPO/LD-DPO 等变体边界。

## 一手来源也进行交叉核对

- v3 Appendix A.4 Eq.21 的 winner/loser 顺序与正文 Eq.7 相反，后面的梯度式也不应直接照抄。正文以主公式、作者执行代码和直接微分为准；数值有限差分另行验证梯度符号。
- 作者与 TRL 源码的 beta→0 注释不能推成“退化 SFT”。对固定有限 log-ratios，beta=0 的实际损失为 ln2，梯度为零；reference_free 是不同设置。
- TRL v0.23.1 的 `tokenize_row` 是分别编码 prompt/chosen/rejected 再拼 IDs 的合同，没有将它误写成必然整体编码原始文本。
- 默认 completion 归约为 sum；IPO、LD-DPO、其他 weighting/mixed losses 不在本文原始算法范围内。

正文以自行构造的有限分布、技术问答和 token 行说明算法，不复制论文实验表格或文档模型训练配方。

## 全文审核与修订

1. 确认 KL 方向为 policy 相对于 Reference，有限分布最优解与 KL identity 符号正确；奖励常数仅在同一 prompt 下抵消。
2. 收紧推导前提：有限回答集合是本节抽象模型的整个支持集，p/q 在其上归一化，q 严格正，奖励固定且有限。明确不是把真实模型候选子集的原始概率直接当作完整分布；零概率项按连续极限定义。
3. 确认 delta 为 policy odds 减 reference odds，loss 为 softplus(-beta*delta)，导数为 `-beta*sigmoid(-beta*delta)`。
4. 确认 `[0.4,0.4,0.2]` 到 `[0.3,0.1,0.6]` 是合法归一化分布反例，且未声称某次独立三分类 SGD 必然沿此轨迹。
5. 确认 beta 的最优分布作用、单样本梯度和实际训练 KL 不能混为单调保证；beta=0 不等于 SFT。
6. 修正标量算例的范围：beta=0 显式返回恒等边界，避免 `0*overflow`；有限输入仍可能产生非有限 delta/margin，现明确拒绝。稳定 softplus 不被当作处理一切浮点溢出的生产框架。
7. 确认 completion log probability 是条件概率链式分解的 sum，不假设 token 独立；teacher forcing、首个目标移位、prompt 可见但不计回答分数、EOS/PAD 同 ID 的位置 mask 均一致。
8. 收紧“四个数”用语：四个值决定一条标量 pair loss，但 policy 分数还须保留模型梯度；清理一处不必要的作者过程旁白。
9. 确认原始字段相同不保证最终条件相同，截断可消除偏好或改变两份回答保留的上下文。
10. 确认 Reference cache 必须绑定权重、模板、IDs、位置/可见性、completion mask、截断与数据版本；补充关闭随机 dropout 的评分条件。不声称 TRL 自动校验全部缓存身份。
11. 确认 reward accuracy/margin 与事实正确率、绝对生成概率有区别；补近邻日志定义链接，避免指标定义无出处。

未发现需要另改的核心损失、梯度符号、算例数字或内部链接。

## 实际执行的验证

`node scripts/reviews/oct13-dpo-examples.mjs`：18 组通过。

脚本直接从正文标记提取并执行 `loss-and-gradient`、`sequence-logprob` 两个 Python 标准库块，数学检查使用独立概率恒等式、有限差分与反例。

- 实际 pair 输出为 `0.693147 0.287682`、导数 `-0.25`，与 ln2、ln(4/3)、ln3 的独立结果一致。
- 合法三回答分布可以降低 pair loss，同时降低 chosen 绝对概率；偏好支持 0.75 不等于生成 chosen 的 0.3。
- Reference odds 非 1 时，减 reference 的方向正确；交换 pair 符号相反。
- 28 个 beta/delta 组合用中心差分核对 policy score 与 reference score 两种符号，没有照抄梯度式充当验证。
- margin 为 ±1000 时稳定函数不发生指数溢出；非有限参数、delta 或 margin 明确拒绝。
- beta=0 包括原本会 overflow 的 delta 输入，仍返回 ln2/零导数/零 margin；不误当 chosen-only SFT。
- beta 在 delta=0 与 delta=5 的梯度权重符合正文数字，不推成一致方向。
- 1,326 个归一化三元素分布（含零质量边界）复算 KL identity 和最优性上界。
- 奖励重构、prompt-only 常数、最优分布与 Bradley–Terry 概率相互一致。
- 实际序列块与另一种 shifted-label/product oracle 一致，完整回答概率为 0.1、log probability 为 ln0.1。
- EOS/PAD 同 ID 按位置保留真正 EOS；错误的按 ID 删除、漏 shift 或加入 prompt 分数都会改变回答事件。
- 独立变长 completion 反例：sum 的 margin 为 +0.1，mean 的 margin 为 -0.1，归约变化会反转排序。
- 相同确定性回答的两分支共同依赖参数，有限差分验证梯度抵消，而非误把两个独立 scalar 参数分别求导。
- 截断删除唯一差异和 keep-end 导致不同保留 prompt 均有具体序列反例。
- 即使 target 文字相同，改变条件也可能令固定 Reference 概率不同，旧 cache 不再合法。
- 所有正文 Python 块通过 AST，技术问答为有效 JSON。

`node scripts/audit-posts.mjs`：90 篇文章、9 个系列、24 个标签的元数据和正文结构检查通过。

`git diff --check`：通过。

## 验证边界

以上未执行真实模型前向、DPO 训练、TRL、GPU、评审 API 或生成质量评测。有限集合、标量 log probabilities 和给定 token 概率不证明优化收敛、真实模型 KL 或输出事实正确性。

## 静止稿移交后的整站与页面检查

Root 在审核稿停止修改后执行了完整检查：

- 所有 20 个 JavaScript 审核算例套件通过；既有 Python 全套算例也通过。
- 生产模式 Jekyll 构建成功；104 个 HTML 页面与 129 个生成文件的站点校验通过。
- 原生数学语法检查覆盖 83 篇生成文章，538 个行内公式与 896 个独立公式，通过。
- 本文的本地浏览器页面实际生成 56 个 MathJax 公式容器，0 个渲染错误；页面宽度与内容宽度均为 1265 像素，没有整页横向溢出。
- 实际查看了奖励重构、Bradley–Terry、相对 margin 与 DPO loss 的页面截图，公式、段落与来源链接排版正常。

这些是离线算例、构建与本地页面证据，不代表真实模型训练结果。发布须在提交推送后另外核对对应提交的工作流和线上 URL，本记录不预先声称部署成功。
