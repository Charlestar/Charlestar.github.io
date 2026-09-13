# 序列打包与变长 Attention 独立最终审核记录

- 审核日期：2026-09-13。
- 正文：`_posts/2026-09-13-sequence-packing-variable-length-attention.md`。
- 角色与范围：writer 停笔交接后，独立完整阅读正文，不仅审阅 diff；核对年份/版本、全部段落的语义前提、公式、图表计数、Python 接口与 labels 例子，并补修。构建、浏览器、提交与部署由主进程另行验证。
- 结论：以下补修与 CPU 参考检查完成后，未发现尚未处理的发布阻断项。本文是语义与固定接口分析，不是 FlashAttention 或分布式训练实测报告。

## 实际独立阅读的一手来源

1. [论文元数据](https://arxiv.org/abs/2107.02027)：核对首版 2021-06-29、所读 v2 为 2022-10-05，未将 2021 写成 packing 发明年份。[论文 v2](https://arxiv.org/html/2107.02027v2)实际阅读引言、§3.1 的装箱概念、§3.2 全部（位置、Attention mask、per-sequence loss）、§3.3 的更新窗口/超参数讨论、Appendix A 的适用假设及 T.2–T.6（位置、mask、loss、值/梯度测试、loss balancing）。未声称通读求解器证明和全部硬件实验附录；因果 decoder 的 shifted labels 与当前 varlen 接口不是原文 BERT 代码的直接结论。
2. [FlashAttention v2.7.4.post1 Python 接口](https://github.com/Dao-AILab/flash-attention/blob/v2.7.4.post1/flash_attn/flash_attn_interface.py)：阅读完整 `flash_attn_varlen_func` 的签名、文档和转发，以及 `maybe_contiguous` 与 `_flash_attn_varlen_forward` 的相关路径。核对 flat Q/K/V、前缀数组长度与 int32、max_seqlen、dropout 和非方形 causal 的右下对齐约定。预审曾参考 v2.8.3，但最终已补读正文指定 v2.7.4.post1，没有用前者代替版本核查。
3. [同版本 flash_api.cpp](https://github.com/Dao-AILab/flash-attention/blob/v2.7.4.post1/csrc/flash_attn/flash_api.cpp)：阅读 `mha_varlen_fwd` 的参数检查、设备 guard、dtype/stride/head dimension 条件、前缀指针传递和执行分支。核对该 CUDA 路径的连续 int32 前缀要求；不将 shape/dtype 检查等同于前缀内容和真实样本边界的验证。
4. [PyTorch v2.5.1 functional.py](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/nn/functional.py)：阅读 SDPA 的接口说明/参考算法中布尔 mask 的 True=允许、因果 mask、dropout 与空行相关行为边界，以及 `cross_entropy` 的 target/ignore_index 说明。[loss.py](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/nn/modules/loss.py)阅读 `CrossEntropyLoss` 的 class-index/概率标签、加权归约、ignore_index、label smoothing 和形状说明。
5. [Transformers v4.46.3 loss_utils.py](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/loss/loss_utils.py)：完整阅读文件，重点复核 `fixed_cross_entropy` 的 sum/num_items 路径与 `ForCausalLMLoss` 的 logits 前移/labels 后移、FP32 logits 和设备对齐。正文主动限定其内部 shift 合同，不推广为所有模型默认行为。
6. [RoFormer v5 §3.2](https://arxiv.org/html/2104.09864v5#S3.SS2)：阅读二维旋转、一般块对角旋转和相对内积公式，核对固定频率共同偏移抵消。[Transformers v4.46.3 Llama](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/llama/modeling_llama.py)阅读 `LlamaRotaryEmbedding` 的初始化、`_dynamic_frequency_update` 与 `forward`；确认动态路径依赖最大 position，并非仅重置 position 就无条件等价。
7. [PyTorch v2.5.1 DDP](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/nn/parallel/distributed.py)：阅读 DDP 类说明中的平均梯度/sum loss、通信合同，以及完整 `no_sync` 说明和上下文实现。核对前向必须进入 no_sync、默认平均下 W/N 的推导条件；未执行 collective 或验证实际训练调度。
8. [Hugging Face Fixing Gradient Accumulation](https://huggingface.co/blog/gradient_accumulation)：阅读问题来源与 sum/全窗口有效项分母修复，区分 mean of means；不把 2024 年发布时的修复承诺当作所有现版本 Trainer 的行为保证。

## 审核中实际修改

- 将“flat 的第 3 个位置”改为 `flat[3]（下标从 0 开始）`，消除 B 首 token 与第三个元素 EOS 的歧义。
- 补充 cu 前缀值及总 token 数必须能用接口的 int32 表示；保留严格递增、末项/总数、max 与真实逻辑段一致的合同。
- 明确 loss 片段的 `input_ids` 为 `torch.long`，与 logits、cu 同设备，所有输入由调用方预先准备，示意并非完整模型。
- 新增仓库 CPU 复算命令及边界，不将 JS 检查声称为 FlashAttention、DDP 或完整模型参数梯度验证。
- 全文复核但未改动的核心结论：独立样本与连续流的不同目标、block-diagonal causal 隔离、EOS 不是 mask、softmax 前保护、RoPE 不代替 mask、首 label 与末 logit 两种 shift 坐标、保留 EOS 终止监督、SFT prompt 的 loss/Attention 区别。
- 重算 token/sample 与默认 DDP 平均的全窗口分母，确认 W/N 不可再除 G；检查本地零监督/global zero 分支以及无跨样本耦合的等价前提。重算 12/9、48/29/81、19/45/26 的计数，未将比值当作实际加速。

## 已执行检查

在仓库根目录运行 `node scripts/reviews/sep13-sequence-packing-examples.mjs`，最终 10 组通过：

1. 前缀半开边界、主例 `[0,3,5,9]`；拒绝非零起点、重复/逆序、末项不符、max 不符和超 int32。另验证 `[0,4,5,9]` 虽结构合法仍会错误分配 `flat[3]`，不能靠结构校验替代样本身份。
2. 非平凡 Q/K/V 的两条独立 causal Attention 与 block-diagonal packed Attention 逐元素相等。
3. 单一全局 causal mask 的跨样本污染；正文 A 的 V 全 1 时，B 首输出从 10 变为 3.25。
4. 只改 A 的 V，隔离后的 B 输出不变，而全局 causal 路径改变。
5. 自行推导的 masked softmax Attention Q/K/V 解析梯度，逐元素与 packed 和独立两条路径的中央有限差分对照；这是 CPU attention 层数学参考，不是完整模型参数梯度实测。
6. 内部 shift 的段首 label 忽略与外部 next-label 的段末忽略等价；主例恰好 6 个 target，并保留 3 个 EOS target。
7. 不等长样本/micro-batch 的 token mean 与 sample mean：包含正文 2.5 对 2 及另一组 4.5 对 6。
8. 默认 DDP 平均的 W/N 数值参考：不等 rank、本地零监督、全窗口按 micro-batch 分解仍得相同目标；未运行 DDP。
9. 固定二维 RoPE 共同偏移内积不变，而仅重置 position 不能替代段间 mask。
10. 两组长度分布的 tokenwise、平方槽位、causal pair 分开计数，包括正文 `[3,2,4]` 的全部表格数字。

## 未执行与限制

- 未安装或运行 PyTorch、FlashAttention、CUDA、DDP/NCCL；未执行正文 Python 片段或构建真实 decoder。
- 未验证 GPU 前后向、分布式 hang/空 rank 的框架实现、动态 RoPE 实际模型运行、dropout 随机映射、MoE、收敛、吞吐或显存。有限差分和纯 JS 归约不能替代这些检查。
- 未实现或 benchmark 论文的装箱求解器；本文关注已有逻辑样本的存储与训练语义合同。
- 仅按上述实际阅读章节/函数作来源核查，不宣称通读全部论文附录、源码仓库或硬件证据。Jekyll/浏览器/部署检查由主进程独立记录。

## 主审发布前复核与本批回归

主审完整阅读正文和最终补修，另核论文 §3.2、Transformers v4.46.3 的实际 shift/sum-count 实现、FlashAttention v2.7.4.post1 varlen 签名与 causal 约定。两个 Python 片段以 UTF-8 输入通过 AST 解析，仅验证语法，不执行框架。

最终三篇正文停止编辑后，执行全仓检查：83 篇 metadata/taxonomy/系列顺序检查、13 个 JavaScript 套件（119 组，其中本批新增 27 组）、Python agent 示例、生产模式 Jekyll 构建、97 个 HTML 页面/122 个生成文件、本地链接与原生数学保留检查，以及 `git diff --check`，均通过。数学检查保留 422 个行内、849 个显示公式，覆盖 76 篇有公式的文章。历史 coverage 仍只陈述原 77 篇审核快照，本批三篇由各自的新记录说明实际全文审阅范围。

通过 computer-use 检查本篇本地真实页面：45 个 MathJax 容器，零数学错误节点、无页面横向溢出；目视检查标题、工作量表格、分布式分母公式和标签位移代码。检查使用默认桌面视口，未声称全设备测试。

本篇提交之前，本批前两篇已各自提交推送，并核实线上：`e864017` 的 [Jekyll 发布运行](https://github.com/Charlestar/Charlestar.github.io/actions/runs/34733589930)与 `699dbe3` 的 [Jekyll 发布运行](https://github.com/Charlestar/Charlestar.github.io/actions/runs/34733808151)均成功；线上混合精度页 52 个、Triton 页 49 个 MathJax 容器无错误，且分别包含 `fused=False` 与乘法前 int64 索引修订。第三篇的线上状态只能在本次提交后另行验证，不把发布前检查写成已部署。
