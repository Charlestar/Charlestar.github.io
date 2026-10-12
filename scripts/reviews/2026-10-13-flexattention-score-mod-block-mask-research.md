# FlexAttention：调研与写作交接

日期：2026-10-13。本文解释可编程 Attention 的语义、编译与块稀疏路径；没有运行 CUDA/PyTorch/Triton kernel 或复现论文性能。

## 选题与编排

现有 FlashAttention/FA2 文章讲 IO 与 work partitioning，序列打包文章讲样本隔离和变长调用；没有独立讲解 `score_mod`、`mask_mod`、`BlockMask`、组合顺序与 decode 全局位置。新文不重复 FlashAttention 全部在线 softmax 推导或 packing loss 推导，链接相应现有文章。

root 已确认 `series: attention-long-context`、`series_order: 65`、`technology_year: 2024`。2024 对应公开编程接口与预印本时期，MLSys 正式论文为 2025；勿称 2025 才首次公开。

建议标题：`FlexAttention：Attention 规则怎样变成可跳过的 GPU 工作`。

## 实际阅读范围与接口快照

1. [PyTorch 首发官方博客](https://pytorch.org/blog/flexattention/)：实际阅读 score-mod/mask-mod、causal/sliding/document 示例及 FAQ 中缓存、重编译、full/partial blocks、旧限制。该页描述 2024 发布时期，后续不可沿用“尚不支持 GQA”“长度必须为 128 倍数”等旧状态作为 2026 通用结论。
2. [MLSys 2025 官方论文 PDF](https://proceedings.mlsys.org/paper_files/paper/2025/file/61a9278dfef5f871b5e472389f8d6fa1-Paper-Conference.pdf)：实际阅读 §1.1、§3.1–3.2、§4.1–4.2、§5.2、§6.1–6.2 的设置和性能限定。未完整阅读所有实验图或 artifact 附录，也未复现。正文短述编译模板与两类 API，主体用原创示意推导，不长译整篇或以作者实验值代替本地测量。
3. [PyTorch 2.8 FlexAttention API](https://docs.pytorch.org/docs/2.8/nn.attention.flex_attention.html)：实际阅读 `flex_attention`、`create_block_mask`、`BlockMask`、full/partial 的字段用途和 prototype warning。代码固定 2.8.0 合同，不称其为 2026-10-13 最新版。Q/K/V 的 shape 为 `[B,Hq,L,E]`、`[B,Hkv,S,E]`、`[B,Hkv,S,Ev]`；本文保守示例先用相同头数与 D=64，避免泛述硬件、dtype、GQA 支持矩阵。
4. [PyTorch v2.8.0 frontend 源码](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/attention/flex_attention.py)：实际阅读 `create_block_mask`、`create_mask`、`_create_empty_block_mask`。核对 bool True=允许、B/H 的 None 广播、block size、未编译构造路径先形成 element mask 再归约为 full/partial blocks。`_compile=True` 已有 deprecation 并建议 `torch.compile(create_block_mask)`，故不照抄首发博客旧建议。
5. [PyTorch v2.8.0 Inductor kernel 源码](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_inductor/kernel/flex_attention.py)：实际阅读性能参数注释、`compute_forward_block_mn`、non-divisible bound checks、safe softmax 和最终 `l_i==0` 处理。核对 QK 缩放后再 score_mod，partial blocks 最后 where mask 为 -inf，full blocks 仍执行 score_mod；ROWS_GUARANTEED_SAFE 只在每行至少一个可见值得到保证时可启用。只读这一版本的相应前向路径，不声称审计所有 backward/decoding/hardware 实现。
6. [PyTorch 官方 inference 后续博客](https://pytorch.org/blog/flexattention-for-inference/)：实际阅读 Working with KV Cache、query offset、BlockMask for Inference、paged logical/physical 映射。正文只用 offset 与缓存生命周期概念，不完整移植其 slicing 代码；网页示例中省略参数或旧 API 不能直接当固定版本可执行代码。
7. [FlexAttention 原始预印本元数据](https://arxiv.org/abs/2412.05496)：核对 2024-12 的公开年份；原始 API 发布文章更早在 2024。只作时间依据。

对单页面衍生摘要限制 200 words 内；正文算法表达、示例、计数、工程判断自行推导。未借助镜像博客或 AI 生成摘要替代论文正文。

## 原创示例与推导

### 把语义和可跳工作分开

对分数 `s=q·k/sqrt(d)`，`score_mod(s,b,h,i,j)` 修改 softmax 之前的值；`mask_mod(b,h,i,j)` 返回连接是否允许。

将非法连接 score 设为 0 会给它 `exp(0)=1`，不能删去它。硬 mask 用 -inf 或独立 bool mask。尤其不能先 `where(allow,s,-inf)` 再对全部结果做 softcap，因为 `c*tanh(-inf/c)=-c` 有限，非法连接重获非零概率。v2.8 kernel 在 score_mod 后应用 mask，与这种错误组合不同。

可用 `score_mod = c*tanh(s/c) + alpha[h]*(p_k-p_q)` 加独立 sliding-causal mask；这是明确教学组合，未声称某个已训练模型采用该确切组合或零成本替换 attention 规则。

### 教学 tile 计数

长度 6 的 causal self-attention 有 21 个允许的连接。在 6×6 分数矩阵上，按 2×2 示意块划分得到 9 块：

```text
P 0 0
F P 0
F F P
```

`0` 三块完全屏蔽，可跳；`F` 三块全可见，计算并执行 score_mod，但不需逐元素 bool mask；`P` 三块部分可见，需要元素 mask。访问 6 块，相当于 24 个 tile-cells，21 个真正可见连接。块级节省 12/36，元素 mask 删去 15/36，不能把元素 sparsity 直接当 GPU 时间缩减比例。2×2 是纸面解释，不是推荐 GPU kernel tile 或将 create_block_mask 参数设为 2 的可运行示例。

### 窗口、packing、位置

本篇窗口 W 定义为“最多 W 个 token，包含 query 自身”，允许 `0 <= p_q-p_k < W`，故 W=3、p_q=5 时可见 k=3,4,5。不要照抄官方旧例 `<= WINDOW` 并把它解释成包含当前的同一个 W；两种定义差 1。

packed lengths `[3,3]` 的 segment IDs 为 `[0,0,0,1,1,1]`，局部位置为 `[0,1,2,0,1,2]`。mask 至少有 `same_segment & causal & window`。只用整条物理 index 下三角会跨文档；本篇保持独立样本语义，若有意连续文档训练则属于另一数据合同。

位置重置不替代硬边界；attention mask 又不替代 shifted label/loss mask。本篇链接既有 packing 文章即可，不重写其全部训练等价性。

### decode 的索引反例

当前 query 全局位置为 5，Q 长度为 1，所以传入 mod 的 q_idx=0。KV 预分配容量 8，但有效 KV 为位置 0..5。

- 错误 mask `k <= q_idx` 只允许 k=0。
- 正确普通 causal 条件为 `k < valid_len=6` 且 `k <= q_idx+offset=5`。
- 正确 W=3 窗口再要求 `0 <= (q_idx+5)-k < 3`，只保留 3,4,5。

captured Tensor 的 offset 数值更新可避免把每步 offset 都变成新的 Python 常量 specialization，但这不自动修改已缓存 BlockMask 的 full/partial/skipped 元数据。mask structure 改变要重建/正确转换相关元数据；只更新 captured tensor 而复用旧结构可漏算新可见块。模板重编译、BlockMask 构造和数值更新是三种工作。

## 推荐文章结构

1. 自定义规则进入 Attention 时，为什么短 PyTorch 表达式还需高效执行路径。
2. 三阶段公式：scaled QK、score_mod 与 mask、softmax V；从一个连接解释 b/h/q/k 的索引含义。
3. 用一个 relative bias/softcap 示例讲 score_mod，并给操作顺序反例。
4. mask_mod 和 BlockMask 连接可见性与 tile 工作；用 6×6 示例解释三类块。
5. 自造 CUDA 示例基于固定 v2.8.0，torch.compile(flex_attention)，不改全局默认训练规则；明确代码未在当前环境 GPU 实跑。
6. 窗口 + packed segment 的组合，局部位置与样本硬边界。
7. decode offset 与有效 KV 长度，用单 query 反例解释为什么二维下三角直觉失效。
8. BlockMask 缓存/构造成本、编译缓存与动态结构变化；注意构造开销可抵消短序列收益。
9. 数值与全部 masked query，正确对齐 reference 前向/反向，benchmark 需计入 warmup、mask construction、steady state。
10. 联系 FA2、packing、PagedAttention；本文提供正确建模和接口理解，不承诺任意新规则性能或模型质量。

## 必须限定的工程结论

- 独立 score_mod 对非法位置产生 -inf 可给相同数学语义，但不足以承诺自动跳块。
- `BlockMask` 一般不会永久存储完整 L×S bool tensor；这不意味着朴素构造阶段从未生成 dense mask。固定 v2.8.0 未编译构造路径可见 dense 临时表示。
- 带 Tensor 数值变化不必每次重编译，是特定 capture/shape/guard 条件下能力，不是任意 Python control flow 皆无需编译。
- 默认 v2.8.0 相应 forward kernel 有安全路径处理全屏蔽行；普通手写 softmax(-inf...-inf) 仍未定义，不保证所有版本和后端相同输出。教学例确保所有有效 query 至少能看自己；无效 padding query 不参与 loss。
- 不启用错误的 ROWS_GUARANTEED_SAFE/BLOCKS_ARE_CONTIGUOUS 断言。它们表达调用者已证明的性质，不是模型学习出的性能开关。
- 切换已训练模型 attention 规则可能改变模型行为，FlexAttention 的数值语义实现不证明模型质量保持。

## 独立审核建议

前向 mask 与 score-mod 顺序；-inf 被 softcap 变回有限值反例；6×6 三类 tile 及 21/24/36 计数；W=3 边界；packing 首 token 隔离；decode offset/valid-length 反例；全 masked reference 行政策；更新 captured offset 却未更新 BlockMask 的反例。CPU arithmetic/reference 检查不作为 GPU 执行或编译路径验证。
