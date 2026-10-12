# FlexAttention 独立审阅记录

日期：2026-10-13。审阅对象为 `_posts/2026-10-13-flexattention-score-mod-block-mask.md`，接口合同固定为 PyTorch v2.8.0，不代表当日最新版本。

## 独立一手核对范围

下列资料由审核者直接打开相应正文或源码阅读，不仅依据调研交接摘要。范围限定为列出的段落和函数，没有宣称通读全部论文附录或审计全部 kernel。

- [PyTorch 2024 首发文章](https://pytorch.org/blog/flexattention/)：分数变换、布尔可见性、BlockMask、full/partial blocks 和缓存说明。旧文章中的限制与 `_compile=True` 建议不直接作为 v2.8.0 的现状。
- [原始预印本](https://arxiv.org/html/2412.05496)：编程模型、块稀疏和编译模板相关正文；核对 2024 公开时间与正式会议年份分开。
- [MLSys 2025 论文](https://proceedings.mlsys.org/paper_files/paper/2025/file/61a9278dfef5f871b5e472389f8d6fa1-Paper-Conference.pdf)：§1–3、§4.1–4.2 的接口和模板、full/partial block 说明、§5.2 的 decode offset，以及 §6.1–6.2 开头的实验设置与限定。没有将论文特定硬件结果改写为本地测量或通用性能保证。
- [PyTorch 2.8 API](https://docs.pytorch.org/docs/2.8/nn.attention.flex_attention.html)：`flex_attention`、`create_block_mask`、`BlockMask` 字段和 prototype 状态。
- [v2.8.0 前端源码](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/attention/flex_attention.py)：通过 raw view 阅读 `create_mask`、`create_block_mask`、相关块转换与广播、编译参数、CPU/prototype 条件。默认未编译构造路径存在 dense 元素 mask 临时量，最终紧凑元数据不说明构造阶段从未稠密化。
- [v2.8.0 higher-order 数学路径](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_higher_order_ops/flex_attention.py)：`_math_attention_inner`、`math_attention` 的 scale、score_mod、布尔 mask 和安全 softmax 次序。
- [v2.8.0 Inductor kernel](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_inductor/kernel/flex_attention.py)：性能选项注释与相应前向模板。QK 缩放后执行 score_mod，partial block 再用 `where(...,-inf)`；full block 仍执行 score_mod。核对全屏蔽行的 `m_ij` 保护与最终 `l_i==0` 处理。没有审核全部 backward/decoding/硬件分支。
- [官方 Inference 后续文章](https://pytorch.org/blog/flexattention-for-inference/)：short query、KV Cache、query offset、captured tensor 与 BlockMask 推理适配部分。不把网页省略参数的 slicing 片段当作本文固定版本的可执行示例。

正文的 score 组合、窗口定义、矩阵计数和坐标反例均为自行定义的教学推导；并不归属某个已训练模型，也没有复制论文实验表格。

## 全文审核与修订

1. 确认默认缩放先于 score_mod、硬 mask 最后排除非法连接、softmax 汇总 Value；没有把零分数误当屏蔽。
2. 确认 softcap-after-mask 会将 `-inf` 映射到有限 `-c`，因此非法连接重新获得非零权重。
3. 确认 full block 只省掉对应元素 mask，不省 score_mod；元素稀疏性与可跳块比例没有混同为 GPU 延迟比例。
4. 确认长度 6、教学 tile 2 的 3 full / 3 partial / 3 skipped、21 可见连接、24 访问 tile 单元格。没有把 tile=2 推荐为实际 GPU 配置。
5. 确认窗口包含自身，`0 <= p_q-p_k < W`，W=3、q=5 得到 k=3,4,5，没有差一错误。
6. 确认 packing 的 segment 条件、逻辑位置和 label/loss 边界各承担不同职责；没有把位置重置当作样本隔离。
7. 确认单 query 的 local index=0 与全局位置=5 的区别、valid KV length 与预分配容量的区别。没有把分页物理地址当作偏置的逻辑距离。
8. 确认 compiler cache、捕获值更新和 BlockMask 结构更新不是同一件事；旧 skipped 与 full 分类都可能因可见性变化变得错误。
9. 补充公式的有限分数前提和数值边界：mask 有 True 不保证自定义 score_mod 后还存在有限分数；全 `-inf`、NaN、正无穷须区分处理。
10. 补充 v2.8 API 的 prototype 状态；清理一处不必要的研究过程旁白。保留固定版本、接口片段未执行和测量范围等会影响使用的限制。

未发现需要另改的公式、窗口定义、示例输出或链接主题错误。

## 实际执行的验证

`node scripts/reviews/oct13-flexattention-examples.mjs`：11 组通过。

脚本直接从正文标记提取并执行 `softcap-order`、`block-counts`、`decode-coordinates` 三个标准库 Python 块，没有另写一份相同实现再把它当作正文验证。

- 实际 softcap 示例输出与独立 logistic 恒等式一致；非法位置的错误权重为 `1/(1+exp(5))`，不是零。
- 零分数仍分得概率；`-inf` 屏蔽连接。
- 正文 reference 对全屏蔽行明确报错；布尔可见性和有限分数条件区分。
- 实际块分类与使用区间端点的独立分类、三角计数一致。
- 独立补充 S=8/tile=4 反例：causal 36 个连接、W=3 有 21 个连接，却同样访问 3 个 tile、48 个 tile 单元格，元素减少不等于整块减少。
- packed `[3,3]` 的段身份与窗口联合得到 12 个连接；仅重置局部位置会保留跨段连接。
- 实际 decode 示例与另一种区间表达一致；不读容量中未填充的位置。
- 严格窗口边界和包含当前 query 的定义一致。
- 独立验证 captured offset 更新但仍复用旧 skipped 元数据会漏掉新连接。
- 独立验证旧 full 元数据变成 partial 后，继续省元素 mask 会放入非法连接。
- 所有正文 Python 块通过 AST 语法检查；PyTorch 调用片段的导入、B/H 广播、长度与 compile 入口做静态检查。

`node scripts/audit-posts.mjs`：89 篇文章、9 个系列、24 个标签的元数据与正文结构检查通过。

`git diff --check`：通过。

## 验证边界

以上运行不导入 PyTorch，不执行 CUDA、torch.compile、FlexAttention forward/backward，也不测量 kernel 速度、显存或模型质量。AST 与人工 API 合同对照不等于实际编译兼容测试。

## 主审交接后的页面验证

最终稿 production Jekyll 构建与生成页面检查通过（103 个 HTML、128 个文件）；原生公式保存检查通过（82 篇含公式文章、495 个行内公式、883 个块公式）。本篇在本机浏览器渲染 24 个公式，MathJax 错误为 0，页面宽度与滚动宽度均为 1265 px。已目视检查 scaled QK、分段 mask 与归一化输出公式的实际页面截图，没有整页水平溢出。线上发布另在提交推送后检查，不将本机预览等同于部署。
