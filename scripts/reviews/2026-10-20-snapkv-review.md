# SnapKV 独立审核

日期：2026-10-20。对象：`2026-10-20-snapkv-observation-window-kv-selection.md` 全文、对应 research 与 `snapkv-examples.mjs`。调研、写作与本次独审分别完成；本记录不代替主代理的构建、浏览器与发布验证。

## 实读来源与公式核验

独立读取论文 v2 的 §3–5，包括观察数据、算法 Listing 1、pooling 消融和实验表；arXiv 提交历史确认 v1 为 2024-04-22、v2 为 2024-06-17。没有把论文的任务平均表现扩为每个任务无损，也没有引用其速度倍数作为本地测量。

独立完整读取官方固定 commit `e216ddc84c5bd210378cbdbbba12ba02102aa640` 的 `snapkv_utils.py`、`llama_hijack_4_37.py` 和 README，并读取 PyTorch 2.8 avg_pool1d/topk 定义。核验结论：

- 观察 query 对完整 K 打分；只在尾窗局部构造未来位置 mask；完整 softmax 完成后才切 prefix 求票。源码 softmax 用 FP32 后转回 query dtype，CPU 教学例不复刻此低精度过程。
- Pooling 先改变分数，再选 C-W 个 prefix 位置；不会因“保留邻域”扩预算，也不保证连续片段。固定实现默认 avgpool，论文部分实验 maxpool；平均池化的补零计入分母。CPU maxpool 对非负票数只取有效邻居，与该域内的边界语义相符。
- 从 Q/K 点积构造的表格、25/26 两行分母、六个票数及平均池化小数与 exact Fraction 复算一致；反例的 .315/.685 对 1.15/.85 确实翻转选择。
- 各层/head 分别筛选，K/V 同索引；独立选择不能在未经设计的情况下仍使用原生 GQA 共享缓存。单 query 读取全部保留历史的同步 K/V 置换是求和置换，不授权重新编号 RoPE 或改写多 query 因果 mask。
- 对固定 Q/K/V，完整输出是 m O_keep + (1-m) O_drop；所选范数下 V 有统一上界时，误差不超过 2 Vmax (1-m)。只在 0<m<1 定义两组内部归一化均值，没有将它误作未来 m、hit rate 或模型质量保证。

## 接入与容量核验

固定 Llama 接入先应用 RoPE，再 repeat_kv；压缩 K/V 交给 cache，但当前 prefill attention 的局部变量仍是完整 K/V。压缩和完整 attention 在同一个 prefill forward 中发生，源码不是“当前完整 attention 完成后才裁剪”。Decode 追加新 K/V，不按固定 C 每步重新裁全部历史；逻辑 kv_seq_len 与实际 compact cache 长度分开。

持久缓存应按此接入的 Hq 计数。长提示词压缩分支为 C+g，短提示词旁路为 L+g，统一 min(L,C)+g；短提示词旁路也没有撤销 head 展开。g 是已 forward/已追加的 token，不包括刚采样但还未处理的 token。文中 2112/768/192 MiB、4/11 与 1/11 逻辑比值正确，第三行假设共享设计没有冒称官方收益。

观察打分的 BHqWL 个 score 和 BHqWLD 点积量纲正确；32 heads、W32、L16384 的单份 FP32 score 张量为 64 MiB。它不是整个峰值显存；仍有完整 prefill K/V、转换/softmax/gather 等临时成本。源码重写传入 mask，README 的历史版本测试范围不能泛化到任意 padding、packed/ragged、chunked prefill 或当前引擎。

## 修正

1. 开头的“留下重要位置及其邻域”改为池化使选择偏向邻近信息，避免先给出邻居必留的保证再在后文撤回。
2. Prefill 章节标题改为“当前 prefill 仍读完整 K/V，压缩缓存服务后续 decode”，消除时间顺序歧义。
3. 缓存增长及 GQA 公式补齐短提示词旁路，以 min(L,C)+g 统一，另明确短提示词仍有 head 展开。
4. CPU 教学域增加输入与点积累加须处于有限 Python 浮点范围的说明，溢出拒绝不伪装成完整数值后端。

## 独立测试和实际执行

作者已有的表格 Fraction oracle、双 head、因果遮挡、完整 softmax 反例、budget/旁路、tie-break 与 K/V 配对测试保留。独立新增：

- 120 组固定随机种子的正整数 exp(logit) 权重，配合三个 pool 宽度与 avg/max 两种模式，共 720 次 Q/K 到选择或短提示词旁路调用。以原整数权重构造 exact Fraction 概率、显式补零池化及 top-k 阈值 oracle，不调用正文的 softmax/vote/pool/gather 函数。精确同分仅检查合法选集，不拿实数 ties 冒充浮点/GPU 稳定次序。
- 4374 组 exact Fraction 概率、标量 V 与非空真子集，验证输出分解、误差恒等式和范数界，包括上界可取等的情况。它们独立于选择器，不证明未来 query 会把质量留在所选集合。
- 二维 RoPE 数学 toy：保留原位置编码且 K/V 同步置换，单 query 输出不变；按 compact 槽位重编码则一般改变输出。只验证代数区别，不执行 Transformers 历史 cache API。
- GQA 短提示词仍展开、g4096 容量交叉点和 g8192 反超原生完整缓存的反例；超大整数与有限输入点积溢出的拒绝测试。

实际执行 `node scripts/reviews/snapkv-examples.mjs`，15 个 CPU 语义组通过；101 篇的 `node scripts/audit-posts.mjs` 元数据/结构/标签/系列与 `git diff --check` 通过。穷举和随机配置数量不等同独立人工审阅数量。

未导入 Torch、运行官方 GPU 实现、加载 checkpoint、执行模型质量评估或复现论文速度/显存实验。未发现剩余阻止发表的概念、公式或教学代码合同问题；构建、页面、匹配提交的 CI 及线上最终正文由主代理分别验证。

## 主代理最终本地检查与发布边界

三篇逐一完成独立审核后，主代理实际执行全部 35 个 JavaScript 专项、既有 Python 检查、101 篇元数据与历史覆盖范围检查，均通过。三篇新增专项分别为 9、12、15 个 CPU 语义组，不将其扩为全部文章的语义正确性证明。生产构建生成 117 个 HTML 页面、142 个文件；94 篇数学文章中的 1534 个原生行内、1055 个独立公式通过结构核对。

浏览器逐篇读取本地最终构建：Profiling 2、RMSNorm 81、SnapKV 76 个 MathJax 容器，均无 MathJax 错误、正文双美元残留或整页横向溢出；关键公式另做目视检查。系列页实际读取 GPU Runtime 的 9 篇和 KV Cache 的 7 篇，三篇新文均位于预期技术年顺序。未改变旧文章、框架或已有 URL。

前两篇已分别提交推送 `5d4e26d31d7feab1b093470bae81256c1f5bcdc9`、`0bfd80ec44667162b3a0f53812a4be5d887c6aaf`。截至 2026-10-20 00:26（Asia/Shanghai），匹配的主发布工作流 `39284002002`、`39284853432` 仍为 queued，不能据此宣布线上发布完成。本篇随本记录单独提交，提交后的 CI 和线上正文仍须另行确认；上述本地检查不代替部署证据。
