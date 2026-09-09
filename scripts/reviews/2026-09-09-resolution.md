# 2026-09-09 评审处理记录

审阅输入：用户提供的 `blog-audit-2026-09-09.md`。核对基线：`bb5e3cc`，共 77 篇文章。

报告的 31 项意见均有修改价值，涉及 36 篇文章。逐项对照正文后，以原论文、官方文档、固定版本源码或独立数学反例确认，已落实修复。这里的“采纳”包括纠错、消除歧义和补充适用范围，不表示每项都属于算法错误。其余文章没有因本报告批量改写。

## 逐项判断与处理

| 编号 | 文章 | 判断与落实 | 主要依据 |
| --- | --- | --- | --- |
| R01 | 基础光流法介绍 | 公式错误：删除二次型左侧多余转置，补充矩阵维度及交叉项。 | [Farnebäck §2](https://www.diva-portal.org/smash/get/diva2:273847/fulltext01.pdf)、矩阵展开 |
| R02 | 推荐算法召回策略 | 概念错误：DR 改为分层离散编码空间，说明物品与路径多对多关联，不再称作物品树。 | [Deep Retrieval §1–2.1](https://arxiv.org/abs/2007.07203) |
| R03 | SmoothQuant | 数学条件遗漏：推导 alpha 两端及变换后幅度，说明绝对 scale 的变化依赖 a×b。 | [SmoothQuant §4](https://arxiv.org/abs/2211.10438)、数值反例 |
| R04 | FP8 推理 | 精度解释需要修改：区分正常数相对精度与下溢，替换无效例子。 | [FP8 表1](https://arxiv.org/abs/2209.05433)、可表示数枚举 |
| R05 | SpecForge、MoE 推测解码、llm-d、NVFP4、P/D 弹性、长请求治理、GPU 互联、结构化生成、Program-Aware Serving、多模态 Serving | 发布故障：正文原始反斜杠定界符被 Markdown 消耗。迁移为 Kramdown 原生数学节点，并增加源文件与构建产物检查。 | [Kramdown 数学语法](https://kramdown.gettalong.org/syntax.html#math-blocks)、[MathJax converter](https://kramdown.gettalong.org/math_engine/mathjax.html)、本地构建及浏览器 |
| A01 | GQA | 时间步错误：明确 prefill 输出 1001，下一次 decode 输入 1001、输出 1002。 | [Transformers 缓存循环](https://huggingface.co/docs/transformers/main/cache_explanation) |
| A02 | Sorting-Free Sampling | 采样错误：Top-k 后归一化，保留首次越过 Top-p 阈值的项，补充 p=1 和 ties 条件。 | [FlashInfer 算法](https://flashinfer.ai/2025/03/10/sampling.html)、[API](https://docs.flashinfer.ai/generated/flashinfer.sampling.top_k_top_p_sampling_from_probs.html) |
| A03 | SpecForge | 对齐错误：统一特征 g_t 与已选 anchor x_(t+1)，首个新增 draft 为 x_(t+2)。 | [EAGLE-3 §3.1](https://arxiv.org/html/2503.01840v2#S3.SS1) |
| A04 | EAGLE、DFlash | 必要条件过强：补充保留首次失配 target 样本的精确验证；只在 min(p/q) 接受规则下要求配套正残差。 | [Transformers v4.57.1](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/generation/utils.py)、[推测采样证明](https://proceedings.mlr.press/v202/leviathan23a.html) |
| A05 | FlashAttention-2 | 定义遗漏：补齐 D 的逐行归约、形状、求和轴及广播，说明 score 缩放的梯度。 | [FA2 Algorithm 2](https://tridao.me/publications/flash2/flash2.pdf)、有限差分 |
| A06 | Ring Attention | 布局错误：Ulysses 在 attention 中使用完整序列、部分 heads，之后恢复序列切分。 | [DeepSpeed Ulysses 设计](https://github.com/deepspeedai/DeepSpeed/blob/master/blogs/deepspeed-ulysses/README.md) |
| A07 | FlashAttention-3 | 符号有歧义：round 改为 E4M3 cast，说明 scale、反量化及下溢。未将原文歧义误判为实现已经执行整数舍入。 | [NVIDIA FP8 primer](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html) |
| A08 | KV Cache 基础 | 容量口径错误：拆分滑动窗口和动态稀疏 attention，说明 DSA 通常仍需全历史主 KV 和索引 cache。 | [DeepSeek-V3.2-Exp 参考实现](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py) |
| A09 | DeepEP | API 单位错误：参数限制 source 输入 token 上界，destination 去重接收和 expert 展开量另外计算。 | [DeepEP 01dc3aa](https://github.com/deepseek-ai/DeepEP/blob/01dc3aaac82068020353dce2c302e38153c0bfaa/csrc/elastic/buffer.hpp) |
| A10 | DFlash | 计数错误：正常轮提交数为接受草稿数加 1；EOS/长度截断改用实际提交数，不重复计算 anchor。 | [DFlash §3.1](https://arxiv.org/html/2602.06036v1#S3.SS1)、成本反例 |
| S01 | LangGraph + MCP | 默认语义缺失：同一步写默认单值字段会报并发更新错误；完整 schema 中直接声明 reducer。 | [LangGraph 错误说明](https://docs.langchain.com/oss/python/langgraph/errors/INVALID_CONCURRENT_GRAPH_UPDATE) |
| S02 | LangGraph + MCP | 图文不一致：补回检索环、次数耗尽分支及审批后的 send_report。 | 本文业务约束与图路径核对 |
| S03 | Model Runner V2 | 术语错误：改为 Unified Virtual Addressing。 | [CUDA Unified Addressing](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__UNIFIED.html) |
| S04 | Continuous Batching | 计数错误：decode 轮数为 max(O−1,0)，表格、公式和时间线同步更新。 | [NVIDIA 两阶段说明](https://developer.nvidia.com/blog/streamlining-ai-inference-performance-and-deployment-with-nvidia-tensorrt-llm-chunked-prefill/) |
| S05 | 推理容量规划 | 成本模型错误：使用 cached/uncached 二维 profile，保留后缀读取历史的 H×U 成本。 | [FlashAttention 非方形 causal mask](https://github.com/Dao-AILab/flash-attention#flashattention)、枚举计数 |
| S06 | NIXL | 并发口径错误：区分单笔速率和成功 payload / wall-clock 窗口；说明跨窗口计数、稳态与 drain。 | 并发传输反例及指标定义 |
| S07 | NIXL、Dynamo | 版本范围过强：保留语义兼容，允许版本明确支持的受限 block 重映射/布局转换；不推广到所有引擎。 | [NixlConnector 兼容矩阵](https://docs.vllm.ai/en/latest/features/nixl_connector_compatibility/) |
| S08 | V1 EngineCore、Chunked Prefill | 模型范围遗漏：保留经典物理块解释，补充 prefix_match_unit 与各 KV group 状态恢复边界。 | [vLLM CacheConfig](https://docs.vllm.ai/en/latest/api/vllm/config/cache/#vllm.config.cache.CacheConfig.prefix_match_unit) |
| S09 | 长请求治理 | 定义域缺失：仅对正预算计算 urgency，过期请求先走取消、降级或独立队列。 | 零分母/负预算反例 |
| S10 | 长请求治理 | 状态区间缺失：补 Emergency，并优先检查 allocator 实际可分配性。 | 水位边界穷举 |
| S11 | Program-Aware Serving | 量纲错误：统一货币等价成本，定义 GPU·秒、byte·秒与传输资源单价；明确这是博客示意模型。 | [InferCept 作者说明](https://mlsys.wuklab.io/posts/infercept/)、量纲核对 |
| S12 | DistServe | 图示错误：TTFT 截止首 token，E2E 截止最后 token；补 TPOT 的 N>1 条件。 | 本文时间定义 |
| S13 | Multi-LoRA | 等价关系过强：KV 内容缓存按身份隔离，执行图按拓扑、shape、地址与 LoRA 模式兼容性复用。 | [vLLM v0.25.0 descriptor](https://docs.vllm.ai/en/v0.25.0/api/vllm/v1/worker/gpu/cudagraph_utils/)、[PyTorch CUDA Graph](https://docs.pytorch.org/docs/main/notes/cuda.html#cuda-graphs) |
| D01 | ZeRO/FSDP | 通信顺序错误：普通混合精度路径先转换本地 shard，再按 param_dtype AllGather；与 optimizer 持久精度分开。 | [FSDP2 MixedPrecisionPolicy](https://docs.pytorch.org/docs/main/distributed.fsdp.fully_shard.html#torch.distributed.fsdp.MixedPrecisionPolicy) |
| D02 | Collective Communication | 计时边界错误：区分 API、Queue、GPU kernel 和下游暴露等待，不再混算。 | [CUDA Stream API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html)、[Nsight Systems](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html) |
| D03 | RLHF/GRPO | 摊销与实际量混用：平均项为 C/k，单轮实际项另外定义；异步只统计关键路径增量。 | [DCP 异步保存](https://docs.pytorch.org/tutorials/recipes/distributed_async_checkpoint_recipe.html)、算术验证 |

交叉复核另修正了长请求文章的相邻 KV 计数公式：在普通自回归每次 forward+sample 完成后，刚输出的最后一个 token 还没有 KV；区分有效 KV、下一步预留和物理 block 占用，消除与 A01/S04 的矛盾。

## 数学发布与可复算验证

R05 在基线共有 146 处原始行内表达式。SpecForge 重写对齐段落时替换其中 11 处，剩余 135 处采用 Kramdown 原生行内 `$$...$$`。该语法让公式成为 Markdown 数学节点，输出时再生成保留完整 TeX 的 MathJax 定界符；没有仅靠修改浏览器的定界符配置来掩盖构建故障。

- `node scripts/audit-posts.mjs`：检查 77 篇的元数据、系列、标签、公式围栏及不安全原始定界符。
- `node scripts/reviews/2026-09-09-examples.mjs`：12 组零依赖数学例子，包含概率枚举、矩阵乘法、FP8 可表示值、causal attention 对数和调度边界。
- 完整 Jekyll production build 后运行 `node scripts/validate-site.mjs` 与 `bundle exec ruby scripts/validate-math.rb`，后者将源文件的 Kramdown 数学节点与实际生成 HTML 逐式比较。
- LangGraph 文内 Python reducer 还经过交换/结合、重复更新、冲突检测与 Annotated 元数据检查；未将这些检查表述为完整工作流运行。
- 上述源文件、数值与生成产物检查均接入 GitHub Actions，在部署前执行。

本地实际结果：77 篇源文件审计通过，12 组数学例子通过，重新生成的 90 个 HTML 页面和 115 个文件通过完整性检查；70 篇数学文章中的 140 个原生行内公式与 770 个块级公式和生成产物一致。浏览器逐页打开 R05 的全部 10 篇，均有实际 MathJax 节点且没有数学错误节点；NVFP4 的 20 个行内公式从原先裸露的 TeX 文本恢复为排版结果。随后修改的长请求 KV 公式也重新构建并在浏览器复核。

本次事实复核与数学例子不等于真实 GPU 集群性能复现。版本相关结论按文中指定的 release/commit 或 2026-09-09 所核对官方文档限定；单美元旧式数学未纳入 Kramdown 原生节点比对，浏览器排版另行检查。
