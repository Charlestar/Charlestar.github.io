# 全文复核：基础、推荐与早期推理文章

基线 `8354568`。以下 20 篇均重新完整通读，不以旧报告或关键词扫描替代全文审阅。核查日期：2026-09-09。主要检查定义、公式、量纲、时序、代码/图文的一致性；引用资料是实际核对的关键章节，并不表示对每篇参考文献全文、全部实验或硬件行为做了复现。

| 文件 | 本轮结论与核查点 | 关键依据 / 边界 |
| --- | --- | --- |
| `2021-03-19-C语言可变参数的函数.md` | 未发现新增需修正问题。检查默认提升、va_list 复制、越界读取、类型化哨兵及 C23 va_start。 | [N1570 §7.16](https://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf)、[N3096 §7.16.1.4](https://www.open-std.org/jtc1/sc22/wg14/www/docs/n3096.pdf)；错误示例是明确的反例，不执行其未定义行为。 |
| `2021-06-08-C语言中函数参数的传递.md` | 未发现新增需修正问题。值传递、指针副本、二级指针与省略号位置一致。 | [N1570 §6.5.2.2、§6.9.1](https://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf)；不把指针传值称为语言级引用传递。 |
| `2022-03-16-树莓派4B使用经验.md` | 未发现新增确定性错误。保留已明确限定为 2022 年经历的系统、VNC、相机和安装记录。 | [64-bit 官方发布](https://www.raspberrypi.com/news/raspberry-pi-os-64-bit/)；没有该历史镜像/硬件，无法复现个人故障或证明其因果关系，不将历史 workaround 当当前建议。 |
| `2022-03-26-互联网工作分享.md` | 未发现新增需修正问题；这是附可信度警示的历史项目索引，不是匿名名单事实认证。 | [WorkingTime 原声明](https://github.com/Robin970822/WorkingTime)；不核实个人劳动争议，也不背书外部社区名单。 |
| `2022-03-29-数据结构整理.md` | 未发现新增需修正问题。检查均摊/期望/最坏口径、DSU、邻接表/矩阵与树种范围。 | [Princeton Union-Find](https://algs4.cs.princeton.edu/15uf/)；复杂度成立于对应实现和计算模型。 |
| `2022-06-04-基础光流法介绍.md` | 修复：欠定不等于无解；稀疏/稠密是输出位置区分，LK 不限于稀疏；理想二次信号的平移恒等式不依赖微小位移。逐式检查 Taylor、法方程、二次型与邻域最小二乘。 | [Farnebäck §2–3](https://www.diva-portal.org/smash/get/diva2:273847/fulltext01.pdf)；保留亮度/局部近似、可逆性与多尺度边界。 |
| `2025-03-26-推荐算法排序.md` | 修复条件后验与总体正类比例混用、粗排/精排输入输出量混用；长度权重改为策略示例。MMoE、校准逆式、标签泄漏检查通过。 | [MMoE 作者说明](https://research.google/pubs/modeling-task-relationships-in-multi-task-learning-with-multi-gate-mixture-of-experts/)、条件概率代数；业务经验无法独立认证，性能效果须实验。 |
| `2025-03-26-推荐算法召回.md` | 修复 DSSM 与所有双塔等同的表述，概率伪码改 text；核对 triplet、logQ 限制、DR 多路径、Bloom false positive。3 张内嵌图片已人工查看。 | [Deep Retrieval §2.1–2.2](https://arxiv.org/pdf/2007.07203)、文内 DSSM/采样论文；保留概率校正的采样模型边界。 |
| `2025-03-27-推荐算法特征交叉.md` | 未发现新增需修正问题。FM 双线性项、DCN 内积/最高阶/复杂度、Gate NU 和 PPNet 调制、FiBiNET 标量/向量区分一致；2 张图片已查看。 | [DCN §3](https://arxiv.org/html/1708.05123v1)、[PEPNet §2.2 式6–9](https://arxiv.org/html/2302.01115v2)；不外推实测收益。 |
| `2025-03-27-推荐算法行为序列建模.md` | 补正 DIN 软加权而非硬筛选、未归一化权重及无位置特征时的置换不变性；明确 DIEN 辅助监督，撤回 BST 必然更快的表述。 | [DIN §4.3 式3](https://arxiv.org/pdf/1706.06978)、[DIEN 式12–13](https://arxiv.org/html/1809.03672v3)、[BST §2](https://arxiv.org/html/1905.06874v1)。 |
| `2026-03-17-2025-2026视频生成研究进展调研.md` | 未发现新增需修正问题。时空/采样时间、VAE 示例、DiT 二次注意力、噪声阶段 MoE 与评测边界一致。 | [Wan2.2 官方架构说明](https://github.com/Wan-Video/Wan2.2)；压缩尺寸是近似算例，真实 causal VAE 还受 padding/帧对齐影响；没有生成视频实测。 |
| `2026-03-17-FlashAttention原理与实现详解.md` | 修复全 mask tile 的 -inf−(-inf) 与零除；限定空行输出约定、Q-major 教学变体和 FA1 中间 O 读写；算术强度明确读+写。 | [FA1 Algorithm 1](https://arxiv.org/html/2205.14135v2)、独立 online softmax 数值检查；不复现 GPU kernel。 |
| `2026-03-17-GQA分组查询注意力详解.md` | 修复后文生命周期仍把刚生成 token 当成已有 K/V 的残留。检查映射、32/8/1 heads 内存、uptraining 和 TP 复制。 | [GQA §2](https://arxiv.org/html/2305.13245v3)、[HF 缓存循环](https://huggingface.co/docs/transformers/main/cache_explanation)；未执行模型转换训练。 |
| `2026-03-17-PagedAttention与vLLM内存管理.md` | 修复第11–13 token 物化/生成歧义，补全 prompt 命中仍需 logits 的边界、物理块与匹配粒度区别。 | [PagedAttention](https://arxiv.org/abs/2309.06180)、[vLLM prefix caching](https://docs.vllm.ai/en/latest/design/prefix_caching/)；不把缓存模型外推到所有 hybrid backend。 |
| `2026-05-11-2026大模型推理引擎全景对比.md` | 区分按请求 goodput 与满足整体 P99 的负载容量；补单输出 TPOT、chunk/ITL 区别；TRT 迁移结论固定文档 commit。 | [TensorRT-LLM c295dd9 迁移页](https://nvidia.github.io/TensorRT-LLM/latest/legacy/tensorrt-backend-removal.html)；没有做引擎性能排名。 |
| `2026-05-12-推测解码SpeculativeDecoding原理与实践.md` | 修复“任何精确采样必须正残差”过强结论；独立匹配须保留原 target 样本；动态长度补非前瞻选择条件。其余单步/联合分布推导检查通过。 | [Leviathan Algorithm 1](https://proceedings.mlr.press/v202/leviathan23a.html)、[Transformers v4.57.1](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/generation/utils.py)、[DSpark §3.2.2/附录A](https://arxiv.org/html/2607.05147v1)。 |
| `2026-05-12-SGLang与RadixAttention详解.md` | 补相同 KV 的模型/位置/Adapter 条件；suffix 仍读取 prefix、全命中 logits 和实际对齐边界；容量惩罚与时间统一量纲。 | [SGLang §3](https://arxiv.org/html/2312.07104v2)、直接因果 attention/量纲检查。 |
| `2026-05-13-flashinfer-deep-dive.md` | 修复共享前缀 attention state 易被误读为跨不同 Query 直接复用；明确各 Query 单独计算，共用 KV 加载。检查 CSR/page 长度、布局和 plan/run 边界。 | [FlashInfer recursive attention](https://docs.flashinfer.ai/tutorials/recursive_attention.html)、[KV layout](https://docs.flashinfer.ai/tutorials/kv_layout.html)；未执行 GPU API。 |
| `2026-05-13-MoE推理优化全景指南.md` | 补 Top-k 索引/tie、路由权重模型差异、空 assignment 零除与 destination 去重；其余参数/显存/并行/容量关系一致。 | [Mixtral §2](https://arxiv.org/html/2401.04088v1)、文内 DeepEP；未测真实通信。 |
| `2026-05-14-langgraph-mcp-practical-guide.md` | 修复审批 truthiness 会误批准字符串/非空字典、schema 漏字段及旧审批载荷绑定；shutdown 改 transport closure；撤回未指定 SDK 版本的泛化。 | [LangGraph interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts)、[MCP 2025-06-18 lifecycle](https://modelcontextprotocol.io/specification/2025-06-18/basic/lifecycle)；审批函数为边界示例，不是完整认证/邮件发送实现。 |

此表不宣称未修改文章绝对无错；它记录本轮核查没有发现新的确定性问题。个人经历、外部匿名数据、真实 GPU 性能和完整分布式运行不在直接复现范围内。
