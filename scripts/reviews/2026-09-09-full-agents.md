# 全文复核：结构化生成、DFlash、A/F 与 Agent/多模态服务

核查日期：2026-09-09；基线为本轮开始时的 `8354568`。以下 5 篇均从开头完整读到参考资料，包含全部公式、伪代码和文本图。改动保持发布日期、URL、系列顺序不变；不是对旧报告的关键词复查。

## 71. `2026-08-26-structured-generation-grammar-constrained-decoding.md`

- 检查：约束层级、词表/字节/UTF-8、grammar 状态与 EOS、前缀接受、JSON Schema 覆盖、Token Healing、Top-p 次序、empty set、推测状态和注入边界。
- 修正：局部 mask 后归一化不等于对整段有效文本做全局条件采样；加入 AX/BY 的反例。正概率质量、后续 temperature/top-k 等处理顺序明确。正则限于可表达成有限自动机的子集；接受态不代表必须立即停止；Token Healing 是拼接前后分词边界变化，不是编码结果停在“半个 token”。
- 修正：多个硬约束必须取交集，回退不能绕过硬规则；draft/target 分别维护可回滚 matcher；正残差公式只与相应拒绝采样配对，独立 target matching 是另一条正确路径；首 token 可由 Prefill logits 生成。
- 一手核对：[XGrammar §3–4](https://arxiv.org/html/2411.15100v3)、[JSON Schema 2020-12 Validation](https://json-schema.org/draft/2020-12/json-schema-validation)、[SGLang Structured Outputs](https://docs.sglang.ai/advanced_features/structured_outputs.html)、[XGrammar 深层嵌套安全公告](https://github.com/mlc-ai/xgrammar/security/advisories/GHSA-7rgv-gqhr-fxg3)。局部与全局分布区别另做独立概率推导。
- 复算：`full-foundations-examples.mjs` 中 AX/BY 反例通过：局部为 `(0.5,0.5)`，全局条件分布为 `(0.9,0.1)`。
- 边界：未执行各 backend 的 grammar、JSON Schema 全功能一致性或安全压力测试；不能把 valid JSON 等同于事实正确、业务授权或安全调用。

## 72. `2026-08-27-dflash-block-diffusion-speculative-decoding.md`

- 检查：候选数/anchor/配置 block size、一次去噪前向、目标上下文 KV injection、训练 attention、衰减 loss、前缀期望、采样保证、KV 提交、ragged/graph 与性能口径。
- 修正：双向表示交互不等于随机采样相关；在固定上下文、确定前向与逐位置独立 RNG 下，proposal 仍可因子分解。拒绝后缀的关键是 target 验证条件变化，不是 DFlash 草稿必须顺序生成。目标特征表示为逐位置投影序列，而非仅缓存最后一个向量。
- 修正：候选接受数与实际提交数分开；ragged/padded 候选槽位不等于包含未物化 anchor 的实际 forward 位置数；逻辑回滚与等待在途工作结束后的物理回收分开。
- 一手核对：[DFlash v1 §3.1、§4.1–4.2、§5](https://arxiv.org/html/2602.06036v1)、[Speculators DFlash 的 Anchor/Sample From Anchor](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/dflash/)、[DSpark v1 的 admission 条件](https://arxiv.org/html/2607.05147v1)。双向表示与概率独立的区别属于在明确假设下的代数分析，不冒充论文实验。
- 复算：`full-programs-examples.mjs` 检查独立分布乘积与候选/anchor 计数。
- 边界：未训练 drafter、未执行 verification kernel、未复现 6× 或并发性能；自适应长度仍须满足精确采样的因果准入要求。

## 73. `2026-08-28-attention-ffn-disaggregation-moe-serving.md`

- 检查：层公式与 residual、A/F 和 P/D 区分、路由/目的地、通信与 credits、周期模型、比率优化、负载长尾、重试、成本/SLO。
- 修正：文字保留了公式中的 Norm/residual；缓存负载使用物化输入位置，而非刚采样输出数。单层 profile 得到 token-layer/s，同质层复用硬件时完整输出吞吐还需除以层数；再区分每实例、每 GPU/每金额吞吐。充分填充流水服务率不能直接当成完整跨层输出延迟。epoch 只能拒绝旧结果，物理地址复用要先 quiesce/drain 或安全隔离。成本分子统一金额、统计区间并避免重复计费，分母不含待验证候选。
- 一手核对：[A/F Ratio v1 §3.1–3.4、Appendix B](https://arxiv.org/html/2601.21351v1)、[GPUDirect RDMA 生命周期/同步](https://docs.nvidia.com/cuda/gpudirect-rdma/index.html)。通信与成本公式为文中假设下的简化，不当作任何 SKU 的实测报告。
- 复算：不同 A/F GPU 数的资源归一化、采样与物化位置计数见 `full-programs-examples.mjs`。
- 边界：未复现真实 A/F pipeline、尾延迟、专家调度、通信故障或特定规模的最优实例比例。

## 74. `2026-08-29-program-aware-agent-serving-scheduling.md`

- 检查：程序与 Call 粒度、PLAS/ATLAS、Fork/Join、工具中断 Preserve/Discard/Swap 成本、缓存身份、placement、deadline/fairness、取消与恢复、指标。
- 修正：PLAS 原定义与其它工作量度量变体分开。ATLAS 论文的实际近似维护 Program scalar，不要求完整依赖 DAG；显式 Join-aware 元数据是文章后文的扩展设计，不是原算法必要条件。
- 修正：placement score 是示意而非原论文式，各项统一时间并避免重复关键路径计数；公平分数明确排序方向/单位与无饥饿边界。Prefix 可逐出后在 miss 时重算；未验证候选与已提交待发送输出分开；取消流程先排空在途访问再释放；KV 错误也影响正确性，只有完整可重算契约下丢弃它才主要影响性能。
- 一手核对：[Autellix v1 §4.2.1、§4.2.2](https://arxiv.org/html/2502.13965v1)。成本量纲、状态先后与最后两层 checkpoint 区别按全文假设逐项推导。
- 复算：同一程序后继 Call 继承累计服务、ATLAS scalar 近似的基本状态例见 `full-programs-examples.mjs`。
- 边界：未部署 Agent runtime、执行工具副作用或测 program makespan；现有参考文献不代表每项建议都是 Autellix 的现成功能。

## 76. `2026-08-31-multimodal-serving-encoder-cache-scheduling.md`

- 检查：不同视觉融合架构、媒体解码与安全边界、Processor 版本、Patch/合并视觉 Token/上下文长度、占位符、四级缓存、租户隔离、显存、Chunked Prefill、并行与量化、TTFT。
- 修正：不能到完整像素分配后才限制解码规模；先验证格式头并全程限制资源。SSRF DNS 检查要绑定实际连接地址，逐跳校验与网络边界配套。
- 修正：区分 Encoder Patch 与进入 LLM 的视觉向量，以 Qwen2-VL 的 256→64+2 为例；Prefill 长度不重复计替换占位符，KV 式限定普通同尺寸 MHA/GQA。Encoder Key 继承完整 Processor/内容身份；随机处理也进入缓存契约。
- 修正：不强制选择最早缓存阶段；取消后完成在途访问再解除引用/释放；TTFT 加上未被计入的 Connector/传输/发送，并说明串行分项与实际关键路径之别。
- 一手核对：[Qwen2-VL v2 §2.1](https://arxiv.org/html/2409.12191v2)、[vLLM 多模态输入文档](https://docs.vllm.ai/en/latest/features/multimodal_inputs/)、[Pillow Image.open 的资源保护](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.open)、[OWASP SSRF](https://cheatsheetseries.owasp.org/cheatsheets/Server_Side_Request_Forgery_Prevention_Cheat_Sheet.html)。
- 复算：256 Patch、64 向量、66 个视觉 span 位置，以及单占位替换后总长度的算术检查通过。
- 边界：未执行媒体攻击、真实模型 OCR/量化质量或 TP/多图压测；安全段落不是完整网络代理/解码沙箱的认证。

5/5 篇均有定点修订。全文已读完、一手关键段落已核对与小例通过，均不等于证明文章在任意模型/软件版本/硬件上绝无错误。
