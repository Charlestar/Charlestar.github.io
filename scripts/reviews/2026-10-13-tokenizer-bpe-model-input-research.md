# Tokenizer 与 BPE：调研与写作交接

日期：2026-10-13。负责范围为一手资料阅读、概念核对与原创大纲；未调用模型 API、下载模型权重或运行生产 tokenizer benchmark。

## 选题与已有覆盖

检索现有 87 篇标题与正文中的 Tokenizer、BPE、SentencePiece、FlexAttention、DPO。已有结构化生成文章讨论 tokenizer 与字符规则的映射；RLHF 数据流文章强调 tokenizer/chat-template contract；多模态 Serving、缓存和推测解码文章提及版本兼容。没有文章独立讲清 BPE 训练与推断、字节和 token 的关系、切片不可加性以及这些性质怎样影响 Serving。

建议标题：`Tokenizer 与 BPE：文本怎样变成模型真正接收的输入`。文章自己从输入路径建立概念，不要求读者先懂推理引擎；避免“新手”“入门”等受众标签。

## 实际阅读范围

1. [Sennrich et al., ACL 2016 论文](https://aclanthology.org/P16-1162.pdf)：实际阅读 §3 动机、§3.2 BPE 描述、Algorithm 1 及 test-time 段，PDF 第 2–4 页有关段落；核对字符级初始符号、词频加权相邻对、词边界、学习合并与推断分离。未完整阅读全部实验、附录或复现机器翻译结果。正文只简要介绍历史与算法来源，不翻译论文实验。
2. [HF 官方 BPE 讲解](https://huggingface.co/docs/course/en/chapter6/5)：实际阅读 Training algorithm、Tokenization algorithm、Implementation 与 tie-breaking 提醒；用于核对学习规则顺序与实际编码的区别。正文全部语料和数字自行构造，不复用教程语料或长段翻译。
3. [HF Tokenizers v0.20.3 pipeline](https://huggingface.co/docs/tokenizers/v0.20.3/en/pipeline)：实际阅读 Normalization、Pre-Tokenization、Model、Post-Processing、Decoding；据此描述四阶段 encode 及独立 decode 合同。固定版本作为可核接口快照，不称其是 2026 最新版。
4. [HF Tokenizers v0.20.3 Models API](https://huggingface.co/docs/tokenizers/v0.20.3/en/api/models)：实际阅读 BPE 参数与 vocab/merges、dropout、unk、byte_fallback 语义；核对 BPE 模型不是完整 tokenizer 配置。未将本篇 toy 实现声称为该库源码复现。
5. [subword-nmt 作者源码 apply_bpe.py](https://github.com/rsennrich/subword-nmt/blob/master/subword_nmt/apply_bpe.py)：实际阅读 `bpe_codes` 初始化、`encode` 中候选相邻对、最小 rank 选择、重叠合并、end-of-word 和 optional vocabulary splitting；该链接是读取当日 master，未固定 SHA，正文不以它承诺生产 API 稳定性。它确认推断不重新统计当前请求的 pair 频率。
6. [SentencePiece 项目 README](https://github.com/google/sentencepiece)：实际阅读 raw sentence、BPE/Unigram、空白符 `▁` 和 model-contained preprocessing 段。没有引用当日首页的性能表或把维护者 benchmark 当本文实测。
7. [SentencePiece v0.2.0 normalization](https://github.com/google/sentencepiece/blob/v0.2.0/doc/normalization.md)：实际阅读完整 50 行文档；默认 `nmt_nfkc` 是 NFKC 的变体并附加空白处理，支持 identity/custom rule。由规范化可能多对一自行推导原始 bytes 不一定可逆，不照抄 README 在特定例子下的“lossless”表述为通用保证。
8. [Transformers 官方 Chat templates](https://huggingface.co/docs/transformers/chat_templating)：实际阅读消息到 token 序列、Using apply_chat_template、special-token duplication 提醒、add_generation_prompt、continue_final_message；正文使用概念合同与自行示意的格式，不把示意特殊 token 当真实模型约定。该网页为 2026-10-13 当日读取，不称其已固定所有未来版本行为。

各来源只用于短概念摘要与事实核对，单一页面衍生文字保持 200 words 内；主体解释、例子、工程推论和教学实现独立组织，不长译单源。

## 原创贯穿例与核对数字

简化前提：输入已经是独立 ASCII pre-token，初始符号为字符；没有 normalization、特殊 token、word-end 标记、byte fallback 或 dropout。训练语料 `ab×6, bc×5, abc×1`。

- 初始相邻对 `(a,b)` 计数 7，`(b,c)` 计数 6；先合并 `(a,b) → ab`。
- 更新后的语料为 `ab×6, [b,c]×5, [ab,c]×1`；下一对 `(b,c)` 计数 5，`(ab,c)` 计数 1；第二合并为 `(b,c) → bc`。
- 第三合并 `(ab,c) → abc`，最终词表包含 `a,b,c,ab,bc,abc`。rank 是三个合并的学习顺序，ID 是词表索引，二者不同。
- 推断 `bcabc`：`[b,c,a,b,c] → [b,c,ab,c] → [bc,ab,c] → [bc,abc]`。不重新训练或按本请求频率重排 rank。
- 令 `A='a', B='b'`：`encode(A)=[a]`、`encode(B)=[b]`，但 `encode(A+B)=[ab]`。token 数不具有任意文本切片可加性；字符前缀 `a` 也未必是完整字符串 `ab` 的 token 前缀。这是存在性反例，不声称每个切分都会变化。
- Unicode 教学字 `中` 的 UTF-8 bytes 为 `E4 B8 AD`；若某个 byte-level tokenizer 把这些字节分到多个 token，逐 token 按完整 UTF-8 decode 可产生替换字符，增量 decoder 应保留未完成字节。这个字节序列是标准编码事实；本篇不声称真实模型必将该字拆为三个 token。

## 建议正文结构

1. 从请求原文 → 模板渲染 → token IDs → embedding → Prefill 建立贯穿路径，解释真实预算为何在输入合同之后计算。
2. 分开 Unicode code point、UTF-8 byte、子词 token 与 ID；词、人眼字符、token 都不能互换。
3. 讲清 normalization、pre-tokenization、模型切分、post-processing；decoder 不是简单把显示字符串拼接。
4. 用上面三条语料逐步学习 pair、更新语料、得到 rank；写一段明确教学范围的可复算实现。
5. 用 unseen 新串执行固定 rank，说明推断为何不同于训练；token ID 不必由 rank 直接推导。
6. 将字符 BPE、byte-level BPE、byte fallback 分开；初始 256 bytes 不意味着最终每 token 只有一个 byte，也不意味着语义理解无未知概念。
7. SentencePiece 是工具，支持 BPE 和 Unigram；不扩写 Unigram 全部推导，解释 normalization 使可逆性成为配置属性。
8. Chat template 增加角色与边界 token；完整模板化后的 token 数才与模型上下文输入一致；避免二次特殊 token。
9. 用切片反例解释预算、缓存和流式 detokenization；缓存依赖相同 token 前缀及模型/adapter/position 等执行身份，tokenizer 哈希是重要但非充分键。
10. 模型部署时核对 tokenizer 文件、normalizer、pre-tokenizer、merges、vocab、special IDs、template、stop 规则与 revision；给相关阅读而非把 checklist 当正文主体。

## 不能写成的结论

- Token 永远等于一个词、一个汉字或一个 UTF-8 byte。
- BPE 一定只采用字节、推断一定按当前请求词频重新合并，或者所有 BPE 库都与 toy API 同构。
- SentencePiece 等于 Unigram，或任何 tokenizer 都保证原文逐字节无损恢复。
- 重新 tokenize 累计文本必然保持之前 token 列表，或随便拼接各片段的 token 数即可得到准确上下文预算。
- 同 tokenizer 保证 KV cache 可共享。模型、adapter、位置与其他执行参数仍须一致。
- 小语料 CPU 例子证明生产 tokenizer 吞吐、模型质量或端到端服务性能。

## 可交独立审核的复算项目

加权 pair 计数及三步合并；rank 与任意 ID 分离；`bcabc` 编码；`a`/`b` 切片不等式；有明确 pre-token 边界时不跨界；UTF-8 增量解码与逐片错误解码；normalization 多对一；模板与额外特殊 token 计数；模型身份和 token-prefix cache 反例。只测试本文明确承诺的性质，不要求复刻真实模型词表。
