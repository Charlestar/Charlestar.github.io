# Tokenizer 与 BPE 新文独立审核

日期：2026-10-13。对象：`_posts/2026-10-13-tokenizer-bpe-model-input.md`。writer 停止编辑后，auditor 通读正文、所有公式、表格、代码、内链与参考链接；修订后移交主审构建、提交和发布。

## 独立核实的一手材料

- [Sennrich et al. ACL 2016](https://aclanthology.org/P16-1162.pdf)：§3 动机、§3.2、Algorithm 1、test-time 操作与词边界段落。没有宣称精读全部实验或复现 BLEU。
- [HF BPE 说明](https://huggingface.co/docs/course/en/chapter6/5)：训练/应用算法及规则顺序；教学语料和计数采用文章自己定义的例子。
- [subword-nmt apply_bpe.py](https://github.com/rsennrich/subword-nmt/blob/master/subword_nmt/apply_bpe.py)：初始化 rank、`encode` 候选选择/重叠处理、词尾和 optional vocabulary 分裂逻辑。此处为当日 master，不承诺永久 API。
- [tiktoken 0.12.0 教学源码](https://github.com/openai/tiktoken/blob/0.12.0/tiktoken/_educational.py)：训练、编码、UTF-8 字节解码和 `mergeable_ranks` 的实际用途；另读当前 `core.py` 中 special token、非法代理字符与流式边界的接口说明。没有安装或跑真实 encoding。
- [HF Tokenizers v0.20.3 pipeline](https://huggingface.co/docs/tokenizers/v0.20.3/en/pipeline)及 [Models](https://huggingface.co/docs/tokenizers/v0.20.3/en/api/models)：四阶段、decoder、BPE vocab/merges/byte_fallback/ignore_merges 参数。也核对当前 Components 与 Added Tokens 页面；正文采用固定版本的组件语义，不称其是当日最新版。
- SentencePiece v0.2.0 [README](https://github.com/google/sentencepiece/blob/v0.2.0/README.md)算法/原始句子/空格段、[normalization.md](https://github.com/google/sentencepiece/blob/v0.2.0/doc/normalization.md)完整文档，以及 [sentencepiece_model.proto](https://github.com/google/sentencepiece/blob/v0.2.0/src/sentencepiece_model.proto)的模型类型、`byte_fallback` 默认值、特殊符号和 Normalizer 配置字段。
- [Transformers Chat templates](https://huggingface.co/docs/transformers/chat_templating)：消息序列、模板化、`add_generation_prompt`、`continue_final_message` 与二次特殊 token 的提醒。

## 已确认及修订

1. 训练语料的加权对计数为 7/6，第一次合并后为 5/1，三条规则的顺序与 `bcabc → [50,60]` 一致。应用执行冻结 rank，不按本次请求频率重新训练；字符例子不冒充 byte-level 或真实模型 Tokenizer。
2. rank 与 ID 有意分离。发现原稿六条词表用 ID 10–60 后直接讨论 `E[60]`，容易让读者误取 `V=6` 发生越界；补充 `V` 是 embedding 行数，例子实际查表至少要 61 行，部署必须核对 ID 范围。
3. 完整基础字节覆盖、单 token 无法独立解码、非法最终 UTF-8 和暂时不完整字节前缀是不同条件。文章没有声称 `中` 在真实词表中必分成三个 token。
4. SentencePiece 是工具而非单一算法；规范化可能丢失原文信息。补充直接的模型配置来源：`byte_fallback=false` 是 v0.2.0 默认值，不能仅凭 HF BPE 参数说明推断 SentencePiece 默认值。
5. 明确 `T(A+B)` 不具无条件可加性，也不保证文本前缀成为 token 前缀。补充合法生成 `[10,20]` 解码再编码成为 `[40]`，解释合法模型输出不必是规范切分；该例没有冒充 canonical `encode` 输出截断一定改变 IDs。
6. 模板化完整输入后计长度、避免重复特殊 token、同 IDs 仍需模型/Adapter/位置/Attention/多模态输入兼容，逻辑成立。stop strings 可跨 token 与网络块；仅检查最新片段不足以实现一般匹配。
7. 没有“新手”受众标签；主线从 ID、BPE 到字节、模板和 Serving，已有内链均引用真实文件。未添加独占标签或改写旧文。

## 已执行验证

`node scripts/reviews/oct13-tokenizer-examples.mjs`：10 组通过。

脚本直接从两处稳定标记提取正文 Python 并执行，不维护第二份正文 `encode` 或 decoder。另用不同组织方式的独立 oracle、标准库和数学反例验证：

- 四个实际打印结果、ID/rank 分离与最大 ID 对矩阵范围的要求。
- 从正文读取三条语料频次，重新学习三步合并并核对每轮统计。
- `{a,b,c}` 上长度 0–7 共 3,280 条字符串与按历史顺序逐条应用合并的 oracle 一致，且全部 roundtrip。
- 冻结规则、空输入与域外字符拒绝；文字拼接、ID 拼接、pre-token 分段边界及前缀反例。
- 正文增量 UTF-8 的四块输出、逐片替换损失，以及 final 时不完整 UTF-8 应报错。
- NFC/大小写的多对一规范化反例；合法非规范 ID 序列重新编码变化。
- 所有 Python 块语法可解析。HF chat 调用片段没有执行，缺少真实 Tokenizer 与消息对象不伪装为接口兼容测试。

`node scripts/audit-posts.mjs` 通过：88 篇、9 系列、24 标签。`git diff --check` 通过。

这些验证只检查教学代码、特定概念反例和元数据，不是生产 Tokenizer benchmark、模型推理或全部文章正确性的证明。

## 主审交接后的页面验证

最终稿 production Jekyll 构建通过，生成页面检查通过（102 个 HTML、127 个文件）；原生公式保存检查通过（81 篇含公式文章、479 个行内公式、875 个块公式）。本篇在本机浏览器渲染 8 个公式，MathJax 错误为 0，页面宽度与滚动宽度均为 1265 px，没有整页水平溢出。检查了实际公式和段落截图，并刷新确认最终补充例子已进入页面。这不等于生产 Tokenizer 性能验证，也不提前宣称线上部署成功。
