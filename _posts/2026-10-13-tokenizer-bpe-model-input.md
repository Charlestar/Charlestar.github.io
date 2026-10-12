---
layout: post
title: "Tokenizer 与 BPE：一段文本怎样成为模型的输入"
subtitle: "从词表、合并顺序和 UTF-8 字节，理解上下文长度、聊天模板与 Prefix Cache 的共同边界"
date: 2026-10-13
last_modified_at: 2026-10-13
author: iStar
catalog: true
series: model-serving-agents
series_order: 5
technology_year: 2016
mathjax: true
tags: [LLM推理, 机器学习, 工程实践]
---

一段文字进入推理服务时，最先影响 GPU 工作量的决定，往往发生在 CPU 上：Tokenizer 把它转换成了多少个 token、每个 token 对应哪个整数，以及聊天模板在文字前后加入了什么。

这些整数决定了模型查哪几行 embedding，也决定了 Prefill 的序列长度、KV Cache 中的位置和可复用前缀。文章看起来相同、字符数量相同，甚至使用同一个模型名称，都不足以保证这些结果相同。

我们从一份可以手算的词表开始，逐步走到服务里的长度预算、缓存复用与流式输出。下面的频率、token ID 和代码均为自制教学示例，没有代替任何生产模型的 Tokenizer。

## 文本、token 与 token ID 是三件事

假设一个人为词表包含这些条目：

| 词表条目 | 人为指定的 ID |
| --- | ---: |
| `a` | 10 |
| `b` | 20 |
| `c` | 30 |
| `ab` | 40 |
| `bc` | 50 |
| `abc` | 60 |

字符串 `abc` 可以被切成 `a / b / c`，也可以成为 `ab / c`，或者直接成为 `abc`。仅有词表并不能告诉我们最后采用哪一种切法；还需要分词算法及其已训练的规则。

一旦规则选择 `abc`，编码结果就是 `[60]`。60 没有“比 10 更复杂”或“比 20 更重要”的含义，它标识 embedding 的一行。对 embedding 矩阵 $$E\in\mathbb{R}^{V\times d}$$，模型读取 $$E[60]$$，得到一个 $$d$$ 维向量。这里 $$V$$ 是矩阵行数；我们的 ID 故意留有空档，所以若真的按这些 ID 查表，至少需要 61 行，不能把上表六个条目直接装进六行矩阵。生产模型中的 ID 与矩阵范围必须一致。向量的内容来自模型参数训练；Tokenizer 自己没有在这一步理解句子的含义。

词表也不必以完整词语为单位。条目可以是一个字、词的一部分、标点、带前导空格的片段，或者一串字节。因此，“一词一个 token”“一个汉字一个 token”都只能在特定词表的特定输入上成立。

词表越小，未被合并的细碎片段通常越多；词表越大，常见片段有机会变成较长的 token，却也扩大 embedding 和输出分类空间。上述 embedding 参数量是 $$Vd$$；若模型有独立的输出投影，那里还要考虑相应矩阵，权重共享时则不能重复计算存储。这里存在取舍，没有一条“词表越大，模型越好”的规律。

## BPE 学的是一份合并顺序

Byte Pair Encoding 原本是一类压缩方法；2016 年的论文将反复合并相邻符号的思路用于机器翻译的子词表示。其字符级构造从较小的符号集合出发，在训练语料中统计相邻对，反复合并最常见的一对，并保留合并次序。现代 byte-level BPE 则可以从字节出发；两个版本应当区分。[论文 §3.2](https://aclanthology.org/P16-1162.pdf)

我们自行构造一份没有空格、每个样本单独处理的语料：

```text
ab   出现 6 次
bc   出现 5 次
abc  出现 1 次
```

暂时不加词尾符号，不做规范化，也不允许不同样本之间发生合并。初始符号只有 `a`、`b`、`c`。

第一轮，`(a,b)` 出现 7 次：来自六个 `ab` 和一个 `abc`；`(b,c)` 出现 6 次：来自五个 `bc` 和一个 `abc`。因此首先合并 `(a,b) → ab`。

注意，下一轮统计的已经不是原始字符串中的字符对，而是更新后的符号序列：

```text
[ab]      × 6
[b, c]    × 5
[ab, c]   × 1
```

现在 `(b,c)` 的次数是 5，`(ab,c)` 的次数是 1。第二轮合并 `(b,c) → bc`，第三轮才合并 `(ab,c) → abc`。得到的规则表是：

| 合并优先级 rank | 相邻符号对 | 新符号 |
| --- | --- | --- |
| 0 | `(a, b)` | `ab` |
| 1 | `(b, c)` | `bc` |
| 2 | `(ab, c)` | `abc` |

rank 表示历史顺序，token ID 表示词表索引。在这份例子里，`ab` 的 rank 是 0，ID 是 40，显然不是同一个数。具体实现可以采用不同的存储方式，但不能脱离它的定义，把词表 ID 普遍当成 BPE 合并优先级。[Hugging Face BPE 说明](https://huggingface.co/docs/course/en/chapter6/5)

训练通常还会设词表规模或合并次数等停止条件；碰到频率相同的候选，也需要确定的 tie-breaking。一个可复现的 Tokenizer 不仅需要数据集名字，还需要这些训练约定。

## 推断时执行旧规则，不重新学习新输入

处理新字符串 `bcabc` 时，我们从：

```text
b / c / a / b / c
```

出发。当前 `(b,c)` 出现两次，`(a,b)` 只有一次，但 `(a,b)` 的 rank 更小，仍先执行它：

```text
b / c / ab / c
bc / ab / c
bc / abc
```

结果为 `[50, 60]`。这是在应用已经冻结的规则，不是“每次给一句话重新统计谁最常见”。它也不是先遍历词表，贪心取当前位置能匹配到的最长字符串。BPE 的核心操作是按已学的优先级合并相邻符号；作者的 [subword-nmt 应用代码](https://github.com/rsennrich/subword-nmt/blob/master/subword_nmt/apply_bpe.py)把这份次序存为 `bpe_codes`。

下面的 Python 代码只支持 `a`、`b`、`c`，实现这份显式字符对规则。相同 rank 的不同位置按左侧优先处理；它没有 Normalizer、PreTokenizer、词尾符号、特殊 token、聊天模板或 byte fallback，也没有实现生产 Tokenizer 的性能优化。

<!-- tokenizer-example: toy-bpe -->
```python
MERGES = [("a", "b"), ("b", "c"), ("ab", "c")]
RANK = {pair: i for i, pair in enumerate(MERGES)}
VOCAB = {"a": 10, "b": 20, "c": 30, "ab": 40, "bc": 50, "abc": 60}
BY_ID = {token_id: piece for piece, token_id in VOCAB.items()}


def encode(text):
    if any(ch not in "abc" for ch in text):
        raise ValueError("This example only accepts a, b, and c")
    pieces = list(text)
    while len(pieces) > 1:
        candidates = []
        for pos in range(len(pieces) - 1):
            pair = (pieces[pos], pieces[pos + 1])
            if pair in RANK:
                candidates.append((RANK[pair], pos))
        if not candidates:
            break
        _, pos = min(candidates)
        pieces[pos:pos + 2] = [pieces[pos] + pieces[pos + 1]]
    return [VOCAB[piece] for piece in pieces]


def decode(ids):
    return "".join(BY_ID[token_id] for token_id in ids)


print(encode("bcabc"))       # [50, 60]
print(decode([50, 60]))      # bcabc
print(encode("a") + encode("b"))  # [10, 20]
print(encode("ab"))          # [40]
assert decode(encode("bcabc")) == "bcabc"
```

一些 byte-level 实现不显式保存“左符号、右符号”两个键，而是对合并后字节串记录 rank。比如 [tiktoken 0.12.0 的教学实现](https://github.com/openai/tiktoken/blob/0.12.0/tiktoken/_educational.py)使用 `mergeable_ranks`，同一表还参与 ID 查找。这里可以看到一个具体实现的两种用途，但它不意味着所有 BPE 文件格式都如此。

也因此，上面的代码用于观察规则次序，不能拿一份真实 `tokenizer.json` 随意抽出词表塞进去，就认为已经重现了该模型的输入。

## 真正的 Tokenizer 还有几道边界

在 [Hugging Face Tokenizers v0.20.3 的流水线](https://huggingface.co/docs/tokenizers/v0.20.3/en/pipeline)中，编码由 Normalization、Pre-tokenization、Model 和 Post-processing 四部分构成。这里的 Model 指 BPE、Unigram 等分词模型，并不是负责生成回答的语言模型。

```text
原始文字
   ↓ Normalizer：按配置转换文本，可为空操作
规范化文本
   ↓ PreTokenizer：划出不可被普通子词合并跨越的片段
候选片段
   ↓ BPE 等模型：片段内分词并映射到 ID
token IDs
   ↓ PostProcessor：按任务约定组织序列、添加特定标记
模型输入
```

这是一种库中的组件划分，不是每个产品都严格采用同一内部流程。特殊 token 也可能由专门的匹配逻辑提前识别；聊天消息如何渲染，是更外层的输入组织问题。

Normalizer 可以做 Unicode 规范化、大小写转换或其他映射。若它把 `A` 与 `a` 统一，后续解码不可能仅凭同一 ID 判断原文究竟是哪一个。规范化不是无损压缩的一部分；是否启用、采用哪套规则，都属于模型输入契约。

PreTokenizer 则决定哪些符号有机会互相合并。若先把一个字符串切成两段，再分别运行 BPE，就不会出现跨段合并。不同实现可能按空格、标点、字母或数字模式切分，且可保留空格作为片段的一部分；它不等于一句简单的 `split()`。

PostProcessor 可以给序列加入 BOS、EOS 等约定标记，但具体行为由模型配置决定。普通文字分词完全一致，若一条路径额外加入 BOS，最终输入仍然不同。把词表、规则、预处理和特殊标记放在一起，才得到我们实际调用的 Tokenizer。

## 字节覆盖，解决不了所有 Unicode 问题

“字”在不同层次下有不同含义。`中` 是一个 Unicode 码点，编码为 UTF-8 时是三个字节 `E4 B8 AD`；具体 Tokenizer 可能把它编码成一个 token，也可能拆成多个。

byte-level BPE 可以以 256 个单字节值为初始集合，再训练较长的字节串。保留这套完整基础字节映射时，任意合法文本的 UTF-8 字节都能被表示，不必因为某个罕见字符未作为整体条目出现，就立刻落到未知符号。[tiktoken 的字节级构造](https://github.com/openai/tiktoken/blob/0.12.0/tiktoken/_educational.py)

但单个 token 对应的字节，未必是完整的 UTF-8 字符。假设一个人为分块把 `中` 拆成 `E4`、`B8`、`AD`，完整拼回后可以解码；逐块强行解码、把错误替换成 `�`，再拼接结果，则已经丢掉信息。

下面用标准库的增量 UTF-8 解码器观察这个区别。三个字节块是我们自己指定的，未声称任何真实模型会采用这种切分。

<!-- tokenizer-example: utf8-stream -->
```python
import codecs

chunks = [bytes([b]) for b in "中".encode("utf-8")]
decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
output = [decoder.decode(chunk, final=False) for chunk in chunks]
output.append(decoder.decode(b"", final=True))

print(output)                    # ['', '', '中', '']
print("".join(output))            # 中
print("".join(chunk.decode("utf-8", errors="replace") for chunk in chunks))
# ���
assert "".join(output) == "中"
```

真实流式 detokenization 还可能包含空格恢复、子词前缀清理或特殊 token 过滤；应使用该 Tokenizer 的 decoder 语义。在某个字节前缀暂时不完整时保留状态，与最终输出确实非法时如何报错或替换，是两种不同情形。

还有一个容易混淆的层次：外观相同的字符，底层编码未必相同。`é` 可以是一个预组合码点，也可以由 `e` 与组合重音构成。Tokenizer 是否将它们统一，取决于 Normalizer；不能根据屏幕显示相同推断 token IDs 相同。

## SentencePiece 是工具，不是 BPE 的别名

[SentencePiece](https://github.com/google/sentencepiece/tree/v0.2.0)支持 BPE 与 Unigram 等子词模型，可以直接从原始句子训练，也将空格纳入自己的表示方式。看到 `tokenizer.model` 或词表中的 `▁`，不能据此断言它一定使用 BPE，更不能据此断言所有 token 都是 UTF-8 字节片段。

BPE 按合并规则逐步构造片段；Unigram 采用不同的概率建模与切分方式。使用同一个工具封装，并不会让两者的编码算法变成同一个算法。

SentencePiece v0.2.0 默认使用名为 `nmt_nfkc` 的规范化规则，包含 NFKC 的变体及空格相关处理，也允许配置其他规则。它的 [Normalization 文档](https://github.com/google/sentencepiece/blob/v0.2.0/doc/normalization.md)明确说明规则被存入模型文件，并提示实现并非完整 NFKC。

因此，不能把 `decode(encode(text)) == text` 当成所有 Tokenizer 的无条件承诺。完整字节覆盖、未知字符处理、Normalizer 是否保留信息、decoder 是否跳过特殊 token，都要分别检查。byte fallback 会将未知片段回退为 UTF-8 字节表示；它是一项具体配置或实现能力，不是所有 SentencePiece 模型自动拥有的特性。SentencePiece v0.2.0 的 [模型配置定义](https://github.com/google/sentencepiece/blob/v0.2.0/src/sentencepiece_model.proto)中，`byte_fallback` 默认是 `false`；Hugging Face 的 [BPE 模型参数](https://huggingface.co/docs/tokenizers/v0.20.3/en/api/models#tokenizers.models.BPE)也将它列为单独选项。

## 消息内容，不等于聊天模型真正读到的输入

用户发送 `{"role": "user", "content": "解释一下 BPE"}` 时，模型一般不会直接读这份字典。Chat Template 将角色、轮次、内容及控制标记串成序列，Tokenizer 再把它转换成模型输入。[Transformers Chat Templates 文档](https://huggingface.co/docs/transformers/chat_templating)

同样一句问题，作为用户消息、系统指令或者已经开始的 assistant 回复，前后标记可能不同。对于多轮对话，还必须决定最后是在开始一条新的 assistant 消息，还是继续补全最后一条消息。这个选择会改变模型看到的前缀。

如果已经加载了与模型匹配的 `tokenizer` 对象，典型用法是直接从完整消息得到 IDs：

```python
ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
)
```

这是官方接口使用片段，需已有 Tokenizer 与 `messages`，不是本文标准库玩具示例。`add_generation_prompt` 是否真正增加标记，也由具体模板决定。

另一条路径是先 `tokenize=False` 得到渲染后的字符串，再分词。文档要求这时设置 `add_special_tokens=False`，以免重复加入模板已经包含的特殊标记。重复 BOS 仍然是合法整数，通常不会触发形状错误，却改变了模型实际消费的序列。

模型与 Tokenizer 还必须共享特殊 ID 的语义。EOS、padding、消息结束符是否相同，应读取该模型的约定；不能看到一个叫 `</s>` 的显示字符串，就认为它在所有模型里都是相同 ID 或相同终止事件。

## token 数量为什么不能简单相加

回到我们的玩具词表，令 `A="a"`、`B="b"`。分别分词得到 `[10]`、`[20]`，拼接两份 ID 是 `[10,20]`；把文字先拼成 `ab` 再分词，结果却是 `[40]`。

若 $$T$$ 表示确定的编码函数、$$\Vert$$ 表示拼接，则存在：

$$
T(A\Vert B)\ne T(A)\Vert T(B)
$$

对应的长度也不同：1 + 1 不等于 1。这不是说每次拼接都会改变分词；预切分边界、特殊 token 或某些输入内容可能阻止跨界合并。这个反例说明，我们不能无条件假设可加性。

同一个反例也揭示了前缀问题：文字 `a` 是 `ab` 的文本前缀，但 `[10]` 不是 `[40]` 的 token 前缀。若应用不断把生成文本重新拼回字符串，再重做分词，边界附近的历史 IDs 有可能发生变化。

因此，统计完整 prompt 的可靠步骤是先按模型契约渲染，再执行匹配的 Tokenizer，最后计算所得 ID 序列的长度。把系统指令、用户内容、工具描述的估算长度分别相加，只能当近似；按字符数乘一个固定系数，同样不能承担严格的上下文上限检查。

长度裁剪也应理解这一点：按字符截取的字符串，不一定保留恰好某个数量的 token；按 token 截取后再解码重编码，也不保证原 ID 前缀保持不变。比如模型可以生成合法的 `[10,20]`，它解码为 `ab`，重新编码却得到 `[40]`；合法生成序列不必是 Tokenizer 对同一文字选出的那份规范切分。服务可以在明确支持 token IDs 的边界上直接使用裁剪后的序列；如果必须回到消息层，应重新检查模板完整性与最终长度。

## 输入契约怎样传到 Serving

对于文本 decoder 模型，输入 IDs 的长度是序列长度的直接依据。它进入 Prefill 调度，影响 Attention 工作量，并沿着各层生成与保存对应位置的 K/V。Padding、packing、滑动窗口、缓存命中与多模态扩展会进一步改变实际资源成本，不能把 token 数直接等同于总 GPU 时间。

在 [RadixAttention]({% post_url 2026-05-12-SGLang与RadixAttention详解 %})这样的前缀缓存里，对比的是实际 token 前缀。字符串显示相似、被不同模板包装，或者经另一份 Tokenizer 分词，都不构成同一前缀的证据。

另一方面，相同 token IDs 也只是缓存正确性的输入条件之一。缓存中的 K/V 还依赖模型权重、Adapter、位置与 Attention 配置；多模态场景还依赖相应输入状态。将同一 `[40]` 交给另一个模型，它可能读取的是完全不同的 embedding 行，旧 K/V 当然不能继续使用。

部署时，最好把以下内容作为同一份输入契约记录：

- 词表与 token-ID 映射、BPE merges 或其他分词模型文件。
- Normalizer、PreTokenizer、decoder 及相关运行参数。
- Chat Template、特殊 token 定义与是否自动添加标记。
- 模型和 Adapter revision，以及影响位置与缓存语义的配置。

调试“相同问题为什么缓存不命中”，可以先取两次渲染后的输入，逐项比较 IDs 找到首个差异，再追到模板或分词边界。调试解码异常时，token 对应的字节十六进制往往比逐 token 打印出的 `�` 更有信息。

终止条件也要分层处理。Stop token IDs 直接针对生成序列；stop string 则是文本匹配，一个字符串可能跨多个 token，甚至跨网络分块。实现必须保留足够的增量状态，不能只检查最新一块，也不能把每个 token 强行解码成独立字符串。这一问题与 [结构化生成的 Tokenizer 适配]({% post_url 2026-08-26-structured-generation-grammar-constrained-decoding %})共享同一字节与序列边界。

本文的两个标准库示例可以独立运行，验证合并次序、ID 映射、分段非可加性与 UTF-8 增量解码；它们没有验证真实词表速度或模型效果。理解 Tokenizer 的价值在于让我们能够直接检查模型究竟读到了什么：先确认序列正确，长度预算、缓存策略与生成行为才有共同的依据。
