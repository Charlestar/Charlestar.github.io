---
layout: post
title: "结构化生成：Grammar 怎样约束每一个 Token"
subtitle: "从 JSON Schema、Regex 与 CFG 到 Token Mask，理解 Constrained Decoding 的正确性与性能边界"
date: 2026-08-26 09:00:00 +0800
last_modified_at: 2026-08-31
author: iStar
catalog: true
series: model-serving-agents
series_order: 20
technology_year: 2023
mathjax: true
tags: [LLM推理, 推理调度, GPU优化]
---

让大语言模型“只输出 JSON”看起来只是一个提示词问题：给出字段名、类型和示例，再强调不要添加解释文字。但只要把它放进自动化链路，问题很快就会暴露出来：模型可能漏掉引号，多写一个逗号，把枚举值换成近义词，或者在 JSON 前后补上一段自然语言。

这些错误对人并不难理解，却足以让解析器失败。重试可以降低失败率，但不能把概率承诺变成接口契约；而且重试会增加延迟、成本，并使同一个请求产生不同结果。

结构化生成解决的是另一类问题：它不是在生成结束后检查结果，而是在每一步采样之前，把所有会让输出离开目标语言的 Token 屏蔽掉。模型仍然决定“在合法答案中选什么”，Grammar 则决定“什么形式绝不能出现”。

本文从解码循环出发，逐层说明 JSON Schema、正则表达式和上下文无关文法如何变成 Token Mask，为什么字符级规则不能直接套在 Tokenizer 上，以及这套机制进入连续批处理、流式输出、工具调用和投机解码之后会遇到哪些工程边界。

## 1. 先区分三种不同的保证

假设服务需要返回如下对象：

```json
{
  "city": "Nanjing",
  "temperature": 28.5,
  "unit": "celsius"
}
```

我们通常希望同时得到三种保证：

1. **语法有效**：引号、逗号、括号都符合 JSON 语法；
2. **结构有效**：字段、类型、枚举和必填项符合给定 Schema；
3. **语义正确**：城市、温度和单位与现实及用户问题相符。

Grammar 主要负责前两层。它可以阻止 `unit` 生成 `kelvin-ish`，却不知道南京此刻到底多少度；也不能判断模型是否把摄氏温度误填成华氏数值。

因此，一个可靠链路通常是：

```text
模型概率分布
    ↓
Grammar 保证可解析、符合结构
    ↓
业务校验检查范围、引用和跨字段关系
    ↓
外部数据或执行结果确认事实
```

把“结构合法”误当成“答案正确”，是结构化生成最危险的使用误区。

## 2. Prompt 约束为什么不是硬约束

提示词会改变模型的概率分布，但不会把非法 Token 的概率严格变成零。

对于输出前缀 \(x_{<t}\)，普通自回归解码在词表 \(V\) 上计算：

$$
p(v \mid x_{<t}), \qquad v \in V
$$

“请只输出 JSON”可能让 `{"` 的概率显著上升，让解释性文字的概率下降。然而只要后者仍大于零，温度、采样随机性、长上下文干扰或模型能力差异都可能让它被选中。

后处理也不是完整替代品：

- 从文本中截取第一个 `{...}`，无法稳妥处理嵌套括号和字符串转义；
- 修复缺失逗号可能改变原意；
- 生成完成后才发现错误，已经浪费了全部解码计算；
- “解析失败后重试”没有确定的最坏延迟。

Constrained Decoding 的关键变化，是把结构规则放进采样路径本身。

## 3. 目标不是一个模板，而是一门语言

结构化输出可以统一描述为：只允许模型生成属于某个语言 \(L\) 的字符串。

不同接口只是描述 \(L\) 的方式不同：

| 接口 | 适合表达什么 | 典型实现 |
| --- | --- | --- |
| Choice | 有限候选，例如 `yes/no` | Trie 或有限状态机 |
| Regex | 编号、日期、受限文本格式 | 正则编译为自动机 |
| JSON Object | 任意合法 JSON 对象 | 预定义 JSON Grammar |
| JSON Schema | 字段、类型、枚举、数组和嵌套对象 | Schema 转换为 Grammar |
| CFG / EBNF | 递归语法、DSL、代码片段 | 下推自动机或等价 Matcher |

例如，以下正则描述一个简化订单号：

```regex
ORD-[0-9]{8}
```

而 JSON Schema 可以进一步表达对象结构：

```json
{
  "type": "object",
  "properties": {
    "city": { "type": "string" },
    "temperature": { "type": "number" },
    "unit": {
      "type": "string",
      "enum": ["celsius", "fahrenheit"]
    }
  },
  "required": ["city", "temperature", "unit"],
  "additionalProperties": false
}
```

这里真正交给解码器的不是“照着这个例子写”，而是一个能够回答下面问题的状态机：

> 在已经生成当前前缀之后，词表里的哪些 Token 仍可能通向至少一个合法终态？

## 4. 解码循环里到底多了什么

普通采样大致包含三步：模型计算 Logits，采样器应用 temperature、top-k 或 top-p，最后选出下一个 Token。

加入 Grammar 后，循环变成：

```text
1. 模型根据当前前缀计算 logits
2. Grammar Matcher 根据自己的状态计算 allowed_tokens
3. 将其他 token 的 logits 设为 -∞
4. 在剩余 token 上执行 temperature / top-k / top-p
5. 选出 token，并推进 Grammar 状态
6. 重复，直到接受 EOS 或到达终态
```

令当前 Grammar 状态下允许的 Token 集合为 \(A_t\)，屏蔽后的分布为：

$$
p'(v \mid x_{<t}) =
\begin{cases}
\dfrac{p(v \mid x_{<t})}{\sum_{u \in A_t}p(u \mid x_{<t})}, & v \in A_t \\
0, & v \notin A_t
\end{cases}
$$

实际系统通常不会显式重新计算上式，而是在 GPU 上用一个 Bitmask 把非法 Logit 写成负无穷，再复用现有采样 Kernel。

这也说明 Grammar 没有替模型决定字段值。在生成 `"city": "` 之后，大量字符串 Token 仍然合法，模型依旧需要从中选择；当生成到 `"unit": "` 时，Grammar 才会把选择空间收窄到枚举值的合法前缀。

## 5. Grammar 状态如何前进

考虑一个有限选择：

```text
"success"
"failed"
```

可以把两个字符串放进一棵 Trie：

```text
root
├── s → u → c → c → e → s → s
└── f → a → i → l → e → d
```

当前缀为空时，只能选择 `s` 或 `f`；生成 `fa` 后，只能继续 `i`。如果 Tokenizer 恰好有一个 `"failed"` Token，那么它也可以一步走到叶子。

Regex 同样可以转换成有限状态自动机。有限状态机只需记录当前节点，适合不需要任意深度嵌套的规则。

JSON 和通用 CFG 更复杂，因为它们包含递归：数组里可以继续放数组，对象值也可以是对象。此时系统不仅要记住“现在处于哪条产生式”，还要保存尚未闭合的调用栈，行为更接近下推自动机。

例如下面的简化 Grammar：

```ebnf
value  ::= string | number | object | array
object ::= "{" members? "}"
members ::= pair ("," pair)*
pair   ::= string ":" value
array  ::= "[" (value ("," value)*)? "]"
```

读到 `{` 后，Matcher 需要记住未来必须遇到匹配的 `}`；进入嵌套数组时，又要把数组的返回位置压栈。仅靠固定数量的有限状态无法表达任意嵌套深度。

## 6. 最大的实现难点：Grammar 读字符，模型吐 Token

文法一般定义在字符、Unicode Code Point 或字节上，模型却按 Token 生成。二者并不是一一对应：

- 一个 Token 可能包含多个字符，例如 `"temperature"`；
- 一个字符串可能被拆成几个 Token；
- Token 可能包含前导空格或标点；
- UTF-8 字符可能涉及多个字节；
- 不同模型使用完全不同的词表与正规化规则。

因此不能简单地说：“Grammar 下一步允许字符 `t`，那就只开放 Token `t`。”一个以 `t` 开头并包含后续合法字符的长 Token 也可能有效；反过来，以合法字符开头、但后半段立即违反规则的 Token 必须被拒绝。

严谨的判断是：从当前 Grammar 状态出发，完整消费某个 Token 对应的字节序列后，是否仍存在通往接受状态的路径。

可以抽象为：

```python
def token_is_allowed(grammar_state, token_bytes):
    state = grammar_state.clone()
    for byte in token_bytes:
        state = state.accept(byte)
        if state.is_dead():
            return False
    return True
```

若每一步都对整个词表执行这段逻辑，成本会很高。高性能实现会在编译阶段预分析 Tokenizer 词表，缓存与 Grammar 节点相关的结果，并把运行时输出压缩成位图。

## 7. 从规则到 Token Mask 的编译路径

一套完整后端通常包含四层：

```text
JSON Schema / Regex / EBNF
            ↓
解析与规范化
            ↓
内部 Grammar 表示（规则、状态、递归栈）
            ↓
结合 Tokenizer 词表预计算可复用信息
            ↓
运行时 Matcher + Token Bitmask
```

编译阶段可能完成：

- 展开 `$ref` 并检查递归引用；
- 把字符串枚举构造成 Trie；
- 把 Regex 转为自动机；
- 消除或改写部分 Grammar 结构；
- 识别与上下文无关的 Token 接受结果；
- 为词表建立稠密 Bitset 或稀疏候选表。

运行时则只维护每个序列自己的状态：当前规则位置、递归栈、已匹配的字节前缀，以及是否可以结束。

这就是为什么第一次使用一个新 Schema 可能明显慢于后续请求。第一次包含 Grammar 编译和 Tokenizer 适配，后续命中编译缓存时，只需创建轻量 Matcher。

## 8. 缓存键必须描述完整语义

Grammar 编译缓存不能只用原始 Schema 字符串做键。至少应考虑：

```text
canonical_schema_hash
+ grammar_backend_and_version
+ tokenizer_id_and_revision
+ vocabulary_hash
+ whitespace_or_format_options
+ strictness_options
```

其中 `canonical_schema_hash` 应来自稳定序列化：对象键顺序、无意义空白和等价表示不应导致无穷多缓存项。

Tokenizer 必须进入缓存键，因为同一个 Grammar 面对不同词表时，可接受 Token 集合不同。模型名称也不一定足够：同名模型的 Tokenizer 文件可能发生修订，或者服务端加载了自定义 Tokenizer。

缓存还需要容量限制。允许租户无限提交从未见过的复杂 Schema，会把编译缓存变成内存耗尽入口。常见策略包括：

- 限制 Schema 字节数、规则数、嵌套深度和枚举总量；
- 对编译设置时间预算；
- 按租户配额统计缓存占用；
- 采用 LRU/LFU，并记录命中率和逐出原因；
- 对常用平台 Schema 预编译、固定驻留。

## 9. JSON Schema 支持并不等于完整实现规范

JSON Schema 本身非常庞大。`type`、`properties`、`required`、`enum` 和数组约束比较直观，但一些关键字很难仅靠逐 Token Grammar 完整表达。

例如：

- `uniqueItems` 需要比较数组中已经生成的完整值；
- 数字的跨字段大小关系不是语法属性；
- 某些复杂的 `oneOf`、条件 Schema 和动态引用会显著增加状态；
- 自定义 `format` 往往需要业务代码判断；
- 字符串长度按字符、字节还是正规化后的 Code Point 计算，需要明确语义。

服务端应明确公布支持的 Schema 子集，并在请求进入排队前拒绝不支持的关键字，而不是静默忽略。静默降级会让调用方误以为已经获得硬保证。

更稳妥的接口返回可以区分：

```json
{
  "grammar_compiled": true,
  "schema_dialect": "2020-12-subset",
  "unsupported_keywords": []
}
```

生成完成后仍可再跑一次标准 JSON Schema Validator。这看似重复，却能捕获后端实现缺陷、版本差异以及不适合放入 Grammar 的语义约束。

## 10. 空白、EOS 和 Token Healing

结构化生成的边界错误往往发生在看似无关紧要的细节上。

### 10.1 空白策略

JSON 允许多处空白。如果 Grammar 对所有空白都完全开放，模型可能生成大量换行和缩进，浪费 Token；如果只允许固定格式，又可能与模型学到的常见输出形态冲突，降低有效概率质量。

因此后端常提供紧凑 JSON、固定分隔符或受限空白模式。选择哪一种是吞吐、可读性和模型适配之间的权衡。

### 10.2 EOS 不是随时合法

只有当前 Grammar 状态已经接受一个完整值时，EOS 才能进入允许集合。若对象还缺少 `}`，提前 EOS 必须被屏蔽。

与此同时，达到 `max_tokens` 属于外部强制截断，Grammar 无法凭空补齐结构。服务应把这种结果标为 `length` 或 `incomplete`，不能宣称它是有效结构化输出。

### 10.3 Prompt 尾部可能切在 Token 边界之外

有些 API 让 Prompt 直接包含输出前缀，例如：

```text
Result: {"city":
```

Prompt 的最后一个 Token 可能还包含额外空格或字符。Token Healing 会重新考虑边界附近的 Token，使生成能够从字符级前缀自然续写。若 Grammar 后端和 Token Healing 各自修改前缀，却没有共享同一份字节视图，就可能错误屏蔽本来合法的续写。

### 10.4 UTF-8 必须按增量状态处理

一个非 ASCII 字符可能跨越多个字节。Matcher 不能把暂时不完整的 UTF-8 序列立刻视为非法，也不能在最终输出里接受永远未闭合的序列。Tokenizer 解码、Grammar 字节流和响应序列化必须采用一致规则。

## 11. 连续批处理中的隐藏 CPU 瓶颈

在普通 LLM Serving 中，GPU 每轮为一个批次计算下一个 Token。加入 Grammar 后，每条序列都有自己的 Matcher 状态，也可能使用完全不同的 Schema：

```text
Request A → schema A → state A → mask A
Request B → regex B  → state B → mask B
Request C → schema C → state C → mask C
```

如果调度线程等待 CPU 逐条生成词表大小的 Mask，再启动 GPU 采样，GPU 会在每个 Decode Step 之间出现气泡。输出越短、模型越小，Grammar 开销占比反而可能越明显。

优化方向包括：

- 预计算与 Grammar 上下文无关的 Token 接受关系；
- 用位运算批量生成或合并 Bitmask；
- 在 GPU 执行当前 Step 时，CPU 准备下一轮状态；
- 把 Mask 写入固定或 Pinned Buffer，降低复制开销；
- 将 Mask 应用与 Sampling Kernel 融合；
- 按 Grammar 后端或相同 Schema 做有限度分组，但避免破坏全局批处理效率。

XGrammar 的核心工作之一，就是把大量 Token 预判从逐步运行时移到预处理阶段，并用适合并行应用的 Bitmask 表示允许集合。它仍然不会让开销凭空消失：复杂 Grammar 的上下文相关部分、状态推进和每请求 Mask 管理依旧需要计算。

评测时不应只看“请求总延迟”，还应拆出：

```text
grammar_compile_ms
grammar_mask_ms_per_step
grammar_state_advance_ms
gpu_sampling_ms
time_to_first_token
inter_token_latency
```

否则模型计算变快之后，CPU Grammar 路径可能悄悄成为新的主瓶颈。

## 12. Mask 应该在采样流水线的什么位置

典型采样器还会执行 repetition penalty、presence penalty、temperature、top-k、top-p、min-p 等操作。Grammar Mask 与这些算子的次序会影响数值和行为。

一个清晰的原则是：非法 Token 在任何情况下都不能复活。因此无论其他 Logit Processor 怎样变换分数，Grammar Mask 都必须在最终抽样前生效，并保证后续算子不会把负无穷变回有限值。

还要处理“合法集合被其他过滤器清空”的情况。例如 Grammar 只允许一个低概率 Token，而极端 top-k 实现先把它删掉。可靠实现应让硬约束优先于概率过滤，或者在集合为空时回退到 Grammar 允许集合，而不是从非法 Token 中采样。

可以把规则分成两层：

```text
硬约束：Grammar / 禁止词 / 安全协议边界
软策略：temperature / top-k / top-p / repetition penalty
```

软策略只能在硬约束允许的空间内工作。

## 13. Beam Search、并行候选与状态复制

当 `n > 1` 或使用 Beam Search 时，同一个请求会同时维护多个输出分支。Grammar 状态必须与每个分支绑定：

```text
prefix P
├── token a → matcher state Sa
├── token b → matcher state Sb
└── token c → matcher state Sc
```

如果所有分支错误地共享一个可变 Matcher，某个分支推进状态后会污染其他分支。实现上需要廉价的状态克隆、持久化数据结构，或基于日志的回滚。

分支重排同样要同步重排 Matcher。当 Beam Search 选择新的 Top-K Beam 时，KV Cache、Token 序列、累计分数和 Grammar 状态必须使用同一份索引映射。

这类错误不一定导致崩溃，反而可能偶发地产生“不该合法却通过”的输出，因此需要专门的分叉与回溯测试。

## 14. 投机解码如何与 Grammar 配合

投机解码让 Draft Model 一次提出多个 Token，再由 Target Model 并行验证。Grammar 又要求每个 Token 都基于前一个 Token 推进状态，两者必须正确组合。

假设 Draft 提出：

```text
[t1, t2, t3, t4]
```

Grammar 需要从当前状态依次检查：

```text
S0 --t1--> S1 --t2--> S2 --t3--> dead
```

那么 `t3` 及其后的 Token 不能被接受，即使 Target Model 的概率检验原本会通过。验证阶段至少要区分两种拒绝原因：

1. Target Model 的接受概率不足；
2. Draft Token 违反 Grammar。

随后由 Target Model 在状态 `S2` 的合法集合中重新采样。不能先接受整段 Draft，再用最终字符串做 Grammar 校验，那会破坏逐 Token 硬约束。

为了性能，系统可以并行计算每个位置的模型 Logits，但 Grammar 状态转换在语义上仍是前缀相关的；实现需要使用扫描、预计算转移或高效的逐 Token Matcher，而不能忽略依赖关系。

## 15. Prefix Cache 与 Grammar Cache 不是一回事

Prompt Prefix Cache 保存的是模型对输入前缀的 KV 状态。Grammar 编译缓存保存的是规则与 Tokenizer 的可执行表示。两者解决不同问题：

| 缓存 | 复用对象 | 主要键 |
| --- | --- | --- |
| Prefix/KV Cache | 模型对 Prompt 的注意力状态 | Token IDs、模型与 Adapter 身份 |
| Grammar Compile Cache | 规则的编译产物 | Schema、Tokenizer、后端版本 |
| Matcher State | 某条输出已走到哪里 | 编译 Grammar + 已生成前缀 |

两个请求可以命中同一个 Prefix Cache，却使用不同 Schema；也可以使用同一个 Schema，却从不同 Prompt 开始。Matcher State 通常不能跨请求共享，因为它随输出前缀变化。

如果服务支持断点续写或恢复生成，除了 Token 与 KV，还必须重放输出 Token 来恢复 Matcher，或者序列化与版本绑定的 Matcher State。只恢复 KV 而让 Grammar 从初始状态开始，会在下一步生成错误 Mask。

## 16. 流式输出：可消费不等于随时可解析

结构化生成保证最终结果属于目标语言，不保证每个流式 Chunk 都是完整 JSON。

例如客户端依次收到：

```text
{"city"
:"Nan
jing","temperature":28.5}
```

每段单独解析都会失败。服务端可以选择三种协议：

1. 原样流式传输 Token，客户端只在结束后解析；
2. 传输增量事件，例如 `field_started`、`string_delta`、`field_completed`；
3. 对顶层字段做缓冲，字段完整后再发送结构化事件。

第一种延迟最低但客户端复杂，后两种提供更稳定的消费语义，却需要服务端把 Grammar 状态映射为协议事件。

无论采用哪一种，都应把“正常到达接受态”“用户取消”“长度截断”“服务端错误”作为不同终止原因。收到最后一个网络 Chunk，并不自动代表 JSON 已闭合。

## 17. 工具调用本质上也是结构化语言

Tool Calling 通常包含两部分：选择工具，以及生成符合该工具参数 Schema 的对象。

```json
{
  "name": "get_weather",
  "arguments": {
    "city": "Nanjing",
    "unit": "celsius"
  }
}
```

静态工具列表可以编译成一个联合 Grammar：先约束 `name` 枚举，再根据选中的名称切换到对应参数 Schema。困难在于工具集合可能随请求变化，工具数量也可能很大。

如果把几千个工具及其 Schema 全部编成一个巨大 Grammar，会增加编译时间、状态数量和首 Token 延迟。更合理的系统可能先进行工具检索或路由，只把候选子集交给解码器；也可以采用支持运行时标签分派的结构化后端，减少重复展开。

但“先检索”引入了另一层召回风险：真正需要的工具若未进入候选集，Grammar 只能保证模型在错误候选里生成合法调用。因此工具选择质量和参数结构正确性要分开评估。

## 18. 不可信 Grammar 是资源输入，也可能是攻击面

开放 API 若允许用户提交 Regex、CFG 或 JSON Schema，就等于允许用户向编译器输入程序。系统应按不可信代码处理，而不是把它当普通字符串。

风险包括：

- 极深递归导致栈或状态爆炸；
- 巨大枚举制造超大 Trie；
- 复杂 Regex 或 Schema 消耗大量编译时间；
- 大量唯一 Schema 冲垮缓存；
- 后端解析器自身的安全缺陷；
- 错误消息回显内部文件、规则或服务实现细节。

2025 年公开披露的 XGrammar 递归处理拒绝服务问题就是一个直接提醒：即使目标只是“约束输出”，Grammar 编译与匹配仍必须有深度限制、资源预算、版本修复和隔离边界。

推荐在进入模型调度队列之前完成：

```text
请求鉴权
  → Schema 大小与关键字检查
  → 递归深度、枚举量和规则数限制
  → 有超时与内存预算的编译
  → 编译成功后才进入推理队列
```

这样可以避免昂贵 GPU Slot 被一个永远无法编译的规则占住。

## 19. 错误处理必须覆盖“合法集合为空”

理论上，正确编译的 Grammar 与正确推进的 Matcher 不应无缘无故进入死状态。但工程系统还可能遇到：

- Prompt 已经包含与 Grammar 冲突的强制输出前缀；
- 自定义 Logit Processor 屏蔽了所有合法 Token；
- Tokenizer 与编译缓存版本不一致；
- Grammar 后端存在实现缺陷；
- Schema 本身描述了空语言；
- 强制 Stop String 与合法终态冲突。

当 \(A_t = \varnothing\) 时，绝不能移除 Grammar Mask 后“随便采一个”。那会把明确的正确性失败伪装成成功。

服务应终止该序列，并返回可区分、可观察的错误，例如：

```json
{
  "finish_reason": "constraint_violation",
  "error": {
    "code": "NO_VALID_TOKEN",
    "generated_tokens": 37,
    "grammar_state_id": "..."
  }
}
```

对外不必泄露内部 Grammar 内容，但日志和 Trace 应能关联编译版本、Tokenizer 指纹与状态位置。

## 20. 正确性测试不能只准备几个示例

结构化解码器适合做属性测试，因为它有一个非常明确的不变量：凡是报告成功的输出，都必须被独立解析器接受。

建议覆盖以下层次。

### 20.1 Grammar 单元测试

- 每条规则的合法与非法字符串；
- 空对象、空数组、转义、Unicode 和极值数字；
- 嵌套深度边界；
- EOS 只在接受态开放；
- 不支持关键字明确拒绝。

### 20.2 Tokenizer 交叉测试

- 同一 Grammar 配合多种 BPE、SentencePiece 与自定义词表；
- 一个 Token 跨越多个 Grammar 字符；
- Token 内含前导空格、引号和 UTF-8 字节；
- 特殊 Token 不得意外进入正文。

### 20.3 随机与差分测试

随机生成 Schema 和合法实例，让 Constrained Decoder 生成大量结果，再交给独立 JSON Schema Validator 检查。还可以让两个 Grammar 后端处理同一组输入，比较接受集合和最终输出。

### 20.4 Serving 集成测试

- 连续批处理中不同请求使用不同 Grammar；
- Beam 分叉、重排与取消；
- Prefix Cache 命中和未命中；
- 投机解码开启与关闭；
- 流式中断、`max_tokens` 截断与服务重启；
- 动态 Adapter 或模型切换后缓存键不串用。

这些测试需要同时检查结果与中间事件。只验证最终 JSON 能解析，可能漏掉 Mask 失效后“模型碰巧仍生成合法结果”的问题。

## 21. 性能评测应拆成冷路径和热路径

结构化生成的性能至少有两个阶段：

1. **冷路径**：解析 Schema、编译 Grammar、适配 Tokenizer、填充缓存；
2. **热路径**：每个 Decode Step 生成并应用 Token Mask、推进状态。

因此应分别报告：

| 指标 | 回答的问题 |
| --- | --- |
| Compile latency p50/p99 | 新规则首次出现有多慢 |
| Compile cache hit rate | 生产请求有多少避开冷路径 |
| TTFT | 编译是否拖慢首 Token |
| Inter-token latency | Matcher 是否拖慢逐 Token 解码 |
| Output tokens/s | 整体吞吐损失 |
| Mask density | 每步通常开放多少词表 Token |
| Grammar CPU utilization | 是否出现 CPU 饱和 |
| GPU bubble time | GPU 是否在等待 Mask |
| Cache bytes / eviction | 编译缓存是否稳定 |

测试集合也不能只有一个简单 Schema。至少要覆盖：小枚举、宽对象、深嵌套数组、复杂 Regex、大型工具集合，以及多租户随机 Schema。

基线应包含无约束解码、Prompt-only、冷缓存约束和热缓存约束。否则一个漂亮的平均吞吐数字无法说明首次请求体验，也无法暴露恶意或长尾规则。

## 22. 一条可落地的服务实现路径

把前面的机制收束起来，可以得到一条相对稳妥的实现顺序。

### 第一步：定义接口契约

明确支持 Choice、Regex、JSON Schema 还是 CFG；列出 JSON Schema 方言与关键字子集；定义截断、取消和约束失败的响应语义。

### 第二步：把编译放在 Admission 阶段

先做大小、深度、关键字和租户配额检查，再在受限资源中编译。只有成功得到编译产物的请求才能进入 GPU 调度队列。

### 第三步：建立版本完整的缓存键

绑定 Schema 规范化结果、Tokenizer 指纹、Backend 版本和格式选项。部署新 Backend 或 Tokenizer 时，让旧缓存自然失效。

### 第四步：把 Matcher 作为 Sequence State

Matcher 与 Token 序列、KV Block、采样参数一样，归属于具体序列。Fork、Beam 重排、暂停、迁移和恢复都必须带上它。

### 第五步：将 Mask 路径纳入调度性能模型

测量 CPU 准备时间、拷贝时间和 GPU 应用时间；让 CPU 与 GPU 重叠工作，并对复杂 Grammar 施加独立限额。

### 第六步：做独立的结束校验

Grammar 成功到达接受态后，再用标准解析器和业务 Validator 检查一次。若不一致，应当视为服务端正确性故障并告警，而不是普通模型错误。

## 23. 从一轮请求观察完整生命周期

以工具调用为例，一轮请求可以拆成：

```text
1. API 收到 tools 与 response schema
2. 规范化 Schema，检查大小和支持范围
3. 查询 Grammar Compile Cache
4. 未命中则编译，并绑定 Tokenizer 指纹
5. Scheduler 创建 Sequence 与 Matcher
6. 模型执行 Prefill
7. 每轮 Decode：
   logits → allowed mask → sampling → matcher advance
8. 到达 Grammar 接受态后开放 EOS
9. 独立解析与 Schema 校验
10. 以 completed / length / cancelled / constraint_violation 结束
```

这条链路展示了结构化生成的位置：它既不是模型内部的新知识，也不是响应结束后的文本清洗，而是一段横跨 API Admission、CPU 编译、GPU Sampling、Sequence State 和响应协议的 Serving 能力。

## 24. 结语

Grammar-Constrained Decoding 把“希望模型遵守格式”变成“非法格式无法被采样”。它的核心并不神秘：编译目标语言，跟踪当前状态，在每一步生成允许 Token 集合，然后将其他 Logit 屏蔽。

真正困难的是字符语言与子词 Token 的映射，以及这份逐序列状态如何进入高吞吐推理系统。编译缓存键少一个 Tokenizer 版本，可能产生错误 Mask；连续批处理里 CPU 慢几百微秒，可能让 GPU 每轮等待；Beam、投机解码和恢复流程漏带 Matcher，又可能破坏原本的硬保证。

因此，评价一套结构化生成实现不能只问“支持 JSON 吗”，还应继续追问：支持哪些 Schema 语义、Mask 在哪里生成、缓存怎样隔离、非法集合为空时如何失败、冷路径和热路径分别多快，以及最终结果是否经过独立验证。

当这些问题都有明确答案时，结构化输出才不只是一个方便的 API 参数，而是可以承载工具调用、Agent 协议和自动化工作流的可靠接口边界。

## 参考资料

- Willard & Louf, [Efficient Guided Generation for Large Language Models](https://arxiv.org/abs/2307.09702)
- Dong et al., [XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models](https://arxiv.org/abs/2411.15100)
- Dong et al., [XGrammar 2: Unified Grammar-Constrained Decoding](https://arxiv.org/abs/2601.04426)
- [XGrammar 官方仓库与文档](https://github.com/mlc-ai/xgrammar)
- [vLLM Structured Outputs 文档](https://docs.vllm.ai/en/latest/features/structured_outputs/)
- [XGrammar GHSA-7rgv-gqhr-fxg3 安全公告](https://github.com/mlc-ai/xgrammar/security/advisories/GHSA-7rgv-gqhr-fxg3)
