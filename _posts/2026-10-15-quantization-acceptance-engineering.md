---
layout: post
title: "量化模型怎样验收：从参考数值到真实内核与服务质量"
subtitle: "用严格的 CPU 检查工具串起量化合同、有效 token 计数、内核对齐和服务证据"
date: 2026-10-15
last_modified_at: 2026-10-15
author: iStar
catalog: true
series: model-quantization
series_order: 120
technology_year: 2026
mathjax: true
tags: [模型量化, LLM推理, 工程实践]
---

一个模型文件缩小到原来的四分之一，随手问几道题也能正常回答，是否已经足以交付？还不够。它可能使用了错误的 zero point，可能把高低四位读反，也可能正确地做了量化，却根本没有进入低精度内核。即使这些都正确，长上下文的任务质量、首 token 延迟和高并发下的失败率仍然是另外几件事。

前面的文章分别解释了量化格式与算法，这一篇把它们放回同一条工程链：**针对一份确定的量化产物，逐步回答数值是否自洽、内核是否正确、任务退化是否可接受，以及服务收益是否真实。** 每一关都需要自己的比较对象，不能让前一关的成功替后一关作答。

配套的 [quantization-acceptance.py](https://github.com/Charlestar/Charlestar.github.io/blob/master/scripts/quantization-acceptance.py) 是一个只依赖 Python 标准库的 CPU 小工具。它读取严格 JSON，计算指定样本的张量误差、已对齐 logits 指标，以及一种明确的 UINT4 打包合同。它不下载模型、不运行 GPU，也不签发“模型已经验收”的总证书。

## 1. 把一次验收拆成六种证据

假设我们把一个 BF16 模型变成量化模型。最容易定位错误的方式，不是从最终回答倒猜原因，而是每次只增加一个变化。

| 关卡 | 应比较什么 | 能得到的证据 | 不能顺带推出什么 |
| --- | --- | --- | --- |
| 浮点图改写 | 原始浮点图与融合、缩放或旋转后的浮点图 | 测试输入上改写误差在预定阈值内 | 量化误差和低精度执行已经合格 |
| 量化数值参考 | 浮点张量与声明规则的 Q/DQ 重构 | 数值规则及量化误差可检查 | packed 字节或 GPU 内核正确 |
| 打包布局 | 已知 codes、固定字节向量、独立解包结果 | 该布局下元素与元数据映射一致 | 内核实际使用了它 |
| 真实内核 | 目标内核与同一 Q/DQ 数据的高精度计算 | 指定形状和后端下的实现误差 | 原模型的任务质量不下降 |
| 模型质量 | 相同评测协议下的基线与量化模型 | 所测任务、样本和上下文内的回归结果 | 任意任务和任意文本都正确 |
| 在线服务 | 固定负载与资源条件下的两个服务 | 延迟、容量、SLO 和成本的实测差异 | 更换流量、硬件后收益仍相同 |

例如，[SmoothQuant]({% post_url 2026-08-02-smoothquant-w8a8-activation-outliers %}) 或 [旋转量化]({% post_url 2026-10-15-quarot-spinquant-rotation %}) 的图改写，应先在不量化时检查。否则，错误的融合与正常的量化舍入叠加在一起，无法判断损失来自哪里。

真实 kernel 也应先对齐“同一份量化数据的参考计算”，再与原始 BF16 做总回归。直接用一个最终输出差值衡量两者，会把算法固有误差和内核实现错误混在一起。前者可能需要重新校准，后者可能只是一次错误的位移或 scale 索引。

## 2. 先确定这份产物和这次评测是什么

数值比较需要固定输入，工程比较还需要固定身份。至少应保留模型及 adapter 的不可变 revision、量化后的实际文件 hash、量化 recipe 与分组轴、校准样本及顺序，以及导出器和后端版本。评估侧还要保留 tokenizer、聊天模板、截断与位置规则、上下文长度、数据集 revision 和样本标识。

这些信息不是一长串装饰字段。比如两个 tokenizer 的词表大小一样，不代表同一个 token ID 的含义一样；两批 logits 都是 $$N\times V$$，也不代表它们来自相同前缀。只检查 shape，不能证明语义对齐。

校准集和独立评估集也要分开。用同一批数据搜索 scale、挑选最好的 recipe，再用它宣布泛化质量不变，会把选择过程泄漏到评测结果中。对基线和量化模型，应使用同样的评估协议，并保存足够定位差异的逐样本记录。

seed 只能固定部分随机性，不是跨环境逐位复现的承诺。[PyTorch 2.8 的可复现性说明](https://github.com/pytorch/pytorch/blob/v2.8.0/docs/source/notes/randomness.rst)明确区分了随机数控制、确定性算法选择和版本、平台之间的可复现边界。真正的记录需要版本、设备和执行配置，而不只是一个 seed。

配套工具的 SHA256 绑定的是**本次实际读入的原始字节**。空格、行尾或字段顺序改变都可能改变 hash，即使解析后的数据相同。它可以回答“这份报告算的是哪个文件”，不能回答“文件是否由指定模型诚实导出”。模型名、GPU 名或 `kernel_pass: true` 写进 JSON，也不会因此变成真实运行证据。

## 3. 运行一份严格但有意收窄的 CPU 检查

在仓库根目录，使用 Python 3.10 或更新版本即可运行：

```sh
python -B scripts/quantization-acceptance.py --self-test
```

自测包含最近偶数舍入、整数 zero point 的加法位置、已知 nibble 向量、奇数尾部、张量指标、mask 与 PPL，以及故意注入的 schema、packing 和数值错误。自测会输出一份严格 JSON；其中 `self_test.status` 为 `passed`，但顶层仍为 `incomplete`，因为真实模型、GPU 和服务关卡没有执行。

工具只有两个互斥模式：`--self-test` 和 `--input PATH`。缺少模式、拼错参数、使用参数缩写或同时传入两个模式都会非零退出，不会悄悄改跑一份 toy 数据。成功时 stdout 只有一份 JSON，错误诊断输出到 stderr。

### 输入的三组数据互不自动关联

v1 顶层要求整数 `schema_version: 1`，并且至少有一组可计算数据。不存在的组标记为 `skipped`，不补默认数据。

| 数据组 | 输入合同 | 工具实际计算 |
| --- | --- | --- |
| tensor | 非空、等长的一维有限 reference/candidate 数组 | 这些展平样本的误差，不推断层或 head 的形状 |
| logits | 同形状非空 $$N\times V$$ 数组，$$V\ge2$$；N 个整数 targets 和 N 个布尔 mask | 同一批 mask=true 行上的 KL、top1、NLL 和 PPL |
| uint4 | codes、packed_bytes、logical_length、正 scale、整数 zero_point，以及固定 rounding/layout | low-first packing parity、该合同下的 DQ 和端点比例 |
| external_evidence | 可选 JSON object | 原样列为 reported_unverified，不升级本地证据 |

下面是一份可以保存为 UTF-8 `evidence.json` 的完整合成输入。第二行 logits 被 mask 排除；外部声称的 kernel 通过，只用来演示它不会被工具采信。

<!-- fixture: quantization-acceptance-v1-evidence -->
```json
{
  "schema_version": 1,
  "tensor": {
    "reference": [1, 2],
    "candidate": [2, 4]
  },
  "logits": {
    "reference": [[0, 0], [0, 0]],
    "candidate": [[0, 0], [0, 2]],
    "targets": [0, 0],
    "mask": [true, false]
  },
  "uint4": {
    "codes": [0, 1, 2],
    "packed_bytes": [16, 2],
    "logical_length": 3,
    "scale": 0.5,
    "zero_point": 1,
    "rounding": "ties_to_even",
    "layout": "low_first"
  },
  "external_evidence": {
    "kernel_pass": true,
    "model_revision": "not authenticated"
  }
}
```

随后运行：

```sh
python -B scripts/quantization-acceptance.py --input evidence.json
```

这份输入应得到 tensor 的 MSE 为 2.5、相对平方 L2 为 1、cosine 约为 1；logits 只评分 1 行，KL 为 0、top1 一致率为 1、两侧 PPL 都为 2；UINT4 解码为 `[-0.5, 0, 0.5]`。第二个 logits 行即使分布明显改变，也不进入任何评分指标，因为 mask 对所有指标采用同一口径。

这三个数据组是三个独立的数值声明。工具不会因为它们在同一个文件中，就假定 `tensor.reference` 是 `uint4` 的原始权重，也不会把这些 logits 与所附模型 revision 自动关联。

### 让输入错误与数值失败真正失败

JSON 数字 1.0、整数 1 和布尔 true 在这里不是可以互换的值。版本、codes、targets、zero point 和长度必须是真正的 JSON 整数；mask 必须是真正的布尔值。用于数值计算的整数还必须能无损转成 binary64：例如 $$2^{53}+1$$ 会被拒绝，而可精确表示的 $$2^{53}+2$$ 仍合法，避免不同整数转换成同一浮点数后出现假零误差。普通小数字面量仍按 binary64 舍入，并不承诺十进制任意精度；不参与计算的外部整数 metadata 则不强制转浮点。工具拒绝空数组、ragged 矩阵、未知字段、重复 key、非法 code、坏 scale 和越界 target。被 mask 排除的行也必须满足 schema，不能用它藏起 NaN 或非法 target。

Python 默认 JSON 行为并不等于这套合同：重复 key 需要显式拒绝，NaN/Infinity 需要拒绝，`1e400` 这类合法 JSON 数字语法也可能超出 binary64 范围；非零字面量直接下溢成零也会被本工具拒绝。工具先严格按 UTF-8 解码，并在解析和字段检查时限制这些边界；输出关闭非标准 NaN/Infinity。[Python JSON 文档](https://docs.python.org/3.13/library/json.html)说明了相关 parser hooks 和默认行为。

有限输入仍可能算出无穷：例如 $$10^{308}-(-10^{308})$$、平方误差累加或 $$\exp(1000)$$。非零误差的平方也可能下溢成零，或者正的 SSE 在求均值、相对误差时除成零；cosine 的非零正负比值也可能在最后一步下溢。工具对这些越出其 binary64 数值合同的情况输出带阶段和字段的 `numeric_error`，不把它们冒充零误差，也不靠随意加入 epsilon 掩盖。reference 真正为零导致相对误差或 cosine 未定义，则返回 `null` 和原因，与计算失败区别开。log-softmax 中极小指数项下溢则允许存在，因为计算不再对该项直接取 log。

退出码 0 只表示当前 CPU 作用域的计算完成；CLI 或输入错误返回 2，数值失败、packing parity 失败和自测失败返回 1。工具没有全局 `accepted: true`，也没有替业务选择默认的质量阈值。

## 4. cosine 很高与饱和率很低都可能误导

对 reference $$x$$ 和 candidate $$y$$，本文使用：

$$
\mathrm{MSE}
=\frac1n\sum_i(y_i-x_i)^2,
\qquad
\mathrm{relative\_squared\_l2}
=\frac{\sum_i(y_i-x_i)^2}{\sum_i x_i^2}.
$$

第二项明确是平方 L2 的比值，不省略定义而统称 NMSE。分母为零时该项未定义。

cosine 只看方向。如果 $$x=[1,2]$$、$$y=[2,4]$$，cosine 等于 1，但所有元素都放大了一倍，MSE 为 2.5，相对平方 L2 为 1。只设一个 cosine 阈值，便可能放过显著的幅值错误。

“饱和率”同样需要先定义。对于整数参考：

$$
u_i=\operatorname{RNE}(x_i/s)+z,
\qquad
q_i=\operatorname{clip}(u_i,q_{\min},q_{\max}),
$$

至少有三个不同的比例：舍入后的 $$u_i$$ 超出 code 范围的比例、最终 $$q_i$$ 落在端点的比例，以及原值 $$x_i$$ 超出重构端点范围的比例。

取 $$s=1$$、$$z=0$$、码域 $$[-7,7]$$，输入为 $$[0,7,7.2,100]$$。7.2 超出了最大重构值 7，但最近偶数舍入仍为 7，没有发生 code clamp；7 本来就合法，也会落在端点。三个比例分别为 1/4、3/4、2/4，不能共用一个“饱和率”名称。

<!-- example: quantization-acceptance-tensor-and-clamp -->
```python
import math

reference, candidate = [1.0, 2.0], [2.0, 4.0]
sse = sum((y - x)**2 for x, y in zip(reference, candidate))
ref_square = sum(x*x for x in reference)
cand_square = sum(y*y for y in candidate)
cosine = sum(x*y for x, y in zip(reference, candidate)) / math.sqrt(ref_square*cand_square)
assert sse / len(reference) == 2.5
assert sse / ref_square == 1.0
assert math.isclose(cosine, 1.0)

values, scale, zero, qmin, qmax = [0.0, 7.0, 7.2, 100.0], 1.0, 0, -7, 7
raw = [round(x / scale) + zero for x in values]
codes = [max(qmin, min(qmax, q)) for q in raw]
clamp_fraction = sum(not qmin <= q <= qmax for q in raw) / len(values)
endpoint_fraction = sum(q in (qmin, qmax) for q in codes) / len(values)
range_fraction = sum(not scale*(qmin-zero) <= x <= scale*(qmax-zero)
                     for x in values) / len(values)
assert raw == [0, 7, 7, 100]
assert codes == [0, 7, 7, 7]
assert (clamp_fraction, endpoint_fraction, range_fraction) == (0.25, 0.75, 0.5)
assert round(0.5) + 1 == 1 and round(0.5 + 1) == 2
print("MSE / relative squared L2 / cosine:", sse / 2, sse / ref_square, cosine)
print("code clamp / endpoint / original range:", clamp_fraction, endpoint_fraction, range_fraction)
```

先舍入再加整数 zero point 的顺序也不是无关紧要的写法：在中点上，`round(0.5)+1` 与 `round(0.5+1)` 不相等。[ONNX QuantizeLinear opset 13](https://onnx.ai/onnx/operators/onnx__QuantizeLinear.html#quantizelinear-13)对其 INT8/UINT8 路径说明了商的最近偶数舍入；这里的 UINT4 是同类仿射规则的独立参考合同，不声称 opset 13 已定义 UINT4 算子。

## 5. packing 检查必须有独立的已知答案

对 v1 的 low-first UINT4，一个字节是：

$$
b_i=q_{2i}+16q_{2i+1}.
$$

codes 为 0 到 15 时，已知字节序列是 `10 32 54 76 98 ba dc fe`，这里按十六进制书写。不能只检查 `unpack(pack(codes))==codes`：打包与解包若同时把两边反过来，往返仍然成功，却都不符合外部布局。

奇数尾部也要规定清楚。示例的三个 code 为 `[0,1,2]`，打包为 `[0x10,0x02]`。第二个字节未使用的 high nibble 必须为零，logical length 为 3；解码出的 padding 不进入任何样本计数。

工具首先检查 byte 数量与 padding，再比较解码 codes 和输入 codes，最后按：

$$
\hat x_i=s(q_i-z)
$$

计算重构。若把示例第一个字节 16 改为 1，就会产生 packing parity 失败；若把 scale 从 0.5 合法地改成 1，packing 本身仍然可以正确，重构则变成另一组数。

后者不能一概叫 schema 错误。只有手中存在独立可信的预期重构 fixture，才能指出“这个 scale 与该 fixture 不一致”。本工具的自测有这样的固定答案，外部 v1 输入却没有原始浮点 source，因而不计算其源量化误差或三种 clamp 比例，只能计算最终端点占比。`rounding` 字段记录的是调用者声明的合同，也不能凭空证明这些 code 当初真的由 RNE 产生。

如果同时一致地改了 codes 与 bytes，packing parity 仍可通过。这恰好说明：**数据完整性和来源真实性是两件事。** 实际 AWQ、QServe、NF4 或 [NVFP4/MXFP4]({% post_url 2026-10-15-nvfp4-mxfp4-format %}) 有自己的码本、scale 与布局，不能把本工具的 UINT4 decoder 当通用 checkpoint 读取器。

## 6. logits 比较必须使用相同前缀和评分口径

给两个模型同一段已固定 token 的前缀，比较同一位置的输出，才是在隔离数值变化。这通常称为 teacher forcing。若让两个模型各自自由生成，某一步选择不同 token 后，后续上下文已经不同，不能继续把它们“第十步的 logits 差值”解释成同一输入下的实现误差。

对已对齐的一行 logits，令：

$$
p=\operatorname{softmax}(\ell_{\mathrm{ref}}),
\qquad
q=\operatorname{softmax}(\ell_{\mathrm{quant}}).
$$

本文使用温度 1 和方向明确的：

$$
D_{\mathrm{KL}}(p\Vert q)
=\sum_jp_j(\log p_j-\log q_j).
$$

top1 一致率与 KL 回答不同问题。top1 相同，其他概率仍可明显改变；top1 的 margin 很小时，微小扰动也可能翻转选择。工具对并列最大值固定取最低索引，避免两侧采用不同 tie-break。

计算时应直接得到稳定的 log-softmax，而不是先把小概率舍入成零，再对零取 log。参考 logits 为 $$[0,0,-1000]$$、candidate 为 $$[0,-1000,0]$$ 时，某些指数项在 binary64 下会下溢成零，但 log-softmax 仍可给出有限的对数概率，KL 约为 500。这里不能用随意 clamp 概率后的值冒充原定义的 KL。

近乎相同的分布还有另一种抵消问题：直接累加正负 $$p_j(\log p_j-\log q_j)$$，可能把极小的正 KL 算成负数。工具使用归一化概率下等价的非负项形式 $$\sum_j[p_j\log(p_j/q_j)+q_j-p_j]$$，并用 `expm1` 与零附近的级数计算；这是利用 $$\sum_j(q_j-p_j)=0$$，不是截断负 KL 或给概率加 epsilon。加权项在对数域组合，避免先把极小概率下溢成零、再乘大对数比时丢失仍可表示的结果。正文下面的小例只展示给定数值的计算，实际工具另外处理这些数值边界。

工具验证的只是输入矩阵和 mask 的合同。tokenizer、前缀、position 和模板是否确实相同，仍是导出与评估流程必须提供的证据，不会因为 shape 相等就被自动认证。

## 7. PPL 最关键的是实际评分了多少目标

对实际参与评分的目标集合 $$\mathcal S$$，PPL 是：

$$
\mathrm{PPL}
=\exp\left(
\frac{\sum_{t\in\mathcal S}-\log p(x_t)}{\lvert\mathcal S\rvert}
\right).
$$

所以需要累计总 NLL 和真正有效的目标数量，不能平均每个窗口的 PPL，也不能不加权地平均每个 batch 的 mean loss。

如果第一个 batch 评分 1 个 token、平均 NLL 为 $$\ln2$$，第二个评分 3 个 token、平均 NLL 为 $$\ln4$$，正确结果是：

$$
\exp\left(\frac{\ln2+3\ln4}{4}\right)
\approx3.3635856610.
$$

直接平均两个 loss 会得到约 2.8284271247，两者不同，尽管没有任何模型计算发生变化。

### 先对齐目标 再直接数 mask

CausalLM 通常用 `logits[:, :-1]` 预测 `labels[:, 1:]`。因此应直接检查 shifted labels 中哪些目标有效，而不是一律用“原 labels 的有效数减 batch size”。

例如两行 labels 为：

```text
[-100,    2,  3, -100]
[-100, -100,  3,    4]
```

移位后的有效目标各有 2 个。原始第一列已经是 -100，再机械地每行减 1，就会错误地各计 1 个。重叠窗口中，前缀只作上下文而不评分，这种边界尤其容易出现。

[Transformers 4.55.4 的 PPL 文档](https://github.com/huggingface/transformers/blob/v4.55.4/docs/source/en/perplexity.md)解释了 tokenizer、上下文和 stride 对 PPL 的影响；其中示例的直接减 batch 计数不能无条件套到第一列已 masked 的窗口。正确计数应从实际 shifted targets 的有效 mask 得到。

本工具则要求导出时已经把每一行 logits 与它要预测的 next-token target 对齐。输入 N 行就检查 N 行，不再 shift 一次；N=1 且 mask=true，就是 1 个评分目标，不会变成零个。KL、top1、两侧 NLL/PPL 均使用同一批 true 行，不能一项排除 padding、另一项又把它算回来。

<!-- example: quantization-acceptance-logits-mask-and-ppl -->
```python
import math

def log_softmax(row):
    peak = max(row)
    shifted = [x - peak for x in row]
    normalizer = math.log(math.fsum(math.exp(x) for x in shifted))
    return [x - normalizer for x in shifted]

ref, cand = [[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 2.0]]
targets, mask = [0, 0], [True, False]
selected = [i for i, keep in enumerate(mask) if keep]
kls, nlls = [], []
for i in selected:
    lp, lq = log_softmax(ref[i]), log_softmax(cand[i])
    kls.append(math.fsum(math.exp(p) * (p-q) for p, q in zip(lp, lq)))
    nlls.append(-lq[targets[i]])
assert len(selected) == 1
assert kls == [0.0]
assert math.exp(math.fsum(nlls) / len(selected)) == 2.0

lp, lq = log_softmax([0, 0, -1000]), log_softmax([0, -1000, 0])
stable_kl = math.fsum(math.exp(p) * (p-q) for p, q in zip(lp, lq))
assert math.isclose(stable_kl, 500.0)

labels = [[-100, 2, 3, -100], [-100, -100, 3, 4]]
shifted_counts = [sum(t != -100 for t in row[1:]) for row in labels]
blanket_counts = [sum(t != -100 for t in row) - 1 for row in labels]
assert shifted_counts == [2, 2] and blanket_counts == [1, 1]
weighted_ppl = math.exp((math.log(2) + 3*math.log(4)) / 4)
wrong_ppl = math.exp((math.log(2) + math.log(4)) / 2)
assert math.isclose(weighted_ppl, 3.363585661014858)
assert math.isclose(wrong_ppl, 2.8284271247461903)
print("scored targets / KL / PPL:", len(selected), kls[0], 2.0)
print("stable extreme-logit KL:", stable_kl)
print("shifted / blanket counts:", shifted_counts, blanket_counts)
print("weighted / unweighted PPL:", weighted_ppl, wrong_ppl)
```

这些指标仍不是问答质量的替代物。相同 PPL 可以掩盖特定任务退化，tokenizer 或模板改变也会让 PPL 失去直接可比性。任务集应该覆盖真实使用场景，并单独分析长上下文、代码、数学、结构化输出等相关维度，而不是从一个平均值推出所有任务等价。

## 8. 真实 GPU 内核需要另一份执行证据

CPU 参考通过之后，下一步才是在目标设备上运行真实 packed kernel。需要固定 GPU、驱动、CUDA、后端版本、矩阵形状、输入 dtype、分组参数和具体布局；先比较它与相同量化数据的高精度参考，再记录其耗时。

这一关有两个独立问题：算得对不对，以及是否真的走了预期内核。一个 fake Q/DQ 后接浮点 Linear 的程序，可以用于模拟数值误差，却不能直接用来报告真实 INT4/FP4 GEMM 的性能。应结合后端入口条件和 trace 检查实际调用，并明确有多少层、调用或耗时走了 fallback；不同覆盖率的分母不能混用。

在单设备上进行后续计时时，应先预热，分开记录 cold start、编译及稳定阶段。wall-clock 计时要在开始前和结束后同步参与计算的设备；也可在依赖明确的 CUDA stream 上使用 events，但 event 时间不是客户端 HTTP 总延迟。多 GPU 或 TP 还需建立所有参与设备、ranks 和通信的完成依赖，不能用一个设备的 event 代表整个请求。[PyTorch 2.8 CUDA notes 的异步执行及 stream 章节](https://github.com/pytorch/pytorch/blob/v2.8.0/docs/source/notes/cuda.rst)解释了这些同步要求。

测量范围也要与结论匹配。若收益声称覆盖 activation 量化、unpack、epilogue 或通信，就不能把这些成本排除在计时区间之外；加载权重、下载、打印或转 CPU 的操作则应分离，除非它们本来就是所报告的范围。

显存同样要分账：活跃 tensor 占用、allocator reserved、阶段峰值与进程占用不是一个指标。`empty_cache()` 不能释放仍被 tensor 使用的内存。基线和量化模型最好隔离进程测量，避免两套模型共存污染峰值。[同版本的 Memory management 说明](https://github.com/pytorch/pytorch/blob/v2.8.0/docs/source/notes/cuda.rst)区分了 allocated、reserved 与缓存释放。

本文工具不包含这一关，也没有 GPU 耗时或显存结果。上述是有目标设备后需要执行并保存的步骤，不是由 CPU 自测推导出的加速结论。

## 9. 任务质量与服务收益要分别收口

运行真实任务评测时，保存完整的评测配置与逐样本记录，比只摘一行总分更有用。以固定版本的 [lm-evaluation-harness 接口](https://github.com/EleutherAI/lm-evaluation-harness/blob/452749513f817315042df9286241a61051392470/docs/interface.md)为例，需要确定任务集合、few-shot、batch、聊天模板、生成参数和 model 参数，并保存 sample logs 与实际 task config。结果缓存应按不可变模型产物隔离，避免量化运行读到另一份模型的结果。只跑少量 `--limit` 样本适合 smoke test，不等于完成整个 benchmark。

业务阈值应在看量化结果之前，根据任务风险确定。cosine 0.99、PPL 变化 1% 或某个平均准确率，都不是跨任务通用的安全线。发现退化时，保留具体样本和前缀，才能回到前面的层输出、logits 或 KV 误差继续定位。

服务侧则要固定到达过程、输入输出长度、并发、KV 策略、scheduler 与并行方案。尤其要分开两类实验：固定相同工作集，检查一次请求到底快了多少；固定 HBM 预算，检查节省的容量能否减少 eviction、容纳更多并发。后者可能改善整体吞吐，即使某个小矩阵 kernel 并没有变快。

以客户端观察口径，TTFT 可以定义为请求开始到首 token 到达的时间，包含相应排队、传输和处理成本。对输出 token 数 $$n>1$$，平均 TPOT 可定义为：

$$
\mathrm{TPOT}=\frac{t_{\mathrm{last}}-t_{\mathrm{first}}}{n-1}.
$$

只有一个输出 token 时，这个指标未定义，应单独处理，不能填零后拉低平均值；流式 chunk 里如果包含多个 token，chunk 间隔也不等于逐 token 间隔。

SLO attainment 要声明分母和失败处理。若报告“单位时间内符合 TTFT/TPOT 目标的完成请求数”，单位是合格完成数/秒；token/s 是另一种量。DistServe 所定义的 per-GPU goodput 则是在指定 attainment 目标下，每 GPU 能承载的最大请求率，需要改变 offered load 找到边界，不能拿一次运行的合格完成数/秒直接当成同一个指标。[DistServe v3 引言](https://arxiv.org/html/2401.09670v3)给出了后一种定义。

最终报告应同时保留超时、错误、未完成请求、P50/P95/P99、重复运行次数和负载范围，而不是只展示成功请求的平均延迟。质量与服务收益分别有了证据，才有资格讨论具体使用条件下是否值得切换。

切换方案还应预先约定回滚条件：哪些质量退化、错误率或 SLO 越界会触发暂停放量或切回基线。保留一份不可变、已知可用的基线及其配套 tokenizer、recipe 和 runtime 版本；路由切回后重新检查健康状态与关键指标，并保留失败的量化产物、输入和日志供定位。这是目标服务上线前应执行并演练的策略，不是本 CPU 工具已经完成了回滚或会自动修改服务。

## 10. 一个可信的结论应带着作用域

CPU 工具的输入模式会给出 `input_sha256`、各数据组结果、`verified_scope` 和 `unverified`。缺失的数据组是 `skipped`；不合法或不可计算的输入会失败；计算完成则仍保留顶层 `incomplete`。这不是对可用结果的否定，而是在防止一小段证据覆盖尚未执行的关卡。

同样，一份完整项目报告可以说：“在指定 revision、评测集、上下文与设备条件下，图改写及内核 parity 通过，任务回归在预定范围内，服务在指定负载下达到目标。”它不应把这些限定删除后，改写成“低比特与原模型完全等价”或者“所有生成内容都正确”。

从最初的 [量化数值基础]({% post_url 2026-10-15-quantization-numerical-foundations %}) 到这里，格式、算法和系统实现已经可以接成一条可检查的链。真正有用的验收不是一个笼统的绿色勾，而是清楚知道：哪些主张已有证据，哪些错误能被复现，哪些关卡仍然需要执行。
