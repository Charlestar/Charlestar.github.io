# 量化端到端验收：终篇研究交接

研究日期：2026-10-15。系列第12篇，2026工程综合，而非声称一种新算法/指标在2026首创。建议标题：**量化模型怎样验收：从参考数值到真实内核与服务质量**。

主线应是一份 artifact 逐关获得证据，不再重复前11篇算法清单。可交付一个 CPU/标准库小工具，附可选 GPU 与真实模型操作。仅 CPU 自测通过时，必须保留模型质量、真实kernel、服务性能为未验证。

## 一、实际阅读来源和范围

- [ONNX QuantizeLinear opset13](https://onnx.ai/onnx/operators/onnx__QuantizeLinear.html#quantizelinear-13)：本次重读 §13 的整数范围、nearest-even、per-axis 与输入类型。只用这版的 INT8/UINT8 数值语义，不借当前页面更高opset的FP4支持反向解释opset13。
- [PyTorch2.8.0 CUDA notes 固定源码](https://github.com/pytorch/pytorch/blob/v2.8.0/docs/source/notes/cuda.rst)：读 Asynchronous execution、CUDA streams 与 Memory management。网页版访问失败，实际读的是官方tag原文；GPU计时必须完成同步，allocator reserved≠allocated，empty_cache不释放活跃tensor。
- [PyTorch2.8.0 Reproducibility 固定源码](https://github.com/pytorch/pytorch/blob/v2.8.0/docs/source/notes/randomness.rst)：读开头、RNG控制、cuDNN选择、deterministic算法段。相同seed不保证跨版本/CPU-GPU逐位复现。
- [Transformers4.55.4 perplexity文档固定源码](https://github.com/huggingface/transformers/blob/v4.55.4/docs/source/en/perplexity.md)：全文实读，包含PPL定义、tokenization、context/stride与示例。网页版访问失败。**不原样复制其示例的 blanket `valid_count-batch_size` 计数方式，见后文反例**。
- lm-evaluation-harness v0.4.9 固定 [`452749513f817315042df9286241a61051392470`](https://github.com/EleutherAI/lm-evaluation-harness/tree/452749513f817315042df9286241a61051392470)，2025-06-19：完整读 [`docs/interface.md`](https://github.com/EleutherAI/lm-evaluation-harness/blob/452749513f817315042df9286241a61051392470/docs/interface.md)，README实读revision、model_args、缓存与logging相关段；未跑任务，未逐行审eval实现。
- [DistServe v3](https://arxiv.org/html/2401.09670v3)：实读引言的TTFT/TPOT/SLO与per-GPUgoodput定义段，不将该文实现和倍速带入量化终篇。
- 算法、packed布局、FP4/FP8版本约束直接链接本系列相应逐篇research已读来源，不另加二手综述充数。

## 二、验收关卡与能证明什么

| 关卡 | 比较对象 | 通过后可声称 | 仍不能声称 |
| --- | --- | --- | --- |
| graph rewrite | 原FP vs融合/旋转后的FP | 在测试输入上改写误差在阈内 | 量化质量/真实低位执行 |
| 数值reference | FP张量 vs声明Q/DQ | 量化规则与误差可检查 | packed bytes正确/kernel正确 |
| packing | 已知codes→bytes→外部referencedecode | 该布局与元数据映射一致 | GPU算子用上此布局 |
| kernel | 目标kernel vs同一Q/DQ后的高精度参考 | 指定shape/dtype/backend正确 | 原模型任务精度不下降 |
| 模型质量 | 同一任务/前缀下FP vsquant | 该评测覆盖内回归可接受 | 任意任务等价/事实绝对正确 |
| 服务 | 固定workload下延迟、容量、SLO | 指定负载有实证 | 换硬件/流量仍相同收益 |

相邻每关尽量只增加一个变量。真实kernel先对比量化reference，而非一开始只对比原BF16：后者混合算法固有误差和实现错误。之后仍要与BF16作任务层总回归。

## 三、最容易写错的指标

### 张量误差与“饱和率”

`MSE=sum((x-xhat)^2)/n`；建议显式命名 `relative_squared_l2=sum(error²)/sum(reference²)`，而不是不解释定义的NMSE。reference零向量时相对误差/cosine需返回null或明确特殊语义，不能偷偷加epsilon再宣传一致。

cosine不检测幅值：reference[1,2]、candidate[2,4]有cosine1，MSE2.5、relative_squared_l2=1。不能凭cosine>0.99验收层输出。

对整数参考 `u=RNE(x/s)+z`：

- `code_clamp_fraction`：u越出code区间的比例；
- `endpoint_fraction`：最终code等于min/max比例；
- `prequant_range_exceed_fraction`：x越出重构端点区间比例。

三者不同。s1、z0、code[-7,7]、x=[0,7,7.2,100]时分别1/4、3/4、2/4。7.2超出最大重构值，但RNE后仍7，没有发生code clamp。若recipe先clip输入，还需单列preclip规则，不能用端点占比冒充真实clip。

### Logits、KL与分叉

在同一teacher-forced prefix、同一tokenizer/模板/position/mask下比较 logits。定义 `p=softmax(logits_ref)`，`q=softmax(logits_quant)`，用 `KL(p||q)=sum p(log p-log q)`，说明方向、温度默认1。稳定计算从log_softmax出发；真实任意概率输入若q=0且p>0，KL为无穷，不能悄悄clamp后说原KL有限。

top1一致不代表分布相同；小logits margin下微小差异就能翻转greedy，但greedy分叉后上下文不同，不能继续把自由生成的第t步logits当同一输入数值回归。任务输出应独立评测。

### PPL的有效token计数

`NLL_total = sum(loss_t over actually scored targets)`；`PPL=exp(NLL_total/N_valid)`。不同batch/window的mean loss必须按真正有效目标数加权，不能平均每段loss/PPL。不同tokenizer、BOS/EOS、chat模板、context/stride、文档连接规则都会改变口径。

CausalLM内部通常移位：logits[:, :-1]预测labels[:, 1:]。所以有效计数应直接由 **shifted labels / scoring mask** 计算。不能总用 `(labels!=-100).sum()-batch_size`：如果原第一列已masked，再减batch会重复扣除。官方文档也需要独立验算。

已执行反例：labels两行 `[-100,2,3,-100]`、`[-100,-100,3,4]`；shift后有效token各2，blanket减1得到各1。另两个batch的平均NLL为ln2/ln4、有效数1/3，正确PPL=3.3635856610；直接平均两个loss给2.8284271247。

## 四、CPU/标准库小工具建议（由writer/root实现，不在本研究代理改脚本）

建议一个完整可执行 `scripts/quantization-acceptance.py` 或文章内独立脚本，Python3标准库即可，不下载模型、不访问网络。范围要小但闭环，避免搭一个未经跑通的大框架。

### 最小功能

1. `--self-test`：nearest-even+clamp、Q/DQ、UINT4 low-first pack/unpack、稳定log_softmax/KL、maskedNLL/PPL、张量误差与三个clamp统计。
2. `--input evidence.json`：可选读取外部导出的同shape reference/candidate、aligned logits+targets+mask、声明的量化contract。无输入时只跑toy，不伪称检查了仓库外模型。
3. 打印机器可读JSON：精确数值contract、输入SHA256、样本数/有效token数、各指标及 `verified_scope`。没有model/GPU证据时状态应明确为`incomplete`，而非输出一个笼统`accepted:true`。
4. 代码应拒绝空集合、不匹配shape/length、非有限输入、非法code/scale/zero、目标越界、空mask；JSON不要写非标准NaN/Infinity，未定义指标用null+reason。

### 不能省略的自测

- RNE .5、1.5、2.5、-.5、-1.5；加整数zero的位置用.5/z1反例。
- pack标准向量0..15应得到hex `[10,32,54,76,98,ba,dc,fe]`；不要只roundtrip，因为encoder/decoder同样错也能往返。
- 故意换nibble、改scale、错zero得到失败，作为注入故障自测，不修改实际用户artifact。
- [1,2]vs[2,4]cosine例、上述clamp例、mask计数、batch加权PPL。
- p=[.5,.25,.25],q=[.25,.5,.25]的KL=0.17328679514nats；同分布KL≈0。
- 末尾odd count padding须保存logical length，不能把补零也算进误差/目标数。

以上数值已用Node标准算术执行，**完整Python工具尚未由本代理实现/运行**。审核代理应实际执行新工具与错误输入，并检查自测的失败例不被catch后误报为成功。

## 五、可选GPU操作：具体但不编造结果

先固定backend/version/GPU/driver/CUDA/模型revision与shape；查真实packed格式/组大小。微基准同时有：高精度FP、quant-reference、realpackedkernel。区别量化误差、kernel parity、kernel速度三条结果，不能只把fakeQ/DQ后F.linear称INT4/INT8性能。

1. 在目标GPU预热到compile/缓存/graph稳定，cold-start另外记录。
2. `torch.cuda.synchronize()`后开始wall-clock，执行已固定次数，再synchronize后停止；或在明确同stream依赖下使用CUDA events。event时间不是CPU调度/HTTP总延迟，多stream要建立正确依赖。
3. 不在计时循环里临时加载、下载、打印、转CPU、调用`.item()`，除非这些就是明示测量范围。明确是否含activation量化、unpack、epilogue、分配与通信。
4. reset peak、按阶段记录allocated/reserved/peak；baseline和quant最好隔离进程，避免两套model共存污染。nvidia-smi、tensorallocated、allocatorreserved含义不同。
5. 覆盖M小decode/大prefill、实际K/N、TPshard、tail、bias、group与fallback；trace核实实际kernel。计数覆盖率需要给分母，按层数、调用数、耗时或FLOPs不是同一个量。

没有GPU时把上述明确列为“待运行步骤”，不要输出假耗时/利用率/显存。CPUtoy不提供相应结论。

## 六、模型与服务验收

固定calibration集与独立eval集（不要校准后再拿同一集合挑最佳recipe并当泛化证据）。manifest记录sampleIDs/hash、采样顺序、模板、tokenizer版本、truncation、position、是否teacherforcing、模型与adapterrevision；seed只是一项而非充分条件。

lm-eval命令应指定已固定的task集合、revision或本地不可变artifact、seed、batch、fewshot、chat模板，保存log_samples与show_config。`--limit`只能smoketest，不把前10样本得分宣传成完整benchmark。缓存要按artifactrevision隔离，避免quant运行读到FP结果；不用tool自动把结果上传外部hub。

服务先固定权重/激活/KV/scheduler/并行/输入输出长度及到达过程，一次只改一个量化轴。分别测“固定工作集”与“固定HBM预算”：后者可能来自更少eviction/更多并发，而不是单个kernel更快。

TTFT要声明客户端请求开始到首token的范围；平均TPOT常取 `(t_last-t_first)/(n_output-1)`，n<=1未定义，应分开统计。streaming chunk可能含多个token，chunk间隔不等于逐token间隔。

goodput必须声明单位和定义：本篇可以报告 `SLO-qualified completions/time` 与 SLO attainment；DistServe所称最大可承载请求率是“满足既定attainment目标时的最大offered rate”，不是对某次运行的合格完成数除时间直接同义。token/s也不是request/s。记录timeout/error、P50/P95/P99、repeat次数与负载生成方式，不隐去失败请求。

质量与性能阈值应预先按任务风险确定，不将cosine0.99、PPL变化1%或单一平均延迟冒充通用安全线。任何“通过”仅覆盖所测任务/形状/分布，不能保证所有文本事实无误。

## 七、本篇详细结构

1. 例子开场：文件更小、回答看似相似，为什么仍不足以上线。
2. 一张六关证据图（上表即可），贯穿同一个artifact。
3. 准备不可变manifest与BF16基线；明确数值、质量、性能是三个独立维度。
4. 数值关：contract、packing已知向量和注入错误，运行CPU工具。
5. 用cosine/clamp反例解释为什么要看多种张量指标。
6. 同prefix logits、KL、teacherforcing与自由生成分叉。
7. 用mask/加权反例讲PPL，不堆通用指标名。
8. 真实kernel parity与实际低精度执行；可选GPU计时，明确尚未实测。
9. 任务质量、校准/评估分离和版本化harness。
10. 服务负载、SLO、容量vs带宽收益，定义goodput而非直接引用倍速。
11. 最终输出一份带pass/fail/skipped/incomplete和证据链接的验收报告；回顾系列各篇提供哪一块，而不是再罗列算法优劣排行榜。

研究未完成/未执行的部分：完整新Python工具、真实checkpoint导出/解包、模型任务eval、GPUkernel与端到端服务bench。写作/审核阶段只可按其实际新证据更新状态，不可将本研究中的建议步骤改成完成时叙述。

## 八、工具前置补查：紧凑 v1 输入与失败契约（本轮冻结）

此处仅约定第12篇写作时可实现的小工具，不现在新增工具、依赖或通用验收框架。

### CLI 与最小 JSON 边界

CLI 只提供互斥的 `--self-test` 与 `--input PATH` 两种模式；未知选项、两者同时出现或都不选应报usage错误，不能因拼错input选项而悄悄运行toy。读取指定本地文件一次为原始bytes，严格UTF-8解码；不执行内容、不下载文件。

v1顶层必须是object，`schema_version`必须是整数1；只允许下列字段，除版本外至少有一个数据组。不存在的数据组标`skipped`，不猜测默认数据。每个存在的组必须完整提供该组字段，不接受额外拼写/未知字段；仅`external_evidence`可保留外部JSON metadata，但不执行、不合并到本地验证状态。

| 数据组 | 最少字段与语义 |
| --- | --- |
| `tensor` | `reference`、`candidate`：等长、非空的一维有限数值数组；只计算此flattened样本误差，不声称保留层/head/shape语义 |
| `logits` | `reference`、`candidate`：相同N×V、非空、非ragged的有限数值二维数组，V>=2；`targets`：长度N，整数且0<=target<V；`mask`：长度N且每项恰为JSON boolean，至少一个true。每行已对齐“一个要评分的next-token target”，工具**不再自动shift一次**；不支持用-100代替此明确mask |
| `uint4` | `codes`：0..15整数数组；`packed_bytes`：0..255整数数组；`logical_length`：正整数，等于codes长度；bytes长度恰为ceil(logical_length/2)。`scale`：有限正数；`zero_point`：0..15整数；`rounding`固定字符串`ties_to_even`；`layout`固定`low_first`。只支持这条per-tensor affine UINT4契约，不声称兼容AWQ/QServe/NF4/FP4真实布局 |
| `external_evidence` | 可选object，仅列为`reported_unverified`；即使包含`kernel_pass:true`、GPU名或链接，也不能把`gpu_kernel_verified`/模型质量/服务验收自动改为true |

整数校验应使用 `type(value) is int`，不是 `isinstance(value,int)`；所有code、target、logical_length、zero_point、schema_version都拒绝bool和浮点1.0。有限实数校验也要排除bool，不接受数字字符串、null或隐式转换。以上是v1有意收窄的输入范围，非宣称这些情形在其他系统没有合法用法。

JSON解析需拒绝重复key与`NaN`/`Infinity`常量；`parse_constant`拒绝不足以兜住所有非有限数，`1e400`按默认float解析也会越出binary64范围，仍需逐数值字段检查。输出使用严格JSON，不能偷偷写NaN/Infinity。不要引入JSON Schema库，几项显式检查足够。

### 数值失败与证据身份

- 有限输入不保证有限中间值：如 `1e308/1e-308`、`1e308-(-1e308)`、平方误差/求和、`exp(NLL)`均可能越界。转换float、Q/DQ商与重构、误差累加、log-softmax/NLL/KL及最终PPL各阶段应检查所需值/分母；遇OverflowError、非有限中间值或本应正值的scale/归一化分母变0，返回带阶段/字段的`numeric_error`，不返回成功指标或用clamp/epsilon掩盖。
- 正常softmax里极小项的exp下溢到0不必一概报错；应使用稳定log-softmax，不能随后直接log(0)。若有限输入的跨度仍超出该实现的binary64能力，明确报告不支持的数值范围，而非把错误解释成模型质量下降。零reference导致cosine/相对误差未定义等既定情况仍用null+reason，与数值计算失败区分。
- SHA256绑定**实际读入的原始bytes**，不对解析后重新dump的JSON计算后再冒称原文件hash；空白、行尾变化可改变hash。hash说明这次算了哪些bytes，**不证明来源真实性、导出过程正确、模型revision属实或GPU曾执行**。外部证据字段不能提升此信任层级。
- 退出0只表示当前CPU作用域的计算/自测完成，报告仍含`verified_scope`与全栈`incomplete`；非法CLI/schema用非零usage/input状态，数值失败、packing parity失败或自测断言失败也必须非零。不要让总`accepted:true`盖过未执行的GPU/模型/服务关卡。

### packing 与负向自测必须真的使错误路径失败

已知codes0..15必须与独立固定bytes `[0x10,0x32,0x54,0x76,0x98,0xba,0xdc,0xfe]` 比较，不能仅encoder/decoder互相往返。奇数尾部规定未使用的high nibble为0；如codes[0,1,2]、logical_length3，bytes须为[0x10,0x02]。解码先验证bytes数量与padding nibble，再只取前logical_length项；padding不计入误差或token数。

负向用例应实际调用同一validator/parity逻辑，逐个断言：替换nibble、logical_length与bytes数不符、非零padding、bool code/target、坏scale、非有限/中间溢出均被预期的具体错误拒绝。合法但被改过的scale/zero不一定违反schema，应在带已知重构值的fixture上断言parity失败，而非误称所有scale变化都非法。使用“预期失败”帮助函数时必须记录确实捕获匹配的错误类型/代码；若函数没有失败，或抛出无关异常，则自测失败，不能用裸`except: pass`吞掉后计成功。

### 本次实际阅读及未执行范围

实读CPython3.13.7官方固定文档相关段：[json](https://github.com/python/cpython/blob/v3.13.7/Doc/library/json.rst)的load/parse_constant/object_pairs_hook、重复key、allow_nan；[stdtypes](https://github.com/python/cpython/blob/v3.13.7/Doc/library/stdtypes.rst)的bool为int子类型与float转换；[math](https://github.com/python/cpython/blob/v3.13.7/Doc/library/math.rst)的isfinite与域/溢出异常；[hashlib](https://github.com/python/cpython/blob/v3.13.7/Doc/library/hashlib.rst)的bytes输入和digest；[argparse](https://github.com/python/cpython/blob/v3.13.7/Doc/library/argparse.rst)的互斥参数与错误退出。上述schema与失败策略是本工具的设计选择，不冒称Python默认已经提供全部严格校验。

本次没有实现/执行Python工具、JSON validator、CLI、hash校验或故障注入；也没有运行Torch/模型/GPU。后续writer逐篇至第12篇才实现，auditor需实际执行这些正负向用例。研究文件仅此补充，本轮冻结，不自行提交。

### Root 补充的设备计时边界

Root 另实读上述 PyTorch 2.8 CUDA notes 的 Asynchronous execution/CUDA streams 段，以及同版本 [`torch/cuda/__init__.py`](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/cuda/__init__.py) 的 `synchronize` 定义。无参数调用只等待当前设备的所有 streams，不自动等待所有 GPU 或分布式 ranks。若示例扩展为多 GPU/TP，需为参与设备和 ranks 建立正确的完成依赖与测量口径，不能把单个 device event 当整个请求的分布式耗时；本文可以只给明确限定为单设备的可选示例。这里只读源码，未实际执行 CUDA。

## 九、审核代理独立预读与待执行核验设计

2026-10-15，仍只读准备，未实现工具或修改正文/测试，未对尚未写成的工具宣告通过。独立实读范围：ONNX 页面中明确标为 opset13 的 summary/axis/输入类型段（不是当前高 opset）；PyTorch2.8 CUDA notes 的 Asynchronous execution、streams、Memory management 以及 synchronize 定义，randomness 文档全文；Transformers4.55.4 perplexity 文档全文；固定 lm-eval `docs/interface.md` 全文；DistServe v3 引言关于 TTFT/TPOT、SLO attainment、per-GPU goodput 定义。另读 CPython3.13.7 json 的 parser hooks/默认重复key/非标准常量/编码、argparse 的 allow_abbrev/required互斥组/error退出、stdtypes 的 bool子类、math 的 isfinite/exception、hashlib 的 bytes/digest 对应段。没有运行 PyTorch、lm-eval、真实模型、GPU 或服务 benchmark。

### 工具设计需要明确而不能由默认库行为代劳的边界

- `ArgumentParser` 默认 `allow_abbrev=True`，所以严格 v1 CLI 应显式 `allow_abbrev=False`；使用必选互斥组，不使用 `parse_known_args` 忽略未知选项。成功报告的 stdout 应是一份严格 JSON，诊断信息留 stderr，避免自测 PASS 文本污染机器读取。
- 若限定 UTF-8，应先对实际读入的 raw bytes 作严格 UTF-8 解码，再 JSON parse；直接把 bytes 交给 `json.loads` 还会按文档接受 UTF-16/32。object_pairs_hook 要拒绝任意层级重复key；parse_constant 不拦截普通浮点字面量 `1e400`，也不能把浮点1.0等同于整数1。
- `external_evidence` 的宽松 metadata 也必须能安全输出为严格 JSON；不能让其中 `1e400` 变成 Infinity 后在最后 dump 时才出现未处理异常。可以在解析层或递归metadata检查中明确拒绝超出工具数值域的数值，不能借其“未验证”身份绕过输出约束。至少一个数据组应明确指可计算的 `tensor` / `logits` / `uint4`，仅外部metadata不应算一次本地数值验收。
- 相同 N×V、targets/mask 长度正确，只证明形状合同，不证明两组 logits 实际来自同 tokenizer、同前缀、同 position/mask 或同模型revision。工具需把这些语义对齐作为调用者前提/外部报告，不能自动生成 `teacher_forcing_verified:true`。同文件的几个数据组也不天然证明全部来自同一份真实 checkpoint。
- hash 仅绑定此次实际读到的 bytes；相同内容拥有相同 SHA256，不认证作者、模型revision、导出过程、GPU执行或任务质量。外部字段不得改变本地 `incomplete` 结论和已验证作用域。
- Python 的普通浮点算术有时产生 Inf，有时由 math 函数抛 OverflowError/ValueError；不能仅抓异常而漏掉非有限结果，也不能仅入口 isfinite 后不再检查所需中间量。合法 zero-reference 未定义指标应与 numeric_error 分开。

### 后续独立测试计划（当前未执行完整工具）

- 以 Fraction 候选码距离/偶码选择核 RNE+整数zero顺序；用固定十六进制向量而非同实现 round-trip 核 low-first UINT4，逐项要求坏nibble/padding/length真正触发匹配的失败代码。
- 以固定重构fixture检查“合法但错的 scale/zero”，不把任意合法scale改变都称 schema 错误。再检查bool/int、未知字段、重复key、空mask、形状、严格JSON以及实际rawbytes hash；错误CLI、numeric_error、parity失败必须非零退出。
- 当前研究中的 KL 示例是置换对称的，用来检查方向不够敏感。计划另用 p=[3/4,1/4]、q=[1/2,1/2] 的高精度 Decimal oracle，正反方向不同；并用 reference logits=[0,0,-1000]、candidate=[0,-1000,0]、target0，检查稳定 log-softmax 得 KL≈500、两侧PPL≈2。它允许极小softmax项下溢，但不应直接对0取log或为掩盖错误而clamp概率。
- 对 PPL 直接检查实际shift后的评分mask计数，再确认 v1 工具输入已经是对齐的 next-token 行、不会重复shift；用不同有效数的窗口检查加权 NLL，而不是平均各段 PPL。
- 以只含 `external_evidence.kernel_pass=true` / GPU名 / 链接的伪造报告验证不会提升 GPU/model/service 证据；未执行关卡仍 skipped/incomplete，不出现笼统 accepted=true。

上述只是之后正式接收静止工具时的核验设计。现在未创建专项、未实际执行CLI/validator、未声称输出已满足这些要求。
