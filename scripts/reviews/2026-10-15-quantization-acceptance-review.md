# 量化验收终篇及 CPU 工具独立审核记录

日期：2026-10-15。对象：`_posts/2026-10-15-quantization-acceptance-engineering.md`、`scripts/quantization-acceptance.py`。收到静止稿后完整读正文、工具全部函数/自测/CLI，再实际执行正文和工具；研究预读与作者自测不代替本次独审。

## 一、一手来源与实际阅读范围

- [ONNX QuantizeLinear opset13](https://onnx.ai/onnx/operators/onnx__QuantizeLinear.html#quantizelinear-13)：读明确标记为13的 summary、INT8/UINT8范围、最近偶数、axis/输入类型。工具 UINT4 是独立合同，不反向声称这个 opset 已定义 UINT4/FP4。
- PyTorch2.8.0 固定 [`CUDA notes`](https://github.com/pytorch/pytorch/blob/v2.8.0/docs/source/notes/cuda.rst)：异步执行、streams、Memory management；读 [`torch.cuda.synchronize` 定义](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/cuda/__init__.py)。无参数只同步当前设备，不等于所有GPU或ranks。另完整读 [`randomness.rst`](https://github.com/pytorch/pytorch/blob/v2.8.0/docs/source/notes/randomness.rst) 对seed、确定性、版本/平台边界的说明。
- Transformers4.55.4 [`perplexity.md`](https://github.com/huggingface/transformers/blob/v4.55.4/docs/source/en/perplexity.md)：全文，包括context/stride/tokenizer与示例。独立核出第一列已masked时，不能无条件从有效labels计数再减batch。
- lm-evaluation-harness 固定 `452749513f817315042df9286241a61051392470`：完整读 [`docs/interface.md`](https://github.com/EleutherAI/lm-evaluation-harness/blob/452749513f817315042df9286241a61051392470/docs/interface.md)，定点读README模型revision、缓存与logging；未执行任务或逐行认证整个评测库。
- [DistServe v3](https://arxiv.org/html/2401.09670v3)：读引言TTFT/TPOT、SLO attainment与per-GPU goodput定义。最大可承载offered rate与单次合格完成数/秒没有混为一谈，未引用其倍速作为本机结果。
- CPython3.13.7 固定文档的 [`json`](https://github.com/python/cpython/blob/v3.13.7/Doc/library/json.rst)、[`argparse`](https://github.com/python/cpython/blob/v3.13.7/Doc/library/argparse.rst)、[`stdtypes`](https://github.com/python/cpython/blob/v3.13.7/Doc/library/stdtypes.rst)、[`math`](https://github.com/python/cpython/blob/v3.13.7/Doc/library/math.rst)、[`hashlib`](https://github.com/python/cpython/blob/v3.13.7/Doc/library/hashlib.rst)：实际读parser hooks、非标准数字/重复key、bytes编码、required互斥/allow_abbrev、bool为int子类、浮点转换/异常、原始bytes digest对应段。严格v1是工具的明确设计，不称Python默认就提供这些保证。

来源预读已记录在research §9；本轮完整读最终正文/实现，复核版本与使用上下文，没有为增加数量重复抓取无关来源。未运行上述Torch、HF模型或lm-eval服务。

## 二、正文概念、公式与作用域

1. 六关区分浮点图改写、Q/DQ算法误差、packing布局、真实kernel parity、任务质量、服务收益；真实kernel先与相同量化数据参考对齐，再做BF16总回归，不混淆两种误差。
2. 不可变revision/manifest、校准与评估隔离、模板/tokenizer/position/context/seed各自含义正确。shape相同不认证语义前缀相同，同JSON数据组不认证同一checkpoint来源。
3. SHA256绑定本次实际raw bytes，不认证导出者、模型revision或真实GPU执行；外部metadata原样reported_unverified，不会修改本地verified_scope或全栈incomplete。
4. MSE、relative_squared_l2明确分母，cosine只看方向。零reference/零norm返回null+原因，不加epsilon；非零平方或正误差比值下溢不冒充0。
5. code_clamp、endpoint、prequant_range_exceed的三种比例不同；[0,7,7.2,100]分别1/4、3/4、1/2正确。RNE(x/s)+z与RNE(x/s+z)在odd-zero tie不能互换。
6. low-first UINT4 golden bytes及odd high padding0正确。v1 UINT4没有原始浮点source，不计算源量化误差或source clamp；rounding只是声明，不证明历史encoder真的用过它。合法scale/zero变化不必schema失败，只有独立重构fixture才可指出其不匹配。
7. teacher-forced同前缀和已对齐next-token行的前提清楚。mask应用到同一批KL/top1/NLL/PPL；false行仍先作schema校验，但合法false行不应参与数值计算。N=1不再次shift。并列top1固定最低索引。
8. KL方向为reference||candidate、温度1。稳定log-softmax不先exp再log；[0,0,-1000]与[0,-1000,0]的KL500/PPL2成立。新数值实现的非负等价式与下述舍入范围已明示。
9. PPL按总NLL/实际有效目标数计算；ln2×1和ln4×3得到2^(7/4)，不是不加权loss平均。shifted labels反例各2而不是各1正确；tokenizer/上下文/窗口协议不同时不直接比较PPL。
10. GPU/服务段只有待执行操作，没有虚构耗时/显存/精度。单device同步、event的stream依赖、HTTP总延迟、allocated/reserved/峰值、fake Q/DQ不等于低位kernel等边界正确。TPOT仅n>1定义、chunk≠token间隔；goodput定义、失败分母及回滚方案均有明确范围。
11. 按root/research计划补上预设质量/错误/SLO触发条件、不可变已知可用baseline及配套tokenizer/recipe/runtime、路由切回后的健康/指标复核、失败artifact与日志留存。明确是待真实服务演练的策略，不是CPU工具自动回滚或已完成部署。

## 三、工具完整审读及本轮实际修正

### 既有实现经独立执行确认的边界

- 必选且互斥 `--self-test`/`--input`，`allow_abbrev=False`；不存在默认toy模式。unknown/missing/abbrev/both的真实CLI都非零退出。
- Path.read_bytes只读一次；严格UTF8，重复key在任意层拒绝，parse_constant拒绝NaN/Infinity，parse_float同时拒绝1e400与非零字面量下溢0。真正零mantissa即使指数非零仍接受。
- 先validate整个文档再算任何一组；版本/codes/targets/长度/zero为严格int而非bool/float，mask严格bool；unknown/missing/ragged/非法字节与全false等失败。外部metadata不借“不验证”身份绕过strictJSON解析。
- input_error退出2、numeric/parity/selftest失败退出1，stdout成功只有JSON，失败诊断在stderr；结果没有accepted:true，总状态incomplete，未提供组skipped。
- SSE及ref/cand norm的非零分量平方、正MSE/relative quotient下溢已有保护；本代理独立通过CLI触发，未以作者自测代替核验。整数/小数/字面量范围是工具有意收窄的binary64合同，不是任意精度指标库。
- pack/unpack、affine_quantize是内部reference helper；外部v1入口先完整validate。没有将内部helper误写成支持任意参数的通用checkpoint读取器。

### 1. KL抵消导致微小负散度

独立构造普通4维近同分布：

```text
reference = [2.452614615935712, -0.10910710576120053,
             3.6311162686894303, 2.9003006069414257]
candidate = [2.4526146219971254, -0.10910710290931795,
             3.631116278287903, 2.9003006017458595]
```

原直接Σp(lp-lq)返回约-6.01559e-17；80/90位Decimal从原始浮点logits重新归一化的真值约+1.976138120905224e-17。这是抵消问题，不是模型产生“负KL”。

修复采用归一化时数学等价的 `Σ[p log(p/q)+q-p]`。d=lp-lq：

- d>=0：p·[d+expm1(-d)]。
- d<0：q·[d·exp(d)-expm1(d)]。
- |d|<1e-3：**无论d正负均取p侧权重**，用 `d²(1/2-d/6+d²/24-d³/120+d⁴/720-d⁵/5040)` 计算p侧kernel，避免零附近再次抵消。n>=8的省略项在阈值内低于binary64舍入量级；测试检查阈值两侧及d两种符号，没有混用q侧权重。
- 在log域组合log_weight+log(kernel)再exp；d0单独0，kernel非有限/负或非零d的kernel意外下溢拒绝。所有指数函数只接非正/可计算输入，不为控制结果随意clamp概率或无条件max(0,KL)。

修复后近同例约+1.976138122956151e-17，与独立原始logits oracle接近；差异来自binary64 log-softmax等舍入，非任意精度证明。另独立检查ref[0,-1000]、cand[0,-1e308]，先exp(lp)会丢掉一个最终仍可表示的加权项；log域组合保留约5.07595889754946e-127。数学等价要求归一化概率；这里由工具log-softmax构造，浮点近似/指数项最终下溢仍有binary64分辨率限制，不保证检测每个理论上非零的分布差异。

### 2. cosine非零正负比值下溢

t=2^-537，ref=[1,1,t]、cand=[1,-1,±t]：非零平方仍可表示，但最后cosine商会变±0。把cosine也接入ratio保护，并令ratio检查非零分子而非仅正分子；现在两种符号都numeric_error。未把真实零dot误判为失败。

### 3. JSON整数转float造成输入假零

ref=[9007199254740993]、cand=[9007199254740992]原会在real转换时合并成同一float、MSE0。计算字段现在拒绝 `int(float(value))!=value` 的整数；并不笼统限值2^53。2^53+2可精确表示，实际CLI仍通过。外部惰性metadata中的10^400不参与float转换，仍可strictJSON往返保留。正文明确普通小数字面量照binary64舍入，不承诺十进制任意精度。

### 4. 自测增加实际回归

从作者交接54项增至最终60项：近同分布正KL、tiny概率但可表示的加权KL、cosine正负ratio下溢、inexact整数拒绝与exact大整数接受。计数与实际names长度一致；并非以增加自测数量替代独立专项。

## 四、独立专项与真实命令行

专项：`scripts/reviews/oct15-quantization-acceptance-examples.mjs`。直接提取正文2个唯一Python块及1个JSON fixture；工具整体通过真实子进程CLI检查，并辅以有限的内存故障注入/read探针。

```powershell
$env:BLOG_PYTHON = 'C:\Users\iStar\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
node scripts/reviews/oct15-quantization-acceptance-examples.mjs
node scripts/audit-posts.mjs
```

最终7组、180次真实CLI全部通过：

1. 正文2Python原块实际执行、JSON fixture原字节交给工具；真实--self-test通过60项且仍incomplete；missing/unknown/abbrev/both/文件读取错误的退出与输出通道。
2. 全组missing/unknown/type/bool/shape/code/scale/target/mask参数，顶层/nested重复key、严格UTF8、NaN/Infinity/1e400/±1e-400；真正0.0/-0.0/0e400/-0.000e-400/0.00e+9999均接受。2^53+1及负号拒绝、2^53+2接受、外部10^400整数安全保留。
3. 中文UTF8内容、空白/LF/CRLF/字段顺序的实际文件hash分别与独立hashlib对应；main路径读探针在第二次read即失败，确认实际只读一次且hash同一rawbytes。external的kernel_pass/accepted/verified_scope/GPU名只在reported_unverified内保留，不升级本地范围；未提供组skipped。
4. Fraction相邻整数距离/偶码oracle，独立核RNE+zero与三clamp比例；固定golden bytes、全部256物理byte及独立码拆分、多个odd长度/padding、nibble/parity失败。合法scale/zero改变仍parity_pass但与known-DQ不同；codes与bytes一致更改可通过但不认证来源。
5. Fraction精确SSE/MSE/relative/max与80位Decimal cosine，正常/零norm反例；差值/平方/求和/DQ溢出、norm和mixed-magnitude平方下溢、正MSE/relative商下溢、cosine正负商下溢的真实非零numeric_error。
6. 从原始logits独立用90位Decimal构造概率/log概率并算原定义KL、NLL/PPL，不复用修正后的非负kernel；双方向非对称KL、500/PPL2、N1、共享mask、近同分布、series阈值两侧/两种符号、tiny加权项、按实际token加权PPL、top1 tie。mask=false的有限极大跨度行不计算，但非法target仍先schema失败。
7. 内存中将同一parse_document对特定duplicate fixture改为“不失败”或“抛错误code”，实际self-test必须转成selftest_error，不能误计成功；恢复函数后继续。整个doc先schema验证的故障注入：早组数值溢出不能掩盖后组非法target，实际返回后者input_error而非开始计算。没有修改真实输入或用户artifact。

专项首次尝试系统默认临时目录时受本机沙盒权限拒绝，尚未调用工具/生成有效fixture；已改为workspace内独立临时目录，验证解析后的绝对路径仍在workspace内，并由本轮测试清理。生产工具没有因此增加文件写入或改变数据读取语义。最终专项所用文件仅是自动生成的合成CLI fixture，不是模型数据。

站点静态审核通过98篇、10series、24tags。正文/工具冻结后告root进行生产构建/MathJax/浏览器/发布检查；静态和CLI证据不能替代这些步骤。

## 五、执行边界与正式移交

执行仅限标准库CPU数值、严格输入/CLI/哈希、合成fixture、独立Fraction/Decimal和故障注入。工具不联网下载模型、不执行输入内容、不更改服务；本轮未运行Torch/HF模型、真实checkpoint、GPU内核/trace、校准/任务质量、HBM/吞吐/延迟或线上回滚。真实模型/服务仍incomplete，不把来源hash或外部声称提升为真实运行证据。

正式移交本篇post、Python工具、专项及本审核记录四文件，审核代理停止写入。root可补最终发布QA；本代理没有修改README/计划/其他文章，也未提交或推送。

## 六、主代理最终检查

主代理完整阅读正文、工具、独立专项及最终修订，独立运行7组专项（180次真实CLI、60项工具自测），全部32个JavaScript套件、既有Python、98篇元数据与历史coverage、diff检查均通过。生产构建生成113个HTML页面、138个文件；91篇数学文章的1399个行内与1031个独立公式通过结构检查。浏览器实际读回本地36个MathJax容器、0个错误，无美元残留和整页横向溢出；PPL推导的实际截图可读，回滚段与GitHub工具链接正确。README仅补充工具用法和CPU作用域说明。待单篇推送后另记exact-head CI与线上证据；没有扩大模型/GPU/服务验证范围。
