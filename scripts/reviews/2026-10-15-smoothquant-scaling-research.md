# SmoothQuant 既有文章逐篇调研交接

日期2026-10-15，目标 `_posts/2026-08-02-smoothquant-w8a8-activation-outliers.md`，保留URL，系列50/技术2022。已读完现有文章，尤其先前alpha端点与GEMM K轴限定，不能在结构改造时退回错误口号。

## 真正阅读的来源

- [SmoothQuant arXiv v7](https://arxiv.org/html/2211.10438v7)：实际逐段读§2–4、§5.1、§5.3–5.5及§7相关文字/表2；不复现模型实验。v7是2024修订，首发仍2022-11-18。
- 官方repo当前main经GitHubcommit API固定为 **`c61476d728e42ae0d8a35e7e78494edcac3237b5`**，提交2024-07-12：
  - [smooth.py](https://github.com/mit-han-lab/smoothquant/blob/c61476d728e42ae0d8a35e7e78494edcac3237b5/smoothquant/smooth.py)：实际读完整147行，含LayerNorm/RMSNorm folding和OPT/Llama/Mixtral分叉。
  - [fake_quant.py](https://github.com/mit-han-lab/smoothquant/blob/c61476d728e42ae0d8a35e7e78494edcac3237b5/smoothquant/fake_quant.py)：实际读Q/DQ functions、W8A8Linear、model转换的相关段落。其weight buffer原本float16、forward调F.linear，是fakequant模拟，不是文件名写W8A8就已经INT8 TensorCore。
  - [calibration.py](https://github.com/mit-han-lab/smoothquant/blob/c61476d728e42ae0d8a35e7e78494edcac3237b5/smoothquant/calibration.py)：已打开原文，准备稿件引用具体收集逻辑时应再定位对应函数；本交接只将论文calibration统计作为证明，不把整文件称已逐行读完。
- 现稿三条TensorRT `latest`链接本轮尚未细读；如继续支撑具体current backend支持须固定版本核查。纯通用Q/DQ数学不需依赖未读当前页面。

## 核心契约与alpha限定

维度沿系列 `X(M,K),W(K,N)`，positive finite smoothing `S=diag(s_k)`；
`XW=(X S^-1)(S W)`。

令 `a_k=max_m|X_mk|>0,b_k=max_n|W_kn|>0`：
`s_k=a_k^alpha/b_k^(1-alpha)`, `a'_k=(a_k b_k)^(1-alpha)`, `b'_k=(a_k b_k)^alpha`。

因此：alpha0不是identity，而是s=1/b；alpha1是s=a；alpha0.5平衡两个range统计，不证明真实误差最优。`ds/dalpha=s log(ab)`，当ab<1时scale随alpha增大反而减小；现稿已正确举64×0.01例，必须保留。

官方代码weight_scales先clamp到1e-5，最后s也clamp到1e-5；这些防零约定使zero channel不再直接套正值纯公式。不同backend的epsilon与scale数据精度可不同，不能声称与理想式逐bit相同。

## Claim/source/审阅意见

| Claim | 证据与判定 | 修改建议 |
| --- | --- | --- |
| activation outlier与weight较平稳 | 论文§3对所测层的观察 | 不断言任意模型weight都易量化或直到SQ才可W8A8 |
| input-channel scaling等价 | §4式3；smooth.py | 正确；smoothing factor不是编码step |
| O1/O2/O3 | 表2三者weight均per-tensor；activation依次per-token dynamic、per-tensor dynamic、per-tensor static | 若加表须准确，不能把O1weight默认改成per-channel |
| activation per-channel沿K难直接epilogue缩放 | §3式2 | 正确；不是“任何硬件永远不支持”，限定传统单次INT8 GEMM路径 |
| fold LayerNorm | smooth.py L37–40 | gamma和beta都除同s；所有共享consumers乘s，不是只一个q_proj |
| fold RMSNorm | L62–64 | gamma除s、没有beta；不把RMSNorm所有结构当LayerNorm |
| sharedQ/K/V | L69–77/L115–123 | weight统计在所有fcs合取max，变换同步；勿对同一norm为每分支独立重复修改 |
| residual/nonlinear | §4式3后文字；graph代数 | 不能任意穿过GELU/SiLU，不同分叉要尊重拓扑；现稿已有边界，建议用一例具体化 |
| fakequant vs真实INT8 | fake_quant.py Q/DQ+floatbuffer+F.linear；论文§5.1区分CUTLASS INT8实现 | 新增一小段读源码，防止读者直接运行fakequant然后宣称加速 |
| alpha0.5/0.75 | §5.1指定OPT/BLOOM/GLM；§5.5消融 | 仅论文经验不是allmodel常数，现稿合理 |
| smoothing后activation仍需Q | 论文§4/§5.1 | 正确，静态scale省runtime统计，不是省掉Q本身 |
| SQ W8A8不自动KV8 | 参数/状态位置的区分 | 保留；论文的INT8BMM也不等于缓存物理存储自动INT8 |
| 整体memory减半 | 论文历史实验有不同batch/seq，KV/activation随流量 | 现稿已保留整体内存组成，不扩大理论weight比例 |
| “SQ§3.1式2”链接v1 | 本轮读v7，相关outer-scale公式位于§3式2；v1编号需另核 | 建议统一改读过的v7定位，避免章节与文章正文不一致 |

## 对本篇的具体结构优化

1. 开头“直到SmoothQuant之后才更容易...”易抹平ZeroQuant/LLM.int8并行研究，改成具体“INT8 GEMM已有，激活异常通道仍让简单W8A8失效；SQ选择离线变换而非运行时分路”。
2. 不重教整套基础Q/DQ，可保留最短对称式与40/127算例，并链接系列基础。
3. 在等价式后立刻给两通道完整before/after与quantized output例，使数学关系不悬空。
4. alpha节保留已审核端点/ab<1反例，但重点落在输入/权重范围乘积，不把导数写成主要正文负担。
5. folding节给LayerNorm gamma+beta和Q/K/V共享因子，代替只说“前序参数可以合并”。加入忘记beta或一分支的失败反例。
6. 解释论文O1–O3是不同activation recipe，不是算法三个阶段；optional小表可用。
7. 确认fake_quant.py不能当真实INT8性能程序；最末部署大清单可缩为本篇特有验证，端到端实验规范留最终篇。
8. 给可直接跑的stdlibtoy，不把现有capture_linear_input/zeros示意Python块称实际可执行；示意块可转text或清楚标伪代码。

## 已执行构造

Node标准库、自写RNE执行（非PyTorch/GPU）：

- `X=[10,1],W=[0.1,1]^T`, baseline2；alpha0.5给s=[10,1]，X'=[1,1],W'=[1,1]，无量化exact2。
- 用教学对称码[-7,7]（不是说SQ变成4bit），对原X/W分别absmax Q/DQ，输出2.857142857；对平滑X'/W'同量化得2。这个构造只展示可改善，不证明alpha0.5一般最优。
- `a=64,b=0.01`,alpha0/0.5/1得到s100/80/64，反驳“alpha越大每通道s必越大”。
- 建议LayerNorm folding反例：设normalized z=[1,-1],gamma=[2,3],beta=[0.5,-0.5]，out=[2.5,-3.5]；S=[10,1]，gamma'/beta'都除s才产生[0.25,-3.5]。若漏beta0.5，则首通道变0.7而非0.25，后续乘s后不等价。
- 建议共享consumer反例：同X送Wq/Wk/Wv，X被除s后只改Wq，则K/V输出改变；正确需一起处理共享norm所有用途。
- 建议positive randomS和多M/K/N验证XW与X'W'，加bias应保持原linear bias不变（fold的是前序norm beta，不是当前linear bias盲目乘除）。

## 验证边界

未量化真实模型，未执行官方Torch或CUTLASS INT8、QAT、TP或serving性能。源码核验只是确保文章解释与指定版本相符。最终稿的stdlib算例需auditor直接提取执行；后续真实模型实验应保持calibration和evaluation分离。
