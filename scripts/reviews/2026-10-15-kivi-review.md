# KIVI 独立审核记录

日期：2026-10-15。审核对象：`_posts/2026-10-15-kivi-kv-cache-axes.md`。作者静止稿交接后独立阅读全文、公式、三个实际 Python 块及参考资料，并作下述局部修正。不是仅根据研究交接或作者自测宣告通过。

## 一、实际读取的一手来源

- [KIVI arXiv v2](https://arxiv.org/html/2402.02750v2)：§2、§3.1–3.3、§4.1、§4.2.4、Appendix A Algorithm 1。首发 2024-02-05，v2 修订 2024-07-25；没有复现论文任务分数或吞吐图。
- 官方固定提交 `876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6`（2025-11-20）：
  - [`models/llama_kivi.py`](https://github.com/jy-yuan/KIVI/blob/876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6/models/llama_kivi.py)：普通 Attention 限制，以及 Flash 路径 decode 的 logits 拼接、K 封存、全局 softmax、V 输出与 FIFO 顺序；另读 prefill 的原始浮点计算和缓存准备。未逐行审核所有模型类/训练/生成路径。
  - [`quant/new_pack.py`](https://github.com/jy-yuan/KIVI/blob/876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6/quant/new_pack.py)：完整读取 Key/Value 分组、min-max、pack/unpack、Triton 最后一维路径。
  - [`quant/matmul.py`](https://github.com/jy-yuan/KIVI/blob/876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6/quant/matmul.py)：重点读取 `cuda_bmm_fA_qB_outer` 的布局、metadata 参数和 Hq/Hkv 关系。没有执行该函数或 CUDA 扩展。
- 对原文 §4.1 的模型架构旁支另查 [Mistral 官方发布说明](https://mistral.ai/news/announcing-mistral-7b/) 与 [Mistral 7B 原论文摘要](https://arxiv.org/abs/2310.06825)：Mistral 使用 GQA，不能照搬 KIVI 该节笼统的 MHA 分类；正文未复制该错误。

## 二、概念、公式及固定实现核对

1. 单 head 使用 `K,V: T×D`、`q: 1×D`。Key 的 group 固定通道、沿连续 token 收集；Value 的 group 固定 token、沿连续通道收集。矩阵和表格没有反轴。
2. `m=min(group)` 是原数值单位的浮点 offset，`Δ=(max-min)/(2^b-1)`；不是整数 affine zero point。`[-1,1]` 的四格点不含零。本文教学 RNE 规则和常量组特殊分支明确限定，未冒充原生 kernel 的 constant-group 行为。
3. `δs=q E_K^T/√D` 与 `δo=(phat-p)V+phat E_V` 的维度和符号正确。Key 统一 logit 平移可被 softmax 抵消，因此不能从 KV 重建 MSE 直接推出最终注意力/任务误差。
4. 修订后的 Key 第二列为 `[0,1.5,0.5,1]`，而非单调等间距；per-token 误差不再仅构成统一 logit 偏移。G=4 的两种分组在相同 4×4 数据上比较，最大概率位置由 token 2 变为 token 4。Value 也使用相同矩阵形状和 G=4，避免把组大小差异混入轴的对比；one-hot p 被明确标为人为隔离 Value 误差的构造。
5. R 控制时间残留策略，G 控制分组；Key 要求 R%G=0，不要求 R=G。持久 K residual 周期为 0…R-1，成熟 V residual 为 R。均匀周期持久 K 平均 `(R-1)/2`，不是逐步最低保护长度。
6. Algorithm 1 先调整状态，再计算当前注意力；固定 Flash decode 先用新追加的 K residual 计算 logits，再封存 K，统一 softmax；V 先用于当前输出，再 FIFO。本文明确选择固定 Flash 顺序，未将论文伪码和源码混成同一时点。prefill 当前注意力先用原浮点 K/V，再准备量化缓存。
7. 第 9 步 K 分界为 8/1，V 为 4/5。只做一次覆盖全部 token 的 softmax，再按 V 自己边界分概率。代码检查的是这一代数合同，不声称缓存近似误差消失。
8. 元数据按实际 dtype 计；FP16 offset+step、G=32、2-bit payload 的量化部分是 3 bits/element。容量公式用实际 Hkv 而非 Hq，含持久化 K/V 两种残留。T4096/D128/G32/R128 的单层单 KV head 示例：quantized 378 KiB + residual 32 KiB = 410 KiB；FP16 原始 2 MiB，约 5 倍而非 8 倍。
9. R4/G2 是状态教学尺寸，不直接满足原生 INT32 的 2-bit packing 容器（16 项/INT32）。教学 byte pack 没有被宣称为原生 artifact。
10. 普通 Attention 的 Hq/Hkv 限制、Flash 部分短缓存路径的 repeat 风险，仅用于说明支持范围未验证；没有将代码阅读风险写成已经执行的模型失败。容量公式也未被用作 GQA 全路径支持证明。
11. tuning-free 被解释为无需离线校准/优化量化参数，不等于无需选择 G/R 或评估质量。吞吐论文语境包含 OOM 前可扩大 batch，不承诺固定 batch 单请求加速。

## 三、审核后局部修正

- 标题由“为什么不能沿同一条轴量化”收窄为“为什么选择不同的量化轴”，与正文“经验设计，不是任意矩阵最优定理”一致。
- 首块增加有限、非空、矩形输入检查，避免 `zip(*matrix)` 对不规则矩阵静默截断。对步长、重建、注意力 logits/差值及加权结果检查有限范围；没有把教学代码扩大为通用高动态范围数值库。
- `unpack2` 明确只接受 bytes/bytearray，长度要求严格整数，拒绝 bool；避免 `[256]` 等非字节序列被掩码后静默变为合法 codes。保留显式逻辑长度，不把 padding 当 token。
- 状态函数对 steps 加非负整数约束；G/R 的原有整数、整除约束保留。
- 第三块对 dot、softmax 和 weighted 增加紧凑形状/有限域检查；稳定 softmax 处理常见大正/负 logits，但超出参考范围的中间差值明确拒绝。
- 在说明中写清：Python float 参考只处理下方有限小矩阵，不提供全动态范围高精度保证、不与 FP16/BF16 或官方实现逐位等价。
- §8 补充 tuning-free 的具体含义与 G/R、独立任务评测之间的边界。

## 四、实际执行与独立 oracle

专项：`scripts/reviews/oct15-kivi-examples.mjs`。脚本从上述 post 的三个唯一 marker 提取当前真实代码，每个块在独立 Python namespace 中执行；不另抄三个“相同示例”替代正文。

本轮实际命令：

```powershell
$env:BLOG_PYTHON = 'C:\Users\iStar\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
node scripts/reviews/oct15-kivi-examples.mjs
```

8 组通过：

1. 三个实际 Python 块分别执行，其正文 assertions 和打印输出全部发生。
2. 980 个有理格点组，用 Fraction 在四个候选重建值间比较距离/偶码来选码，不重写同一个除法+round 算法自证；另查 ties 及 nextafter 邻点、非整数原点、常量、subnormal 和步长 underflow/范围 overflow。
3. 所有 256 个 byte 的独立基 4 位提取、已知 `0xE4`/`0x24`、部分长度、空包及非法 code/payload/bool length。
4. 按显式坐标集合分组的 Fraction 矩阵 oracle；70 位 Decimal 的独立 softmax；验证非恒定 logit 误差、精确输出误差分解，并转置构造反例说明轴并非普遍优劣定理。
5. 用闭式 token 编号集合，而非另一套 append/flush 过程，核对 R=1/2/4/6/8/12/32/128、各整除 G 的多周期快照；当前计算与持久化边界、每个 token 不漏不重、平均长度、G/R/steps 非法输入。
6. 独立分数概率 `1/17` 与 `9/17`、`117/17` 输出及分段 softmax 错误 `27/2`；其他 logits 用 70 位 Decimal 校验，遍历各非空 Value 分界重建输出。
7. 不规则/空矩阵、非有限输入、有限输入但乘积或中间差值溢出等拒绝路径；常见 ±1000 等 logits 的稳定归一化。
8. 显式枚举完整量化 group、metadata 和残留元素字节，核对总公式与 410 KiB；不同 metadata bytes、短序列、高精度残留比例、Hkv/Hq 和原生容器对齐的条件。

专项起草过程曾有测试代码自身把 float 传入双参数 Fraction 构造器的问题，已修正为先转 Fraction 再除；其失败不来自正文，也未被计为“发现文章错误”。最终以上八组均以实际通过结果记录，不将内部 assertions 数量当成独立实验数量。

## 五、验证边界与移交

本轮只执行标准库 CPU 数值、状态及字节预算检查。未运行官方 Torch 模型、Triton/CUDA pack/BMM、真实 GQA/cache 接口、完整模型长期 decode、推测解码回滚、任务质量、GPU 显存或性能实验。未以教学 pack round-trip 或 token 编号测试证明真实低位 artifact/runtime 已验证。

正文无裸单美元数学、无数学中的裸 pipe；日历 date、author、series/order、文件名和 URL 保持约定。本文为当天日期，不带会导致 future-post 排除的未来小时。

正文已冻结供 root 最终构建、全站回归和浏览器检查；这些发布层结果由 root 独立补记，不能由本专项代替。完成后正式移交本篇 post、本专项脚本及本审核记录，审核代理停止写入这三个文件。

## 六、Root 本地验收

三个实际代码块与 8 组专项复跑通过；全部 28 个 JavaScript 审核套件、既有 Python、95 篇元数据和历史 coverage 通过，`git diff --check` 无空白错误。实际生产构建生成 110 个 HTML 页面和 135 个文件；88 篇文章的 1104 个原生行内与 978 个独立公式通过结构检查。

本地浏览器读到最终标题“为什么选择不同的量化轴”，117 个 MathJax 容器、0 个错误、无单双美元残留，1265 像素视口无整页横向溢出；计算/持久化两时点的时序截图正常。这里没有运行真实模型或 GPU，提交与部署证据另记。
