# Residual + RMSNorm 独立审核

日期：2026-10-20。对象：`2026-10-20-residual-rmsnorm-kernel-fusion.md` 全文、对应 research 和 `residual-rmsnorm-examples.mjs`。调研、写作与本次审核由不同代理承担。这里只记录内容与实际 CPU 检查，不代替主代理的构建、浏览器或发布证据。

## 独立来源核对

独立读取 FlashInfer GitHub 的 `refs/tags/v0.7.2`，确认 annotated tag `ca0f620c9eca61759dc488e343e9e1e6ba9e43c3` 指向 commit `d84335d8c54e0c8f9f7e22faa15c7e34a58abf94`。随后通过该固定 SHA 读取以下源码，而非依赖动态主分支缓存：

- `include/flashinfer/norm.cuh` L416–542：float 加法与平方和、低精度 residual 独立写回、未低精度舍入的 x 存入 FP32 shared memory，再用于输出；warp 与 cross-warp 归约、实际 D 分母和 shared 配置。
- `flashinfer/norm/__init__.py` L58–101、267–305：显式 CUDA 配置、CuTe 可用性/架构回退；双 mutation、None 返回、API 默认 eps=1e-6。底层 C++ helper 的默认 1e-5 不等同这个 Python API 的默认值，正文未混用。
- `flashinfer/norm/kernels/fused_add_rmsnorm.py` L359–411：CuTe 路径在 residual cast/store 后仍用 FP32 h 计算均方与 Y；源码明确 fastmath rsqrt，因此 CPU 模型不承诺逐 bit 设备复现。
- `tests/utils/test_norm.py` L85–94：官方参考同样对 FP32 相加结果归一化，新残差只是单独 cast。没有误引相邻的 Gemma 参考函数或把 weight_bias=1 当作本 API。

另打开并读了 RMSNorm 2019 原论文 §3–4.1 以及 PyTorch 2.8 RMSNorm 文档。2019 年没有被当作 FlashInfer 融合实现的发布时间。

## 逐段结论

1. 双输出与 Pre/Post-Norm：局部数据流保留未归一化新残差和归一化 Y，未改变模型结构。原地语义只是覆盖指定 input/residual，不保证相互 alias 安全。
2. 公式：均方不是方差，无减均值；gamma=1 时输出均方为 q/(q+eps)，非任意 epsilon 下严格单位 RMS。只对非零向量的正比例缩放讨论 eps=0 的理想不变性，避免负比例或固定 epsilon 的过度推广。eps 位于根号内，PyTorch 与 FlashInfer 默认参数差异有官方依据。
3. 浮点契约：Z32、R_new cast 和 Y cast 的位置与固定源码一致。FP16 中点反例的两个 R_new 相同而 Y 不同，证明仅检查残差不足。表格值是宿主教学模型值，不是 GPU 测量。
4. 归约与数据保存：补齐列的零贡献不改变实际 D 分母；实际 CUDA 路径保存 FP32 行数据在 shared memory，未冒称全部寄存器常驻。D4096、两字节元素得到 vec8、512 线程、16 warps、16448 字节/block，与 launch 源码相符；未将其等同 occupancy 或 HBM 流量。
5. 流量账本：两方案采用同一 FP32 Z 语义，split=5Ns+8N、fused=5Ns，差额为 FP32 临时张量的一写一读。M128/D4096/s2 对应 9437184、5242880、4194304 字节。另列的低精度 6Ns→5Ns 是不同数值契约，没有与前一语义拼成等价速度主张；逻辑 global、shared 容量/流量与实测 HBM 明确分开。
6. 性能与验证：资源占用、行数、stride/对齐/宽地址、两个输出、不同后端与原地状态累积均有相关约束，未承诺任意形状/任意 view。没有用逻辑字节比例宣称设备加速或局部加速必然改善请求。

## 修正与独立测试

CPU 示例原先只在宿主精度检查 epsilon 正数，`1e-100` 转 FP32 会成为零；全零输入随后会除零。现显式要求转换后的 eps32 仍为有限正数。有限检查也先做受控 float 转换，超大整数不再从 math.isfinite 泄漏未解释的转换行为；模式开关限定 bool。输入、残差或输出超出 binary16 范围时拒绝，不模拟硬件 Inf/NaN 传播，正文已限定此域。

保留作者 Decimal 数学参考与不同宽度测试。独立新增不使用 struct 的 binary16 码字解码器和 Fraction 最近偶数舍入 oracle，核对全部 31743 对相邻正有限数的中点、负镜像以及中点两侧，共 126972 次 midpoint/neighbor 比较，含零/次正规和次正规/正规边界。另用 90 位 Decimal 与独立 Fraction half 量化器复算表中两条路径。它们验证舍入节点和示例表，不宣称模拟 GPU FP32 树形归约或 fast rsqrt。

增补 epsilon 下溢/溢出、超大整数、输入 FP16 溢出、残差相加后溢出、归一化后增益导致 Y 溢出、非法模式，以及函数式示例不修改原列表的检查。函数式 CPU 示例与实际原地 GPU API 已分别描述，不拿前者的不修改输入性质当作后者合同。

实际执行 `node scripts/reviews/residual-rmsnorm-examples.mjs`，12 个 CPU 语义组通过；`node scripts/audit-posts.mjs` 的 100 篇结构/元数据/系列/标签检查和 `git diff --check` 通过。穷举输入数不是独立人工审核数量。

本次没有导入 Torch、运行 FlashInfer/CuTe/CUDA、采集性能或执行真实模型。未发现剩余阻止发表的概念、公式或教学代码合同问题；主代理仍需分别确认构建与呈现、匹配 CI 和线上正文。
