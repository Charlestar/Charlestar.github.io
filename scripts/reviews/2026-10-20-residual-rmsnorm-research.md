# Residual + RMSNorm 研究交接

日期：2026-10-20。research_oct20 查阅主要论文及固定源码并制定大纲，writer_oct20 在第一篇独立提交后写本篇；独立审核另存记录。

## 实读来源与数值契约

- 原论文 https://arxiv.org/html/1910.07467 ，2019-10-16，§3–4.1。RMSNorm 去除均值中心化，不等同 LayerNorm。理想缩放性质不能直接推广到固定 epsilon 的所有浮点实现。
- FlashInfer v0.7.2，固定 SHA `d84335d8c54e0c8f9f7e22faa15c7e34a58abf94`。
  - include/flashinfer/norm.cuh L416–542：FP32 加法，低精度残差写回，FP32 和留在 shared memory 供 norm 使用；warp/cross-warp 归约、实际 D 分母与动态 shared memory 配置。
  - flashinfer/norm/__init__.py L58–101、267–305：默认可选 CuTe，CUDA fallback；input/residual 双 mutation、返回 None、默认 eps=1e-6。
  - flashinfer/norm/kernels/fused_add_rmsnorm.py L381–411：FP32 h 在低精度残差写回后仍用于 norm，寄存器与行调度组织不同。
  - tests/utils/test_norm.py L85–94：官方参考 FP32 相加结果参与归一化，不能替换成先低精度 add_ 再 norm 的另一契约。
- https://docs.pytorch.org/docs/2.8/generated/torch.nn.RMSNorm.html ：eps=None 使用输入 dtype finfo.eps，与 FlashInfer 默认值不同，比较应显式统一。

## 复算与边界

输入 X=[1,2]、R=[2^-11,0]、gamma=1、eps=1e-6。两路线 R_new 同为 binary16 [1,2]，Y 第一项分别 0.6328125 与 0.63232421875。CPU 的 struct 半精度/单精度舍入顺序模型仅解释舍入节点，不模拟 GPU 树归约、FMA、fast rsqrt 或 IEEE 溢出传播。独审补齐域约束：eps 转 FP32 后必须仍为有限正数，超大整数或半精度输入/输出溢出均拒绝，不尝试模拟设备 Inf/NaN 传播。

同 FP32 Z 契约下，逻辑 global-memory split=5Ns+8N，fused=5Ns；M128/D4096/s2 分别9,437,184/5,242,880字节，差4MiB。不是实测HBM流量、带宽或speedup。CUDA shared行数据仍需约8N逻辑读写；每block shared容量D4096为16,448字节。

不放未验证的教学 Triton 内核，不把公共API分派当作必走CUDA路径，不承诺任意stride/alias/partial overlap。GPU执行、数值误差分布和性能均待真实环境验证。
