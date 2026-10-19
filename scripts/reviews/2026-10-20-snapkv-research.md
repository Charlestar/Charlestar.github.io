# SnapKV 研究交接

日期：2026-10-20。research_oct20 制定大纲并实读论文与官方源码，writer_oct20 在前两篇分别提交后写本篇；独立审核另存记录。

## 实读来源

- https://arxiv.org/html/2404.14469v2 ，§3–5：观察窗口、先 full causal softmax 再 prefix 投票、pooling、prompt cache 预算和实验范围。v1为2024-04-22，v2为2024-06-17。论文§5.2使用maxpool5，不能与代码默认avgpool混称。
- 官方固定 SHA `e216ddc84c5bd210378cbdbbba12ba02102aa640`：
  - snapkv/monkeypatch/snapkv_utils.py 全文：完整K打分、窗口内因果mask、FP32 softmax转回query dtype后求和、池化topk、同索引gather、观察窗口全留。传入attention_mask被重新赋值，不能宣称支持任意padding/ragged。
  - snapkv/monkeypatch/llama_hijack_4_37.py：先RoPE，再repeat_kv到Hq，再压缩写past；当前prefill仍用未压缩局部KV；decode追加不持续重筛；逻辑进度单独维护。
  - README.md：tested transformers4.37.0，flash-attn2.4.0；不推断当前栈兼容。
- https://docs.pytorch.org/docs/2.8/generated/torch.nn.functional.avg_pool1d.html ：默认count_include_pad=True。
- https://docs.pytorch.org/docs/2.8/generated/torch.topk.html ：ties不保证稳定；教学CPU小索引tie-break明确不冒充GPU复现。

## 复算例与实现边界

L8/W2/C5/D2，由Q6=[sqrt2,0]、Q7=[0,sqrt2]和Kj=[ln(a_j),ln(b_j)]构造：a=[1,2,10,8,1,1,2,1]、b=[1,1,8,12,1,1,1,1]；query6的最后位置因果屏蔽。avgpool3后保留[2,3,4]+[6,7]。独立反例A/B fullsoftmax票[.315,.685]，错误prefix-only票[1.15,.85]，选择翻转。

固定q/K/V保留质量m分解是直接代数说明，不是论文额外实验：O=mOkeep+(1-m)Odrop，有界V下局部误差<=2Vmax(1-m)。不推出未来query的m或完整模型质量。

GQA持久缓存要使用官方展开后的Hq，而非原生Hkv。B1/layers32/Hq32/Hkv8/D128/s2/L16384/C1024/g512时原生full2112MiB、展开Snap768MiB、另一种保持8heads的假设设计192MiB。生成cache在压缩分支为C+g，短提示词旁路为L+g，统一min(L,C)+g，与逻辑位置L+g分开；g指已处理并追加的token。独审同时澄清接入次序：压缩写cache发生在当前完整prefill attention调用前，不能把“当前仍使用完整KV”写成“先完成prefill才压缩”。

CPU代码是Python浮点算法示例，不含官方query dtype回舍入，也不是GPU topk、数值误差、模型任务或性能复现。观察打分额外O(BHqWLD)与O(BHqWL)空间，prefill全KV和临时峰值不可当作持久缓存同比减少。
