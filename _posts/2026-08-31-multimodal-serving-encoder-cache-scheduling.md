---
layout: post
title: "多模态 Serving：图片进入 LLM 之前发生了什么"
subtitle: "拆解 Media I/O、Processor、Encoder、Connector、Embedding Cache 与语言模型调度"
date: 2026-08-31 09:00:00 +0800
last_modified_at: 2026-09-03
author: iStar
catalog: true
series: model-serving-agents
series_order: 30
technology_year: 2023
mathjax: true
tags: [LLM推理, 推理调度, KV Cache]
---

给多模态模型发送一张图片时，API 看起来只比文本请求多了一个 URL 或 Base64 字段：

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        { "type": "image_url", "image_url": { "url": "..." } },
        { "type": "text", "text": "图中有什么？" }
      ]
    }
  ]
}
```

但图片并不会直接进入语言模型的 Tokenizer。服务端必须先获取媒体、解码文件、执行模型特定的缩放和切块，再调用视觉编码器，把二维像素变成一串向量，最后通过 Connector 对齐到语言模型的隐藏空间。

这条路径引入了文本 Serving 没有的资源：网络带宽、图片解码 CPU、Processor Cache、Encoder GPU 时间、视觉 Embedding，以及随视觉 Token 增长的 KV Cache。它也引入了新的正确性与安全边界：同一张图片是否真的命中了同一份缓存，Prompt 中的占位符能否和视觉向量严格对齐，远程 URL 会不会变成 SSRF 入口。

本文沿着一次图片请求的生命周期，说明多模态 Serving 如何把 Media Pipeline 与 LLM Prefill/Decode 连接起来，并进一步讨论动态分辨率、缓存键、连续批处理、显存规划和可观测性。

## 1. 图片不是一个 Token

文本请求经过 Tokenizer 后，得到一维 Token ID 序列：

$$
[t_1, t_2, \ldots, t_n]
$$

模型通过 Embedding Table 把每个 ID 映射为隐藏向量。图片没有天然对应的词表 ID，也不能只用一个特殊 Token 完整表达数百万像素。

视觉语言模型通常先将图片转换为多个视觉特征：

$$
I \xrightarrow{\text{processor}} P
  \xrightarrow{\text{vision encoder}} H_v
  \xrightarrow{\text{connector}} E_v
$$

其中：

- \(I\) 是原始图片；
- \(P\) 是缩放、归一化或切块后的 Pixel Tensor；
- \(H_v\) 是视觉编码器输出；
- \(E_v\) 是已经对齐到语言模型隐藏维度的视觉 Embedding。

语言模型最终接收的是文本 Embedding 与视觉 Embedding 组合后的序列，而不是 JPEG 字节本身。

LLaVA 展示了一种经典结构：用预训练视觉编码器提取特征，再通过可训练 Projection 将视觉特征接到语言模型。后来的模型加入动态分辨率、多尺度切块、视频帧、Cross-Attention 等机制，但“媒体输入必须先变成模型能消费的表示”这一点没有改变。

## 2. 不要假定所有模型都以同一种方式插入视觉信息

多模态模型的 Serving 接口可以相似，内部结构却可能完全不同。

### 2.1 视觉前缀或占位符替换

一类 Decoder-only 模型在文本 Prompt 中放入特殊占位符，再把占位符对应位置的普通 Token Embedding 替换为视觉 Embedding：

```text
[BOS] 请描述 [IMAGE_PLACEHOLDER] 的内容
                  ↓
[BOS] 请描述 [visual embedding 1 ... visual embedding m] 的内容
```

组合后的序列一起进入语言模型 Prefill，后续 Decode 继续使用生成的 KV Cache。

### 2.2 Encoder-Decoder 或 Cross-Attention

另一类模型让视觉编码器输出作为独立 Memory，语言模型在特定层通过 Cross-Attention 读取。此时视觉特征不一定等价于“占用同样数量的文本上下文 Token”，缓存形态和并行策略也会不同。

### 2.3 多阶段视觉处理

高分辨率模型可能同时使用缩略图和局部 Tiles；视频模型还要处理帧采样与时序位置。某些模型会对视觉 Token 再做 Pooling 或 Resampling，以固定或压缩长度。

因此推理框架需要模型专用的 Processing Info 与输入映射，不能仅凭 API 里存在 `image_url` 就假设执行路径相同。调度器估算 Token 和显存前，也必须询问具体模型，而不是套用统一常数。

## 3. 一次请求的完整数据路径

从 URL 到第一个输出 Token，一般会经过如下阶段：

```text
请求解析与限额检查
        ↓
Media Fetch / Base64 Decode
        ↓
图片或视频文件解码
        ↓
模型 Processor：resize / crop / normalize / frame sample
        ↓
Vision / Audio Encoder
        ↓
Projector / Resampler / Connector
        ↓
视觉 Embedding 与 Prompt 占位符对齐
        ↓
LLM Prefill
        ↓
LLM Decode
```

这些阶段可能分布在不同设备：

- Fetch 使用网络线程；
- JPEG/PNG 解码和部分预处理使用 CPU；
- Processor 可能运行在独立进程池；
- Vision Encoder 与 LLM 可能共享 GPU，也可能在不同 GPU 池；
- 缓存可能跨 API Worker、Engine Process 和 GPU Worker。

端到端延迟不能只记为“模型推理耗时”。只有把每一段独立计时，才能知道一次慢请求是在等远程图片、解码超大 PNG、排队等待 Encoder，还是卡在长视觉序列的 LLM Prefill。

## 4. Admission Control 必须发生在昂贵处理之前

文本请求可以用 Token 数快速估算成本。媒体请求在下载与解析前，往往连真实尺寸和帧数都不知道。

服务端至少需要两层限制。

### 4.1 获取前可判断的限制

- 每个请求允许的媒体数量；
- URL Scheme 与允许域名；
- 请求体和 Base64 字节数；
- 下载超时、最大重定向次数；
- 声明的 MIME Type；
- 租户并发与带宽配额。

### 4.2 解码后才能确认的限制

- 图片宽高与总像素；
- 动图或视频帧数、时长与采样后帧数；
- Processor 生成的 Tiles 与视觉 Token 数；
- Encoder 输入 Tensor 的实际形状；
- 合并后的模型上下文长度。

第二层检查必须在进入 GPU 调度队列之前完成。否则一个超高分辨率请求可能占住 Queue Slot，直到 Encoder 分配显存时才失败，既浪费前置计算，也干扰其他租户。

## 5. Media I/O 是一个真实的安全边界

允许服务端代替用户获取任意 URL，会让推理 API 同时承担网络代理职责。主要风险包括：

- 访问云环境 Metadata Endpoint 或内网管理地址；
- DNS Rebinding 绕过最初的主机检查；
- 重定向从公网地址跳到私网地址；
- 超慢响应长期占用连接；
- 压缩率极高的图片在解码后膨胀为巨大像素缓冲区；
- 伪造 MIME Type，把非图片内容交给脆弱解码器；
- SVG 等主动内容间接触发外部资源访问。

安全实现应在每次解析和重定向后校验目标地址，默认拒绝 Loopback、Link-local、私网与保留网段；对下载字节数、时间、解码像素和帧数设置硬限制；并使用及时更新、受隔离的媒体解码库。

如果业务不需要任意公网抓取，更稳妥的接口是让用户先把媒体上传到受控对象存储，再向推理服务传递短期签名引用或内容摘要。这样可以把抓取策略、病毒扫描和生命周期治理放在独立边界。

## 6. Processor 不只是 `resize(224, 224)`

模型 Processor 决定像素如何变成 Encoder 输入。常见操作包括：

- EXIF Orientation 校正；
- RGB 通道转换与 Alpha 处理；
- Resize、Center Crop 或保持比例缩放；
- 按模型训练参数归一化；
- 高分辨率图片切成多个 Tiles；
- 视频抽帧、限帧与时间采样；
- 生成 Patch Grid、Aspect Ratio ID 或 Position Metadata；
- 在文本模板中扩展相应数量的视觉占位符。

Processor 版本是模型正确性的一部分。即使 Encoder 权重完全相同，Resize 插值、均值方差、最大像素或 Patch 规则变化，也会产生不同视觉 Embedding。

因此模型部署不能只记录权重 Revision，还应固定：

```text
processor_class
processor_revision
image_config
video_sampling_config
chat_template_revision
tokenizer_revision
```

把 Processor 当成无状态的通用图片工具，会让缓存和灰度发布产生难以解释的差异。

## 7. 动态分辨率怎样变成视觉 Token

固定分辨率模型通常把图片缩放到固定尺寸，再按 Patch 切分。若 Processor 将 \(H \times W\) 的处理后图片分别补齐到最近的 \(p\) 的整数倍，Patch 大小为 \(p \times p\)，忽略额外特殊 Token 时，视觉 Patch 数是：

$$
N_v = \left\lceil \frac{H}{p} \right\rceil
          \left\lceil \frac{W}{p} \right\rceil
$$

这里的向上取整来自补齐操作，并不是所有视觉编码器的通式：若 Patch Embedding 使用不带 Padding 的步长卷积，数量应按实际处理后尺寸取下整；若 Processor 先 Resize/Crop 到固定可整除尺寸，则直接使用处理后的网格大小。Serving 侧应读取模型 Processor 给出的真实 Grid，而不是仅凭原图宽高套公式。

动态分辨率模型不再让所有图片都产生固定 \(N_v\)。高分辨率文档、长图或宽屏截图会保留更多细节，也会生成更多视觉 Token。Qwen2-VL 将这种能力称为 Naive Dynamic Resolution，并进一步用多模态位置编码表达图片与视频的空间、时间信息。

动态长度提升了信息保真度，却把成本控制问题交给 Serving：

- Encoder 计算随像素或 Patch 数增加；
- LLM Prefill 序列更长；
- Decoder-only 架构的 KV Cache 占用增加；
- 同一批请求的形状差异扩大；
- 最大上下文限制需要同时容纳视觉与文本 Token。

因此 API 的 `max_pixels`、`min_pixels`、帧数与 FPS 并非单纯画质参数，它们也是延迟和显存预算参数。

## 8. 视觉 Token 的成本不等于输出 Token

对 Decoder-only 模型，视觉 Embedding 通常只在 Prefill 阶段输入一次，却会影响后续每一层的 KV 状态。

假设文本 Prompt 有 \(N_t\) 个 Token，视觉部分产生 \(N_v\) 个 Embedding，总 Prefill 长度近似为：

$$
N_{\text{prefill}} = N_t + N_v
$$

标准 Attention 的直接计算随序列增长很快。即使使用高效 Attention Kernel，长视觉前缀仍会增加读写量、Prefill 时间与 KV Block 数量。

KV Cache 的粗略元素量可写为：

$$
M_{KV} \propto
2 \times L \times N_{\text{prefill}} \times H_{KV} \times D
$$

其中 \(L\) 是层数，\(H_{KV}\) 是 KV Head 数，\(D\) 是 Head Dimension，系数 2 对应 Key 与 Value。实际字节数还取决于数据类型、Block 对齐和并行切分。

这解释了为什么一张图片虽然在 API 层只是一个对象，却可能消耗数百或数千个“上下文位置”。调度器若只按文本 Token 收费和限流，会系统性低估多模态请求。

## 9. 占位符与 Embedding 必须严格对齐

Processor 经常在 Prompt 中插入一个或多个特殊 Token，告诉模型视觉 Embedding 放在哪里。模型执行前必须验证：

- 媒体数量与占位符组数一致；
- 每个媒体产生的 Embedding 数与预留位置一致；
- 多张图片的顺序与用户消息顺序一致；
- Batch Padding 没有改变有效位置；
- Position ID、Attention Mask 与 Modality Mask 同步更新；
- Chat Template 没有重复或吞掉特殊 Token。

一个常见错误是 Processor 根据真实图片生成了 576 个特征，而 Prompt 仍按固定模板预留 256 个位置。简单实现可能在 Tensor 拼接时报 Shape Error；更糟糕的实现会截断或错位，让文本 Token 与错误的视觉区域对应。

因此框架应把“媒体到占位区间”的映射作为结构化元数据保存：

```json
{
  "media_index": 0,
  "modality": "image",
  "prompt_token_start": 42,
  "prompt_token_length": 576,
  "embedding_shape": [576, 4096]
}
```

在 LLM Prefill 前验证这份映射，比依赖若干隐式长度相等更容易定位问题。

## 10. Encoder 与 LLM 是两种不同的计算形态

视觉 Encoder 常以较大的矩阵乘和二维 Patch Batch 为主，LLM Decode 则是每条序列每轮增加一个 Token、显存带宽敏感。将两者塞进同一个调度队列，会遇到资源竞争：

- 大图片 Encoder 占用 GPU Compute，阻塞正在 Decode 的低延迟请求；
- LLM KV Cache 占满显存后，Encoder 临时 Activation 无处可放；
- 不同图片尺寸造成 Encoder Batch Padding 浪费；
- Encoder 完成顺序与 LLM Scheduler 可接纳顺序不一致。

一种设计是在同一 GPU 上分阶段执行，但为 Encoder 预留受控显存和时间片；另一种是部署独立 Encoder Worker Pool，把视觉 Embedding 传给 LLM Worker。

拆分部署能隔离负载、独立扩容，却带来新的开销：Embedding 体积可能很大，跨进程、跨 PCIe 或跨网络传输会抵消收益；而且模型与 Processor Revision 必须在两端严格一致。

选择前应先测量：Encoder 占总延迟多少、Embedding 多大、重复媒体比例多高，以及 LLM 与 Encoder 的峰值是否真的同时发生。

## 11. Encoder Batch 也需要自己的调度策略

文本连续批处理通常按 Token Budget 组织 Batch。视觉 Encoder 的输入是不同尺寸的 Pixel Tensor 或不同数量的 Tiles，Batching 维度不完全相同。

如果把一张小图和一张超宽文档图直接 Pad 到相同尺寸，大部分计算可能落在 Padding 上。可采用：

- 按分辨率 Bucket；
- 按 Patch 或 Tile 数量设 Batch Budget；
- 让动态形状 Kernel 处理变长序列；
- 分开图片、视频和音频队列；
- 为等待过久的小请求设置 Aging，防止一直被大批次挤压。

Encoder Scheduler 的输出还要与 LLM Scheduler 协调。完成视觉编码并不代表可以立刻 Prefill：LLM 侧可能没有 KV Block 或 Token Budget。若 Embedding 暂存在 GPU 上等待太久，会占用宝贵显存；若立刻移到 CPU，又产生传输成本。

因此系统需要明确的中间状态：

```text
MEDIA_QUEUED
→ PROCESSING
→ ENCODER_QUEUED
→ ENCODING
→ EMBEDDING_READY
→ LLM_PREFILL_QUEUED
→ DECODING
```

每个状态都应支持超时、取消和资源释放。

## 12. 多模态缓存至少有四层

“缓存图片”不是一个足够精确的设计。不同阶段都可以复用，但代价和正确性键不同。

| 层级 | 缓存内容 | 主要节省 | 主要代价 |
| --- | --- | --- | --- |
| Media Cache | 原始或已验证的媒体字节 | 网络下载 | 存储、隐私与过期治理 |
| Processor Cache | Pixel Tensor、Tiles、形状元数据 | 解码和 CPU 预处理 | CPU/共享内存占用 |
| Encoder Cache | 视觉 Embedding | Encoder GPU 计算 | 较大内存或传输成本 |
| LLM Prefix/KV Cache | 组合序列的 KV Block | LLM Prefill | GPU/CPU KV 容量 |

它们不能用同一个模糊的“图片 URL”作为键。URL 指向的内容可能变化；同一字节经过不同 Processor、Encoder 或 Connector 也会得到不同结果。

缓存层级越靠后，单次命中节省的计算通常越多，但键中必须纳入的模型语义也越多。

## 13. 正确的缓存键应从内容身份开始

可靠缓存最好以经过验证的媒体内容摘要为根：

```text
media_digest = hash(canonical_media_bytes)
```

然后按不同缓存层加入处理身份。

Processor Cache Key 可以包含：

```text
media_digest
+ processor_class_and_revision
+ resize_crop_normalize_options
+ frame_sampling_options
+ output_tensor_dtype_layout
```

Encoder Cache Key 还应加入：

```text
vision_encoder_weights_revision
+ connector_or_projector_revision
+ quantization_and_precision_mode
+ model_specific_processing_options
```

如果 Adapter 会修改视觉 Encoder 或 Connector，Adapter 身份也必须加入；若 LoRA 只作用于语言模型层，则要根据实际注入位置判断，而不是无条件加入或省略。

LLM Prefix Cache 则依赖最终输入 Embedding、位置编码、模型权重和 Adapter 身份。Encoder Cache 命中并不意味着 KV Cache 一定命中，因为用户问题、Chat Template 或图片在 Prompt 中的位置可能不同。

## 14. URL、Base64 与摘要各有什么问题

### 14.1 直接用 URL 做键

计算便宜，但同一 URL 的内容可以变化，签名 URL 也会频繁变化；查询参数还可能造成同内容多键。只有在对象存储提供不可变版本 ID 或可信 ETag 时，URL 元数据才适合参与身份判断。

### 14.2 对下载字节做 Hash

内容寻址更稳妥，但必须先完整读取媒体，也会消耗 CPU。对于大视频，可以在流式下载时增量计算摘要，同时执行字节数上限。

### 14.3 对解码后像素做 Hash

可以让元数据不同但视觉内容相同的文件归一化，却必须承担完整解码成本；颜色配置、方向和 Alpha 合成规则也会影响“相同像素”的定义。

工程上常用原始内容摘要作为基础，再把 Processor 配置写入下一级键。重要的是明确所承诺的等价关系，并保证键与实际计算保持一致。

## 15. 缓存命中也涉及租户隔离与隐私

跨用户复用同一张公开图片的 Encoder Embedding 很诱人，但缓存命中本身可能成为侧信道：攻击者通过延迟差异推测某个媒体是否被其他用户处理过。缓存内容也可能包含敏感图片的高维表示。

需要根据产品边界决定：

- 缓存是请求级、会话级、租户级还是全局；
- 是否允许跨租户去重；
- 原始媒体和 Embedding 的保留时间；
- 删除请求如何传播到各级缓存；
- 日志能否记录 URL、摘要或缩略图；
- 缓存数据是否加密，哪些进程可读取。

如果业务处理私有文档，默认租户隔离通常比极致命中率更重要。公开模型演示与内部知识库也不应无意间共享同一套缓存策略。

## 16. Processor Cache 为什么会消耗大量主机内存

压缩图片文件可能只有几百 KB，解码成 Float Tensor 后体积会显著增加。以 RGB、FP32 为例，单张 \(H \times W\) 图片的原始 Tensor 大约占：

$$
M \approx H \times W \times 3 \times 4\ \text{bytes}
$$

如果 Processor 还生成多块 Tiles、多个尺度或一组视频帧，占用会继续放大。多进程 Engine 若各自维护相同容量的 Processor Cache，总内存会按进程数乘开。

一些框架允许选择本地 LRU、进程间共享内存或关闭多模态 Processor Cache。共享缓存减少重复内容，却增加序列化、IPC、锁竞争和生命周期复杂度；本地缓存简单，但每个 API/Engine Process 会重复保存。

容量规划要看解码后对象的真实字节数，而不是上传文件大小。还应分别记录逻辑条目大小、共享内存分配大小和进程 RSS，避免“配置了 4 GB 缓存，实际多进程用了几十 GB”的误判。

## 17. GPU Encoder Cache 与 KV Cache 会争夺同一块显存

把视觉 Embedding 留在 GPU 上，可以让重复图片跳过 Encoder 和 Host-to-Device 复制。但 GPU 内存同时还要容纳：

```text
模型权重
+ CUDA Graph / Workspace
+ Encoder 峰值 Activation
+ 视觉 Embedding Cache
+ LLM KV Cache
+ 临时 Sampling 与通信 Buffer
```

多给 Encoder Cache 一 GB，就可能少一 GB KV Cache，降低并发 Decode 能力。反之，如果 KV Cache 把显存吃到极限，偶发大图的 Encoder 峰值会触发 OOM。

合理的启动 Profiling 应同时覆盖最坏媒体形状与代表性 LLM Batch，而不是只用纯文本请求估算可用 KV Block。对于共享 GPU 的设计，还可以设置独立水位：当 LLM KV 压力升高时，先逐出可重算的视觉 Embedding，而不是让运行中序列失败。

逐出策略也不应只看最近访问。一个需要几百毫秒重算的大视频 Embedding，与几毫秒可重算的小图不应具有相同价值。可将 `recompute_cost / bytes` 纳入保留优先级。

## 18. 多轮对话怎样复用同一张图

用户通常先上传图片，再连续追问：

```text
Turn 1: 这张图展示了什么？
Turn 2: 左上角的数字是多少？
Turn 3: 把表格整理成 JSON。
```

若每轮都重新下载、处理和编码图片，浪费非常明显。可以按层复用：

1. 会话保存不可变媒体引用与摘要；
2. Processor Cache 复用 Pixel Tensor；
3. Encoder Cache 复用视觉 Embedding；
4. 若完整 Prompt 前缀一致，KV Cache 还可以复用 LLM Prefill。

不过对话模板经常把历史消息重新串接，图片占位符的位置可能变化；模型也可能要求每一轮都保留完整视觉前缀。Encoder Embedding 通常与文本位置无关，但加入位置编码或模型专用融合之后未必如此。

因此系统应缓存“模型定义允许复用的最早阶段”，再在当前 Prompt 中重新完成位置相关组合，而不是假设上轮的最终 KV 可以任意搬到新上下文。

## 19. Scheduler 需要同时理解三种预算

一个多模态请求的资源至少有三部分：

```text
媒体处理预算：bytes / pixels / frames / CPU time
Encoder 预算：patches / tiles / encoder FLOPs / activation bytes
LLM 预算：prefill tokens / decode tokens / KV blocks
```

只按 `prompt_tokens + max_tokens` 调度，会漏掉前两层；只按图片数量调度，又无法区分小图标和高分辨率文档。

一种可行流程是：Processor 先输出确定的形状元数据和视觉 Token 数，Admission Controller 据此估算 Encoder 与 KV 成本，再将请求放入对应队列。此时调度器可以做更合理的决策：

- 限制同一时刻的大图 Encoder 数量；
- 在 Prefill Chunking 中把长视觉前缀分段；
- 为短文本、已命中 Encoder Cache 的请求提供低延迟路径；
- 防止持续到来的小请求让大图永远饥饿；
- 对同租户同时占用 Media、Encoder 与 Decode 资源做联合配额。

调度策略需要明确优化目标：交互式问图重视 TTFT，离线视频理解重视吞吐，二者不应共享完全相同的 Queue Policy。

## 20. Chunked Prefill 不能切坏多模态边界

长 Prompt 可以通过 Chunked Prefill 分多轮计算，避免单个 Prefill 独占整个 Token Budget。多模态序列还多了一层要求：视觉 Embedding 区间可能对应一个不可任意拆分的模型输入单元。

例如某些模型的视觉模块需要一次处理完整 Tile 集合，或者 Connector 输出后才知道准确 Token 数。Scheduler 在切分前必须获得模型声明的 MultiModal Boundaries。

切分方案可能是：

```text
chunk 1: system + user text prefix
chunk 2: complete image embedding span
chunk 3: remaining user text
```

也可能允许在视觉 Embedding 序列内部按 Token 切块，但前提是模型实现和 Attention Metadata 支持。不能仅根据全局 Token Index 机械切分。

取消请求时，尚未进入 LLM 的 Encoder 输出、已分配但未写满的 KV Block，以及 Processor 中间对象都要释放。跨阶段状态越多，取消路径越需要显式资源所有权。

## 21. Tensor Parallel 下视觉编码器怎样放置

语言模型可能采用 Tensor Parallel 切分权重。视觉 Encoder 的规模和通信模式不同，不一定适合同样切分。

常见选择包括：

- 每个 TP Rank 都复制 Encoder，但只处理自己的请求或相同输入；
- 只在一个 Rank 执行 Encoder，再把视觉 Embedding 广播到其他 Rank；
- 对 Encoder 本身做 Tensor Parallel；
- 使用独立 Encoder Data Parallel Worker，再分发结果给 LLM TP Group。

复制 Encoder 简单，却增加权重显存；单 Rank 编码节省权重，但它可能成为吞吐瓶颈，并产生 Embedding Broadcast；独立 Worker 隔离最好，却增加跨设备数据传输。

Connector 也不能遗漏。若 Projector 将视觉维度映射到 LLM Hidden Size，它的输出可能需要按语言模型并行布局 Shard。广播完整 Tensor 还是直接生成 Shard，会影响通信量和实现复杂度。

最终方案必须以具体模型支持为准。框架的“支持多模态”列表通常还会区分是否支持 PP、TP、视频、多图片或特定 Attention Backend；部署前应验证目标模型组合，而不是只看到模型能单卡运行。

## 22. 量化不会自动覆盖整条多模态链路

加载一个量化语言模型，并不代表视觉 Encoder、Connector 和视觉 Embedding 都使用同一种精度。

可能出现：

- LLM 权重为 INT4，Vision Encoder 仍为 BF16；
- Projector 使用 FP16，输出再转换为 LLM Activation Dtype；
- Processor 在 CPU 产生 FP32 Tensor，上传 GPU 后转换；
- KV Cache 使用 FP8，但视觉 Encoder Activation 不支持 FP8；
- 特定量化 Backend 只实现文本模型算子。

这些转换会影响显存、带宽与数值。性能报告应分别列出 LLM、Encoder、Connector 和 KV 的 Dtype，不能用一个“模型是 4-bit”概括整条路径。

正确性验证还要关注 OCR、小字体、图表数值等对量化敏感的任务。文本困惑度变化不大，不代表视觉细节能力保持不变。

## 23. 流式输出之前，服务端可能已经做了大量不可流式工作

文本生成可以在 Prefill 完成后逐 Token 返回。图片请求在第一个 Token 之前，通常必须完成下载、解码、Processor、Encoder、Connector 和 LLM Prefill。

因此多模态 TTFT 可以分解为：

$$
T_{TTFT} = T_{fetch} + T_{decode} + T_{process}
+ T_{encoder\_queue} + T_{encoder}
+ T_{llm\_queue} + T_{prefill} + T_{sample}
$$

如果只暴露一个 TTFT，用户看到的抖动很难定位。更好的 Trace 会记录每个阶段开始、结束和缓存命中事件。

某些场景可以提前返回“媒体已接收”“处理完成”等进度事件，但不能在 Encoder 完成前生成依赖图片内容的答案。所谓流式多模态 Serving，主要改善的是 Decode 阶段输出体验，不会消除前置视觉计算。

## 24. 可观测性要记录形状，而不是记录敏感内容

多模态问题通常与输入形状和缓存身份相关，但直接把图片、URL 或 Base64 写入日志会制造严重隐私风险。

建议记录结构化、去内容化的字段：

```text
request_id
tenant_scope
modality_counts
download_bytes
decoded_width / decoded_height / frame_count
processor_revision
tile_count / visual_token_count
processor_cache_hit
encoder_cache_hit
encoder_queue_ms / encoder_ms
llm_prefill_tokens / llm_prefill_ms
kv_blocks_allocated
finish_reason
```

内容摘要也要谨慎：如果摘要可被低成本字典攻击，它仍可能暴露用户是否上传过某张公开图片。可以使用租户域内的 Keyed Hash，或者只在受控 Trace 中保存截断标识。

日志还应保留模型、Processor、Tokenizer 与 Adapter Revision，确保一次异常能够重放到相同处理链路。

## 25. 性能评测需要构造真实的媒体分布

只用一张固定 224×224 图片循环压测，会让 Processor Cache 和 Encoder Cache 接近 100% 命中，也无法暴露动态分辨率、网络下载与形状异构成本。

建议至少准备以下维度：

| 维度 | 代表场景 |
| --- | --- |
| 小图 / 中图 / 高分辨率长图 | 图标、照片、文档截图 |
| 单图 / 多图 | 问图、图片对比 |
| 短视频 / 长视频不同采样率 | 动作理解、事件检索 |
| 缓存冷 / 热 / 部分复用 | 首次请求、多轮追问 |
| 本地上传 / 对象存储 / 远程 URL | 不同 Media I/O 路径 |
| 短输出 / 长输出 | OCR 提取、详细描述 |
| 同形状 / 混合形状 Batch | 理想与生产负载 |

核心指标应分阶段统计：

- Media Fetch 与 Decode p50/p95/p99；
- Processor CPU 时间与队列长度；
- Encoder Tokens 或 Patches/s；
- Encoder Batch Padding 比例；
- TTFT 与 Inter-token Latency；
- Processor、Encoder、KV 三类缓存命中率；
- GPU 利用率、峰值显存和 OOM/Admission Reject；
- 每个请求的视觉 Token、文本 Prefill Token 与输出 Token；
- 取消后资源回收延迟。

还要分别报告冷缓存和热缓存结果，并限制压测端缓存，避免客户端本身改变输入分布。

## 26. 正确性验证要跨越每一道转换

多模态 Serving 的错误可能发生在模型之前。可以把验证分为四层。

### 26.1 Media 与 Processor

- EXIF 旋转后的方向正确；
- 不同颜色模式、透明通道与动图处理符合契约；
- 超限像素、帧数和下载大小在 Admission 阶段拒绝；
- 同输入与同 Revision 产生确定的 Tensor 形状和摘要。

### 26.2 Encoder 与 Connector

- 缓存命中和重新计算的 Embedding 在允许误差内一致；
- Precision 或量化切换有独立基准；
- TP/DP 模式下输出与单卡参考一致；
- 多图片的 Embedding 顺序不交换。

### 26.3 LLM 组合

- 占位符区间与视觉 Token 数完全一致；
- Position ID、Attention Mask 和 Padding 正确；
- Chunked Prefill 与非 Chunked 结果一致；
- Prefix Cache 命中不会串用不同媒体或 Adapter。

### 26.4 Serving 生命周期

- 下载、Processor、Encoder 和 LLM 每个阶段都可取消；
- 超时与错误不会留下缓存半成品；
- Worker 重启后不会读取版本不兼容的持久缓存；
- 多租户缓存和日志策略满足隔离要求。

最终答案质量评测仍不可缺少，但它不能替代这些系统不变量。模型偶然答对，并不能证明图片方向、缓存身份和 Token 对齐都是正确的。

## 27. 一条可落地的部署路径

如果要把文本 Serving 扩展为多模态，可以按下面顺序建立能力。

### 第一步：先固定模型处理契约

锁定 Model、Tokenizer、Processor、Chat Template 与 Connector Revision，明确支持的媒体类型、数量、像素、帧数和上下文上限。

### 第二步：在 GPU 队列前完成安全处理

把 URL 策略、下载限制、解码隔离、像素/帧检查和视觉 Token 估算放在 Admission 阶段。失败请求不应占用 Encoder 或 LLM Slot。

### 第三步：先实现正确的无缓存路径

验证媒体顺序、占位符长度、Embedding Shape、Position 与 Attention Metadata。没有可靠基线时加入缓存，只会让错误变得随机。

### 第四步：按收益逐层加缓存

先测 CPU Processor 与 Encoder 各自成本，再选择缓存层。为每层定义完整版本键、租户范围、容量、逐出和删除策略。

### 第五步：让 Scheduler 看见真实成本

将像素、Tiles、视觉 Token、Encoder 形状、LLM Prefill Token 与 KV Block 都变成 Admission 和调度输入，而不是只传一个图片数量。

### 第六步：建立分阶段 Trace 与基准

覆盖冷/热缓存、混合分辨率、多图、视频、取消、超时、TP 和量化组合。上线后观察队列、命中率、显存水位和长尾 TTFT。

## 28. 从一张图片回看完整生命周期

把前面的组件串起来，一轮请求可以表示为：

```text
1. API 验证媒体数量、Scheme、租户配额
2. 安全下载并增量计算内容摘要
3. 解码后检查像素、帧数和 MIME
4. 查询 Processor Cache
5. Processor 生成 Pixel Tensor、Tiles 与视觉 Token 元数据
6. Admission Controller 估算 Encoder 和 LLM 成本
7. 查询 Encoder Cache；未命中则排队编码
8. Connector 生成与 LLM Hidden Size 对齐的 Embedding
9. 校验媒体顺序、占位符区间、Position 和 Mask
10. LLM Scheduler 分配 KV Block 并执行 Prefill
11. Decode Scheduler 连续生成文本 Token
12. 请求结束、取消或超时后释放中间资源
```

这条链路解释了多模态 Serving 为什么不能被简化成“在文本接口里支持图片”。它实际上把媒体系统、CPU 数据处理、异构 GPU 计算、缓存治理和 LLM 调度连接成了一条新的数据平面。

## 29. 结语

图片进入 LLM 之前，已经完成了一次从不可信媒体到模型专用向量的编译过程。Media I/O 决定能否安全取得内容，Processor 决定像素如何解释，Encoder 与 Connector 决定视觉信息如何映射到语言空间，占位符和 Position Metadata 决定它在 Prompt 中位于哪里。

真正稳定的多模态服务需要同时回答四组问题：输入怎样限额与隔离，视觉长度怎样计入调度，哪些中间结果能够在什么范围复用，以及 Encoder 峰值、Embedding Cache 与 KV Cache 怎样共享有限显存。

当这些状态都显式化之后，很多“模型偶尔看错图”的问题才能被分解：是图片方向错了、Processor Revision 变了、Encoder Cache 串了、占位符错位了，还是模型本身的视觉理解失败。也只有完成这种分解，多模态能力才从一次可运行的 Demo，变成可以观测、扩容、复现和治理的 Serving 系统。

## 参考资料

- Liu et al., [Visual Instruction Tuning (LLaVA)](https://arxiv.org/abs/2304.08485)
- Wang et al., [Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution](https://arxiv.org/abs/2409.12191)
- [vLLM Multimodal Processing Cache API](https://docs.vllm.ai/en/latest/api/vllm/multimodal/cache/)
- [vLLM Optimization and Tuning](https://docs.vllm.ai/en/stable/configuration/optimization/)
- [vLLM Supported Models](https://docs.vllm.ai/en/latest/models/supported_models/)
