# Charlestar

子辰的个人技术博客，主要记录 AI Infra、大模型推理、GPU 性能优化与云原生工程实践。

访问：[charlestar.github.io](https://charlestar.github.io)

## 技术栈

- Jekyll 4：静态站点与内容管理
- GitHub Actions / Pages：自动构建与发布
- Liquid / Markdown：页面模板与文章内容
- 原生现代 CSS：Grid、Custom Properties、响应式与深色模式
- 原生 JavaScript：搜索、目录、代码复制、阅读进度与 PWA
- Giscus：基于 GitHub Discussions 的评论

项目不依赖 Bootstrap、jQuery、前端打包器或第三方主题运行时。

## 本地开发

需要 Ruby 3.1+ 与 Bundler；运行下方完整检查还需要 Node.js 20+ 和 Python 3.10+。

```bash
bundle install
bundle exec jekyll serve --livereload
```

打开 <http://127.0.0.1:4000>。

构建后可运行本地完整性检查：

```bash
bundle exec jekyll build
node scripts/audit-posts.mjs
node scripts/reviews/check-coverage.mjs
node scripts/reviews/run-examples.mjs
python scripts/reviews/full-agent-examples.py
node scripts/validate-site.mjs
bundle exec ruby scripts/validate-math.rb
```

文章审阅记录与可复算反例保存在 `scripts/reviews/`。Python 示例检查仅使用标准库（Python 3.10+）；这些检查验证特定公式、计数和状态反例，不等于真实 GPU 实测或全部内容的正确性证明。覆盖检查固定核对 2026-09-09 的 77 篇审阅记录，后续新增文章会另行提示，不冒充已经审阅。

量化系列另提供一个只依赖 Python 标准库的小工具：

```bash
python -B scripts/quantization-acceptance.py --self-test
python -B scripts/quantization-acceptance.py --input evidence.json
```

输入格式和可复算样本见[量化验收篇](_posts/2026-10-15-quantization-acceptance-engineering.md)。两个模式互斥；输入错误、数值范围错误和 packing parity 失败会非零退出。成功输出仍保留全栈 `incomplete`：它只检查所提供的 CPU 数值/布局数据，不认证来源、不同数据组的同源关系、真实 GPU 内核、模型任务质量或服务性能。独立专项由既有 `run-examples.mjs` 自动执行。

## 撰写文章

复制 `_drafts/template.md`，在 `_posts/` 中创建 `YYYY-MM-DD-slug.md`：

```yaml
---
layout: post
title: 文章标题
subtitle: 一句话说明文章解决的问题
date: 2026-08-09
author: 子辰
catalog: true
mathjax: false
tags: [LLM推理, KV Cache]
---
```

文章内图片统一放在 `assets/images/`，并使用 `/assets/images/...` 引用。

数学文章设置 `mathjax: true`。新写的行内公式采用 Kramdown 原生语法，例如正文中的 `$$H_{kv}$$`；独立公式将起止 `$$` 各放在一行。不要在 Markdown 正文直接写 `\(...\)` 或 `\[...\]`，这些反斜杠会被 Markdown 转义处理。原生数学节点由 Kramdown 转换为 MathJax 定界符，构建后的检查会逐式核对它们是否完整保留。浏览器是否成功加载 MathJax、公式是否正确排版仍需页面实测。

行内绝对值和范数使用 `\lvert x\rvert`、`\lVert x\rVert`，避免裸竖线被 GFM 误识别为表格分隔符。数学检查还会拒绝生成正文中残留的双美元定界符；只检查已经成功解析的数学节点，无法发现所有 Markdown 吞掉公式的情况。

标签从 `_data/tags.yml` 中选择，每篇使用 1–3 个；第一项应是最适合相关推荐的主主题。不要为仅一篇文章创建项目名标签，除非已有后续文章计划并能同步形成稳定专题。

## 发布

推送到 `master` 后，`.github/workflows/pages.yml` 会构建并发布站点。仓库首次启用该流程时，需要在 GitHub 的 **Settings → Pages** 中将 Source 设置为 **GitHub Actions**。

## License

站点代码采用 MIT License；文章与个人图片版权归作者所有。
