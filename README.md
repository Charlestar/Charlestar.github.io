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

需要 Ruby 3.1+ 与 Bundler。

```bash
bundle install
bundle exec jekyll serve --livereload
```

打开 <http://127.0.0.1:4000>。

构建后可运行本地完整性检查：

```bash
bundle exec jekyll build
node scripts/validate-site.mjs
```

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

标签从 `_data/tags.yml` 中选择，每篇使用 1–3 个；第一项应是最适合相关推荐的主主题。不要为仅一篇文章创建项目名标签，除非已有后续文章计划并能同步形成稳定专题。

## 发布

推送到 `master` 后，`.github/workflows/pages.yml` 会构建并发布站点。仓库首次启用该流程时，需要在 GitHub 的 **Settings → Pages** 中将 Source 设置为 **GitHub Actions**。

## License

站点代码采用 MIT License；文章与个人图片版权归作者所有。
