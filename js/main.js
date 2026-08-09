(() => {
  'use strict';

  const root = document.documentElement;
  const header = document.querySelector('[data-site-header]');
  const nav = document.getElementById('site-nav');
  const navToggle = document.querySelector('.nav-toggle');
  const themeToggle = document.querySelector('.theme-toggle');

  const setTheme = (theme, persist = true) => {
    root.dataset.theme = theme;
    if (persist) localStorage.setItem('theme', theme);
    const icon = document.querySelector('.theme-icon');
    if (icon) icon.textContent = theme === 'dark' ? '☀' : '◐';
    const giscus = document.querySelector('iframe.giscus-frame');
    if (giscus?.contentWindow) {
      giscus.contentWindow.postMessage({ giscus: { setConfig: { theme: theme === 'dark' ? 'dark' : 'light' } } }, 'https://giscus.app');
    }
  };

  setTheme(root.dataset.theme || 'light', false);
  themeToggle?.addEventListener('click', () => setTheme(root.dataset.theme === 'dark' ? 'light' : 'dark'));

  const colorPreference = matchMedia('(prefers-color-scheme: dark)');
  colorPreference.addEventListener?.('change', event => {
    if (!localStorage.getItem('theme')) setTheme(event.matches ? 'dark' : 'light', false);
  });

  navToggle?.addEventListener('click', () => {
    const open = navToggle.getAttribute('aria-expanded') !== 'true';
    navToggle.setAttribute('aria-expanded', String(open));
    nav?.classList.toggle('is-open', open);
  });

  document.addEventListener('click', event => {
    if (!nav?.classList.contains('is-open')) return;
    if (!event.target.closest('.nav-shell')) {
      nav.classList.remove('is-open');
      navToggle?.setAttribute('aria-expanded', 'false');
    }
  });

  const updateScrollUI = () => {
    header?.classList.toggle('is-scrolled', scrollY > 12);
    const article = document.querySelector('[data-article-body]');
    const progress = document.querySelector('.reading-progress');
    if (!article || !progress) return;
    const start = article.offsetTop;
    const distance = Math.max(article.offsetHeight - innerHeight, 1);
    const value = Math.min(100, Math.max(0, ((scrollY - start + 80) / distance) * 100));
    progress.style.width = `${value}%`;
  };
  addEventListener('scroll', updateScrollUI, { passive: true });
  updateScrollUI();

  const article = document.querySelector('[data-article-body]');
  if (article) {
    const text = article.textContent.trim();
    const minutes = Math.max(1, Math.ceil(text.length / 500));
    const readingTime = document.querySelector('[data-reading-time]');
    if (readingTime) readingTime.textContent = `约 ${minutes} 分钟阅读`;

    article.querySelectorAll('pre').forEach(pre => {
      const code = pre.querySelector('code');
      if (!code) return;

      const tools = document.createElement('span');
      tools.className = 'code-tools';
      const languageClass = [...code.classList].find(name => name.startsWith('language-'));
      if (languageClass) {
        const label = document.createElement('span');
        label.className = 'code-language';
        label.textContent = languageClass.replace('language-', '');
        tools.append(label);
      }

      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'code-copy';
      button.textContent = '复制';
      button.setAttribute('aria-label', '复制代码');
      button.addEventListener('click', async () => {
        try {
          await navigator.clipboard.writeText(code.innerText);
          button.textContent = '已复制';
          button.classList.add('is-copied');
          setTimeout(() => {
            button.textContent = '复制';
            button.classList.remove('is-copied');
          }, 1600);
        } catch (_) {
          button.textContent = '复制失败';
        }
      });
      tools.append(button);
      pre.append(tools);
    });

    const headings = [...article.querySelectorAll('h2, h3')];
    const tocTargets = document.querySelectorAll('[data-toc], [data-toc-mobile]');
    if (headings.length && tocTargets.length) {
      headings.forEach((heading, index) => {
        if (!heading.id) heading.id = `section-${index + 1}`;
      });

      tocTargets.forEach(target => {
        headings.forEach(heading => {
          const link = document.createElement('a');
          link.href = `#${heading.id}`;
          link.textContent = heading.textContent;
          link.className = `toc-level-${heading.tagName.slice(1)}`;
          target.append(link);
        });
      });

      if ('IntersectionObserver' in window) {
        const links = [...document.querySelectorAll('[data-toc] a')];
        const observer = new IntersectionObserver(entries => {
          entries.forEach(entry => {
            if (!entry.isIntersecting) return;
            links.forEach(link => link.classList.toggle('is-active', link.hash === `#${entry.target.id}`));
          });
        }, { rootMargin: '-18% 0px -72% 0px' });
        headings.forEach(heading => observer.observe(heading));
      }
    }
  }

  const copyLink = document.querySelector('[data-copy-link]');
  copyLink?.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(location.href);
      copyLink.textContent = '链接已复制';
      setTimeout(() => { copyLink.textContent = '复制链接'; }, 1600);
    } catch (_) {
      copyLink.textContent = '请复制地址栏链接';
    }
  });

  const searchInput = document.getElementById('site-search');
  const searchResults = document.getElementById('search-results');
  const searchData = document.getElementById('search-data');
  if (searchInput && searchResults && searchData) {
    let posts = [];
    try { posts = JSON.parse(searchData.textContent); } catch (_) {}

    const normalize = value => value.toLocaleLowerCase('zh-CN').replace(/\s+/g, ' ').trim();
    const renderResults = value => {
      const query = normalize(value);
      searchResults.replaceChildren();
      if (!query) {
        searchResults.hidden = true;
        return;
      }

      const matches = posts
        .map(post => {
          const title = normalize(post.title || '');
          const tags = normalize((post.tags || []).join(' '));
          const excerpt = normalize(post.excerpt || '');
          const score = (title.includes(query) ? 5 : 0) + (tags.includes(query) ? 3 : 0) + (excerpt.includes(query) ? 1 : 0);
          return { post, score };
        })
        .filter(item => item.score > 0)
        .sort((a, b) => b.score - a.score)
        .slice(0, 8);

      if (!matches.length) {
        const empty = document.createElement('p');
        empty.className = 'search-empty';
        empty.textContent = '没有找到相关文章';
        searchResults.append(empty);
      } else {
        matches.forEach(({ post }) => {
          const link = document.createElement('a');
          link.className = 'search-result';
          link.href = post.url;
          link.setAttribute('role', 'option');
          const title = document.createElement('strong');
          title.textContent = post.title;
          const meta = document.createElement('span');
          meta.textContent = `${post.date} · ${(post.tags || []).slice(0, 2).join(' / ')}`;
          link.append(title, meta);
          searchResults.append(link);
        });
      }
      searchResults.hidden = false;
    };

    let searchTimer;
    searchInput.addEventListener('input', () => {
      clearTimeout(searchTimer);
      searchTimer = setTimeout(() => renderResults(searchInput.value), 100);
    });
    searchInput.addEventListener('keydown', event => {
      if (event.key === 'Escape') {
        searchInput.value = '';
        searchResults.hidden = true;
        searchInput.blur();
      }
    });
    document.addEventListener('keydown', event => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') {
        event.preventDefault();
        searchInput.focus();
      }
    });
    document.addEventListener('click', event => {
      if (!event.target.closest('.search-card')) searchResults.hidden = true;
    });
  }

  if ('serviceWorker' in navigator) {
    addEventListener('load', () => navigator.serviceWorker.register('/sw.js').catch(() => {}));
  }
})();
