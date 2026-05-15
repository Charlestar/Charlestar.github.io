/* ============================================================
   Blog Optimizations - JavaScript
   Dark Mode, Copy Button, Reading Progress, Mobile Catalog,
   Related Posts, Search
   ============================================================ */

(function() {
  'use strict';

  // ── Dark Mode ──
  function initDarkMode() {
    const html = document.documentElement;
    const stored = localStorage.getItem('theme');
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;

    if (stored === 'dark' || (!stored && prefersDark)) {
      html.setAttribute('data-theme', 'dark');
    }

    // Update toggle button icon
    updateThemeIcon();

    // Listen for system theme changes
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', function(e) {
      if (!localStorage.getItem('theme')) {
        html.setAttribute('data-theme', e.matches ? 'dark' : 'light');
        updateThemeIcon();
      }
    });
  }

  function toggleTheme() {
    const html = document.documentElement;
    const current = html.getAttribute('data-theme');
    const next = current === 'dark' ? 'light' : 'dark';
    html.setAttribute('data-theme', next);
    localStorage.setItem('theme', next);
    updateThemeIcon();
  }

  function updateThemeIcon() {
    const icon = document.querySelector('.theme-toggle i');
    if (icon) {
      const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
      icon.className = isDark ? 'fas fa-sun' : 'fas fa-moon';
    }
  }

  // ── Code Block Copy Button ──
  function initCopyButtons() {
    const pres = document.querySelectorAll('.post-container pre');
    pres.forEach(function(pre) {
      // Detect language from class
      const codeEl = pre.querySelector('code');
      let lang = '';
      if (codeEl) {
        const classes = codeEl.className.split(' ');
        for (let i = 0; i < classes.length; i++) {
          const match = classes[i].match(/language-(\w+)/);
          if (match) {
            lang = match[1];
            break;
          }
        }
      }

      // Language label
      if (lang) {
        const label = document.createElement('span');
        label.className = 'code-lang-label';
        label.textContent = lang;
        pre.appendChild(label);
      }

      // Copy button
      const btn = document.createElement('button');
      btn.className = 'code-copy-btn';
      btn.textContent = 'Copy';
      btn.setAttribute('aria-label', 'Copy code to clipboard');

      btn.addEventListener('click', function() {
        const text = pre.textContent.replace(/Copy$/, '').trim();
        navigator.clipboard.writeText(text).then(function() {
          btn.textContent = '✓ Copied!';
          btn.classList.add('copied');
          setTimeout(function() {
            btn.textContent = 'Copy';
            btn.classList.remove('copied');
          }, 2000);
        }).catch(function() {
          // Fallback for older browsers
          const textarea = document.createElement('textarea');
          textarea.value = text;
          textarea.style.position = 'fixed';
          textarea.style.opacity = '0';
          document.body.appendChild(textarea);
          textarea.select();
          document.execCommand('copy');
          document.body.removeChild(textarea);
          btn.textContent = '✓ Copied!';
          btn.classList.add('copied');
          setTimeout(function() {
            btn.textContent = 'Copy';
            btn.classList.remove('copied');
          }, 2000);
        });
      });

      pre.appendChild(btn);
    });
  }

  // ── Reading Progress Bar ──
  function initReadingProgress() {
    // Create progress bar element
    const bar = document.createElement('div');
    bar.className = 'reading-progress-bar';
    document.body.appendChild(bar);

    function updateProgress() {
      const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
      const docHeight = document.documentElement.scrollHeight - document.documentElement.clientHeight;
      const progress = docHeight > 0 ? (scrollTop / docHeight) * 100 : 0;
      bar.style.width = Math.min(progress, 100) + '%';
    }

    window.addEventListener('scroll', updateProgress, { passive: true });
    updateProgress();
  }

  // ── Mobile Catalog ──
  function initMobileCatalog() {
    const catalogToggle = document.querySelector('.mobile-catalog-toggle');
    const mobileCatalog = document.querySelector('.mobile-catalog');

    if (catalogToggle && mobileCatalog) {
      catalogToggle.addEventListener('click', function(e) {
        e.preventDefault();
        mobileCatalog.classList.toggle('open');
      });

      // Sync with desktop catalog
      const desktopBody = document.querySelector('.side-catalog .catalog-body');
      const mobileBody = document.querySelector('.mobile-catalog-body');
      if (desktopBody && mobileBody) {
        mobileBody.innerHTML = '<ul>' + desktopBody.innerHTML + '</ul>';
      }
    }
  }

  // ── Related Posts ──
  function initRelatedPosts() {
    const container = document.getElementById('related-posts');
    if (!container) return;

    const currentTags = JSON.parse(container.getAttribute('data-tags') || '[]');
    const currentUrl = container.getAttribute('data-current-url') || '';
    const allPosts = JSON.parse(container.getAttribute('data-posts') || '[]');

    if (!allPosts || allPosts.length === 0) {
      container.style.display = 'none';
      return;
    }

    // Score posts by tag overlap
    const scored = allPosts
      .filter(function(post) { return post.url !== currentUrl; })
      .map(function(post) {
        let score = 0;
        if (post.tags) {
          post.tags.forEach(function(tag) {
            if (currentTags.indexOf(tag) !== -1) score++;
          });
        }
        return { post: post, score: score };
      })
      .sort(function(a, b) { return b.score - a.score; })
      .slice(0, 3);

    if (scored.length === 0) {
      container.style.display = 'none';
      return;
    }

    const section = document.createElement('div');
    section.className = 'related-posts';

    const heading = document.createElement('h4');
    heading.textContent = '📚 Related Articles';
    section.appendChild(heading);

    scored.forEach(function(item) {
      const a = document.createElement('a');
      a.className = 'related-post-item';
      a.href = item.post.url;

      const title = document.createElement('span');
      title.className = 'related-title';
      title.textContent = item.post.title;

      const date = document.createElement('span');
      date.className = 'related-date';
      date.textContent = item.post.date;

      a.appendChild(title);
      a.appendChild(date);
      section.appendChild(a);
    });

    container.appendChild(section);
  }

  // ── Search (Lunr.js integration) ──
  function initSearch() {
    const searchInput = document.getElementById('site-search');
    const searchResults = document.getElementById('search-results');
    const searchData = document.getElementById('search-data');

    if (!searchInput || !searchResults) return;

    let idx = null;

    function buildIndex() {
      if (!searchData) return;
      try {
        const data = JSON.parse(searchData.textContent);
        idx = lunr(function() {
          this.ref('url');
          this.field('title', { boost: 10 });
          this.field('tags', { boost: 5 });
          this.field('content');
          data.forEach(function(doc) {
            this.add(doc);
          }.bind(this));
        });
      } catch(e) {
        console.warn('Search index build failed:', e);
      }
    }

    function search(query) {
      if (!idx || !query || query.length < 2) {
        searchResults.style.display = 'none';
        return;
      }

      const results = idx.search(query);
      if (results.length === 0) {
        searchResults.innerHTML = '<div class="search-result-item"><span style="color:var(--text-muted)">No results found</span></div>';
        searchResults.style.display = 'block';
        return;
      }

      searchResults.innerHTML = '';
      results.forEach(function(result) {
        const doc = idx.get(result.ref);
        if (!doc) return;

        const item = document.createElement('div');
        item.className = 'search-result-item';

        const link = document.createElement('a');
        link.href = doc.url;
        link.textContent = doc.title;

        const date = document.createElement('div');
        date.className = 'search-result-date';
        date.textContent = doc.date;

        const excerpt = document.createElement('div');
        excerpt.className = 'search-result-excerpt';
        const content = doc.content || '';
        excerpt.textContent = content.substring(0, 120) + (content.length > 120 ? '...' : '');

        item.appendChild(link);
        item.appendChild(date);
        item.appendChild(excerpt);
        searchResults.appendChild(item);
      });

      searchResults.style.display = 'block';
    }

    let debounceTimer;
    searchInput.addEventListener('input', function() {
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(function() {
        if (!idx) buildIndex();
        search(searchInput.value.trim());
      }, 200);
    });

    // Close search results on outside click
    document.addEventListener('click', function(e) {
      if (!e.target.closest('.search-container')) {
        searchResults.style.display = 'none';
      }
    });
  }

  // ── Initialize ──
  document.addEventListener('DOMContentLoaded', function() {
    initDarkMode();
    initCopyButtons();
    initReadingProgress();
    initMobileCatalog();
    initRelatedPosts();
    initSearch();
  });

  // ── Global Functions ──
  window.toggleTheme = toggleTheme;

})();
