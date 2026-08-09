import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { join, relative } from 'node:path';

const root = process.cwd();
const postsDir = join(root, '_posts');
const seriesFile = join(root, '_data', 'series.yml');
const posts = readdirSync(postsDir)
  .filter((name) => name.endsWith('.md'))
  .sort();
const errors = [];
const warnings = [];
const seriesText = existsSync(seriesFile) ? readFileSync(seriesFile, 'utf8') : '';
const seriesIds = new Set([...seriesText.matchAll(/^\s*- id:\s*([a-z][a-z0-9-]+)\s*$/gm)].map(match => match[1]));
const seriesPositions = new Map();

if (!seriesIds.size) errors.push('_data/series.yml: no series definitions found');

for (const name of posts) {
  const file = join(postsDir, name);
  const text = readFileSync(file, 'utf8');
  const lines = text.split(/\r?\n/);
  const secondDelimiter = lines.indexOf('---', 1);

  if (lines[0] !== '---' || secondDelimiter < 2) {
    errors.push(`${name}: invalid YAML front matter`);
    continue;
  }

  const front = lines.slice(1, secondDelimiter).join('\n');
  for (const field of ['layout', 'title', 'date', 'last_modified_at', 'author', 'tags']) {
    if (!new RegExp(`^${field}:`, 'm').test(front)) {
      errors.push(`${name}: missing ${field}`);
    }
  }

  const series = front.match(/^series:\s*([^\s#]+)\s*$/m)?.[1];
  const seriesOrder = front.match(/^series_order:\s*(\d+)\s*$/m)?.[1];
  const technologyYear = front.match(/^technology_year:\s*(\d{4})\s*$/m)?.[1];
  if (series) {
    if (!seriesIds.has(series)) errors.push(`${name}: unknown series ${series}`);
    if (!seriesOrder) errors.push(`${name}: series requires a positive numeric series_order`);
    if (!technologyYear) errors.push(`${name}: series requires a four-digit technology_year`);
    if (seriesOrder) {
      const position = `${series}:${seriesOrder}`;
      if (seriesPositions.has(position)) {
        errors.push(`${name}: duplicate series position ${position} also used by ${seriesPositions.get(position)}`);
      } else {
        seriesPositions.set(position, name);
      }
    }
  } else if (seriesOrder || technologyYear) {
    errors.push(`${name}: series_order and technology_year require series`);
  }

  let inFence = false;
  const headings = new Map();
  for (let index = secondDelimiter + 1; index < lines.length; index += 1) {
    const line = lines[index];
    if (/^\s*```/.test(line)) {
      inFence = !inFence;
      continue;
    }
    if (inFence) continue;

    const heading = line.match(/^(#{1,6})\s+(.+?)\s*$/);
    if (!heading) continue;
    if (heading[1].length === 1) {
      errors.push(`${name}:${index + 1}: H1 duplicates the layout-rendered title`);
    }
    const key = heading[2].replace(/[`*_]/g, '').trim().toLowerCase();
    if (headings.has(key)) {
      errors.push(`${name}:${index + 1}: duplicate heading "${heading[2]}"`);
    } else {
      headings.set(key, index + 1);
    }
  }
  if (inFence) errors.push(`${name}: unclosed fenced code block`);

  for (const match of text.matchAll(/!\[[^\]]*\]\((\/[^)\s]+)(?:\s+"[^"]*")?\)/g)) {
    const localPath = decodeURIComponent(match[1]).replace(/^\//, '');
    if (!existsSync(join(root, localPath))) {
      errors.push(`${name}: missing image ${match[1]}`);
    }
  }

  if (/\b(?:TODO|TBD|FIXME)\b|To be continue|lorem ipsum/i.test(text)) {
    errors.push(`${name}: contains unfinished-content marker`);
  }
  if (/https?:\/\/(?:[^/]+\.)?example\.com/i.test(text)) {
    errors.push(`${name}: contains placeholder domain example.com`);
  }
  if (/根据我们的(?:基准|测试|实验)|某大型(?:公司|企业|平台)/.test(text)) {
    warnings.push(`${name}: contains a claim that needs an explicit source`);
  }
}

if (warnings.length) {
  console.warn(`Content warnings (${warnings.length}):\n- ${warnings.join('\n- ')}`);
}
if (errors.length) {
  console.error(`Content audit failed (${errors.length}):\n- ${errors.join('\n- ')}`);
  process.exit(1);
}

console.log(`Audited ${posts.length} posts and ${seriesIds.size} series: metadata, headings, code fences, local images, and residual markers passed.`);
