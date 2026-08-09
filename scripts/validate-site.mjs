import fs from 'node:fs';
import path from 'node:path';
import process from 'node:process';

const root = path.resolve(process.argv[2] || '_site');
const failures = [];

if (!fs.existsSync(root)) {
  console.error(`Build output not found: ${root}`);
  process.exit(1);
}

const walk = directory => fs.readdirSync(directory, { withFileTypes: true }).flatMap(entry => {
  const fullPath = path.join(directory, entry.name);
  return entry.isDirectory() ? walk(fullPath) : [fullPath];
});

const files = walk(root);
const htmlFiles = files.filter(file => file.endsWith('.html'));
const sourceFiles = files.filter(file => /\.(?:html|css|js|xml)$/i.test(file));

for (const file of files) {
  if (fs.statSync(file).size === 0) failures.push(`Empty file: ${path.relative(root, file)}`);
}

for (const file of htmlFiles) {
  const html = fs.readFileSync(file, 'utf8');
  const attributePattern = /\b(?:href|src)=(?:"([^"]+)"|'([^']+)')/gi;
  for (const match of html.matchAll(attributePattern)) {
    const raw = match[1] || match[2];
    if (!raw || /^(?:[a-z]+:|\/\/|#)/i.test(raw)) continue;

    let pathname = raw.split(/[?#]/, 1)[0];
    try { pathname = decodeURIComponent(pathname); } catch {}
    if (!pathname) continue;

    let target = pathname.startsWith('/')
      ? path.join(root, pathname.slice(1))
      : path.resolve(path.dirname(file), pathname);

    if (pathname.endsWith('/') || (fs.existsSync(target) && fs.statSync(target).isDirectory())) {
      target = path.join(target, 'index.html');
    }

    if (!fs.existsSync(target)) {
      failures.push(`Missing ${raw} referenced by ${path.relative(root, file)}`);
    }
  }

  const jsonLdPattern = /<script\b[^>]*type=["']application\/ld\+json["'][^>]*>([\s\S]*?)<\/script>/gi;
  for (const match of html.matchAll(jsonLdPattern)) {
    try { JSON.parse(match[1]); }
    catch (error) { failures.push(`Invalid JSON-LD in ${path.relative(root, file)}: ${error.message}`); }
  }

  const embeddedJsonPattern = /<script\b[^>]*type=["']application\/json["'][^>]*>([\s\S]*?)<\/script>/gi;
  for (const match of html.matchAll(embeddedJsonPattern)) {
    try { JSON.parse(match[1]); }
    catch (error) { failures.push(`Invalid embedded JSON in ${path.relative(root, file)}: ${error.message}`); }
  }
}

const forbidden = /qiubaiying|hux[ -]?blog|jquery|bootstrap|gruntfile|gitcafe/i;
for (const file of sourceFiles) {
  const content = fs.readFileSync(file, 'utf8');
  if (forbidden.test(content)) failures.push(`Legacy template marker in ${path.relative(root, file)}`);
}

JSON.parse(fs.readFileSync(path.join(root, 'pwa', 'manifest.json'), 'utf8'));

if (failures.length) {
  console.error(failures.join('\n'));
  process.exit(1);
}

console.log(`Validated ${htmlFiles.length} HTML pages and ${files.length} generated files.`);
