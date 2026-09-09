import assert from 'node:assert/strict';
import { readFileSync, readdirSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const directory = dirname(fileURLToPath(import.meta.url));
const root = resolve(directory, '../..');
const posts = readdirSync(join(root, '_posts')).filter(name => name.endsWith('.md'));
const records = { foundations: 20, attention: 20, serving: 18, training: 12, agents: 5, network: 2 };
const covered = new Set();

for (const [group, expectedCount] of Object.entries(records)) {
  const record = readFileSync(join(directory, `2026-09-09-full-${group}.md`), 'utf8');
  const members = posts.filter(name => record.includes(name));
  assert.equal(members.length, expectedCount, `${group}: missing, renamed or unexpected article in review record`);
  for (const name of members) {
    assert.ok(!covered.has(name), `Duplicate assignment in review records: ${name}`);
    covered.add(name);
  }
}
assert.equal(covered.size, 77, 'The 2026-09-09 snapshot must cover exactly 77 unique posts');
const newer = posts.filter(name => !covered.has(name));
console.log(`Verified 77 unique article records in the 2026-09-09 audit snapshot; ${posts.length} current posts.`);
if (newer.length) console.log(`Later posts are not covered by this historical audit:\n${newer.join('\n')}`);
console.log('Coverage records are evidence of review scope, not a proof of semantic correctness or a check of later edits.');
