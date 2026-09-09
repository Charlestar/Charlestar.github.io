import { readdirSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

const directory = dirname(fileURLToPath(import.meta.url));
const root = resolve(directory, '../..');
const files = readdirSync(directory).filter(name => name.endsWith('-examples.mjs') && name !== 'run-examples.mjs').sort();
if (files.length === 0) throw new Error('No review examples found');

for (const name of files) {
  console.log(`\nRunning ${name}`);
  const result = spawnSync(process.execPath, [join(directory, name)], { cwd: root, stdio: 'inherit' });
  if (result.error) throw result.error;
  if (result.status !== 0) process.exit(result.status ?? 1);
}
console.log(`\nAll ${files.length} JavaScript review suites passed. Run full-agent-examples.py separately.`);
