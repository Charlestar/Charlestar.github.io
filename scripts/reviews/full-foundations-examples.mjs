import assert from 'node:assert/strict';

// Small deterministic mathematical counterexamples, not GPU benchmarks.
const close = (a, b) => assert.ok(Math.abs(a - b) < 1e-11, `${a} != ${b}`);
let checks = 0;
function check(name, run) { run(); checks++; console.log(`PASS ${name}`); }

check('conditional downsampling posterior and inverse, including endpoints', () => {
  for (const p of [0, 0.001, 0.05, 0.5, 0.9, 1]) {
    for (const alpha of [0.01, 0.2, 1]) {
      const sampled = p / (p + alpha * (1 - p));
      close(alpha * sampled / (1 - sampled + alpha * sampled), p);
    }
  }
});

function reference(scores, values, allowed) {
  const valid = scores.map((_, i) => i).filter(i => allowed[i]);
  if (!valid.length) return 0; // Explicit API convention, not softmax(empty).
  const max = Math.max(...valid.map(i => scores[i]));
  const weights = valid.map(i => Math.exp(scores[i] - max));
  return weights.reduce((sum, w, j) => sum + w * values[valid[j]], 0)
    / weights.reduce((sum, w) => sum + w, 0);
}
function streamed(scores, values, allowed, tileSize) {
  let m = -Infinity, l = 0, o = 0;
  for (let start = 0; start < scores.length; start += tileSize) {
    const ids = scores.map((_, i) => i).slice(start, start + tileSize).filter(i => allowed[i]);
    if (!ids.length) continue;
    const next = Math.max(m, ...ids.map(i => scores[i]));
    const alpha = l === 0 ? 0 : Math.exp(m - next);
    const p = ids.map(i => Math.exp(scores[i] - next));
    l = alpha * l + p.reduce((sum, value) => sum + value, 0);
    o = alpha * o + p.reduce((sum, value, j) => sum + value * values[ids[j]], 0);
    m = next;
  }
  return l > 0 ? o / l : 0;
}
check('online softmax: all 64 masks, empty first/middle/all tiles, extreme logits', () => {
  const scores = [-1000, 4, -3, 900, 899, 2];
  const values = [2, -1, 0.4, 3, 7, -2];
  for (let bits = 0; bits < 64; bits++) {
    const allowed = scores.map((_, i) => Boolean(bits & (1 << i)));
    for (const tile of [1, 2, 3, 6]) close(streamed(scores, values, allowed, tile), reference(scores, values, allowed));
  }
});

check('GQA binary units and materialized KV cursor', () => {
  for (const [heads, bytes] of [[32, 2 * 2 ** 30], [8, 512 * 2 ** 20], [1, 64 * 2 ** 20]]) {
    assert.equal(2 * 32 * 4096 * heads * 128 * 2, bytes);
  }
  const prompt = 1000;
  for (let outputs = 1; outputs <= 10; outputs++) {
    const kv = prompt + outputs - 1;
    assert.equal(kv + 1, prompt + outputs);
  }
});

check('shared K/V does not imply shared attention output for different queries', () => {
  const keys = [-1, 1], values = [0, 1];
  const a = reference(keys.map(k => 2 * k), values, [true, true]);
  const b = reference(keys.map(k => -2 * k), values, [true, true]);
  assert.ok(a > 0.98 && b < 0.02);
  close(a + b, 1);
});

check('independent matching must preserve the SAME target draw', () => {
  const p = [0.7, 0.3], q = [0.5, 0.5];
  const exact = [0, 0], incorrect = [0, 0];
  for (let y = 0; y < 2; y++) for (let z = 0; z < 2; z++) {
    const mass = q[y] * p[z];
    exact[z] += mass;
    if (y === z) incorrect[z] += mass;
    else for (let x = 0; x < 2; x++) incorrect[x] += mass * p[x];
  }
  exact.forEach((value, i) => close(value, p[i]));
  // This q happens to be uniform: use a nonuniform proposal to expose resampling bias.
  const nonuniform = [0.9, 0.1];
  const reject = 1 - p.reduce((sum, value, i) => sum + value * nonuniform[i], 0);
  assert.ok(Math.abs(nonuniform[0] * p[0] + reject * p[0] - p[0]) > 0.1);
});

check('local grammar renormalization differs from globally conditioned sequence law', () => {
  // Only AX and BY are valid. First tokens A/B both have legal continuations.
  const local = [0.5, 0.5];
  const rawValid = [0.5 * 0.9, 0.5 * 0.1];
  const global = rawValid.map(x => x / rawValid.reduce((a, b) => a + b, 0));
  close(global[0], 0.9);
  assert.notDeepEqual(local, global);
});

console.log(`Validated ${checks} foundation/serving mathematical examples; no GPU or full-engine execution claimed.`);
