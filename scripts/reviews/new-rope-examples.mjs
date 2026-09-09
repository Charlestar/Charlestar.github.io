// CPU-only independent checks for the RoPE/context-extension article.
// No model quality, CUDA kernel, or long-context benchmark is measured.
import assert from 'node:assert/strict';

const sum = (xs) => xs.reduce((a, b) => a + b, 0);
const dot = (a, b) => sum(a.map((x, i) => x * b[i]));
const close = (a, b, eps = 1e-10) => assert.ok(
  Number.isFinite(a) && Number.isFinite(b) && Math.abs(a - b) <= eps,
  `${a} != ${b}`,
);
let passed = 0;
function check(name, fn) {
  fn();
  passed += 1;
  console.log(`PASS ${name}`);
}
function rotate(x, position, frequencies) {
  assert.equal(x.length, 2 * frequencies.length);
  return frequencies.flatMap((frequency, j) => {
    const c = Math.cos(position * frequency), s = Math.sin(position * frequency);
    return [c * x[2 * j] - s * x[2 * j + 1],
      s * x[2 * j] + c * x[2 * j + 1]];
  });
}
const frequencies = (d, base = 10000) =>
  Array.from({ length: d / 2 }, (_, j) => base ** (-2 * j / d));
const split = (x) => [
  ...x.filter((_, i) => i % 2 === 0), ...x.filter((_, i) => i % 2 === 1),
];
function rotateSplit(x, position, freqs) {
  const out = Array(x.length), half = x.length / 2;
  freqs.forEach((frequency, j) => {
    const c = Math.cos(position * frequency), s = Math.sin(position * frequency);
    out[j] = c * x[j] - s * x[j + half];
    out[j + half] = s * x[j] + c * x[j + half];
  });
  return out;
}

check('rotation norm, relative-displacement sign, and common shift', () => {
  const q = [1, 2, -3, 0.5, 0.2, -0.4, 7, -2];
  const k = [-0.5, 1, 4, 2, -1, 0.3, 0.2, 1];
  const f = frequencies(q.length);
  for (const [m, n] of [[0, 0], [3, 9], [99, 12], [-2, 5]]) {
    const rq = rotate(q, m, f), rk = rotate(k, n, f);
    close(dot(rq, rq), dot(q, q));
    close(dot(rq, rk), dot(q, rotate(k, n - m, f)));
    close(dot(rq, rk), dot(rotate(q, m + 100, f), rotate(k, n + 100, f)));
  }
  assert.ok(Math.abs(dot(q, rotate(k, 6, f)) - dot(q, rotate(k, -6, f))) > 0.1);
});

check('article two-dimensional example and post-projection rotation', () => {
  const q = [1, 2], k = [3, 4], f = [Math.PI / 6];
  close(dot(rotate(q, 2, f), rotate(k, 5, f)), 2);
  close(dot(rotate(q, 12, f), rotate(k, 15, f)), 2);
  // W = diag(2, 1) does not commute with a quarter-turn.
  const projection = ([x, y]) => [2 * x, y];
  assert.notDeepEqual(rotate(projection([1, 1]), 1, [Math.PI / 2]),
    projection(rotate([1, 1], 1, [Math.PI / 2])));
});

check('article frequency/wavelength table and base 160000', () => {
  const f = frequencies(8);
  const wavelengths = [6.283, 62.832, 628.319, 6283.185];
  const rotations = [651.899, 65.190, 6.519, 0.652];
  f.forEach((value, j) => {
    close(2 * Math.PI / value, wavelengths[j], 0.000501);
    close(4096 * value / (2 * Math.PI), rotations[j], 0.000501);
  });
  close(10000 * 8 ** (8 / 6), 160000);
  frequencies(8, 160000).forEach((value, j) =>
    close(value, [1, 0.05, 0.0025, 0.000125][j]));
});

check('interleaved and split-half require a consistent coordinate permutation', () => {
  const q = [1, 2, 3, 4, 5, 6, 7, 8], k = [2, -1, 0.5, 1, -3, 4, 0.2, -0.7];
  const f = frequencies(8);
  split(rotate(q, 7, f)).forEach((x, i) => close(x, rotateSplit(split(q), 7, f)[i]));
  close(dot(rotate(q, 7, f), rotate(k, 13, f)),
    dot(rotateSplit(split(q), 7, f), rotateSplit(split(k), 13, f)));
  assert.ok(Math.abs(dot(rotate(q, 7, f), rotate(k, 13, f))
    - dot(rotateSplit(q, 7, f), rotateSplit(k, 13, f))) > 0.1);
});

check('PI scales positions or every frequency, including the highest', () => {
  const length = 4096, target = 32768, scale = target / length;
  const q = [1, 2, 3, 4, 5, 6, 7, 8], f = frequencies(8);
  for (const position of [0, 1, length - 1, target - 1]) {
    rotate(q, position / scale, f).forEach((x, i) =>
      close(x, rotate(q, position, f.map((v) => v / scale))[i]));
  }
  close((target - 1) / scale, 4095.875);
  assert.ok((target - 1) / scale < length);
  close(f[0] / scale, 0.125);
});

check('NTK-aware base change is not uniform position interpolation', () => {
  const d = 8, scale = 8, base = 10000;
  const f = frequencies(d, base), adjusted = frequencies(d, base * scale ** (d / (d - 2)));
  adjusted.forEach((value, j) => close(value / f[j], scale ** (-2 * j / (d - 2))));
  close(adjusted[0], f[0]);
  close(adjusted.at(-1), f.at(-1) / scale);
  assert.notEqual(adjusted[0], f[0] / scale);
});

check('orthogonal RoPE does not imply monotonically decaying attention scores', () => {
  const q = [1, 0], f = [1];
  const score = (distance) => dot(q, rotate(q, distance, f));
  close(score(0), 1);
  assert.ok(score(4) > score(3));
  assert.ok(score(6) > score(4));
});

check('YaRN amplitude squares logits only over the components actually scaled', () => {
  const amplitude = 1 + 0.1 * Math.log(8), temperature = 1 / amplitude ** 2;
  close(amplitude, 1.207944, 0.000001);
  close(amplitude ** 2, 1.45913, 0.00001);
  close(1 / temperature, amplitude ** 2);
  const qR = [1, 2], kR = [3, 4], qU = [2, -1], kU = [5, 3];
  const headDim = qR.length + qU.length;
  const score = (dot(qR, kR) + dot(qU, kU)) / Math.sqrt(headDim);
  const partial = (amplitude ** 2 * dot(qR, kR) + dot(qU, kU)) / Math.sqrt(headDim);
  assert.ok(Math.abs(partial - amplitude ** 2 * score) > 0.1);
  const gamma = (r) => Math.max(0, Math.min(1, (r - 1) / 31));
  close(gamma(0.652), 0);
  close(gamma(65.190), 1);
  close(gamma(16.5), 0.5);
});

check('cache keys cannot silently mix two positional coordinate systems', () => {
  const q = [1, 0, 0, 0], k = [1, 0, 0, 0], oldF = frequencies(4);
  const newF = oldF.map((f) => f / 4), m = 23, n = 7;
  const oldKey = rotate(k, n, oldF), newQuery = rotate(q, m, newF);
  const target = dot(newQuery, rotate(k, n, newF));
  assert.ok(Math.abs(dot(newQuery, oldKey) - target) > 0.1);
  // This only changes the rotation of a fixed content key; deeper-layer hidden
  // states can differ after changing the scheme, so full prefill is not proven equivalent.
  const correctedKey = rotate(oldKey, n, newF.map((f, j) => f - oldF[j]));
  close(dot(newQuery, correctedKey), target);
});

console.log(`${passed} RoPE reference groups passed; no model/GPU qualification.`);
