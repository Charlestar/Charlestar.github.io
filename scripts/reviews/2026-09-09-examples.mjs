import assert from 'node:assert/strict';

// Recomputable mathematical examples for the 2026-09-09 blog review.
// No model, GPU, framework, network, or measured performance is involved.
// Run: node scripts/reviews/2026-09-09-examples.mjs
const sum = (values) => values.reduce((a, b) => a + b, 0);
const close = (actual, expected) => assert.ok(
  Number.isFinite(actual) && Math.abs(actual - expected) < 1e-12,
  `${actual} != ${expected}`,
);
let passed = 0;
function example(id, run) {
  run();
  passed += 1;
  console.log(`PASS ${id}`);
}

example('R01: quadratic form dimensions and cross term', () => {
  const [r4, r5, r6] = [5, 7, 11];
  const matrix = [[r4, r6 / 2], [r6 / 2, r5]];
  for (let x = -3; x <= 3; x += 1) {
    for (let y = -3; y <= 3; y += 1) {
      const vector = [x, y];
      const Av = matrix.map((row) => sum(row.map((v, j) => v * vector[j])));
      const quadratic = sum(vector.map((v, i) => v * Av[i]));
      close(quadratic, r4 * x ** 2 + r5 * y ** 2 + r6 * x * y);
    }
  }
});

example('R03: SmoothQuant endpoints and non-monotone absolute scaling', () => {
  const scale = (a, b, alpha) => a ** alpha / b ** (1 - alpha);
  const alphas = [0, 0.5, 1];
  assert.deepEqual(alphas.map((alpha) => scale(64, 0.01, alpha)), [100, 80, 64]);
  assert.deepEqual(alphas.map((alpha) => scale(64, 1, alpha)), [1, 8, 64]);
  for (const alpha of alphas) {
    const a = 64, b = 0.01, s = scale(a, b, alpha);
    close(a / s, (a * b) ** (1 - alpha));
    close(b * s, (a * b) ** alpha);
    close((a / s) * (b * s), a * b);
  }
  close(0.01 * scale(64, 0.01, 0), 1);
  close(64 / scale(64, 0.01, 1), 1);
});

example('R04: enumerate positive finite E4M3 encodings', () => {
  // FP8 Formats for Deep Learning, Table 1: bias 7, code 0x7f is NaN.
  const values = Array.from({ length: 127 }, (_, code) => {
    const exponent = code >> 3, fraction = code & 7;
    return { code, value: exponent === 0 ? fraction * 2 ** -9
      : (1 + fraction / 8) * 2 ** (exponent - 7) };
  });
  const nearest = (x) => values.reduce((best, candidate) => {
    const delta = Math.abs(candidate.value - x) - Math.abs(best.value - x);
    return delta < 0 || (delta === 0 && candidate.code % 2 === 0) ? candidate : best;
  }).value;
  close(values.at(-1).value, 448);
  close(nearest(1.1), 1.125);
  close(nearest(1.1 * 256) / 256, 1.125);
  close(nearest(1e-4 / (400 / 448)), 0);
  close(nearest(2 ** -9), 2 ** -9);
  close(nearest(2 ** -10), 0); // nearest-even at half the minimum subnormal
});

example('D03: actual checkpoint increments versus average amortization', () => {
  const k = 10, cost = 100;
  const increments = Array.from({ length: 30 }, (_, i) => (i + 1) % k === 0 ? cost : 0);
  close(increments[0], 0);
  close(increments[9], 100);
  close(sum(increments) / increments.length, cost / k);
  close(sum(increments) / increments.length, 10);
});

example('A02: top-p crosses its threshold; top-k-first renormalizes', () => {
  const probabilities = [0.4, 0.3, 0.2, 0.1]; // no ties in this example
  const nucleus = (k, p) => {
    const top = probabilities.map((value, index) => ({ value, index }))
      .sort((a, b) => b.value - a.value).slice(0, k);
    const mass = sum(top.map(({ value }) => value));
    const result = [];
    let cumulative = 0;
    for (const { value, index } of top) {
      result.push(index);
      cumulative += value / mass;
      if (cumulative >= p) break;
    }
    return result;
  };
  assert.deepEqual(nucleus(4, 0.6), [0, 1]);
  assert.deepEqual(nucleus(2, 0.5), [0]); // 0.4 / 0.7 > 0.5
  assert.ok(0.4 < 0.5 && 0.4 + 0.3 >= 0.5); // skipping renormalization keeps two
  assert.ok(0.4 <= 0.6 && 0.4 + 0.3 > 0.6); // old <= mask incorrectly keeps one
});

example('A04: independent matching and maximal coupling both preserve target', () => {
  const p = [0.7, 0.3], q = [0.4, 0.6], output = [0, 0];
  let matching = 0;
  for (let x = 0; x < p.length; x += 1) {
    for (let y = 0; y < q.length; y += 1) {
      const mass = p[x] * q[y];
      output[x === y ? y : x] += mass; // preserve this same target draw on mismatch
      if (x === y) matching += mass;
    }
  }
  output.forEach((value, i) => close(value, p[i]));
  close(matching, 0.46);
  const accepted = q.map((value, i) => value * Math.min(1, p[i] / value));
  const residual = p.map((value, i) => Math.max(0, value - q[i]));
  const rejection = 1 - sum(accepted);
  accepted.forEach((value, i) => close(value + rejection * residual[i] / sum(residual), p[i]));
  close(sum(accepted), 0.7);
  assert.ok(sum(accepted) > matching);
  const incorrectlyResampled = p.map((value, i) => value * q[i] + (1 - matching) * value);
  close(incorrectlyResampled[0], 0.658); // mismatch + a fresh target draw is biased here
});

example('S04: prefill produces token one; remaining outputs require decode', () => {
  for (const outputs of [0, 1, 8, 64, 20]) {
    const phases = Array.from({ length: outputs }, (_, position) => position ? 'decode' : 'prefill');
    assert.equal(phases.filter((phase) => phase === 'decode').length, Math.max(outputs - 1, 0));
    if (outputs > 0) {
      const inputTokens = 1000;
      const materializedKV = inputTokens + phases.filter((phase) => phase === 'decode').length;
      assert.equal(materializedKV, inputTokens + outputs - 1);
    }
  }
});

example('S05: suffix queries still attend to cached prefix keys', () => {
  const N = 100, H = 50;
  let full = 0, remaining = 0;
  for (let query = 0; query < N; query += 1) {
    for (let key = 0; key <= query; key += 1) {
      full += 1;
      if (query >= H) remaining += 1;
    }
  }
  assert.deepEqual([full, remaining, full - remaining], [5050, 3775, 1275]);
  const isolatedSuffix = (N - H) * (N - H + 1) / 2;
  assert.equal(remaining - isolatedSuffix, H * (N - H));
});

example('S06: overlapping transfers use wall-clock window for aggregate goodput', () => {
  const MiB = 2 ** 20, transfers = [{ bytes: MiB, start: 0, end: 1 }, { bytes: MiB, start: 0, end: 1 }];
  const bytes = sum(transfers.map((t) => t.bytes));
  const durations = sum(transfers.map((t) => t.end - t.start));
  const window = Math.max(...transfers.map((t) => t.end)) - Math.min(...transfers.map((t) => t.start));
  close(bytes / window, 2 * MiB);
  close(bytes / durations, MiB);
});

example('A10: committed tokens include correction/bonus in nonterminal rounds', () => {
  const expectations = [0.2, 0.3], costs = [1, 1.3];
  assert.ok(costs[0] / expectations[0] > costs[1] / expectations[1]);
  assert.ok(costs[0] / (expectations[0] + 1) < costs[1] / (expectations[1] + 1));
  const accepted = 0, committed = accepted + 1;
  close(costs[0] / committed, 1); // EOS/length truncation instead requires actual committed count
});

example('S09: urgency ratio applies only to positive remaining budget', () => {
  const classify = (service, budget) => budget <= 0 ? { state: 'expired' }
    : { state: 'pending', urgency: service / budget };
  close(classify(1, 0.1).urgency, 10);
  assert.equal(classify(1, 0).state, 'expired');
  assert.equal(classify(1, -0.1).state, 'expired');
  assert.equal(Number.isFinite(1 / 0), false);
  close(1 / -0.1, -10); // raw ratio would invert the intended priority
});

example('S10: every valid watermark state is covered, including emergency', () => {
  const state = (usage, allocatable) => !allocatable ? 'exhausted'
    : usage < 0.6 ? 'normal' : usage < 0.8 ? 'pressure' : usage < 0.9 ? 'critical' : 'emergency';
  assert.deepEqual([0, 0.6, 0.8, 0.9, 0.95, 1].map((u) => state(u, u < 1)),
    ['normal', 'pressure', 'critical', 'emergency', 'emergency', 'exhausted']);
  assert.equal(state(0.5, false), 'exhausted'); // no allocatable block takes precedence
  assert.ok(!(0.95 < 0.6 || (0.95 >= 0.6 && 0.95 < 0.8) || (0.95 >= 0.8 && 0.95 < 0.9)));
});

console.log(`Verified ${passed} mathematical examples; no GPU performance claims.`);
