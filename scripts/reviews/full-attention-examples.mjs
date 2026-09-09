// Small CPU reference checks for the 2026-09-09 independent article review.
// Run: node scripts/reviews/full-attention-examples.mjs
// These do not execute CUDA, vLLM, SpecForge, or distributed kernels.
import assert from 'node:assert/strict';

let groups = 0;
function test(name, fn) {
  fn();
  groups += 1;
  console.log(`PASS ${name}`);
}
const sum = (xs) => xs.reduce((a, b) => a + b, 0);
const dot = (a, b) => sum(a.map((x, i) => x * b[i]));
function close(a, b, eps = 1e-10) {
  assert.ok(Math.abs(a - b) <= eps, `${a} != ${b}`);
}
function softmax(xs) {
  const m = Math.max(...xs);
  const exps = xs.map((x) => Math.exp(x - m));
  const z = sum(exps);
  return exps.map((x) => x / z);
}

test('input computation frontier is not committed sequence length', () => {
  const prompt = 1000;
  const scheduledInput = prompt;
  const materializedInput = prompt;
  const committedSequence = prompt + 1; // Prefill logits sample one output.
  assert.ok(scheduledInput >= materializedInput);
  assert.ok(materializedInput < committedSequence);
});

test('FairBatching budget floor is not a hard deadline guarantee', () => {
  const slackMs = [10, 80];
  const tpotMs = [50, 100];
  const budget = Math.max(Math.min(...slackMs), Math.min(...tpotMs));
  assert.equal(budget, 50);
  assert.ok(budget > Math.min(...slackMs));
});

test('DSA selects indices, excludes future positions, and handles ties', () => {
  function indices(scores, position, k) {
    // Index order is this reference's tie rule, not a claim about CUDA topk.
    return scores.map((score, index) => ({ score, index }))
      .filter(({ index }) => index <= position)
      .sort((a, b) => b.score - a.score || a.index - b.index)
      .slice(0, Math.min(k, position + 1))
      .map(({ index }) => index);
  }
  const scores = [3, 2, 2, 1];
  const selectedValues = new Set([3, 2]);
  assert.equal(scores.filter((x) => selectedValues.has(x)).length, 3);
  assert.deepEqual(indices(scores, 3, 2), [0, 1]);
  assert.deepEqual(indices([1, 100, 200], 0, 2048), [0]);
});

test('single-pivot sampling distribution and finite worst-case bound', () => {
  const scores = [11, 7, 5, 3, 1];
  for (let k = 1; k <= scores.length; k += 1) {
    function recurse(threshold) {
      const active = scores.map((p, i) => ({ p, i }))
        .filter(({ p }) => p > threshold);
      const total = sum(active.map(({ p }) => p));
      const output = scores.map(() => 0);
      let maxRounds = 1;
      for (const { p, i } of active) {
        if (i < k) {
          output[i] += p / total;
        } else {
          const rest = recurse(p);
          rest.output.forEach((q, j) => { output[j] += p / total * q; });
          maxRounds = Math.max(maxRounds, 1 + rest.maxRounds);
        }
      }
      return { output, maxRounds };
    }
    const result = recurse(0);
    const topMass = sum(scores.slice(0, k));
    result.output.forEach((p, i) => close(p, i < k ? scores[i] / topMass : 0));
    assert.equal(result.maxRounds, scores.length - k + 1);
  }
});

test('continuous batching consumes both token and sequence budgets', () => {
  function schedule(running, waiting, tokenBudget, seqBudget) {
    const batch = [];
    for (const requested of [...running, ...waiting]) {
      if (tokenBudget <= 0 || seqBudget <= 0) break;
      const work = Math.min(requested, tokenBudget);
      if (work <= 0) continue;
      batch.push(work);
      tokenBudget -= work;
      seqBudget -= 1;
    }
    return batch;
  }
  assert.deepEqual(schedule([1, 1, 1], [4096], 2048, 2), [1, 1]);
  assert.deepEqual(schedule([1, 1], [4096], 2048, 3), [1, 1, 2046]);
  assert.deepEqual(schedule([], [4096], 0, 3), []);
  assert.deepEqual(schedule([1], [4096], 2048, 0), []);
});

test('chunked prefill token and block arithmetic', () => {
  const chunks = [2016, 2016, 2016, 952];
  assert.equal(sum(chunks), 7000);
  chunks.forEach((c) => assert.ok(c + 32 <= 2048));
  assert.equal(Math.ceil(1000 / 16), 63);
  // One free slot at the tail is not enough to eliminate the next block.
  assert.equal(Math.ceil((15 + 1000) / 16) - Math.ceil(15 / 16), 63);
  // Eight free slots are sufficient in this specific example.
  assert.equal(Math.ceil((8 + 1000) / 16) - Math.ceil(8 / 16), 62);
});

function onlineRow(tiles, valueDim) {
  let m = -Infinity;
  let denominator = 0;
  let numerator = Array(valueDim).fill(0);
  for (const { scores, values } of tiles) {
    const active = scores.map((score, i) => ({ score, value: values[i] }))
      .filter(({ score }) => score !== -Infinity);
    if (active.length === 0) continue;
    assert.ok(active.every(({ score }) => Number.isFinite(score)));
    const nextM = Math.max(m, ...active.map(({ score }) => score));
    const rescale = denominator === 0 ? 0 : Math.exp(m - nextM);
    numerator = numerator.map((x) => x * rescale);
    denominator *= rescale;
    for (const { score, value } of active) {
      const mass = Math.exp(score - nextM);
      denominator += mass;
      numerator = numerator.map((x, j) => x + mass * value[j]);
    }
    m = nextM;
  }
  return denominator === 0 ? Array(valueDim).fill(0)
    : numerator.map((x) => x / denominator);
}

test('online softmax handles empty first, middle, and global rows', () => {
  const empty = { scores: [-Infinity], values: [[100, 100]] };
  const a = { scores: [1000, 1001], values: [[1, 2], [3, 4]] };
  const b = { scores: [998], values: [[7, -2]] };
  const weights = softmax([...a.scores, ...b.scores]);
  const values = [...a.values, ...b.values];
  const reference = [0, 1].map((j) => sum(weights.map((p, i) => p * values[i][j])));
  for (const tiles of [[empty, a, empty, b], [b, empty, a], [a, b, empty]]) {
    onlineRow(tiles, 2).forEach((x, j) => close(x, reference[j]));
  }
  assert.deepEqual(onlineRow([empty, empty], 2), [0, 0]);
  assert.deepEqual(onlineRow([], 2), [0, 0]);
  assert.ok(Number.isNaN(Math.exp(-Infinity - -Infinity)));
});

test('FA2 rowwise D identity and softmax backward finite differences', () => {
  const scores = [0.2, -0.3, 1.1];
  const values = [[1, 2], [-1, 3], [4, -2]];
  const upstream = [0.7, -0.2];
  const p = softmax(scores);
  const o = [0, 1].map((j) => sum(p.map((q, i) => q * values[i][j])));
  const dp = values.map((value) => dot(upstream, value));
  const d = dot(p, dp);
  close(d, dot(o, upstream));
  const grad = p.map((q, i) => q * (dp[i] - d));
  function loss(s) { return dot(softmax(s), dp); }
  scores.forEach((_, i) => {
    const plus = [...scores];
    const minus = [...scores];
    plus[i] += 1e-5;
    minus[i] -= 1e-5;
    close(grad[i], (loss(plus) - loss(minus)) / 2e-5, 1e-8);
  });
  close(sum(grad), 0);
});

test('Ring visits each owner with P computations and P-1 transfers', () => {
  for (const size of [1, 2, 4, 8]) {
    for (let rank = 0; rank < size; rank += 1) {
      const owners = [];
      let transfers = 0;
      for (let step = 0; step < size; step += 1) {
        owners.push((rank - step + size) % size);
        if (step + 1 < size) transfers += 1;
      }
      assert.equal(new Set(owners).size, size);
      assert.equal(transfers, size - 1);
    }
  }
  const time = (p, compute, comm) => (p - 1) * Math.max(compute, comm) + compute;
  assert.equal(time(8, 3, 2), 24);
  assert.equal(time(8, 3, 5), 38);
  assert.equal(time(1, 3, 5), 3);
  const sent = 2 * (8 - 1) * (128 * 1024 / 8) * 32 * 128 * 2;
  assert.equal(sent, 1879048192);
  assert.equal(sent + sent, 3758096384);
});

test('MLA absorption preserves score/value and original softmax scale', () => {
  const upK = [[1, 2, -1], [-0.5, 0.3, 2]]; // [d_h=2, d_c=3]
  const upV = [[0.2, 1, -1], [2, -0.5, 0.1]];
  const q = [0.4, -0.7];
  const latent = [[1, 0, 2], [-1, 1, 0.5], [0.3, -0.8, 1.1]];
  const qr = [0.1, 0.4];
  const kr = [[0.2, 0.3], [-0.5, 0.1], [0.7, -0.2]];
  const absorbedQ = [0, 1, 2].map((j) => sum(upK.map((row, i) => row[j] * q[i])));
  const explicit = latent.map((c, i) => dot(q, upK.map((row) => dot(row, c))) + dot(qr, kr[i]));
  const absorbed = latent.map((c, i) => dot(absorbedQ, c) + dot(qr, kr[i]));
  explicit.forEach((x, i) => close(x, absorbed[i]));
  const p = softmax(explicit.map((x) => x / Math.sqrt(2 + 2)));
  const incorrect = softmax(absorbed.map((x) => x / Math.sqrt(3 + 2)));
  assert.ok(p.some((x, i) => Math.abs(x - incorrect[i]) > 1e-4));
  const weightedLatent = [0, 1, 2].map((j) => sum(latent.map((c, i) => p[i] * c[j])));
  upV.forEach((row) => close(
    sum(latent.map((c, i) => p[i] * dot(row, c))), dot(row, weightedLatent),
  ));
});

test('QuantSpec buffer rotation waits for confirmed materialized KV', () => {
  const group = 128;
  const tentative = 128;
  const acceptedKV = 12;
  assert.equal(tentative, group);
  assert.ok(acceptedKV < group);
  const rotate = (confirmedKV) => confirmedKV >= group;
  assert.equal(rotate(acceptedKV), false);
  assert.equal(rotate(group), true);
  assert.equal(Math.min(31, group), 31); // Short prompts cannot retain G entries.
});

test('EAGLE time per token is total time / total tokens, not mean of ratios', () => {
  const times = [5, 20];
  const tokens = [1, 10];
  const renewalRatio = sum(times) / sum(tokens);
  const meanRoundRatio = sum(times.map((t, i) => t / tokens[i])) / times.length;
  close(renewalRatio, (sum(times) / 2) / (sum(tokens) / 2));
  assert.notEqual(renewalRatio, meanRoundRatio);
  close(sum(softmax([-1, 0, 2])), 1);
});

test('NVFP4 payload, scale overhead, and illustrative KV sizes', () => {
  const elements = 2 * 80 * 8 * 128;
  assert.equal(elements * 2 / 1024, 320);
  assert.equal(elements * 2 * (128 * 1024) / 1024 ** 3, 40);
  const bits = (16 * 4 + 8) / 16;
  assert.equal(bits, 4.5);
  close(16 / bits, 32 / 9);
});

console.log(`All ${groups} reference groups passed (CPU-only; no production-kernel qualification).`);
