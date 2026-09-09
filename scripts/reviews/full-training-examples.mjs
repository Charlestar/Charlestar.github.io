// Reproducible arithmetic/finite-state examples for the 2026-09-09 full audit.
// No GPU, NCCL, training convergence, or wall-clock performance is measured here.
import assert from 'node:assert/strict';

const sum = (xs) => xs.reduce((a, b) => a + b, 0);
const near = (a, b, eps = 1e-10) => assert.ok(Math.abs(a - b) <= eps, `${a} != ${b}`);
let passed = 0;
function check(name, fn) {
  fn();
  passed++;
  console.log(`PASS ${name}`);
}

check('CP: skip fully masked blocks, then merge online softmax', () => {
  const blocks = [[[-Infinity, 99]], [[2, 3], [1, 5]], [], [[4, 7]]];
  let state = null;
  for (const block of blocks) {
    const valid = block.filter(([s]) => s !== -Infinity);
    if (!valid.length) continue;
    const m = Math.max(...valid.map(([s]) => s));
    const l = sum(valid.map(([s]) => Math.exp(s - m)));
    const u = sum(valid.map(([s, v]) => Math.exp(s - m) * v));
    if (state === null) { state = { m, l, u }; continue; }
    const nextM = Math.max(state.m, m);
    const a = Math.exp(state.m - nextM), b = Math.exp(m - nextM);
    state = { m: nextM, l: a * state.l + b * l, u: a * state.u + b * u };
  }
  const valid = blocks.flat().filter(([s]) => s !== -Infinity);
  const reference = sum(valid.map(([s, v]) => Math.exp(s) * v))
    / sum(valid.map(([s]) => Math.exp(s)));
  near(state.u / state.l, reference);
  assert.ok(Number.isNaN(Math.exp(-Infinity - (-Infinity))));
});

check('CP: zigzag positions cannot use a single rank offset', () => {
  const shards = [[0, 1, 6, 7], [2, 3, 4, 5]];
  assert.deepEqual(shards.flat().sort((a, b) => a - b), [0, 1, 2, 3, 4, 5, 6, 7]);
  assert.notDeepEqual(shards[0].map((_, j) => shards[0][0] + j), shards[0]);
  // Enumerate allowed causal keys, rather than assuming equal token count => equal work.
  const counts = shards.map((qs) => sum(qs.map((q) =>
    Array.from({ length: 8 }, (_, k) => Number(k <= q)).reduce((a, b) => a + b, 0))));
  assert.deepEqual(counts, [18, 18]);
});

check('RLHF: k3 is unbiased under current policy, not arbitrary old samples', () => {
  const p = [0.7, 0.3], q = [0.4, 0.6], old = [0.5, 0.5];
  const k3 = p.map((x, i) => q[i] / x - Math.log(q[i] / x) - 1);
  const kl = sum(p.map((x, i) => x * Math.log(x / q[i])));
  near(sum(p.map((x, i) => x * k3[i])), kl);
  assert.ok(k3.every((x) => x >= 0));
  assert.ok(Math.abs(sum(old.map((x, i) => x * k3[i])) - kl) > 0.01);
  near(sum(old.map((x, i) => x * (p[i] / x) * k3[i])), kl);
});

check('GAE: truncated value bootstrap does not join a new packed episode', () => {
  const reward = 1, value = 3, nextValue = 4, gamma = 0.9, lambda = 0.95;
  const terminalDelta = reward - value;
  const truncatedDelta = reward + gamma * nextValue - value;
  near(terminalDelta, -2);
  near(truncatedDelta, 1.6);
  near(truncatedDelta + gamma * lambda * 0, 1.6);
  assert.notEqual(truncatedDelta + gamma * lambda * 100, truncatedDelta);
});

check('FSDP: two parameter gathers and one gradient scatter use distinct bytes', () => {
  const ranks = 4, elements = 10, paramBytes = elements * 2, gradBytes = elements * 4;
  const phaseInputs = [paramBytes, paramBytes, gradBytes];
  const enumerated = sum(phaseInputs.flatMap((bytes) =>
    Array.from({ length: ranks - 1 }, () => bytes / ranks)));
  near(enumerated, (ranks - 1) / ranks * (2 * paramBytes + gradBytes));
  near(enumerated, 60);
  assert.notEqual(enumerated, 3 * (ranks - 1) / ranks * paramBytes);
});

check('EP: expert assignments, rank-deduplicated rows, and remote rows differ', () => {
  const expertRank = [0, 0, 1, 1], routes = [[0, 1, 2], [1, 2, 3]];
  const sourceRank = 0;
  const destinations = routes.map((route) => [...new Set(route.map((e) => expertRank[e]))]);
  assert.equal(sum(routes.map((r) => r.length)), 6);
  assert.equal(sum(destinations.map((r) => r.length)), 4);
  assert.equal(sum(destinations.map((r) => r.filter((x) => x !== sourceRank).length)), 2);
  const imbalance = (cols) => sum(cols) ? Math.max(...cols) / (sum(cols) / cols.length) : null;
  assert.equal(imbalance([0, 0]), null);
  near(imbalance([2, 2]), 1);
});

check('Checkpoint: async staging must finish before the first mutation', () => {
  // Two values copied at logical times 3 and 8; a simultaneous update changes both.
  const capture = (mutationTime) => [3, 8].map((readTime) => readTime < mutationTime ? 0 : 1);
  assert.deepEqual(capture(6), [0, 1]); // Torn snapshot: never a committed version.
  assert.deepEqual(capture(9), [0, 0]);
  const stagingDuration = 8, readOnlyCompute = 5;
  assert.equal(Math.max(0, stagingDuration - readOnlyCompute), 3);
  // Durations are invented units illustrating overlap, not measured checkpoint latency.
});

check('Startup: resource-aware DAG time differs from sum of stage durations', () => {
  const jobs = [
    ['read0', 10, []], ['read1', 10, ['read0']],
    ['copy0', 4, ['read0']], ['copy1', 4, ['read1', 'copy0']],
    ['runtime', 15, []], ['verify', 3, ['copy1', 'runtime']],
  ];
  const done = new Map();
  for (const [name, duration, deps] of jobs) {
    done.set(name, Math.max(0, ...deps.map((dep) => done.get(dep))) + duration);
  }
  assert.equal(done.get('verify'), 27);
  assert.equal(sum(jobs.map(([, d]) => d)), 46);
});

check('Zero-Bubble: AdamW algebraic rollback has nonsingular denominators', () => {
  const beta1 = 0.9, beta2 = 0.99, lr = 0.1, decay = 0.2, eps = 1e-8, g = 0.4;
  const old = { theta: 1, m: 0.2, v: 0.3, t: 3 };
  const next = { t: old.t + 1, m: beta1 * old.m + (1 - beta1) * g,
    v: beta2 * old.v + (1 - beta2) * g ** 2 };
  const delta = lr * (next.m / (1 - beta1 ** next.t))
    / (Math.sqrt(next.v / (1 - beta2 ** next.t)) + eps);
  next.theta = old.theta * (1 - lr * decay) - delta;
  near((next.theta + delta) / (1 - lr * decay), old.theta);
  near((next.m - (1 - beta1) * g) / beta1, old.m);
  near((next.v - (1 - beta2) * g ** 2) / beta2, old.v);
  // With beta1 = 0, two different previous moments map to the same next moment.
  assert.equal(0 * 0.2 + g, 0 * 9 + g);
  // With lr * decay = 1, the old parameter disappears from the forward update.
  assert.equal(1 * (1 - 1) - delta, 9 * (1 - 1) - delta);
});

check('Serving: first-token and mean-TPOT boundaries', () => {
  const prefillEnd = 100, firstTokenVisible = prefillEnd + 2 + 3;
  assert.equal(firstTokenVisible, 105);
  assert.notEqual(firstTokenVisible, firstTokenVisible + 50); // Next decode is not TTFT.
  const tpot = (times) => times.length < 2 ? null
    : (times.at(-1) - times[0]) / (times.length - 1);
  assert.equal(tpot([105]), null);
  near(tpot([105, 115, 155]), 25);
  assert.deepEqual([115 - 105, 155 - 115], [10, 40]); // Mean hides the tail.
});

check('Topology: bit/s to byte/s must include the factor of eight', () => {
  const rawBitsPerLane = 8, lanes = 4, encoding = 1, protocol = 0.8;
  const payloadBits = rawBitsPerLane * lanes * encoding * protocol;
  const payloadBytes = payloadBits / 8;
  near(payloadBytes, 3.2);
  near(payloadBytes * 8, payloadBits);
});

console.log(`${passed} arithmetic/state examples passed; no GPU performance was measured.`);
