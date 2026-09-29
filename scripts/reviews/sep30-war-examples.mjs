import assert from 'node:assert/strict';

// Deterministic CPU reference examples, not a WAR or veRL implementation.
let passed = 0;
const sum = (values) => values.reduce((total, value) => total + value, 0);
function close(actual, expected, tolerance = 1e-12) {
  assert.ok(Number.isFinite(actual));
  assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} != ${expected}`);
}
function closeVector(actual, expected) {
  assert.equal(actual.length, expected.length);
  actual.forEach((value, index) => close(value, expected[index]));
}
function check(name, run) {
  run();
  passed += 1;
  console.log(`PASS ${name}`);
}

check('synchronous rollout waits for the last trajectory', () => {
  const completionTimes = [2, 2, 2, 10];
  close(sum(completionTimes) / completionTimes.length, 4);
  close(Math.max(...completionTimes), 10);
  const complete = new Set([0, 1, 2]);
  assert.equal(complete.size === completionTimes.length, false);
  complete.add(3);
  assert.equal(complete.size === completionTimes.length, true);
});

check('global routing and local speculation are independent decisions', () => {
  // Illustrative thresholds only; these are not extracted WAR settings.
  const routeMode = (outstanding) => outstanding >= 32 ? 'cache-aware' : 'sticky';
  const decodeMode = (localBatch) => localBatch < 8 ? 'speculative' : 'ordinary';
  assert.equal(routeMode(64), 'cache-aware');
  assert.deepEqual([2, 20].map(decodeMode), ['speculative', 'ordinary']);
  assert.equal(routeMode(2), 'sticky');
  assert.equal(decodeMode(20), 'ordinary');
});

check('suffix pattern proposals do not establish prefix KV compatibility', () => {
  const history = [10, 20, 30, 40, 50];
  const current = [99, 20, 30];
  const pattern = current.slice(-2);
  const start = history.findIndex((_, index) => pattern.every((token, offset) => history[index + offset] === token));
  assert.equal(start, 1);
  assert.deepEqual(history.slice(start + pattern.length), [40, 50]);
  assert.notDeepEqual(history.slice(0, 3), current);
  // Minimal compatibility model; real engines also track layout, adapters, etc.
  const kvCompatible = (cached, request) => cached.policy === request.policy
    && cached.tokenizer === request.tokenizer
    && cached.prefix.every((token, index) => token === request.prefix[index]);
  const cached = { policy: 7, tokenizer: 'ids-v1', prefix: [10, 20, 30] };
  assert.equal(kvCompatible(cached, { ...cached, prefix: current }), false);
  assert.equal(kvCompatible(cached, { ...cached, policy: 8 }), false);
  assert.equal(kvCompatible(cached, { ...cached, prefix: [10, 20, 30, 60] }), true);
});

function speculativeMass(p, q) {
  assert.equal(p.length, q.length);
  for (const distribution of [p, q]) {
    close(sum(distribution), 1);
    assert.ok(distribution.every((probability) => probability >= 0));
  }
  const accepted = p.map((probability, index) => Math.min(probability, q[index]));
  const residual = p.map((probability, index) => Math.max(probability - q[index], 0));
  const rejected = Math.max(0, 1 - sum(accepted));
  const residualSum = sum(residual);
  if (residualSum < 1e-14) return { accepted, rejected, residualDistribution: null, output: accepted };
  const residualDistribution = residual.map((probability) => probability / residualSum);
  const output = accepted.map((probability, index) => probability + rejected * residualDistribution[index]);
  return { accepted, rejected, residualDistribution, output };
}

check('linear rejection sampling preserves target mass, resampling p does not', () => {
  const p = [0.6, 0.3, 0.1];
  const result = speculativeMass(p, [0.2, 0.5, 0.3]);
  closeVector(result.accepted, [0.2, 0.3, 0.1]);
  close(result.rejected, 0.4);
  closeVector(result.residualDistribution, [1, 0, 0]);
  closeVector(result.output, p);
  const wrong = result.accepted.map((probability, index) => probability + result.rejected * p[index]);
  closeVector(wrong, [0.44, 0.42, 0.14]);
  assert.ok(wrong.some((probability, index) => Math.abs(probability - p[index]) > 0.01));
});

check('equal, zero-support and 4356 grid-distribution pairs preserve mass', () => {
  const same = speculativeMass([0.6, 0.4, 0], [0.6, 0.4, 0]);
  close(same.rejected, 0);
  assert.equal(same.residualDistribution, null);
  closeVector(same.output, [0.6, 0.4, 0]);
  const disjoint = speculativeMass([1, 0, 0], [0, 1, 0]);
  close(disjoint.rejected, 1);
  closeVector(disjoint.output, [1, 0, 0]);
  const distributions = [];
  for (let a = 0; a <= 10; a += 1) {
    for (let b = 0; b <= 10 - a; b += 1) distributions.push([a / 10, b / 10, (10 - a - b) / 10]);
  }
  assert.equal(distributions.length ** 2, 4356);
  for (const p of distributions) {
    for (const q of distributions) closeVector(speculativeMass(p, q).output, p);
  }
});

check('fixed-batch throughput can fall despite accepting more tokens', () => {
  const speedup = (batch, tau, ordinaryMs, speculativeMs) => (batch * tau / speculativeMs) / (batch / ordinaryMs);
  close(speedup(2, 3, 4, 6), 2);
  close(speedup(32, 6, 6, 42), 6 / 7);
  assert.ok(speedup(32, 6, 6, 42) < 1);
  close(sum([1, 3, 5]) / 6, 1.5);
});

check('cache hit ratio is not missing-token cost and EMA is not a time oracle', () => {
  close(2000 * (1 - 0.9), 200);
  close(20000 * (1 - 0.7), 6000);
  assert.ok(20000 * 0.7 > 2000 * 0.9);
  const ema = (previous, observation, alpha) => alpha * observation + (1 - alpha) * previous;
  close(ema(10, 6, 0.25), 9);
  close(ema(10, 6, 1), 6);
});

check('the printed inflight clamp has idle and fractional boundary cases', () => {
  // Counterexamples to treating the printed equation as a complete controller.
  const printedBound = (estimate, inflight, queued, replicas) => Math.min(Math.max(1, estimate), inflight + queued / replicas);
  close(printedBound(8, 0, 0, 4), 0);
  close(printedBound(8, 0, 1, 4), 0.25);
  close(printedBound(8, 2, 8, 4), 4);
});

check('illustrative tree branches and external tool side effects stay isolated', () => {
  const parent = [null, 0, 0, 1]; // root, branch A, branch B, child of A.
  const visible = (query, key) => {
    for (let node = query; node !== null; node = parent[node]) {
      if (node === key) return true;
    }
    return false;
  };
  assert.equal(visible(3, 1), true);
  assert.equal(visible(3, 2), false);
  assert.equal(visible(1, 2), false);
  assert.equal(visible(2, 0), true);
  let sideEffects = 0;
  const executeTool = (committed) => { if (committed) sideEffects += 1; };
  executeTool(false);
  assert.equal(sideEffects, 0);
  executeTool(true);
  assert.equal(sideEffects, 1);
});

check('less work after TTL truncation is not same-task acceleration', () => {
  // Normalize the reported relative changes; not an empirical rerun.
  const baseline = { time: 100, tokens: 100, turns: 100, complete: true };
  const truncated = { time: 53.1, tokens: 58.7, turns: 61.1, complete: false };
  const change = (key) => 100 * (truncated[key] / baseline[key] - 1);
  close(change('time'), -46.9);
  close(change('tokens'), -41.3);
  close(change('turns'), -38.9);
  assert.ok(truncated.time < baseline.time);
  assert.equal(truncated.complete, false);
});

check('rollout speedup is not full training iteration speedup', () => {
  const before = 60 + 40;
  const after = 60 / 1.5 + 40;
  close(after, 80);
  close(before / after, 1.25);
  assert.ok(before / after < 1.5);
});

console.log(`${passed} WAR reference groups passed; no veRL, tree verifier, sandbox, or GPU execution.`);
