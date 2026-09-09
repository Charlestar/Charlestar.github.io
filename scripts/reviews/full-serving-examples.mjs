import assert from 'node:assert/strict';

// Independent arithmetic checks for the full reread of posts 41–58.
// Run: node scripts/reviews/full-serving-examples.mjs
// These are model-free examples, not GPU/kernel or serving benchmarks.
const sum = (xs) => xs.reduce((a, b) => a + b, 0);
const close = (a, b) => assert.ok(Number.isFinite(a) && Math.abs(a - b) < 1e-10, `${a} != ${b}`);
let passed = 0;
function check(name, fn) {
  fn();
  passed += 1;
  console.log(`PASS ${name}`);
}

check('KV: forward positions vs sampled outputs, including one output', () => {
  for (const L of [1, 4, 512]) {
    for (const T of [1, 2, 17]) {
      const uncached = sum(Array.from({ length: T }, (_, t) => L + t));
      const cached = L + sum(Array(T - 1).fill(1));
      assert.equal(uncached, T * L + T * (T - 1) / 2);
      assert.equal(cached, L + T - 1);
      assert.equal(cached - L, T - 1);
    }
  }
});

check('DistServe: OPT-66B KV exact SI and IEC units', () => {
  const bytes = 2 * 512 * 64 * 72 * 128 * 2;
  assert.equal(bytes, 1207959552);
  close(bytes / 2 ** 30, 1.125);
  close(bytes / 1e9, 1.207959552);
  close(10 * bytes / 1e9, 12.07959552);
  close(10 * bytes * 8 / 1e9, 96.63676416);
});

check('Mooncake/Dynamo: cached-prefix suffix attention and equal endpoints', () => {
  // Count causal query-key pairs, not GPU time. A real profile also includes MLP etc.
  const pairs = (H, U) => H * U + U * (U + 1) / 2;
  const H = 100, U = 10;
  assert.equal(pairs(H, U), 1055);
  assert.equal(pairs(0, U), 55);
  assert.equal(pairs(0, H + U), 6105);
  const load = 5800; // Hypothetical cost expressed in the same toy units.
  assert.ok(load + pairs(0, U) < pairs(0, H + U)); // Incorrectly claims a win.
  assert.ok(load + pairs(H, U) > pairs(0, H + U)); // Complete path loses.
  const queue = 50;
  close((queue + load + pairs(H, U)) - (queue + pairs(0, H + U)),
    load + pairs(H, U) - pairs(0, H + U));
});

check('DSpark Appendix A: retrospective admission biases target sampling', () => {
  const p = [0.7, 0.3], q = [0.5, 0.5], c1 = 0.8;
  const sps = [1, 0.5, 0.45];
  const utilities = (c2) => [sps[0], (1 + c1) * sps[1], (1 + c1 + c1 * c2) * sps[2]];
  const argmax = (xs) => xs.indexOf(Math.max(...xs));
  assert.equal(argmax(utilities(0.9)), 2);
  assert.equal(argmax(utilities(0)), 0);
  close(utilities(0.9)[2], 1.134);
  close(utilities(0)[2], 0.81);
  // A proposal is admitted and accepted; a B proposal triggers fresh p sampling.
  const output = [q[0] * Math.min(1, p[0] / q[0]) + q[1] * p[0], q[1] * p[1]];
  close(output[0], 0.85);
  close(output[1], 0.15);
  close(sum(output), 1);
  assert.notDeepEqual(output, p);
  // Early-stop rejects the first capacity increment without looking at c2.
  assert.ok(utilities(0)[1] <= utilities(0)[0]);
  assert.deepEqual(q.map(() => p).reduce((acc, row, i) =>
    acc.map((v, j) => v + q[i] * row[j]), [0, 0]), p);
});

check('Capacity: output rate, decode work, KV positions and recovery units', () => {
  const osl = [1, 3], lambda = 10;
  assert.equal(lambda * sum(osl) / osl.length, 20);
  assert.equal(lambda * sum(osl.map((n) => Math.max(n - 1, 0))) / osl.length, 10);
  const prompt = 100, generated = 3;
  assert.equal(prompt + generated - 1, 102);
  const recoveryGpuSeconds = [4, 6, 10], recoverySeconds = 5;
  assert.equal(sum(recoveryGpuSeconds) / recoverySeconds, 4);
  const pReplicas = Math.ceil(12 * 1.25 / 4);
  const dReplicas = Math.ceil(12 * 1.2 / 7);
  assert.equal(pReplicas * 2 + dReplicas * 4, 20);
});

check('EP/DeepEP: logical assignments need not equal physical rows', () => {
  const topk = [[0, 1], [0, 1], [2, 3]];
  const owner = [0, 0, 1, 1];
  const logical = sum(topk.map((experts) => experts.length));
  const rows = sum(topk.map((experts) => new Set(experts.map((e) => owner[e])).size));
  assert.equal(logical, 6);
  assert.equal(rows, 3);
  assert.equal(logical, topk.length * 2);
});

check('EPLB: greedy replicas and padded inverse map', () => {
  const loads = [100, 55, 25, 20], counts = [1, 1, 1, 1];
  for (let i = 0; i < 2; i += 1) {
    const costs = loads.map((load, e) => load / counts[e]);
    counts[costs.indexOf(Math.max(...costs))] += 1;
  }
  assert.deepEqual(counts, [2, 2, 1, 1]);
  const inverse = [[0, 4], [1, 5], [2, -1], [3, -1]];
  inverse.forEach((slots, e) => assert.ok(slots.slice(0, counts[e]).every((p) => p >= 0)));
  assert.equal(inverse[2][1], -1); // Unfiltered modulo can select an invalid slot.
  assert.equal((256 + 32) / 32, 9);
  assert.equal((256 + 32) / 144, 2);
  assert.equal(16 * 256 / 8, 512); // Logical-expert target only.
  assert.equal((512 * 8 / 256) / 2, 8); // Two replicas receive 8 rows each on average.
});

check('CUDA Graph: padding denominators and unique-pool accounting', () => {
  close((24 - 19) / 24, 5 / 24);
  close((24 - 19) / 19, 5 / 19);
  assert.notEqual((24 - 19) / 24, (24 - 19) / 19);
  const graphPools = [{ id: 'shared', bytes: 128 }, { id: 'shared', bytes: 128 }];
  const unique = new Map(graphPools.map((pool) => [pool.id, pool.bytes]));
  assert.equal(sum([...unique.values()]) + 32, 160);
});

check('torch.compile: positive-savings call break-even and no-gain boundary', () => {
  const calls = (compile, eager, compiled) => eager > compiled
    ? Math.ceil(compile / (eager - compiled)) : Infinity;
  assert.equal(calls(100, 3, 1), 50);
  assert.equal(calls(101, 3, 1), 51);
  assert.equal(calls(100, 1, 1), Infinity);
  assert.equal(calls(100, 1, 2), Infinity);
});

check('FP8/AWQ: zero groups have positive scales, not 0/0', () => {
  const scale = (xs, maxCode) => {
    assert.ok(xs.length > 0 && xs.every(Number.isFinite));
    const amax = Math.max(...xs.map(Math.abs));
    return amax === 0 ? 1 : amax / maxCode;
  };
  for (const maxCode of [448, 7]) {
    const s = scale([0, 0, 0], maxCode);
    assert.equal(s, 1);
    assert.deepEqual([0, 0, 0].map((x) => x / s), [0, 0, 0]);
  }
});

check('FP8/SmoothQuant: scale factorization is valid only outside K', () => {
  const xq = [1, 2], wq = [3, 4], sx = 0.5, sw = 2;
  close(sum(xq.map((x, k) => (x * sx) * (wq[k] * sw))), sx * sw * sum(xq.map((x, k) => x * wq[k])));
  const kScales = [1, 3];
  const scaledDot = sum(xq.map((x, k) => x * wq[k] * kScales[k]));
  assert.equal(scaledDot, 27);
  assert.notEqual(scaledDot, kScales[0] * sum(xq.map((x, k) => x * wq[k])));
});

check('QServe: overflow example and protected-range rounding bound', () => {
  const roundEven = (x) => {
    const lo = Math.floor(x), frac = x - lo;
    return frac === 0.5 ? (lo % 2 === 0 ? lo : lo + 1) : Math.round(x);
  };
  const s = roundEven((120 - (-113)) / 15), z = roundEven(113 / s);
  assert.equal(s, 16);
  assert.equal(z, 7);
  assert.equal((roundEven(120 / s) + z - z) * s, 128);
  // A protected group's rounded scale is <= 16; nearest rounding adds <= 8.
  for (let groupMin = -119; groupMin <= 119; groupMin += 1) {
    for (let groupMax = groupMin; groupMax <= 119; groupMax += 1) {
      const groupScale = Math.max(1, roundEven((groupMax - groupMin) / 15));
      assert.ok(groupScale <= 16);
      for (const q of [groupMin, groupMax]) {
        const restored = roundEven(q / groupScale) * groupScale;
        assert.ok(restored >= -128 && restored <= 127);
        assert.ok(Math.abs(restored - q) <= groupScale / 2);
      }
    }
  }
  close(2.5 / 128, 0.01953125);
  close(2.5 / 32, 0.078125);
});

console.log(`Validated ${passed} full-serving mathematical examples; no GPU execution claimed.`);
