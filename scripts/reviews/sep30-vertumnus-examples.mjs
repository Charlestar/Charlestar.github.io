// Independent CPU arithmetic and state references, not RTP-LLM or distributed GPU tests.
import assert from 'node:assert/strict';

let passed = 0;
const check = (name, fn) => { fn(); passed += 1; console.log(`PASS ${name}`); };
const close = (actual, expected, tolerance = 1e-10) => {
  assert.ok(Number.isFinite(actual) && Math.abs(actual - expected) <= tolerance,
    `${actual} != ${expected}`);
};
const sum = values => values.reduce((a, b) => a + b, 0);
const pairs = (remaining, prefix) => remaining * prefix + remaining * (remaining + 1) / 2;
const service = (degree, a, b) => a + b / degree;
const cost = ({ degree, minDegree = 1, a, b, queue = 0, beta = 1, lambda = 1 }) =>
  queue + beta * service(degree, a, b)
    + lambda * (degree * service(degree, a, b) - minDegree * service(minDegree, a, b));

check('cached causal prefix removes only its completed triangle, not suffix-to-prefix pairs', () => {
  assert.equal(pairs(8, 0), 36);
  assert.equal(pairs(2, 6), 15);
  assert.notEqual(pairs(2, 6), pairs(2, 0));
  for (let length = 0; length <= 64; length += 1) {
    for (let prefix = 0; prefix <= length; prefix += 1) {
      const enumerated = sum(Array.from({ length: length - prefix }, (_, r) => prefix + r + 1));
      assert.equal(pairs(length - prefix, prefix), enumerated);
      assert.equal(enumerated, (length * (length + 1) - prefix * (prefix + 1)) / 2);
    }
  }
});

check('request latency and GPU-time can move in opposite directions', () => {
  assert.ok(140 < 200);
  assert.equal(2 * 200, 400);
  assert.equal(4 * 140, 560);
  assert.ok(4 * 140 > 2 * 200);
  assert.equal(pairs(0, 8), 0);
  assert.ok(service(4, 2, 0) > 0); // no remaining pairs does not mean zero service or TTFT
});

check('extra GPU-time cancels the parallel component only at the same cache condition', () => {
  for (const a of [0, 0.5, 2, 9]) {
    for (const b of [0, 4, 24, 99]) {
      for (const minDegree of [1, 2, 4]) {
        for (const degree of [minDegree, minDegree * 2, minDegree * 4]) {
          close(degree * service(degree, a, b) - minDegree * service(minDegree, a, b),
            (degree - minDegree) * a);
        }
      }
    }
  }
  const ownPrefixBaseline = service(1, 2, 4);
  const unrelatedColdWorker = service(1, 2, 24);
  assert.equal(4 * service(4, 2, 4) - ownPrefixBaseline, 6);
  assert.equal(4 * service(4, 2, 4) - unrelatedColdWorker, -14);
});

check('cache-dependent remaining work and the resource penalty can reverse degree selection', () => {
  assert.equal(service(1, 2, 24), 26);
  assert.equal(service(4, 2, 24), 8);
  assert.equal(cost({ degree: 1, a: 2, b: 24 }), 26);
  assert.equal(cost({ degree: 4, a: 2, b: 24 }), 14);
  assert.equal(cost({ degree: 1, a: 2, b: 4 }), 6);
  assert.equal(cost({ degree: 4, a: 2, b: 4 }), 9);
  assert.equal(cost({ degree: 4, a: 2, b: 24, lambda: 4 }), 32);
});

check('queue and service weighting explain both hot and cold placement decisions', () => {
  const hot = beta => 20 + beta * 4;
  const cold = beta => beta * 14;
  assert.equal(hot(1), 24);
  assert.equal(cold(1), 14);
  assert.ok(cold(1) < hot(1));
  assert.equal(hot(3), 32);
  assert.equal(cold(3), 42);
  assert.ok(hot(3) < cold(3));
});

check('candidate filtering precedes cost ranking and placement reserves load immediately', () => {
  const workers = [
    { id: 'draining', status: 'draining', fits: true, queue: 0 },
    { id: 'too-small', status: 'ready', fits: false, queue: 0 },
    { id: 'a', status: 'ready', fits: true, queue: 0 },
    { id: 'b', status: 'ready', fits: true, queue: 0 },
  ];
  const place = () => {
    const eligible = workers.filter(w => w.status === 'ready' && w.fits);
    const selected = eligible.reduce((best, w) => w.queue < best.queue ? w : best);
    selected.queue += 4;
    return selected.id;
  };
  assert.equal(place(), 'a');
  assert.equal(place(), 'b');
  assert.equal(workers[0].queue, 0);
});

check('split and merge conserve the GPU budget, not the number of independent workers', () => {
  const configurations = [[4, 4], [4, 2, 2], [2, 2, 2, 2]];
  configurations.forEach(degrees => assert.equal(sum(degrees), 8));
  assert.deepEqual(configurations.map(degrees => degrees.length), [2, 3, 4]);
});

check('a new worker becomes visible only after draining and consistent epoch acknowledgment', () => {
  const ready = state => state.drained && state.metadataEpoch === state.targetEpoch
    && state.rankEpochs.every(epoch => epoch === state.targetEpoch);
  const state = { drained: false, targetEpoch: 8, metadataEpoch: 7, rankEpochs: [7, 7, 7, 7] };
  assert.equal(ready(state), false);
  state.drained = true;
  state.rankEpochs = [8, 8, 8, 7];
  state.metadataEpoch = 8;
  assert.equal(ready(state), false);
  state.rankEpochs[3] = 8;
  assert.equal(ready(state), true);
  // This is only a publication invariant. It does not implement collectives or engine steps.
});

check('logical presence and resident source blocks do not imply a ready destination prefix', () => {
  const required = ['a', 'b', 'c'];
  const destination = new Set(['a']);
  const source = new Set(required);
  const visible = () => required.every(block => destination.has(block));
  assert.equal(visible(), false);
  assert.ok(required.every(block => source.has(block)));
  destination.add('b');
  assert.equal(visible(), false);
  destination.add('c');
  assert.equal(visible(), true);
  const services = [26, 18, 9];
  const marginalBenefits = [services[0] - services[1], services[1] - services[2]];
  assert.equal(sum(marginalBenefits), services[0] - services[2]);
  assert.notEqual((services[0] - services[1]) + (services[0] - services[2]), sum(marginalBenefits));
});

check('input-token-weighted attainment differs from request attainment', () => {
  const lengths = [100, 900], successes = [true, false];
  close(successes.filter(Boolean).length / successes.length, 0.5);
  close(sum(lengths.map((length, i) => successes[i] ? length : 0)) / sum(lengths), 0.1);
});

console.log(`${passed} Vertumnus reference groups passed; no RTP-LLM, collective, or GPU execution.`);
