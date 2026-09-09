// Independent scalar/analytic checks; this does not execute PyTorch or a GPU.
import assert from 'node:assert/strict';

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

check('uniform chain proxy has a square-root optimum, not constant memory', () => {
  const layers = 64;
  const candidates = Array.from({ length: layers }, (_, i) => i + 1)
    .map((segment) => ({ segment, units: Math.ceil(layers / segment) + segment }));
  const best = Math.min(...candidates.map(({ units }) => units));
  assert.equal(best, 16);
  assert.equal(candidates.find(({ segment }) => segment === 8).units, best);
  assert.equal(candidates.find(({ segment }) => segment === 1).units, 65);
  assert.equal(candidates.find(({ segment }) => segment === 64).units, 65);
  // This is A*(ceil(L/k)+k), excluding endpoints, gradients and graph lifetimes.
  // It deliberately does not claim an exact framework peak-allocation count.
});

check('article MiB table, asymmetric boundary optimum, and attention matrix', () => {
  const layers = 64, perLayerMiB = 32;
  assert.equal(layers * perLayerMiB, 2048);
  assert.deepEqual([4, 8, 16].map((k) => (layers / k + k) * perLayerMiB), [640, 512, 640]);
  const boundary = 128, internal = 8;
  const optimum = Math.sqrt(layers * boundary / internal);
  close(optimum, 32);
  const cost = (k) => layers / k * boundary + k * internal;
  for (let k = 1; k <= 64; k += 1) assert.ok(cost(k) >= cost(optimum));
  assert.equal(1 * 4096 * 4096 * 2 / 2 ** 20, 32);
  assert.equal(1 * 32 * 4096 ** 2 * 2 / 2 ** 30, 1);
});

const sum = (xs) => xs.reduce((a, b) => a + b, 0);
const transpose = (matrix) => matrix[0].map((_, j) => matrix.map((row) => row[j]));
const matmul = (left, right) => {
  assert.equal(left[0].length, right.length);
  return left.map((row) => transpose(right).map((column) =>
    sum(row.map((x, j) => x * column[j]))));
};
const mapMatrix = (matrix, fn) => matrix.map((row, i) => row.map((x, j) => fn(x, i, j)));

check('all two-layer MLP matrix gradients agree with finite differences', () => {
  const X = [[0.1, 0.3], [-0.2, 0.4]];
  const W1 = [[0.5, -0.2, 0.7], [0.8, -0.1, -0.4]];
  const W2 = [[0.2, -0.5], [0.6, 0.7], [-0.3, 0.9]];
  const G = [[0.4, -0.2], [0.5, 0.3]];
  const forward = (x, w1, w2) => {
    const Z = matmul(x, w1), A = mapMatrix(Z, (z) => Math.tanh(z));
    return { Z, A, Y: matmul(A, w2) };
  };
  const { A } = forward(X, W1, W2);
  const dA = matmul(G, transpose(W2));
  const dZ = mapMatrix(dA, (g, i, j) => g * (1 - A[i][j] ** 2));
  const gradients = [matmul(dZ, transpose(W1)), matmul(transpose(X), dZ), matmul(transpose(A), G)];
  const values = [X, W1, W2];
  const loss = (args) => sum(forward(...args).Y.flatMap((row, i) => row.map((x, j) => x * G[i][j])));
  const delta = 1e-5;
  values.forEach((matrix, which) => matrix.forEach((row, i) => row.forEach((_, j) => {
    const plus = values.map((m) => m.map((r) => [...r]));
    const minus = values.map((m) => m.map((r) => [...r]));
    plus[which][i][j] += delta;
    minus[which][i][j] -= delta;
    close(gradients[which][i][j], (loss(plus) - loss(minus)) / (2 * delta), 1e-8);
  })));
});

const step = (x, weight) => Math.tanh(weight * x + 0.1);
const fullForward = (input, weights) => weights.reduce(
  (xs, weight) => [...xs, step(xs.at(-1), weight)], [input],
);
function fullGradient(input, weights) {
  const xs = fullForward(input, weights), grads = Array(weights.length);
  let upstream = xs.at(-1); // loss = output^2 / 2
  for (let j = weights.length - 1; j >= 0; j -= 1) {
    const preactivationGrad = upstream * (1 - xs[j + 1] ** 2);
    grads[j] = preactivationGrad * xs[j];
    upstream = preactivationGrad * weights[j];
  }
  return { output: xs.at(-1), inputGrad: upstream, grads };
}
function checkpointGradient(input, weights, segment) {
  const boundaries = new Map();
  let output = input;
  weights.forEach((weight, j) => {
    if (j % segment === 0) boundaries.set(j, output);
    output = step(output, weight);
  });
  const grads = Array(weights.length);
  let upstream = output, recomputed = 0;
  const starts = [...boundaries.keys()].reverse();
  for (const start of starts) {
    const end = Math.min(start + segment, weights.length);
    const local = fullForward(boundaries.get(start), weights.slice(start, end));
    recomputed += end - start;
    for (let j = end - 1; j >= start; j -= 1) {
      const offset = j - start;
      const preactivationGrad = upstream * (1 - local[offset + 1] ** 2);
      grads[j] = preactivationGrad * local[offset];
      upstream = preactivationGrad * weights[j];
    }
  }
  return { output, inputGrad: upstream, grads, recomputed };
}

check('pure-function segment replay preserves loss, input and parameter gradients', () => {
  const weights = [0.8, -0.7, 1.1, 0.5, -0.9, 1.2, 0.6], input = 0.3;
  const reference = fullGradient(input, weights);
  for (const segment of [1, 2, 3, 7]) {
    const replay = checkpointGradient(input, weights, segment);
    close(replay.output, reference.output);
    close(replay.inputGrad, reference.inputGrad);
    replay.grads.forEach((g, j) => close(g, reference.grads[j]));
    assert.equal(replay.recomputed, weights.length);
  }
  const loss = (x, ws) => fullForward(x, ws).at(-1) ** 2 / 2;
  const delta = 1e-5;
  close(reference.inputGrad, (loss(input + delta, weights) - loss(input - delta, weights)) / (2 * delta), 1e-8);
  weights.forEach((_, j) => {
    const plus = [...weights], minus = [...weights];
    plus[j] += delta;
    minus[j] -= delta;
    close(reference.grads[j], (loss(input, plus) - loss(input, minus)) / (2 * delta), 1e-8);
  });
});

check('changing the replay dropout mask changes the backward function', () => {
  const input = 1, weight = 1, keepProbability = 0.5, originalMask = 1, wrongReplayMask = 0;
  const output = weight * input * originalMask / keepProbability;
  const upstream = output; // loss = output^2 / 2
  close(output ** 2 / 2, 2);
  close(upstream * input * originalMask / keepProbability, 4);
  close(upstream * input * wrongReplayMask / keepProbability, 0);
});

check('matching shapes do not prove equal values or Jacobians', () => {
  const input = 2, originalWeight = 3, changedWeight = 4;
  const originalOutput = input * originalWeight;
  const trueGrad = originalOutput * originalWeight;
  const invalidReplayGrad = originalOutput * changedWeight;
  close(trueGrad, 18);
  close(invalidReplayGrad, 24);
  assert.equal(typeof originalOutput, typeof (input * changedWeight));
});

check('one extra forward is one-third extra FLOPs only when backward is twice forward', () => {
  const forward = 100, backward = 200, replay = 100;
  close(replay / (forward + backward), 1 / 3);
  close((forward + backward + replay) / (forward + backward), 4 / 3);
  close((forward + backward) / (forward + backward + replay), 3 / 4);
  const layers = 64, segment = 8;
  close((layers - segment) / layers, 7 / 8); // If the final segment is not checkpointed.
});

check('peak savings from two regions are not independently additive', () => {
  const phases = [100, 80]; // Invented bytes at two different instants.
  const peak = (a, b) => Math.max(phases[0] - a, phases[1] - b);
  const baseline = peak(0, 0);
  const savingA = baseline - peak(30, 0), savingB = baseline - peak(0, 30);
  close(savingA, 20);
  close(savingB, 0);
  close(baseline - peak(30, 30), 30);
  assert.notEqual(savingA + savingB, baseline - peak(30, 30));
});

console.log(`${passed} rematerialization groups passed; no PyTorch/GPU performance qualification.`);
