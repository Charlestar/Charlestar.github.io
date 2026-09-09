import assert from 'node:assert/strict';

let checks = 0;
const check = (name, fn) => {
  fn();
  checks++;
  console.log(`PASS ${name}`);
};
const close = (a, b) => assert.ok(Math.abs(a - b) < 1e-12, `${a} != ${b}`);

check('DFlash: independent draws factorize despite shared deterministic features', () => {
  const q1 = [0.7, 0.3];
  const q2 = [0.2, 0.8];
  const joint = q1.map(a => q2.map(b => a * b));
  close(joint.flat().reduce((a, b) => a + b, 0), 1);
  joint.forEach((row, i) => close(row.reduce((a, b) => a + b, 0), q1[i]));
  q2.forEach((mass, j) => close(joint[0][j] + joint[1][j], mass));
  close(joint[0][0] / q1[0], q2[0]);
  close(joint[1][0] / q1[1], q2[0]);
});

check('DFlash: candidate slots differ from target inputs with an anchor', () => {
  const candidates = [1, 3, 5];
  const ragged = candidates.reduce((a, b) => a + b, 0);
  const padded = candidates.length * Math.max(...candidates);
  assert.equal(ragged, 9);
  assert.equal(padded, 15);
  assert.equal(candidates.reduce((a, b) => a + b + 1, 0), 12);
  assert.equal(candidates.length * (1 + Math.max(...candidates)), 18);
  assert.equal(ragged + candidates.length, 12);
});

check('A/F: sampled output count and materialized input count differ', () => {
  const prompt = 100;
  const generated = 4;
  const beforeNextForward = prompt + generated - 1;
  const afterNextForward = beforeNextForward + 1;
  assert.equal(beforeNextForward, 103);
  assert.equal(afterNextForward, 104);
});

check('A/F: per-instance normalization is not per-GPU normalization', () => {
  const r = 2, batch = 10, cycle = 5, gpuA = 1, gpuF = 4;
  const workRate = r * batch / cycle;
  const perInstance = workRate / (r + 1);
  const perGPU = workRate / (r * gpuA + gpuF);
  close(perInstance, 2 * perGPU);
  const layers = 4;
  // If the same pool repeats homogeneous layer work, one output needs four units.
  const outputRate = workRate / layers;
  close(outputRate * layers, workRate);
  assert.notEqual(outputRate, workRate);
});

check('ATLAS: program scalar carries observed service into later calls', () => {
  let longestObserved = 4;
  const inherited = longestObserved;
  const finish = service => { longestObserved = Math.max(longestObserved, inherited + service); };
  finish(2);
  assert.equal(longestObserved, 6);
  finish(5);
  assert.equal(longestObserved, 9);
  // This is a scalar state example, not an implementation or optimality proof.
  const nextCallPriority = longestObserved;
  assert.equal(nextCallPriority, 9);
});

check('Multimodal: patches, merged embeddings and prompt spans are distinct', () => {
  const grid = 224 / 14;
  const patches = grid * grid;
  const visualEmbeddings = patches / 4;
  assert.equal(patches, 256);
  assert.equal(visualEmbeddings, 64);
  assert.equal(visualEmbeddings + 2, 66);
  const promptIncludingOnePlaceholder = 77;
  const retainedTextAndMarkers = promptIncludingOnePlaceholder - 1;
  assert.equal(retainedTextAndMarkers + visualEmbeddings, 140);
  assert.notEqual(promptIncludingOnePlaceholder + visualEmbeddings, 140);
});

console.log(`${checks} program/multimodal examples passed; no model or GPU execution claimed.`);
