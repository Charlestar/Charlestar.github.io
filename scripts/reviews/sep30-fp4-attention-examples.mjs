// Mathematical CPU references, not Direct-P converter, PTX, TMEM, CUDA, or training tests.
import assert from 'node:assert/strict';

let passed = 0;
const check = (name, fn) => { fn(); passed += 1; console.log(`PASS ${name}`); };
const close = (actual, expected, tolerance = 1e-12) => {
  assert.ok(Number.isFinite(actual) && Math.abs(actual - expected) <= tolerance,
    `${actual} != ${expected}`);
};
const sum = values => values.reduce((a, b) => a + b, 0);
const dot = (a, b) => sum(a.map((x, i) => x * b[i]));
const magnitude = values => Math.hypot(...values);
const normalize = values => values.map(value => value / sum(values));
const e2m1 = [0, 0.5, 1, 1.5, 2, 3, 4, 6];
// Positive, finite, saturating mathematical nearest-even codebook model.
// Even means the low bit of the nonnegative E2M1 code, not an even numeric value.
function quantize(value) {
  assert.ok(Number.isFinite(value) && value >= 0);
  let index = 0;
  for (let i = 1; i < e2m1.length; i += 1) {
    const distance = Math.abs(value - e2m1[i]);
    const oldDistance = Math.abs(value - e2m1[index]);
    if (distance < oldDistance || (distance === oldDistance && i % 2 === 0)) index = i;
  }
  return e2m1[index];
}

check('E2M1 midpoint boundaries and illustrative nearest-even outcomes', () => {
  const boundaries = e2m1.slice(1).map((right, i) => (e2m1[i] + right) / 2);
  assert.deepEqual(boundaries, [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5]);
  assert.deepEqual(boundaries.map(quantize), [0, 1, 1, 2, 2, 4, 4]);
  boundaries.forEach((boundary, i) => {
    assert.equal(quantize(boundary - 1e-6), e2m1[i]);
    assert.equal(quantize(boundary + 1e-6), e2m1[i + 1]);
  });
});

check('affine code classification can agree or disagree with exact exponential rounding', () => {
  const approximate = x => quantize(Math.max(0, 1.5 * x + 1.2));
  assert.equal(quantize(2 ** 0), 1);
  assert.equal(approximate(0), 1);
  assert.equal(quantize(2 ** 1), 2);
  assert.equal(approximate(1), 3);
  const z = -0.7, reference = 0.3, exponent = -2;
  const x = (z - reference) * Math.LOG2E - exponent + Math.log2(6);
  close(2 ** x, Math.exp(z - reference) / (2 ** exponent / 6));
});

check('alpha over six belongs to this representation contract; two operands require over 36', () => {
  const alphaP = 1, alphaV = 2, qP = [6, 2], qV = [3, 0.5];
  const decodedP = qP.map(q => alphaP * q / 6);
  const decodedV = qV.map(q => alphaV * q / 6);
  close(dot(decodedP, decodedV), alphaP * alphaV * dot(qP, qV) / 36);
  assert.notEqual(dot(decodedP, decodedV), alphaP * alphaV * dot(qP, qV) / 6);
  assert.deepEqual([1, 0.4].map(p => quantize(6 * p)), [6, 2]);
});

check('represented numerator and denominator preserve a constant value unlike mismatched normalization', () => {
  const original = [1, 0.4];
  const represented = original.map(p => quantize(6 * p) / 6);
  const values = [2, 2];
  close(dot(represented, values) / sum(represented), 2);
  close(dot(represented, values) / sum(original), 40 / 21);
  // A positive-denominator condition is essential; an all-masked row cannot use this division.
  assert.equal(sum([0, 0]), 0);
  assert.ok(Number.isNaN(dot([0, 0], values) / sum([0, 0])));
});

check('consistent normalization does not restore exact attention values', () => {
  const original = [1, 0.4], represented = [1, 1 / 3], values = [0, 6];
  close(dot(original, values) / sum(original), 12 / 7);
  close(dot(represented, values) / sum(represented), 1.5);
  close(dot(represented, values) / sum(original), 10 / 7);
  assert.notEqual(dot(original, values) / sum(original), dot(represented, values) / sum(represented));
});

check('each block scale matters and independent per-tile softmax changes row weights', () => {
  const codes = [6, 6], amplitudes = [1, 0.25], values = [0, 10];
  const represented = codes.map((code, i) => amplitudes[i] * code / 6);
  close(dot(normalize(represented), values), 2);
  close(dot(normalize(codes), values), 5);
  const weights = [1, 3, 2, 4], rowValues = [1, 2, 10, 20];
  const entireRow = dot(weights, rowValues) / sum(weights);
  const separateTiles = dot(normalize(weights.slice(0, 2)), rowValues.slice(0, 2))
    + dot(normalize(weights.slice(2)), rowValues.slice(2));
  close(entireRow, 10.7);
  assert.notEqual(entireRow, separateTiles);
  assert.notEqual(entireRow, separateTiles / 2);
});

check('evaluation order differs in an explicit mathematical FP32 output-FTZ model', () => {
  const minNormal = 2 ** -126;
  const roundAndFlush = value => {
    const rounded = Math.fround(value);
    return Math.abs(rounded) < minNormal ? 0 : rounded;
  };
  const alpha = minNormal, codeSum = 12;
  const wrongOrder = roundAndFlush(roundAndFlush(alpha / 6) * codeSum);
  const safeForThisCase = roundAndFlush(alpha * roundAndFlush(codeSum / 6));
  assert.equal(wrongOrder, 0);
  assert.equal(safeForThisCase, 2 ** -125);
  // This custom FTZ rule is not a measurement of JavaScript or a GPU's hardware mode.
});

check('paired K and V permutation preserves noncausal attention but not an unchanged causal mask', () => {
  const weights = [0.2, 0.3, 0.5], values = [10, 20, 30], permutation = [2, 0, 1];
  const permutedWeights = permutation.map(i => weights[i]);
  const permutedValues = permutation.map(i => values[i]);
  close(dot(weights, values), dot(permutedWeights, permutedValues));
  assert.notEqual(dot(permutedWeights, values), dot(weights, values));
  // Query at original position zero may attend only original key zero.
  assert.equal(values[0], 10);
  assert.equal(permutedValues[0], 30); // a naive unchanged triangular mask now exposes the wrong key
  assert.equal(permutedValues[permutation.indexOf(0)], values[0]);
});

check('TMEM capacity and consumer granularity do not change when probability payload shrinks', () => {
  assert.equal(128 * 512 * 4, 256 * 1024);
  assert.equal(2 * 128 + 2 * 128, 512);
  const readyFragments = new Set();
  const firstHalfReady = () => readyFragments.has(0) && readyFragments.has(1);
  readyFragments.add(0);
  assert.equal(firstHalfReady(), false);
  readyFragments.add(1);
  assert.equal(firstHalfReady(), true);
  let owner = 'score';
  const canWriteNextQK = () => owner === 'free';
  owner = 'probability';
  assert.equal(canWriteNextQK(), false);
  owner = 'pv-reading';
  assert.equal(canWriteNextQK(), false);
  owner = 'free';
  assert.equal(canWriteNextQK(), true);
});

check('serial Amdahl arithmetic and expanded timing scope limit a local speedup', () => {
  close(0.4 + 0.6 / 4, 0.55);
  close(1 / (0.4 + 0.6 / 4), 20 / 11);
  close(1 / (0.4 + 0.6 / 4 + 0.1), 20 / 13);
  assert.ok(0.501 / 0.356 > 1);
  assert.ok(0.501 / 0.508 < 1);
  close(854.516 / 751.722, 1.136745, 1e-6);
});

check('cosine can hide magnitude errors and training arithmetic is not quality evidence', () => {
  const reference = [1, 2, -3], candidate = reference.map(value => 2 * value);
  close(dot(reference, candidate) / (magnitude(reference) * magnitude(candidate)), 1);
  close(magnitude(candidate.map((value, i) => value - reference[i])) / magnitude(reference), 1);
  assert.equal(64 * 4 * 4, 1024);
  close(2.3948 - 2.3048, 0.09);
});

console.log(`${passed} FP4 attention reference groups passed; no converter, GPU, or training execution.`);
