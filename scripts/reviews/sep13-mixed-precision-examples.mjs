// Independent CPU arithmetic references, not PyTorch, CUDA, or optimizer integration tests.
import assert from 'node:assert/strict';

let passed = 0;
const close = (a, b, eps = 1e-12) => assert.ok(Number.isFinite(a) && Number.isFinite(b) && Math.abs(a - b) <= eps, `${a} != ${b}`);
const check = (name, fn) => { fn(); console.log(`PASS ${name}`); passed += 1; };
const sum = (xs) => xs.reduce((a, b) => a + b, 0);
const tiesEven = (x) => {
  const lo = Math.floor(x), fraction = x - lo;
  return fraction < 0.5 ? lo : fraction > 0.5 ? lo + 1 : lo + lo % 2;
};
// A mathematical round-to-nearest-even binary format model; it does not model GPU FTZ.
function quantize(value, fractionBits, minExponent, maxExponent) {
  if (!Number.isFinite(value) || value === 0) return value;
  const x = Math.abs(value), sign = Math.sign(value);
  const exponent = Math.max(minExponent, Math.floor(Math.log2(x)));
  const quantum = 2 ** (exponent - fractionBits);
  const result = tiesEven(x / quantum) * quantum;
  return sign * (result >= 2 ** (maxExponent + 1) ? Infinity : result);
}
const fp16 = (x) => quantize(x, 10, -14, 15);
const bf16 = (x) => quantize(x, 7, -126, 127);

check('FP16 and BF16 have different range and spacing, including subnormals', () => {
  assert.equal((2 - 2 ** -10) * 2 ** 15, 65504);
  close(2 ** -14, 0.00006103515625);
  close(2 ** -24, 0.000000059604644775390625);
  assert.equal(fp16(65504), 65504);
  assert.equal(fp16(65520), Infinity);
  assert.equal(fp16(2 ** -25), 0); // halfway, ties to zero
  assert.equal(fp16(0.75 * 2 ** -24), 2 ** -24); // below min subnormal need not round to zero
  assert.equal(fp16(2 ** -24), 2 ** -24);
  assert.equal(bf16((2 - 2 ** -7) * 2 ** 127), (2 - 2 ** -7) * 2 ** 127);
  assert.equal(bf16(2 ** -133), 2 ** -133);
  assert.equal(bf16(2 ** -134), 0);
  assert.equal(fp16(1 + 2 ** -10) - 1, 2 ** -10);
  assert.equal(bf16(1 + 2 ** -7) - 1, 2 ** -7);
  assert.equal(bf16(1 + 2 ** -10), 1);
});

check('scaling can preserve a tiny backward value but cannot repair forward infinity', () => {
  const gradient = 2 ** -26, scale = 2 ** 12;
  assert.equal(fp16(gradient), 0);
  assert.equal(gradient * scale, 2 ** -14);
  assert.equal(fp16(gradient * scale) / scale, gradient);
  assert.equal(fp16(70000), Infinity);
  assert.equal(fp16(70000) / scale, Infinity);
});

const clip = (g, limit) => {
  const norm = Math.hypot(...g);
  return g.map((x) => x * Math.min(1, limit / norm));
};
check('unscale then clip uses the intended threshold; reverse order shrinks it by S', () => {
  const gradient = [3, 4], scale = 128, limit = 1;
  const scaled = gradient.map((x) => x * scale);
  const correct = clip(scaled.map((x) => x / scale), limit);
  const incorrect = clip(scaled, limit).map((x) => x / scale);
  correct.forEach((x, i) => close(x, [0.6, 0.8][i]));
  close(Math.hypot(...correct), limit);
  close(Math.hypot(...incorrect), limit / scale);
  incorrect.forEach((x, i) => close(x, [0.0046875, 0.00625][i]));
  assert.deepEqual(clip([0, 0], limit), [0, 0]);
});

check('clipping before accumulation is not clipping the effective-batch gradient', () => {
  assert.deepEqual(clip([3 - 2], 1), [1]);
  assert.equal(clip([3], 1)[0] + clip([-2], 1)[0], 0);
});

check('an accumulation window needs one common scale and one final unscale', () => {
  const gradients = [2, 10], scale = 128;
  close(sum(gradients.map((x) => x * scale)) / scale, 12);
  const wrongMixedScales = (gradients[0] * 128 + gradients[1] * 256) / 256;
  close(wrongMixedScales, 11);
  const wrongEarlyUnscale = (gradients[0] + gradients[1] * scale) / scale;
  assert.notEqual(wrongEarlyUnscale, 12);
  assert.equal(2 * 8 + 4 * 16, 80);
  assert.equal((2 * 8 + 4 * 16) / 16, 5); // article example: should be 2 + 4 = 6
});

check('unequal microbatches require effective-element weighting, not mean of means', () => {
  const counts = [3, 1], meanGradients = [2, 10];
  const correct = sum(meanGradients.map((g, i) => g * counts[i])) / sum(counts);
  close(correct, 4);
  close(sum(meanGradients) / meanGradients.length, 6);
  const scale = 1024;
  close(sum(meanGradients.map((g, i) => scale * g * counts[i] / sum(counts))) / scale, correct);
});

check('DDP average of local sums needs world-size over global valid count', () => {
  const localSums = [6, 10], counts = [3, 1], worldSize = 2;
  const globalMean = sum(localSums) / sum(counts);
  close(sum(localSums.map((x) => x * worldSize / sum(counts))) / worldSize, globalMean);
  close(sum(localSums.map((x, i) => x / counts[i])) / worldSize, 6);
  close(globalMean, 4);
});

check('nonfinite gradients skip a toy update and reduce the scale', () => {
  const toyStep = (weight, gradient, scale) => Number.isFinite(gradient)
    ? { weight: weight - 0.1 * gradient / scale, scale, updated: true }
    : { weight, scale: scale / 2, updated: false };
  assert.deepEqual(toyStep(1, Infinity, 128), { weight: 1, scale: 64, updated: false });
  assert.deepEqual(toyStep(1, NaN, 128), { weight: 1, scale: 64, updated: false });
  close(toyStep(1, 128, 128).weight, 0.9);
  // This deliberately does not simulate GradScaler internals or scheduler behavior.
});

check('low-precision parameter updates can disappear; retained FP32 updates accumulate', () => {
  let lowWeight = 1, fp32Weight = 1;
  const update = 2 ** -14;
  for (let i = 0; i < 16; i += 1) {
    lowWeight = fp16(lowWeight - update);
    fp32Weight = Math.fround(fp32Weight - update);
  }
  assert.equal(lowWeight, 1);
  assert.equal(fp32Weight, 1 - 2 ** -10);
  assert.equal(fp16(fp32Weight), 1 - 2 ** -10);
  const articleWeight = 1.5, articleUpdate = 2 ** -12;
  assert.equal(fp16(articleWeight - articleUpdate), articleWeight);
  assert.equal(fp16(articleWeight - 4 * articleUpdate), articleWeight - 2 ** -10);
  assert.equal(2 + 4 + 2 + 4 + 4, 16); // explicit low-precision copies plus master/moments
  assert.equal(2 + 4 + 4 + 4 + 4, 18); // same layout with FP32 gradients
  assert.equal(4 + 4 + 4 + 4, 16); // FP32 parameters/grads/moments; no extra master
});

console.log(`${passed} mixed-precision reference groups passed; no PyTorch/GPU execution.`);
