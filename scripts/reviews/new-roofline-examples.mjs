// Standard-library arithmetic checks for Roofline reasoning, not GPU measurements.
import assert from 'node:assert/strict';

const sum = (xs) => xs.reduce((a, b) => a + b, 0);
const close = (a, b, eps = 1e-10) => assert.ok(
  Number.isFinite(a) && Number.isFinite(b) && Math.abs(a - b) <= eps * Math.max(1, Math.abs(b)),
  `${a} != ${b}`,
);
let passed = 0;
function check(name, fn) {
  fn();
  passed += 1;
  console.log(`PASS ${name}`);
}
const lowerTime = ({ flops, bytes, compute, bandwidth }) =>
  Math.max(flops / compute, bytes / bandwidth);

check('ridge dimensions and FLOP/s upper bound equal a time lower bound', () => {
  const compute = 100e12, bandwidth = 2e12, flops = 1e12, bytes = 40e9;
  const intensity = flops / bytes;
  close(compute / bandwidth, 50);
  close(intensity, 25);
  close(Math.min(compute, bandwidth * intensity), 50e12);
  close(lowerTime({ flops, bytes, compute, bandwidth }), 0.02);
});

check('FP32 SAXPY reads x/y and writes y; FMA counts as two FLOPs', () => {
  const n = 2 ** 20, flops = 2 * n, bytes = (2 + 1) * n * 4;
  close(flops / bytes, 1 / 6);
  assert.equal(bytes, 12 * 2 ** 20);
});

check('cold-cache GEMM and GEMV model includes output writes and optional old-output reads', () => {
  const m = 4096, n = 4096, k = 4096, elementBytes = 2;
  const flops = 2 * m * n * k, modelBytes = elementBytes * (m * k + k * n + m * n);
  assert.equal(flops, 137438953472);
  assert.equal(modelBytes, 100663296);
  close(flops / modelBytes, 4096 / 3);
  const betaNonzeroBytes = modelBytes + elementBytes * m * n;
  assert.equal(betaNonzeroBytes, 134217728);
  const vectorFlops = 2 * n * k;
  const vectorModelBytes = elementBytes * (n * k + k + n);
  close(vectorFlops / vectorModelBytes, 4096 / 4098);
  assert.ok(vectorFlops / vectorModelBytes < 1);
});

check('article GEMM batch sweep on hypothetical 120 TFLOP/s and 2 TB/s roofs', () => {
  const compute = 120e12, bandwidth = 2e12, k = 4096, n = 4096, elementBytes = 2;
  close(compute / bandwidth, 60);
  const expected = [
    { m: 1, flops: 33554432, bytes: 33570816, intensity: 4096 / 4098, microseconds: 16.785408,
      roundedIntensity: '0.9995', roundedComputeUs: '0.280', roundedMemoryUs: '16.785' },
    { m: 32, flops: 1073741824, bytes: 34078720, intensity: 2048 / 65, microseconds: 17.03936,
      roundedIntensity: '31.5077', roundedComputeUs: '8.948', roundedMemoryUs: '17.039' },
    { m: 128, flops: 4294967296, bytes: 35651584, intensity: 2048 / 17, microseconds: 35.79139413333333,
      roundedIntensity: '120.4706', roundedComputeUs: '35.791', roundedMemoryUs: '17.826' },
  ];
  for (const row of expected) {
    const flops = 2 * row.m * k * n;
    const bytes = elementBytes * (row.m * k + k * n + row.m * n);
    assert.equal(flops, row.flops);
    assert.equal(bytes, row.bytes);
    close(flops / bytes, row.intensity);
    assert.equal((flops / bytes).toFixed(4), row.roundedIntensity);
    assert.equal((flops / compute * 1e6).toFixed(3), row.roundedComputeUs);
    assert.equal((bytes / bandwidth * 1e6).toFixed(3), row.roundedMemoryUs);
    close(lowerTime({ flops, bytes, compute, bandwidth }) * 1e6, row.microseconds);
  }
});

check('serial heterogeneous stages cannot hide each other with aggregate arithmetic intensity', () => {
  const stages = [
    { flops: 90, bytes: 10, compute: 100, bandwidth: 100 },
    { flops: 10, bytes: 90, compute: 100, bandwidth: 100 },
  ];
  const serial = sum(stages.map(lowerTime));
  const aggregate = lowerTime({ flops: 100, bytes: 100, compute: 100, bandwidth: 100 });
  close(serial, 1.8);
  close(aggregate, 1);
  assert.ok(serial > aggregate);
  close(100 / serial, 500 / 9);
});

check('article serial two-stage lower bound is 200 us, not aggregate 101 us', () => {
  const compute = 120e12, bandwidth = 2e12;
  const stages = [
    { flops: 0.12e9, bytes: 200e6, compute, bandwidth },
    { flops: 12e9, bytes: 2e6, compute, bandwidth },
  ];
  stages.forEach((stage) => close(lowerTime(stage) * 1e6, 100));
  close(sum(stages.map(lowerTime)) * 1e6, 200);
  const aggregated = lowerTime({ flops: sum(stages.map((s) => s.flops)),
    bytes: sum(stages.map((s) => s.bytes)), compute, bandwidth });
  close(aggregated * 1e6, 101);
  assert.ok(aggregated < sum(stages.map(lowerTime)));
});

check('hierarchical rooflines pair each traffic level with its own bandwidth', () => {
  const flops = 1000, hbmBytes = 100, l2Bytes = 500;
  const compute = 500, hbmBandwidth = 20, l2Bandwidth = 40;
  const hbmCeiling = hbmBandwidth * flops / hbmBytes;
  const l2Ceiling = l2Bandwidth * flops / l2Bytes;
  close(hbmCeiling, 200);
  close(l2Ceiling, 80);
  close(Math.min(compute, hbmCeiling, l2Ceiling), 80);
  close(Math.max(flops / compute, hbmBytes / hbmBandwidth, l2Bytes / l2Bandwidth), 12.5);
});

check('less work can lower time even while FLOP/s falls, matching the article example', () => {
  const before = { flops: 100, seconds: 10 }, after = { flops: 40, seconds: 5 };
  close(before.flops / before.seconds, 10);
  close(after.flops / after.seconds, 8);
  assert.ok(after.seconds < before.seconds);
  assert.ok(after.flops / after.seconds < before.flops / before.seconds);
  close(before.seconds / after.seconds, 2);
});

console.log(`${passed} Roofline arithmetic groups passed; no GPU performance was measured.`);
