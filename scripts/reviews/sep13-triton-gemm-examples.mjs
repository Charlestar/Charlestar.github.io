// CPU indexing and counting model; no Triton compilation, GPU execution, or timing.
import assert from 'node:assert/strict';

let passed = 0;
const check = (name, fn) => { fn(); console.log(`PASS ${name}`); passed += 1; };
const close = (a, b, eps = 1e-10) => assert.ok(Number.isFinite(a) && Number.isFinite(b) && Math.abs(a - b) <= eps, `${a} != ${b}`);
function view(rows, columns, stride0, stride1, fill) {
  const storage = Array((rows - 1) * stride0 + (columns - 1) * stride1 + 1).fill(NaN);
  for (let i = 0; i < rows; i += 1) for (let j = 0; j < columns; j += 1) storage[i * stride0 + j * stride1] = fill(i, j);
  return { rows, columns, stride0, stride1, storage };
}
const load = (a, i, j) => {
  assert.ok(i >= 0 && i < a.rows && j >= 0 && j < a.columns);
  return a.storage[i * a.stride0 + j * a.stride1];
};
function reference(a, b) {
  return Array.from({ length: a.rows }, (_, m) => Array.from({ length: b.columns }, (_, n) => {
    let acc = 0;
    for (let k = 0; k < a.columns; k += 1) acc += load(a, m, k) * load(b, k, n);
    return acc;
  }));
}
function tiled(a, b, bm, bn, bk) {
  const M = a.rows, N = b.columns, K = a.columns;
  assert.equal(K, b.rows);
  const result = view(M, N, N + 3, 1, () => NaN), writes = Array(M * N).fill(0);
  const gridM = Math.ceil(M / bm), gridN = Math.ceil(N / bn);
  for (let pid = 0; pid < gridM * gridN; pid += 1) {
    const pidM = Math.floor(pid / gridN), pidN = pid % gridN;
    const acc = Array.from({ length: bm }, () => Array(bn).fill(0));
    for (let k0 = 0; k0 < K; k0 += bk) {
      for (let i = 0; i < bm; i += 1) for (let j = 0; j < bn; j += 1) for (let kk = 0; kk < bk; kk += 1) {
        const m = pidM * bm + i, n = pidN * bn + j, k = k0 + kk;
        const av = m < M && k < K ? load(a, m, k) : 0;
        const bv = k < K && n < N ? load(b, k, n) : 0;
        acc[i][j] += av * bv;
      }
    }
    for (let i = 0; i < bm; i += 1) for (let j = 0; j < bn; j += 1) {
      const m = pidM * bm + i, n = pidN * bn + j;
      if (m < M && n < N) { result.storage[m * result.stride0 + n] = acc[i][j]; writes[m * N + n] += 1; }
    }
  }
  assert.ok(writes.every((n) => n === 1));
  return result;
}

check('M/N/K edge masks and flat program mapping cover each output exactly once', () => {
  for (const [M, N, K] of [[1, 1, 1], [5, 7, 3], [17, 19, 23], [31, 33, 29], [64, 32, 16]]) {
    const a = view(M, K, K, 1, (m, k) => (m + 2 * k) % 7 - 3);
    const b = view(K, N, N, 1, (k, n) => (3 * k - n) % 5);
    const expected = reference(a, b);
    for (const [bm, bn, bk] of [[16, 16, 16], [32, 16, 16], [16, 32, 32]]) {
      const actual = tiled(a, b, bm, bn, bk);
      expected.forEach((row, m) => row.forEach((x, n) => close(load(actual, m, n), x)));
    }
  }
});

check('element strides support transpose views and padded rows without assuming contiguous input', () => {
  const M = 7, N = 9, K = 5;
  for (const transposedA of [false, true]) for (const transposedB of [false, true]) {
    const a = view(M, K, transposedA ? 1 : K + 4, transposedA ? M + 2 : 1, (m, k) => m - k / 2);
    const b = view(K, N, transposedB ? 1 : N + 3, transposedB ? K + 2 : 1, (k, n) => k / 4 + n);
    const expected = reference(a, b), actual = tiled(a, b, 16, 16, 16);
    expected.forEach((row, m) => row.forEach((x, n) => close(load(actual, m, n), x)));
  }
  const xStorage = [10, 11, 12, 20, 21, 22];
  assert.equal(xStorage[1 * 1 + 0 * 3], 11); // transpose B[1,0], stride=(1,3)
  assert.equal(xStorage[1 * 2 + 0 * 1], 12); // incorrectly treating B as contiguous
});

check('masking only stores cannot prevent invalid K loads from poisoning valid outputs', () => {
  const K = 3, bk = 4, a = [1, 2, 3, NaN], b = [4, 5, 6, NaN];
  let masked = 0, missingMask = 0;
  for (let k = 0; k < bk; k += 1) {
    masked += (k < K ? a[k] : 0) * (k < K ? b[k] : 0);
    missingMask += a[k] * b[k];
  }
  assert.equal(masked, 32);
  assert.ok(Number.isNaN(missingMask));
  let moduloK = 0;
  for (let k = 0; k < bk; k += 1) moduloK += a[k % K] * b[k % K];
  assert.equal(moduloK, 36); // wrapping K adds another real contribution, not zero padding
});

check('index promotion must precede stride multiplication, not follow int32 overflow', () => {
  const row = 65536n, stride = 65536n, column = 3n;
  const correctOffset = row * stride + column;
  assert.equal(correctOffset, 4294967299n);
  const wrappedProduct = BigInt.asIntN(32, row * stride);
  assert.equal(wrappedProduct, 0n);
  assert.equal(wrappedProduct + column, 3n);
  assert.notEqual(wrappedProduct + column, correctOffset);
  // A wider type after the multiply retains the wrong value; no huge allocation needed.
  assert.equal(BigInt.asIntN(64, wrappedProduct), 0n);
  assert.equal(BigInt.asIntN(64, row) * stride, 4294967296n);
});

check('article 130x70x65 shape matches direct GEMM including its last 2x6 output tile', () => {
  const M = 130, N = 70, K = 65, bm = 64, bn = 32, bk = 32;
  assert.equal(Math.ceil(M / bm) * Math.ceil(N / bn), 9);
  assert.equal(Math.ceil(K / bk), 3);
  assert.equal((M - 2 * bm) * (N - 2 * bn), 12);
  assert.equal(K - 2 * bk, 1);
  const a = view(M, K, K, 1, (m, k) => (m + k) % 7 - 3);
  const b = view(K, N, N, 1, (k, n) => (k + n) % 5 - 2);
  const expected = reference(a, b), actual = tiled(a, b, bm, bn, bk);
  expected.forEach((row, m) => row.forEach((x, n) => close(load(actual, m, n), x)));
});

check('useful GEMM FLOPs and padded tile arithmetic are different counts', () => {
  const M = 33, N = 35, K = 17, bm = 32, bn = 32, bk = 16;
  const useful = 2 * M * N * K;
  const padded = 2 * Math.ceil(M / bm) * bm * Math.ceil(N / bn) * bn * Math.ceil(K / bk) * bk;
  assert.equal(useful, 39270);
  assert.equal(padded, 262144);
  close(useful / padded, 0.14980316162109375);
  const articleUseful = 2 * 130 * 70 * 65, articlePadded = 2 * (3 * 64) * (3 * 32) * (3 * 32);
  assert.equal(articleUseful, 1183000);
  assert.equal(articlePadded, 3538944);
  close(articleUseful / articlePadded * 100, 33.428, 0.001);
  // Padded arithmetic is a logical tile upper-bound model, not measured instructions.
});

check('tile input reuse intensity ignores cache traffic, output stores, and register spill', () => {
  const bm = 64, bn = 64, bk = 32, elementBytes = 2;
  const flops = 2 * bm * bn * bk;
  const inputBytes = (bm * bk + bk * bn) * elementBytes;
  assert.equal(flops, 262144);
  assert.equal(inputBytes, 8192);
  assert.equal(flops / inputBytes, 32);
  assert.equal(bm * bn, 4096); // logical accumulator elements, not per-thread register count
  assert.equal(4 * 64 * 64 / 1024, 16);
  assert.equal(4 * 128 * 128 / 1024, 64);
  const stageIntensity = (tileM, tileN, tileK) => 2 * tileM * tileN * tileK / (2 * (tileM * tileK + tileK * tileN));
  assert.equal(stageIntensity(128, 128, 32), 64);
  assert.equal(stageIntensity(128, 128, 64), 64);
});

check('FP32 accumulation has rounding and does not guarantee exact real arithmetic', () => {
  const values = [2 ** 24, 1, -(2 ** 24)];
  const serialFP32 = values.reduce((acc, x) => Math.fround(acc + x), 0);
  const realSum = values.reduce((acc, x) => acc + x, 0);
  assert.equal(serialFP32, 0);
  assert.equal(realSum, 1);
});

console.log(`${passed} GEMM reference groups passed; no Triton/PyTorch/GPU execution.`);
