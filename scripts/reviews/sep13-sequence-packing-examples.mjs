// Independent CPU attention/loss semantics; no FlashAttention, PyTorch, or GPU execution.
import assert from 'node:assert/strict';

let passed = 0;
const check = (name, fn) => { fn(); console.log(`PASS ${name}`); passed += 1; };
const close = (a, b, eps = 1e-11) => assert.ok(Number.isFinite(a) && Number.isFinite(b) && Math.abs(a - b) <= eps, `${a} != ${b}`);
const sum = (xs) => xs.reduce((a, b) => a + b, 0);
const dot = (a, b) => sum(a.map((x, i) => x * b[i]));
const cumulative = (lengths) => lengths.reduce((xs, x) => [...xs, xs.at(-1) + x], [0]);
function validateCu(boundaries, total, maximum) {
  assert.ok(boundaries.length >= 2 && boundaries[0] === 0);
  assert.ok(boundaries.every((x) => Number.isInteger(x) && x >= 0 && x <= 2 ** 31 - 1));
  const differences = boundaries.slice(1).map((x, i) => x - boundaries[i]);
  assert.ok(differences.every((x) => x > 0));
  assert.equal(boundaries.at(-1), total);
  assert.equal(Math.max(...differences), maximum);
}
function attention(q, k, v, allowed) {
  return q.map((qi, i) => {
    const scores = k.map((kj, j) => allowed(i, j) ? dot(qi, kj) / Math.sqrt(qi.length) : -Infinity);
    const maximum = Math.max(...scores);
    assert.ok(Number.isFinite(maximum), 'test inputs must not contain fully masked query rows');
    const exp = scores.map((s) => Math.exp(s - maximum)), denominator = sum(exp);
    return v[0].map((_, d) => sum(v.map((vj, j) => exp[j] * vj[d])) / denominator);
  });
}
const lengths = [3, 2], cu = cumulative(lengths), segment = [0, 0, 0, 1, 1];
const q = [[1, 0], [0, 1], [1, 1], [-1, 0.5], [0.5, -1]];
const k = [[0.5, 1], [1, -0.5], [-0.5, 0.2], [1, 1], [-1, -0.5]];
const v = [[1, 2], [2, -1], [3, 0.5], [10, 4], [20, -3]];
const blockMask = (i, j) => segment[i] === segment[j] && j <= i;

check('cu_seqlens is an exclusive prefix boundary with total length at the last entry', () => {
  assert.deepEqual(cu, [0, 3, 5]);
  assert.equal(cu.length, lengths.length + 1);
  assert.equal(cu.at(-1), q.length);
  assert.equal(Math.max(...lengths), 3);
  cu.slice(0, -1).forEach((start, i) => assert.equal(cu[i + 1] - start, lengths[i]));
  assert.deepEqual(cumulative([3, 2, 4]), [0, 3, 5, 9]);
  validateCu([0, 3, 5, 9], 9, 4);
  for (const [badCu, total, maximum] of [
    [[1, 3, 5, 9], 9, 4], [[0, 3, 3, 9], 9, 6], [[0, 5, 3, 9], 9, 6],
    [[0, 3, 5, 8], 9, 3], [[0, 3, 5, 9], 9, 9], [[0, 2 ** 31], 2 ** 31, 2 ** 31],
  ]) assert.throws(() => validateCu(badCu, total, maximum));
  // Shape/dtype checks cannot detect a semantically wrong but structurally valid split.
  validateCu([0, 4, 5, 9], 9, 4);
  const ownerAt = (boundaries, i) => boundaries.findIndex((end) => end > i) - 1;
  assert.equal(ownerAt([0, 3, 5, 9], 3), 1);
  assert.equal(ownerAt([0, 4, 5, 9], 3), 0);
});

check('block-diagonal causal attention equals running each sequence independently', () => {
  const packed = attention(q, k, v, blockMask);
  const independent = lengths.flatMap((_, sequence) => {
    const start = cu[sequence], end = cu[sequence + 1];
    return attention(q.slice(start, end), k.slice(start, end), v.slice(start, end), (i, j) => j <= i);
  });
  packed.forEach((row, i) => row.forEach((x, d) => close(x, independent[i][d])));
});

check('one global causal triangle leaks the preceding sample into the next sample', () => {
  const zeros = Array.from({ length: 5 }, () => [0]);
  const values = [[1], [2], [3], [10], [20]];
  const packed = attention(zeros, zeros, values, blockMask);
  const leaking = attention(zeros, zeros, values, (i, j) => j <= i);
  close(packed[3][0], 10);
  close(leaking[3][0], 4);
  const articleValues = [[1], [1], [1], [10], [10]];
  close(attention(zeros, zeros, articleValues, blockMask)[3][0], 10);
  close(attention(zeros, zeros, articleValues, (i, j) => j <= i)[3][0], 3.25);
});

check('block isolation also gives zero cross-sample output sensitivity', () => {
  const modifiedV = v.map((row) => [...row]);
  modifiedV[0][0] += 100;
  const before = attention(q, k, v, blockMask), after = attention(q, k, modifiedV, blockMask);
  after.slice(3).forEach((row, offset) => row.forEach((x, d) => close(x, before[offset + 3][d])));
  const leakingBefore = attention(q, k, v, (i, j) => j <= i);
  const leakingAfter = attention(q, k, modifiedV, (i, j) => j <= i);
  assert.ok(Math.abs(leakingAfter[3][0] - leakingBefore[3][0]) > 1);
});

check('masked attention analytic Q/K/V gradients agree with central finite differences', () => {
  const upstream = [[0.2, -0.1], [0.3, 0.4], [-0.2, 0.7], [0.5, -0.3], [0.1, 0.6]];
  const gradients = [q, k, v].map((matrix) => matrix.map((row) => row.map(() => 0)));
  const [dq, dk, dv] = gradients;
  q.forEach((qi, i) => {
    const scores = k.map((kj, j) => blockMask(i, j) ? dot(qi, kj) / Math.sqrt(qi.length) : -Infinity);
    const maximum = Math.max(...scores), exp = scores.map((s) => Math.exp(s - maximum));
    const probabilities = exp.map((x) => x / sum(exp));
    const dp = v.map((vj) => dot(upstream[i], vj)), average = dot(probabilities, dp);
    k.forEach((kj, j) => {
      const ds = probabilities[j] * (dp[j] - average);
      qi.forEach((x, d) => { dq[i][d] += ds * kj[d] / Math.sqrt(qi.length); dk[j][d] += ds * x / Math.sqrt(qi.length); });
      upstream[i].forEach((x, d) => { dv[j][d] += probabilities[j] * x; });
    });
  });
  const values = [q, k, v], delta = 1e-5;
  const objective = (args) => sum(attention(...args, blockMask).map((row, i) => dot(row, upstream[i])));
  const independentObjective = (args) => sum(lengths.map((_, sequence) => {
    const start = cu[sequence], end = cu[sequence + 1];
    const out = attention(...args.map((matrix) => matrix.slice(start, end)), (i, j) => j <= i);
    return sum(out.map((row, i) => dot(row, upstream[start + i])));
  }));
  values.forEach((matrix, which) => matrix.forEach((row, i) => row.forEach((_, d) => {
    const plus = values.map((m) => m.map((r) => [...r])), minus = values.map((m) => m.map((r) => [...r]));
    plus[which][i][d] += delta;
    minus[which][i][d] -= delta;
    close(gradients[which][i][d], (objective(plus) - objective(minus)) / (2 * delta), 1e-8);
    close(gradients[which][i][d], (independentObjective(plus) - independentObjective(minus)) / (2 * delta), 1e-8);
  })));
});

check('cross-sample next-token labels must be masked at the correct shift convention', () => {
  const ids = [10, 11, 12, 20, 21], ignore = -100;
  const explicitNext = ids.map((_, i) => i + 1 < ids.length && segment[i] === segment[i + 1] ? ids[i + 1] : ignore);
  assert.deepEqual(explicitNext, [11, 12, ignore, 21, ignore]);
  assert.equal(explicitNext.filter((x) => x !== ignore).length, 3);
  // For an internally shifted model, the target labels at each sequence start are ignored.
  const internallyShiftedLabels = [...ids];
  for (const start of cu.slice(0, -1)) internallyShiftedLabels[start] = ignore;
  assert.deepEqual(internallyShiftedLabels, [ignore, 11, 12, ignore, 21]);
  assert.deepEqual(internallyShiftedLabels.slice(1), explicitNext.slice(0, -1));
  const eos = 99, articleIds = [10, 11, eos, 20, eos, 30, 31, 32, eos];
  const articleLabels = [...articleIds];
  for (const start of [0, 3, 5]) articleLabels[start] = ignore;
  assert.deepEqual(articleLabels, [ignore, 11, eos, ignore, eos, ignore, 31, 32, eos]);
  assert.equal(articleLabels.slice(1).filter((x) => x !== ignore).length, 6);
  assert.equal(articleLabels.filter((x) => x === eos).length, 3);
});

check('token mean, sequence mean, and mean of microbatch means differ with unequal lengths', () => {
  const losses = [[1, 3, 5], [9]];
  const tokenMean = sum(losses.flat()) / losses.flat().length;
  const sequenceMean = sum(losses.map((xs) => sum(xs) / xs.length)) / losses.length;
  close(tokenMean, 4.5);
  close(sequenceMean, 6);
  const counts = losses.map((xs) => xs.length), localMeans = losses.map((xs) => sum(xs) / xs.length);
  close(sum(localMeans.map((x, i) => x * counts[i])) / sum(counts), tokenMean);
  close((2 * 1 + 6 * 3) / (2 + 6), 2.5);
  close((1 + 3) / 2, 2);
});

check('DDP mean requires global effective-token denominator including unequal ranks', () => {
  const localSums = [9, 9], validTokens = [3, 1], world = 2;
  const actual = sum(localSums.map((x) => world * x / sum(validTokens))) / world;
  close(actual, 4.5);
  close(sum(localSums.map((x, rank) => x / validTokens[rank])) / world, 6);
  const localZero = [0, 18], countsZero = [0, 4];
  close(sum(localZero.map((x) => world * x / sum(countsZero))) / world, 4.5);
  close((world * 2 / 8 + world * 18 / 8) / world, 2.5);
  close((2 / 2 + 18 / 6) / world, 2);
  const microbatchSums = [[1, 1], [6, 12]];
  close(sum(microbatchSums.map((rank) => sum(rank.map((lossSum) => world * lossSum / 8)))) / world, 2.5);
  // All-zero global counts require skipping/handling the batch, never division by zero.
});

check('RoPE common offsets preserve relative dot products but do not create attention isolation', () => {
  const rotate = ([x, y], angle) => [x * Math.cos(angle) - y * Math.sin(angle), x * Math.sin(angle) + y * Math.cos(angle)];
  const left = [0.3, -0.7], right = [0.9, 0.2], p = 2, r = 5, shift = 11, frequency = 0.4;
  close(dot(rotate(left, p * frequency), rotate(right, r * frequency)), dot(rotate(left, (p + shift) * frequency), rotate(right, (r + shift) * frequency)));
  const resetPositions = [0, 1, 2, 0, 1];
  const rq = q.map((x, i) => rotate(x, resetPositions[i] * frequency));
  const rk = k.map((x, i) => rotate(x, resetPositions[i] * frequency));
  const isolated = attention(rq, rk, v, blockMask), leaking = attention(rq, rk, v, (i, j) => j <= i);
  assert.ok(Math.abs(isolated[3][0] - leaking[3][0]) > 1);
});

check('padding removal has separate tokenwise and attention-pair ideal work ratios', () => {
  const lens = [2, 4, 8], maximum = 8;
  assert.equal(sum(lens), 14);
  assert.equal(lens.length * maximum, 24);
  assert.equal(sum(lens.map((x) => x * x)), 84);
  assert.equal(lens.length * maximum ** 2, 192);
  assert.equal(sum(lens.map((x) => x * (x + 1) / 2)), 49);
  assert.equal(sum(lens) * (sum(lens) + 1) / 2, 105);
  const articleLengths = [3, 2, 4];
  assert.equal(sum(articleLengths), 9);
  assert.equal(articleLengths.length * Math.max(...articleLengths), 12);
  assert.equal(articleLengths.length * Math.max(...articleLengths) ** 2, 48);
  assert.equal(sum(articleLengths.map((x) => x * x)), 29);
  assert.equal(sum(articleLengths) ** 2, 81);
  assert.equal(sum(articleLengths.map((x) => x * (x + 1) / 2)), 19);
  assert.equal(9 * 10 / 2, 45);
  assert.equal(45 - 19, 26);
  // Counts are theoretical work, not end-to-end speedup promises.
});

console.log(`${passed} sequence-packing reference groups passed; no framework/GPU execution.`);
