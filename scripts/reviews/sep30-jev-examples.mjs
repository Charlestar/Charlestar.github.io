import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';

// Offline arithmetic and application-policy examples only. No Jev API or SDK.
const postUrl = new URL('../../_posts/2026-09-30-jev-structured-decision-model.md', import.meta.url);
const article = readFileSync(fileURLToPath(postUrl), 'utf8');
let passed = 0;
const sum = (values) => values.reduce((total, value) => total + value, 0);
const mean = (values) => sum(values) / values.length;
function close(actual, expected, tolerance = 1e-12) {
  assert.ok(Number.isFinite(actual));
  assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} != ${expected}`);
}
function check(name, run) {
  run();
  passed += 1;
  console.log(`PASS ${name}`);
}
function codeBlocks(language) {
  return [...article.matchAll(new RegExp('```' + language + '\\r?\\n([\\s\\S]*?)```', 'g'))]
    .map((match) => match[1]);
}

const jsonBlocks = codeBlocks('json');
assert.equal(jsonBlocks.length, 1);
const request = JSON.parse(jsonBlocks[0]);

check('the actual article JSON parses and defines three typed questions', () => {
  assert.deepEqual(Object.keys(request).sort(), ['model', 'questions', 'state']);
  assert.equal(request.model, 'jev-1.13.0');
  assert.equal(typeof request.state.ticket, 'string');
  assert.deepEqual(Object.keys(request.questions).sort(), ['impact', 'owner', 'workaround']);
  for (const question of Object.values(request.questions)) {
    assert.ok(question.instructions.length > 0);
    assert.ok(['choice', 'score', 'noul'].includes(question.type));
  }
  assert.equal(request.questions.owner.type, 'choice');
  assert.ok(Object.keys(request.questions.owner.criteria).length <= 255);
  assert.ok(Object.hasOwn(request.questions.owner.criteria, 'other'));
  assert.equal(request.questions.impact.type, 'score');
  assert.ok(request.questions.impact.criteria.length >= 2);
  assert.ok(request.questions.impact.criteria.length <= 10);
  assert.equal(request.questions.workaround.type, 'noul');
  // This checks the illustrated request shape, not the complete official schema.
});

const javascriptBlocks = codeBlocks('javascript');
assert.equal(javascriptBlocks.length, 1);
const logged = [];
const proposePriority = vm.runInNewContext(
  `${javascriptBlocks[0]}\nproposePriority;`,
  { console: { log: (...values) => logged.push(values.join(' ')) } },
  { timeout: 1000, contextCodeGeneration: { strings: false, wasm: false } },
);

check('the actual article JavaScript executes its three printed examples', () => {
  assert.equal(typeof proposePriority, 'function');
  assert.deepEqual(logged, ['normal', 'review', 'elevated']);
});

function expectedLevel(probabilities) {
  assert.ok(probabilities.every((p) => Number.isFinite(p) && p >= 0 && p <= 1));
  close(sum(probabilities), 1);
  return sum(probabilities.map((p, level) => level * p));
}
// Extract these arrays from the prose rather than keeping a separate fixture.
const inlineArrays = [...article.matchAll(/`(\[[\d., ]+\])`/g)].map((match) => JSON.parse(match[1]));
check('the printed Score distribution has expected index 1.65', () => {
  const distribution = inlineArrays.find((values) => values[0] === 0.05);
  assert.ok(distribution);
  close(expectedLevel(distribution), 1.65);
  assert.ok(expectedLevel(distribution) <= request.questions.impact.criteria.length - 1);
});

check('equal Score means do not identify the underlying distribution', () => {
  const middle = inlineArrays.find((values) => values.length === 3 && values[1] === 1);
  const split = inlineArrays.find((values) => values.length === 3 && values[0] === 0.5);
  assert.ok(middle && split);
  close(expectedLevel(middle), 1);
  close(expectedLevel(split), 1);
  const variance = (values) => sum(values.map((p, level) => p * (level - expectedLevel(values)) ** 2));
  close(variance(middle), 0);
  close(variance(split), 1);
  assert.notEqual(middle.at(-1), split.at(-1));
});

check('classification accuracy and a single calibration-bin gap differ', () => {
  const predictions = Array(100).fill(0.8);
  const observed80 = Array.from({ length: 100 }, (_, index) => Number(index < 80));
  const observed100 = Array(100).fill(1);
  const accuracy = (labels) => mean(labels.map((label, index) => Number(Number(predictions[index] > 0.5) === label)));
  close(accuracy(observed80), 0.8);
  close(accuracy(observed100), 1);
  close(Math.abs(mean(predictions) - mean(observed80)), 0);
  close(Math.abs(mean(predictions) - mean(observed100)), 0.2);
  // These are artificial finite samples, not a statistical calibration proof.
});

check('pooled frequency can hide equal and opposite subgroup errors', () => {
  const groupA = Array.from({ length: 10 }, (_, index) => Number(index < 9));
  const groupB = Array.from({ length: 10 }, (_, index) => Number(index < 1));
  close(mean([...groupA, ...groupB]), 0.5);
  close(Math.abs(mean(groupA) - 0.5), 0.4);
  close(Math.abs(mean(groupB) - 0.5), 0.4);
});

check('two-action Bayes threshold follows asymmetric error costs', () => {
  const falsePositiveCost = 1;
  const falseNegativeCost = 20;
  const threshold = falsePositiveCost / (falsePositiveCost + falseNegativeCost);
  close(threshold, 1 / 21);
  close(threshold * falseNegativeCost, (1 - threshold) * falsePositiveCost);
  assert.ok(0.01 * falseNegativeCost < (1 - 0.01) * falsePositiveCost);
  assert.ok(0.1 * falseNegativeCost > (1 - 0.1) * falsePositiveCost);
});

check('review interval includes both exact ties and excludes adjacent values', () => {
  const cases = [
    [0, 'normal'], [0.02 - 1e-9, 'normal'], [0.02, 'review'],
    [0.02 + 1e-9, 'review'], [1 / 21, 'review'], [0.5, 'review'],
    [0.6 - 1e-9, 'review'], [0.6, 'review'], [0.6 + 1e-9, 'elevated'],
    [1, 'elevated'],
  ];
  for (const [probability, action] of cases) assert.equal(proposePriority(probability).action, action);
});

check('the extracted policy minimizes its stated risk on a 10001-point grid', () => {
  for (let index = 0; index <= 10000; index += 1) {
    const p = index / 10000;
    const result = proposePriority(p);
    close(result.risk.normal, p * 20);
    close(result.risk.elevated, 1 - p);
    close(result.risk.review, 0.4);
    close(result.risk[result.action], Math.min(...Object.values(result.risk)));
  }
});

check('invalid or missing probabilities fail closed to review', () => {
  const invalid = [undefined, null, NaN, Infinity, -Infinity, -0.01, 1.01, '0.8', '', true, false, {}, [], [0.8]];
  for (const value of invalid) {
    const result = proposePriority(value);
    assert.equal(result.action, 'review');
    assert.equal(result.reason, 'invalid_probability');
  }
});

check('mock transport failures and non-Noul outputs are not negative evidence', () => {
  // A local boundary stub, not an HTTP client, SDK adapter, or timeout test.
  // Only the requested Noul can supply p; confidence/Score may not substitute.
  const fromOutcome = (outcome) => proposePriority(
    outcome.status === 'ok' && outcome.answer?.type === 'noul' ? outcome.answer.noul : undefined,
  );
  assert.equal(fromOutcome({ status: 'ok', answer: { type: 'noul', noul: 0.8 } }).action, 'elevated');
  assert.equal(fromOutcome({ status: 'timeout', answer: { type: 'noul', noul: 0 } }).action, 'review');
  assert.equal(fromOutcome({ status: 'error' }).action, 'review');
  assert.equal(fromOutcome({ status: 'ok' }).action, 'review');
  assert.equal(fromOutcome({ status: 'ok', answer: { type: 'choice', confidence: 1 } }).action, 'review');
  assert.equal(fromOutcome({ status: 'ok', answer: { type: 'score', score: 0.8 } }).action, 'review');
  assert.equal(fromOutcome({ status: 'ok', answer: { type: 'noul', confidence: 1 } }).action, 'review');
});

check('equal marginal probabilities do not imply independent events', () => {
  const joint = (outcomes) => sum(outcomes.filter(([a, b]) => a && b).map(([, , weight]) => weight));
  const examples = [
    { outcomes: [[true, true, 0.5], [false, false, 0.5]], expected: 0.5 },
    { outcomes: [[true, false, 0.5], [false, true, 0.5]], expected: 0 },
    { outcomes: [[true, true, 0.25], [true, false, 0.25], [false, true, 0.25], [false, false, 0.25]], expected: 0.25 },
  ];
  for (const { outcomes, expected } of examples) {
    close(sum(outcomes.filter(([a]) => a).map(([, , weight]) => weight)), 0.5);
    close(sum(outcomes.filter(([, b]) => b).map(([, , weight]) => weight)), 0.5);
    close(joint(outcomes), expected);
  }
});

console.log(`${passed} Jev offline reference groups passed; no API, SDK, model inference, or network execution.`);
