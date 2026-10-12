import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';

const article = readFileSync(new URL('../../_posts/2026-10-13-dpo-preference-optimization.md', import.meta.url), 'utf8');
function markedBlock(name) {
  const parts = article.split(`<!-- dpo-example: ${name} -->`);
  assert.equal(parts.length, 2, `Missing or duplicate marker ${name}`);
  const match = parts[1].match(/^\s*```python\r?\n([\s\S]*?)```/);
  assert.ok(match, `Missing Python block after ${name}`);
  return match[1];
}
const payload = {
  pair: markedBlock('loss-and-gradient'),
  sequence: markedBlock('sequence-logprob'),
  allPython: [...article.matchAll(/```python\r?\n([\s\S]*?)```/g)].map(match => match[1]),
  allJson: [...article.matchAll(/```json\r?\n([\s\S]*?)```/g)].map(match => match[1]),
};
// Actual prose code is the implementation under test. Independent distribution
// identities and finite differences check its mathematics, not a mirrored loss.
const program = String.raw`
import ast
import contextlib
import io
import json
import math
import sys

data = json.load(sys.stdin)
passed = 0
def check(name, action):
    global passed
    action()
    passed += 1
    print("PASS", name)

def run_article_block(key):
    scope, printed = {}, io.StringIO()
    with contextlib.redirect_stdout(printed):
        exec(compile(data[key], "article:" + key, "exec"), scope)
    return scope, printed.getvalue().splitlines()

pair, pair_output = run_article_block("pair")
dpo_pair, sigmoid = pair["dpo_pair"], pair["sigmoid"]
def actual_pair():
    assert pair_output == ["0.693147 0.287682", "-0.25"]
    old, new = pair["old"], pair["new"]
    assert math.isclose(old[0], math.log(2), rel_tol=1e-14)
    assert math.isclose(new[0], math.log(4 / 3), rel_tol=1e-14)
    assert math.isclose(new[2], math.log(3), rel_tol=1e-14)
    assert math.isclose(new[1], -0.25, rel_tol=1e-14)
check("actual pair code matches the published probability and derivative results", actual_pair)

def absolute_chosen_counterexample():
    reference, old, new = [0.4, 0.4, 0.2], [0.4, 0.4, 0.2], [0.3, 0.1, 0.6]
    for distribution in (reference, old, new):
        assert math.isclose(math.fsum(distribution), 1) and all(p > 0 for p in distribution)
    old_loss = dpo_pair(*map(math.log, old[:2]), *map(math.log, reference[:2]), 1)[0]
    new_loss = dpo_pair(*map(math.log, new[:2]), *map(math.log, reference[:2]), 1)[0]
    assert new_loss < old_loss and new[0] < old[0]
    assert math.isclose(sigmoid(pair["new"][2]), 0.75)
    assert not math.isclose(sigmoid(pair["new"][2]), new[0])
check("normalized distributions improve pair loss while chosen absolute probability falls", absolute_chosen_counterexample)

def reference_ratio_direction():
    # Equal policy odds need not mean zero margin: reference odds are two.
    inputs = [math.log(0.3), math.log(0.3), math.log(0.4), math.log(0.2)]
    loss, derivative, margin = dpo_pair(*inputs, 1)
    assert math.isclose(margin, -math.log(2), rel_tol=1e-14)
    assert math.isclose(loss, math.log(3), rel_tol=1e-14)
    assert math.isclose(derivative, -2 / 3, rel_tol=1e-14)
    swapped = dpo_pair(inputs[1], inputs[0], inputs[3], inputs[2], 1)
    assert math.isclose(swapped[2], -margin, rel_tol=1e-14)
check("reference subtraction and swapped pair signs have the correct direction", reference_ratio_direction)

def finite_difference_gradient():
    step = 1e-5
    for beta in (0.1, 0.3, 1.0, 2.0):
        for delta in (-10.0, -4.0, -1.0, 0.0, 1.0, 4.0, 10.0):
            # All four arguments remain valid nonpositive scalar logprobs.
            args = [-20 + delta, -20, -20, -20]
            derivative = dpo_pair(*args, beta)[1]
            plus, minus = args.copy(), args.copy()
            plus[0] += step
            minus[0] -= step
            numerical = (dpo_pair(*plus, beta)[0] - dpo_pair(*minus, beta)[0]) / (2 * step)
            assert math.isclose(derivative, numerical, rel_tol=3e-5, abs_tol=2e-9), (beta, delta)
            # The frozen-reference score enters delta with the opposite sign.
            plus, minus = args.copy(), args.copy()
            plus[2] += step
            minus[2] -= step
            numerical_ref = (dpo_pair(*plus, beta)[0] - dpo_pair(*minus, beta)[0]) / (2 * step)
            assert math.isclose(-derivative, numerical_ref, rel_tol=3e-5, abs_tol=2e-9)
check("28 independent central differences verify the delta derivative and reference sign", finite_difference_gradient)

def stable_loss_range():
    positive = dpo_pair(-1, -1001, -1, -1, 1)
    negative = dpo_pair(-1001, -1, -1, -1, 1)
    assert positive == (0.0, -0.0, 1000)
    assert negative == (1000.0, -1.0, -1000)
    assert sigmoid(1000) == 1.0 and sigmoid(-1000) == 0.0
check("actual stable functions handle margins plus and minus 1000 without exponential overflow", stable_loss_range)

def numeric_rejections():
    invalid = [
        (-1, -1, -1, -1, -0.1),
        (math.nan, -1, -1, -1, 1),
        (-1, -1, math.inf, -1, 1),
        (-1e308, 0, 0, -1e308, 1),  # finite inputs, overflowing delta
        (-1, -2, -1, -1, 1e308 * 2),  # nonfinite beta
        (-1, -3, -1, -1, 1e308),  # finite delta/beta, overflowing margin
    ]
    for args in invalid:
        try:
            dpo_pair(*args)
        except ValueError:
            pass
        else:
            raise AssertionError(("Accepted nonfinite arithmetic", args))
check("finite inputs do not silently turn overflowing delta or margin into a valid result", numeric_rejections)

def beta_zero():
    for scores in ((-2, -4, -3, -5), (-1e308, 0, 0, -1e308)):
        loss, derivative, margin = dpo_pair(*scores, 0)
        assert loss == math.log(2) and derivative == 0 and margin == 0
    assert -math.log(0.3) != math.log(2)  # Chosen-only SFT is another objective.
check("beta zero is a constant-loss zero-gradient boundary even before delta overflow", beta_zero)

def beta_gradient_weights():
    near_small = -dpo_pair(-10, -10, -10, -10, 0.1)[1]
    near_large = -dpo_pair(-10, -10, -10, -10, 1)[1]
    far_small = -dpo_pair(-5, -10, -10, -10, 0.1)[1]
    far_large = -dpo_pair(-5, -10, -10, -10, 1)[1]
    assert math.isclose(near_small, 0.05) and math.isclose(near_large, 0.5)
    assert round(far_small, 6) == 0.037754 and round(far_large, 6) == 0.006693
    assert near_large > near_small and far_large < far_small
check("the published beta examples do not imply monotonic per-step gradient size", beta_gradient_weights)

def kl(probability, reference):
    return math.fsum(p * math.log(p / q) for p, q in zip(probability, reference) if p > 0)
def optimal_distribution(reference, reward, beta):
    weights = [q * math.exp(r / beta) for q, r in zip(reference, reward)]
    z = math.fsum(weights)
    return [weight / z for weight in weights], z
def kl_identity():
    q, reward, beta = [0.2, 0.5, 0.3], [-0.6, 1.4, 0.2], 0.7
    assert math.isclose(math.fsum(q), 1)
    optimum, z = optimal_distribution(q, reward, beta)
    optimum_j = math.fsum(p * r for p, r in zip(optimum, reward)) - beta * kl(optimum, q)
    assert math.isclose(optimum_j, beta * math.log(z), abs_tol=1e-14)
    checked = 0
    for i in range(51):
        for j in range(51 - i):
            p = [i / 50, j / 50, (50 - i - j) / 50]
            value = math.fsum(pv * r for pv, r in zip(p, reward)) - beta * kl(p, q)
            identity = beta * math.log(z) - beta * kl(p, optimum)
            assert math.isclose(value, identity, abs_tol=2e-14)
            assert value <= optimum_j + 2e-14
            checked += 1
    assert checked == 1326
check("1326 normalized finite distributions verify the KL identity including zero-mass edges", kl_identity)

def reward_constant_cancellation():
    q, reward, beta = [0.2, 0.5, 0.3], [-0.6, 1.4, 0.2], 0.7
    optimum, z = optimal_distribution(q, reward, beta)
    shifted, _ = optimal_distribution(q, [r + 3 for r in reward], beta)
    assert all(math.isclose(p, p2, rel_tol=1e-14) for p, p2 in zip(optimum, shifted))
    recovered = [beta * math.log(p / ref) + beta * math.log(z) for p, ref in zip(optimum, q)]
    assert all(math.isclose(r, r2, abs_tol=1e-14) for r, r2 in zip(reward, recovered))
    margin = beta * (math.log(optimum[1] / q[1]) - math.log(optimum[0] / q[0]))
    assert math.isclose(margin, reward[1] - reward[0], abs_tol=1e-14)
    assert math.isclose(sigmoid(margin), sigmoid((reward[1] + 3) - (reward[0] + 3)))
check("reward recovery and prompt-only constants preserve the optimum and preference probability", reward_constant_cancellation)

sequence, sequence_output = run_article_block("sequence")
def actual_sequence_shift():
    assert sequence_output == ["[3, 4, 5]", "-2.302585", "-0.767528"]
    ids, masks, probabilities = sequence["ids"], sequence["loss_mask"], sequence["next_token_probs"]
    # Independently align next labels with the zipped shifted stream.
    selected = [(label, probability) for label, mask, probability in zip(ids[1:], masks[1:], probabilities) if mask]
    assert [label for label, _ in selected] == [20, 21, 2]
    joint = math.prod(probability for _, probability in selected)
    assert math.isclose(joint, 0.1)
    assert math.isclose(sequence["logp"], math.log(joint), rel_tol=1e-14)
    assert math.isclose(sequence["logp"] / 3, math.log(joint ** (1 / 3)), rel_tol=1e-14)
check("actual sequence code matches a shifted-label product oracle rather than a mean", actual_sequence_shift)

def eos_pad_same_id():
    ids, masks = sequence["ids"], sequence["loss_mask"]
    assert ids[5] == ids[6] == 2 and masks[5] and not masks[6]
    wrong_positions = [pos for pos in sequence["target_positions"] if ids[pos] != 2]
    assert wrong_positions == [3, 4]
    wrong_logp = math.log(0.5 * 0.25)
    assert not math.isclose(wrong_logp, sequence["logp"])
check("position masks retain genuine EOS despite a shared padding ID", eos_pad_same_id)

def incorrect_shift_and_prompt_loss():
    probabilities, masks = sequence["next_token_probs"], sequence["loss_mask"]
    wrong_shift = math.prod(probability for probability, mask in zip(probabilities, masks[:-1]) if mask)
    assert math.isclose(wrong_shift, 0.25 * 0.8 * 0.6) and not math.isclose(wrong_shift, 0.1)
    prompt_included = math.prod(probabilities[:5])
    assert math.isclose(prompt_included, 0.9 * 0.7 * 0.1) and not math.isclose(prompt_included, 0.1)
check("unshifted masks and prompt-inclusive scores change the compared answer event", incorrect_shift_and_prompt_loss)

def mean_changes_ranking():
    ref_token, chosen_token, rejected_token = 0.25, 0.25 * math.exp(0.2), 0.25 * math.exp(0.3)
    assert 0 < chosen_token < 1 and 0 < rejected_token < 1
    chosen_logp, rejected_logp = 2 * math.log(chosen_token), math.log(rejected_token)
    chosen_ref, rejected_ref = 2 * math.log(ref_token), math.log(ref_token)
    sum_margin = dpo_pair(chosen_logp, rejected_logp, chosen_ref, rejected_ref, 1)[2]
    mean_margin = dpo_pair(chosen_logp / 2, rejected_logp, chosen_ref / 2, rejected_ref, 1)[2]
    assert math.isclose(sum_margin, 0.1, abs_tol=1e-14)
    assert math.isclose(mean_margin, -0.1, abs_tol=1e-14)
check("replacing completion sums by token means can reverse the pair margin", mean_changes_ranking)

def identical_pair_gradient():
    # In a deterministic scoring function, both branches depend on the same
    # parameter. Varying it moves them together, unlike differentiating two
    # unrelated scalar arguments in isolation.
    def loss_for_parameter(theta):
        score = math.log(sigmoid(theta))
        return dpo_pair(score, score, -2, -2, 1)[0]
    for theta in (-3, 0, 3):
        assert loss_for_parameter(theta) == math.log(2)
        derivative = (loss_for_parameter(theta + 1e-4) - loss_for_parameter(theta - 1e-4)) / 2e-4
        assert derivative == 0
check("identical deterministic answer branches cancel parameter gradients", identical_pair_gradient)

def truncation_changes_contract():
    chosen, rejected = [20, 21, 99], [20, 21, 98]
    assert chosen != rejected and chosen[:2] == rejected[:2]
    prompt, long_answer, short_answer = [10, 11, 12], [20, 21, 22], [30]
    maximum = 4
    chosen_kept, rejected_kept = (prompt + long_answer)[-maximum:], (prompt + short_answer)[-maximum:]
    assert chosen_kept == [12, 20, 21, 22] and rejected_kept == [10, 11, 12, 30]
    assert chosen_kept[:1] != rejected_kept[:3]  # Retained conditioning differs.
check("truncation can remove the preference or retain different prompt contexts", truncation_changes_contract)

def reference_cache_counterexample():
    # A reference model can give the same target different probabilities
    # under different retained contexts, with each conditional normalized.
    reference = {"full_prompt": [0.8, 0.2], "truncated_prompt": [0.2, 0.8]}
    assert all(math.isclose(math.fsum(row), 1) for row in reference.values())
    cached = math.log(reference["full_prompt"][0])
    current = math.log(reference["truncated_prompt"][0])
    assert not math.isclose(cached, current)
check("unchanged target text does not validate a cache under changed conditioning", reference_cache_counterexample)

def syntax_and_pair_record():
    for block in data["allPython"]:
        ast.parse(block)
    assert len(data["allJson"]) == 1
    record = json.loads(data["allJson"][0])
    assert set(record) == {"prompt", "chosen", "rejected"}
    assert all(isinstance(value, str) and value for value in record.values())
    assert record["chosen"] != record["rejected"]
check("all article code parses and the constructed preference record is valid JSON", syntax_and_pair_record)

print(f"{passed} DPO groups passed; actual stdlib code, finite differences, and finite-distribution counterexamples only, no model training or TRL execution.")
`;
const python = process.env.BLOG_PYTHON ?? (process.platform === 'win32' ? 'python' : 'python3');
const result = spawnSync(python, ['-c', program], {
  input: JSON.stringify(payload), encoding: 'utf8',
  env: { ...process.env, PYTHONUTF8: '1' },
});
if (result.stdout) process.stdout.write(result.stdout);
if (result.stderr) process.stderr.write(result.stderr);
if (result.error) throw result.error;
assert.equal(result.status, 0, 'DPO prose example verification failed');
