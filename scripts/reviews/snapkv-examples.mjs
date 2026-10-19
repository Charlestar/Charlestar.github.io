// CPU algorithm checks extracted from the post; no model, torch, or GPU execution.
import assert from 'node:assert/strict';
import { readFileSync, existsSync } from 'node:fs';
import { spawnSync } from 'node:child_process';

const post = readFileSync('_posts/2026-10-20-snapkv-observation-window-kv-selection.md', 'utf8');
const blocks = [...post.matchAll(/<!-- example: snapkv-select -->\s*```python\r?\n([\s\S]*?)\r?\n```/g)];
assert.equal(blocks.length, 1, 'one actual QK-to-selection example');
const tests = String.raw`
from fractions import Fraction
import itertools
import json
import random

def close(a, b, tolerance=1e-12):
    assert math.isclose(a, b, rel_tol=tolerance, abs_tol=tolerance), (a, b)

def expect_error(fn):
    try:
        fn()
    except (ValueError, OverflowError):
        return
    raise AssertionError("expected invalid input to fail")

# Independent exact-rational oracle from the chosen exp(logit) weights.
expected_votes = [Fraction(a[j], 25) + Fraction(b[j], 26) for j in range(6)]
expected_pool = [sum(expected_votes[max(0, j-1):min(6, j+2)], Fraction())/3
                 for j in range(6)]
for actual, expected in zip(result["votes"], expected_votes):
    close(actual, float(expected))
for actual, expected in zip(result["pooled"], expected_pool):
    close(actual, float(expected))
assert [f"{x:.6f}" for x in result["votes"]] == [
    "0.078462", "0.118462", "0.707692", "0.781538", "0.078462", "0.078462"]
assert [f"{x:.6f}" for x in result["pooled"]] == [
    "0.065641", "0.301538", "0.535897", "0.522564", "0.312821", "0.052308"]
assert len(result["indices"]) == 5 and len(set(result["indices"])) == 5
assert result["indices"][-2:] == [6, 7]
assert result["keys"] == [keys[i] for i in result["indices"]]
assert result["values"] == [values[i] for i in result["indices"]]

# Another head has a different prefix pattern and must not inherit head one's indices.
a2, b2 = [12, 10, 1, 1, 1, 1, 2, 1], [10, 8, 1, 1, 1, 1, 1, 1]
k2 = [[math.log(x), math.log(y)] for x, y in zip(a2, b2)]
head2 = select_prompt_kv(queries, k2, values, 2, 5)
assert head2["indices"][:3] != result["indices"][:3]
assert set(head2["indices"][:3]) == {0, 1, 2}

# Full softmax vs prefix-only softmax selects opposite positions.
ar, br = [9, 1, 90, 1], [1, 3, 2/9, 2/9]
kr = [[math.log(x), math.log(y)] for x, y in zip(ar, br)]
qr = [[0.0, 0.0], [0.0, 0.0], [math.sqrt(2), 0.0], [0.0, math.sqrt(2)]]
vr = [[float(i), -float(i)] for i in range(4)]
counterexample = select_prompt_kv(qr, kr, vr, 2, 3, kernel_size=1)
close(counterexample["votes"][0], 0.315)
close(counterexample["votes"][1], 0.685)
assert counterexample["indices"][0] == 1
wrong_votes = [9/10 + 1/4, 1/10 + 3/4]
assert wrong_votes == [1.15, 0.85] and wrong_votes.index(max(wrong_votes)) == 0

# A future key is invisible to query 6 but visible to query 7.
k_changed = [list(row) for row in keys]
k_changed[7][0] = 10000.0  # q7's first component is zero.
masked = select_prompt_kv(queries, k_changed, values, 2, 5)
assert masked["indices"] == result["indices"]
for actual, expected in zip(masked["votes"], result["votes"]):
    close(actual, expected)

unchanged = select_prompt_kv(queries, keys, values, 2, 9)
assert unchanged["indices"] == list(range(8))
assert unchanged["keys"] == keys and unchanged["keys"] is not keys
full_budget = select_prompt_kv(queries, keys, values, 2, 8)
assert set(full_budget["indices"]) == set(range(8))
all_zero = [[0.0, 0.0] for _ in range(8)]
tie = select_prompt_kv(all_zero, all_zero, values, 2, 5, kernel_size=1)
assert tie["indices"] == [0, 1, 2, 6, 7]  # Explicit teaching tie-break.
maxpool = select_prompt_kv(queries, keys, values, 2, 5, pooling="max")
assert len(maxpool["indices"]) == 5

for kwargs in [{"window": 0, "capacity": 5}, {"window": 2, "capacity": 2},
               {"window": 9, "capacity": 10}, {"window": True, "capacity": 5},
               {"window": 2, "capacity": 5, "kernel_size": 2},
               {"window": 2, "capacity": 5, "pooling": "unknown"}]:
    expect_error(lambda kwargs=kwargs: select_prompt_kv(queries, keys, values, **kwargs))
expect_error(lambda: select_prompt_kv([], [], [], 1, 2))
expect_error(lambda: select_prompt_kv(queries[:-1], keys, values, 2, 5))
bad_key = [list(row) for row in keys]
bad_key[0][0] = float("nan")
expect_error(lambda: select_prompt_kv(queries, bad_key, values, 2, 5))
for bad in [10 ** 1000, 1e308]:
    extreme = [list(row) for row in queries]
    extreme[6][0] = bad
    # Finite inputs alone cannot guarantee a finite dot product; this teaching
    # implementation rejects values/products outside its Python-float domain.
    enormous_key = [list(row) for row in keys]
    enormous_key[0][0] = 1e308
    expect_error(lambda: select_prompt_kv(extreme, enormous_key, values, 2, 5))

# Auditor-generated positive integer exp(logit) weights give an exact rational
# probability/pooling oracle across 720 QK-to-selection runs. The oracle does
# not invoke the article's softmax, vote, pool, or gather implementation.
rng = random.Random(20261020)
for case in range(120):
    length = rng.randrange(3, 11)
    width = rng.randrange(1, min(3, length-1)+1)
    capacity_case = rng.randrange(width+1, length+2)
    prefix_case = length-width
    exp_weights = [[rng.randrange(1, 20) for _ in range(length)] for _ in range(width)]
    q_case = [[0.0]*width for _ in range(length)]
    for row in range(width):
        q_case[prefix_case+row][row] = math.sqrt(width)
    k_case = [[math.log(exp_weights[row][j]) for row in range(width)] for j in range(length)]
    v_case = [[float(j + 10*d) for d in range(width)] for j in range(length)]
    for kernel_case, mode in itertools.product([1, 3, 5], ["avg", "max"]):
        selected_case = select_prompt_kv(q_case, k_case, v_case, width, capacity_case,
                                         kernel_size=kernel_case, pooling=mode)
        if length < capacity_case:
            assert selected_case["indices"] == list(range(length))
            assert selected_case["votes"] == selected_case["pooled"] == []
            continue
        exact_votes = [sum((Fraction(exp_weights[r][j],
                                     sum(exp_weights[r][:prefix_case+r+1]))
                            for r in range(width)), Fraction()) for j in range(prefix_case)]
        radius = kernel_case//2
        padded = [Fraction()]*radius + exact_votes + [Fraction()]*radius
        exact_pool = []
        for j in range(prefix_case):
            neighborhood = padded[j:j+kernel_case]
            exact_pool.append(sum(neighborhood)/kernel_case if mode == "avg" else max(neighborhood))
        for actual, exact in zip(selected_case["votes"], exact_votes):
            close(actual, float(exact))
        for actual, exact in zip(selected_case["pooled"], exact_pool):
            close(actual, float(exact))
        k_budget = capacity_case-width
        selected_prefix = selected_case["indices"][:k_budget]
        threshold = sorted(exact_pool, reverse=True)[k_budget-1]
        assert len(selected_case["indices"]) == len(set(selected_case["indices"])) == capacity_case
        assert all(exact_pool[j] >= threshold for j in selected_prefix)
        assert all(j in selected_prefix for j in range(prefix_case) if exact_pool[j] > threshold)
        assert selected_case["indices"][k_budget:] == list(range(prefix_case, length))
        assert selected_case["keys"] == [k_case[j] for j in selected_case["indices"]]
        assert selected_case["values"] == [v_case[j] for j in selected_case["indices"]]

# Softmax deletion is lossy without any rounding or quantization.
alpha = [0.5, 0.5]
v = [0.0, 10.0]
full_output = sum(a*x for a, x in zip(alpha, v))
keep_output, drop_output, retained_mass = v[0], v[1], alpha[0]
assert full_output == 5 and keep_output == 0
close(keep_output-full_output, (1-retained_mass)*(keep_output-drop_output))
assert abs(keep_output-full_output) <= 2*max(abs(x) for x in v)*(1-retained_mass)

# Exact-rational decomposition and scalar-norm bound on all nontrivial subsets,
# independent of the selector. This includes cases attaining the 2*Vmax bound.
for weights in itertools.product([1, 2, 4], repeat=3):
    probabilities = [Fraction(w, sum(weights)) for w in weights]
    for value_case in itertools.product([-3, 0, 3], repeat=3):
        full = sum(p*x for p, x in zip(probabilities, value_case))
        for mask in range(1, 7):
            kept = [j for j in range(3) if mask & (1 << j)]
            removed = [j for j in range(3) if j not in kept]
            mass = sum(probabilities[j] for j in kept)
            kept_out = sum(probabilities[j]*value_case[j] for j in kept)/mass
            removed_out = sum(probabilities[j]*value_case[j] for j in removed)/(1-mass)
            assert full == mass*kept_out + (1-mass)*removed_out
            assert kept_out-full == (1-mass)*(kept_out-removed_out)
            assert abs(kept_out-full) <= 2*max(abs(x) for x in value_case)*(1-mass)
assert abs(Fraction(3)-Fraction(0)) == 2*3*(1-Fraction(1, 2))

# A two-dimensional RoPE toy checks the mathematical position distinction and
# synchronous K/V permutation, not the historical Transformers cache runtime.
def rotate(vector, position):
    c, s = math.cos(position), math.sin(position)
    return [c*vector[0]-s*vector[1], s*vector[0]+c*vector[1]]

def single_query_output(query, key_rows, value_rows):
    logits = [sum(a*b for a, b in zip(query, key))/math.sqrt(2) for key in key_rows]
    exp_values = [math.exp(x-max(logits)) for x in logits]
    denominator = math.fsum(exp_values)
    return math.fsum(w*v for w, v in zip(exp_values, value_rows))/denominator

positions = [2, 3, 4, 6, 7]
base_keys = [[1.0, 0.25*i] for i in range(5)]
positioned_keys = [rotate(key, p) for key, p in zip(base_keys, positions)]
positioned_values = [1.0, -2.0, 4.0, 0.5, 3.0]
next_query = rotate([1.0, 0.2], 8)
reference_output = single_query_output(next_query, positioned_keys, positioned_values)
permutation = [4, 0, 3, 1, 2]
close(single_query_output(next_query, [positioned_keys[i] for i in permutation],
                          [positioned_values[i] for i in permutation]), reference_output)
reindexed_keys = [rotate(key, p) for p, key in enumerate(base_keys)]
assert not math.isclose(single_query_output(next_query, reindexed_keys, positioned_values),
                        reference_output, rel_tol=1e-6)

# Independent symbolic capacity and logical-position accounting.
batch, layers, hq, hkv, dimension, size = 1, 32, 32, 8, 128, 2
prompt, capacity, generated_in_cache = 16384, 1024, 512
def kv_bytes(heads, tokens):
    return batch*layers*heads*tokens*dimension*size*2
full_bytes = kv_bytes(hkv, prompt+generated_in_cache)
expanded_bytes = kv_bytes(hq, capacity+generated_in_cache)
native_hypothesis = kv_bytes(hkv, capacity+generated_in_cache)
assert [x//1024**2 for x in (full_bytes, expanded_bytes, native_hypothesis)] == [2112, 768, 192]
close(expanded_bytes/full_bytes, 4/11)
close(native_hypothesis/full_bytes, 1/11)
assert 32*32*16384*4 == 64*1024**2  # One FP32 Hq*W*L score matrix, not total peak.
assert prompt+generated_in_cache == 16896
assert capacity+generated_in_cache == 1536
assert prompt+generated_in_cache != capacity+generated_in_cache
assert kv_bytes(hq, min(32, capacity)) / kv_bytes(hkv, 32) == 4  # Short-prompt bypass still expanded.
assert kv_bytes(hq, capacity+4096) == kv_bytes(hkv, prompt+4096)
assert kv_bytes(hq, capacity+8192) > kv_bytes(hkv, prompt+8192)
print(json.dumps({"cpu_groups": 15, "model_gpu_runtime": "not executed"}))
`;
const result = spawnSync(process.env.BLOG_PYTHON || 'python', ['-B', '-c', `${blocks[0][1]}\n${tests}`], { encoding: 'utf8' });
assert.equal(result.status, 0, `${result.error || ''}\n${result.stdout}\n${result.stderr}`);
// post_url targets are checked here too because this series grew on multiple dates.
for (const [, slug] of post.matchAll(/\{% post_url ([^ ]+) %\}/g)) {
  assert.ok(existsSync(`_posts/${slug}.md`), `missing internal post ${slug}`);
}
console.log(result.stdout.trim());
console.log('PASS extracted SnapKV selection and independent probability/storage checks; no model or GPU benchmark.');
