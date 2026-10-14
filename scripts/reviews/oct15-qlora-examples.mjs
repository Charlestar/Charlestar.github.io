import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';

const article = readFileSync(new URL('../../_posts/2026-10-15-qlora-nf4-frozen-base.md', import.meta.url), 'utf8');
function extract(name) {
  const parts = article.split(`<!-- example: qlora-${name} -->`);
  assert.equal(parts.length, 2, `Missing or duplicate QLoRA marker: ${name}`);
  const block = parts[1].match(/^\s*```python\r?\n([\s\S]*?)```/);
  assert.ok(block, `Missing actual Python block: ${name}`);
  return block[1];
}
const blocks = { nf4: extract('nf4-block-and-packing'), grads: extract('frozen-base-gradients') };
assert.equal([...article.matchAll(/^```python\s*$/gm)].length, 2, 'Exercise every QLoRA Python block');

// The implementation is the real manuscript. Independent oracles use normal
// quantiles, rational interval boundaries, bitstrings, enumerated block records,
// and loss finite differences. No ML framework or GPU is imported.
const program = String.raw`
import ast
import bisect
import contextlib
from fractions import Fraction as F
from hashlib import sha256
import io
import json
import math
from statistics import NormalDist
import struct
import sys

blocks = json.load(sys.stdin)
scopes, outputs = {}, {}
passed = 0
def check(name, action):
    global passed
    action()
    passed += 1
    print("PASS", name)

def close(actual, expected, rel_tol=1e-10, abs_tol=1e-11):
    assert math.isfinite(actual) and math.isclose(actual, float(expected), rel_tol=rel_tol, abs_tol=abs_tol), (actual, expected)

def transpose(a):
    return [list(row) for row in zip(*a)]

def multiply(a, b):
    return [[math.fsum(x*y for x, y in zip(row, col)) for col in transpose(b)] for row in a]

def actual_blocks():
    for name, source in blocks.items():
        ast.parse(source)
        scope, captured = {}, io.StringIO()
        with contextlib.redirect_stdout(captured):
            exec(compile(source, "article:qlora-"+name, "exec"), scope)
        scopes[name], outputs[name] = scope, ast.literal_eval(captured.getvalue().strip())
    assert outputs["nf4"]["codes"] == [7, 9, 3, 15]
    assert outputs["nf4"]["bytes"] == "793f"
    close(outputs["grads"]["output"], F(-7, 20))
    for actual, expected in zip(outputs["grads"]["dX"], [F(-1431, 2000), F(297, 500)]):
        close(actual, expected)
    close(outputs["grads"]["dB"], F(-27, 40))
check("both actual manuscript blocks execute independently with their own assertions and stated outputs", actual_blocks)
cbook = scopes["nf4"]["NF4"]
encode, pack, unpack = (scopes["nf4"][key] for key in ["encode_nf4_block", "pack_codes", "unpack_codes"])

def independently_constructed_nf4():
    # The pinned create_normal_map's asymmetric probability grids provide a
    # different oracle from copying 16 manuscript literals. NormalDist uses
    # binary64, not the original torch FP32 linspace/scipy sequence, so the
    # construction comparison is approximate, not a bit-exact library claim.
    distribution, offset = NormalDist(), 0.9677083
    positives = [distribution.inv_cdf(offset+(0.5-offset)*i/8) for i in range(8)]
    negatives = [-distribution.inv_cdf(offset+(0.5-offset)*i/7) for i in range(7)]
    normalization = positives[0]
    independent = sorted([v/normalization for v in negatives+positives]+[0.0])
    # Exact data fixture independently read from the pinned 0.47.0
    # backends/utils.py table, not generated from the manuscript. Quantile
    # construction verifies the design; these FP32 words catch one-ULP typos.
    source_words = ["bf800000", "bf3239b1", "bf066b30", "beca32a0",
                    "be91a24d", "be3d353f", "bdba7871", "00000000",
                    "3da2faff", "3e24cae3", "3e7c04dd", "3ead033a",
                    "3ee1a4b8", "3f1007ab", "3f3913b3", "3f800000"]
    assert [struct.pack(">f", value).hex() for value in cbook] == source_words
    assert len(cbook) == len(independent) == 16
    assert sum(v < 0 for v in cbook) == 7 and sum(v == 0 for v in cbook) == 1 and sum(v > 0 for v in cbook) == 8
    assert cbook[0] == -1 and cbook[7] == 0 and cbook[-1] == 1
    assert all(a < b for a, b in zip(cbook, cbook[1:]))
    for actual, expected in zip(cbook, independent):
        close(actual, expected, rel_tol=2e-6, abs_tol=2e-7)
        assert struct.unpack("<f", struct.pack("<f", actual))[0] == actual
    assert any(cbook[i] != -cbook[-1-i] for i in range(8))
    # Normalized E2M1 has a different, sign-symmetric finite codebook.
    e2m1_positive = [F(0), F(1, 12), F(1, 6), F(1, 4), F(1, 3), F(1, 2), F(2, 3), F(1)]
    assert set(map(F, cbook)) != set(e2m1_positive+[-v for v in e2m1_positive])
    assert scopes["grads"]["endpoint_table"] == {0: cbook[0], 15: cbook[15]}
check("normal-quantile construction independently checks all NF4 levels, asymmetry, exact zero and distinction from E2M1", independently_constructed_nf4)

def quantizer_thresholds_and_finite_domain():
    thresholds = [(F(a)+F(b))/2 for a, b in zip(cbook, cbook[1:])]
    for scale in [2.0**-20, 1.0, 2.0**20]:
        candidates = [float(F(i, 512)*F(scale)) for i in range(-512, 513)]
        candidates += [scale*v for v in cbook]
        for threshold in thresholds:
            midpoint = float(threshold)*scale
            candidates += [math.nextafter(midpoint, -math.inf), midpoint, math.nextafter(midpoint, math.inf)]
        codes, restored, actual_scale = encode([-scale, scale]+candidates)
        assert actual_scale == scale
        for value, code, rebuilt in zip(candidates, codes[2:], restored[2:]):
            # Boundaries live in reconstructed weight units. bisect_left puts
            # exact equal distances at the lower code, matching the declared tie.
            boundaries = [v*F(scale) for v in thresholds]
            expected = bisect.bisect_left(boundaries, F(value))
            assert code == expected, (scale, value, code, expected)
            assert F(rebuilt) == F(scale)*F(cbook[expected])
    assert encode([0, -0.0])[0] == [7, 7] and encode([0, 0])[2] == 1
    for values in [[], [math.nan], [math.inf], [-math.inf]]:
        try:
            encode(values)
        except ValueError:
            continue
        raise AssertionError(("invalid block accepted", values))
    # Endpoints remain representable at both binary64 extremes. Other
    # subnormal reconstruction rounding is not an exact-real guarantee.
    for value in [math.ulp(0.0), sys.float_info.max]:
        codes, rebuilt, scale = encode([-value, value])
        assert codes == [0, 15] and rebuilt == [-value, value] and scale == value
check("the actual encoder matches rational reconstructed-midpoint oracles for all boundaries and finite block policies", quantizer_thresholds_and_finite_domain)

def independent_packing_and_malformed_inputs():
    for a in range(16):
        for b in range(16):
            payload, length = pack([a, b])
            independent_byte = int(format(a, "04b")+format(b, "04b"), 2)
            assert payload == bytes([independent_byte]) and length == 2
            assert unpack(bytes([independent_byte]), 2) == [a, b]
    assert pack([0, 1, 2]) == (bytes([0x01, 0x27]), 3)
    assert pack([7, 9, 3, 15]) == (bytes([0x79, 0x3f]), 4)
    assert unpack(bytearray([0xf7]), 1) == [15]
    assert pack([]) == (b"", 0) and unpack(b"", 0) == []
    for codes in [[-1], [16], [1.0], [True], ["7"]]:
        try:
            pack(codes)
        except ValueError:
            continue
        raise AssertionError(("invalid code accepted", codes))
    for payload, length in [(b"", 1), (b"\x77", 0), (b"\xff", 1),
                            (b"\x77", -1), (b"\x77", 1.0), (b"\x77", True),
                            ([256], 2), ([-1], 2), ("77", 2), (memoryview(b"\x77"), 2)]:
        try:
            unpack(payload, length)
        except ValueError:
            continue
        raise AssertionError(("invalid payload accepted", payload, length))
check("independent bitstrings check all 256 byte/code pairs, known bytes, odd padding and strict byte-container validation", independent_packing_and_malformed_inputs)

def double_quantization_budget_and_offset():
    asymptotic = F(4)+F(8, 64)+F(32, 64*256)
    assert asymptotic == F(2113, 512) and float(asymptotic) == 4.126953125
    assert F(9, 2)-asymptotic == F(191, 512)
    for p in [1, 2, 63, 64, 65, 127, 128, 255, 256, 16383, 16384, 16385]:
        # Enumerate actual first/second-level records rather than using ceil
        # to obtain both the subject expression and its expected answer.
        first_records = [list(range(start, min(start+64, p))) for start in range(0, p, 64)]
        second_records = [first_records[start:start+256] for start in range(0, len(first_records), 256)]
        payload, logical_length = pack([7]*p)
        bits = 8*len(payload)+sum(8 for _ in first_records)+sum(32 for _ in second_records)+32
        n1, n2 = (p+63)//64, (len(first_records)+255)//256
        assert len(first_records) == n1 and len(second_records) == n2 and logical_length == p
        assert bits == 8*((p+1)//2)+8*n1+32*n2+32
        if p == 16384:
            assert F(bits, p) == asymptotic+F(32, p)
        if p == 1:
            assert bits == 80 > 8+32  # Tiny-tensor overhead can exceed plain scales.
    # Center/decenter identity and the correlated scale-error term do not
    # depend on assuming a hardware E4M3 cast. This does NOT execute bnb's
    # actual signed-dynamic 8-bit second-level quantizer.
    scales = [F(1, 2), F(2), F(5)]
    offset = sum(scales)/len(scales)
    centered = [v-offset for v in scales]
    assert any(v < 0 for v in centered) and any(v > 0 for v in centered)
    assert [v+offset for v in centered] == scales
    assert centered != scales
    delta_scale = F(1, 64)
    indices = [0, 7, 9, 15]
    for code in indices:
        a, cv = scales[1], F(cbook[code])
        assert (a+delta_scale)*cv-a*cv == delta_scale*cv
check("enumerated block records verify ceil/tail/offset budgets and exact centering plus correlated scale-error algebra", double_quantization_budget_and_offset)

def actual_gradient_coordinates_and_initialization():
    scope = scopes["grads"]
    gradients, objective = scope["gradients"], scope["loss_flat"]
    parameters = scope["parameters"]
    expected = [F(-1431, 2000), F(297, 500), F(-81, 200), F(-81, 100), F(-27, 40)]
    # A dyadic step and multiple sizes make this independent from the one
    # epsilon hardcoded in the article, while differentiating its real loss.
    for step in [2.0**-12, 2.0**-16, 2.0**-20]:
        for index, derivative in enumerate(expected):
            plus, minus = parameters.copy(), parameters.copy()
            plus[index] += step; minus[index] -= step
            numeric = (objective(plus)-objective(minus))/(2*step)
            close(numeric, derivative, rel_tol=1e-7, abs_tol=1e-9)
    x, a = scope["x"], scope["a"]
    dx, da, db = gradients(x, a, 0)
    assert da == [0, 0] and db != 0
    assert all(v != 0 for v in dx)
    next_b = -0.05*db
    assert any(v != 0 for v in gradients(x, a, next_b)[1])
    # With c=0 the base alone still propagates dX. Omitting this term is not
    # equivalent to freezing base parameters.
    old_c = scope["c"]
    scope["c"] = 0
    dx_base, da_base, db_base = gradients(x, a, scope["b"])
    assert da_base == [0, 0] and db_base == 0 and any(v != 0 for v in dx_base)
    for i in range(2):
        plus, minus = parameters.copy(), parameters.copy()
        plus[i] += 2**-16; minus[i] -= 2**-16
        close((objective(plus)-objective(minus))/(2*2**-16), dx_base[i], rel_tol=1e-7, abs_tol=1e-9)
    scope["c"] = old_c
    # General M=2,K=3,r=2,N=2 checks all 16 X/A/B coordinates and c!=1.
    x0 = [[1, -0.5, 0.25], [-1, 2, 0.5]]
    a0 = [[0.125, -0.25], [0.5, 0.125], [-0.125, 0.25]]
    b0 = [[0.25, -0.5], [0.125, 0.375]]
    _, decoded, _ = encode([0.5, -0.49, 0.125, -0.25, 0.375, 0.0])
    base = [decoded[i:i+2] for i in range(0, 6, 2)]
    target, c = [[0.5, -0.25], [0, 1]], 2/3
    def prediction(x, a, b):
        main, adapter = multiply(x, base), multiply(multiply(x, a), b)
        return [[v+c*adapter[i][j] for j, v in enumerate(row)] for i, row in enumerate(main)]
    y = prediction(x0, a0, b0)
    g = [[v-target[i][j] for j, v in enumerate(row)] for i, row in enumerate(y)]
    da = multiply(multiply(transpose(x0), g), transpose(b0))
    db = multiply(multiply(transpose(a0), transpose(x0)), g)
    dx_base, dx_adapter = multiply(g, transpose(base)), multiply(multiply(g, transpose(b0)), transpose(a0))
    dx = [[v+c*dx_adapter[i][j] for j, v in enumerate(row)] for i, row in enumerate(dx_base)]
    da, db = [[[c*v for v in row] for row in matrix] for matrix in [da, db]]
    def matrix_loss(x, a, b):
        y = prediction(x, a, b)
        return 0.5*math.fsum((v-target[i][j])**2 for i, row in enumerate(y) for j, v in enumerate(row))
    for matrix_index, analytic in enumerate([dx, da, db]):
        for i, row in enumerate(analytic):
            for j, derivative in enumerate(row):
                plus = [[r.copy() for r in matrix] for matrix in [x0, a0, b0]]
                minus = [[r.copy() for r in matrix] for matrix in [x0, a0, b0]]
                step = 2**-16
                plus[matrix_index][i][j] += step; minus[matrix_index][i][j] -= step
                numeric = (matrix_loss(*plus)-matrix_loss(*minus))/(2*step)
                close(numeric, derivative, rel_tol=1e-7, abs_tol=1e-9)
    assert any(abs(v) > 0.01 for row in dx_base for v in row)
check("every real toy derivative and every rectangular X/A/B coordinate matches independent loss finite differences, including B=0 and base-only dX", actual_gradient_coordinates_and_initialization)

def merge_requantization_and_frozen_evidence():
    scope = scopes["grads"]
    indices = unpack(scope["base_packet"], 2)
    assert indices == [15, 0]
    base = [scope["base_scale"]*cbook[code] for code in indices]
    assert tuple(base) == scope["base"]
    delta = [scope["c"]*v*scope["b"] for v in scope["a"]]
    merged = [v+d for v, d in zip(base, delta)]
    for actual, expected in zip(merged, [F(53, 100), F(-11, 25)]):
        close(actual, expected)
    original_output = scope["prediction"](scope["x"], scope["a"], scope["b"])
    close(math.fsum(v*w for v, w in zip(scope["x"], merged)), original_output)
    codes, requantized, new_scale = encode(merged)
    assert codes == [15, 1] and new_scale == max(map(abs, merged))
    close(new_scale, F(53, 100))
    assert new_scale != scope["base_scale"]
    close(requantized[1], new_scale*cbook[1])
    changed_output = math.fsum(v*w for v, w in zip(scope["x"], requantized))
    close(changed_output, -0.207964369058609)
    assert abs(changed_output-original_output) > 0.1
    # This unquantized W produces the SAME NF4 base codes and scale but merging
    # into W instead of DQ(Q(W)) is still a different trained function.
    original_w = [0.5, -0.49]
    assert encode(original_w)[0] == [15, 0] and encode(original_w)[1] == base
    wrong_merged = [v+d for v, d in zip(original_w, delta)]
    assert not math.isclose(math.fsum(v*w for v, w in zip(scope["x"], wrong_merged)), original_output)
    # The checksum covers only this toy packet and binary64 scale, not all
    # model parameters, bias/norm freeze settings or framework optimizers.
    digest = sha256(scope["base_packet"]+struct.pack("<d", scope["base_scale"])).digest()
    assert digest == scope["before"]
    assert scope["loss_flat"](scope["x"]+scope["updated_a"]+[scope["updated_b"]]) < scope["loss_flat"](scope["parameters"])
check("merge preserves the pre-update two-branch function, fresh-absmax requantization changes it, and frozen-byte evidence stays limited to the toy", merge_requantization_and_frozen_evidence)

assert passed == 7
print(json.dumps({"groups": passed, "scope": "actual manuscript Python and independent standard-library arithmetic; no bitsandbytes/PyTorch, GPU, paged optimizer, LLM training or model-quality experiment"}))
`;

const result = spawnSync(process.env.BLOG_PYTHON || 'python', ['-c', program], {
  input: JSON.stringify(blocks), encoding: 'utf8', maxBuffer: 4 * 1024 * 1024,
});
if (result.stdout) process.stdout.write(result.stdout);
if (result.stderr) process.stderr.write(result.stderr);
assert.equal(result.error, undefined, result.error?.message);
assert.equal(result.status, 0, 'QLoRA manuscript examples or independent oracles failed');
