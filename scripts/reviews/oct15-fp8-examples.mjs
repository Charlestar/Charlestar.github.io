import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';

const article = readFileSync(new URL('../../_posts/2026-08-01-fp8-formats-scaling-llm-inference.md', import.meta.url), 'utf8');
const parts = article.split('<!-- example: fp8-format-and-scaling-reference -->');
assert.equal(parts.length, 2, 'The FP8 reference marker must be unique');
const found = parts[1].match(/^\s*```python\r?\n([\s\S]*?)```/);
assert.ok(found, 'Missing actual FP8 Python reference');
assert.equal([...article.matchAll(/^```python\s*$/gm)].length, 1, 'All FP8 prose Python code must execute');

const program = String.raw`
import ast
import contextlib
from fractions import Fraction as F
import io
import json
import math
import sys

source = json.load(sys.stdin)["source"]
passed = 0
scope = {}
def check(name, action):
    global passed
    action()
    passed += 1
    print("PASS", name)

def close(a, b):
    assert math.isfinite(a) and math.isclose(a, float(b), rel_tol=1e-12, abs_tol=1e-12), (a, b)

def actual_reference():
    ast.parse(source)
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        exec(compile(source, "article:fp8-format-and-scaling-reference", "exec"), scope)
    assert ast.literal_eval(captured.getvalue().strip()) == {
        "E4M3FN_max": 448, "E5M2_max": 57344, "small_value_with_outlier": 0,
    }
check("the actual FP8 article reference executes with all its assertions and stated output", actual_reference)
decode, encode, quant = scope["decode_fp8"], scope["encode_finite"], scope["quantize_scaled"]

# Independent codebooks are constructed as ordered arithmetic progressions
# within binades. They do not parse exponent/fraction fields from each byte.
def pow2(exponent):
    return F(2**exponent) if exponent >= 0 else F(1, 2**(-exponent))

e4 = [F(i, 512) for i in range(8)]
for exponent in range(-6, 8):
    e4.extend(i*pow2(exponent-3) for i in range(8, 16))
e4.extend(F(i) for i in range(256, 449, 32))
e5 = [F(i, 65536) for i in range(4)]
for exponent in range(-14, 16):
    e5.extend(i*pow2(exponent-2) for i in range(4, 8))
levels = {"E4M3FN": e4, "E5M2": e5}

def independent_all_bytes():
    for fmt, positive in levels.items():
        assert all(a < b for a, b in zip(positive, positive[1:]))
        nan_count = inf_count = zero_count = finite_count = 0
        for code in range(256):
            actual = decode(code, fmt)
            unsigned = code if code < 128 else code-128
            sign = 1 if code < 128 else -1
            if unsigned < len(positive):
                assert F(actual) == sign*positive[unsigned], (fmt, code, actual)
                assert math.copysign(1, actual) == sign
                assert encode(actual, fmt) == code
                finite_count += 1
                zero_count += actual == 0
            elif fmt == "E5M2" and unsigned == 124:
                assert math.isinf(actual) and math.copysign(1, actual) == sign
                inf_count += 1
            else:
                assert math.isnan(actual)
                nan_count += 1
        assert (finite_count, zero_count, inf_count, nan_count) == (
            (254, 2, 0, 2) if fmt == "E4M3FN" else (248, 2, 2, 6))
    assert e4[1] == F(1, 512) and e4[8] == F(1, 64) and e4[-1] == 448
    assert e5[1] == F(1, 65536) and e5[4] == F(1, 16384) and e5[-1] == 57344
    assert decode(0x38, "E4M3FN") == 1 and decode(0x38, "E5M2") == 0.5
check("independently constructed binade codebooks verify all 512 format/byte pairs and special-value counts", independent_all_bytes)

def tie_boundaries_and_saturation():
    for fmt, positive in levels.items():
        for code, (lower, upper) in enumerate(zip(positive, positive[1:])):
            midpoint = float((lower+upper)/2)
            expected = code if code % 2 == 0 else code+1
            for sign, signbit in [(1, 0), (-1, 128)]:
                assert encode(sign*midpoint, fmt) == (expected | signbit)
                assert encode(sign*math.nextafter(midpoint, -math.inf), fmt) == (code | signbit)
                assert encode(sign*math.nextafter(midpoint, math.inf), fmt) == ((code+1) | signbit)
        largest_code = len(positive)-1
        assert encode(sys.float_info.max, fmt) == largest_code
        assert encode(-sys.float_info.max, fmt) == (largest_code | 128)
        assert encode(math.ulp(0.0), fmt) == 0
        assert encode(-math.ulp(0.0), fmt) == 128
        assert encode(0.0, fmt) == 0 and encode(-0.0, fmt) == 128
    assert encode(2**-10, "E4M3FN") == 0
    assert encode(3*2**-10, "E4M3FN") == 2
check("all finite adjacent midpoints, both neighbors and signs obey even rounding including zero and normal/subnormal edges", tie_boundaries_and_saturation)

def normal_power_two_invariance():
    for fmt, positive in levels.items():
        smallest_normal = F(1, 64) if fmt == "E4M3FN" else F(1, 16384)
        for lower, upper in zip(positive, positive[1:]):
            if lower < smallest_normal:
                continue
            for numerator in [1, 2, 3]:
                value = lower+(upper-lower)*F(numerator, 4)
                ordinary = decode(encode(float(value), fmt), fmt)
                for exponent in [-3, -1, 1, 3]:
                    scaled = value*pow2(exponent)
                    if smallest_normal <= scaled <= positive[-1]:
                        restored = decode(encode(float(scaled), fmt), fmt)/float(pow2(exponent))
                        assert restored == ordinary, (fmt, value, exponent)
    assert decode(encode(1.1)) == decode(encode(1.1*256))/256 == 1.125
check("normal-range power-of-two scaling shifts exponent without increasing representational accuracy", normal_power_two_invariance)

def scale_contract_and_underflow():
    for fmt in levels:
        codes, restored, scale = quant([0.0, -0.0], fmt)
        assert codes == [0, 128] and scale == 1
        assert math.copysign(1, restored[1]) == -1
        assert quant([-0.0], fmt)[0] == [128]
    for values, fmt, scale in [
        ([], "E4M3FN", None), ([math.nan], "E4M3FN", None),
        ([math.inf], "E4M3FN", None), ([1], "invalid", None),
        ([1], "E4M3FN", 0), ([1], "E4M3FN", -1),
        ([1], "E4M3FN", math.inf), ([1], "E4M3FN", math.nan),
        ([math.ulp(0.0)], "E4M3FN", None),
        ([1.79e308], "E4M3FN", 1e307),
    ]:
        try:
            quant(values, fmt, scale)
        except ValueError:
            continue
        raise AssertionError(("invalid/unrepresentable scale case accepted", values, fmt, scale))
    for code in [-1, 256, 1.0, True]:
        try:
            decode(code)
        except ValueError:
            continue
        raise AssertionError("invalid code accepted")
    for value in [math.nan, math.inf, -math.inf]:
        try:
            encode(value)
        except ValueError:
            continue
        raise AssertionError("non-finite encoding accepted")
    # Finite saturation deliberately handles normalized +/-infinity caused by
    # finite input / tiny scale; this is not a claim about hardware overflow.
    code, rebuilt, _ = quant([1e308, -1e308], scale=1e-308)
    assert code == [126, 254] and all(math.isfinite(x) for x in rebuilt)
    assert rebuilt[0] > 0 and rebuilt[1] < 0
    close(rebuilt[0], 448*1e-308)
    _, shared, _ = quant([1e-4, 400])
    _, alone, _ = quant([1e-4])
    assert shared[0] == 0 and alone[0] > 0
    close(alone[0], 1e-4)
check("actual scaled quantizer preserves signed zero and enforces finite scale policies while exposing the outlier-underflow example", scale_contract_and_underflow)

def history_distribution_jump():
    stale = quant([10], scale=1/448)
    current = quant([10])
    close(stale[1][0], 1)
    close(current[1][0], 10)
    # In a stable distribution the old range can still be appropriate.
    stable_old = quant([1], scale=1/448)[1][0]
    stable_now = quant([1])[1][0]
    close(stable_old, stable_now)
    # A history maximum can linger after a large value disappears.
    _, stale_small, _ = quant([1e-4], scale=400/448)
    _, current_small, _ = quant([1e-4])
    assert stale_small[0] == 0 < current_small[0]
check("current/stale-scale examples isolate distribution jumps without claiming delayed scaling is universally worse", history_distribution_jump)

def gemm_scale_axes():
    x8 = [decode(0x38), decode(0x38)]
    w8 = [decode(0x38), decode(0x38)]
    sx, sw = [F(1), F(2)], [F(1), F(5)]
    reconstructed = sum(F(x8[k])*sx[k]*F(w8[k])*sw[k] for k in range(2))
    grouped = sum(sx[k]*sw[k]*F(x8[k]*w8[k]) for k in range(2))
    wrongly_factored = sx[0]*sw[0]*sum(x8[k]*w8[k] for k in range(2))
    assert reconstructed == grouped == 11 and wrongly_factored == 2
    # Row and output-column scales stay constant within each dot product.
    x = [[decode(0x38), decode(0x40)], [decode(0x30), decode(0x38)]]
    w = [[decode(0x38), decode(0x30)], [decode(0x40), decode(0x38)]]
    row_scales, col_scales = [F(1, 2), F(4)], [F(2), F(1, 4)]
    for i in range(2):
        for j in range(2):
            independent = sum((F(x[i][k])*row_scales[i])*(F(w[k][j])*col_scales[j]) for k in range(2))
            factored = row_scales[i]*col_scales[j]*sum(F(x[i][k])*F(w[k][j]) for k in range(2))
            assert independent == factored
check("reconstructed GEMM distinguishes valid row/column factoring from invalid K-group scale extraction", gemm_scale_axes)

def directional_blocks_and_requantization():
    original = [[1e-4]*32 for _ in range(32)]
    original[0][0] = 400
    transpose = lambda a: [list(row) for row in zip(*a)]
    def rows_quantized(a):
        result = []
        for row in a:
            amax = max(abs(v) for v in row)
            scale = 2.0**math.ceil(math.log2(amax/448)) if amax else 1.0
            result.append(quant(row, scale=scale)[1])
        return result
    rows = rows_quantized(original)
    original_direction = transpose(rows)
    direct_new_direction = rows_quantized(transpose(original))
    double_quantized = rows_quantized(transpose(rows))
    assert original_direction[1][0] == 0
    assert direct_new_direction[1][0] > 0
    assert double_quantized[1][0] == 0
    assert direct_new_direction != double_quantized
check("a 32-value power-of-two-scaled construction shows direction changes require the original high-precision values", directional_blocks_and_requantization)

print(f"{passed} FP8 groups passed; actual stdlib reference, independent rational codebooks and scale/layout counterexamples only, no Transformer Engine, CUDA cast, GPU GEMM or model experiment.")
`;

const python = process.env.BLOG_PYTHON ?? (process.platform === 'win32' ? 'python' : 'python3');
const result = spawnSync(python, ['-c', program], {
  input: JSON.stringify({ source: found[1] }), encoding: 'utf8',
  env: { ...process.env, PYTHONUTF8: '1' },
});
if (result.stdout) process.stdout.write(result.stdout);
if (result.stderr) process.stderr.write(result.stderr);
if (result.error) throw result.error;
assert.equal(result.status, 0, 'FP8 article reference audit failed');
