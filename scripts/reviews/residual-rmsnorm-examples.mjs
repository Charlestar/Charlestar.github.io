// CPU numerical-contract examples extracted from the article; no FlashInfer/GPU execution.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';

const post = readFileSync('_posts/2026-10-20-residual-rmsnorm-kernel-fusion.md', 'utf8');
const marker = /<!-- example: residual-rmsnorm-rounding -->\s*```python\r?\n([\s\S]*?)\r?\n```/g;
const blocks = [...post.matchAll(marker)];
assert.equal(blocks.length, 1, 'exactly one actual numerical example');
const tests = String.raw`
from decimal import Decimal, localcontext
from fractions import Fraction
from bisect import bisect_left
import json

def expect_error(fn):
    try:
        fn()
    except (ValueError, OverflowError):
        return
    raise AssertionError("expected invalid example input to fail")

# An independent high-precision mathematical reference, not a GPU rounding oracle.
def decimal_reference(x, r, w, eps):
    with localcontext() as context:
        context.prec = 60
        z = [Decimal.from_float(f16(a)) + Decimal.from_float(f16(b))
             for a, b in zip(x, r)]
        q = sum(v * v for v in z) / Decimal(len(z))
        denom = (q + Decimal.from_float(eps)).sqrt()
        return [float(v / denom * Decimal.from_float(f16(g)))
                for v, g in zip(z, w)]

for d in [1, 3, 111, 4096]:
    x = [((i * 7) % 19 - 9) / 8 for i in range(d)]
    r = [((i * 3) % 11 - 5) / 32 for i in range(d)]
    w = [1 + (i % 5 - 2) / 8 for i in range(d)]
    y, new_r = residual_rmsnorm_fp16(x, r, w)
    expected = decimal_reference(x, r, w, 1e-6)
    assert len(y) == len(new_r) == d
    for actual, target in zip(y, expected):
        assert math.isclose(actual, target, rel_tol=0.002, abs_tol=2e-5)
    assert new_r == [f16(f32(f16(a) + f16(b))) for a, b in zip(x, r)]

assert residual_rmsnorm_fp16([0, 0], [0, 0], [1, 2]) == ([0, 0], [0, 0])
near_zero, _ = residual_rmsnorm_fp16([2**-20], [0], [1])
assert 0 < near_zero[0] < 0.002
positive, _ = residual_rmsnorm_fp16([1, 1], [0, 0], [1, 1])
assert positive == [1, 1]  # FP16 rounds an ideal value slightly below one.
assert sum(positive) / len(positive) != 0

z, eps = [1.0, 2.0, 3.0], 1e-6
q = sum(v * v for v in z) / len(z)
normalized = [v / math.sqrt(q + eps) for v in z]
assert math.isclose(sum(v * v for v in normalized) / 3, q / (q + eps))
wrong_padding = [v / math.sqrt(sum(t*t for t in z) / 4 + eps) for v in z]
assert max(abs(a - b) for a, b in zip(normalized, wrong_padding)) > 0.1
correct_eps, _ = residual_rmsnorm_fp16([0.001], [0], [1], eps=1e-6)
wrong_eps = f16(f16(0.001) / (math.sqrt(f16(0.001)**2) + 1e-6))
assert abs(correct_eps[0] - wrong_eps) > 0.2

# A positive fixed epsilon breaks exact scale invariance near zero.
def ideal_row(row, epsilon):
    scale = math.sqrt(sum(v*v for v in row)/len(row) + epsilon)
    return [v/scale for v in row]
assert ideal_row([0.001], 1e-6) != ideal_row([0.002], 1e-6)
scaled_gamma, _ = residual_rmsnorm_fp16([1, 1], [0, 0], [2, 1])
assert sum(v*v for v in scaled_gamma)/2 > 2

for invalid in [0, -1, True, float("nan"), float("inf"), "1e-6"]:
    expect_error(lambda invalid=invalid: residual_rmsnorm_fp16([1], [0], [1], eps=invalid))
for invalid in [True, float("nan"), float("inf"), "1"]:
    expect_error(lambda invalid=invalid: residual_rmsnorm_fp16([invalid], [0], [1]))
expect_error(lambda: residual_rmsnorm_fp16([], [], []))
expect_error(lambda: residual_rmsnorm_fp16([1, 2], [0], [1, 1]))
for invalid in [1e-100, 1e100, 10 ** 1000]:
    expect_error(lambda invalid=invalid: residual_rmsnorm_fp16([0], [0], [1], eps=invalid))
expect_error(lambda: residual_rmsnorm_fp16([10 ** 1000], [0], [1]))
expect_error(lambda: residual_rmsnorm_fp16([70000], [0], [1]))
expect_error(lambda: residual_rmsnorm_fp16([65504], [65504], [1]))
expect_error(lambda: residual_rmsnorm_fp16([1, 0], [0, 0], [65504, 1]))
expect_error(lambda: residual_rmsnorm_fp16([1], [0], [1], round_before_norm="false"))

# Auditor's independent binary16 decoder and exact-rational nearest-even quantizer.
# Neither construction uses struct.pack/unpack or the article's f16 helper.
def half_value(code):
    exponent, mantissa = code >> 10, code & 1023
    if exponent == 0:
        return Fraction(mantissa, 2 ** 24)
    significand = Fraction(1024 + mantissa)
    power = exponent - 25
    return significand * (2 ** power if power >= 0 else Fraction(1, 2 ** -power))

half_values = [half_value(code) for code in range(0x7c00)]

def half_rne_exact(value):
    value = Fraction(value)
    sign = -1 if value < 0 else 1
    magnitude = abs(value)
    if magnitude >= 65520:
        raise ValueError("binary16 result would overflow")
    high = bisect_left(half_values, magnitude)
    if high == 0:
        chosen = half_values[0]
    elif high == len(half_values):
        chosen = half_values[-1]
    else:
        low = high - 1
        dl, dh = magnitude - half_values[low], half_values[high] - magnitude
        chosen = half_values[low if dl < dh or (dl == dh and low % 2 == 0) else high]
    return sign * chosen

# Every adjacent finite positive half pair, their negative mirrors, and both sides
# of the midpoint. The zero/subnormal and subnormal/normal boundaries are included.
for code, low in enumerate(half_values[:-1]):
    high = half_values[code + 1]
    middle = (low + high) / 2
    expected = half_rne_exact(middle)
    assert f16(float(middle)) == float(expected)
    assert f16(-float(middle)) == -float(expected)
    assert f16(math.nextafter(float(middle), -math.inf)) == float(low)
    assert f16(math.nextafter(float(middle), math.inf)) == float(high)

# Independent high-precision check of both table rows; input and final FP16
# conversion use the rational quantizer, not the article's half-rounding helper.
def table_oracle(round_first):
    with localcontext() as context:
        context.prec = 90
        z = [Fraction(1) + Fraction(1, 2048), Fraction(2)]
        if round_first:
            z = [half_rne_exact(v) for v in z]
        dz = [Decimal(v.numerator) / Decimal(v.denominator) for v in z]
        q = sum(v*v for v in dz) / 2
        divisor = (q + Decimal("0.000001")).sqrt()
        return [float(half_rne_exact(Fraction(v / divisor))) for v in dz]

assert table_oracle(False) == fused_y
assert table_oracle(True) == rounded_y
assert [half_rne_exact(Fraction(1) + Fraction(1, 2048)), half_rne_exact(2)] == [1, 2]

# The example is functional: it returns two new rows and never mutates its lists.
originals = ([1, -2, 0], [2 ** -11, 0, 0], [1, 2, -1])
snapshots = [list(row) for row in originals]
y_check, r_check = residual_rmsnorm_fp16(*originals)
assert list(originals) == snapshots
assert y_check is not r_check

# Explicit logical bytes, independently enumerated by stage and tensor.
m, d, s = 128, 4096, 2
n = m*d
add = {"read_x": n*s, "read_r": n*s, "write_r": n*s, "write_z32": n*4}
norm = {"read_z32": n*4, "read_gamma_per_row": n*s, "write_y": n*s}
fused = {"read_x": n*s, "read_r": n*s, "read_gamma_per_row": n*s,
         "write_r": n*s, "write_y": n*s}
assert sum(add.values()) + sum(norm.values()) == 9437184 == 5*n*s + 8*n
assert sum(fused.values()) == 5242880 == 5*n*s
assert 9437184 - 5242880 == 4194304 == 4*1024**2
vec = math.gcd(16//s, d)
block = min(1024, d//vec)
warps = (block+31)//32
smem = (((warps+3)//4)*4 + d)*4
assert (vec, block, warps, smem) == (8, 512, 16, 16448)
print(json.dumps({"cpu_groups": 12, "gpu_runtime": "not executed"}))
`;
const python = process.env.BLOG_PYTHON || 'python';
const result = spawnSync(python, ['-B', '-c', `${blocks[0][1]}\n${tests}`], { encoding: 'utf8' });
assert.equal(result.status, 0, `${result.error || ''}\n${result.stdout}\n${result.stderr}`);
console.log(result.stdout.trim());
console.log('PASS extracted residual RMSNorm contracts, boundary examples, and logical traffic; no GPU measured.');
