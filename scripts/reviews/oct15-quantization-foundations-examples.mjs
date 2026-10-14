import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';

const article = readFileSync(new URL('../../_posts/2026-10-15-quantization-numerical-foundations.md', import.meta.url), 'utf8');
function extract(name) {
  const parts = article.split(`<!-- example: quant-foundations-${name} -->`);
  assert.equal(parts.length, 2, `Missing or duplicated article marker: ${name}`);
  const block = parts[1].match(/^\s*```python\r?\n([\s\S]*?)```/);
  assert.ok(block, `Missing Python code after ${name}`);
  return block[1];
}
const blocks = { affine: extract('affine'), dot: extract('dot'), ste: extract('ste') };
assert.equal([...article.matchAll(/^```python\s*$/gm)].length, 3, 'All three prose Python blocks must be exercised');

// The implementation under test is extracted from the actual manuscript.
// Rational nearest-grid search and reconstructed matrix multiplication are
// independent oracles; this suite does not import a ML framework or run a GPU.
const program = String.raw`
import ast
import contextlib
from fractions import Fraction as F
import io
import json
import math
import struct
import sys

blocks = json.load(sys.stdin)
passed = 0
scopes = {}
outputs = {}
def check(name, action):
    global passed
    action()
    passed += 1
    print("PASS", name)

def close(actual, expected):
    assert math.isclose(actual, float(expected), rel_tol=1e-12, abs_tol=1e-12), (actual, expected)

def actual_blocks():
    for name, source in blocks.items():
        ast.parse(source)
        scope, captured = {}, io.StringIO()
        with contextlib.redirect_stdout(captured):
            exec(compile(source, "article:" + name, "exec"), scope)
        scopes[name], outputs[name] = scope, ast.literal_eval(captured.getvalue().strip())
    assert [r[0] for r in outputs["affine"]] == [-2, -2, 0, 0, 2, 6, 7]
    assert outputs["dot"] == {"centered_sum": 7, "output": 0.875, "group_scaled_output": 1.625}
    assert outputs["ste"] == {"finite_difference": 0.0, "surrogate_gradient": 1.0}
check("all three actual article blocks run independently and produce their stated examples", actual_blocks)
quantize = scopes["affine"]["quantize_scalar"]
fake_quant = scopes["ste"]["fake_quant_and_mask"]
ste_dx = scopes["ste"]["ste_dx"]

def rational_nearest_grid():
    cases = 0
    for lo, hi, zero in [(-7, 7, 0), (-5, 4, -1), (0, 15, 7), (-128, 127, 0), (0, 255, 128)]:
        positions = range(8 * (lo-zero-2), 8 * (hi-zero+2)+1)
        if hi - lo > 32:
            positions = sorted(set([8*j+d for j in [lo-zero-1, lo-zero, -1, 0, 1, hi-zero, hi-zero+1] for d in [-4, -1, 0, 1, 4]]))
        for scale in [F(1, 8), F(1, 2), F(2)]:
            for position in positions:
                value = scale * F(position, 8)
                code, restored, clipped = quantize(float(value), float(scale), zero, lo, hi)
                # Search reconstructed levels directly; ties use the parity of
                # the rounded normalized integer, not the shifted code.
                expected = min(range(lo, hi+1), key=lambda c: (abs(scale*(c-zero)-value), (c-zero) % 2, c))
                base = position // 8
                unconstrained = min(range(base-2, base+3), key=lambda n: (abs(F(n)-F(position, 8)), n % 2, n))
                assert code == expected, (value, scale, zero, code, expected)
                close(restored, scale*(expected-zero))
                assert clipped == (not lo <= unconstrained+zero <= hi)
                if scale*(lo-zero) <= value <= scale*(hi-zero):
                    assert abs(F(restored)-value) <= scale/2
                cases += 1
    print("  rational grid cases:", cases)
check("actual affine code agrees with rational nearest-level search including signed ties and shifted zeros", rational_nearest_grid)

def rejects_bad_parameters():
    bad = [
        (math.nan, 1, 0, -7, 7), (math.inf, 1, 0, -7, 7),
        (0, math.nan, 0, -7, 7), (0, math.inf, 0, -7, 7),
        (0, 0, 0, -7, 7), (0, -1, 0, -7, 7),
        (0, 1, 0.0, -7, 7), (0, 1, True, -7, 7),
        (0, 1, 0, -7.0, 7), (0, 1, 8, -7, 7),
        (0, 1, 0, 7, 7), (0, 1, 0, 8, -7),
        (0, 1, 0, -129, 127), (0, 1, 0, 0, 256),
        (0, 1, 0, -128, 255), (0, 1, 0, -1, 255),
        (1e308, 1e-308, 0, -7, 7), (1.7e308, 1e308, 0, -7, 7),
    ]
    for args in bad:
        try:
            quantize(*args)
        except ValueError:
            continue
        raise AssertionError(("invalid input accepted", args))
    for zero, lo, hi in [(0, -127, 127), (128, 0, 255), (7, 0, 15)]:
        assert quantize(0.0, 1.0, zero, lo, hi) == (zero, 0.0, False)
    assert quantize(0, 1, 0, -128, 127)[0] == 0
    assert quantize(0, 1, 128, 0, 255)[0] == 128
check("finite-input guards reject invalid byte ranges and intermediate overflow while zero groups remain valid", rejects_bad_parameters)

def clipping_and_zero_domain():
    data = [1.0]*100 + [10.0]
    errors = [math.fsum((quantize(x, s, 0, -1, 1)[1]-x)**2 for x in data) for s in [10.0, 1.0]]
    assert errors == [100.0, 81.0]
    assert quantize(7.2, 1, 0, -7, 7) == (7, 7, False)
    assert quantize(7.6, 1, 0, -7, 7) == (7, 7, True)
    assert abs(quantize(100, 1, 0, -7, 7)[1]-100) == 93
    assert 0.1*(0-128) == -12.8 and quantize(0, 0.1, 128, 0, 255)[0] == 128
    # Moving an odd zero point inside round changes both positive and negative ties.
    for t in [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]:
        assert quantize(t, 1, 1, -7, 7)[0] != round(t+1)
check("clipping tradeoff, endpoint/code distinction and zero-point units match concrete counterexamples", clipping_and_zero_domain)

def matmul(a, b):
    return [[sum((a[i][k]*b[k][j] for k in range(len(b))), F(0)) for j in range(len(b[0]))] for i in range(len(a))]

def matrix_and_group_contract():
    x = [[F(1, 8), F(3, 4), F(-1, 2), F(3, 2), F(-2)], [F(-3, 8), F(1), F(1, 4), F(-1), F(1, 2)]]
    w = [[F(k-2, 4), F(3-k, 2)] for k in range(5)]
    sx, zx = F(1, 4), 2
    sw, zw = [F(1, 8), F(1, 2)], [-1, 3]
    qx = [[quantize(float(v), float(sx), zx, -7, 7)[0] for v in row] for row in x]
    qw = [[quantize(float(w[k][n]), float(sw[n]), zw[n], -7, 7)[0] for n in range(2)] for k in range(5)]
    xhat = [[sx*(v-zx) for v in row] for row in qx]
    what = [[sw[n]*(qw[k][n]-zw[n]) for n in range(2)] for k in range(5)]
    decoded = matmul(xhat, what)
    for m in range(2):
        for n in range(2):
            raw = sum(qx[m][k]*qw[k][n] for k in range(5))
            corrected = raw-zw[n]*sum(qx[m])-zx*sum(qw[k][n] for k in range(5))+5*zx*zw[n]
            assert sx*sw[n]*corrected == decoded[m][n]
    # Non-divisible K group sizes and varying group zero points/scales.
    gs, gz = [F(1, 8), F(1, 4), F(1, 2)], [-1, 2, 0]
    grouped_what = [[gs[k//2]*(qw[k][n]-gz[k//2]) for n in range(2)] for k in range(5)]
    decoded_grouped = matmul(xhat, grouped_what)
    for m in range(2):
        for n in range(2):
            partials = [sum((qx[m][k]-zx)*(qw[k][n]-gz[g]) for k in range(2*g, min(2*g+2, 5))) for g in range(3)]
            assert sx*sum(gs[g]*partials[g] for g in range(3)) == decoded_grouped[m][n]
    assert scopes["dot"]["grouped"] != scopes["dot"]["wrong"]
check("reconstructed rectangular matrix products verify zero corrections and K-group scales against independent GEMM", matrix_and_group_contract)

def full_error_identity():
    x = [[F(1, 2), F(-3), F(2)], [F(0), F(1, 4), F(4)]]
    w = [[F(1), F(-2)], [F(1, 4), F(3)], [F(-1), F(1, 2)]]
    dx = [[F(1, 8), F(-1, 4), F(0)], [F(-1, 8), F(0), F(1, 4)]]
    dw = [[F(-1, 8), F(0)], [F(1, 8), F(1, 4)], [F(0), F(-1, 4)]]
    add = lambda a, b: [[v+b[i][j] for j, v in enumerate(row)] for i, row in enumerate(a)]
    original, perturbed = matmul(x, w), matmul(add(x, dx), add(w, dw))
    terms = add(add(matmul(dx, w), matmul(x, dw)), matmul(dx, dw))
    assert terms == [[perturbed[i][j]-original[i][j] for j in range(2)] for i in range(2)]
    errors = [[F(0), F(1, 10)], [F(1, 5), F(0)]]
    mse = [sum(v*v for v in e)/2 for e in errors]
    out = [abs(e[0]+100*e[1]) for e in errors]
    assert mse == [F(1, 200), F(1, 50)] and out == [F(10), F(1, 5)]
    assert mse[0] < mse[1] and out[0] > out[1]
check("matrix error decomposition includes cross terms and weight MSE can rank output errors backward", full_error_identity)

def granularity_and_budget():
    matrix = [[0.1, 100], [0.2, 80]]
    global_values = [[quantize(v, 100/7, 0, -7, 7)[1] for v in row] for row in matrix]
    channel_values = [[quantize(v, [0.2/7, 100/7][n], 0, -7, 7)[1] for n, v in enumerate(row)] for row in matrix]
    assert global_values[0][0] == global_values[1][0] == 0
    close(channel_values[0][0], F(4, 35))
    close(channel_values[1][0], F(1, 5))
    assert sum((channel_values[k][0]-matrix[k][0])**2 for k in range(2)) < 0.1**2+0.2**2
    for k in [1, 2, 31, 32, 33, 127, 128, 129]:
        for n in [1, 3]:
            for group in [16, 32, 128]:
                groups = [list(range(start, min(start+group, k))) for _ in range(n) for start in range(0, k, group)]
                actual_bits = sum(4*len(chunk)+16 for chunk in groups)
                assert actual_bits == 4*k*n+16*n*math.ceil(k/group)
    assert 4*128+16*4 == 576 and F(576, 128) == F(9, 2)
    # K groups cannot merge across independent output channels.
    assert 4*1*3+16*3 != 4*1*3+16*math.ceil(1*3/32)
check("channel reconstruction and enumerated per-column tail groups validate the metadata budget", granularity_and_budget)

def accumulator_range():
    term, limit = (-128)*(-128), 2**31-1
    k = limit//term
    assert k == 131071
    assert struct.unpack("<i", struct.pack("<i", k*term))[0] == k*term
    for value in [(k+1)*term, k*term+16384]:
        try:
            struct.pack("<i", value)
        except struct.error:
            continue
        raise AssertionError("overflowing signed INT32 value accepted")
check("legal INT8 operands do not imply unbounded signed INT32 accumulation or bias addition", accumulator_range)

def ste_boundaries():
    for value in [-7.6, -7.2, -6.2, -0.2, 0.2, 6.2, 7.2, 7.6]:
        eps = 1e-5
        derivative = (fake_quant(value+eps)[0]-fake_quant(value-eps)[0])/(2*eps)
        assert derivative == 0.0
        expected_mask = -7.5 < value < 7.5
        assert fake_quant(value)[1] == expected_mask
        assert ste_dx(value, -3) == (-3 if expected_mask else 0)
    assert fake_quant(-7.5) == (-7, False) and fake_quant(7.5) == (7, False)
    assert fake_quant(math.nextafter(7.5, -math.inf)) == (7, True)
    assert fake_quant(math.nextafter(-7.5, math.inf)) == (-7, True)
    assert ste_dx(0.5) == 1
    # At an actual jump the finite-difference quotient does not approach 1.
    for eps in [0.01, 0.001]:
        jump = (fake_quant(0.5+eps)[0]-fake_quant(0.5-eps)[0])/(2*eps)
        assert jump == 1/(2*eps) and jump != ste_dx(0.5)
    for invalid in [math.nan, math.inf, -math.inf]:
        try:
            fake_quant(invalid)
        except ValueError:
            continue
        raise AssertionError("non-finite fake quant input accepted")
check("actual STE code differs from true local derivatives and handles both tie-adjacent mask edges", ste_boundaries)

print(f"{passed} quantization-foundations groups passed; actual stdlib blocks and rational/matrix oracles only, no PyTorch, packed kernel, GPU, QAT or model quality experiment.")
`;

const python = process.env.BLOG_PYTHON ?? (process.platform === 'win32' ? 'python' : 'python3');
const result = spawnSync(python, ['-c', program], {
  input: JSON.stringify(blocks), encoding: 'utf8',
  env: { ...process.env, PYTHONUTF8: '1' },
});
if (result.stdout) process.stdout.write(result.stdout);
if (result.stderr) process.stderr.write(result.stderr);
if (result.error) throw result.error;
assert.equal(result.status, 0, 'Actual foundations prose example verification failed');
