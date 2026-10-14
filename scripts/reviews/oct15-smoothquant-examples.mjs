import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';

const article = readFileSync(new URL('../../_posts/2026-08-02-smoothquant-w8a8-activation-outliers.md', import.meta.url), 'utf8');
function extract(name) {
  const parts = article.split(`<!-- example: smoothquant-${name} -->`);
  assert.equal(parts.length, 2, `Missing or duplicate SmoothQuant marker: ${name}`);
  const block = parts[1].match(/^\s*```python\r?\n([\s\S]*?)```/);
  assert.ok(block, `Missing actual Python block: ${name}`);
  return block[1];
}
const blocks = { two: extract('two-channel-int8'), folding: extract('shared-norm-folding') };
assert.equal([...article.matchAll(/^```python\s*$/gm)].length, 2, 'Exercise all actual SmoothQuant Python blocks');
assert.match(article, /^date: 2026-08-02 09:00:00 \+0800$/m, 'Preserve original publication date');
assert.match(article, /^author: iStar$/m);
assert.match(article, /^series_order: 50$/m);

// Actual manuscript code is executed unchanged. Independent oracles search
// rational reconstruction levels and multiply reconstructed rectangular
// matrices. No framework, official smoothing function or GPU is executed.
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
scopes, outputs = {}, {}
passed = 0
def check(name, action):
    global passed
    action()
    passed += 1
    print("PASS", name)

def close(actual, expected):
    assert math.isfinite(actual) and math.isclose(actual, float(expected), rel_tol=1e-11, abs_tol=1e-11), (actual, expected)

def transpose(a):
    return [list(row) for row in zip(*a)]

def multiply(a, b):
    return [[sum((F(x)*F(y) for x, y in zip(row, col)), F(0)) for col in transpose(b)] for row in a]

def actual_blocks():
    for name, source in blocks.items():
        ast.parse(source)
        scope, captured = {}, io.StringIO()
        with contextlib.redirect_stdout(captured):
            exec(compile(source, "article:smoothquant-"+name, "exec"), scope)
        scopes[name], outputs[name] = scope, ast.literal_eval(captured.getvalue().strip())
    assert outputs["two"]["factors"] == [10, 1]
    close(outputs["two"]["original"], 2)
    close(outputs["two"]["INT8_before"], F(260, 127))
    close(outputs["two"]["INT8_after"], 2)
    assert set(outputs["folding"]["original"]) == {"q", "k", "v"}
check("both actual manuscript blocks execute independently with their assertions and stated two-channel output", actual_blocks)
factor, quant = scopes["two"]["smoothing_factor"], scopes["two"]["qdq_int8"]

def rational_quantization_oracle():
    for scale in [F(1, 8), F(1, 2), F(1), F(8)]:
        positions = [F(i, 4) for i in range(-508, 509)]
        values = [float(scale*v) for v in positions]
        codes, restored, actual_scale = quant(values)
        assert F(actual_scale) == scale
        for value, code, rebuilt in zip(values, codes, restored):
            expected = min(range(-127, 128), key=lambda q: (abs(F(value)-scale*q), q % 2, q))
            assert code == expected
            assert F(rebuilt) == scale*expected
            assert abs(F(value)-F(rebuilt)) <= scale/2
    # Include +/-127 to keep a binary-exact unit scale while crossing ties.
    for integer in [-126, -3, -2, -1, 0, 1, 2, 125]:
        tie = integer+0.5
        for value in [math.nextafter(tie, -math.inf), tie, math.nextafter(tie, math.inf)]:
            code = quant([127.0, -127.0, value])[0][2]
            expected = min(range(-127, 128), key=lambda q: (abs(F(value)-q), q % 2, q))
            assert code == expected
    assert quant([0.0, -0.0]) == ([0, 0], [0, 0], 1.0)
check("the real Q/DQ function agrees with rational nearest-level/parity search at signed ties, neighbors and zero groups", rational_quantization_oracle)

def alpha_ranges_and_clamps():
    # Powers of two provide an independent exponent-domain oracle, rather than
    # copying a**alpha / b**(1-alpha) from the manuscript.
    for a_exp in [-4, 0, 4, 8]:
        for b_exp in [-4, 0, 4, 8]:
            a, b = 2.0**a_exp, 2.0**b_exp
            for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
                s = factor(a, b, alpha)
                close(s, 2.0**((a_exp+b_exp)*alpha-b_exp))
                close(a/s, 2.0**((a_exp+b_exp)*(1-alpha)))
                close(b*s, 2.0**((a_exp+b_exp)*alpha))
    assert factor(64, 0.01, 0) == 100
    close(factor(64, 0.01, 0.5), 80)
    assert factor(64, 0.01, 1) == 64
    assert factor(64, 0.01, 0.25) > factor(64, 0.01, 0.75)
    assert factor(64, 1, 0.25) < factor(64, 1, 0.75)
    for alpha in [0, 0.25, 0.5, 0.75, 1]:
        close(factor(4, 0.25, alpha), 4)  # ab=1 makes s constant.
    # The ideal positive-statistic identities do not extend through clamping.
    assert factor(0, 0, 0.5) == 1e-5
    assert factor(0, 0, 1) == 1e-5
    close(factor(0, 0, 0), 1e5)  # Explicit Python 0**0 = 1 convention.
    close(factor(1, 0, 0.5), math.sqrt(1e5))
    s = factor(1, 1e-12, 0.5)
    assert not math.isclose(1/s, 1e-12*s)
    # Any positive s still preserves a zero product in real arithmetic; zero
    # does not require asserting the unclamped range-balancing formula.
    assert F(0)/F(factor(0, 0, 0.5))*F(0)*F(factor(0, 0, 0.5)) == 0
check("alpha endpoints and ab<1 behavior match independent exponent identities while zero/epsilon cases remain separate", alpha_ranges_and_clamps)

def rectangular_invariance_and_gemm_scales():
    x = [[F(1), F(-2), F(3, 8)], [F(-1, 2), F(4), F(5, 4)]]
    w = [[F(1, 4), F(2), F(-1), F(3)], [F(1), F(-3, 4), F(2), F(0)],
         [F(-2), F(1, 2), F(3), F(1, 8)]]
    s = [F(8), F(1, 2), F(2)]
    xs = [[v/s[j] for j, v in enumerate(row)] for row in x]
    ws = [[v*s[i] for v in row] for i, row in enumerate(w)]
    assert multiply(xs, ws) == multiply(x, w)
    wrong_ws = [[v/s[i] for v in row] for i, row in enumerate(w)]
    assert multiply(xs, wrong_ws) != multiply(x, w)
    # Real manuscript quantizer supplies each input row and output column.
    qx, xhat, sx = [], [], []
    qw_columns, what_columns, sw = [], [], []
    for row in xs:
        codes, rebuilt, scale = quant(list(map(float, row)))
        qx.append(codes); xhat.append(rebuilt); sx.append(scale)
    for column in transpose(ws):
        codes, rebuilt, scale = quant(list(map(float, column)))
        qw_columns.append(codes); what_columns.append(rebuilt); sw.append(scale)
    qw, what = transpose(qw_columns), transpose(what_columns)
    direct = multiply(xhat, what)
    for m in range(len(xs)):
        for n in range(len(ws[0])):
            integer_accumulator = sum(qx[m][k]*qw[k][n] for k in range(len(ws)))
            # Independent reconstructed GEMM vs outer-dimension epilogue.
            expected = F(sx[m])*F(sw[n])*integer_accumulator
            close(float(direct[m][n]), expected)
    # Same integer sum and the same external row/output scales cannot encode
    # two different per-K scale products without additional information.
    codes = [[1, 0], [0, 1]]
    k_factors = [F(1), F(2)]
    assert [sum(row) for row in codes] == [1, 1]
    assert [sum(v*k_factors[j] for j, v in enumerate(row)) for row in codes] == [1, 2]
check("exact rectangular GEMM checks the K-axis transformation and reconstructed Q/DQ verifies outer scales versus per-K factors", rectangular_invariance_and_gemm_scales)

def folding_and_graph_boundaries():
    scope = scopes["folding"]
    z, gamma, beta = list(map(F, scope["z"])), list(map(F, scope["gamma"])), list(map(F, scope["beta"]))
    original_x = [v*g+b for v, g, b in zip(z, gamma, beta)]
    assert original_x == [F(5, 2), F(-7, 2)]
    for name, weight in scope["weights"].items():
        expected = sum(v*F(w) for v, w in zip(original_x, weight))+F(scope["biases"][name])
        close(scope["original"][name], expected)
        close(scope["folded"][name], expected)
    assert scope["wrong_beta_q"] != scope["original"]["q"]
    assert scope["wrong_k"] != scope["original"]["k"]
    close(scope["wrong_beta_x"][0], F(7, 10))
    close(scope["xs"][0], F(1, 4))
    # Multi-row affine output, with Q/K/V projections of different output widths.
    zrows = [[F(1), F(-1), F(1, 2)], [F(-2), F(1, 4), F(0)]]
    gamma, beta = [F(2), F(-3), F(1, 2)], [F(1, 2), F(1, 4), F(-1)]
    x = [[v*gamma[j]+beta[j] for j, v in enumerate(row)] for row in zrows]
    projections = [
        [[F(1), F(2)], [F(-1), F(1, 4)], [F(3), F(0)]],
        [[F(1, 2)], [F(4)], [F(-1)]],
        [[F(2), F(0), F(-1)], [F(1), F(2), F(1)], [F(0), F(-2), F(3)]],
    ]
    biases = [[F(1, 4), F(-1)], [F(2)], [F(0), F(1), F(-2)]]
    # Aggregate each input direction over all output coordinates/consumers.
    a = [max(abs(row[j]) for row in x) for j in range(3)]
    b = [max(abs(v) for w in projections for v in w[j]) for j in range(3)]
    s = list(map(F, [factor(float(aa), float(bb), 0.5) for aa, bb in zip(a, b)]))
    new_gamma, new_beta = [v/t for v, t in zip(gamma, s)], [v/t for v, t in zip(beta, s)]
    xs = [[v*new_gamma[j]+new_beta[j] for j, v in enumerate(row)] for row in zrows]
    assert xs == [[v/s[j] for j, v in enumerate(row)] for row in x]
    for w, bias in zip(projections, biases):
        ws = [[v*s[j] for v in row] for j, row in enumerate(w)]
        before, after = multiply(x, w), multiply(xs, ws)
        assert [[v+bias[j] for j, v in enumerate(row)] for row in before] == [[v+bias[j] for j, v in enumerate(row)] for row in after]
    # Omitting a shared consumer or modifying the current Linear bias breaks
    # the graph, despite a correct Q branch and positive scales.
    assert multiply(xs, projections[1]) != multiply(x, projections[1])
    assert [biases[0][j]/s[j] for j in range(2)] != biases[0]
    # A residual branch consuming the same X must also be accounted for.
    residual_w = [[v*s[j] for v in row] for j, row in enumerate(projections[2])]
    linear_before, linear_after = multiply(x, projections[2]), multiply(xs, residual_w)
    assert linear_before == linear_after
    assert [[v+x[i][j] for j, v in enumerate(row)] for i, row in enumerate(linear_before)] != [[v+xs[i][j] for j, v in enumerate(row)] for i, row in enumerate(linear_after)]
    # SiLU is not positively homogeneous; scaling cannot cross it arbitrarily.
    silu = lambda v: v/(1+math.exp(-v))
    assert not math.isclose(silu(2), 2*silu(1))
    # Scaling the input of LayerNorm is not scaling its affine output.
    norm = lambda row: [(v-math.fsum(row)/len(row))/math.sqrt(math.fsum((u-math.fsum(row)/len(row))**2 for u in row)/len(row)) for v in row]
    assert norm([0.5, 3]) != [v/t for v, t in zip(norm([1, 3]), [2, 1])]
check("actual shared-norm folding agrees with independent affine GEMM and multi-consumer/bias/residual/nonlinear counterexamples", folding_and_graph_boundaries)

def smoothing_is_not_error_monotonic():
    # Both vectors are exactly representable before smoothing. Alpha=1 keeps
    # them unchanged, while alpha=.5 balances ranges but adds output error.
    x, w = [1.0, 1.0], [127.0, 1.0]
    reference = 128.0
    quantized_dot = lambda a, b: math.fsum(v*t for v, t in zip(quant(a)[1], quant(b)[1]))
    assert quantized_dot(x, w) == reference
    results = {}
    for alpha in [0, 0.5, 1]:
        s = [factor(abs(a), abs(b), alpha) for a, b in zip(x, w)]
        xs, ws = [v/t for v, t in zip(x, s)], [v*t for v, t in zip(w, s)]
        close(math.fsum(v*t for v, t in zip(xs, ws)), reference)
        results[alpha] = quantized_dot(xs, ws)
    close(results[0], reference)
    close(results[1], reference)
    close(results[0.5], F(16250, 127))
    assert abs(results[0.5]-reference) > 0.04
    # In contrast, the manuscript's other matrix does become exactly
    # representable after smoothing. Neither outcome is universal.
    assert outputs["two"]["INT8_after"] == outputs["two"]["original"]
check("a different exactly representable input disproves guaranteed improvement or alpha=.5 optimality without contradicting the article's zero-error example", smoothing_is_not_error_monotonic)

def finite_scope_and_dtype():
    for args in [(math.nan, 1, 0.5), (1, math.inf, 0.5), (1, 1, math.nan),
                 (-1, 1, 0.5), (1, -1, 0.5), (1, 1, -0.1), (1, 1, 1.1)]:
        try:
            factor(*args)
        except ValueError:
            continue
        raise AssertionError(("invalid statistics accepted", args))
    for values in [[], [math.inf], [math.nan], [math.ulp(0.0)]]:
        try:
            quant(values)
        except ValueError:
            continue
        raise AssertionError(("unsupported Q/DQ values accepted", values))
    # This demonstrates a limitation stated beside the code, not an expected
    # guarantee of a general-purpose transform API.
    s = factor(1e308, 1e308, 1)
    assert math.isfinite(s) and math.isinf(s*1e308)
    assert s > 0 and 1e-308/s == 0
    # Python's binary64 reference is not FP16 arithmetic. Binary packing here
    # checks the representational limit only; no actual FP16 kernel executes.
    large = factor(0, 0, 0)
    assert large > 65504
    try:
        struct.pack("e", large)
    except OverflowError:
        pass
    else:
        raise AssertionError("half-precision overflow boundary not exposed")
check("actual toy guards and explicit arithmetic counterexamples bound zero/epsilon/dtype and transformed-intermediate claims", finite_scope_and_dtype)

assert passed == 7
print(json.dumps({"groups": passed, "scope": "actual manuscript Python and independent standard-library arithmetic; no official SmoothQuant, CUTLASS, model, GPU or serving experiment"}))
`;

const result = spawnSync(process.env.BLOG_PYTHON || 'python', ['-c', program], {
  input: JSON.stringify(blocks), encoding: 'utf8', maxBuffer: 4 * 1024 * 1024,
});
if (result.stdout) process.stdout.write(result.stdout);
if (result.stderr) process.stderr.write(result.stderr);
assert.equal(result.error, undefined, result.error?.message);
assert.equal(result.status, 0, 'SmoothQuant manuscript code or independent oracles failed');
