import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';

const article = readFileSync(new URL('../../_posts/2026-08-03-awq-activation-aware-weight-quantization.md', import.meta.url), 'utf8');
function extract(name) {
  const parts = article.split(`<!-- example: awq-${name} -->`);
  assert.equal(parts.length, 2, `Missing or duplicate AWQ marker: ${name}`);
  const block = parts[1].match(/^\s*```python\r?\n([\s\S]*?)```/);
  assert.ok(block, `Missing actual AWQ Python: ${name}`);
  return block[1];
}
const blocks = { scaling: extract('channel-scaling-and-grid'), clipping: extract('clipping-local-versus-global'), folding: extract('scaled-activation-folding') };
assert.equal([...article.matchAll(/^```python\s*$/gm)].length, 3, 'Exercise every AWQ Python block');
assert.match(article, /^date: 2026-08-03 09:00:00 \+0800$/m);
assert.match(article, /^author: iStar$/m);
assert.match(article, /^series_order: 70$/m);

// Test the actual manuscript, not a rewritten implementation. Fraction grids,
// Decimal forward search, full candidate enumeration and reconstructed matrix
// products provide different oracles. Source-layout checks below are numerical
// contracts, not execution of the external Torch/CUDA packing implementation.
const program = String.raw`
import ast
import contextlib
from decimal import Decimal as D, localcontext, ROUND_HALF_EVEN
from fractions import Fraction as F
import io
import itertools
import json
import math
import sys

blocks = json.load(sys.stdin)
scopes, printed = {}, {}
passed = 0
def check(name, action):
    global passed
    action()
    passed += 1
    print("PASS", name)

def close(actual, expected, rel_tol=1e-11, abs_tol=1e-11):
    assert math.isfinite(actual) and math.isclose(actual, float(expected), rel_tol=rel_tol, abs_tol=abs_tol), (actual, expected)

def transpose(a):
    return [list(row) for row in zip(*a)]

def multiply(a, b):
    return [[sum((F(x)*F(y) for x, y in zip(row, col)), F(0)) for col in transpose(b)] for row in a]

def exact_qdq(values, bound=7, threshold=None):
    values = list(map(F, values))
    peak = max(map(abs, values)) if threshold is None else F(threshold)
    if peak == 0:
        return [F(0)]*len(values)
    step = peak/bound
    return [step*min(range(-bound, bound+1), key=lambda code: (abs(value-step*code), code % 2, code)) for value in values]

def actual_blocks():
    for name, source in blocks.items():
        ast.parse(source)
        scope, captured = {}, io.StringIO()
        with contextlib.redirect_stdout(captured):
            exec(compile(source, "article:awq-"+name, "exec"), scope)
        scopes[name], printed[name] = scope, captured.getvalue()
        assert printed[name].strip()
    assert len(scopes["scaling"]["history"]) == 20
    close(scopes["scaling"]["best"][1], F(17, 20))
    close(sum(scopes["clipping"]["baseline"]), F(1, 2))
    close(sum(scopes["clipping"]["selected"]), F(167, 350))
    assert len(scopes["folding"]["reference"]) == 2
check("all three real manuscript blocks execute independently with their assertions and recorded outputs", actual_blocks)
sym, scaled = scopes["scaling"]["sym_qdq"], scopes["scaling"]["scaled_forward"]
clip = scopes["clipping"]["quantize_clipped"]

def rational_grid_and_numeric_domain():
    for step in [F(1, 8), F(1, 2), F(1), F(8)]:
        values = [float(step*F(i, 8)) for i in range(-56, 57)]
        codes, restored, actual_step = sym(values)
        assert actual_step == step
        expected = exact_qdq(values)
        assert list(map(F, restored)) == expected
        for value, rebuilt in zip(values, restored):
            assert abs(F(value)-F(rebuilt)) <= step/2
        for integer in range(-7, 7):
            tie = float(step*(F(integer)+F(1, 2)))
            for value in [math.nextafter(tie, -math.inf), tie, math.nextafter(tie, math.inf)]:
                data = [-float(7*step), float(7*step), value]
                assert F(sym(data)[1][-1]) == exact_qdq(data)[-1]
    assert sym([0, -0.0]) == ([0, 0], [0, 0], 1)
    for values in [[], [math.nan], [math.inf], [math.ulp(0.0)]]:
        try:
            sym(values)
        except ValueError:
            continue
        raise AssertionError(("unsupported group accepted", values))
    bad = [([], [], []), ([1], [1, 2], [1]), ([math.nan], [1], [1]),
           ([1], [1], [0]), ([1], [1], [-1]), ([1], [1], [math.inf]),
           ([1e308], [1], [1e-308]),
           ([1e-300], [1e100], [1e100]),
           ([1e100], [1e-300], [1e-100]),
           ([1], [1e308], [1e308]),
           ([1e-200], [1e-200], [1]),
           ([1e308, 1e308], [1, 1], [1, 1])]
    for x, w, s in bad:
        try:
            scaled(x, w, s)
        except ValueError:
            continue
        raise AssertionError(("unsupported transform accepted", x, w, s))
check("real symmetric Q/DQ matches rational level search and rejects unsupported transforms, products and floating ranges", rational_grid_and_numeric_domain)

def matrix_error_and_group_coupling():
    x = [[F(2), F(-1), F(1, 2)], [F(-1), F(3), F(2)]]
    w = [[F(1, 4), F(1)], [F(-1, 2), F(1, 8)], [F(2), F(-1)]]
    s = [F(4), F(1, 2), F(2)]
    xs, ws = [[v/s[j] for j, v in enumerate(row)] for row in x], [[v*s[i] for v in row] for i, row in enumerate(w)]
    assert multiply(xs, ws) == multiply(x, w)
    reconstructed = transpose([exact_qdq(column) for column in transpose(w)])
    ew = [[v-reconstructed[i][j] for j, v in enumerate(row)] for i, row in enumerate(w)]
    output_error = multiply(x, ew)
    sse = sum(v*v for row in output_error for v in row)
    xtx = multiply(transpose(x), x)
    proxy = multiply(multiply(transpose(ew), xtx), ew)
    assert sse == sum(proxy[i][i] for i in range(len(proxy)))
    # Equal meanabs does not encode cross-channel cancellation.
    error = [[F(1)], [F(-1)]]
    assert multiply([[1, 1]], error) == [[0]]
    assert multiply([[1, -1]], error) == [[2]]
    for scale, expected in [(1, F(17, 7)), (10, F(2)), (100, F(17, 7))]:
        y, _, step = scaled([10.0, 1.0], [0.1, 1.0], [scale, 1])
        close(y, expected)
        close(step, F(1, 7) if scale <= 10 else F(10, 7))
    # Common scaling leaves this ideal grid unchanged in original weight units.
    original = [F(1, 10), F(1)]
    for common in [F(1, 16), F(1, 10), F(3), F(16)]:
        assert [v/common for v in exact_qdq([v*common for v in original])] == exact_qdq(original)
    # Check the transformed single-element bound, not a claim of actual
    # error monotonicity for every member of a moving group grid.
    for s in [[F(1), F(1)], [F(10), F(1)], [F(100), F(1)]]:
        transformed = [v*t for v, t in zip(original, s)]
        step = max(map(abs, transformed))/7
        rebuilt = exact_qdq(transformed)
        for v, q, t in zip(original, rebuilt, s):
            assert abs(v-q/t) <= step/(2*t)
check("independent rectangular GEMM verifies the output-error proxy, K-axis equivalence and group-step coupling/bounds", matrix_error_and_group_coupling)

def independent_forward_grid_search():
    # High-precision Decimal evaluates the UNNORMALIZED scale family.
    # The common normalizer cancels algebraically; this differs from the
    # article's binary64 normalized candidates and Python round.
    with localcontext() as ctx:
        ctx.prec = 60
        independent = []
        for i in range(20):
            ratio = D(i)/20
            relative = D(10)**ratio
            weights = [relative/10, D(1)]
            step = max(weights)/7
            codes = [int((v/step).to_integral_value(rounding=ROUND_HALF_EVEN)) for v in weights]
            prediction = D(10)/relative*step*codes[0]+step*codes[1]
            independent.append(((prediction-2)**2, ratio, prediction))
        expected_best = min(independent, key=lambda row: row[0])
        assert expected_best[1] == D("0.85")
        assert sum(row[0] == expected_best[0] for row in independent) == 1
        for actual, expected in zip(scopes["scaling"]["history"], independent):
            for a, b in zip(actual, expected):
                close(a, b)
        assert expected_best[0] > 0 and scaled([10.0, 1.0], [0.1, 1.0], [10, 1])[0] == 2
    assert scopes["scaling"]["x"] == [10, 1] and scopes["scaling"]["w"] == [0.1, 1]
    x, w = [10.0, 1.0], [0.1, 1.0]
    original = x.copy(), w.copy()
    for i in range(20):
        raw = [max(abs(v)**(i/20), 1e-4) for v in x]
        normalized = [v/math.sqrt(max(raw)*min(raw)) for v in raw]
        scaled(x, w, normalized)
    assert (x, w) == original
    raw = [1e-4, 1e12]
    normalized = [v/math.sqrt(max(raw)*min(raw)) for v in raw]
    assert min(normalized) < 1e-4
    assert [max(0.0**0, 1e-4)]*2 == [1, 1]
check("an independent 60-digit unnormalized forward search selects the same 20-candidate optimum without calling it a global optimum", independent_forward_grid_search)

def local_clipping_and_global_objectives():
    groups = [([F(1), F(2)], [F(1, 10), F(1, 5)]),
              ([F(-1), F(1, 2)], [F(1, 10), F(1, 5)])]
    factors = [F(1)-F(i, 20) for i in range(10)]
    records = []
    for x, w in groups:
        target = sum(a*b for a, b in zip(x, w))
        candidates = []
        for factor in factors:
            threshold = max(map(abs, w))*factor
            expected = exact_qdq(w, threshold=threshold)
            actual = clip(list(map(float, w)), float(factor))
            for a, b in zip(actual, expected):
                close(a, b)
            prediction = sum(a*b for a, b in zip(x, expected))
            candidates.append(((prediction-target)**2, factor, prediction-target))
        records.append(candidates)
    selected = [min(rows, key=lambda row: row[0]) for rows in records]
    assert [row[1] for row in selected] == [F(19, 20), F(4, 5)]
    assert [row[2] for row in selected] == [F(-2, 175)]*2
    assert [rows[0][2] for rows in records] == [F(1, 70), F(-1, 70)]
    before = [rows[0][2] for rows in records]
    after = [row[2] for row in selected]
    assert sum(v*v for v in after) < sum(v*v for v in before)
    assert sum(before)**2 == 0 < sum(after)**2
    # All 100 joint choices expose the difference from independent group MSE.
    joint = [(a[2]+b[2])**2 for a, b in itertools.product(*records)]
    assert min(joint) == 0 and sum(after)**2 > min(joint)
    for errors in [before, after]:
        assert sum(errors)**2 == sum(v*v for v in errors)+2*errors[0]*errors[1]
    assert clip([0, 0], 0.55) == [0, 0]
    for group, factor in [([], 1), ([math.nan], 1), ([1], 0), ([1], 1.1),
                          ([1], math.inf), ([1e-300], 1e-300), ([math.ulp(0.0)], 1)]:
        try:
            clip(group, factor)
        except ValueError:
            continue
        raise AssertionError(("unsupported clipping input accepted", group, factor))
check("exact rational candidate and joint-grid enumeration confirms local clipping improvements can destroy global cancellation", local_clipping_and_global_objectives)

def nonlinear_folding():
    scope = scopes["folding"]
    h, scales, weights = scope["h"], scope["scales"], scope["weights"]
    # erfc and independent reconstructed GEMM, not the article's erf/linear.
    activation = [[0.5*v*math.erfc(-v/math.sqrt(2)) for v in h]]
    expected = multiply(activation, weights)[0]
    for actual in [scope["reference"], scope["correct"]]:
        for a, b in zip(actual, expected):
            close(a, b)
    assert max(abs(a-b) for a, b in zip(scope["wrong"], scope["correct"])) > 0.01
    # Move inverse scaling AFTER the nonlinear output at general KxN shape.
    rows = [[-2, 0, 0.5], [1, -0.5, 2]]
    weights = [[F(1), F(2)], [F(-1), F(1, 2)], [F(3), F(-1)]]
    s = [F(2), F(1, 4), F(4)]
    activation = [[F(0.5*v*math.erfc(-v/math.sqrt(2))) for v in row] for row in rows]
    xs = [[v/s[j] for j, v in enumerate(row)] for row in activation]
    ws = [[v*s[j] for v in row] for j, row in enumerate(weights)]
    assert multiply(xs, ws) == multiply(activation, weights)
    # A gate is not generally homogeneous either; scaling the linear 'up'
    # branch after a fixed gate is different from rescaling its SiLU input.
    silu = lambda v: v/(1+math.exp(-v))
    gate, up, scale = 1.0, 3.0, 2.0
    close(silu(gate)*(up/scale)*scale, silu(gate)*up)
    assert not math.isclose(silu(gate/scale)*up*scale, silu(gate)*up)
check("actual GELU folding matches independent erfc/GEMM and explicit nonlinear/gated counterexamples", nonlinear_folding)

def source_numeric_contract_and_metadata():
    # Numerical contract read from pinned quantizer.py: round normalized w,
    # THEN add integer zero. This is not execution of the Torch function.
    def source_qdq(w):
        w = list(map(F, w))
        step = max(max(w)-min(w), F(1, 100000))/15
        rounded_zero = min(range(-100, 101), key=lambda z: (abs(-min(w)/step-z), z % 2, z))
        zero = max(0, min(15, rounded_zero))
        codes = []
        for value in w:
            normalized = value/step
            base = normalized.numerator//normalized.denominator
            nearest = min(range(base-2, base+3), key=lambda n: (abs(normalized-n), n % 2, n))
            codes.append(max(0, min(15, nearest+zero)))
        return step, zero, codes
    step, zero, codes = source_qdq([-1, F(1, 2), 14])
    assert step == 1 and zero == 1 and codes == [0, 1, 15]
    assert round(F(1, 2)+zero) == 2 != codes[1]
    step, zero, codes = source_qdq([0, 0])
    assert step > 0 and zero == 0 and codes == [0, 0]
    for step, zero in [(F(1, 8), 3), (F(1, 2), 15)]:
        scaled_zero = -step*zero
        for code in range(16):
            assert step*code+scaled_zero == step*(code-zero)
    assert F(4)+F(16+16, 128) == F(17, 4)
    assert F(16)/F(17, 4) == F(64, 17)
    # Source qmodule uses int16 qweight, four output rows interleaved.
    # Count its padded buffers from the pinned shapes; no '8 nibbles/int32'
    # storage assumption and no claim that this executes pack_intweight.
    def buffers(k, n, group, meta_bytes=2):
        assert k % group == 0 and k % 64 == 0 and n % 8 == 0
        qweight_elements = (n//4)*k
        records = list(range(0, k, group))
        multiplier = 1 if group >= 128 else 2 if group == 64 else 4
        # Each metadata stride covers 8 groups, then is rounded to the
        # source's size multiplier (2 for G64, 4 for G32).
        strides = (len(records)+7)//8
        strides = ((strides+multiplier-1)//multiplier)*multiplier
        padded_groups = strides*8
        return qweight_elements*2, padded_groups*n*meta_bytes*2
    assert buffers(1024, 8, 128) == (4096, 256)
    assert F(sum(buffers(1024, 8, 128))*8, 1024*8) == F(17, 4)
    assert buffers(128, 8, 128) == (512, 256)
    assert F(sum(buffers(128, 8, 128))*8, 128*8) == 6  # Padding, not 4.25.
    assert buffers(192, 8, 64) == (768, 512)
    assert buffers(128, 8, 32) == (512, 1024)
    assert buffers(1024, 8, 128, meta_bytes=4) == (4096, 512)
check("fixed-source rounding/zero-offset algebra and actual int16/padded-buffer byte contracts support the conditional 4.25-bit budget", source_numeric_contract_and_metadata)

assert passed == 7
print(json.dumps({"groups": passed, "scope": "actual manuscript Python plus independent standard-library arithmetic and source-layout budgets; no official AWQ calibration/packing, Torch, TinyChat, GPU or model-quality experiment"}))
`;

const result = spawnSync(process.env.BLOG_PYTHON || 'python', ['-c', program], {
  input: JSON.stringify(blocks), encoding: 'utf8', maxBuffer: 4 * 1024 * 1024,
});
if (result.stdout) process.stdout.write(result.stdout);
if (result.stderr) process.stderr.write(result.stderr);
assert.equal(result.error, undefined, result.error?.message);
assert.equal(result.status, 0, 'AWQ manuscript examples or independent oracles failed');
