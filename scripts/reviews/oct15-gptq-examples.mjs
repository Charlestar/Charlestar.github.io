import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';

const article = readFileSync(new URL('../../_posts/2026-10-15-gptq-second-order-reconstruction.md', import.meta.url), 'utf8');
function extract(name) {
  const parts = article.split(`<!-- example: gptq-${name} -->`);
  assert.equal(parts.length, 2, `Missing or duplicate GPTQ example marker: ${name}`);
  const found = parts[1].match(/^\s*```python\r?\n([\s\S]*?)```/);
  assert.ok(found, `Missing actual Python block: ${name}`);
  return found[1];
}
const blocks = {
  two: extract('two-dimensional-compensation'),
  discrete: extract('discrete-counterexample'),
};
assert.equal([...article.matchAll(/^```python\s*$/gm)].length, 2, 'Exercise every Python block in this article');

// The subject is the manuscript's real code. Oracles use exact rational
// elimination, constrained derivatives and exhaustive integer residuals,
// not a second copy of the manuscript's inverse/factor calculation.
const program = String.raw`
import ast
import contextlib
from fractions import Fraction as F
import io
import itertools
import json
import math
import sys

blocks = json.load(sys.stdin)
scopes, outputs = {}, {}
passed = 0
def check(name, action):
    global passed
    action()
    passed += 1
    print("PASS", name)

def close(actual, expected, tolerance=1e-11):
    assert math.isfinite(actual) and math.isclose(actual, float(expected), rel_tol=tolerance, abs_tol=tolerance), (actual, expected)

def transpose(a):
    return [list(row) for row in zip(*a)]

def multiply(a, b):
    return [[sum((x*y for x, y in zip(row, col)), F(0)) for col in transpose(b)] for row in a]

def inverse(a):
    # Exact Gauss-Jordan with rational pivots, independent of the 2x2 direct
    # determinant formula and floating Cholesky used in the article.
    n = len(a)
    work = [[F(v) for v in row] + [F(i == j) for j in range(n)] for i, row in enumerate(a)]
    for q in range(n):
        pivot = next((i for i in range(q, n) if work[i][q]), None)
        if pivot is None:
            raise ValueError("singular rational matrix")
        work[q], work[pivot] = work[pivot], work[q]
        divisor = work[q][q]
        work[q] = [v/divisor for v in work[q]]
        for i in range(n):
            if i != q:
                ratio = work[i][q]
                work[i] = [v-ratio*u for v, u in zip(work[i], work[q])]
    return [row[n:] for row in work]

def trailing(a, q):
    return [row[q:] for row in a[q:]]

def upper(a):
    n = len(a)
    lower = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1):
            residual = float(a[i][j])-math.fsum(lower[i][k]*lower[j][k] for k in range(j))
            lower[i][j] = math.sqrt(residual) if i == j else residual/lower[j][j]
    return transpose(lower)

def grid(v):
    # Search distances to all reconstruction levels, with parity as tie-break.
    v = F(v)
    return min(range(-8, 8), key=lambda code: (abs(v-code), code % 2, code))

def loss(x, original, candidate):
    return sum((sum((F(v)*(q-w) for v, q, w in zip(row, candidate, original)), F(0)))**2 for row in x)

def quadratic(h, delta):
    return sum(delta[i]*F(h[i][j])*delta[j] for i in range(len(delta)) for j in range(len(delta)))/2

def raw_path(h, w, wrong_full_inverse=False):
    work, trace = list(map(F, w)), []
    full = inverse(h)
    for q in range(len(w)):
        pf = [row[q:] for row in full[q:]] if wrong_full_inverse else inverse(trailing(h, q))
        chosen = grid(work[q])
        residual = work[q]-chosen
        for j in range(q, len(w)):
            work[j] -= residual*pf[j-q][0]/pf[0][0]
        assert work[q] == chosen
        trace.append(work.copy())
    return work, trace

def actual_blocks():
    for name, source in blocks.items():
        ast.parse(source)
        scope, output = {}, io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(compile(source, "article:gptq-"+name, "exec"), scope)
        scopes[name], outputs[name] = scope, ast.literal_eval(output.getvalue().strip())
    assert outputs["two"]["quantized"] == [0, 1]
    close(outputs["two"]["SSE"], F(2503, 10000))
    assert sum((F(q)-F(49, 100))**2 for q in outputs["two"]["quantized"])/2 == F(2501, 10000)
    assert sum((F(0)-F(49, 100))**2 for _ in range(2))/2 == F(2401, 10000)
    assert outputs["discrete"]["global_best"] == [(-1, 0, 0)]
    close(outputs["discrete"]["sequential_SSE"], F(21, 25))
    close(outputs["discrete"]["RTN_SSE"], F(11, 25))
check("both actual manuscript blocks execute independently with their assertions and stated output", actual_blocks)
quantize, gptq_two = scopes["two"]["quantize_int4"], scopes["two"]["gptq_two"]

def fixed_grid_rounding():
    for quant in [quantize, scopes["discrete"]["quantize"]]:
        for numerator in range(-80, 81):
            value = F(numerator, 4)
            assert quant(float(value)) == grid(value)
        for integer in range(-9, 9):
            tie = integer+0.5
            for value in [math.nextafter(tie, -math.inf), tie, math.nextafter(tie, math.inf)]:
                assert quant(value) == grid(F(value)), (integer, value)
        assert quant(-sys.float_info.max) == -8 and quant(sys.float_info.max) == 7
    for value in [math.inf, -math.inf, math.nan]:
        try:
            quantize(value)
        except ValueError:
            continue
        raise AssertionError("nonfinite value accepted")
check("the real quantizers agree with nearest-level distance/parity oracles at ties, neighbors and saturation", fixed_grid_rounding)

def finite_difference_and_damping():
    # Evaluate the actual scalar reconstruction objective at rational offsets;
    # central finite differences are exact here because the objective is quadratic.
    fixtures = [
        (scopes["two"]["x"], scopes["two"]["h"]),
        (scopes["discrete"]["x"], scopes["discrete"]["h"]),
        ([[1, -2, 0], [0, 1, 3], [2, 0, -1], [1, 1, 1]], None),
    ]
    for x0, h0 in fixtures:
        # The 2D sqrt(3)/2 input is binary floating, so its Gram agrees within
        # floating tolerance rather than pretending sqrt(3) is rational.
        x = [[F(v) for v in row] for row in x0]
        h = [[2*v for v in row] for row in multiply(transpose(x), x)]
        if h0 is not None:
            for i, row in enumerate(h):
                for j, value in enumerate(row):
                    close(h0[i][j], value)
        n, step = len(h), F(1, 8)
        origin = [F(i-1, 4) for i in range(n)]
        def objective(v):
            return loss(x, [F(0)]*n, v)
        def point(i, si, j=None, sj=0):
            v = origin.copy()
            v[i] += si*step
            if j is not None:
                v[j] += sj*step
            return objective(v)
        for i in range(n):
            assert (point(i, 1)-2*objective(origin)+point(i, -1))/step**2 == h[i][i]
            for j in range(i):
                mixed = (point(i, 1, j, 1)-point(i, 1, j, -1)-point(i, -1, j, 1)+point(i, -1, j, -1))/(4*step**2)
                assert mixed == h[i][j]
        lam = F(3, 16)
        damped = [[v+(lam if i == j else 0) for j, v in enumerate(row)] for i, row in enumerate(h)]
        assert quadratic(damped, origin) == objective(origin)+lam*sum(v*v for v in origin)/2
    # The undamped calibration [1,1] is rank deficient; damping changes the
    # objective and enables the finite 2D demonstration, not the original inverse.
    try:
        gptq_two([[2.0, 2.0], [2.0, 2.0]], [0.49, 0.49])
    except ValueError:
        pass
    else:
        raise AssertionError("singular Hessian accepted")
    assert gptq_two([[2.1, 2.0], [2.0, 2.1]], [0.49, 0.49])[0] == [0, 1]
    # Algebraic accumulation corresponding to the pinned add_batch recurrence.
    # This executes the recurrence, NOT official PyTorch add_batch.
    batches = [([[1, 2], [3, 4]], 1), ([[2, -1], [1, 0], [-2, 3], [1, -1]], 2)]
    buffer, count, collected = [[F(0), F(0)], [F(0), F(0)]], 0, []
    for values, tmp in batches:
        gram = multiply(transpose(values), values)
        buffer = [[v*F(count, count+tmp)+2*F(gram[i][j], count+tmp) for j, v in enumerate(row)] for i, row in enumerate(buffer)]
        count += tmp
        collected.extend(values)
    expected = multiply(transpose(collected), collected)
    assert buffer == [[2*F(v, count) for v in row] for row in expected]
    assert count == 3 != len(collected)
check("finite differences verify the layer Hessian, lambda/2 and the separate batch-normalization count", finite_difference_and_damping)

def continuous_minimum_and_two_dimensional_paths():
    cases = 0
    for a, b, c in itertools.product([1, 2, 5], [1, 3, 7], [-2, -1, 0, 1, 2]):
        if a*b <= c*c:
            continue
        h = [[a, c], [c, b]]
        for w in [[F(49, 100), F(49, 100)], [F(-3, 8), F(9, 8)], [F(37, 4), F(-39, 4)]]:
            actual, trace, p, u = gptq_two(h, list(map(float, w)))
            chosen = grid(w[0])
            delta0 = chosen-w[0]
            # Different oracle: minimize directly in the remaining variable.
            delta1 = -F(c, b)*delta0
            expected_middle = [F(chosen), w[1]+delta1]
            close(trace[0][0], expected_middle[0])
            close(trace[0][1], expected_middle[1])
            assert actual == [chosen, grid(expected_middle[1])]
            assert F(c)*delta0+F(b)*delta1 == 0
            minimum = quadratic(h, [delta0, delta1])
            for shift in [F(-1), F(-1, 8), F(1, 8), F(1)]:
                assert quadratic(h, [delta0, delta1+shift])-minimum == F(b)*shift**2/2
            exact_p = inverse(h)
            for i in range(2):
                for j in range(2):
                    close(p[i][j], exact_p[i][j])
                    close(sum(u[k][i]*u[k][j] for k in range(2)), exact_p[i][j])
            close(u[1][1]**2, F(1, b))
            assert minimum == (w[0]-chosen)**2/(2*exact_p[0][0])
            # Positive scalar Hessian rescaling preserves the undamped path.
            for scale in [F(1, 16), F(16)]:
                scaled = [[float(v*scale) for v in row] for row in h]
                scaled_result, scaled_trace, _, _ = gptq_two(scaled, list(map(float, w)))
                assert scaled_result == actual
                for left, right in zip(scaled_trace, trace):
                    for v, expected in zip(left, right):
                        close(v, expected)
            cases += 1
    assert cases > 0
    print("  2D SPD/weight fixtures:", cases)
check("actual 2D trajectories satisfy independent constrained derivatives and positive Hessian-scale invariance", continuous_minimum_and_two_dimensional_paths)

def current_inverse_schur_and_wrong_full_p():
    sample = scopes["discrete"]
    h = sample["h"]
    p, u = inverse(h), sample["u"]
    exact, trace = raw_path(h, [F(-2, 5), F(-2, 5), F(1, 5)])
    for q in range(3):
        pf = inverse(trailing(h, q))
        gram = multiply(transpose(trailing(u, q)), trailing(u, q))
        for i in range(len(pf)):
            for j in range(len(pf)):
                close(gram[i][j], pf[i][j])
        close(u[q][q]**2, pf[0][0])
        for actual, expected in zip(sample["trace"][q], trace[q]):
            close(actual, expected)
        if q+1 < 3:
            schur = [[pf[i][j]-pf[i][0]*pf[0][j]/pf[0][0] for j in range(1, len(pf))] for i in range(1, len(pf))]
            assert schur == inverse(trailing(h, q+1))
    assert inverse(trailing(h, 1))[0][0] == F(3, 4) != p[1][1]
    assert inverse(trailing(h, 2))[0][0] == F(1, 6) != p[2][2]
    assert exact == [0, -1, 0]
    # An independently chosen SPD example makes reusing the original P fail
    # at a later rounding decision, not merely differ by floating roundoff.
    x = [[1, 1, 0], [0, 1, 1], [1, 0, 1]]
    h = [[2*v for v in row] for row in multiply(transpose(x), x)]
    w = [F(49, 100)]*3
    correct, _ = raw_path(h, w)
    wrong, _ = raw_path(h, w, wrong_full_inverse=True)
    assert correct == [0, 1, 0] and wrong == [0, 1, 1]
    assert loss(x, w, correct) == F(2403, 2500)
    assert loss(x, w, wrong) == F(2603, 2500)
check("exact current-free-set inverses verify Schur and upper-factor semantics while a different SPD case rejects stale full-P reuse", current_inverse_schur_and_wrong_full_p)

def lazy_blocks_and_layout():
    # Non-square KxN weights and an incomplete block tail exercise the exact
    # transpose in the article. This is arithmetic, not an exporter/kernel.
    x = [[1, 2, 0, -1, 1], [0, 1, 1, 0, -1], [2, 0, 1, 1, 0],
         [1, -1, 0, 2, 1], [0, 1, -1, 1, 2], [1, 0, 2, -1, 1]]
    h = [[2*v for v in row] for row in multiply(transpose(x), x)]
    w = [[F(3*i-2*j-5, 16) for j in range(3)] for i in range(5)]
    expected = transpose([raw_path(h, column)[0] for column in transpose(w)])
    def blocked(h, w, size):
        u, work = upper(inverse(h)), [list(map(float, row)) for row in w]
        n, outputs = len(work), len(work[0])
        result = [[0]*outputs for _ in range(n)]
        for start in range(0, n, size):
            end = min(start+size, n)
            errors = []
            for q in range(start, end):
                chosen = [grid(F(v)) for v in work[q]]
                error = [(v-code)/u[q][q] for v, code in zip(work[q], chosen)]
                errors.append(error)
                result[q] = chosen
                for k in range(q, end):
                    work[k] = [v-u[q][k]*e for v, e in zip(work[k], error)]
                work[q] = list(map(float, chosen))
            # U[B,tail]^T (tail x B) times E (B x N).
            update = multiply(transpose([row[end:] for row in u[start:end]]), errors) if end < n else []
            for k in range(end, n):
                work[k] = [v-update[k-end][j] for j, v in enumerate(work[k])]
        return result
    for size in [1, 2, 3, 128]:
        assert blocked(h, w, size) == expected
    perm = [3, 0, 4, 1, 2]
    hp = [[h[i][j] for j in perm] for i in perm]
    wp = [w[i] for i in perm]
    xp = [[row[i] for i in perm] for row in x]
    assert multiply(xp, wp) == multiply(x, w)
    qp = blocked(hp, wp, 2)
    assert qp == transpose([raw_path(hp, column)[0] for column in transpose(wp)])
    restored = [None]*len(perm)
    for ordered, original in enumerate(perm):
        restored[original] = qp[ordered]
    assert multiply(x, restored) == multiply(xp, qp)
check("KxN lazy block updates match independent rational per-output paths including partial blocks and restored permutations", lazy_blocks_and_layout)

def exact_discrete_enumeration():
    sample = scopes["discrete"]
    scores = []
    for a, b, c in itertools.product(range(-8, 8), repeat=3):
        # Integer residual numerators for w=(-2/5,-2/5,1/5).
        da, db, dc = 5*a+2, 5*b+2, 5*c-1
        numerator = (da+db+dc)**2+(db+dc)**2+dc**2
        scores.append((numerator, (a, b, c)))
        close(sample["sse"]([a, b, c]), F(numerator, 25))
    best = min(score for score, _ in scores)
    minimizers = [candidate for score, candidate in scores if score == best]
    assert len(scores) == 4096 and best == 6 and minimizers == [(-1, 0, 0)]
    assert sample["best"] == minimizers
    close(sample["best_loss"], F(best, 25))
    assert [loss(sample["x"], [F(-2, 5), F(-2, 5), F(1, 5)], row) for row in
            [[0, F(-3, 5), F(1, 5)], [0, -1, F(7, 15)], [0, -1, 0]]] == [F(2, 25), F(14, 75), F(21, 25)]
    assert best < 11 < 21  # Exact global < RTN < sequential numerators.
check("all 4096 integer-residual candidates establish the exact unique optimum and the stated worse-than-RTN boundary", exact_discrete_enumeration)

def numerical_contract():
    bad = [
        ([], [0, 0]), ([[1]], [0, 0]), ([[1, 0], [0, 1]], [0]),
        ([[2, 1], [0, 2]], [0, 0]),
        ([[math.nan, 0], [0, 1]], [0, 0]), ([[math.inf, 0], [0, 1]], [0, 0]),
        ([[1, 0], [0, 1]], [math.inf, 0]), ([[1, 0], [0, 1]], [math.nan, 0]),
        ([[-1, 0], [0, 1]], [0, 0]), ([[1, 2], [2, 1]], [0, 0]),
        ([[1, 1], [1, 1]], [0, 0]), ([[0, 0], [0, 0]], [0, 0]),
        ([[1e200, 0], [0, 1e200]], [0, 0]),
        ([[1e-200, 0], [0, 1e-200]], [0, 0]),
        ([[1e-309, 0], [0, 1]], [0, 0]),
        ([[0.25, 0.5], [0.5, math.nextafter(1.0, math.inf)]], [0.49, 0.49]),
        ([[1e150, 1e-100], [1e-100, 1e150]], [0, 0]),
        ([[1e-154, math.ulp(0.0)], [math.ulp(0.0), 1e154]], [0, 0]),
        ([[1e-150, 0.5e-150], [0.5e-150, 1e-150]], [1e-300, 1e-300]),
        ([[1e150, 0], [0, 1e150]], [1e308, 0]),
        ([[1, 1e-150], [1e-150, 1]], [1e-320, 0]),
        ([[1e150, 0.5], [0.5, 1e-150]], [1e200, 0]),
        ([[1e150, 0.5], [0.5, 1e-150]], [2e158, 1.5e308]),
    ]
    for h, w in bad:
        try:
            gptq_two(h, w)
        except ValueError:
            continue
        raise AssertionError(("unsupported numeric case accepted", h, w))
    assert gptq_two([[2, 0], [0, 2]], [0.0, -0.0])[0] == [0, 0]
    assert gptq_two([[1e150, 0], [0, 1e-150]], [1, 0.49])[0] == [1, 0]
    # The guarded intermediate underflow does not mean the mathematical
    # compensation is zero: direct rational minimization gives 5e-301.
    c, b = F(0.5e-150), F(1e-150)
    assert -c/b*(-F(1e-300)) > 0
    assert float(-c/b*(-F(1e-300))) == 5e-301
check("the real 2D reference rejects unsupported shapes/SPD/floating ranges including recoverable-result intermediate underflow", numerical_contract)

assert passed == 8
print(json.dumps({"groups": passed, "scope": "actual manuscript Python and independent standard-library arithmetic only; no official GPTQ, packing, GPU or LLM experiment"}))
`;

const python = process.env.BLOG_PYTHON || 'python';
const result = spawnSync(python, ['-c', program], {
  input: JSON.stringify(blocks), encoding: 'utf8', maxBuffer: 4 * 1024 * 1024,
});
if (result.stdout) process.stdout.write(result.stdout);
if (result.stderr) process.stderr.write(result.stderr);
assert.equal(result.error, undefined, result.error?.message);
assert.equal(result.status, 0, 'GPTQ manuscript examples or independent oracles failed');
