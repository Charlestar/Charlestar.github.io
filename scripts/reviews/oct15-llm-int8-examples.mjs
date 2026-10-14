import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';

const article = readFileSync(new URL('../../_posts/2026-10-15-llm-int8-outlier-decomposition.md', import.meta.url), 'utf8');
function extract(name) {
  const parts = article.split(`<!-- example: llm-int8-${name} -->`);
  assert.equal(parts.length, 2, `Missing or repeated example marker: ${name}`);
  const found = parts[1].match(/^\s*```python\r?\n([\s\S]*?)```/);
  assert.ok(found, `Missing Python block after ${name}`);
  return found[1];
}
const payload = { single: extract('single-row'), matrix: extract('matrix-decomposition') };
assert.equal([...article.matchAll(/^```python\s*$/gm)].length, 2, 'All prose Python examples must execute');

const program = String.raw`
import ast
import contextlib
from fractions import Fraction as F
import io
import json
import math
import sys

data = json.load(sys.stdin)
scopes = {}
printed = {}
passed = 0
def check(name, action):
    global passed
    action()
    passed += 1
    print("PASS", name)

def close(actual, expected):
    assert math.isfinite(actual) and math.isclose(actual, float(expected), rel_tol=1e-12, abs_tol=1e-12), (actual, expected)

def actual_blocks():
    for key, code in data.items():
        ast.parse(code)
        namespace, out = {}, io.StringIO()
        with contextlib.redirect_stdout(out):
            exec(compile(code, "article:"+key, "exec"), namespace)
        scopes[key], printed[key] = namespace, ast.literal_eval(out.getvalue().strip())
    assert printed["single"]["codes"] == [0, 0, 127]
    close(printed["single"]["original"], F(10031, 100))
    close(printed["single"]["int8"], 100)
    assert printed["matrix"]["outlier_cols"] == [2]
    assert printed["matrix"]["outlier"] == [[100], [0.5]]
    close(printed["matrix"]["output"][0][0], F(100)+F(197, 635))
check("both actual article blocks execute independently with their published outputs", actual_blocks)
mixed = scopes["matrix"]["mixed_mm"]
quant = scopes["matrix"]["quant_vector"]
branch = scopes["matrix"]["int8_branch"]

def nearest(value, scale):
    value, scale = F(value), F(scale)
    # Compare all 255 reconstructed levels, rather than calling round.
    return min(range(-127, 128), key=lambda code: (abs(scale*code-value), code % 2, code))

def rational_vector_oracle():
    for scale in [F(1, 4), F(1), F(2)]:
        for sign in [-1, 1]:
            for value in [F(-253, 2), F(-3, 2), F(-1, 2), F(0), F(1, 2), F(3, 2), F(253, 2)]:
                vector = [sign*127*scale, value*scale, F(0)]
                codes, actual_scale = quant([float(x) for x in vector])
                assert F(actual_scale) == scale
                assert codes == [nearest(x, scale) for x in vector]
    assert quant([]) == ([], 1) and quant([0, 0]) == ([0, 0], 1)
    x = [[127.0, 1.5, -2.5, 0], [0.5, -63.5, 1.5, 0]]
    w = [[127.0, 1, 0.5], [1.5, 254, -1.5], [-2.5, 3, 63.5], [0, 0, 1]]
    xsteps = [F(1), F(1, 2)]
    wsteps = [F(1), F(2), F(1, 2)]
    decoded_x = [[xsteps[i]*nearest(v, xsteps[i]) for v in row] for i, row in enumerate(x)]
    decoded_w = [[wsteps[n]*nearest(w[k][n], wsteps[n]) for n in range(3)] for k in range(4)]
    actual = branch(x, w, list(range(4)))
    for i in range(2):
        for n in range(3):
            expected = sum(decoded_x[i][k]*decoded_w[k][n] for k in range(4))
            close(actual[i][n], expected)
check("rational nearest-grid search and reconstructed rectangular GEMM verify row/column scale broadcasting", rational_vector_oracle)

def partition_oracle():
    x = [[0.11, 0.2, 100], [0.2, -0.3, 0.5], [-7, 0.0, 0.25]]
    w = [[1.0, 2.0], [0.5, 1.0], [1.5, -2.0]]
    result = mixed(x, w, 6)
    assert result["outlier_cols"] == [0, 2]
    # These independent per-column products include the non-outlier 0.5 in
    # a selected column, for every output column and every token row.
    for i in range(len(x)):
        for n in range(2):
            high = F(x[i][0])*F(w[0][n])+F(x[i][2])*F(w[2][n])
            close(result["outlier"][i][n], high)
            low_reference = F(x[i][1])*F(w[1][n])
            close(result["regular"][i][n], low_reference)
            close(result["output"][i][n], high+low_reference)
    assert result["outlier"][1] == [0.95, -0.6]
    # With quantization disabled in this independent algebra, every possible
    # partition is exactly the same rational full matrix multiplication.
    for bits in range(1 << 3):
        selected = [k for k in range(3) if bits & (1 << k)]
        regular = [k for k in range(3) if k not in selected]
        for row in x:
            for n in range(2):
                full = sum(F(row[k])*F(w[k][n]) for k in range(3))
                split = sum((F(row[k])*F(w[k][n]) for k in selected), F(0))+sum((F(row[k])*F(w[k][n]) for k in regular), F(0))
                assert split == full
check("selected-column union covers all rows without double counting and every unquantized partition preserves GEMM", partition_oracle)

def threshold_and_empty_branches():
    w = [[1.0, -2.0], [2.0, 0.5], [3.0, 0.0]]
    x = [[6.0, -6.0, math.nextafter(6, -math.inf)]]
    assert mixed(x, w, 6)["outlier_cols"] == [0, 1]
    assert mixed([[6.0]], [[2.0]], math.nextafter(6, math.inf))["outlier_cols"] == []
    assert mixed(x, w, 0)["outlier_cols"] == []
    assert mixed(x, w, -0.0)["outlier_cols"] == []
    all_high = mixed([[7, -8], [0, 1]], [[2, 0], [-1, 0]], 6)
    assert all_high["outlier_cols"] == [0, 1]
    assert all_high["regular"] == [[0, 0], [0, 0]]
    assert all_high["output"] == [[22, 0], [-1, 0]]
    all_low = mixed([[1, -2], [0, 0]], [[1, 0], [2, 0]], 6)
    assert all_low["outlier"] == [[0, 0], [0, 0]]
    assert all_low["output"][1] == [0, 0]
    assert all(row[1] == 0 for row in all_low["output"])
    assert mixed([[7]], [[2]], 6)["output"] == [[14]]
    original = [[0.1, 0.2, 0.3]]
    appended = original+[[7, -8, 9]]
    a = set(mixed(original, w, 6)["outlier_cols"])
    b = set(mixed(appended, w, 6)["outlier_cols"])
    assert a <= b and b == {0, 1, 2}
check("threshold equality, zero disable, empty regular/outlier sets and zero vectors follow one explicit convention", threshold_and_empty_branches)

def bad_inputs_and_intermediates():
    for x, w, threshold in [
        ([], [[1]], 6), ([[]], [[1]], 6), ([[1], [1, 2]], [[1]], 6),
        ([[1]], [], 6), ([[1]], [[1], [2]], 6), ([[math.inf]], [[1]], 6),
        ([[1]], [[math.nan]], 6), ([["invalid"]], [[1]], 6),
        ([[10**1000]], [[1]], 6), ([[1]], [[1]], math.inf),
        ([[1]], [[1]], math.nan), ([[1]], [[1]], -1),
        ([[1]], [[1]], None), ([[1]], [[1]], 10**1000),
        ([[math.ulp(0.0)]], [[1]], 0),
        ([[1e-160]], [[1e-160]], 0),
        ([[1e308]], [[1e308]], 0),
        ([[1e308]], [[1e308]], 6),
        ([[1e308, 1e308]], [[1], [1]], 6),
        ([[1e308, 1]], [[1], [1e308]], 6),
    ]:
        try:
            mixed(x, w, threshold)
        except ValueError:
            continue
        raise AssertionError(("invalid/unrepresentable case accepted", x, w, threshold))
    # The mathematical final product is representable, but the naive scale
    # product underflows first. The article now rejects instead of returning 0.
    assert 1e-160*1e-160 > 0 and (1e-160/127)*(1e-160/127) == 0
    # Accumulation and combination are guarded, not just input validation.
    try:
        scopes["matrix"]["finite_sum"]([1e308, 1e308])
    except ValueError:
        pass
    else:
        raise AssertionError("non-finite accumulation accepted")
check("actual input guards and scale/rescaling/accumulation guards reject finite intermediate overflow or underflow", bad_inputs_and_intermediates)

def implementation_boundary_counterexamples():
    # This is a numerical counterexample to substituting original W for a
    # CB-dequantized W, not an execution of bitsandbytes.
    original_weight = F(1, 10)
    stored_codes, step = quant([1.0, float(original_weight)])
    assert stored_codes == [127, 13]
    reconstructed = F(step)*stored_codes[1]
    assert reconstructed != original_weight
    paper_style = mixed([[0, 100]], [[1], [0.1]], 6)
    assert paper_style["outlier"] == [[10]]
    assert not math.isclose(float(100*reconstructed), paper_style["outlier"][0][0])
    # For this pinned default source, only threshold-crossing elements are
    # excluded from statistics before all selected columns are zeroed in CA.
    source_row = [F(1, 5), F(-3, 10), F(1, 2)]
    source_amax = max(abs(v) for v in source_row if abs(v) < 6)
    teaching_amax = max(abs(source_row[k]) for k in [0, 1])
    assert source_amax == F(1, 2) and teaching_amax == F(3, 10)
    source_codes = [nearest(v, source_amax/127) for v in source_row[:2]]
    teaching_codes = [nearest(v, teaching_amax/127) for v in source_row[:2]]
    assert source_codes != teaching_codes
check("CB weight reconstruction and element-filtered statistics differ from the stated teaching decomposition", implementation_boundary_counterexamples)

def improvement_is_not_monotonic():
    x = [[0.125, 0.125, 0.125, 0.375, 100.0]]
    w = [[1.0], [1.0], [1.0], [-1.0], [1.0]]
    baseline = math.fsum(x[0][k]*w[k][0] for k in range(5))
    naive = mixed(x, w, 0)["output"][0][0]
    hybrid = mixed(x, w, 6)["output"][0][0]
    close(baseline, 100)
    close(naive, baseline)
    assert abs(hybrid-baseline) > 0.002
    # The original published example improves; a different cancellation
    # pattern worsens. Neither one is a model-quality benchmark.
    example_error = abs(printed["matrix"]["output"][0][0]-100.31)
    assert 0 < example_error < abs(printed["single"]["int8"]-100.31)
check("a constructive cancellation counterexample rejects a universal monotonic-error improvement claim", improvement_is_not_monotonic)

print(f"{passed} LLM.int8 groups passed; actual stdlib article blocks and rational/matrix counterexamples only, no bitsandbytes, PyTorch, FP16 kernel, GPU or model-quality benchmark.")
`;

const python = process.env.BLOG_PYTHON ?? (process.platform === 'win32' ? 'python' : 'python3');
const result = spawnSync(python, ['-c', program], {
  input: JSON.stringify(payload), encoding: 'utf8',
  env: { ...process.env, PYTHONUTF8: '1' },
});
if (result.stdout) process.stdout.write(result.stdout);
if (result.stderr) process.stderr.write(result.stderr);
if (result.error) throw result.error;
assert.equal(result.status, 0, 'LLM.int8 manuscript example audit failed');
