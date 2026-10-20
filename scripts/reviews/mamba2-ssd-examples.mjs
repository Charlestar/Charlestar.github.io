// Extract actual article algorithms and compare against exact rational path accounting.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';

const post = readFileSync('_posts/2026-10-21-mamba2-ssd-state-chunk-matrix.md', 'utf8');
function block(marker) {
  const pattern = new RegExp(`<!-- example: ${marker} -->\\s*\x60\x60\x60python\\r?\\n([\\s\\S]*?)\\r?\\n\x60\x60\x60`, 'g');
  const matches = [...post.matchAll(pattern)];
  assert.equal(matches.length, 1, `expected one ${marker} block`);
  return matches[0][1];
}
const code = ['mamba2-ssd-recurrence', 'mamba2-ssd-dense', 'mamba2-ssd-chunked', 'mamba2-ssd-example'].map(block).join('\n');
const tests = String.raw`
import copy
import itertools
import random

def rejects(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("expected ValueError")

# Independent exact oracle: keep each write as a separate historical contribution.
# It does not materialize M or combine writes into one recurrent state.
def tagged_contributions(a, b, c, x, initial):
    n, p = len(initial), len(initial[0])
    pieces = [[row[:] for row in initial]]
    outputs = []
    for t in range(len(a)):
        pieces = [[[a[t] * value for value in row] for row in piece] for piece in pieces]
        pieces.append([[b[t][i] * x[t][j] for j in range(p)] for i in range(n)])
        outputs.append([sum(c[t][i] * piece[i][j]
                            for piece in pieces for i in range(n)) for j in range(p)])
    final = [[sum(piece[i][j] for piece in pieces) for j in range(p)] for i in range(n)]
    return outputs, final

def assert_float_equal(actual, expected):
    for left_matrix, right_matrix in zip(actual, expected):
        assert len(left_matrix) == len(right_matrix)
        for left_row, right_row in zip(left_matrix, right_matrix):
            assert len(left_row) == len(right_row)
            for left, right in zip(left_row, right_row):
                assert math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-10), (left, right)

assert reference == tagged_contributions(a, b, c, x, initial)
for gates in itertools.product([F(0), F(1, 2), F(1)], repeat=4):
    bq = [[F(i % 3 - 1), F(2)] for i in range(4)]
    cq = [[F(1), F(1 - i)] for i in range(4)]
    xq = [[F(i - 2), F(3 - i)] for i in range(4)]
    hq = [[F(1), F(-2)], [F(3), F(1)]]
    expected = tagged_contributions(gates, bq, cq, xq, hq)
    assert recurrent(gates, bq, cq, xq, hq) == expected
    assert dense(gates, bq, cq, xq, hq) == expected
    for size in [1, 2, 3, 4, 7]:
        assert chunked(gates, bq, cq, xq, hq, size) == expected

rng = random.Random(19)
for length in range(9):
    for n in [1, 2, 3]:
        for p in [1, 2, 4]:
            af = [rng.random() for _ in range(length)]
            bf = [[rng.uniform(-2,2) for _ in range(n)] for _ in range(length)]
            cf = [[rng.uniform(-2,2) for _ in range(n)] for _ in range(length)]
            xf = [[rng.uniform(-2,2) for _ in range(p)] for _ in range(length)]
            hf = [[rng.uniform(-2,2) for _ in range(p)] for _ in range(n)]
            saved = copy.deepcopy((af,bf,cf,xf,hf))
            expected = tagged_contributions(af,bf,cf,xf,hf)
            assert_float_equal(recurrent(af,bf,cf,xf,hf), expected)
            assert_float_equal(dense(af,bf,cf,xf,hf), expected)
            for size in [1, 2, 5, 12]:
                assert_float_equal(chunked(af,bf,cf,xf,hf,size), expected)
            assert (af,bf,cf,xf,hf) == saved

# One-step independent numeric result; current write is NOT multiplied by a.
assert recurrent([F(1,2)], [[F(2)]], [[F(3)]], [[F(4)]], [[F(2)]]) == ([[F(27)]], [[F(9)]])
# Strong decay underflows long prefix products, but recent writes remain meaningful.
length = 300
af, bf, cf, xf, hf = [0.001]*length, [[1.0]]*length, [[1.0]]*length, [[1.0]]*length, [[2.0]]
expected = recurrent(af,bf,cf,xf,hf)
assert all(math.isfinite(row[0]) for row in expected[0])
assert_float_equal(dense(af,bf,cf,xf,hf), expected)
for size in [1,7,64,400]:
    assert_float_equal(chunked(af,bf,cf,xf,hf,size), expected)
# Zero input still preserves decaying initial-state contributions.
assert recurrent([F(1,2),F(1,2)], [[F(1)]]*2, [[F(1)]]*2, [[F(0)]]*2, [[F(4)]]) == ([[F(2)],[F(1)]], [[F(1)]])

# Affine block composition is associative but not commutative.
def compose(second, first):
    a2,s2 = second
    a1,s1 = first
    return a2*a1, a2*s1+s2
f1,f2,f3 = (F(1,2),F(1)), (F(1,3),F(2)), (F(1,4),F(3))
assert compose(f3, compose(f2,f1)) == compose(compose(f3,f2), f1)
assert compose(f2,f1) != compose(f1,f2)

# Independent audit: recover the temporal mixing matrix using basis inputs to the
# actual recurrence, then check exact cross-cut factors and ranks by elimination.
def impulse_matrix(gates, writes, reads):
    length, n = len(gates), len(writes[0])
    columns = []
    for source in range(length):
        inputs = [[F(int(t == source))] for t in range(length)]
        out, _ = recurrent(gates, writes, reads, inputs, [[F(0)] for _ in range(n)])
        columns.append([row[0] for row in out])
    return [[columns[s][t] for s in range(length)] for t in range(length)]

def exact_rank(matrix):
    work = [list(map(F, row)) for row in matrix]
    rows, cols, pivot_row = len(work), len(work[0]), 0
    for col in range(cols):
        pivot = next((row for row in range(pivot_row, rows) if work[row][col]), None)
        if pivot is None:
            continue
        work[pivot_row], work[pivot] = work[pivot], work[pivot_row]
        scale = work[pivot_row][col]
        work[pivot_row] = [value / scale for value in work[pivot_row]]
        for row in range(rows):
            if row != pivot_row:
                multiplier = work[row][col]
                work[row] = [left - multiplier * right
                             for left, right in zip(work[row], work[pivot_row])]
        pivot_row += 1
        if pivot_row == rows:
            break
    return pivot_row

full_rank = impulse_matrix([F(1,2)]*6, [[F(1)]]*6, [[F(1)]]*6)
assert exact_rank(full_rank) == 6  # N=1 does not make the full matrix rank one.
assert all(full_rank[t][s] == 0 for t in range(6) for s in range(t+1, 6))
audit_rng = random.Random(103)
for n in (1, 2, 3):
    for case in range(40):
        gates = [audit_rng.choice((F(0), F(1,2), F(1))) for _ in range(7)]
        writes = [[F(audit_rng.randrange(-2,3)) for _ in range(n)] for _ in range(7)]
        reads = [[F(audit_rng.randrange(-2,3)) for _ in range(n)] for _ in range(7)]
        matrix = impulse_matrix(gates, writes, reads)
        for cut in range(6):
            cross = [row[:cut+1] for row in matrix[cut+1:]]
            assert exact_rank(cross) <= n
            for t in range(cut+1, 7):
                future = math.prod(gates[cut+1:t+1])
                for s in range(cut+1):
                    past = math.prod(gates[s+1:cut+1])
                    factored = sum((future*reads[t][i]) * (past*writes[s][i]) for i in range(n))
                    assert matrix[t][s] == factored
            if gates[cut+1] == 0:
                assert exact_rank(cross) == 0

# Restart at every boundary, including empty prefix/suffix, carrying the full state.
for split in range(len(a)+1):
    left, boundary = recurrent(a[:split], b[:split], c[:split], x[:split], initial)
    right, final = chunked(a[split:], b[split:], c[split:], x[split:], boundary, 3)
    assert (left+right, final) == reference
# The first decay affects only an incoming state, not the same-step write.
zero_initial = [[F(0),F(0)], [F(0),F(0)]]
changed_a0 = list(a)
changed_a0[0] = F(0)
assert recurrent(a,b,c,x,zero_initial) == recurrent(changed_a0,b,c,x,zero_initial)
assert recurrent(a,b,c,x,initial) != recurrent(changed_a0,b,c,x,initial)
# A future input perturbation cannot alter earlier outputs at fixed coefficients.
changed_x = copy.deepcopy(x)
changed_x[3][0] += F(17)
assert recurrent(a,b,c,changed_x,initial)[0][:3] == reference[0][:3]
# Signed mixing weights need not sum to one; no softmax probabilities are implied.
signed = impulse_matrix([F(1), F(1)], [[F(1)], [F(-1)]], [[F(1)], [F(1)]])
assert signed[1] == [F(1), F(-1)] and sum(signed[1]) == 0
# Fixed nonzero initial state makes the input map affine, not generally linear.
zero_x = [[F(0),F(0)] for _ in x]
doubled_x = [[2*value for value in row] for row in x]
offset = recurrent(a,b,c,zero_x,initial)
doubled = recurrent(a,b,c,doubled_x,initial)
for matrix, once, base in zip(doubled, reference, offset):
    assert matrix == [[2*value - shift for value, shift in zip(row, base_row)]
                      for row, base_row in zip(once, base)]

# Storage calculation is separate from model quality or real deployment peak.
assert 1*32*32*64*128*2 == 16*1024*1024
assert 1*32*(32*64+2*128)*4*2 == 589824

for bad in [-1, 2, float('nan'), float('inf'), True, None, '0.5']:
    rejects(lambda bad=bad: recurrent([bad], [[1]], [[1]], [[1]], [[0]]))
for bad in [float('nan'), float('inf'), True, '1', 10**1000]:
    rejects(lambda bad=bad: dense([1], [[bad]], [[1]], [[1]], [[0]]))
for initial_bad in [[], [[]], [[0],[0,1]]]:
    rejects(lambda initial_bad=initial_bad: recurrent([], [], [], [], initial_bad))
rejects(lambda: recurrent([1], [], [[1]], [[1]], [[0]]))
rejects(lambda: recurrent([1], [[1,2]], [[1]], [[1]], [[0]]))
rejects(lambda: recurrent([1], [[1]], [[1]], [[1,2]], [[0]]))
for bad in [0,-1,True,1.5,None]:
    rejects(lambda bad=bad: chunked([1], [[1]], [[1]], [[1]], [[0]], bad))
print('Mamba-2 SSD examples: exact/float algorithm equivalence, cross-cut rank/factors, restart/causality/affine-state oracles and storage arithmetic passed; no Torch/GPU/model run.')
`;
const python = process.env.BLOG_PYTHON || 'python';
const result = spawnSync(python, ['-B', '-c', code + '\n' + tests], { encoding: 'utf8' });
if (result.error) throw result.error;
process.stdout.write(result.stdout || '');
process.stderr.write(result.stderr || '');
assert.equal(result.status, 0);
