import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';

const post = readFileSync('_posts/2026-11-10-fused-linear-cross-entropy.md', 'utf8');
function block(marker) {
  const expression = new RegExp(`<!-- example: ${marker} -->\\s*\x60\x60\x60python\\r?\\n([\\s\\S]*?)\\r?\\n\x60\x60\x60`, 'g');
  const matches = [...post.matchAll(expression)];
  assert.equal(matches.length, 1, `one ${marker} block required`);
  return matches[0][1];
}
const code = ['flce-reference', 'flce-small-example'].map(block).join('\n');
const tests = String.raw`
import copy
import random

def close(a, b, tolerance=1e-9):
    if isinstance(a, (list, tuple)):
        assert len(a) == len(b), (a, b)
        for left, right in zip(a, b):
            close(left, right, tolerance)
    else:
        assert math.isclose(a, b, rel_tol=tolerance, abs_tol=tolerance), (a, b)

# Independent dense oracle for small, bounded logits uses unshifted exp/log.
def dense_oracle(x, w, targets, reduction='mean', ignore=-100):
    count = sum(y != ignore for y in targets)
    if reduction == 'mean' and not count:
        raise ValueError('undefined mean')
    denom = count if reduction == 'mean' else 1
    z = [[sum(a*b for a,b in zip(row, classifier)) for classifier in w] for row in x]
    p = [[math.exp(value)/sum(math.exp(other) for other in row) for value in row] for row in z]
    losses = [-math.log(p[i][y]) for i,y in enumerate(targets) if y != ignore]
    dz = [[((p[i][j] - int(j == y))/denom if y != ignore else 0.0)
           for j in range(len(w))] for i,y in enumerate(targets)]
    dx = [[sum(dz[i][j]*w[j][h] for j in range(len(w)))
           for h in range(len(w[0]))] for i in range(len(x))]
    dw = [[sum(dz[i][j]*x[i][h] for i in range(len(x)))
           for h in range(len(w[0]))] for j in range(len(w))]
    return sum(losses)/denom, dx, dw

def finite_difference(x, w, targets, reduction, scalar, which):
    values = x if which == 'x' else w
    answer = [[0.0]*len(row) for row in values]
    delta = 1e-5
    for i, row in enumerate(values):
        for j in range(len(row)):
            old = row[j]
            row[j] = old + delta
            plus = scalar*dense_oracle(x, w, targets, reduction)[0]
            row[j] = old - delta
            minus = scalar*dense_oracle(x, w, targets, reduction)[0]
            row[j] = old
            answer[i][j] = (plus-minus)/(2*delta)
    return answer

close(dx, [[-0.21239480869777932, 0.07734894894220859], [0,0],
           [0.23178332674055258, -0.2682166732594474]])
close(dw, [[0.07654825346759508, 0.2439277755801842],
           [-0.13370798889241697, -0.2560722244198158],
           [0.057159735424821834, 0.012144448839631602]])
rng = random.Random(1102)
for rows, hidden, vocab in [(1,2,3), (3,2,3), (5,3,4), (7,2,5)]:
    inputs = [[rng.uniform(-0.7,0.7) for _ in range(hidden)] for _ in range(rows)]
    weights = [[rng.uniform(-0.7,0.7) for _ in range(hidden)] for _ in range(vocab)]
    labels = [rng.randrange(vocab) if i%3 != 1 else -100 for i in range(rows)]
    saved = copy.deepcopy((inputs, weights, labels))
    for reduction in ('mean','sum'):
        expected = dense_oracle(inputs, weights, labels, reduction)
        for chunk in (1,2,3,20):
            actual = linear_ce_chunked(inputs, weights, labels, chunk, reduction)
            close(actual, expected)
        for upstream in (0.0,0.3,-1.7):
            local_dx, local_dw = scale_saved_gradients(actual[1], actual[2], upstream)
            close(local_dx, finite_difference(inputs, weights, labels, reduction, upstream, 'x'), 1e-7)
            close(local_dw, finite_difference(inputs, weights, labels, reduction, upstream, 'w'), 1e-7)
    assert (inputs, weights, labels) == saved

# Entirely ignored chunks must not introduce a local divide by zero.
ignored_prefix = [[0.1,0.2], [0.2,-0.1], [0.3,0.2], [0.1,0.7], [-0.2,0.1]]
labels = [-100,-100,0,1,2]
close(linear_ce_chunked(ignored_prefix,weight,labels,2), dense_oracle(ignored_prefix,weight,labels))
# Unequal valid counts / tail: global mean is not an unweighted mean of means.
assert (2*1+1*4)/3 == 2
assert (1+4)/2 == 2.5

def rejects(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError('ValueError expected')

rejects(lambda: linear_ce_chunked(x, weight, [-100]*3))
all_ignored = linear_ce_chunked(x, weight, [-100]*3, reduction='sum')
close(all_ignored, (0, [[0,0]]*3, [[0,0]]*3))
close(linear_ce_chunked([], weight, [], reduction='sum'), (0, [], [[0,0]]*3))
rejects(lambda: linear_ce_chunked([], weight, []))
# Ignored projection is skipped by this CPU reference, even for finite inputs
# whose dot product would overflow. This says nothing about context visibility.
close(linear_ce_chunked([[1e308]], [[2.0],[0.0]], [-100], reduction='sum'),
      (0.0, [[0.0]], [[0.0],[0.0]]))
# ignore_index=0 excludes target rows, not vocabulary column 0 from other rows.
close(linear_ce_chunked(x, weight, [0,1,2], ignore_index=0),
      dense_oracle(x, weight, [0,1,2], ignore=0))
single = linear_ce_chunked([[1.0]], [[0.0],[0.0],[0.0]], [1], ignore_index=0)
assert math.isclose(single[0], math.log(3))

# An independently formed max-shift normalizer matches the online recurrence.
for values in [[0], [1,-2,4,0], [1000,999,-1000], [-1000,-1001,-999]]:
    m,s = online_stats(values)
    maximum = max(values)
    close((m,s), (maximum, math.fsum(math.exp(v-maximum) for v in values)))
# Merge vocabulary partitions independently of the article's scalar recurrence.
pieces = [[-3.0, 7.0], [1.0], [-9.0, 6.0, 2.0]]
merged_m, merged_s = -math.inf, 0.0
for piece in pieces:
    local_m = max(piece)
    local_s = math.fsum(math.exp(v-local_m) for v in piece)
    next_m = max(merged_m, local_m)
    merged_s = merged_s*math.exp(merged_m-next_m) + local_s*math.exp(local_m-next_m)
    merged_m = next_m
close((merged_m, merged_s), online_stats([v for piece in pieces for v in piece]))
large = linear_ce_chunked([[1.0]], [[1e16],[1e16],[1e16]], [0])
assert math.isclose(large[0], math.log(3))
lost = (1e16 + math.log(3)) - 1e16
assert not math.isclose(lost, math.log(3))
for shift in (-1000, 0, 1000):
    # Last input coordinate=1 implements a common logit bias through all W rows.
    shifted = linear_ce_chunked([[1.0,1.0]], [[1.0,shift],[2.0,shift],[-1.0,shift]], [1])
    expected_loss = math.log(math.exp(-1)+1+math.exp(-3))
    assert math.isclose(shifted[0], expected_loss)

for chunk in (0,-1,True,1.5):
    rejects(lambda chunk=chunk: linear_ce_chunked(x,weight,targets,chunk))
for reduction in ('none','median',''):
    rejects(lambda reduction=reduction: linear_ce_chunked(x,weight,targets,reduction=reduction))
for labels in ([3,0,1],[-1,0,1],[True,0,1],[0.5,0,1]):
    rejects(lambda labels=labels: linear_ce_chunked(x,weight,labels))
rejects(lambda: linear_ce_chunked([[1]],weight,[0]))
rejects(lambda: linear_ce_chunked(x,weight,[0]))
rejects(lambda: linear_ce_chunked(x,[],targets))
rejects(lambda: linear_ce_chunked([[float('nan'),0]],weight,[0]))
rejects(lambda: linear_ce_chunked(x,[[float('inf'),0]],targets))
rejects(lambda: scale_saved_gradients(dx,dw,float('nan')))
for ignore in (True, 1.5):
    rejects(lambda ignore=ignore: linear_ce_chunked(x,weight,targets,ignore_index=ignore))
for invalid in (True, '1', 10**1000):
    rejects(lambda invalid=invalid: linear_ce_chunked([[invalid]],[[0.0],[0.0]],[0]))
    rejects(lambda invalid=invalid: scale_saved_gradients(dx,dw,invalid))
# Finite inputs do not imply finite projection, loss, or gradients.
rejects(lambda: linear_ce_chunked([[1e308]], [[2.0],[0.0]], [0]))
rejects(lambda: linear_ce_chunked([[1e308,1e308]], [[1.0,1.0],[0.0,0.0]], [0]))
rejects(lambda: linear_ce_chunked([[1.0]], [[1e308],[-1e308]], [1]))
rejects(lambda: linear_ce_chunked([[1e308]]*4, [[0.0],[0.0]], [0]*4, reduction='sum'))
rejects(lambda: linear_ce_chunked([[0.0]], [[-1.7e308]]+[[1.7e308]]*3, [0]))
rejects(lambda: scale_saved_gradients([[1e308]], [[0.0]], 2.0))

# Flatten-then-shift creates a supervised pair across independent samples.
samples = [[10,11,12], [20,21,22]]
correct_pairs = [(row[i],row[i+1]) for row in samples for i in range(len(row)-1)]
flat = [token for row in samples for token in row]
wrong_pairs = list(zip(flat[:-1],flat[1:]))
assert (12,20) not in correct_pairs and (12,20) in wrong_pairs
# Source-style in-place saved-gradient scaling does not promise repeatability.
saved_gradient = 5.0
saved_gradient *= 2.0
saved_gradient *= 3.0
assert saved_gradient == 30.0 and saved_gradient != 3.0*5.0

assert 8192*131072*2 == 2*1024**3
assert 8192*131072*4 == 4*1024**3
assert 256*131072*2 == 64*1024**2
assert 8192*4096*2 == 64*1024**2
assert 131072*4096*2 == 1024**3
print('FLCE article examples: dense/finite-difference gradients, chunks/masks/reductions/upstream, normalizer merge, numeric overflow, shift/repeated-backward counterexamples and byte models passed; CPU only.')
`;
const result = spawnSync(process.env.BLOG_PYTHON || 'python', ['-B', '-c', code + '\n' + tests], { encoding: 'utf8' });
if (result.error) throw result.error;
process.stdout.write(result.stdout || '');
process.stderr.write(result.stderr || '');
assert.equal(result.status, 0);
