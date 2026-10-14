import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';

const article = readFileSync(new URL('../../_posts/2026-10-15-kivi-kv-cache-axes.md', import.meta.url), 'utf8');
function extract(name) {
  const pieces = article.split(`<!-- example: kivi-${name} -->`);
  assert.equal(pieces.length, 2, `Missing or duplicate KIVI marker: ${name}`);
  const block = pieces[1].match(/^\s*```python\r?\n([\s\S]*?)```/);
  assert.ok(block, `Missing actual KIVI Python: ${name}`);
  return block[1];
}
const blocks = {
  axes: extract('axes-and-2bit-reconstruction'),
  state: extract('streaming-compute-versus-persist'),
  attention: extract('global-softmax-and-value-boundary'),
};
assert.equal([...article.matchAll(/^```python\s*$/gm)].length, 3, 'Exercise every KIVI Python block');
assert.match(article, /^date: 2026-10-15$/m);
assert.match(article, /^author: iStar$/m);
assert.match(article, /^series_order: 80$/m);
assert.match(article, /为什么选择不同的量化轴/);

// Execute exactly the post's blocks. Fraction nearest-code distance,
// high-precision exponentials, closed-form token-index sets and enumerated
// allocations are independent oracles, not execution of official GPU kernels.
const program = String.raw`
import ast
import contextlib
from decimal import Decimal as D, localcontext
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
    print('PASS', name)

def close(actual, expected, rel_tol=1e-11, abs_tol=1e-12):
    assert math.isfinite(actual) and math.isclose(actual, float(expected), rel_tol=rel_tol, abs_tol=abs_tol), (actual, expected)

def fails(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return
    raise AssertionError(('expected ValueError', fn.__name__, args))

def exact_group(values):
    values = list(map(F, values))
    lo, hi = min(values), max(values)
    if lo == hi:
        return [0]*len(values), values
    grid = [lo+(hi-lo)*F(c, 3) for c in range(4)]
    codes = [min(range(4), key=lambda c: (abs(v-grid[c]), c % 2, c)) for v in values]
    return codes, [grid[c] for c in codes]

def exact_matrix(matrix, axis):
    result = [[F(0) for _ in row] for row in matrix]
    groups = ([[(t,d) for d in range(len(matrix[0]))] for t in range(len(matrix))]
              if axis == 'token' else
              [[(t,d) for t in range(len(matrix))] for d in range(len(matrix[0]))])
    for group in groups:
        _, rebuilt = exact_group([matrix[t][d] for t,d in group])
        for (t,d), value in zip(group, rebuilt):
            result[t][d] = value
    return result

def actual_blocks():
    for name, source in blocks.items():
        ast.parse(source)
        scope, captured = {}, io.StringIO()
        with contextlib.redirect_stdout(captured):
            exec(compile(source, 'article:kivi-'+name, 'exec'), scope)
        scopes[name], printed[name] = scope, captured.getvalue()
        assert printed[name].strip()
    assert len(scopes['state']['states']) == 9
    close(scopes['attention']['mixed'][0], F(117,17))
    assert scopes['axes']['key_channel'] == scopes['axes']['key']
check('all three actual manuscript blocks execute independently with their assertions and outputs', actual_blocks)
quant = scopes['axes']['quantize_group']

def nearest_code_and_float_domain():
    cases = 0
    # Exactly representable dyadic grids; oracle chooses among four rational
    # reconstruction points, not another division/round implementation.
    for origin in [F(-8), F(-1,2), F(0), F(1,2), F(8)]:
        for step in [F(1,8), F(1,2), F(1), F(8)]:
            for i in range(49):
                values = [origin, origin+step*F(i,16), origin+3*step]
                codes, minimum, actual_step, rebuilt = quant(list(map(float, values)))
                expected_codes, expected = exact_group(values)
                assert codes == expected_codes
                assert F(minimum) == origin and F(actual_step) == step
                assert list(map(F, rebuilt)) == expected
                assert all(abs(v-r) <= step/2 for v,r in zip(values, expected))
                cases += 1
    assert cases == 980
    for midpoint in [.5, 1.5, 2.5]:
        for v in [math.nextafter(midpoint, -math.inf), midpoint, math.nextafter(midpoint, math.inf)]:
            codes, _, _, rebuilt = quant([0., v, 3.])
            expected_codes, expected = exact_group([0., v, 3.])
            assert codes == expected_codes and list(map(F, rebuilt)) == expected
    # Zero is not a reconstruction point of this min/max grid, unlike an
    # integer affine zero point. RNE selects upper middle code at x=0.
    codes, origin, step, rebuilt = quant([-1., 0., 1.])
    assert codes == [0,2,3] and origin == -1
    close(rebuilt[1], F(1,3))
    assert F(-origin)/F(step) != int(F(-origin)/F(step))
    tiny = math.ulp(0.)
    for v in [0., -5., 5., tiny, -tiny, sys.float_info.max]:
        assert quant([v]*4)[3] == [v]*4
    assert quant([0., 3*tiny])[3] == [0., 3*tiny]
    for values in [[], [math.inf], [math.nan], [-1e308, 1e308], [0., tiny]]:
        fails(quant, values)
check('980 rational grids, tie neighbours, floating offset, constant and tiny/overflow contracts', nearest_code_and_float_domain)

def packet_contract():
    pack, unpack = scopes['axes']['pack2'], scopes['axes']['unpack2']
    for byte in range(256):
        expected = [(byte//(4**i)) % 4 for i in range(4)]
        packet, n = pack(expected)
        assert packet == bytes([byte]) and n == 4
        assert unpack(packet, n) == expected
        assert unpack(bytearray(packet), n) == expected
    assert pack([0,1,2,3]) == (bytes([0xE4]),4)
    assert pack([0,1,2]) == (bytes([0x24]),3)
    assert pack([3,2,1,0,3]) == (bytes([0x1B,0x03]),5)
    for n in range(18):
        values = [(i*3+1) % 4 for i in range(n)]
        packet, length = pack(values)
        assert len(packet) == (n+3)//4 and unpack(packet,length) == values
        if n % 4:
            assert packet[-1] < 4**(n % 4)  # Padding bits are not logical values.
    assert unpack(b'', 0) == []
    for bad in [[True], [4], [-1], [1.0], [math.nan]]:
        fails(pack, bad)
    for data,n in [([256],1), ([-1],1), ('x',1), (b'\x00',True), (b'\x00',-1), (b'\x00',5), (b'',1), (b'\x00',1.5)]:
        fails(unpack,data,n)
check('all 256 bytes, known low-bit order, partial logical lengths and strict payload/code types', packet_contract)

def axes_and_error_propagation():
    scope = scopes['axes']
    keys, values = scope['key'], scope['value']
    for matrix in [keys,values, [[-2.,1.,4.],[3.,-1.,2.]]]:
        for axis, fn in [('token',scope['qdq_rows']), ('channel',scope['qdq_columns'])]:
            actual, expected = fn(matrix), exact_matrix(matrix,axis)
            for row_a,row_e in zip(actual,expected):
                for a,e in zip(row_a,row_e):
                    close(a,e)
    original_s = [F(row[1])/2 for row in keys]
    token_s = [F(row[1])/2 for row in scope['key_token']]
    assert len(set(b-a for a,b in zip(original_s,token_s))) > 1  # Not a uniform softmax shift.
    assert max(range(4),key=original_s.__getitem__) == 1
    assert max(range(4),key=token_s.__getitem__) == 3
    with localcontext() as ctx:
        ctx.prec = 70
        logits = [D(v.numerator)/D(v.denominator) for v in original_s]
        exps = [x.exp() for x in logits]
        expected_p = [x/sum(exps) for x in exps]
        for actual,expected in zip(scope['p_original'],expected_p):
            close(actual,expected)
    assert scope['out_token'] == [1.,0.,1.,0.]
    assert all(v == F(100,3) for v in exact_matrix(values,'channel')[1])
    # Matrix output decomposition uses changed probabilities and changed V.
    # Verify it independently with exact rational arithmetic, not the post's
    # weighted helper. This is an algebraic identity for any p and phat.
    p, phat = [F(1,4)]*4, [F(1,10),F(2,10),F(3,10),F(4,10)]
    v0, v1 = [list(map(F,row)) for row in values], exact_matrix(values,'channel')
    for d in range(4):
        lhs = sum(phat[t]*v1[t][d]-p[t]*v0[t][d] for t in range(4))
        rhs = sum((phat[t]-p[t])*v0[t][d]+phat[t]*(v1[t][d]-v0[t][d]) for t in range(4))
        assert lhs == rhs
    # Reversing the fixed-outlier example makes the other axis exact: there
    # is no universal per-channel dominance theorem.
    transposed = [list(col) for col in zip(*keys)]
    assert scope['qdq_rows'](transposed) == transposed
    assert scope['qdq_columns'](transposed) != transposed
check('independent matrix grouping, nonconstant logit error, Decimal probabilities and output-error identity', axes_and_error_propagation)

def closed_form_state_oracle():
    trace = scopes['state']['trace']
    for r in [1,2,4,6,8,12,32,128]:
        for g in [g for g in range(1,r+1) if r % g == 0]:
            with contextlib.redirect_stdout(io.StringIO()):
                snapshots = trace(r,g,3*r+3)
            for t,(k_compute,v_compute,kr,vr) in enumerate(snapshots,1):
                # Closed-form membership, unlike the append/flush/FIFO code.
                k_old = ((t-1)//r)*r
                v_old = max(0,t-1-r)
                assert k_compute == (list(range(1,k_old+1)),list(range(k_old+1,t+1)))
                assert v_compute == (list(range(1,v_old+1)),list(range(v_old+1,t+1)))
                assert kr == list(range((t//r)*r+1,t+1))
                assert vr == list(range(max(1,t-r+1),t+1))
                for history,full in [k_compute,v_compute]:
                    assert set(history).isdisjoint(full)
                    assert len(set(history+full)) == t
                    assert sorted(history+full) == list(range(1,t+1))
                # Paper Algorithm 1 uses the just-persisted partition for
                # current compute, so it differs at K flush/V FIFO boundaries.
                if t % r == 0:
                    assert len(k_compute[1]) == r and not kr
                if t > r:
                    assert len(v_compute[1]) == r+1 and len(vr) == r
            # Uniform persisted cycle average is (R-1)/2, not a lower bound.
            assert F(sum(len(s[2]) for s in snapshots[:r]),r) == F(r-1,2)
    for steps in [0,1]:
        with contextlib.redirect_stdout(io.StringIO()):
            assert len(trace(4,2,steps)) == steps
    for r,g,n in [(0,1,1),(4,0,1),(5,2,1),(True,1,1),(4,True,1),(4,2,-1),(4,2,True),(4,2,1.5)]:
        fails(trace,r,g,n)
check('closed-form token membership across complete cycles, distinct compute/persist boundaries and invalid G/R/steps', closed_form_state_oracle)

def global_attention_oracle():
    scope = scopes['attention']
    softmax, weighted = scope['softmax'], scope['weighted']
    exact_p = [F(1,17)]*8+[F(9,17)]
    for p,e in zip(scope['probabilities'],exact_p):
        close(p,e)
    assert sum(exact_p) == 1
    expected = sum(p*t for t,p in enumerate(exact_p,1))
    assert expected == F(117,17)
    close(scope['mixed'][0],expected)
    close(scope['mixed'][1],-expected)
    close(scope['wrong'][0],F(27,2))
    # Check arbitrary logits against 70-digit exponentials and all valid
    # Value boundaries, not merely the one favourable 8/1 + 4/5 example.
    for raw in [[-3.,0.,2.,1.],[-1000.,-1001.,-999.],[0.],[4.,4.,4.,4.,4.]]:
        with localcontext() as ctx:
            ctx.prec = 70
            exponentials = [D(str(x)).exp() for x in raw]
            expected_p = [x/sum(exponentials) for x in exponentials]
            actual_p = softmax(raw)
            for a,e in zip(actual_p,expected_p):
                close(a,e)
        rows = [[float(t),float(t*t-2)] for t in range(1,len(raw)+1)]
        dense = weighted(actual_p,rows)
        for split in range(1,len(raw)):
            pieces = [weighted(actual_p[:split],rows[:split]),weighted(actual_p[split:],rows[split:])]
            for d in range(2):
                close(sum(part[d] for part in pieces),dense[d])
        for a,b in zip(actual_p,softmax([x+20 for x in raw])):
            close(a,b)
    fails(weighted,scope['probabilities'][:8],scope['value_quant_restored'])
check('exact global probability masses, Decimal softmax and all nonempty Value split reconstructions', global_attention_oracle)

def shape_and_finite_guards():
    a, attention = scopes['axes'], scopes['attention']
    for matrix in [[],[[]],[[1.],[2.,3.]],[[math.nan]],[[math.inf]]]:
        fails(a['qdq_columns'],matrix)
        fails(a['qdq_rows'],matrix)
    fails(a['attention_probabilities'],[[1.,2.]])
    original_query = a['query']
    try:
        a['query'] = [0.,2.,0.,0.]
        fails(a['attention_probabilities'],[[0.,1e308,0.,0.]])
    finally:
        a['query'] = original_query
    for weighted in [a['weighted'],attention['weighted']]:
        for p,rows in [([1.],[]),([1.],[[]]),([1.,0.],[[1.],[2.,3.]]),([1.],[[math.inf]]),([-1.],[[2.]]),([math.nan],[[2.]]),([2.],[[1e308]])]:
            fails(weighted,p,rows)
    for left,right in [([],[]),([1.],[1.,2.]),([math.inf],[1.]),([1e308],[2.]),([1e308,-1e308],[2.,2.])]:
        fails(attention['dot'],left,right)
    for logits in [[],[math.inf],[math.nan],[-1e308,1e308]]:
        fails(attention['softmax'],logits)
    assert attention['softmax']([1000.,1000.]) == [.5,.5]
    assert attention['softmax']([-1000.,-1000.]) == [.5,.5]
check('ragged/empty matrices, byte/type boundaries and finite inputs with overflowing intermediates are explicitly bounded', shape_and_finite_guards)

def enumerated_cache_budget():
    def allocations(t,d,g,r,b=2,e=2,f=2):
        rk,rv = t % r,min(t,r)
        # Enumerate groups and residual elements; no use of the aggregate
        # expression under test. Groups are assumed complete and no padding.
        key_groups = [(channel,start) for channel in range(d) for start in range(0,t-rk,g)]
        value_groups = [(token,start) for token in range(t-rv) for start in range(0,d,g)]
        code_bits = len(key_groups+value_groups)*g*b
        metadata = len(key_groups+value_groups)*2*e
        residual = sum(f for _ in range((rk+rv)*d))
        actual = F(code_bits,8)+metadata+residual
        formula = d*((2*t-rk-rv)*(F(b,8)+F(2*e,g))+(rk+rv)*f)
        assert actual == formula
        return actual, F(code_bits,8)+metadata,residual
    for t in [0,1,3,4,5,7,8,9,16,17]:
        for d,g,r in [(4,2,4),(8,4,8),(16,4,12),(32,32,128)]:
            for e in [2,4]:
                allocations(t,d,g,r,e=e)
    total,quantized,residual = allocations(4096,128,32,128)
    assert quantized == 378*1024 and residual == 32*1024 and total == 410*1024
    full = 2*4096*128*2
    assert full == 2*1024**2 and F(full,total) == F(1024,205)
    assert F(2,1)+F(16+16,32) == 3
    assert F(16,3) != 8
    # Actual head storage, not the number of Query heads.
    layers,batch,hkv,hq = 32,3,8,32
    assert full*layers*batch*hq == 4*(full*layers*batch*hkv)
    # Native 2-bit INT32 stores 16 elements, unlike the byte teaching pack.
    assert 32//2 == 16 and 4 % 16 != 0
check('enumerated complete-group metadata/residual allocations, conditional 410 KiB and Hkv rather than Hq', enumerated_cache_budget)

assert passed == 8
print(json.dumps({'groups':passed,'scope':'actual manuscript Python with independent CPU arithmetic/state/budget oracles; no official Torch/Triton/CUDA, packing artifact, model quality or GPU benchmark'}))
`;

const result = spawnSync(process.env.BLOG_PYTHON || 'python', ['-c', program], {
  input: JSON.stringify(blocks), encoding: 'utf8', maxBuffer: 4 * 1024 * 1024,
});
if (result.stdout) process.stdout.write(result.stdout);
if (result.stderr) process.stderr.write(result.stderr);
assert.equal(result.error, undefined, result.error?.message);
assert.equal(result.status, 0, 'KIVI manuscript examples or independent oracles failed');
