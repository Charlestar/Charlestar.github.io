import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';

const post = readFileSync('_posts/2026-11-10-swiglu-transformer-ffn.md', 'utf8');
function block(marker) {
  const expression = new RegExp(`<!-- example: ${marker} -->\\s*\x60\x60\x60python\\r?\\n([\\s\\S]*?)\\r?\\n\x60\x60\x60`, 'g');
  const matches = [...post.matchAll(expression)];
  assert.equal(matches.length, 1, `one ${marker} block required`);
  return matches[0][1];
}
const code = ['swiglu-reference', 'swiglu-small-example'].map(block).join('\n');
const tests = String.raw`
import copy
import random
import struct

def close(a,b,tolerance=1e-9):
    if isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for left,right in zip(a,b):
            close(left,right,tolerance)
    else:
        assert math.isclose(a,b,rel_tol=tolerance,abs_tol=tolerance),(a,b)

# Scalar-loop oracle does not call the article's matrix multiplication helpers.
def scalar_loss(inputs, gate, up, down, upstream):
    total=0.0
    for i,row in enumerate(inputs):
        for j in range(len(down[0])):
            output=0.0
            for k in range(len(down)):
                a=sum(row[t]*gate[t][k] for t in range(len(row)))
                b=sum(row[t]*up[t][k] for t in range(len(row)))
                output+=(a/(1+math.exp(-a)))*b*down[k][j]
            total+=upstream[i][j]*output
    return total

def finite_difference(arrays,upstream,which):
    values=arrays[which]
    answer=[[0.0]*len(row) for row in values]
    delta=1e-5
    for i,row in enumerate(values):
        for j in range(len(row)):
            previous=row[j]
            row[j]=previous+delta
            plus=scalar_loss(*arrays,upstream)
            row[j]=previous-delta
            minus=scalar_loss(*arrays,upstream)
            row[j]=previous
            answer[i][j]=(plus-minus)/(2*delta)
    return answer

close(y,[[0.7310585786300049,-0.5378828427399902]])
close(dx,[[1.8519048263915063,1.558259373884048]])
assert math.isclose(scalar_loss(x,wg,wu,wd,g),-0.3447071068499755)
assert silu(-1)<0 and silu(2)>1
assert silu(0)==0 and silu_prime(0)*3==1.5
assert sigmoid(-1000)==0 and sigmoid(1000)==1
assert silu_prime(-1000)==0 and silu_prime(1000)==1

rng=random.Random(1103)
for n,d,h in [(1,2,2),(2,3,5),(3,2,3),(2,4,1)]:
    arrays=[[[rng.uniform(-0.6,0.6) for _ in range(d)] for _ in range(n)],
            [[rng.uniform(-0.6,0.6) for _ in range(h)] for _ in range(d)],
            [[rng.uniform(-0.6,0.6) for _ in range(h)] for _ in range(d)],
            [[rng.uniform(-0.6,0.6) for _ in range(d)] for _ in range(h)]]
    upstream=[[rng.uniform(-1,1) for _ in range(d)] for _ in range(n)]
    original=copy.deepcopy(arrays)
    output,cache=swiglu_forward(*arrays)
    gradients=swiglu_backward(*arrays,cache,upstream)
    close(sum(upstream[i][j]*output[i][j] for i in range(n) for j in range(d)),
          scalar_loss(*arrays,upstream))
    for which in range(4):
        close(gradients[which],finite_difference(arrays,upstream,which),1e-7)
    assert arrays==original
    inputs,gate,up,down=arrays

    # Merged gate/up GEMM: pack columns, then split using the actual h boundary.
    packed=mm(inputs,[left+right for left,right in zip(gate,up)])
    merged_u=[[silu(row[k])*row[h+k] for k in range(h)] for row in packed]
    close(mm(merged_u,down),output)

    # Uneven teaching shards cover tail channels; sum partial down projections.
    partial_y=[]
    partial_dx=[]
    shard_dw_gate=[[] for _ in range(d)]
    shard_dw_up=[[] for _ in range(d)]
    shard_dw_down=[]
    for start in range(0,h,2):
        stop=min(start+2,h)
        local_gate=[row[start:stop] for row in gate]
        local_up=[row[start:stop] for row in up]
        local_down=down[start:stop]
        local_y,local_cache=swiglu_forward(inputs,local_gate,local_up,local_down)
        local_grads=swiglu_backward(inputs,local_gate,local_up,local_down,local_cache,upstream)
        partial_y.append(local_y)
        partial_dx.append(local_grads[0])
        for row in range(d):
            shard_dw_gate[row].extend(local_grads[1][row])
            shard_dw_up[row].extend(local_grads[2][row])
        shard_dw_down.extend(local_grads[3])
    close([[sum(part[i][j] for part in partial_y) for j in range(d)] for i in range(n)], output)
    close([[sum(part[i][j] for part in partial_dx) for j in range(d)] for i in range(n)], gradients[0])
    close((shard_dw_gate,shard_dw_up,shard_dw_down),gradients[1:])

    # A consistent channel permutation changes only internal names.
    permutation=list(reversed(range(h)))
    perm_gate=[[row[k] for k in permutation] for row in gate]
    perm_up=[[row[k] for k in permutation] for row in up]
    perm_down=[down[k] for k in permutation]
    perm_output,perm_cache=swiglu_forward(inputs,perm_gate,perm_up,perm_down)
    close(perm_output,output)
    perm_grads=swiglu_backward(inputs,perm_gate,perm_up,perm_down,perm_cache,upstream)
    close(perm_grads[0],gradients[0])
    close(perm_grads[1],[[row[k] for k in permutation] for row in gradients[1]])
    close(perm_grads[2],[[row[k] for k in permutation] for row in gradients[2]])
    close(perm_grads[3],[gradients[3][k] for k in permutation])

# Shape-compatible channel errors and nonlinear folding counterexamples.
wrong_up=[[row[1],row[0]] for row in wu]
wrong_y=swiglu_forward(x,wg,wrong_up,wd)[0]
assert any(not math.isclose(a,b) for a,b in zip(y[0],wrong_y[0]))
double_y=swiglu_forward([[2*v for v in row] for row in x],wg,wu,wd)[0]
assert any(not math.isclose(a,2*b) for a,b in zip(double_y[0],y[0]))
# Naively halving [all_gate,all_up] gives ranks the wrong local gate/up pairs.
packed=mm(x,[left+right for left,right in zip(wg,wu)])
wrong_partial=[]
for rank in range(2):
    local=packed[0][2*rank:2*rank+2]
    wrong_partial.append([silu(local[0])*local[1]*value for value in wd[rank]])
wrong_sum=[sum(part[j] for part in wrong_partial) for j in range(2)]
assert any(not math.isclose(a,b) for a,b in zip(wrong_sum,y[0]))
# Residual Jacobian is the identity in addition to the FFN branch derivative.
residual_dx=[[dx[i][j]+g[i][j] for j in range(2)] for i in range(1)]
assert residual_dx != dx
# Equal zero outputs do not determine the local gate derivative; U alone loses
# information required by the backward unless other inputs are saved/recomputed.
for up_value in (3.0,6.0):
    zero_x,zero_gate,zero_up,zero_down=[[1.0]],[[0.0]],[[up_value]],[[1.0]]
    zero_y,zero_saved=swiglu_forward(zero_x,zero_gate,zero_up,zero_down)
    zero_grads=swiglu_backward(zero_x,zero_gate,zero_up,zero_down,zero_saved,[[1.0]])
    close(zero_y,[[0.0]])
    close(zero_grads,([[0.0]],[[up_value/2]],[[0.0]],[[0.0]]))

# BF16 round-to-nearest-even bookkeeping for moderate finite positive numbers.
# This models rounding locations, not Triton sigmoid accuracy or GPU execution.
def bf16_rne(value):
    bits=struct.unpack('>I',struct.pack('>f',value))[0]
    rounded=(bits+0x7fff+((bits>>16)&1))&0xffff0000
    return struct.unpack('>f',struct.pack('>I',rounded))[0]
separate=bf16_rne(bf16_rne(silu(1.0))*9.0)
fused_model=bf16_rne(silu(1.0)*9.0)
assert separate==6.5625 and fused_model==6.59375

def rejects(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError('ValueError expected')
rejects(lambda: swiglu_forward([],wg,wu,wd))
rejects(lambda: swiglu_forward([[1]],wg,wu,wd))
rejects(lambda: swiglu_forward(x,wg,[[1],[2]],wd))
rejects(lambda: swiglu_backward(x,wg,wu,wd,saved,[[1]]))

d,h,n=4096,11008,8192
baseline=8*d*d
gated=3*d*h
assert baseline==134217728 and gated==135266304
assert (gated/baseline-1)*100==0.78125
assert baseline*2==256*1024**2 and gated*2==258*1024**2
assert n*h*2==172*1024**2 and 3*n*h*2==516*1024**2
assert n*d*2==64*1024**2
assert 6*n*d*h+12*n*d*h==18*n*d*h
# For divisible exact-budget widths the dominant forward work matches exactly.
d,h0,h=12,48,32
assert 2*d*h0==3*d*h and 4*7*d*h0==6*7*d*h
print('SwiGLU article examples: full finite-difference gradients, channel-shard/permutation weight gradients, pairing/zero-state/rounding counterexamples and budget arithmetic passed; CPU only.')
`;
const result = spawnSync(process.env.BLOG_PYTHON || 'python', ['-B', '-c', code + '\n' + tests], { encoding: 'utf8' });
if (result.error) throw result.error;
process.stdout.write(result.stdout || '');
process.stderr.write(result.stderr || '');
assert.equal(result.status, 0);
