import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';

const article = readFileSync(new URL('../../_posts/2026-08-04-w4a8-progressive-quantization-kernel.md', import.meta.url), 'utf8');
function extract(name) {
  const parts = article.split(`<!-- example: w4a8-${name} -->`);
  assert.equal(parts.length, 2, `Missing or duplicate W4A8 marker: ${name}`);
  const block = parts[1].match(/^\s*```python\r?\n([\s\S]*?)```/);
  assert.ok(block, `Missing real W4A8 Python: ${name}`);
  return block[1];
}
const blocks = {
  range: extract('two-level-reference-and-protected-range'),
  tile: extract('tile-bytes-lanes-and-correction'),
  attention: extract('smoothattention-rope-pairing'),
};
assert.equal([...article.matchAll(/^```python\s*$/gm)].length, 3, 'Exercise every W4A8 Python block');
assert.match(article, /^date: 2026-08-04 09:00:00 \+0800$/m);
assert.match(article, /^author: iStar$/m);
assert.match(article, /^series_order: 100$/m);

// The source's two generic reshape/permutation steps, rational rounding and
// dense matrix reconstruction are different oracles from the post's closed
// tile mapping. These checks do not execute official Torch/CUDA/export code.
const program = String.raw`
import ast
import contextlib
from fractions import Fraction as F
import io
import itertools
import json
import math
import sys

blocks=json.load(sys.stdin)
scopes={}
passed=0
def check(name,action):
    global passed
    action();passed+=1;print('PASS',name)

def close(a,b):
    assert math.isfinite(a) and math.isclose(a,float(b),rel_tol=1e-11,abs_tol=1e-12),(a,b)

def fails(fn,*args):
    try: fn(*args)
    except ValueError: return
    raise AssertionError(('expected ValueError',fn.__name__,args))

def nearest_integer(value):
    lower=value.numerator//value.denominator
    return min([lower,lower+1],key=lambda code:(abs(value-code),code%2))

def actual_blocks():
    for name,source in blocks.items():
        ast.parse(source)
        captured,scope=io.StringIO(),{}
        with contextlib.redirect_stdout(captured):
            exec(compile(source,'article:w4a8-'+name,'exec'),scope)
        assert captured.getvalue().strip()
        scopes[name]=scope
    assert scopes['range']['count']==2303960
    assert (scopes['range']['observed_min'],scopes['range']['observed_max'])==(-121,121)
    close(scopes['range']['output'],F(22,25))
check('all three manuscript Python blocks execute, including the full posted 2303960-case loop',actual_blocks)

def rational_protected_range():
    rne,encode=scopes['range']['rne_ratio'],scopes['range']['encode_group']
    # Compare exact distances to adjacent integer candidates, rather than
    # repeating the manuscript's remainder/doubled-remainder tests.
    for denominator in range(1,31):
        for numerator in range(-256,257):
            assert rne(numerator,denominator)==nearest_integer(F(numerator,denominator))
    nearest={s:{q:nearest_integer(F(q,s)) for q in range(-119,120)} for s in range(1,17)}
    expected_count,lo_seen,hi_seen=0,0,0
    for lo in range(-119,120):
        for hi in range(lo,120):
            s=max(1,nearest_integer(F(hi-lo,15)))
            z=min(15,max(0,nearest_integer(F(-lo,s))))
            targets=list(range(lo,hi+1))
            u,actual_s,actual_z,restored=encode(targets)
            assert (actual_s,actual_z)==(s,z) and 1<=s<=16
            expected_u=[min(15,max(0,nearest[s][q]+z)) for q in targets]
            assert u==expected_u
            for q,code,value in zip(targets,expected_u,restored):
                unbounded=nearest[s][q]*s
                assert abs(F(unbounded)-q)<=F(s,2)
                assert value==(code-z)*s
                assert min(unbounded,0)<=value<=max(unbounded,0)
                assert -127<=value<=127
                lo_seen=min(lo_seen,value);hi_seen=max(hi_seen,value)
                expected_count+=1
    assert expected_count==2303960 and (lo_seen,hi_seen)==(-121,121)
    assert encode([100,119])==([15,15],1,0,[15,15])
    assert max(abs(a-b) for a,b in zip([100,119],encode([100,119])[3]))==104>8
    # Final reconstructed s8 need not still be in the original +/-119 range.
    assert encode([-49,119])[3]==[-44,121]
    assert encode([-119,49])[3]==[-121,44]
    assert rne(120,16)+7==15 and rne(120+7*16,16)==14
    assert (rne(120,16)+7-7)*16==128
    for args in [(True,1),(1,True),(1.,2),(1,0),(1,-2)]: fails(rne,*args)
    for values in [[],[True],[1.],[-120],[120],[math.inf]]: fails(encode,values)
check('independent Fraction RNE and all endpoint/value groups, safe-not-accurate clamp and final 121 contracts',rational_protected_range)

def permute_flat(flat,shape,permutation):
    newshape=[shape[axis] for axis in permutation]
    strides=[math.prod(shape[i+1:]) for i in range(len(shape))]
    reordered=[]
    for target_index in itertools.product(*(range(size) for size in newshape)):
        source_index=[0]*len(shape)
        for new_axis,old_axis in enumerate(permutation):
            source_index[old_axis]=target_index[new_axis]
        reordered.append(flat[sum(i*s for i,s in zip(source_index,strides))])
    return reordered,newshape

def source_tile_oracle():
    s=scopes['tile']
    # Immutable source: w4a8_linear.py at 02b2925... from_linear.
    # Replay its generic shape/permute operations, not tile_offset's formula.
    shape=[1,2,2,8,1,2,4,4]
    symbols,shape=permute_flat(list(range(1024)),shape,[0,4,3,6,1,5,2,7])
    symbols,shape=permute_flat(symbols,shape,[0,1,2,3,5,6,7,4])
    assert shape==[1,1,8,4,2,2,4,2] and sorted(symbols)==list(range(1024))
    seen=set()
    for n in range(32):
        for k in range(32):
            offset,shift=s['tile_offset'](n,k)
            assert shift in [0,4] and 0<=offset<512
            assert symbols[2*offset+shift//4]==n*32+k
            seen.add((offset,shift))
    assert len(seen)==1024
    assert [(symbols[2*i]//32,symbols[2*i]%32,symbols[2*i+1]//32,symbols[2*i+1]%32) for i in [0,4,8]]==[(0,0,16,0),(8,0,24,0),(0,16,16,16)]
    for values in [s['tile'],[[(n*11+k*7+3)%16 for k in range(32)] for n in range(32)]]:
        logical=[v for row in values for v in row]
        expected=bytes(logical[symbols[2*i]]+(logical[symbols[2*i+1]]<<4) for i in range(512))
        assert s['pack_tile'](values)==expected
    # Independent arbitrary physical bytes exercise all 256 bit patterns;
    # decoding is not checked solely on this same encoder's output.
    arbitrary=bytes((i*37+128)%256 for i in range(512))
    assert set(arbitrary)==set(range(256))
    expected=[0]*1024
    for i,byte in enumerate(arbitrary):
        expected[symbols[2*i]]=byte%16
        expected[symbols[2*i+1]]=byte//16
        signed=byte if byte<128 else byte-256
        assert (signed&255)==byte  # int8 container reinterprets the same bits.
    expected=[expected[start:start+32] for start in range(0,1024,32)]
    assert s['unpack_tile'](arbitrary)==expected
    assert s['unpack_tile'](bytearray(arbitrary))==expected
    for n,k in [(-1,0),(32,0),(0,32),(0,-1),(True,0),(0,1.)]: fails(s['tile_offset'],n,k)
    fails(s['pack_tile'],[])
    for bad in [True,-1,16,1.]:
        matrix=[[0]*32 for _ in range(32)];matrix[0][0]=bad;fails(s['pack_tile'],matrix)
    for packet in [[256]*512,[-1]*512,b'\x00'*511,b'\x00'*513,'x'*512]: fails(s['unpack_tile'],packet)
check('generic source reshape/permutation oracle identifies every 1024 nibble position and all physical byte patterns',source_tile_oracle)

def packed_lane_arithmetic():
    # Every 4-lane UINT4 combination at the largest protected scale; word
    # multiplication matches independent byte multiplication, without carry.
    for digits in itertools.product(range(16),repeat=4):
        word=int.from_bytes(bytes(digits),'little')
        product=word*16
        assert product<2**32
        assert product.to_bytes(4,'little')==bytes(d*16 for d in digits)
    for scale in range(1,17):
        for code in range(16):
            for zero in range(16):
                product=code*scale
                assert 0<=product<=240<256
                offset=(-zero*scale)%256
                byte=(product+offset)%256
                interpreted=byte if byte<128 else byte-256
                target=(code-zero)*scale
                if -128<=target<=127: assert interpreted==target
                else: assert interpreted!=target
    assert (15*16+128)%256==112
    assert (0*16+128)%256==128  # interpreted as signed -128, not unsigned 128.
    assert (15*16)%256==240 and 240-256==-16  # invalid signed target 240 wraps.
    s=scopes['tile']
    assert s['signed']==[-128,-16,0,112]
    assert s['normal_add']!=s['lane_add']
    # A larger scalar can cause inter-byte carry; byte-wise add must not.
    assert int.from_bytes(b'\x0f\x00\x00\x00','little')*18==270
    assert (270>>8)&255==1
check('all boundary-scale lane words, unsigned 128/240, final signed interpretation and non-saturating isolated byte addition',packed_lane_arithmetic)

def dense_groups_and_corrections():
    m,k,n,g=3,6,2,2
    a=[[(i*7+j*3)%15-7 for j in range(k)] for i in range(m)]
    sx=[F(1,2),F(1,4),F(1,8)];s0=[F(1,8),F(1,4)]
    u=[[(j+channel*3)%16 for channel in range(n)] for j in range(k)]
    scales=[[1,2],[3,1],[2,4]];zeros=[[2,3],[1,2],[3,0]]
    qw=[[(u[j][channel]-zeros[j//g][channel])*scales[j//g][channel] for channel in range(n)] for j in range(k)]
    assert all(-128<=v<=127 for row in qw for v in row)
    dense_x=[[sx[i]*v for v in row] for i,row in enumerate(a)]
    dense_w=[[s0[channel]*v for channel,v in enumerate(row)] for row in qw]
    for i in range(m):
        for channel in range(n):
            reconstructed=sum(dense_x[i][j]*dense_w[j][channel] for j in range(k))
            grouped=sum(scales[group][channel]*sum(a[i][j]*(u[j][channel]-zeros[group][channel]) for j in range(group*g,(group+1)*g)) for group in range(k//g))*sx[i]*s0[channel]
            accumulator=sum(a[i][j]*qw[j][channel] for j in range(k))*sx[i]*s0[channel]
            assert reconstructed==grouped==accumulator
    assert sum([1*2,2*3])==8 and 2*sum([1,2])==6
    s=scopes['tile']
    assert s['exact']==F(5,16) and s['extra_approximation']==F(7,20)
    assert s['extra_approximation']-s['exact']==F(3,80)
    # Exact expansion, with a nonzero column zero point and independent
    # original-X correction; this is not a GPU arithmetic precision test.
    xhat,x,u,z,sw=s['xhat'],s['x'],s['u'],s['z'],s['sw']
    direct=sum(v*(code-z)*sw for v,code in zip(xhat,u))
    assert direct==s['exact']
    assert s['extra_approximation']-direct==z*sw*(sum(xhat)-sum(x))
check('rectangular dense rational GEMM matches group restoration/epilogue while global scale and original-X correction differ',dense_groups_and_corrections)

def smoothattention_pair_oracle():
    s=scopes['attention']
    assert s['original']==-1 and s['dot'](s['pre_q'],s['rope90'](s['pre_k']))!=s['original']
    def rotate(row,turns,pairs):
        result=list(row)
        for turn,(a,b) in zip(turns,pairs):
            if turn%4==0: result[a],result[b]=row[a],row[b]
            elif turn%4==1: result[a],result[b]=-row[b],row[a]
            elif turn%4==2: result[a],result[b]=-row[a],-row[b]
            else: result[a],result[b]=row[b],-row[a]
        return result
    q,k=[F(1),F(2),F(0),F(1)],[F(1,2),F(-1),F(2),F(0)]
    pairs=[(0,2),(1,3)]  # split-half, unlike adjacent pairs.
    qp,kp=rotate(q,[0,1],pairs),rotate(k,[1,0],pairs)
    expected=sum(a*b for a,b in zip(qp,kp))
    for scales in [[F(2),F(3),F(2),F(3)],[F(1,2),F(4),F(1,2),F(4)]]:
        pre_q,pre_k=s['scale_pair'](list(map(float,q)),list(map(float,k)),list(map(float,scales)))
        actual=s['dot'](rotate(pre_q,[0,1],pairs),rotate(pre_k,[1,0],pairs))
        close(actual,expected)
    # Paired the wrong way (adjacent rather than split-half) does not commute.
    bad=[2.,2.,3.,3.]
    pre_q,pre_k=s['scale_pair'](list(map(float,q)),list(map(float,k)),bad)
    assert not math.isclose(s['dot'](rotate(pre_q,[0,1],pairs),rotate(pre_k,[1,0],pairs)),float(expected))
    # After RoPE, any finite positive diagonal scale can cancel algebraically.
    post_q,post_k=s['scale_pair'](list(map(float,qp)),list(map(float,kp)),bad)
    close(s['dot'](post_q,post_k),expected)
    assert s['common_scale']>0
    for scale in [[0.,1.],[-1.,1.],[math.inf,1.],[math.nan,1.]]: fails(s['scale_pair'],[1.,0.],[0.,1.],scale)
    for q,k,scale in [([],[],[]),([1.],[1.,2.],[1.]),([math.inf,0.],[0.,1.],[1.,1.]),([1e308],[1.],[2.]),([1.],[1e308],[1e-308]),([1e-200],[1e-100],[1e-200]),([1e-100],[1e-200],[1e200])]: fails(s['scale_pair'],q,k,scale)
    fails(s['dot'],[1e308],[2.]);fails(s['dot'],[1.],[1.,2.])
    for row in [[],[1.],[1.,2.,3.],[math.inf,0.]]: fails(s['rope90'],row)
check('independent split-half RoPE pair conditions, arbitrary post-RoPE cancellation and finite/underflow domain',smoothattention_pair_oracle)

def source_buffer_budget():
    # Tensor allocations in the fixed Python class: qweight int8[N,K/2],
    # s1_scales fp16[N], and s2_scale/s2_scaledzero int8[K/G,N].
    for k,n,g in [(128,32,128),(4096,32,128),(4096,64,128),(8192,96,128)]:
        payload=sum(1 for _ in range(n) for _ in range(k//2))
        first_scales=sum(2 for _ in range(n))
        group_scales=sum(1 for _ in range(k//g) for _ in range(n))
        scaled_zeros=sum(1 for _ in range(k//g) for _ in range(n))
        actual=F(payload+first_scales+group_scales+scaled_zeros,k*n)
        assert actual==F(1,2)+F(2,g)+F(2,k)
    assert F(1,2)+F(2,128)+F(2,4096)==F(1057,2048)
    close(float(F(1057,2048)),.51611328125)
    # Optional bias is another fp16[N]; per-channel uses fp16 scaled-zero,
    # not the same grouped int8 metadata budget.
    assert F(2*32,128*32)==F(2,128)
    assert F(128*32//2+2*32+2*32,128*32)==F(1,2)+F(4,128)
    # Byte dtype alone certifies neither raw zero 0..15 nor all group sizes;
    # only the fixed G128 dispatch is sourced, not this small CPU group G2.
    assert 20<128 and not 0<=20<=15
check('enumerated actual metadata container byte costs, bias/channel distinction and logical-domain/dispatch limits',source_buffer_budget)

assert passed==7
print(json.dumps({'groups':passed,'scope':'actual manuscript Python with independent CPU rational/reshape/lane/GEMM/index oracles; no official converter, LMQuant, Torch, CUDA, model-quality or GPU performance execution'}))
`;

const result = spawnSync(process.env.BLOG_PYTHON || 'python', ['-c', program], {
  input: JSON.stringify(blocks), encoding: 'utf8', maxBuffer: 4 * 1024 * 1024,
});
if (result.stdout) process.stdout.write(result.stdout);
if (result.stderr) process.stderr.write(result.stderr);
assert.equal(result.error, undefined, result.error?.message);
assert.equal(result.status, 0, 'W4A8 manuscript examples or independent oracles failed');
