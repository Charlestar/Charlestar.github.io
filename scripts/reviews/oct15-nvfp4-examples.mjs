import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';

const article = readFileSync(new URL('../../_posts/2026-10-15-nvfp4-mxfp4-format.md', import.meta.url), 'utf8');
function extract(name) {
  const parts = article.split(`<!-- example: nvfp4-mxfp4-${name} -->`);
  assert.equal(parts.length, 2, `Missing or duplicate FP4 marker: ${name}`);
  const block = parts[1].match(/^\s*```python\r?\n([\s\S]*?)```/);
  assert.ok(block, `Missing real FP4 Python: ${name}`);
  return block[1];
}
const blocks = {
  codes: extract('e2m1-codes-ties-packing'),
  nv: extract('scales-error-and-encoder'),
  mx: extract('groups-transpose-and-byte-budget'),
};
assert.equal([...article.matchAll(/^```python\s*$/gm)].length, 3, 'Exercise all FP4 Python blocks');
assert.match(article, /^date: 2026-10-15$/m);
assert.match(article, /^author: iStar$/m);
assert.match(article, /^series_order: 110$/m);

// Exact-rational binades and a search through legal E8M0 thresholds do not
// reuse the post's bit decoder, frexp selector, or exporter implementation.
const program = String.raw`
import ast
import contextlib
from fractions import Fraction as F
import io
import itertools
import json
import math
import struct
import sys

blocks=json.load(sys.stdin)
scopes={}
passed=0
def check(name,action):
    global passed
    action();passed+=1;print('PASS',name)

def close(a,b):
    assert math.isfinite(a) and math.isclose(a,float(b),rel_tol=2e-12,abs_tol=1e-28),(a,b)

def fails(fn,*args):
    try: fn(*args)
    except ValueError: return
    raise AssertionError(('expected ValueError',fn.__name__,args))

def pow2(exponent):
    return F(2**exponent) if exponent>=0 else F(1,2**-exponent)

# Generate values by exponent binades, independent of bit shifts/divmod.
payload=[F(0),F(1,2)]+[F(significand,2)*pow2(e) for e in range(3) for significand in [2,3]]
e4=[F(k,512) for k in range(8)]
for exponent in range(-6,9):
    e4.extend(F(significand,8)*pow2(exponent) for significand in range(8,15 if exponent==8 else 16))
e8=[pow2(exponent) for exponent in range(-127,128)]
assert len(payload)==8 and len(e4)==127 and len(e8)==255

def nearest_code(value,values):
    return min(range(len(values)),key=lambda i:(abs(values[i]-value),i%2))

def payload_oracle(value):
    sign=8 if math.copysign(1.,value)<0 else 0
    return sign+nearest_code(min(F.from_float(abs(float(value))),F(6)),payload)

def actual_blocks():
    for name,source in blocks.items():
        ast.parse(source)
        scope,capture={},io.StringIO()
        with contextlib.redirect_stdout(capture):
            exec(compile(source,'article:nvfp4-mxfp4-'+name,'exec'),scope)
        assert capture.getvalue().strip()
        scopes[name]=scope
    assert scopes['codes']['packed']==bytes.fromhex('1032547698badcfe')
    assert (scopes['mx']['mx_bytes'],scopes['mx']['nv_bytes'])==(544,580)
check('all three actual manuscript Python blocks execute in independent namespaces',actual_blocks)

def e2m1_and_bytes():
    s,nv,mx=scopes['codes'],scopes['nv'],scopes['mx']
    for code in range(16):
        expected=float(payload[code%8])*(-1 if code>=8 else 1)
        for decoder in [s['decode4'],nv['d4']]:
            actual=decoder(code)
            assert actual==expected
            assert math.copysign(1.,actual)==(-1. if code>=8 else 1.)
        assert s['encode4'](expected)==nv['q4'](expected)==code
    midpoints=[(a+b)/2 for a,b in zip(payload,payload[1:])]
    assert [payload[nearest_code(x,payload)] for x in midpoints]==[0,1,1,2,2,4,4]
    values=[n/64 for n in range(-1024,1025)]+[0.,-0.,1e-320,-1e-320,sys.float_info.max,-sys.float_info.max]
    for midpoint in midpoints:
        for value in [math.nextafter(float(midpoint),-math.inf),float(midpoint),math.nextafter(float(midpoint),math.inf)]:
            values.extend([value,-value])
    for value in values:
        expected=payload_oracle(value)
        assert s['encode4'](value)==nv['q4'](value)==expected
        decoded=mx['qdq4'](value)
        assert decoded==float(payload[expected%8])*(-1 if expected>=8 else 1)
        assert math.copysign(1.,decoded)==math.copysign(1.,value)
    # Decode physical input not produced solely by the same packer.
    for byte in range(256):
        expected=[byte%16,byte//16]
        assert s['unpack4'](bytes([byte]))==expected
        assert s['unpack4'](bytearray([byte]))==expected
        assert s['pack4'](expected)==bytes([byte])
    assert s['pack4'](list(range(16)))==bytes.fromhex('1032547698badcfe')
    for bad in [-1,16,True,1.,'1']:
        fails(s['decode4'],bad);fails(nv['d4'],bad);fails(s['pack4'],[0,bad])
    for bad in [[256],[-1],[1],1,'a',memoryview(b'x')]: fails(s['unpack4'],bad)
    fails(s['pack4'],[1,2,3])
    for bad in [math.nan,math.inf,-math.inf]:
        fails(s['encode4'],bad);fails(nv['q4'],bad);fails(mx['qdq4'],bad)
check('independent E2M1 binades, signed zero, every tie/neighbor and all 256 physical bytes',e2m1_and_bytes)

def scale_codebooks_and_source_boundaries():
    s=scopes['nv']
    for code,expected in enumerate(e4):
        assert s['positive_e4m3'](code)==float(expected)
        assert s['E4'][code]==float(expected)
        assert s['scale_code'](float(expected))==code
    assert e4[1]==F(1,512) and e4[8]==F(1,64) and e4[-1]==448
    for left,right in zip(e4,e4[1:]):
        midpoint=float((left+right)/2)
        for value in [math.nextafter(midpoint,-math.inf),midpoint,math.nextafter(midpoint,math.inf)]:
            assert s['scale_code'](value)==nearest_code(F.from_float(value),e4)
    assert s['scale_code'](sys.float_info.max)==126
    for bad in [-1,127,255,True,1.]: fails(s['positive_e4m3'],bad)
    for bad in [-1.,math.inf,math.nan]: fails(s['scale_code'],bad)
    assert e8[0]==F(1,2**127) and e8[-1]==2**127
    assert all(value>0 and math.isfinite(float(value)) for value in e8)
    # The finite 255 E8M0 values are codes 0..254, not code 255 (NaN).
    maximum_fp32=(F(2)-pow2(-23))*pow2(127)
    assert 6*e8[-1]>maximum_fp32
    try: struct.pack('!f',float(6*e8[-1]))
    except OverflowError: pass
    else: raise AssertionError('FP32 overflow boundary was not exercised')
    # Source-read distinction, not Torch execution: its MX cast counts strict
    # greater-than midpoints; its NV sign comparison does not read signbit.
    mids=[(a+b)/2 for a,b in zip(payload,payload[1:])]
    lower=[sum(midpoint>bound for bound in mids) for midpoint in mids]
    rne=[nearest_code(midpoint,payload) for midpoint in mids]
    assert lower==list(range(7)) and [i for i,(a,b) in enumerate(zip(lower,rne)) if a!=b]==[1,3,5]
    assert not (-0.0<0) and math.copysign(1.,-0.)<0
check('all finite positive E4M3/E8M0 scales, E4 tie neighbors, FP32 product overflow and distinct source contracts',scale_codebooks_and_source_boundaries)

def nv_global_scale_and_mse():
    s=scopes['nv']
    # Derive binary32's quantization quantum from exact rational exponents,
    # then compare its neighboring integer multiples; no struct oracle.
    def round_fp32(value):
        exponent=value.numerator.bit_length()-value.denominator.bit_length()
        if value<pow2(exponent): exponent-=1
        quantum=pow2(max(-149,exponent-23))
        normalized=value/quantum
        lower=normalized.numerator//normalized.denominator
        rounded=min([lower,lower+1],key=lambda code:(abs(code-normalized),code%2))
        return rounded*quantum
    maxima=[1.,2688.,2688.*.1,2688.*1.5,2688.*2**-149,2688.*2**-140,2688.*2**100,
            2688.*(1.+2**-24),2688.*(1.+3*2**-24),2688.*float((2-pow2(-23))*pow2(127))]
    for amax in maxima:
        a,scale_codes,codes,actual=s['nv_reference']([amax]*16+[-amax/2]*16)
        # The manuscript divides using Python double, then converts to FP32.
        assert F.from_float(a)==round_fp32(F.from_float(amax/2688))
        assert len(scale_codes)==2 and len(codes)==len(actual)==32
        assert all(0<c<127 for c in scale_codes)
        for g in range(2):
            b=e4[scale_codes[g]]
            for i in range(g*16,(g+1)*16):
                expected=float(F.from_float(a)*b*payload[codes[i]%8])*(-1 if codes[i]>=8 else 1)
                assert actual[i]==expected
        gamma=1/a
        close(1/(gamma*gamma),a*a)
    # Exact decimal input/error oracle, independent of the article's doubles.
    def reconstructed_exact(values,scale):
        return [scale*payload[nearest_code(abs(v/scale),payload)]*(-1 if v<0 else 1) for v in values]
    def exact_mse(values,rebuilt): return sum((x-y)**2 for x,y in zip(values,rebuilt))/len(values)
    pattern=[F(0),F(3,10),F(3,5),F(9,10),F(6,5),F(9,5),F(12,5),F(18,5)]*2
    for scale,expected in [(F(3,5),F(0)),(F(5,8),F(137,25600)),(F(1),F(23,400))]:
        assert exact_mse(pattern,reconstructed_exact(pattern,scale))==expected
        close(s['mse'](list(map(float,pattern)),s['reconstruct'](list(map(float,pattern)),float(scale))),expected)
    counter=[F(2,5)]*15+[F(18,5)]
    default=exact_mse(counter,reconstructed_exact(counter,F(5,8)))
    alternative=exact_mse(counter,reconstructed_exact(counter,F(7,8)))
    assert (default,alternative)==(F(879,102400),F(199,102400)) and alternative<default
    assert s['nv_reference']([-0.]*16)[0:3]==(1.,[56],[8]*16)
    assert all(math.copysign(1.,x)<0 for x in s['nv_reference']([-0.]*16)[3])
    assert s['nv_reference']([2688.]*16+[0.]*16)[1][1]==56
    fails(s['nv_reference'],[2688.*2**-150]*16)
    just_above=math.nextafter(2688.*2**-150,math.inf)
    assert s['nv_reference']([just_above]*16)[0]==2**-149
    for bad in [[],[1.]*17,[math.nan]*16,[math.inf]*16,[1e-50]*16,[sys.float_info.max]*16,[2688.]*16+[1e-10]*16]:
        fails(s['nv_reference'],bad)
check('NV global FP32 rounding, exact scale/error counterexamples, reciprocal direction and zero/tiny/overflow policy',nv_global_scale_and_mse)

def exact_e8_encoder_boundaries():
    s=scopes['mx']
    def scale_oracle(value):
        if value==0: return F(1)
        maximum=F.from_float(value)
        matches=[scale for scale in e8 if maximum<=6*scale]
        if not matches: raise ValueError('outside finite E8 range')
        return matches[0]
    tests=[0.,math.nextafter(0.,1.),1e-50,1.,7.,sys.float_info.max]
    for scale in e8:
        boundary=float(6*scale)
        tests.extend([math.nextafter(boundary,0.),boundary,math.nextafter(boundary,math.inf)])
    for value in tests:
        try: expected=scale_oracle(value)
        except ValueError: fails(s['mx_scale'],value);continue
        assert F.from_float(s['mx_scale'](value))==expected,(value,s['mx_scale'](value),expected)
        assert F.from_float(value)<=6*expected
        if expected>e8[0] and value!=0: assert F.from_float(value)>3*expected
    # Reproduce the old log2 failure independently, so a regression test is
    # not restricted to the ordinary 0.3/6 transpose construction.
    value=math.nextafter(6.*2**100,math.inf)
    assert math.ceil(math.log2(value/6))==100
    assert s['mx_scale'](value)==2.**101
    fails(s['mx_scale'],math.nextafter(6.*2**127,math.inf))
    for value in [-1.,math.nan,math.inf]: fails(s['mx_scale'],value)
    # OCP recommended conversion and this independent ceil encoder differ.
    ocp_scale=F(4,4)
    assert ocp_scale==1 and s['mx_scale'](7.)==2.
    assert float(ocp_scale*payload[nearest_code(F(7)/ocp_scale,payload)])==6.
check('exact search of all E8M0 thresholds and adjacent doubles, including log2 false-integer and upper-domain rejection',exact_e8_encoder_boundaries)

def transpose_and_axis_oracle():
    s=scopes['mx']
    def quantize_rows_oracle(matrix):
        result=[]
        for row in matrix:
            rebuilt=[]
            for start in range(0,len(row),32):
                block=list(map(F.from_float,row[start:start+32]))
                maximum=max(map(abs,block))
                scale=next((scale for scale in e8 if maximum<=6*scale),None) if maximum else F(1)
                assert scale is not None
                rebuilt.extend(float(scale*payload[nearest_code(abs(x/scale),payload)])*(-1 if math.copysign(1.,float(x))<0 else 1) for x in block)
            result.append(rebuilt)
        return result
    weight=[[.3]*32 for _ in range(32)];weight[0][0]=6.
    original_rows=quantize_rows_oracle(weight)
    transposed=[[weight[row][column] for row in range(32)] for column in range(32)]
    transposed_rows=quantize_rows_oracle(transposed)
    actual_a=s['transpose'](s['mx_rows'](weight))
    actual_b=s['mx_rows'](s['transpose'](weight))
    assert actual_a==[[original_rows[row][column] for row in range(32)] for column in range(32)]
    assert actual_b==transposed_rows
    assert actual_a[0][1]==F(1,4) and actual_b[0][1]==F(1,2)
    assert sum(a!=b for ra,rb in zip(actual_a,actual_b) for a,b in zip(ra,rb))==62
    # Rectangular input with two independent groups per row, not only a
    # symmetric matrix where transpose superficially preserves the input.
    matrix=[[float(((row*13+col*7)%31-15)*pow2((col//32)*3-row)) for col in range(64)] for row in range(3)]
    assert s['mx_rows'](matrix)==quantize_rows_oracle(matrix)
    assert s['transpose'](matrix)==[[matrix[row][col] for row in range(3)] for col in range(64)]
    # Flat regrouping can cross a logical row even if the total is divisible.
    logical=[list(range(row*17,(row+1)*17)) for row in range(32)]
    flat=list(itertools.chain.from_iterable(logical))
    assert len(flat)%32==0 and 17%32!=0 and flat[16:18]==[16,17]
    fails(s['mx_rows'],logical)
    for bad in [[],[[]],[[1.]*31],[[1.]*32,[1.]*31],[[math.inf]*32]]: fails(s['mx_rows'],bad)
    for bad in [[],[[]],[[1.,2.],[3.]]]: fails(s['transpose'],bad)
check('independent exact group reconstruction, all transpose entries and rectangular/flat-axis counterexamples',transpose_and_axis_oracle)

def grouped_dense_gemm():
    s=scopes['mx']
    assert s['group_result']==s['direct_result']==-120 and s['wrong_result']==-8
    m,k,n,g=3,64,2,16
    px=[[payload[(i*3+j)%8]*(-1 if (i+j)%3==0 else 1) for j in range(k)] for i in range(m)]
    pw=[[payload[(j+channel*5)%8]*(-1 if (j+channel)%2 else 1) for channel in range(n)] for j in range(k)]
    ax,aw=F(1,2),F(3,4)
    bx=[[F(i+group+1,4) for group in range(k//g)] for i in range(m)]
    bw=[[F(group+channel+1,8) for channel in range(n)] for group in range(k//g)]
    x=[[ax*bx[i][j//g]*px[i][j] for j in range(k)] for i in range(m)]
    w=[[aw*bw[j//g][channel]*pw[j][channel] for channel in range(n)] for j in range(k)]
    for i in range(m):
        for channel in range(n):
            dense=sum(x[i][j]*w[j][channel] for j in range(k))
            grouped=ax*aw*sum(bx[i][group]*bw[group][channel]*sum(px[i][j]*pw[j][channel] for j in range(group*g,(group+1)*g)) for group in range(k//g))
            assert dense==grouped
    # Different group boundaries require a common refinement, not matching
    # their group indices. Both groups here are small CPU algebra examples.
    xscales,wscales=[F(1),F(2),F(4)],[F(1,2),F(3)]
    xv,wv=[F(j-2) for j in range(12)],[F((j%5)-2) for j in range(12)]
    direct=sum(xv[j]*xscales[j//4]*wv[j]*wscales[j//6] for j in range(12))
    edges=sorted(set([0,4,8,12]+[0,6,12]))
    refined=sum(xscales[left//4]*wscales[left//6]*sum(xv[j]*wv[j] for j in range(left,right)) for left,right in zip(edges,edges[1:]))
    assert direct==refined
check('non-square Fraction GEMM and different-axis common refinement versus a globally extracted wrong scale',grouped_dense_gemm)

def bytes_padding_and_scalar_guards():
    s,nv=scopes['mx'],scopes['nv']
    for group,global_bytes in [(16,4),(32,0)]:
        for groups in [1,2,7,32]:
            length=group*groups
            payload_bytes=sum(1 for _ in range(0,length,2))
            local_bytes=sum(1 for _ in range(groups))
            assert s['storage_bytes'](length,group,global_bytes)==payload_bytes+local_bytes+global_bytes
    assert (s['storage_bytes'](1024,32,0),s['storage_bytes'](1024,16,4))==(544,580)
    assert F(s['storage_bytes'](16,16,4)*8,16)==F(13,2)
    # Independent hypothetical row-padding budget, not an implemented exporter.
    # Two rows of 17 logical values need 2 complete NV groups per row, not
    # ceil(total/G)=3 groups; retaining original shape does not delete padding.
    rows,width,group=2,17,16
    group_count=sum(1 for row in range(rows) for start in range(0,width,group))
    payload_bytes=sum(group//2 for _ in range(group_count))
    padded=group_count*group
    assert group_count==4 and payload_bytes+group_count+4==40
    assert padded==64 and rows*width==34 and math.ceil(rows*width/group)==3
    for args in [(True,16,4),(16,True,4),(16,16,True),(16.,16,4),(16,0,4),(16,15,4),(17,16,4),(16,16,-1),(0,16,4)]: fails(s['storage_bytes'],*args)
    for args in [([],[]),([1.],[1.,2.]),([math.inf],[0.]),([1e308],[-1e308]),([1e-200],[0.]),([1e-161]+[0.]*999,[0.]*1000)]: fails(nv['mse'],*args)
    fails(nv['reconstruct'],[1.],0.)
    fails(nv['reconstruct'],[math.inf],1.)
    fails(nv['reconstruct'],[1e308],1e-308)
    fails(nv['reconstruct'],[sys.float_info.max],sys.float_info.max/5.5)
check('explicit complete-block and per-row ceil/padding byte counts plus finite/typed scalar failure contracts',bytes_padding_and_scalar_guards)

assert passed==8
print(json.dumps({'groups':passed,'scope':'actual standard-library manuscript Python; independent exact binade, threshold, grouped GEMM, byte and shape oracles; no Torch/ModelOpt/CUDA, model or GPU execution'}))
`;

const result = spawnSync(process.env.BLOG_PYTHON || 'python', ['-c', program], {
  input: JSON.stringify(blocks), encoding: 'utf8', maxBuffer: 4 * 1024 * 1024,
});
if (result.stdout) process.stdout.write(result.stdout);
if (result.stderr) process.stderr.write(result.stderr);
assert.equal(result.error, undefined, result.error?.message);
assert.equal(result.status, 0, 'FP4 manuscript examples or independent oracles failed');
