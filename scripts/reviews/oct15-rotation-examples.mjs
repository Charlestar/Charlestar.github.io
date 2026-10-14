import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';

const article = readFileSync(new URL('../../_posts/2026-10-15-quarot-spinquant-rotation.md', import.meta.url), 'utf8');
function extract(name) {
  const parts = article.split(`<!-- example: rotation-${name} -->`);
  assert.equal(parts.length, 2, `Missing or duplicate rotation marker: ${name}`);
  const block = parts[1].match(/^\s*```python\r?\n([\s\S]*?)```/);
  assert.ok(block, `Missing actual rotation Python: ${name}`);
  return block[1];
}
const blocks = {
  quant: extract('helpful-and-harmful-quantization'),
  graph: extract('norm-residual-graph-and-rope'),
  cayley: extract('cayley-orthogonality-principle'),
};
assert.equal([...article.matchAll(/^```python\s*$/gm)].length, 3, 'Exercise every rotation Python block');
assert.match(article, /^date: 2026-10-15$/m);
assert.match(article, /^author: iStar$/m);
assert.match(article, /^series_order: 90$/m);
assert.match(article, /2024-03-30/);
assert.match(article, /2024-05-26/);

// Fraction code-grid distances, Decimal nonlinear network evaluation, explicit
// token/head indices, and exact Gauss-Jordan solves provide distinct oracles.
// None of these runs the external Torch optimizer or INT4 GPU kernels.
const program = String.raw`
import ast
import cmath
import contextlib
import copy
from decimal import Decimal as D, localcontext
from fractions import Fraction as F
import io
import json
import math
import sys

blocks = json.load(sys.stdin)
scopes = {}
passed = 0
def check(name, action):
    global passed
    action()
    passed += 1
    print('PASS',name)

def close(a,b,rel_tol=1e-10,abs_tol=1e-11):
    assert math.isfinite(a) and math.isclose(a,float(b),rel_tol=rel_tol,abs_tol=abs_tol),(a,b)

def vector_close(a,b):
    assert len(a)==len(b)
    for x,y in zip(a,b): close(x,y)

def fails(fn,*args):
    try: fn(*args)
    except ValueError: return
    raise AssertionError(('expected ValueError',fn.__name__,args))

def transpose(a):
    return [list(col) for col in zip(*a)]

def product(a,b):
    return [[sum(x*y for x,y in zip(row,col)) for col in transpose(b)] for row in a]

def eye(n):
    return [[F(i==j) for j in range(n)] for i in range(n)]

def exact_qdq(values,bound=7):
    values=list(map(F,values))
    peak=max(map(abs,values))
    if peak==0: return [F(0)]*len(values)
    grid=[peak*F(c,bound) for c in range(-bound,bound+1)]
    return [min(grid,key=lambda v:(abs(v-x),(v*bound/peak).numerator%2)) for x in values]

def actual_blocks():
    for name,source in blocks.items():
        ast.parse(source)
        captured,scope=io.StringIO(),{}
        with contextlib.redirect_stdout(captured):
            exec(compile(source,'article:rotation-'+name,'exec'),scope)
        assert captured.getvalue().strip()
        scopes[name]=scope
    vector_close(scopes['graph']['logits'],scopes['graph']['rotated_logits'])
    assert scopes['cayley']['gram_error'](scopes['cayley']['cayley'])<1e-12
check('three real manuscript Python blocks execute separately with their printed outputs and assertions',actual_blocks)

def exact_grid_and_rotated_dot():
    quant,had,dot=(scopes['quant'][name] for name in ['qdq','hadamard2','dot'])
    for step in [F(1,8),F(1,2),F(1),F(8)]:
        for i in range(-56,57):
            values=[-7*step,step*F(i,8),7*step]
            assert list(map(F,quant(list(map(float,values)))))==exact_qdq(values)
    # In two dimensions each rotated operand has a common 1/sqrt(2).
    # It cancels from code selection and its product is exactly 1/2, so
    # the independent dot oracle contains no floating Hadamard transform.
    x=[F(10),F(1,5)]
    for w,expected_plain,expected_rotated in [([F(1),F(1)],F(10),F(51,5)),([F(1,10),F(1)],F(10,7),F(561,700))]:
        raw_x=[sum(x),x[0]-x[1]]
        raw_w=[sum(w),w[0]-w[1]]
        plain=sum(a*b for a,b in zip(exact_qdq(x),exact_qdq(w)))
        rotated=sum(a*b for a,b in zip(exact_qdq(raw_x),exact_qdq(raw_w)))/2
        assert plain==expected_plain and rotated==expected_rotated
        actual=dot(quant(had(list(map(float,x)))),quant(had(list(map(float,w)))))
        close(actual,rotated)
    h4=[[F((-1)**((i&j).bit_count()),2) for j in range(4)] for i in range(4)]
    assert product(transpose(h4),h4)==eye(4)
    # A uniform row becomes concentrated, while an impulse spreads.
    assert max(map(abs,product([[F(1)]*4],h4)[0]))==2
    assert max(map(abs,product([[F(4),F(0),F(0),F(0)]],h4)[0]))==2
    raw=[[int(v*2) for v in row] for row in h4]
    assert product(transpose(raw),raw)==[[4*e for e in row] for row in eye(4)]
check('rational nearest-code grids, exact irrational-factor cancellation and normalized/unnormalized Hadamard contracts',exact_grid_and_rotated_dot)

def decimal_reference(state,blocks,head_gamma,epsilon,head,bias):
    def dec(x): return D(str(x))
    def mat(row,matrix):
        return [sum((x*dec(matrix[k][j]) for k,x in enumerate(row)),D(0)) for j in range(len(matrix[0]))]
    def norm(row,gamma,eps):
        denom=(sum((x*x for x in row),D(0))/len(row)+dec(eps)).sqrt()
        return [x*dec(g)/denom for x,g in zip(row,gamma)]
    state=list(map(dec,state))
    for block in blocks:
        n=norm(state,block['gamma'],block['epsilon'])
        gates=[x+dec(b) for x,b in zip(mat(n,block['wg']),block['bg'])]
        up=[x+dec(b) for x,b in zip(mat(n,block['wu']),block['bu'])]
        z=[g/(D(1)+(-g).exp())*u for g,u in zip(gates,up)]
        projected=mat(z,block['wd'])
        state=[x+y+dec(b) for x,y,b in zip(state,projected,block['bd'])]
    return [x+dec(b) for x,b in zip(mat(norm(state,head_gamma,epsilon),head),bias)]

def graph_all_embeddings_and_rectangular():
    s=scopes['graph']
    for embedding in s['embedding']:
        state=list(embedding)
        rotated=s['right'](state,s['rotation'])
        for block in s['blocks']:
            state=s['block_forward'](state,block)
            rotated=s['block_forward'](rotated,s['rotate_block'](block))
            vector_close(rotated,s['right'](state,s['rotation']))
        expected=s['add'](s['right'](s['gain'](s['rms'](state,s['head_epsilon']),s['head_gamma']),s['head']),s['head_bias'])
        actual=s['add'](s['right'](s['rms'](rotated,s['head_epsilon']),s['rotated_head']),s['head_bias'])
        with localcontext() as ctx:
            ctx.prec=70
            oracle=decimal_reference(embedding,s['blocks'],s['head_gamma'],s['head_epsilon'],s['head'],s['head_bias'])
        vector_close(expected,oracle)
        vector_close(actual,oracle)
    # A nonsymmetric rational 4D orthogonal matrix catches using the same
    # direction rather than inverse. Hidden4 -> gate/up3 -> hidden4 -> logits5.
    h4=[[F((-1)**((i&j).bit_count()),2) for j in range(4)] for i in range(4)]
    r=[[row[(j+1)%4] for j in range(4)] for row in h4]
    assert r!=transpose(r) and product(transpose(r),r)==eye(4)
    matrix=lambda m,n,seed:[[(i*7+j*3+seed)%11/10-.5 for j in range(n)] for i in range(m)]
    test_blocks=[dict(gamma=[1.2,.7,1.1,.6],epsilon=eps,wg=matrix(4,3,l),wu=matrix(4,3,l+2),wd=matrix(3,4,l+4),bg=[.1,-.2,.05],bu=[-.05,.2,.07],bd=[.03,-.04,.06,.02]) for l,eps in [(1,1e-5),(3,1e-4)]]
    embedding=matrix(5,4,4)
    head,bias,gamma,eps=matrix(4,5,2),[.1,-.05,.02,.06,-.03],[.9,1.1,.8,1.2],3e-5
    original_rotation,original_inverse=s['rotation'],s['inverse']
    try:
        s['rotation']=[list(map(float,row)) for row in r]
        s['inverse']=transpose(s['rotation'])
        transformed=[s['rotate_block'](b) for b in test_blocks]
        transformed_head=s['mm'](s['inverse'],s['scale_rows'](head,gamma))
        for start in embedding:
            a,b=list(start),s['right'](start,s['rotation'])
            for ba,bb in zip(test_blocks,transformed):
                a,b=s['block_forward'](a,ba),s['block_forward'](b,bb)
                vector_close(b,s['right'](a,s['rotation']))
            with localcontext() as ctx:
                ctx.prec=70
                expected=decimal_reference(start,test_blocks,gamma,eps,head,bias)
            actual=s['add'](s['right'](s['rms'](b,eps),transformed_head),bias)
            vector_close(actual,expected)
    finally:
        s['rotation'],s['inverse']=original_rotation,original_inverse
check('all embedding rows, Decimal nonlinear network oracle and nonsymmetric 4D/rectangular two-block graph',graph_all_embeddings_and_rectangular)

def graph_mutations_and_nonlinearity():
    s=scopes['graph']
    block=s['blocks'][0]
    start=s['right'](s['embedding'][1],s['rotation'])
    correct=s['rotate_block'](block)
    target=s['right'](s['block_forward'](s['embedding'][1],block),s['rotation'])
    mutations=[]
    missing_up=copy.deepcopy(correct)
    missing_up['wu']=s['mm'](s['inverse'],block['wu'])
    mutations.append(missing_up)
    missing_gate=copy.deepcopy(correct)
    missing_gate['wg']=s['mm'](s['inverse'],block['wg'])
    mutations.append(missing_gate)
    old_bias=copy.deepcopy(correct);old_bias['bd']=block['bd'];mutations.append(old_bias)
    old_basis=copy.deepcopy(correct);old_basis['wd']=block['wd'];mutations.append(old_basis)
    old_epsilon=copy.deepcopy(correct);old_epsilon['epsilon']=.1;mutations.append(old_epsilon)
    for bad in mutations:
        actual=s['block_forward'](start,bad)
        assert max(abs(x-y) for x,y in zip(actual,target))>1e-7
    # The external oracle must also be sensitive to the network endpoints,
    # not only the local block. Leaving embedding/head in the old basis fails.
    correct_state=s['rotated_state']
    expected_logits=s['logits']
    wrong_head=s['add'](s['right'](s['rms'](correct_state,s['head_epsilon']),s['head']),s['head_bias'])
    assert max(abs(x-y) for x,y in zip(wrong_head,expected_logits))>1e-3
    wrong_state=list(s['embedding'][1])
    for block in s['blocks']:
        wrong_state=s['block_forward'](wrong_state,s['rotate_block'](block))
    wrong_embedding_logits=s['add'](s['right'](s['rms'](wrong_state,s['head_epsilon']),s['rotated_head']),s['head_bias'])
    assert max(abs(x-y) for x,y in zip(wrong_embedding_logits,expected_logits))>1e-3
    x=[1.,3.]
    assert not s['close'](s['rms'](x,1e-5),s['rms'](x,.1))
    before=s['right']([s['silu'](v) for v in x],s['rotation'])
    after=[s['silu'](v) for v in s['right'](x,s['rotation'])]
    assert max(abs(a-b) for a,b in zip(before,after))>.1
    gate,up=[1.,2.],[3.,4.]
    z=s['gain'](gate,up)
    assert not s['close'](s['right'](z,s['rotation']),s['gain'](s['right'](gate,s['rotation']),s['right'](up,s['rotation'])))
    wd=[[.2,.5,-.3],[.8,-.4,.1]]
    vector_close(s['right'](s['right'](z,s['rotation']),s['mm'](s['inverse'],wd)),s['right'](z,wd))
    # A mean-removing LayerNorm counterexample, with normalization separately
    # computed in Decimal, rather than only trusting the post's own rms.
    with localcontext() as ctx:
        ctx.prec=70
        def ln(row):
            row=list(map(lambda v:D(str(v)),row));mean=sum(row)/len(row)
            centered=[v-mean for v in row]
            denom=(sum(v*v for v in centered)/len(row)+D('0.00001')).sqrt()
            return [v/denom for v in centered]
        a=ln(s['right'](x,s['rotation']))
        b=s['right'](list(map(float,ln(x))),s['rotation'])
        assert max(abs(float(v)-w) for v,w in zip(a,b))>.1
        expected_score=-D(1)/D(2).sqrt()
        close(s['original_score'],expected_score)
        close(s['post_score'],expected_score)
        close(s['pre_score'],-expected_score)
check('missing consumer/bias/epsilon/residual-basis mutations fail, and nonlinear/gating/LayerNorm movement is not valid',graph_mutations_and_nonlinearity)

def multihead_qk_value_and_mask():
    s=scopes['graph']
    h=s['rotation']; r2=[[.6,-.8],[.8,.6]]
    hq,hkv,dim,tokens=4,2,2,3
    # Two Q heads share each KV head, only as a mathematical GQA mapping;
    # this is not certification of any model wrapper's implementation.
    queries=[[[.2+head*.1,-.3],[.7,-.2+head*.05]] for head in range(hq)]
    keys=[[[1.+head*.2,.2],[-.4,.5],[.3,-.7]] for head in range(hkv)]
    values=[[[.1+head,.2],[-.2,.4],[.8,-.3]] for head in range(hkv)]
    masks=[[True,False,False],[True,True,False]]
    out=[];out_rot=[]
    for head in range(hq):
        kv=head//2; dense_rows=[];rot_rows=[]
        for m,q in enumerate(queries[head]):
            logits=[sum(F(x)*F(y) for x,y in zip(q,k)) for k in keys[kv]]
            actual=[s['dot'](s['right'](q,h),s['right'](k,h)) for k in keys[kv]]
            for a,b in zip(actual,logits): close(a,b)
            with localcontext() as ctx:
                ctx.prec=70
                exps=[(D(v.numerator)/D(v.denominator)/D(2).sqrt()).exp() if allow else D(0) for v,allow in zip(logits,masks[m])]
                probabilities=[float(v/sum(exps)) for v in exps]
            assert sum(probabilities)==1. and probabilities[2]==0.
            dense_rows.append([sum(p*v[d] for p,v in zip(probabilities,values[kv])) for d in range(dim)])
            vr=[s['right'](row,r2) for row in values[kv]]
            rot_rows.append([sum(p*v[d] for p,v in zip(probabilities,vr)) for d in range(dim)])
        out.append(dense_rows);out_rot.append(rot_rows)
    concat=[[v for head in range(hq) for v in out[head][m]] for m in range(2)]
    concat_rot=[[v for head in range(hq) for v in out_rot[head][m]] for m in range(2)]
    wo=[[(i*3+j)%7/10-.3 for j in range(3)] for i in range(hq*dim)]
    block_diag=[[r2[i%2][j%2] if i//2==j//2 else 0. for j in range(hq*dim)] for i in range(hq*dim)]
    corrected=s['mm'](s['transpose'](block_diag),wo)
    for a,b in zip(s['mm'](concat_rot,corrected),s['mm'](concat,wo)): vector_close(a,b)
    wrong=s['mm'](concat_rot,wo)
    assert max(abs(a-b) for row_a,row_b in zip(wrong,s['mm'](concat,wo)) for a,b in zip(row_a,row_b))>1e-4
    # Mixing all coordinates across independent heads preserves the sum of
    # their dots, not each head's score and its distinct softmax.
    h4=[[(-1)**((i&j).bit_count())/2 for j in range(4)] for i in range(4)]
    mixed=s['right']([1.,0.,0.,0.],h4)
    assert [sum(v*v for v in mixed[:2]),sum(v*v for v in mixed[2:])]==[.5,.5]
    assert [.5,.5]!=[1.,0.]
    assert not math.isclose(s['dot']([1.,0.],[.2,.8]),s['dot'](s['right']([1.,0.],h),[.2,.8]))
    # Wrongly using an unnormalized 2D Hadamard scales logits by two, before
    # the separate Attention 1/sqrt(head_dim) factor.
    raw_h=[[1.,1.],[1.,-1.]]
    q,k=[.2,.7],[-.4,.5]
    close(s['dot'](s['right'](q,raw_h),s['right'](k,raw_h)),2*s['dot'](q,k))
check('explicit multihead/GQA index mapping, shared R2 Value/output pairing, masks, head-mixing and normalization counterexamples',multihead_qk_value_and_mask)

def exact_inverse(matrix):
    n=len(matrix)
    rows=[[F(v) for v in row]+identity for row,identity in zip(matrix,eye(n))]
    for col in range(n):
        pivot=next(i for i in range(col,n) if rows[i][col])
        rows[col],rows[pivot]=rows[pivot],rows[col]
        value=rows[col][col];rows[col]=[v/value for v in rows[col]]
        for i in range(n):
            if i!=col:
                factor=rows[i][col]
                rows[i]=[a-factor*b for a,b in zip(rows[i],rows[col])]
    return [row[n:] for row in rows]

def cayley_independent_solve():
    s=scopes['cayley']
    for n in [2,3,4]:
        a=[[F((i-j)*(i+j+1),5) for j in range(n)] for i in range(n)]
        assert transpose(a)==[[-v for v in row] for row in a]
        for t in [F(0),F(1,10),F(-1,4),F(1)]:
            identity=eye(n)
            minus=[[identity[i][j]-t*a[i][j] for j in range(n)] for i in range(n)]
            plus=[[identity[i][j]+t*a[i][j] for j in range(n)] for i in range(n)]
            c=product(exact_inverse(minus),plus)
            assert product(transpose(c),c)==identity
            for row_a,row_b in zip(s['mm'](s['transpose'](c),c),identity): assert row_a==row_b
    expected=[[F(99,101),F(-20,101)],[F(20,101),F(99,101)]]
    for a,b in zip(s['cayley'],expected): vector_close(a,b)
    complex_reference=cmath.exp(200j*math.atan(.1))
    vector_close(s['composed'][0],[complex_reference.real,-complex_reference.imag])
    vector_close(s['composed'][1],[complex_reference.imag,complex_reference.real])
    # The fixed point equation's five finite iterations are not an exact
    # inverse. This is a standard-library arithmetic model of that fact,
    # not execution of SGDG/Torch or a statement about its model quality.
    w=[[F(0),F(-1)],[F(1),F(0)]];x=eye(2);t=F(1,5)
    y=[[x[i][j]+t*w[i][j] for j in range(2)] for i in range(2)]
    for _ in range(5):
        midpoint=[[(x[i][j]+y[i][j])/2 for j in range(2)] for i in range(2)]
        correction=product(w,midpoint)
        y=[[x[i][j]+t*correction[i][j] for j in range(2)] for i in range(2)]
    assert product(transpose(y),y)!=eye(2)
    assert max(abs(a-b) for row_a,row_b in zip(product(transpose(y),y),eye(2)) for a,b in zip(row_a,row_b))>F(1,10**8)
check('exact Gauss-Jordan Cayley solves in 2D/3D/4D, complex composition oracle and nonexact five-iteration Gram',cayley_independent_solve)

def finite_and_shape_domain():
    q,g,c=scopes['quant'],scopes['graph'],scopes['cayley']
    for values in [[],[math.nan],[math.inf],[math.ulp(0.)]]: fails(q['qdq'],values)
    for values in [[],[1.],[1.,2.,3.],[math.inf,0.],[1e308,1e308]]: fails(q['hadamard2'],values)
    for a,b in [([],[]),([1.],[1.,2.]),([math.inf],[1.]),([1e308],[2.])]: fails(q['dot'],a,b)
    for matrix in [[],[[]],[[1.],[2.,3.]],[[math.inf]],[[math.nan]]]:
        fails(g['transpose'],matrix);fails(c['transpose'],matrix)
    fails(g['right'],[1.],[[1.,2.],[3.,4.]])
    fails(g['right'],[1e308],[[2.]])
    fails(g['gain'],[1.,2.],[1.])
    fails(g['scale_rows'],[[1.],[2.]],[1.])
    for epsilon in [0.,-1.,math.inf,math.nan]: fails(g['rms'],[1.,2.],epsilon)
    fails(g['rms'],[],1e-5);fails(g['rms'],[1e308,0.],1e-5)
    assert g['silu'](-1000.)==0. and g['silu'](1000.)==1000.
    fails(g['silu'],math.inf)
    for fn in [g['mm'],c['mm']]:
        fails(fn,[[1.]],[[1.,2.],[3.,4.]])
        fails(fn,[[1e308]],[[2.]])
    assert q['qdq']([0.,-0.])==[0.,0.]
check('finite small-reference domain rejects ragged shapes, bad epsilon, underflowing steps and overflowing arithmetic',finite_and_shape_domain)

assert passed==7
print(json.dumps({'groups':passed,'scope':'actual manuscript Python with independent CPU math/network/index oracles; no official Torch optimizer, complete Transformer, packing artifact, model quality or GPU benchmark'}))
`;

const result = spawnSync(process.env.BLOG_PYTHON || 'python', ['-c', program], {
  input: JSON.stringify(blocks), encoding: 'utf8', maxBuffer: 4 * 1024 * 1024,
});
if (result.stdout) process.stdout.write(result.stdout);
if (result.stderr) process.stderr.write(result.stderr);
assert.equal(result.error, undefined, result.error?.message);
assert.equal(result.status, 0, 'Rotation manuscript examples or independent oracles failed');
