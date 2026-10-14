import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';

const article = readFileSync(new URL('../../_posts/2026-10-15-quantization-acceptance-engineering.md', import.meta.url), 'utf8');
function extract(marker, language) {
  const parts = article.split(marker);
  assert.equal(parts.length, 2, `Missing or duplicate acceptance marker: ${marker}`);
  const block = parts[1].match(new RegExp(`^\\s*\x60\x60\x60${language}\\r?\\n([\\s\\S]*?)\x60\x60\x60`));
  assert.ok(block, `Missing actual ${language} block`);
  return block[1];
}
const blocks = {
  tensor: extract('<!-- example: quantization-acceptance-tensor-and-clamp -->', 'python'),
  logits: extract('<!-- example: quantization-acceptance-logits-mask-and-ppl -->', 'python'),
  fixture: extract('<!-- fixture: quantization-acceptance-v1-evidence -->', 'json'),
};
assert.equal([...article.matchAll(/^```python\s*$/gm)].length, 2);
assert.equal([...article.matchAll(/^```json\s*$/gm)].length, 1);
assert.match(article, /^date: 2026-10-15$/m);
assert.match(article, /^author: iStar$/m);
assert.match(article, /^series_order: 120$/m);
assert.match(article, /回滚/);

const program = String.raw`
import ast
import contextlib
import copy
from decimal import Decimal as D, localcontext
from fractions import Fraction as F
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import tempfile

blocks=json.load(sys.stdin)
root=Path.cwd()
tool=root/'scripts/quantization-acceptance.py'
spec=importlib.util.spec_from_file_location('quantization_acceptance_audit',tool)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
passed=0
cli_calls=0
fixture=json.loads(blocks['fixture'])

def check(name,action):
    global passed
    action();passed+=1;print('PASS',name,flush=True)

def close(actual,expected,relative=2e-12,absolute=1e-14):
    assert math.isfinite(actual) and math.isclose(actual,float(expected),rel_tol=relative,abs_tol=absolute),(actual,expected)

def strict_json(text):
    def bad(token): raise AssertionError(('nonstandard JSON',token))
    def pairs(items):
        result={}
        for key,value in items:
            assert key not in result,('duplicate output key',key)
            result[key]=value
        return result
    result=json.loads(text,parse_constant=bad,object_pairs_hook=pairs)
    def walk(value):
        if type(value) is float: assert math.isfinite(value)
        elif type(value) is list:
            for item in value: walk(item)
        elif type(value) is dict:
            for item in value.values(): walk(item)
    walk(result)
    return result

def invoke(args):
    global cli_calls
    cli_calls+=1
    return subprocess.run([sys.executable,'-B',str(tool),*args],capture_output=True,text=True,encoding='utf-8',timeout=15)

cache=(root/'.jekyll-cache').resolve()
assert cache.is_relative_to(root.resolve()),'temporary test directory must stay in workspace'
cache.mkdir(exist_ok=True)
temporary=tempfile.TemporaryDirectory(prefix='quant-acceptance-independent-',dir=cache)
folder=Path(temporary.name)
assert folder.resolve().parent==cache
sequence=0
def raw_cli(raw,code=0,error=None):
    global sequence
    sequence+=1
    path=folder/('样本 evidence '+str(sequence)+'.json')
    path.write_bytes(raw)
    result=invoke(['--input',str(path)])
    assert result.returncode==code,(code,result.returncode,result.stdout,result.stderr)
    if code==0:
        assert result.stderr=='' and result.stdout.endswith('\n')
        report=strict_json(result.stdout)
        assert report['input_sha256']==hashlib.sha256(raw).hexdigest()
        assert report['status']=='incomplete' and report['cpu_checks_status']=='completed'
        assert 'accepted' not in report
    else:
        assert result.stdout==''
        report=strict_json(result.stderr)
        assert report['status']=='failed' and report['code']==error,report
        assert report['stage'] and report.get('field'),report
    return report

def doc_cli(doc,code=0,error=None):
    return raw_cli(json.dumps(doc,ensure_ascii=False,allow_nan=False).encode('utf-8'),code,error)

def tensor_doc(ref,cand): return {'schema_version':1,'tensor':{'reference':ref,'candidate':cand}}
def logits_doc(ref,cand,targets=None,mask=None):
    return {'schema_version':1,'logits':{'reference':ref,'candidate':cand,
        'targets':targets if targets is not None else [0]*len(ref),
        'mask':mask if mask is not None else [True]*len(ref)}}

def uint4_doc(codes,packed=None,scale=.5,zero=1):
    if packed is None:
        packed=[sum(codes[i+j]*16**j for j in range(min(2,len(codes)-i))) for i in range(0,len(codes),2)]
    return {'schema_version':1,'uint4':{'codes':codes,'packed_bytes':packed,'logical_length':len(codes),
        'scale':scale,'zero_point':zero,'rounding':'ties_to_even','layout':'low_first'}}

def actual_manuscript_and_cli():
    for name in ['tensor','logits']:
        ast.parse(blocks[name]);capture=io.StringIO()
        with contextlib.redirect_stdout(capture): exec(compile(blocks[name],'article:'+name,'exec'),{})
        assert capture.getvalue().strip()
    report=raw_cli(blocks['fixture'].encode('utf-8'))
    assert report['results']['tensor']['mse']==2.5
    assert report['results']['tensor']['relative_squared_l2']==1.
    close(report['results']['tensor']['cosine'],1.)
    assert report['results']['logits']['scored_rows']==1
    assert report['results']['logits']['candidate']['perplexity']==2.
    assert report['results']['uint4']['dequantized']==[-.5,0.,.5]
    result=invoke(['--self-test'])
    assert result.returncode==0 and result.stderr==''
    report=strict_json(result.stdout)
    assert report['status']=='incomplete' and report['input_sha256'] is None
    tests=report['self_test']
    assert tests['status']=='passed' and tests['checks']==len(tests['names'])==60
    for args in [[],['--self-tes'],['--in','x'],['--input'],['--self-test','--input','x'],['--self-test','--typo'],['evidence.json']]:
        result=invoke(args)
        assert result.returncode==2 and result.stdout=='' and 'usage:' in result.stderr.lower(),(args,result)
    result=invoke(['--input',str(folder/'does-not-exist.json')])
    assert result.returncode==2 and result.stdout=='' and strict_json(result.stderr)['stage']=='read'
check('two actual Python blocks, actual JSON fixture, 60-check CLI and strict required/unambiguous CLI modes',actual_manuscript_and_cli)

def strict_schema_and_json():
    malformed=[None,[],{}, {'schema_version':1}, {'schema_version':1,'external_evidence':{'kernel_pass':True}}]
    for version in [True,False,1.,'1',2,None]:
        doc=copy.deepcopy(fixture);doc['schema_version']=version;malformed.append(doc)
    for key in ['typo','accepted','gpu_kernel_verified']:
        doc=copy.deepcopy(fixture);doc[key]=True;malformed.append(doc)
    for group in ['tensor','logits','uint4']:
        for key in fixture[group]:
            doc=copy.deepcopy(fixture);del doc[group][key];malformed.append(doc)
        doc=copy.deepcopy(fixture);doc[group]['typo']=1;malformed.append(doc)
        doc=copy.deepcopy(fixture);doc[group]=None;malformed.append(doc)
    for group,key,values in [
        ('tensor','reference',[[],[True],['1'],[None],[[1]]]),
        ('tensor','candidate',[[1.]]),
        ('logits','reference',[[[0]],[[0,0],[0]],[]]),
        ('logits','candidate',[[[0,0]],[[0,0],[0,0,0]]]),
        ('logits','targets',[[True,0],[0,False],[0,2],[0,-1],[0,0.],[0],None]),
        ('logits','mask',[[1,0],[True],[False,False],['true',False]]),
        ('uint4','codes',[[False,1,2],[-1,1,2],[16,1,2],[0.,1,2],[]]),
        ('uint4','packed_bytes',[[True,2],[-1,2],[256,2],[16.,2],[16],[]]),
        ('uint4','logical_length',[True,0,-1,3.,4]),
        ('uint4','zero_point',[True,-1,16,1.,'1']),
        ('uint4','scale',[True,0,-1,'1',None]),
        ('uint4','rounding',['nearest','TIES_TO_EVEN',None]),
        ('uint4','layout',['high_first',None]),
    ]:
        for value in values:
            doc=copy.deepcopy(fixture);doc[group][key]=value;malformed.append(doc)
    for doc in malformed: doc_cli(doc,2,'input_error')
    raw_bad=[b'',b'{',b'{"schema_version":1,"schema_version":1}',
        b'{"schema_version":1,"tensor":{"reference":[1],"reference":[1],"candidate":[1]}}',
        b'{"schema_version":1,"tensor":{"reference":[1],"candidate":[1]},"external_evidence":{"x":1,"x":2}}',
        b'{"x":NaN}',b'{"x":Infinity}',b'{"x":-Infinity}',
        b'{"x":1e400}',b'{"x":-1e400}',b'{"x":1e-400}',b'{"x":-1e-400}',
        b'{"external_evidence":{"nested":[1e400]}}',b'{"external_evidence":{"nested":[1e-400]}}',
        b'\xff',json.dumps(fixture).encode('utf-16')]
    for raw in raw_bad: raw_cli(raw,2,'input_error')
    # Real zero mantissas may have nonzero exponent digits. They must NOT
    # be confused with nonzero literals rounded to zero.
    for literal in [b'0.0',b'-0.0',b'0e400',b'-0.000e-400',b'0.00e+9999']:
        raw=b'{"schema_version":1,"tensor":{"reference":['+literal+b'],"candidate":[0]},"external_evidence":{"zero":'+literal+b'}}'
        assert raw_cli(raw)['results']['tensor']['mse']==0.
    doc=tensor_doc([10**400],[0]);doc_cli(doc,2,'input_error')
    for value in [2**53+1,-(2**53+1)]: doc_cli(tensor_doc([value],[2**53]),2,'input_error')
    assert doc_cli(tensor_doc([2**53+2],[2**53+2]))['results']['tensor']['mse']==0.
    # A huge exact integer in unused JSON metadata is still valid JSON;
    # do not invent a float-conversion requirement for that inert field.
    doc=tensor_doc([1],[1]);doc['external_evidence']={'huge_integer':10**400,'text':'1e400'}
    assert doc_cli(doc)['external_evidence']['metadata']==doc['external_evidence']
check('strict schema/type/bool/shape/full-mask validation, duplicate keys, UTF8 and float-literal overflow versus genuine zero',strict_schema_and_json)

def raw_identity_and_untrusted_evidence():
    doc=tensor_doc([1],[1]);doc['external_evidence']={'kernel_pass':True,'accepted':True,'GPU':'claimed',
        'verified_scope':['all GPUs'],'nested':{'model_task_quality':True},'note':'中文'}
    variants=[json.dumps(doc,ensure_ascii=False).encode('utf-8'),json.dumps(doc,indent=2,ensure_ascii=False).encode('utf-8')+b'\r\n',
              json.dumps(doc,sort_keys=True,ensure_ascii=False).encode('utf-8')+b'\n']
    hashes=[]
    for raw in variants:
        report=raw_cli(raw);hashes.append(report['input_sha256'])
        assert report['external_evidence']=={'status':'reported_unverified','metadata':doc['external_evidence']}
        assert {'source_authenticity','cross_group_identity','real_gpu_kernel','teacher_forcing_semantics','model_task_quality','service_performance'}<=set(report['unverified'])
        assert report['results']['logits']['status']==report['results']['uint4']['status']=='skipped'
        assert report['verified_scope']==['tensor: supplied CPU numeric data']
    assert len(set(hashes))==3
    # Observe main's real read boundary, rather than inferring a single read
    # from two hashes of separately dumped JSON. A second read would fail.
    raw=variants[0];reads=[];old=Path.read_bytes
    def read_once(path):
        reads.append(str(path));assert len(reads)==1,'input read more than once';return raw
    Path.read_bytes=read_once
    stdout,stderr=io.StringIO(),io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(stderr):
            code=module.main(['--input',str(folder/'one-read.json')])
    finally: Path.read_bytes=old
    assert code==0 and len(reads)==1 and stderr.getvalue()==''
    assert strict_json(stdout.getvalue())['input_sha256']==hashlib.sha256(raw).hexdigest()
check('actual raw UTF8/LF/CRLF/field-order hashes, instrumented one-read main and inert external claims without scope promotion',raw_identity_and_untrusted_evidence)

def integer_rounding_and_packing():
    def nearest_integer(value):
        low=value.numerator//value.denominator
        return min([low,low+1],key=lambda c:(abs(value-c),c%2))
    values=[F(i,8) for i in range(-160,161)]
    for scale,zero in [(F(1),0),(F(1,2),1),(F(2),7),(F(1,4),15)]:
        codes,stats=module.affine_quantize(list(map(float,values)),float(scale),zero,0,15)
        raw=[nearest_integer(v/scale)+zero for v in values]
        expected=[min(15,max(0,c)) for c in raw]
        assert codes==expected
        assert stats=={'code_clamp_fraction':sum(c<0 or c>15 for c in raw)/len(raw),
            'endpoint_fraction':sum(c in [0,15] for c in expected)/len(expected),
            'prequant_range_exceed_fraction':sum(v<scale*(0-zero) or v>scale*(15-zero) for v in values)/len(values)}
    assert nearest_integer(F(1,2))+1==1 and nearest_integer(F(3,2))==2
    known=[0x10,0x32,0x54,0x76,0x98,0xba,0xdc,0xfe]
    assert module.pack_uint4(list(range(16)))==known
    assert doc_cli(uint4_doc(list(range(16)),known))['results']['uint4']['byte_count']==8
    packed=list(range(256));codes=[c for b in packed for c in [b%16,b//16]]
    result=doc_cli(uint4_doc(codes,packed,.25,3))['results']['uint4']
    assert result['dequantized']==[float(F(c-3,4)) for c in codes]
    assert result['endpoint_fraction']==.125 and result['byte_count']==256
    for length in [1,3,5,15,17,31]:
        codes=[(i*7)%16 for i in range(length)]
        result=doc_cli(uint4_doc(codes))['results']['uint4']
        assert result['byte_count']==(length+1)//2 and len(result['dequantized'])==length
    for packed in [[1,2],[16,18]]: doc_cli(uint4_doc([0,1,2],packed),1,'parity_error')
    for scale,zero in [(1.,1),(.5,2)]:
        result=doc_cli(uint4_doc([0,1,2],[16,2],scale,zero))['results']['uint4']
        assert result['dequantized']!=[-.5,0.,.5] and result['status']=='parity_pass'
    consistent=uint4_doc([1,1,2],[17,2]);report=doc_cli(consistent)
    assert report['results']['uint4']['status']=='parity_pass' and report['status']=='incomplete'
check('Fraction RNE/zero/clamp, independent golden and all 256 byte values, odd padding, actual parity faults and legal metadata changes',integer_rounding_and_packing)

def exact_tensor_oracles_and_numeric_failures():
    rng=random.Random(614)
    cases=[([1.,2.],[2.,4.]),([0.,0.],[0.,0.]),([0.,0.],[1.,2.]),([1.,2.],[0.,0.])]
    cases.extend(([rng.randint(-16,16)/4 for _ in range(n)],[rng.randint(-16,16)/4 for _ in range(n)]) for n in range(1,9))
    with localcontext() as context:
        context.prec=80
        for ref,cand in cases:
            actual=doc_cli(tensor_doc(ref,cand))['results']['tensor']
            r,c=list(map(F.from_float,ref)),list(map(F.from_float,cand))
            sse=sum((a-b)**2 for a,b in zip(r,c));r2=sum(x*x for x in r);c2=sum(x*x for x in c)
            close(actual['mse'],sse/len(r));close(actual['max_absolute_error'],max(abs(a-b) for a,b in zip(r,c)))
            if r2: close(actual['relative_squared_l2'],sse/r2)
            else: assert actual['relative_squared_l2'] is None and actual['relative_squared_l2_reason']
            if r2 and c2:
                dot=sum(a*b for a,b in zip(r,c))
                def decimal(value): return D(value.numerator)/D(value.denominator)
                close(actual['cosine'],decimal(dot)/(decimal(r2)*decimal(c2)).sqrt())
            else: assert actual['cosine'] is None and actual['cosine_reason']
    tiny=2.**-537
    cases=[([1e308],[-1e308]),([1e200],[0.]),([1e154,1e154],[0.,0.]),([1e-300],[1e-300]),
        ([1.,1e-200],[1.,0.]),([0.,0.],[tiny,0.]),([2.,0.],[2.,tiny]),
        ([1.,1.,tiny],[1.,-1.,tiny]),([1.,1.,tiny],[1.,-1.,-tiny])]
    for ref,cand in cases: doc_cli(tensor_doc(ref,cand),1,'numeric_error')
    huge=uint4_doc([15],[15],1e308,0);doc_cli(huge,1,'numeric_error')
check('independent Fraction tensor metrics/Decimal cosine, genuine zero semantics and signed finite/intermediate underflow or overflow failures',exact_tensor_oracles_and_numeric_failures)

def decimal_logits_oracles():
    def oracle(ref,cand,targets,mask):
        with localcontext() as context:
            context.prec=90
            def distribution(row):
                values=[D.from_float(float(x)) for x in row]
                peak=max(values);weights=[(x-peak).exp() for x in values]
                denominator=sum(weights);logs=[x-peak-denominator.ln() for x in values]
                return [w/denominator for w in weights],logs
            kl,loss_r,loss_q,top=[],[],[],[]
            for r,q,target,keep in zip(ref,cand,targets,mask):
                if not keep: continue
                p,lp=distribution(r);_,lq=distribution(q)
                kl.append(sum(v*(a-b) for v,a,b in zip(p,lp,lq)))
                loss_r.append(-lp[target]);loss_q.append(-lq[target])
                top.append(max(range(len(r)),key=r.__getitem__)==max(range(len(q)),key=q.__getitem__))
            return {'kl':sum(kl)/len(kl),'r_sum':sum(loss_r),'q_sum':sum(loss_q),
                'r_ppl':(sum(loss_r)/len(loss_r)).exp(),'q_ppl':(sum(loss_q)/len(loss_q)).exp(),
                'top':sum(top)/len(top),'count':len(top)}
    cases=[([[math.log(.75),math.log(.25)]],[[math.log(.5),math.log(.5)]],[0],[True]),
        ([[math.log(.5),math.log(.5)]],[[math.log(.75),math.log(.25)]],[0],[True]),
        ([[0.,0.,-1000.]],[[0.,-1000.,0.]],[0],[True]),
        ([[0.,0.]],[[0.,0.]],[1],[True]),
        ([[0.,0.],[0.,0.]],[[0.,0.],[0.,2.]],[0,0],[True,False]),
        ([[2.,1.,0.],[-1.,0.,1.],[3.,3.,1.]],[[2.2,.5,0.],[1.,0.,-1.],[3.,3.,1.]],[2,0,1],[True,False,True])]
    for ref,cand,targets,mask in cases:
        expected=oracle(ref,cand,targets,mask)
        actual=doc_cli(logits_doc(ref,cand,targets,mask))['results']['logits']
        close(actual['kl_reference_to_candidate_nats'],expected['kl'])
        assert actual['scored_rows']==actual['reference']['scored_targets']==actual['candidate']['scored_targets']==expected['count']
        for key,side in [('reference','r'),('candidate','q')]:
            close(actual[key]['nll_sum'],expected[side+'_sum']);close(actual[key]['perplexity'],expected[side+'_ppl'])
        assert actual['top1_agreement']==expected['top']
    # Check the corrected near-equal case against original-logit Decimal
    # probabilities, not the new nonnegative summation's own kernels.
    r=[2.452614615935712,-.10910710576120053,3.6311162686894303,2.9003006069414257]
    q=[2.4526146219971254,-.10910710290931795,3.631116278287903,2.9003006017458595]
    expected=oracle([r],[q],[0],[True])['kl']
    actual=doc_cli(logits_doc([r],[q]))['results']['logits']['kl_reference_to_candidate_nats']
    assert actual>0;close(actual,expected,relative=2e-6,absolute=0.)
    # Both small-d signs and both sides of the series/expm1 boundary.
    for delta in [-.002004,-.001996,-1e-5,1e-5,.001996,.002004]:
        r,q=[[0.,0.]],[[0.,delta]]
        actual=doc_cli(logits_doc(r,q))['results']['logits']['kl_reference_to_candidate_nats']
        close(actual,oracle(r,q,[0],[True])['kl'],relative=1e-9,absolute=0.)
    # log-domain weighting recovers exp(-1000)*1e308, although exp(-1000)
    # alone underflows in binary64. This is still not arbitrary precision.
    tiny=doc_cli(logits_doc([[0.,-1000.]],[[0.,-1e308]]))['results']['logits']['kl_reference_to_candidate_nats']
    with localcontext() as context:
        context.prec=90
        close(tiny,D(-1000).exp()*D.from_float(1e308),relative=1e-12,absolute=0.)
    # Weight one ln2 target and three ln4 targets through real aligned rows.
    rows=[[0.,0.,-1000.,-1000.]]+[[0.,0.,0.,0.]]*3
    result=doc_cli(logits_doc(rows,rows))['results']['logits']
    with localcontext() as context:
        context.prec=60;close(result['reference']['perplexity'],D(2)**(D(7)/4))
    assert result['scored_rows']==4
    # Masked finite huge spans are validated but never evaluated.
    doc=logits_doc([[0.,0.],[1e308,-1e308]],[[0.,0.],[-1e308,1e308]],[0,0],[True,False])
    assert doc_cli(doc)['results']['logits']['candidate']['perplexity']==2.
    doc['logits']['targets'][1]=2;doc_cli(doc,2,'input_error')
    doc=logits_doc([[1e308,-1e308]],[[0.,0.]],[0]);doc_cli(doc,1,'numeric_error')
    doc=logits_doc([[0.,0.]],[[0.,-1000.]],[1]);doc_cli(doc,1,'numeric_error')
check('90-digit original-logit KL/NLL/PPL oracles, both KL directions, near-equal/series boundaries, weighted tiny terms and shared one-row masks',decimal_logits_oracles)

def selftest_negative_paths_and_schema_before_compute():
    # Mutate only this in-memory test module, never production/user files.
    # The same expect-helper must reject a missing or wrong-code exception.
    original=module.parse_document
    def mutated(no_failure):
        def parse(raw):
            if raw==b'{"x":{"a":1,"a":2}}':
                if no_failure: return {}
                raise module.EvidenceError('numeric_error','injected','test','wrong code')
            return original(raw)
        return parse
    try:
        for no_failure in [True,False]:
            module.parse_document=mutated(no_failure)
            try: module.self_test()
            except module.EvidenceError as error:
                assert error.code=='selftest_error' and error.field=='duplicate nested key'
                assert ('expected failure did not occur' if no_failure else 'wrong failure code') in str(error)
            else: raise AssertionError('injected faulty self-test passed')
    finally: module.parse_document=original
    # Entire document is schema-validated before ANY metric computation:
    # a numeric failure in an earlier group must not hide later bad schema.
    doc=copy.deepcopy(fixture);doc['tensor']={'reference':[1e308],'candidate':[-1e308]}
    doc['logits']['targets'][1]=True
    result=doc_cli(doc,2,'input_error');assert result['field']=='logits.targets[1]'
check('actual missing/wrong-failure-code fault injection and full-document schema validation before numeric evaluation',selftest_negative_paths_and_schema_before_compute)

assert passed==7
temporary.cleanup()
print(json.dumps({'groups':passed,'actual_cli_invocations':cli_calls,'scope':'two real post examples and JSON fixture; actual strict CLI with independent Fraction/Decimal oracles and fault injection; no model, Torch, GPU or service execution'}))
`;

const result = spawnSync(process.env.BLOG_PYTHON || 'python', ['-B', '-c', program], {
  input: JSON.stringify(blocks), encoding: 'utf8', maxBuffer: 4 * 1024 * 1024,
  cwd: new URL('../../', import.meta.url), timeout: 120000,
});
if (result.stdout) process.stdout.write(result.stdout);
if (result.stderr) process.stderr.write(result.stderr);
assert.equal(result.error, undefined, result.error?.message);
assert.equal(result.status, 0, 'Acceptance manuscript/CLI/oracle checks failed');
