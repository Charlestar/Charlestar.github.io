// Toy bookkeeping plus stubbed wrapper control flow, never real Torch/CUDA execution.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';

const post = readFileSync('_posts/2026-10-21-pytorch-cuda-memory-allocator-snapshot.md', 'utf8');
function block(marker) {
  const pattern = new RegExp(`<!-- example: ${marker} -->\\s*\x60\x60\x60python\\r?\\n([\\s\\S]*?)\\r?\\n\x60\x60\x60`, 'g');
  const matches = [...post.matchAll(pattern)];
  assert.equal(matches.length, 1, `expected one ${marker} block`);
  return matches[0][1];
}
const cpu = block('allocator-snapshot-ledger');
const gpu = block('allocator-snapshot-capture');
const tests = String.raw`
import ast
import copy
import itertools
import json
import contextlib
import io
import sys
from types import SimpleNamespace

def rejects(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("expected ValueError")

assert ledger == {'reserved': 24*MIB, 'allocated': 6*MIB, 'pending': 2*MIB,
                  'inactive': 16*MIB, 'requested_live': 11*MIB//2,
                  'whole_inactive': 8*MIB, 'largest_inactive': 8*MIB}
assert ledger['reserved'] - ledger['allocated'] == 18*MIB
assert ledger['reserved'] == ledger['allocated'] + ledger['pending'] + ledger['inactive']
assert ledger['allocated'] + ledger['pending'] == 8*MIB

# Independent byte ownership oracle, avoiding arithmetic summation by state.
for states in itertools.product(sorted(STATES), repeat=4):
    blocks = [(i+1, state, i if state != 'inactive' else 0)
              for i, state in enumerate(states)]
    result = summarize_segments([{'address': 16, 'total_size': 10, 'blocks': blocks}])
    owners, requested_addresses, offset = {}, set(), 16
    for size, state, requested in blocks:
        for address in range(offset, offset+size):
            owners[address] = state
        if state == 'active_allocated':
            requested_addresses.update(range(offset, offset+requested))
        offset += size
    assert result['allocated'] == len([a for a, state in owners.items() if state == 'active_allocated'])
    assert result['pending'] == len([a for a, state in owners.items() if state == 'active_awaiting_free'])
    assert result['inactive'] == len([a for a, state in owners.items() if state == 'inactive'])
    assert result['requested_live'] == len(requested_addresses)
    assert result['whole_inactive'] == (10 if set(owners.values()) == {'inactive'} else 0)
    assert result['reserved'] == len(owners)
    assert result['reserved'] - result['allocated'] == result['pending'] + result['inactive']
    assert result['requested_live'] <= result['allocated']

# Translation/scaling leave the classification unchanged; this is a byte ledger,
# not a simulator whose result depends on an arbitrary numeric base address.
scaled = copy.deepcopy(example)
for segment in scaled:
    segment['address'] = segment['address'] * 3 + 7
    segment['total_size'] *= 3
    segment['blocks'] = [(size * 3, state, requested * 3)
                         for size, state, requested in segment['blocks']]
assert summarize_segments(scaled) == {key: value * 3 for key, value in ledger.items()}

# Whole-segment release is a static classification, not an empty_cache simulation.
no_whole = [
    {'address': 0, 'total_size': 16, 'blocks': [(6,'active_allocated',6), (4,'inactive',0),
                                              (2,'active_allocated',2), (4,'inactive',0)]},
    {'address': 32, 'total_size': 8, 'blocks': [(4,'active_allocated',4), (4,'inactive',0)]},
]
fragmented = summarize_segments(no_whole)
assert fragmented['inactive'] == 12 and fragmented['largest_inactive'] == 4
assert fragmented['whole_inactive'] == 0
assert fragmented['largest_inactive'] < 6 < fragmented['inactive']

# A recorded event completion can move pending into inactive; allocated stays fixed.
completed = copy.deepcopy(example)
completed[0]['blocks'][2] = (2*MIB, 'inactive', 0)
after = summarize_segments(completed)
assert after['pending'] == 0 and after['inactive'] == 18*MIB
assert after['allocated'] == ledger['allocated']
assert after['whole_inactive'] == 8*MIB  # live 6 MiB still pins the first fixed segment

# Address order does not affect the result; adjacency is legal but overlap is not.
assert summarize_segments(list(reversed(example))) == ledger
adjacent = [{'address': 0, 'total_size': 2, 'blocks': [(2,'inactive',0)]},
            {'address': 2, 'total_size': 1, 'blocks': [(1,'active_allocated',1)]}]
assert summarize_segments(adjacent)['reserved'] == 3
assert all(value == 0 for value in summarize_segments([]).values())
for address in [0, 1]:
    overlapping = copy.deepcopy(adjacent)
    overlapping[1]['address'] = address
    rejects(lambda: summarize_segments(overlapping))

for bad in [-1, 0.5, True, None, '1']:
    rejects(lambda bad=bad: summarize_segments([{'address': bad, 'total_size': 1, 'blocks': [(1,'inactive',0)]}]))
for bad in [-1, 0, 0.5, True, None, '1']:
    rejects(lambda bad=bad: summarize_segments([{'address': 0, 'total_size': bad, 'blocks': [(1,'inactive',0)]}]))
    rejects(lambda bad=bad: summarize_segments([{'address': 0, 'total_size': 1, 'blocks': [(bad,'inactive',0)]}]))
for bad in [-1, 2, 0.5, True, None, '1']:
    rejects(lambda bad=bad: summarize_segments([{'address': 0, 'total_size': 1, 'blocks': [(1,'active_allocated',bad)]}]))
for blocks in [[], [(1,'unknown',0)], [(1,'inactive',1)], [(2,'inactive',0)]]:
    rejects(lambda blocks=blocks: summarize_segments([{'address': 0, 'total_size': 1, 'blocks': blocks}]))

# Syntax only for the GPU wrapper. Confirm its record/dump API names without import.
tree = ast.parse(json.loads(GPU_SOURCE))
functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
assert 'capture_window' in functions
assert any(isinstance(node, ast.ExceptHandler) for node in ast.walk(tree))

# Independent audit: execute only the extracted function with explicit stand-ins.
# No Torch import, CUDA allocation, filesystem write, or real snapshot is involved.
class FakeOOM(RuntimeError):
    pass

class FakePath:
    def __init__(self, value):
        self.value = str(value)
        self.parent = self
    def mkdir(self, **kwargs):
        pass
    def __str__(self):
        return self.value

class FakeMemory:
    def __init__(self, fail_dump=False, fail_stop=False):
        self.calls = []
        self.fail_dump = fail_dump
        self.fail_stop = fail_stop
    def get_allocator_backend(self):
        return 'stub-native'
    def _record_memory_history(self, **kwargs):
        self.calls.append(('record', kwargs))
        if kwargs['enabled'] is None and self.fail_stop:
            raise RuntimeError('stub stop failure')
    def _dump_snapshot(self, name):
        self.calls.append(('dump', name))
        if self.fail_dump:
            raise RuntimeError('stub dump failure')

def stubbed_capture(memory):
    fake = SimpleNamespace(__version__='stub', version=SimpleNamespace(cuda='stub'),
                           cuda=SimpleNamespace(is_available=lambda: True,
                                                memory=memory, OutOfMemoryError=FakeOOM))
    namespace = {'torch': fake, 'sys': sys, 'Path': FakePath}
    module = ast.Module(body=[functions['capture_window']], type_ignores=[])
    exec(compile(module, '<article wrapper, stub APIs>', 'exec'), namespace)
    return namespace['capture_window']

token = object()
memory = FakeMemory()
with contextlib.redirect_stdout(io.StringIO()):
    assert stubbed_capture(memory)(lambda: token, 'case') is token
assert memory.calls[0][1]['max_entries'] == 100000
assert memory.calls[1] == ('dump', 'case-ok.pickle')
assert memory.calls[-1] == ('record', {'enabled': None})

original = FakeOOM('original oom')
def oom_workload():
    raise original
for fail_dump, fail_stop in itertools.product((False, True), repeat=2):
    memory = FakeMemory(fail_dump, fail_stop)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        try:
            stubbed_capture(memory)(oom_workload, 'case')
        except FakeOOM as caught:
            assert caught is original
        else:
            raise AssertionError('original OOM must propagate')
    assert ('dump', 'case-oom.pickle') in memory.calls
    assert memory.calls[-1] == ('record', {'enabled': None})
# Non-OOM workload errors also keep their identity, with or without stop failure.
other_error = ValueError('workload error')
def bad_workload():
    raise other_error
for fail_stop in (False, True):
    memory = FakeMemory(fail_stop=fail_stop)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        try:
            stubbed_capture(memory)(bad_workload, 'case')
        except ValueError as caught:
            assert caught is other_error
        else:
            raise AssertionError('workload error must propagate')
    assert not any(kind == 'dump' for kind, _ in memory.calls)
# A stop failure after an otherwise successful capture must remain visible.
with contextlib.redirect_stdout(io.StringIO()):
    try:
        stubbed_capture(FakeMemory(fail_stop=True))(lambda: token, 'case')
    except RuntimeError as caught:
        assert str(caught) == 'stub stop failure'
    else:
        raise AssertionError('stop failure must not become silent success')
print('Allocator snapshot examples: CPU ledger/ownership/boundaries and stubbed capture exception control flow passed; no Torch/CUDA run.')
`;
const python = process.env.BLOG_PYTHON || 'python';
const result = spawnSync(python, ['-B', '-c', cpu + '\nGPU_SOURCE = ' + JSON.stringify(JSON.stringify(gpu)) + '\n' + tests], { encoding: 'utf8' });
if (result.error) throw result.error;
process.stdout.write(result.stdout || '');
process.stderr.write(result.stderr || '');
assert.equal(result.status, 0);
