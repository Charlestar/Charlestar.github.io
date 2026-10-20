// Standard-library CPU models extracted from the actual article, not PyTorch execution.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';

const post = readFileSync('_posts/2026-10-21-training-data-pipeline-dataloader.md', 'utf8');
function block(marker) {
  const pattern = new RegExp(`<!-- example: ${marker} -->\\s*\x60\x60\x60python\\r?\\n([\\s\\S]*?)\\r?\\n\x60\x60\x60`, 'g');
  const matches = [...post.matchAll(pattern)];
  assert.equal(matches.length, 1, `expected one ${marker} block`);
  return matches[0][1];
}
const code = ['dataloader-two-stage', 'dataloader-two-drop-last', 'dataloader-iterable-shards'].map(block).join('\n');
const tests = String.raw`
from collections import Counter
from itertools import permutations

def rejects(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("expected ValueError")

# Independently advance integer ticks and move tokens between two servers.
def tick_pipeline(count, prepare, compute):
    if not count:
        return []
    producer_left, consumer_left = prepare, None
    issued, finished, ready, tick = 0, 0, [], 0
    starts, ends, ready_at = {}, {}, {}
    consuming = None
    while finished < count:
        tick += 1
        if producer_left is not None:
            producer_left -= 1
            if producer_left == 0:
                ready.append(issued)
                ready_at[issued] = tick
                issued += 1
                producer_left = prepare if issued < count else None
        if consumer_left is not None:
            consumer_left -= 1
            if consumer_left == 0:
                ends[consuming] = tick
                consumer_left = None
                finished += 1
        if consumer_left is None and ready:
            consuming = ready.pop(0)
            starts[consuming] = tick
            consumer_left = compute
    return [(i, ready_at[i], starts[i], ends[i]) for i in range(count)]

for count in range(9):
    for prepare in range(1, 8):
        for compute in range(1, 8):
            result = two_stage(count, prepare, compute)
            assert result == tick_pipeline(count, prepare, compute)
            if count:
                assert result[-1][3] == prepare + compute + (count - 1) * max(prepare, compute)
assert two_stage(0, 1, 2) == []
assert two_stage(6, 6, 12)[-1][3] == 78
assert two_stage(2, 3, 3)[-1][3] == 9

# Oracle builds an equal-length matrix by cycling original positions, then transposes.
# It does not call the implementation's padding or step-slicing logic.
for size in range(33):
    for replicas in range(1, 9):
        for drop in [False, True]:
            rows = [rank_indices(size, replicas, rank, drop) for rank in range(replicas)]
            count = size // replicas + int(not drop and size % replicas != 0)
            matrix = [[(row * replicas + col) % size for col in range(replicas)]
                      for row in range(count)]
            expected = [[matrix[row][col] for row in range(count)] for col in range(replicas)]
            assert rows == expected
            flattened = [index for row in rows for index in row]
            if drop:
                assert sorted(flattened) == list(range(count * replicas))
            elif size:
                assert set(flattened) == set(range(size))
                assert max(Counter(flattened).values()) - min(Counter(flattened).values()) <= 1
            for batch_size in range(1, 7):
                for loader_drop in [False, True]:
                    batches = [local_batches(row, batch_size, loader_drop) for row in rows]
                    expected_count = count // batch_size + int(not loader_drop and count % batch_size != 0)
                    assert all(len(groups) == expected_count for groups in batches)
                    assert all(len(group) == batch_size for groups in batches for group in groups[:-1])
                    for original, groups in zip(rows, batches):
                        retained = [item for group in groups for item in group]
                        limit = count - count % batch_size if loader_drop else count
                        assert retained == original[:limit]
assert rank_indices(1, 4, 3) == [0]
assert rank_indices(1, 4, 3, True) == []

# Independent audit: arbitrary shared permutations preserve ownership before the
# sampler's documented pad/drop policy. No Python permutation mimics torch.randperm.
for size in range(1, 7):
    for order in permutations(range(size)):
        for replicas in range(1, 5):
            for sampler_drop in (False, True):
                rows = [[order[index] for index in rank_indices(size, replicas, rank, sampler_drop)]
                        for rank in range(replicas)]
                counts = Counter(item for row in rows for item in row)
                expected_total = ((size // replicas) if sampler_drop
                                  else -(-size // replicas)) * replicas
                assert sum(counts.values()) == expected_total
                if sampler_drop:
                    assert counts == Counter(order[:expected_total])
                else:
                    cycles, extra = divmod(expected_total, size)
                    assert counts == Counter({item: cycles + int(position < extra)
                                              for position, item in enumerate(order)})
# A separate rank-local shuffle is not a partition of one shared permutation.
rank0_order = list(range(6))
rank1_order = rank0_order[1:] + rank0_order[:1]
bad_rank_rows = [[order[index] for index in rank_indices(6, 2, rank)]
                 for rank, order in enumerate((rank0_order, rank1_order))]
assert bad_rank_rows == [[0, 2, 4], [2, 4, 0]]
assert set(item for row in bad_rank_rows for item in row) != set(range(6))

# Independent ownership oracle assigns each item to a consumer bucket.
for size in range(32):
    for replicas in range(1, 5):
        for workers in range(1, 5):
            shards = iterable_shards(size, replicas, workers)
            buckets = [[] for _ in range(replicas * workers)]
            consumer = 0
            for item in range(size):
                buckets[consumer].append(item)
                consumer += 1
                if consumer == len(buckets):
                    consumer = 0
            assert shards == buckets
            assert sorted(item for shard in shards for item in shard) == list(range(size))
            assert max(map(len, shards)) - min(map(len, shards)) <= 1

# Independent audit: omitting rank identity duplicates the entire source per rank.
local_only = iterable_shards(23, 1, 2)
bad_consumption = [item for _rank in range(2) for shard in local_only for item in shard]
assert Counter(bad_consumption) == Counter({item: 2 for item in range(23)})
# Exact coverage can still have mismatched step counts after per-worker tail drop.
owned = iterable_shards(23, 2, 2)
steps_with_tail = [sum(len(local_batches(owned[rank * 2 + worker], 3, False))
                       for worker in range(2)) for rank in range(2)]
steps_without_tail = [sum(len(local_batches(owned[rank * 2 + worker], 3, True))
                          for worker in range(2)) for rank in range(2)]
assert steps_with_tail == [4, 4]
assert steps_without_tail == [4, 3]
assert sum(len(shard) % 3 for shard in owned) == 2
# Resume boundary toy: dispatch and committed consumption are different states.
epoch_batches = list(range(1, 121))
consumed, dispatched = 100, 116
assert epoch_batches[consumed:dispatched] == list(range(101, 117))
assert epoch_batches[consumed:] == list(range(101, 121))
assert epoch_batches[dispatched:] == list(range(117, 121))

for bad in [-1, 1.5, True, '1', None]:
    rejects(lambda bad=bad: two_stage(bad, 1, 1))
    rejects(lambda bad=bad: rank_indices(bad, 1, 0))
    rejects(lambda bad=bad: iterable_shards(bad, 1, 1))
for bad in [0, -1, 1.5, True, '1', None]:
    rejects(lambda bad=bad: two_stage(1, bad, 1))
    rejects(lambda bad=bad: two_stage(1, 1, bad))
    rejects(lambda bad=bad: rank_indices(1, bad, 0))
    rejects(lambda bad=bad: local_batches([0], bad))
    rejects(lambda bad=bad: iterable_shards(1, 1, bad))
    rejects(lambda bad=bad: iterable_shards(1, bad, 1))
for bad in [-1, 2, 0.5, True, None]:
    rejects(lambda bad=bad: rank_indices(4, 2, bad))
for bad in [0, 1, None, 'True']:
    rejects(lambda bad=bad: rank_indices(4, 2, 0, bad))
    rejects(lambda bad=bad: local_batches([0], 1, bad))
print('DataLoader CPU examples: timing, drop_last, ownership, shared-permutation, rank-omission and resume-boundary oracles passed; no Torch/CUDA run.')
`;
const python = process.env.BLOG_PYTHON || 'python';
const result = spawnSync(python, ['-B', '-c', code + '\n' + tests], { encoding: 'utf8' });
if (result.error) throw result.error;
process.stdout.write(result.stdout || '');
process.stderr.write(result.stderr || '');
assert.equal(result.status, 0);
