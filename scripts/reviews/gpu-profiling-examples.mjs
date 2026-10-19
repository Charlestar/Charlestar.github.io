// Extract the article's actual CPU block; no CUDA runtime or profiler is invoked.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';

const post = readFileSync('_posts/2026-10-20-gpu-profiling-nsight-systems-compute.md', 'utf8');
function block(marker) {
  const pattern = new RegExp(`<!-- example: ${marker} -->\\s*\x60\x60\x60python\\r?\\n([\\s\\S]*?)\\r?\\n\x60\x60\x60`, 'g');
  const matches = [...post.matchAll(pattern)];
  assert.equal(matches.length, 1, `expected one ${marker} block`);
  return matches[0][1];
}
const cpu = block('gpu-profiling-critical-path');
const gpu = block('gpu-profiling-workload');
const python = process.env.BLOG_PYTHON || 'python';
const tests = String.raw`
import itertools
import json

def expect_value_error(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("expected ValueError")

# Independent interval oracle: endpoints partition the union into cells.
def union_oracle(intervals):
    endpoints = sorted(set(x for pair in intervals for x in pair))
    return sum(b - a for a, b in zip(endpoints, endpoints[1:])
               if any(s <= (a + b) / 2 < e for s, e in intervals))

for n in range(4):
    choices = [(0, 0), (0, 1), (0, 3), (1, 2), (2, 5), (5, 5)]
    for intervals in itertools.product(choices, repeat=n):
        assert interval_union(intervals) == union_oracle(intervals)
assert interval_union([(5, 7), (0, 2)]) == 4
assert interval_union([(0, 4), (1, 2), (4, 6)]) == 6

for bad in [-1, float("inf"), float("nan"), True, "2"]:
    expect_value_error(lambda bad=bad: earliest_finish({"a": (bad, ())}))
    expect_value_error(lambda bad=bad: interval_union([(0, bad)]))
expect_value_error(lambda: nonnegative_time(10 ** 1000))
expect_value_error(lambda: earliest_finish({"a": (1e308, ()), "b": (1e308, ("a",))}))
expect_value_error(lambda: interval_union([(3, 2)]))
expect_value_error(lambda: earliest_finish({"a": (1, ("missing",))}))
expect_value_error(lambda: earliest_finish({"a": (1, ("a",))}))
expect_value_error(lambda: earliest_finish({"a": (1, ("b",)), "b": (1, ("a",))}))
assert earliest_finish({}) == {}
assert earliest_finish({"zero": (0, ()), "end": (0, ("zero",))})["end"] == 0

# Independent path enumeration oracle on all increasing-edge DAGs of four nodes.
names = ["a", "b", "c", "d"]
edges = [(i, j) for i in range(4) for j in range(i + 1, 4)]
for flags in itertools.product([False, True], repeat=len(edges)):
    predecessors = {name: [] for name in names}
    for (i, j), present in zip(edges, flags):
        if present:
            predecessors[names[j]].append(names[i])
    graph = {name: (i + 1, tuple(predecessors[name])) for i, name in enumerate(names)}
    def all_path_weights(name):
        duration, deps = graph[name]
        return ([duration] if not deps else
                [duration + p for dep in deps for p in all_path_weights(dep)])
    expected = {name: max(all_path_weights(name)) for name in names}
    # Reverse insertion order checks that dictionary order does not define dependency order.
    assert earliest_finish(dict(reversed(list(graph.items())))) == expected

assert earliest_finish(faster_communication)["finalize"] == 9
assert 12 / earliest_finish(faster_communication)["finalize"] == 4 / 3

# Auditor-added discrete occupancy oracle, independent of interval-endpoint merging.
integer_intervals = [(a, b) for a in range(5) for b in range(a, 5)]
for intervals in itertools.product(integer_intervals, repeat=3):
    occupied = {tick for a, b in intervals for tick in range(a, b)}
    assert interval_union(intervals) == len(occupied)

# Auditor-added event simulation: unlimited workers, fixed durations, dependencies only.
# This schedules all ready work and advances to actual completion events; it does not
# evaluate the longest-path recurrence or enumerate paths.
def event_oracle(graph):
    pending, running, done = set(graph), {}, {}
    now = 0
    while pending or running:
        while True:
            ready = [n for n in pending if all(p in done for p in graph[n][1])]
            if not ready:
                break
            for name in ready:
                pending.remove(name)
                end = now + graph[name][0]
                if end == now:
                    done[name] = now
                else:
                    running[name] = end
        if running:
            now = min(running.values())
            ending = [n for n, end in running.items() if end == now]
            for name in ending:
                done[name] = running.pop(name)
        elif pending:
            raise AssertionError("oracle unexpectedly encountered a cycle")
    return done

for flags in itertools.product([False, True], repeat=len(edges)):
    predecessors = {name: [] for name in names}
    for (i, j), present in zip(edges, flags):
        if present:
            predecessors[names[j]].append(names[i])
    for durations in itertools.product([0, 1, 3], repeat=4):
        graph = {n: (d, tuple(predecessors[n])) for n, d in zip(names, durations)}
        assert earliest_finish(graph) == event_oracle(graph)

# Serial resource exclusion is NOT inferred: it has to be represented by an edge.
independent = {"a": (6, ()), "b": (6, ())}
serialized = {"a": (6, ()), "b": (6, ("a",))}
assert max(earliest_finish(independent).values()) == 6
assert max(earliest_finish(serialized).values()) == 12
print(json.dumps({"cpu_groups": 9, "gpu_runtime": "not executed"}))
`;
const result = spawnSync(python, ['-B', '-c', `${cpu}\n${tests}`], { encoding: 'utf8' });
assert.equal(result.status, 0, `${result.error || ''}\n${result.stdout}\n${result.stderr}`);
const syntax = spawnSync(python, ['-B', '-c', `import ast\nast.parse(${JSON.stringify(gpu)})\nprint('GPU workload syntax parsed; not executed')`], { encoding: 'utf8' });
assert.equal(syntax.status, 0, `${syntax.error || ''}\n${syntax.stderr}`);
console.log(result.stdout.trim());
console.log(syntax.stdout.trim());
console.log('PASS GPU profiling arithmetic and extracted-code checks; no GPU performance measured.');
