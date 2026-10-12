import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';

const article = readFileSync(new URL('../../_posts/2026-10-13-flexattention-score-mod-block-mask.md', import.meta.url), 'utf8');
function markedBlock(name) {
  const marker = `<!-- flex-example: ${name} -->`;
  const parts = article.split(marker);
  assert.equal(parts.length, 2, `Missing or duplicate marker ${name}`);
  const match = parts[1].match(/^\s*```python\r?\n([\s\S]*?)```/);
  assert.ok(match, `Missing Python block after ${name}`);
  return match[1];
}
const payload = {
  softcap: markedBlock('softcap-order'),
  blocks: markedBlock('block-counts'),
  decode: markedBlock('decode-coordinates'),
  api: markedBlock('pytorch-api'),
  all: [...article.matchAll(/```python\r?\n([\s\S]*?)```/g)].map(match => match[1]),
};
// Directly execute the three actual standard-library examples from the post.
// The CUDA API fragment is parsed and inspected, never mocked or executed.
const program = String.raw`
import ast
import contextlib
import io
import json
import math
import sys

data = json.load(sys.stdin)
passed = 0
def check(name, action):
    global passed
    action()
    passed += 1
    print("PASS", name)

def run_article_block(key):
    scope, printed = {}, io.StringIO()
    with contextlib.redirect_stdout(printed):
        exec(compile(data[key], "article:" + key, "exec"), scope)
    return scope, printed.getvalue().splitlines()

softcap, softcap_output = run_article_block("softcap")
def softcap_order():
    assert softcap_output == ["[1.0, 0.0]", "[0.993307, 0.006693]"]
    assert softcap["softmax"](softcap["correct"]) == [1.0, 0.0]
    # Independently derive the illegal connection's restored weight.
    expected_illegal = 1 / (1 + math.exp(softcap["cap"]))
    actual_illegal = softcap["softmax"](softcap["incorrect"])[1]
    assert math.isclose(actual_illegal, expected_illegal, rel_tol=1e-14)
    assert actual_illegal > 0
check("article code and independent logistic identity expose mask-before-softcap error", softcap_order)

def zero_is_not_mask():
    assert softcap["softmax"]([0.0, 0.0]) == [0.5, 0.5]
    assert softcap["softmax"]([0.0, -math.inf]) == [1.0, 0.0]
check("zero scores retain probability while negative infinity removes a connection", zero_is_not_mask)

def all_masked_policy():
    try:
        softcap["softmax"]([-math.inf, -math.inf])
    except ValueError:
        pass
    else:
        raise AssertionError("The teaching reference silently accepted an all-masked row")
    # A True structural mask need not imply a finite post-score_mod value.
    allowed, transformed = [True], [-math.inf]
    assert any(allowed) and not any(ok and math.isfinite(s) for ok, s in zip(allowed, transformed))
check("reference all-masked policy and finite-score condition are explicit", all_masked_policy)

blocks, blocks_output = run_article_block("blocks")
def causal_blocks():
    assert blocks_output == ["P 0 0", "F P 0", "F F P", "{'F': 3, 'P': 3, '0': 3}", "21 24"]
    length, tile = blocks["length"], blocks["tile"]
    assert length % tile == 0
    # Interval endpoints classify causal rectangles without enumerating the
    # post's per-cell booleans or copying its any/all classification loop.
    kinds = []
    for qmin in range(0, length, tile):
        qmax = qmin + tile - 1
        for kmin in range(0, length, tile):
            kmax = kmin + tile - 1
            kinds.append("F" if kmax <= qmin else ("0" if kmin > qmax else "P"))
    independent_counts = {kind: kinds.count(kind) for kind in ("F", "P", "0")}
    assert independent_counts == blocks["counts"]
    assert blocks["allowed"] == length * (length + 1) // 2 == 21
    assert blocks["visited_cells"] == 24
    assert length * length - blocks["allowed"] == 15
    assert blocks["counts"]["0"] * tile * tile == 12
check("actual block example agrees with interval and triangular-count oracles", causal_blocks)

def window_sparsity_counterexample():
    # S=8, tile=4: causal and W=3 retain the same three block rectangles.
    # Count block occupancy via visible coordinates, not simulated GPU work.
    causal = {(q, k) for q in range(8) for k in range(q + 1)}
    window = {(q, k) for q in range(8) for k in range(max(0, q - 2), q + 1)}
    causal_tiles = {(q // 4, k // 4) for q, k in causal}
    window_tiles = {(q // 4, k // 4) for q, k in window}
    assert len(causal) == 36 and len(window) == 21
    assert causal_tiles == window_tiles == {(0, 0), (1, 0), (1, 1)}
    assert len(causal_tiles) * 16 == len(window_tiles) * 16 == 48
check("fewer visible elements need not remove more whole tiles", window_sparsity_counterexample)

def packed_boundaries():
    segment = [0, 0, 0, 1, 1, 1]
    position = [0, 1, 2, 0, 1, 2]
    visible = {(q, k) for q in range(6) for k in range(6)
               if segment[q] == segment[k] and 0 <= position[q] - position[k] < 3}
    assert len(visible) == 2 * (3 * 4 // 2) == 12
    assert {k for q, k in visible if q == 3} == {3}
    reset_only = {k for k in range(6) if 0 <= position[3] - position[k] < 3}
    assert reset_only == {0, 3}  # Local positions alone leak across the boundary.
    assert {k for k in range(6) if k <= 3} == {0, 1, 2, 3}
check("packing requires sample identity in addition to positions and causal ordering", packed_boundaries)

decode, decode_output = run_article_block("decode")
def decode_coordinates():
    assert decode_output == ["[3, 4, 5]"]
    offset, local = decode["q_offset"], decode["q_idx"]
    assert local == 0 and offset == 5
    expected = list(range(max(0, offset + local - decode["window"] + 1),
                          min(decode["valid_length"], offset + local + 1)))
    assert decode["visible"] == expected == [3, 4, 5]
    assert [k for k in range(decode["capacity"]) if k <= local] == [0]
    assert {6, 7}.isdisjoint(decode["visible"])
check("actual decode example agrees with a global-coordinate range oracle", decode_coordinates)

def strict_window_boundary():
    assert [k for k in range(6) if 0 <= 5 - k < 3] == [3, 4, 5]
    assert [k for k in range(6) if 0 <= 5 - k <= 3] == [2, 3, 4, 5]
    assert [k for k in range(6) if 0 <= 0 - k < 3] == [0]
check("window includes self and has a strict left-distance bound", strict_window_boundary)

def stale_skipped_blocks():
    # A one-query causal pattern at global 0 originally keeps only tile 0.
    # Moving the query to global 5 does not modify those stored tile indices.
    old_keys, new_keys = set(range(1)), set(range(6))
    stored_blocks = {k // 2 for k in old_keys}
    required_blocks = {k // 2 for k in new_keys}
    assert stored_blocks == {0} and required_blocks == {0, 1, 2}
    visited_with_old_metadata = {k for k in new_keys if k // 2 in stored_blocks}
    assert new_keys - visited_with_old_metadata == {2, 3, 4, 5}
check("captured offset updates cannot resurrect blocks absent from stored metadata", stale_skipped_blocks)

def stale_full_blocks():
    # At global 5, blocks [0,1], [2,3], [4,5] are full for one query.
    # At global 2, the old full block [2,3] is now partial; skipping its new
    # element mask would incorrectly retain k=3, even with the same shape.
    old_full = [set(range(start, start + 2)) for start in (0, 2, 4)]
    new_visible = set(range(3))
    assert old_full[1] & new_visible == {2}
    assert old_full[1] - new_visible == {3}
check("stale full-block classifications can bypass a newly necessary mask", stale_full_blocks)

def parse_api_only():
    for block in data["all"]:
        ast.parse(block)
    tree = ast.parse(data["api"])
    imports = [node for node in tree.body if isinstance(node, ast.ImportFrom)]
    assert any(node.module == "torch.nn.attention.flex_attention"
               and {alias.name for alias in node.names} == {"create_block_mask", "flex_attention"}
               for node in imports)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    mask_calls = [node for node in calls if isinstance(node.func, ast.Name) and node.func.id == "create_block_mask"]
    assert len(mask_calls) == 1
    kwargs = {kw.arg: kw.value for kw in mask_calls[0].keywords}
    assert ast.literal_eval(kwargs["B"]) is None and ast.literal_eval(kwargs["H"]) is None
    assert ast.literal_eval(kwargs["Q_LEN"]) == ast.literal_eval(kwargs["KV_LEN"]) == 256
    assert "_compile" not in kwargs
    assert any(isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
               and node.func.value.id == "torch" and node.func.attr == "compile"
               and isinstance(node.args[0], ast.Name) and node.args[0].id == "flex_attention"
               for node in calls)
check("all Python parses and pinned API fragment is statically checked only", parse_api_only)

print(f"{passed} FlexAttention groups passed; stdlib arithmetic and AST only, no torch import, CUDA, compilation, or GPU benchmark.")
`;
const python = process.env.BLOG_PYTHON ?? (process.platform === 'win32' ? 'python' : 'python3');
const result = spawnSync(python, ['-c', program], {
  input: JSON.stringify(payload), encoding: 'utf8',
  env: { ...process.env, PYTHONUTF8: '1' },
});
if (result.stdout) process.stdout.write(result.stdout);
if (result.stderr) process.stderr.write(result.stderr);
if (result.error) throw result.error;
assert.equal(result.status, 0, 'FlexAttention prose example verification failed');
