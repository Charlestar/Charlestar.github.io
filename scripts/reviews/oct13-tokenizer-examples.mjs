import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';

const article = readFileSync(new URL('../../_posts/2026-10-13-tokenizer-bpe-model-input.md', import.meta.url), 'utf8');
function markedBlock(name) {
  const marker = `<!-- tokenizer-example: ${name} -->`;
  const tail = article.split(marker);
  assert.equal(tail.length, 2, `Missing or duplicate marker ${name}`);
  const match = tail[1].match(/^\s*```python\r?\n([\s\S]*?)```/);
  assert.ok(match, `Missing Python block after ${name}`);
  return match[1];
}
const training = article.match(/ab\s+出现 (\d+) 次\r?\nbc\s+出现 (\d+) 次\r?\nabc\s+出现 (\d+) 次/);
assert.ok(training, 'Missing weighted corpus in the article');
const payload = {
  toy: markedBlock('toy-bpe'),
  stream: markedBlock('utf8-stream'),
  pythonBlocks: [...article.matchAll(/```python\r?\n([\s\S]*?)```/g)].map(match => match[1]),
  frequencies: [...training].slice(1).map(Number),
};
// Execute the actual prose examples. No tokenizer package, model, API, or GPU.
// Python's standard library supplies independent Unicode and training checks.
const program = String.raw`
import ast
import codecs
import contextlib
import io
import itertools
import json
import sys
import unicodedata
from collections import Counter

data = json.load(sys.stdin)
passed = 0
def check(name, action):
    global passed
    action()
    passed += 1
    print("PASS", name)

toy = {}
printed = io.StringIO()
with contextlib.redirect_stdout(printed):
    exec(compile(data["toy"], "article:toy-bpe", "exec"), toy)
encode, decode = toy["encode"], toy["decode"]

def direct_examples():
    assert printed.getvalue().splitlines() == ["[50, 60]", "bcabc", "[10, 20]", "[40]"]
    assert encode("bcabc") == [50, 60]
    assert toy["RANK"][("a", "b")] == 0
    assert toy["VOCAB"]["ab"] == 40
    assert max(toy["VOCAB"].values()) + 1 == 61
check("actual article code prints its four declared results", direct_examples)

def weighted_training():
    corpus = [(list(word), count) for word, count in zip(("ab", "bc", "abc"), data["frequencies"])]
    expected = [{("a", "b"): 7, ("b", "c"): 6}, {("b", "c"): 5, ("ab", "c"): 1}, {("ab", "c"): 1}]
    learned = []
    for expected_stats in expected:
        stats = Counter()
        for pieces, count in corpus:
            for pair in zip(pieces, pieces[1:]):
                stats[pair] += count
        assert dict(stats) == expected_stats
        pair = max(stats, key=stats.get)
        learned.append(pair)
        # Independent training transformation, not a second copy of encode().
        updated = []
        for pieces, count in corpus:
            merged, pos = [], 0
            while pos < len(pieces):
                if tuple(pieces[pos:pos+2]) == pair:
                    merged.append("".join(pair)); pos += 2
                else:
                    merged.append(pieces[pos]); pos += 1
            updated.append((merged, count))
        corpus = updated
    assert learned == toy["MERGES"]
check("weighted corpus learns the three published merges", weighted_training)

def independent_oracle(text):
    # Apply the learned operations once in historical order, rather than
    # searching all currently eligible ranks after each single merge.
    pieces = list(text)
    for left, right in toy["MERGES"]:
        result, pos = [], 0
        while pos < len(pieces):
            if pieces[pos:pos+2] == [left, right]:
                result.append(left + right); pos += 2
            else:
                result.append(pieces[pos]); pos += 1
        pieces = result
    return [toy["VOCAB"][piece] for piece in pieces]
def exhaustive_small_alphabet():
    count = 0
    for length in range(8):
        for chars in itertools.product("abc", repeat=length):
            text = "".join(chars)
            assert encode(text) == independent_oracle(text), text
            assert decode(encode(text)) == text
            count += 1
    assert count == 3280
check("3280 strings agree with a differently organized merge oracle", exhaustive_small_alphabet)

def frozen_rules_and_domain():
    ranks = toy["RANK"].copy()
    for text in ("", "bcabc", "bcbcbcab", "aaaa"):
        encode(text)
    assert toy["RANK"] == ranks
    assert encode("") == [] and decode([]) == ""
    for text in ("A", "中", "a b", "d"):
        try:
            encode(text)
        except ValueError:
            continue
        raise AssertionError("Accepted a symbol outside the stated alphabet")
check("encoding leaves learned ranks fixed and rejects unsupported symbols", frozen_rules_and_domain)

def concat_and_prefix():
    a, b, joined = encode("a"), encode("b"), encode("ab")
    assert a + b == [10, 20] and joined == [40]
    assert len(a) + len(b) != len(joined)
    assert joined[:len(a)] != a
    # Explicit pre-token boundaries prevent the cross-boundary merge.
    assert [token for part in ("a", "b") for token in encode(part)] == [10, 20]
check("text concat, token concat, and pre-token boundaries differ", concat_and_prefix)

stream = {}
stream_printed = io.StringIO()
with contextlib.redirect_stdout(stream_printed):
    exec(compile(data["stream"], "article:utf8-stream", "exec"), stream)
def stream_example():
    assert stream["output"] == ["", "", "中", ""]
    assert stream_printed.getvalue().splitlines() == ["['', '', '中', '']", "中", "���"]
    assert b"".join(stream["chunks"]) == bytes.fromhex("e4 b8 ad")
check("actual article incremental decoder preserves all three UTF8 bytes", stream_example)

def partial_utf8():
    for split in (1, 2):
        decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
        encoded = "中".encode("utf-8")
        assert decoder.decode(encoded[:split], final=False) == ""
        try:
            decoder.decode(b"", final=True)
        except UnicodeDecodeError:
            pass
        else:
            raise AssertionError("A final incomplete UTF8 sequence was silently accepted")
    assert "".join(part.decode("utf-8", "replace") for part in stream["chunks"]) != "中"
check("unfinished prefixes and invalid final UTF8 have different semantics", partial_utf8)

def normalizer_information_loss():
    composed, decomposed = "é", "e\u0301"
    assert composed != decomposed
    assert len(composed) == 1 and len(decomposed) == 2
    assert unicodedata.normalize("NFC", composed) == unicodedata.normalize("NFC", decomposed)
    assert "A".lower() == "a".lower()
    assert "A" != "a"
check("normalization can map distinct raw strings to the same result", normalizer_information_loss)

def token_crop_reencoding():
    # These are valid toy vocabulary IDs, but need not be its canonical encode.
    original_prefix = [10, 20]
    assert decode(original_prefix) == "ab"
    assert encode(decode(original_prefix)) == [40] != original_prefix
check("decode then reencode need not preserve a supplied token prefix", token_crop_reencoding)

def syntax_check():
    for block in data["pythonBlocks"]:
        ast.parse(block)
check("all article Python blocks parse without running the external HF fragment", syntax_check)
print(f"{passed} Tokenizer groups passed; stdlib teaching examples only, no production tokenizer or model benchmark.")
`;
const python = process.env.BLOG_PYTHON ?? (process.platform === 'win32' ? 'python' : 'python3');
const result = spawnSync(python, ['-c', program], {
  input: JSON.stringify(payload), encoding: 'utf8',
  env: { ...process.env, PYTHONUTF8: '1' },
});
if (result.stdout) process.stdout.write(result.stdout);
if (result.stderr) process.stderr.write(result.stderr);
if (result.error) throw result.error;
assert.equal(result.status, 0, 'Tokenizer prose example verification failed');
