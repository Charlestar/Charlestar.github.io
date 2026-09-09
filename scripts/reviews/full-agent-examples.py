"""Execute the actual article schema and approval functions with a fake interrupt.

No LangGraph deployment or external messages are executed.
"""
import ast
from copy import deepcopy
from pathlib import Path
import re

article = Path(__file__).resolve().parents[2] / "_posts/2026-05-14-langgraph-mcp-practical-guide.md"
source = article.read_text(encoding="utf-8")
blocks = re.findall(r"```python\s*\n(.*?)```", source, re.S)
namespace = {}
for code in blocks:
    if "class ResearchState" not in code and "def proposed_action" not in code:
        continue
    tree = ast.parse(code)
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.ImportFrom) and node.module == "langgraph.types"
    )]
    exec(compile(tree, str(article), "exec"), namespace)

state = {
    "request_id": "unit-test-request",
    "draft": "Verified report",
    "recipients": ["recipient@invalid.test"],
    "subject": "Report",
    "report_revision": "1",
    "approval": "pending",
    "approved_action_id": None,
}
action = namespace["proposed_action"](state)
for decision in ["rejected", "false", True, {}, {"approve": False},
                 {"approve": "true", "action_id": action["action_id"]},
                 {"approve": True, "action_id": "old"}]:
    namespace["interrupt"] = lambda proposed, response=decision: response
    assert namespace["approval_node"](state)["approval"] == "rejected"

namespace["interrupt"] = lambda proposed: {"approve": True, "action_id": proposed["action_id"]}
approved = {**state, **namespace["approval_node"](state)}
assert namespace["require_current_approval"](approved)["action_id"] == action["action_id"]
for field, value in [("draft", "Changed"), ("recipients", ["new@invalid.test"]),
                     ("subject", "Changed"), ("report_revision", "2")]:
    changed = {**approved, field: value}
    try:
        namespace["require_current_approval"](changed)
    except PermissionError:
        pass
    else:
        raise AssertionError(f"Stale approval accepted for {field}")

annotations = namespace["ResearchState"].__annotations__
for field in state:
    assert field in annotations, field

merge = namespace["merge_evidence"]
a = {"source_id": "a", "title": "A", "uri": "urn:a", "excerpt": "A", "retrieved_at": "fixed"}
b = {**a, "source_id": "b"}
assert merge([a], [b]) == merge([b], [a])
assert merge([a], [deepcopy(a)]) == [a]
try:
    merge([a], [{**a, "excerpt": "conflicting"}])
except ValueError:
    pass
else:
    raise AssertionError("Conflicting evidence was accepted")
print("Validated article approval types, revision binding, schema fields and reducer; no external side effects.")
