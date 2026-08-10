import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "src/mapping/build_turn_commit_map.py"
SPEC = importlib.util.spec_from_file_location("turn_commit_map", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_edit_delta_removes_unchanged_context():
    added, deleted = MODULE.edit_delta("a\nold\nz", "a\nnew\nz")
    assert added == ["new"]
    assert deleted == ["old"]


def test_parse_unified_patch_keeps_changed_text_per_file():
    patch = """diff --git a/pkg/a.py b/pkg/a.py
--- a/pkg/a.py
+++ b/pkg/a.py
@@ -1 +1 @@
-old()
+new()
diff --git a/pkg/b.py b/pkg/b.py
--- a/pkg/b.py
+++ b/pkg/b.py
@@ -0,0 +1 @@
+created = True
"""
    parsed = MODULE.parse_unified_patch(patch)
    assert parsed["pkg/a.py"].added == ["new()"]
    assert parsed["pkg/a.py"].deleted == ["old()"]
    assert parsed["pkg/b.py"].added == ["created = True"]


def test_path_mapping_prefers_repo_relative_suffix():
    paths = ["src/a.py", "tests/a.py"]
    assert MODULE.canonical_path("/tmp/repo/src/a.py", paths) == "src/a.py"
    assert MODULE.canonical_path("/tmp/repo/a.py", paths) is None


def test_multiset_recall_does_not_double_count_repeated_lines():
    assert MODULE.multiset_recall(["x", "x"], ["x"]) == 0.5


def test_exact_match_requires_changed_text_on_present_sides():
    import pandas as pd

    action = {
        "session_id": "s",
        "user_turn_number": 1,
        "conversation_turn_number": 0,
        "tool_turn_number": 2,
        "tool_name": "Edit",
        "file_path": "/repo/pkg/a.py",
        "added_lines": ["new()"],
        "deleted_lines": ["old()"],
    }
    commit = pd.Series({
        "commit_sha": "abc",
        "commit_message": "fix",
        "commit_date": None,
        "file_attribution": "{}",
        "patch": """diff --git a/pkg/a.py b/pkg/a.py
--- a/pkg/a.py
+++ b/pkg/a.py
@@ -1 +1 @@
-old()
+new()
""",
    })
    edge = MODULE.match_action(action, commit)
    assert edge is not None
    assert edge["confidence"] == "exact"
    assert edge["changed_text_recall"] == 1.0


def test_new_text_survival_can_be_exact_when_old_text_was_uncommitted():
    import pandas as pd

    action = {
        "session_id": "s", "user_turn_number": 1,
        "conversation_turn_number": 0, "tool_turn_number": 2,
        "tool_name": "Edit", "file_path": "/repo/CLAUDE.md",
        "added_lines": ["new guidance"], "deleted_lines": ["temporary guidance"],
    }
    commit = pd.Series({
        "commit_sha": "abc", "commit_message": "docs", "commit_date": None,
        "file_attribution": "{}",
        "patch": """diff --git a/CLAUDE.md b/CLAUDE.md
--- a/CLAUDE.md
+++ b/CLAUDE.md
@@ -0,0 +1 @@
+new guidance
""",
    })
    edge = MODULE.match_action(action, commit)
    assert edge["added_recall"] == 1.0
    assert edge["deleted_recall"] == 0.0
    assert edge["survival_recall"] == 1.0
    assert edge["confidence"] == "exact"
