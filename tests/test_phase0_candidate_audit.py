from pathlib import Path
import importlib.util

import pandas as pd


MODULE_PATH = Path(__file__).parents[1] / "src/evaluation/phase0_candidate_audit.py"
SPEC = importlib.util.spec_from_file_location("phase0_candidate_audit", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_chronological_split_keeps_at_least_three_past_sessions():
    frame = pd.DataFrame({
        "session_id": list("abcde"),
        "created_at": pd.date_range("2025-01-01", periods=5, tz="UTC"),
    })
    past, future = MODULE.chronological_split(frame, min_past=3, future_fraction=0.30)
    assert past.session_id.tolist() == ["a", "b", "c"]
    assert future.session_id.tolist() == ["d", "e"]


def test_axis_hints_excludes_chat_only_axes_and_zeros():
    row = pd.Series({"score_R02": -1, "score_R06": 0, "score_R11": 1})
    assert MODULE.axis_hints(row) == ["R02 Abstraction Level (-1)"]


def test_action_summary_preserves_changed_text():
    action = {
        "file_path": "/repo/a.py",
        "tool_turn_number": 9,
        "added_lines": ["return value"],
        "deleted_lines": ["return None"],
    }
    paths, added, deleted = MODULE.action_summary([action])
    assert paths == "/repo/a.py"
    assert "return value" in added
    assert "return None" in deleted
