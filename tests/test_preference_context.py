import pandas as pd

from src.extraction.preference_context import build_preference_context


def _row(turn, role, turn_type, content="", conversational=False, **extra):
    return {
        "session_id": "s1",
        "user_id": "u1",
        "repo_id": "r1",
        "turn_number": turn,
        "conversation_turn_number": None,
        "role": role,
        "turn_type": turn_type,
        "is_conversational": conversational,
        "content": content,
        "timestamp": None,
        "checkpoint_pk": "cp1",
        "tool_name": None,
        "tool_call_id": None,
        "file_path": None,
        "tool_input_json": None,
        **extra,
    }


def test_context_uses_mapper_prompt_boundaries_and_attaches_edges():
    conversation = pd.DataFrame([
        _row(1, "user", "user_prompt", "u1", True),
        _row(2, "assistant", "assistant_response", "a1", True),
        _row(3, "user", "user_prompt", "u2", True),
        _row(4, "tool_use", "tool_use", tool_name="Read", tool_call_id="r", tool_input_json='{"file_path":"a.py"}'),
        _row(5, "tool_result", "tool_result", "old code", tool_name="Read", tool_call_id="r"),
        _row(6, "user", "user_prompt", "u3", True),
        _row(
            7, "tool_use", "tool_use", tool_name="Edit", tool_call_id="e",
            file_path="/repo/a.py",
            tool_input_json='{"file_path":"/repo/a.py","old_string":"old()","new_string":"new()"}',
        ),
        _row(8, "tool_result", "tool_result", "updated", tool_name="Edit", tool_call_id="e"),
        _row(9, "assistant", "assistant_response", "done", True),
        _row(10, "user", "user_prompt", "u4", True),
    ])
    edges = pd.DataFrame([{
        "session_id": "s1", "user_turn_number": 6,
        "tool_turn_number": 7, "action_index": 0,
        "commit_sha": "abc", "survival_recall": 1.0,
        "confidence": "exact",
    }])
    summaries = pd.DataFrame([{
        "session_id": "s1", "user_turn_number": 6,
        "commit_sha": "abc", "mean_text_recall": 1.0,
    }])

    context = build_preference_context(conversation, 6, edges, summaries)

    assert context["event"]["target_user_message_index"] == 3
    assert [row["turn_number"] for row in context["previous_change"]["rows"]] == [4]
    assert [row["turn_number"] for row in context["next_change"]["rows"]] == [7, 9]
    change = context["next_change"]["rows"][0]["tool"]["code_changes"][0]
    assert change["added_lines_normalized"] == ["new()"]
    assert change["deleted_lines_normalized"] == ["old()"]
    assert context["commit_survival_evidence"]["action_edges"][0]["commit_sha"] == "abc"
    assert "checkpoint_pk" not in context["target_user_message"]
    assert "content" not in context["next_change"]["rows"][0]
    assert "tool_call_id" not in context["next_change"]["rows"][0]["tool"]
    assert "role" not in context["next_change"]["rows"][0]
    assert context["next_change"]["rows"][0]["tool"]["result"] == "updated"


def test_non_judge_metadata_rows_are_removed():
    conversation = pd.DataFrame([
        _row(1, "user", "user_prompt", "u1", True),
        _row(2, "metadata", "file_snapshot", "large snapshot"),
        _row(3, "assistant", "assistant_response", "done", True),
        _row(4, "metadata", "system_event", "hook internals"),
        _row(5, "user", "user_prompt", "u2", True),
        _row(6, "metadata", "queue_operation", "queue internals"),
        _row(7, "assistant", "assistant_response", "working", True),
        _row(8, "user", "user_prompt", "u3", True),
    ])

    context = build_preference_context(conversation, 5)

    assert [row["turn_number"] for row in context["previous_change"]["rows"]] == [3]
    assert [row["turn_number"] for row in context["next_change"]["rows"]] == [7]


def test_snapshot_is_kept_only_when_it_contains_file_content():
    conversation = pd.DataFrame([
        _row(1, "user", "user_prompt", "u1", True),
        _row(2, "metadata", "file_snapshot", '{"snapshot":{"trackedFileBackups":{"a.py":{"backupFileName":"v1"}}}}'),
        _row(3, "metadata", "file_snapshot", '{"snapshot":{"files":{"a.py":{"content":"x = 1"}}}}'),
        _row(4, "user", "user_prompt", "u2", True),
        _row(5, "assistant", "assistant_response", "done", True),
        _row(6, "user", "user_prompt", "u3", True),
    ])

    context = build_preference_context(conversation, 4)

    assert context["previous_change"]["rows"] == [{
        "turn_number": 3,
        "role": "file_snapshot",
        "files": [{"path": "a.py", "content": "x = 1"}],
    }]


def test_non_prompt_target_is_rejected():
    conversation = pd.DataFrame([
        _row(1, "user", "user_prompt", "u1", True),
        _row(2, "assistant", "assistant_response", "a1", True),
    ])
    try:
        build_preference_context(conversation, 2)
    except ValueError as error:
        assert "not a mapper-compatible" in str(error)
    else:
        raise AssertionError("expected ValueError")
