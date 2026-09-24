import pandas as pd

from src.pilot_study.extract_four_user_preferences import (
    aggregate_user,
    compute_session_metrics,
    leakage_safe_prior_sessions,
    rank_backfill_sessions,
    render_interventions,
    turn_evidence_packet,
)


def test_leakage_safe_sessions_intersect_selected_tasks():
    rows = [
        {"prior_session_ids": ["a", "b"], "prior_sessions": [
            {"session_id": "a", "timestamp": "2026-01-01"},
            {"session_id": "b", "timestamp": "2026-01-02"},
        ]},
        {"prior_session_ids": ["a"], "prior_sessions": [
            {"session_id": "a", "timestamp": "2026-01-01"},
        ]},
    ]
    assert [item["session_id"] for item in leakage_safe_prior_sessions(rows)] == ["a"]


def test_pushback_rate_uses_conversational_user_turns_as_denominator():
    frame = pd.DataFrame([
        {"turn_number": 0, "role": "user", "is_conversational": True,
         "prompt_pushback": "correction"},
        {"turn_number": 1, "role": "assistant", "is_conversational": True,
         "prompt_pushback": None},
        {"turn_number": 2, "role": "user", "is_conversational": True,
         "prompt_pushback": "non_pushback"},
        {"turn_number": 3, "role": "user", "is_conversational": False,
         "prompt_pushback": "correction"},
    ])
    metrics = compute_session_metrics(frame)
    assert metrics["conversational_turns"] == 3
    assert metrics["conversational_user_turns"] == 2
    assert metrics["pushback_turns"] == 1
    assert metrics["pushback_rate"] == 0.5


def _record(session, turn, label, description):
    return {
        "user": "u", "session_id": session, "turn_number": turn,
        "axis_id": "solution_scope", "axis_name": "Solution Scope",
        "active_rationale": "direct correction", "user_message": f"User message {turn}",
        "method1": {"label": label, "confidence": "high", "evidence_quote": "x", "rationale": "x"},
        "method2": {"useful": True, "description": description,
                    "evidence_quote": "x", "confidence": "high"},
    }


def test_aggregation_and_three_renderers_keep_contextual_exception():
    records = [
        _record("s1", 1, "low", "Keep narrowly scoped fixes localized."),
        _record("s2", 2, "low", "Avoid unrelated adjacent cleanup."),
        _record("s3", 3, "high", "Complete directly related follow-up work."),
    ]
    profile = aggregate_user("u", records, threshold=0.6, min_sessions=2,
                             descriptions_per_axis=4)
    axis = next(a for a in profile["axes"] if a["axis_id"] == "solution_scope")
    assert axis["direction"] == "low"
    assert axis["high_confidence"] is True
    rendered = render_interventions(profile, max_chars=4000)
    assert set(rendered) == {
        "1_high_confidence_directional",
        "2_high_confidence_contextual",
        "3_rubric_descriptions_no_polarity",
    }
    contextual = rendered["2_high_confidence_contextual"]
    assert "2 turns scored low" in contextual
    assert "1 turn scored high" in contextual
    assert "**High:**" in contextual
    assert "**Low:**" in contextual
    assert "User turn: User message 3" in contextual
    assert "Keep narrowly scoped fixes localized" in rendered["3_rubric_descriptions_no_polarity"]


def test_descriptive_intervention_keeps_every_turn_description_without_confidence_cap():
    records = [
        _record(f"s{i}", i, "low", f"Preference description {i}.")
        for i in range(1, 6)
    ]
    records[-1]["method2"]["confidence"] = "low"
    profile = aggregate_user("u", records, threshold=0.99, min_sessions=10,
                             descriptions_per_axis=1)
    rendered = render_interventions(profile, max_chars=120)
    detailed = rendered["3_rubric_descriptions_no_polarity"]
    for i in range(1, 6):
        assert f"Preference description {i}." in detailed


def test_contextual_intervention_caps_user_messages_at_three_per_rubric():
    records = [
        _record("s1", 1, "high", "d1"),
        _record("s2", 2, "high", "d2"),
        _record("s3", 3, "high", "d3"),
        _record("s4", 4, "low", "d4"),
        _record("s5", 5, "low", "d5"),
    ]
    profile = aggregate_user("u", records, threshold=0.5, min_sessions=1,
                             descriptions_per_axis=1)
    contextual = render_interventions(profile, max_chars=4000)[
        "2_high_confidence_contextual"
    ]
    assert contextual.count("User turn:") == 3
    assert "User message 1" in contextual
    assert "User message 4" in contextual


def test_ranked_backfill_puts_strict_sessions_first():
    sessions = [
        {"session_id": "backfill", "conversational_turns": 30,
         "pushback_rate": 0.6, "pushback_turns": 9, "timestamp": "2026-02-02"},
        {"session_id": "strict", "conversational_turns": 11,
         "pushback_rate": 0.7, "pushback_turns": 7, "timestamp": "2026-01-01"},
    ]
    ranked = rank_backfill_sessions(sessions)
    assert [item["session_id"] for item in ranked] == ["strict", "backfill"]
    assert ranked[0]["meets_strict_filter"] is True


def test_turn_packet_labels_response_and_later_feedback():
    frame = pd.DataFrame([
        {"turn_number": 0, "role": "user", "is_conversational": True,
         "content": "initial", "prompt_pushback": "non_pushback"},
        {"turn_number": 1, "role": "assistant", "is_conversational": True,
         "content": "first response", "prompt_pushback": None},
        {"turn_number": 2, "role": "user", "is_conversational": True,
         "content": "keep it narrow", "prompt_pushback": "correction"},
        {"turn_number": 3, "role": "assistant", "is_conversational": True,
         "content": "revised", "prompt_pushback": None},
        {"turn_number": 4, "role": "user", "is_conversational": True,
         "content": "yes, but add a test", "prompt_pushback": "requirement_change"},
        {"turn_number": 5, "role": "assistant", "is_conversational": True,
         "content": "test added", "prompt_pushback": None},
    ])
    packet = turn_evidence_packet(frame, 2)
    assert "## Current user message being scored" in packet
    assert "keep it narrow" in packet
    assert "## Agent response after the scored message" in packet
    assert "revised" in packet
    assert "## Subsequent user feedback" in packet
    assert "yes, but add a test" in packet
