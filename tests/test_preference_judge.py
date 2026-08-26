import copy

import pytest

from src.extraction.preference_judge import (
    JUDGE_RESPONSE_SCHEMA,
    RUBRIC_IDS,
    aggregate_judgments,
    aggregate_user_sessions,
    build_judge_input,
    validate_judgment,
)


def _judgment(overrides=None):
    overrides = overrides or {}
    axes = {}
    for axis_id in RUBRIC_IDS:
        label = overrides.get(axis_id, "na")
        axes[axis_id] = {
            "label": label,
            "confidence": "medium",
            "rationale": "clear" if label != "na" else "no signal",
            "evidence": [] if label == "na" else [{
                "source": "target_user_message", "turn_number": 5, "quote": "please do this"
            }],
        }
    return validate_judgment({"axes": axes})


def test_schema_requires_every_axis():
    assert set(JUDGE_RESPONSE_SCHEMA["properties"]["axes"]["required"]) == set(RUBRIC_IDS)


def test_build_judge_input_marks_context_as_data():
    rendered = build_judge_input({"target_user_message": {"content": "test"}})
    assert rendered.startswith("<preference_context_json>")
    assert rendered.endswith("</preference_context_json>")


def test_validation_adds_ternary_scores_and_rejects_security_low():
    result = _judgment({"solution_scope": "high"})
    assert result["axes"]["solution_scope"]["score"] == 1
    assert result["axes"]["abstraction_preference"]["score"] == 0

    invalid = {"axes": {axis: {
        "label": "na", "confidence": "low", "rationale": "none", "evidence": []
    } for axis in RUBRIC_IDS}}
    invalid["axes"]["security"] = {
        "label": "low", "confidence": "high", "rationale": "bad", "evidence": [{
            "source": "target_user_message", "turn_number": 1, "quote": "x"
        }]
    }
    with pytest.raises(ValueError):
        validate_judgment(invalid)


def test_session_aggregation_uses_support_and_recency_tiebreak():
    judgments = [
        _judgment({"solution_scope": "high"}),
        _judgment({"solution_scope": "low"}),
        _judgment(),
    ]
    session = aggregate_judgments(judgments, level="session")
    axis = session["axes"]["solution_scope"]
    assert axis["support"] == 2
    assert axis["mean_score"] == 0.0
    assert axis["majority_score"] == -1
    assert axis["recent_score"] == -1
    assert axis["conflicted"] is True


def test_user_aggregation_weights_supported_sessions_equally():
    high = aggregate_judgments([_judgment({"solution_scope": "high"})], level="session")
    low = aggregate_judgments([_judgment({"solution_scope": "low"})], level="session")
    user = aggregate_user_sessions([high, low])
    axis = user["axes"]["solution_scope"]
    assert axis["supported_sessions"] == 2
    assert axis["mean_session_score"] == 0.0
    assert axis["majority_score"] == -1
