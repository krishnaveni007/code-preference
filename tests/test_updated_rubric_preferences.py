from src.pilot_study.extract_updated_rubric_preferences import (
    RUBRICS,
    aggregate,
    render_all_confidence_bands,
    render_contexts,
    validate_preferences,
)


def test_updated_rubric_has_four_one_sided_axes():
    assert len(RUBRICS) == 15
    assert {rubric["id"] for rubric in RUBRICS if rubric["low"] is None} == {
        "documentation_workflow",
        "test_execution",
        "refactoring_tolerance",
        "config_externalization",
    }


def test_validation_drops_low_for_one_sided_axis():
    raw = {"preferences": [{
        "axis_id": "documentation_workflow",
        "label": "low",
        "confidence": "high",
        "context": "When the change is small.",
        "preference": "Do not update documentation.",
        "rationale": "Direct request.",
    }]}
    assert validate_preferences(raw) == []


def test_context_renderer_uses_natural_preference_headings():
    records = [
        {
            "user": "Soph", "session_id": "s1", "turn_number": 1,
            "axis_id": "git_automation", "label": "high", "confidence": "high",
            "context": "When delivery is fully delegated.",
            "preference": "Complete the git workflow.", "rationale": "Direct request.",
        },
        {
            "user": "Soph", "session_id": "s2", "turn_number": 2,
            "axis_id": "git_automation", "label": "low", "confidence": "high",
            "context": "When review is requested before commit.",
            "preference": "Wait before git steps.", "rationale": "Direct request.",
        },
    ]
    rendered = render_contexts(aggregate(records, "Soph"))
    assert "User prefers the agent to create branches" in rendered
    assert "User prefers the agent to stop at code changes" in rendered
    assert "When delivery is fully delegated." in rendered
    assert "Contexts where the preference is High" not in rendered
    assert "Aggregated preference" not in rendered


def test_all_confidence_bands_uses_requested_thresholds_and_headings():
    profile = {"user": "Soph", "axes": [
        {"axis_id": "test_execution", "axis_name": "Test Execution", "direction": "high",
         "confidence": 0.90, "supporting_sessions": 3},
        {"axis_id": "agent_autonomy", "axis_name": "Agent Autonomy", "direction": "low",
         "confidence": 0.75, "supporting_sessions": 2},
        {"axis_id": "delivery_phasing", "axis_name": "Delivery Phasing", "direction": "high",
         "confidence": 0.60, "supporting_sessions": 1},
    ]}
    rendered = render_all_confidence_bands(profile)
    assert "## Follow these consistently:" in rendered
    assert "## Follow these by default, unless the task calls for otherwise:" in rendered
    assert "## These were seen only once or twice — weigh them, do not treat them as rules:" in rendered
    assert "Run the test suite continuously" in rendered
    assert "Pause to ask clarifying questions" in rendered
    assert "Fix critical items first" in rendered
