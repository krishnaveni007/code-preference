import pandas as pd

from src.pilot_study.generate_personalized_skills import (
    FRONT_MATTER,
    STAGE1_INSTRUCTIONS,
    enforce_fixed_front_matter,
    serialize_sessions,
    validate_skill,
)


def test_stage1_has_fixed_front_matter_requirement():
    assert f"exactly this YAML front matter:\n\n{FRONT_MATTER}" in STAGE1_INSTRUCTIONS


def test_session_serialization_uses_only_conversational_user_messages():
    frame = pd.DataFrame([
        {"session_id": "a", "turn_number": 1, "role": "user", "is_conversational": True, "content": "first"},
        {"session_id": "a", "turn_number": 2, "role": "assistant", "is_conversational": True, "content": "reply"},
        {"session_id": "a", "turn_number": 3, "role": "user", "is_conversational": False, "content": "tool result"},
        {"session_id": "b", "turn_number": 1, "role": "user", "is_conversational": True, "content": "second"},
    ])
    block, count = serialize_sessions(frame, ["a", "b"])
    assert count == 2
    assert "=== Session: a ===" in block and "=== Session: b ===" in block
    assert "first" in block and "second" in block
    assert "reply" not in block and "tool result" not in block


def test_skill_validation_requires_fixed_front_matter_and_core_sections():
    text = FRONT_MATTER + "\n\n# Scope\nx\n# Communication\nx\n# Work Style\nx\n# Follow-Up Handling\nx\n"
    validate_skill(text)


def test_fixed_front_matter_replaces_model_metadata_without_changing_body():
    raw = "---\nname: invented\ndescription: changed\n---\n\n# Scope\nBody\n"
    fixed = enforce_fixed_front_matter(raw)
    assert fixed.startswith(FRONT_MATTER + "\n\n")
    assert fixed.endswith("# Scope\nBody\n")
    assert "name: invented" not in fixed
