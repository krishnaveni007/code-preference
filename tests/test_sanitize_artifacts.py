from scripts.sanitize_artifacts import sanitize_file


def test_sanitize_file_redacts_credentials_without_touching_other_text(tmp_path):
    artifact = tmp_path / "result.json"
    artifact.write_text(
        '{"OPENAI_API_KEY":"sk-proj-' + "a" * 40
        + '","GEMINI_API_KEY":"AIza' + "b" * 31
        + '","status":"complete"}'
    )

    assert sanitize_file(artifact) == 2
    assert artifact.read_text() == (
        '{"OPENAI_API_KEY":"[REDACTED_OPENAI_KEY]",'
        '"GEMINI_API_KEY":"[REDACTED_GOOGLE_KEY]",'
        '"status":"complete"}'
    )
