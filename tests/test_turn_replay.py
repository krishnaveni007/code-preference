from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from src.turn_replay.models import ReplayInstance, ReplayValidationError
from src.turn_replay.runner import (
    ExecutionOutcome,
    PREFERENCE_END,
    PREFERENCE_START,
    ReplayConfig,
    compose_agents,
    dry_run_plan,
    run_instance,
    serialize_context,
    tree_hash,
)


def make_instance(tmp_path: Path, *, agents: str | None = "# Existing\nKeep tests.\n") -> Path:
    root = tmp_path / "eval_instances" / "u1_s1_t2"
    repo = root / "repo"
    repo.mkdir(parents=True)
    (repo / "app.py").write_text("VALUE = 1\n")
    if agents is not None:
        (repo / "AGENTS.md").write_text(agents)
    (root / "conversation.json").write_text(json.dumps({
        "history": [
            {"role": "user", "content": "Earlier request"},
            {"role": "assistant", "content": "Earlier answer"},
        ],
        "current_user_message": "Make the requested change",
    }))
    (root / "metadata.json").write_text(json.dumps({
        "instance_id": "u1_s1_t2",
        "user_id": "u1",
        "session_id": "s1",
        "turn_index": 2,
        "snapshot_confidence": "validated",
        "original_agent": "claude-code",
        "environment": {"working_directory": ".", "setup_command": None},
        "custom_field": {"preserve": True},
    }))
    (root / "evaluation.json").write_text(json.dumps({
        "next_user_message": "SECRET_FUTURE_MESSAGE"
    }))
    (root / "preference.md").write_text("Prefer small focused changes.\n")
    return root


class RecordingExecutor:
    def __init__(self, *, returncode: int = 0, stderr: str = ""):
        self.calls: list[dict] = []
        self.returncode = returncode
        self.stderr = stderr

    def execute(self, *, command, prompt, cwd, timeout, final_response_path):
        self.calls.append({
            "command": list(command), "prompt": prompt, "cwd": cwd, "timeout": timeout
        })
        (cwd / "agent_created.txt").write_text("created\n")
        response = "completed" if self.returncode == 0 else "partial response"
        return ExecutionOutcome(
            command=list(command),
            returncode=self.returncode,
            stdout='{"type":"turn.completed"}\n',
            stderr=self.stderr,
            final_response=response,
        )


def config(tmp_path: Path, *conditions: str) -> ReplayConfig:
    return ReplayConfig(
        work_root=tmp_path / "work",
        results_root=tmp_path / "results",
        conditions=conditions or ("baseline", "preference"),
        codex_binary=sys.executable,
        timeout=30,
    )


def test_load_valid_instance_and_parse_history(tmp_path):
    instance = ReplayInstance.load(make_instance(tmp_path))
    assert instance.instance_id == "u1_s1_t2"
    assert [message.role for message in instance.conversation.history] == ["user", "assistant"]
    assert instance.conversation.current_user_message == "Make the requested change"


@pytest.mark.parametrize("missing", ["repo", "conversation.json", "metadata.json"])
def test_rejects_missing_required_inputs(tmp_path, missing):
    root = make_instance(tmp_path)
    path = root / missing
    if path.is_dir():
        path.rmdir() if not any(path.iterdir()) else (path / "app.py").unlink()
        if path.exists() and (path / "AGENTS.md").exists():
            (path / "AGENTS.md").unlink()
        if path.exists():
            path.rmdir()
    else:
        path.unlink()
    with pytest.raises(ReplayValidationError, match="missing required"):
        ReplayInstance.load(root)


def test_rejects_missing_current_user_message(tmp_path):
    root = make_instance(tmp_path)
    (root / "conversation.json").write_text(json.dumps({"history": []}))
    with pytest.raises(ReplayValidationError, match="current_user_message"):
        ReplayInstance.load(root)


def test_evaluation_json_is_never_read_and_future_message_not_in_context(tmp_path, monkeypatch):
    root = make_instance(tmp_path)
    original = Path.read_text
    evaluation_reads = []

    def guarded_read(path, *args, **kwargs):
        if path.name == "evaluation.json":
            evaluation_reads.append(path)
            raise AssertionError("evaluation.json was read")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read)
    instance = ReplayInstance.load(root)
    supplied, prompt = serialize_context(instance)
    assert evaluation_reads == []
    assert "SECRET_FUTURE_MESSAGE" not in prompt
    assert "next_user_message" not in json.dumps(supplied)


def test_baseline_does_not_read_or_inject_preference(tmp_path, monkeypatch):
    root = make_instance(tmp_path)
    original = Path.read_text

    def guarded_read(path, *args, **kwargs):
        if path.name == "preference.md":
            raise AssertionError("baseline read preference.md")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read)
    runner = RecordingExecutor()
    result = run_instance(
        ReplayInstance.load(root), config(tmp_path, "baseline"), executor=runner
    )[0]
    data = json.loads(result.result_path.read_text())
    assert data["learned_preferences_injected"] is False
    assert data["preference_file_path"] is None
    assert "LEARNED USER PREFERENCES" not in (tmp_path / "work/u1_s1_t2/baseline/AGENTS.md").read_text()


def test_preference_injection_preserves_existing_agents(tmp_path):
    root = make_instance(tmp_path)
    original = (root / "repo/AGENTS.md").read_text()
    runner = RecordingExecutor()
    result = run_instance(
        ReplayInstance.load(root), config(tmp_path, "preference"), executor=runner
    )[0]
    injected = (tmp_path / "work/u1_s1_t2/preference/AGENTS.md").read_text()
    assert injected.startswith(original.rstrip() + "\n\n")
    assert PREFERENCE_START in injected and PREFERENCE_END in injected
    assert injected.count("Prefer small focused changes.") == 1
    data = json.loads(result.result_path.read_text())
    assert data["original_agents_md"] == original
    assert data["final_agents_md"] == injected


def test_explicit_preference_takes_precedence(tmp_path):
    root = make_instance(tmp_path)
    explicit = tmp_path / "explicit.md"
    explicit.write_text("Use the explicit profile.\n")
    replay_config = config(tmp_path, "preference")
    replay_config = ReplayConfig(**{**replay_config.__dict__, "preference_file": explicit})
    run_instance(ReplayInstance.load(root), replay_config, executor=RecordingExecutor())
    injected = (tmp_path / "work/u1_s1_t2/preference/AGENTS.md").read_text()
    assert "Use the explicit profile." in injected
    assert "Prefer small focused changes." not in injected


def test_source_immutable_copies_identical_and_mutations_isolated(tmp_path):
    root = make_instance(tmp_path)
    before = tree_hash(root / "repo")
    runner = RecordingExecutor()
    results = run_instance(ReplayInstance.load(root), config(tmp_path), executor=runner)
    assert tree_hash(root / "repo") == before
    rows = [json.loads(result.result_path.read_text()) for result in results]
    assert rows[0]["starting_snapshot_hash"] == rows[1]["starting_snapshot_hash"] == before
    assert rows[0]["copied_snapshot_hash"] == rows[1]["copied_snapshot_hash"] == before
    baseline_agents = (tmp_path / "work/u1_s1_t2/baseline/AGENTS.md").read_text()
    preference_agents = (tmp_path / "work/u1_s1_t2/preference/AGENTS.md").read_text()
    assert PREFERENCE_START not in baseline_agents
    assert PREFERENCE_START in preference_agents
    (tmp_path / "work/u1_s1_t2/baseline/only_baseline.txt").write_text("x")
    assert not (tmp_path / "work/u1_s1_t2/preference/only_baseline.txt").exists()


def test_dry_run_never_invokes_codex_or_creates_worktrees(tmp_path):
    root = make_instance(tmp_path)
    plan = dry_run_plan(ReplayInstance.load(root), config(tmp_path))
    assert plan["dry_run"] is True
    assert plan["conversation_history_length"] == 2
    assert plan["current_user_message"] == "Make the requested change"
    assert not (tmp_path / "work").exists()
    assert not (tmp_path / "results").exists()


def test_failed_codex_call_preserves_diagnostics(tmp_path):
    root = make_instance(tmp_path)
    runner = RecordingExecutor(returncode=7, stderr="provider failed")
    result = run_instance(
        ReplayInstance.load(root), config(tmp_path, "baseline"), executor=runner
    )[0]
    output = result.result_path.parent
    assert result.exit_status == "failed"
    assert (output / "stderr.log").read_text() == "provider failed"
    assert (output / "transcript.jsonl").exists()
    assert (output / "patch.diff").exists()
    data = json.loads(result.result_path.read_text())
    assert data["returncode"] == 7
    assert data["errors"]


def test_result_metadata_serialization_preserves_input_metadata(tmp_path):
    root = make_instance(tmp_path)
    result = run_instance(
        ReplayInstance.load(root), config(tmp_path, "baseline"), executor=RecordingExecutor()
    )[0]
    data = json.loads(result.result_path.read_text())
    assert data["input_metadata"]["custom_field"] == {"preserve": True}
    assert data["agent"] == "codex"
    assert data["condition"] == "baseline"
    assert data["exit_status"] == "success"
    assert "agent_created.txt" in data["changed_files"]
    assert (result.result_path.parent / "stdout.log").exists()
    assert (result.result_path.parent / "supplied_context.json").exists()


def test_compose_agents_without_existing_file():
    rendered = compose_agents(None, "Preference")
    assert rendered.startswith(PREFERENCE_START)
    assert rendered.endswith(PREFERENCE_END + "\n")


def test_compose_agents_preserves_original_bytes_as_prefix():
    original = "instructions\n\n\n"
    assert compose_agents(original, "Preference").startswith(original)
