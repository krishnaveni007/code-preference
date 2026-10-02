from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, Sequence

from src.pilot_study.common import write_json
from .models import ReplayInstance, ReplayValidationError


PREFERENCE_START = "<!-- BEGIN LEARNED USER PREFERENCES -->"
PREFERENCE_END = "<!-- END LEARNED USER PREFERENCES -->"
CONDITIONS = ("baseline", "preference")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def tree_manifest(root: Path, *, include_git: bool = True) -> dict[str, str]:
    """Return a deterministic map of path to type/mode/content digest."""
    root = root.resolve()
    entries: dict[str, str] = {}
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        base = Path(directory)
        if not include_git:
            dirnames[:] = [name for name in dirnames if name != ".git"]
        for name in sorted(dirnames):
            path = base / name
            rel = path.relative_to(root).as_posix()
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode):
                entries[rel] = f"link:{stat.S_IMODE(info.st_mode):o}:{os.readlink(path)}"
            else:
                entries[rel + "/"] = f"dir:{stat.S_IMODE(info.st_mode):o}"
        for name in sorted(filenames):
            path = base / name
            rel = path.relative_to(root).as_posix()
            if not include_git and (rel == ".git" or rel.startswith(".git/")):
                continue
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode):
                entries[rel] = f"link:{stat.S_IMODE(info.st_mode):o}:{os.readlink(path)}"
                continue
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            entries[rel] = f"file:{stat.S_IMODE(info.st_mode):o}:{digest.hexdigest()}"
    return dict(sorted(entries.items()))


def manifest_hash(manifest: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for path, value in manifest.items():
        digest.update(path.encode())
        digest.update(b"\0")
        digest.update(value.encode())
        digest.update(b"\0")
    return digest.hexdigest()


def tree_hash(root: Path, *, include_git: bool = True) -> str:
    return manifest_hash(tree_manifest(root, include_git=include_git))


def changed_paths(before: dict[str, str], after: dict[str, str]) -> list[str]:
    return sorted(
        path.rstrip("/")
        for path in before.keys() | after.keys()
        if before.get(path) != after.get(path) and not path.endswith("/")
    )


def resolve_codex_binary(binary: str) -> str:
    resolved = shutil.which(binary)
    if resolved is None:
        bundled = Path("/Applications/ChatGPT.app/Contents/Resources/codex")
        resolved = str(bundled) if bundled.is_file() else None
    if not resolved:
        raise RuntimeError(f"Codex CLI was not found: {binary}")
    return resolved


def codex_version(binary: str) -> str | None:
    try:
        proc = subprocess.run(
            [binary, "--version"], text=True, capture_output=True, timeout=15
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return (proc.stdout or proc.stderr).strip() or None


def compose_agents(original: str | None, preference: str) -> str:
    # Keep every original byte as a prefix. Only add the minimum separator
    # needed before the intervention block.
    prefix = original or ""
    if prefix and not prefix.endswith("\n"):
        prefix += "\n"
    if prefix and not prefix.endswith("\n\n"):
        prefix += "\n"
    return (
        prefix
        + PREFERENCE_START
        + "\n\n"
        + preference.strip()
        + "\n\n"
        + PREFERENCE_END
        + "\n"
    )


def serialize_context(instance: ReplayInstance) -> tuple[dict[str, Any], str]:
    supplied = {
        "serialization": "role-labelled-json-v1",
        "history": [
            {"role": item.role, "content": item.content}
            for item in instance.conversation.history
        ],
        "current_user_message": instance.conversation.current_user_message,
    }
    history_json = json.dumps(supplied["history"], ensure_ascii=False, indent=2)
    prompt = (
        "Continue the coding task represented by the conversation below. Earlier messages are "
        "context only; act exactly once on CURRENT USER MESSAGE. Preserve their role ordering. "
        "Repository instructions such as AGENTS.md apply normally.\n\n"
        "PRIOR CONVERSATION (JSON, in chronological order):\n"
        f"{history_json}\n\n"
        "CURRENT USER MESSAGE:\n"
        f"{instance.conversation.current_user_message}"
    )
    supplied["rendered_prompt"] = prompt
    return supplied, prompt


@dataclass(frozen=True)
class ReplayConfig:
    work_root: Path = Path("work")
    results_root: Path = Path("results")
    conditions: tuple[str, ...] = CONDITIONS
    preference_file: Path | None = None
    codex_binary: str = "codex"
    model: str | None = None
    reasoning_effort: str | None = None
    timeout: int = 3600
    overwrite: bool = False
    run_setup: bool = True


@dataclass
class ExecutionOutcome:
    command: list[str]
    returncode: int | None
    stdout: str
    stderr: str
    final_response: str
    timed_out: bool = False
    error: str | None = None


class Executor(Protocol):
    def execute(
        self,
        *,
        command: Sequence[str],
        prompt: str,
        cwd: Path,
        timeout: int,
        final_response_path: Path,
    ) -> ExecutionOutcome: ...


class SubprocessExecutor:
    def execute(
        self,
        *,
        command: Sequence[str],
        prompt: str,
        cwd: Path,
        timeout: int,
        final_response_path: Path,
    ) -> ExecutionOutcome:
        command_list = list(command)
        try:
            proc = subprocess.run(
                command_list,
                input=prompt,
                cwd=cwd,
                text=True,
                capture_output=True,
                timeout=timeout,
            )
            response = (
                final_response_path.read_text(errors="replace")
                if final_response_path.exists()
                else ""
            )
            return ExecutionOutcome(
                command=command_list,
                returncode=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
                final_response=response,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode(errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode(errors="replace")
            response = (
                final_response_path.read_text(errors="replace")
                if final_response_path.exists()
                else ""
            )
            return ExecutionOutcome(
                command=command_list,
                returncode=None,
                stdout=stdout,
                stderr=stderr,
                final_response=response,
                timed_out=True,
                error=f"Codex timed out after {timeout} seconds",
            )
        except OSError as exc:
            return ExecutionOutcome(
                command=command_list,
                returncode=None,
                stdout="",
                stderr="",
                final_response="",
                error=f"could not launch Codex: {exc}",
            )


@dataclass
class PreparedCondition:
    condition: str
    worktree: Path
    output_dir: Path
    copied_snapshot_hash: str
    original_agents: str | None


@dataclass
class RunResult:
    instance_id: str
    condition: str
    exit_status: str
    returncode: int | None
    result_path: Path
    errors: list[str] = field(default_factory=list)


def _safe_replace_directory(path: Path, *, overwrite: bool) -> None:
    if not path.exists():
        return
    if not overwrite:
        raise FileExistsError(f"output already exists (use --overwrite): {path}")
    resolved = path.resolve()
    if resolved == Path(resolved.anchor) or len(resolved.parts) < 3:
        raise RuntimeError(f"refusing to replace unsafe path: {resolved}")
    shutil.rmtree(resolved)


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args], text=True, capture_output=True
    )


def git_commit(repo: Path) -> str | None:
    proc = _git(repo, "rev-parse", "HEAD")
    return proc.stdout.strip() if proc.returncode == 0 else None


def collect_git_artifacts(
    repo: Path, starting_commit: str | None, source_repo: Path
) -> tuple[str, str]:
    status_proc = _git(repo, "status", "--short", "--untracked-files=all")
    status = status_proc.stdout if status_proc.returncode == 0 else status_proc.stderr
    if status_proc.returncode != 0:
        directory_diff = subprocess.run(
            ["git", "diff", "--no-index", "--binary", "--", str(source_repo), str(repo)],
            text=True,
            capture_output=True,
        )
        patch = (
            directory_diff.stdout
            if directory_diff.returncode in {0, 1}
            else directory_diff.stderr
        )
        return patch, "not a git repository\n"
    if starting_commit:
        diff_proc = _git(repo, "diff", "--binary", starting_commit, "--")
    else:
        diff_proc = _git(repo, "diff", "--binary", "--")
    patch = diff_proc.stdout if diff_proc.returncode == 0 else diff_proc.stderr
    if status_proc.returncode == 0:
        for line in status.splitlines():
            if not line.startswith("?? "):
                continue
            relative = line[3:]
            extra = subprocess.run(
                ["git", "diff", "--no-index", "--binary", "--", "/dev/null", relative],
                cwd=repo,
                text=True,
                capture_output=True,
            )
            if extra.returncode in {0, 1}:
                patch += extra.stdout
    return patch, status


def _setup_command(instance: ReplayInstance) -> str | None:
    environment = instance.metadata.get("environment") or {}
    if not isinstance(environment, dict):
        return None
    command = environment.get("setup_command")
    return command if isinstance(command, str) and command.strip() else None


def _run_setup(command: str | None, cwd: Path, timeout: int) -> ExecutionOutcome | None:
    if not command:
        return None
    try:
        proc = subprocess.run(
            command,
            cwd=cwd,
            shell=True,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        return ExecutionOutcome(
            command=[command],
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            final_response="",
        )
    except subprocess.TimeoutExpired as exc:
        return ExecutionOutcome(
            command=[command],
            returncode=None,
            stdout=exc.stdout or "",
            stderr=exc.stderr or "",
            final_response="",
            timed_out=True,
            error=f"setup command timed out after {timeout} seconds",
        )


def select_preference(instance: ReplayInstance, explicit: Path | None) -> Path | None:
    """Explicit --preference-file takes precedence over instance/preference.md."""
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_file():
            raise ReplayValidationError(f"preference file does not exist: {path}")
        return path
    local = instance.root / "preference.md"
    return local if local.is_file() else None


def build_codex_command(
    config: ReplayConfig,
    binary: str,
    cwd: Path,
    final_response_path: Path,
) -> list[str]:
    command = [
        binary,
        "exec",
        "--json",
        "--ephemeral",
        "--ignore-user-config",
        "--ignore-rules",
        "--sandbox",
        "workspace-write",
        "--skip-git-repo-check",
        "--cd",
        str(cwd),
        "--output-last-message",
        str(final_response_path),
    ]
    if config.model:
        command += ["--model", config.model]
    if config.reasoning_effort:
        command += ["--config", f'model_reasoning_effort="{config.reasoning_effort}"']
    command.append("-")
    return command


def dry_run_plan(instance: ReplayInstance, config: ReplayConfig) -> dict[str, Any]:
    invalid = set(config.conditions) - set(CONDITIONS)
    if invalid:
        raise ReplayValidationError(f"unknown condition(s): {sorted(invalid)}")
    snapshot_hash = tree_hash(instance.repo)
    binary = resolve_codex_binary(config.codex_binary)
    preference = (
        select_preference(instance, config.preference_file)
        if "preference" in config.conditions
        else None
    )
    agents = instance.repo / "AGENTS.md"
    supplied, _ = serialize_context(instance)
    commands: dict[str, list[str]] = {}
    worktrees: dict[str, str] = {}
    outputs: dict[str, str] = {}
    for condition in config.conditions:
        worktree = (config.work_root / instance.instance_id / condition).resolve()
        output = (config.results_root / instance.instance_id / condition).resolve()
        cwd = worktree / str(
            (instance.metadata.get("environment") or {}).get("working_directory", ".")
        )
        commands[condition] = build_codex_command(
            config, binary, cwd, output / "final_response.txt"
        )
        worktrees[condition] = str(worktree)
        outputs[condition] = str(output)
    return {
        "dry_run": True,
        "instance_id": instance.instance_id,
        "repo_snapshot_path": str(instance.repo),
        "snapshot_hash": snapshot_hash,
        "conversation_history_length": len(instance.conversation.history),
        "current_user_message": instance.conversation.current_user_message,
        "context_serialization": supplied["serialization"],
        "agent": "codex",
        "model": config.model or "authenticated CLI default",
        "timeout_seconds": config.timeout,
        "original_agents_md": "present" if agents.is_file() else "absent",
        "preference_file_selected": str(preference) if preference else None,
        "preference_precedence": "explicit CLI path, then instance/preference.md",
        "agents_md_composition": (
            "append delimited learned preferences after original AGENTS.md"
            if preference
            else "no learned preference injection"
        ),
        "working_directories": worktrees,
        "output_directories": outputs,
        "codex_commands": commands,
    }


def _prepare_conditions(
    instance: ReplayInstance, config: ReplayConfig, snapshot_hash: str
) -> dict[str, PreparedCondition]:
    prepared: dict[str, PreparedCondition] = {}
    for condition in config.conditions:
        worktree = (config.work_root / instance.instance_id / condition).resolve()
        output = (config.results_root / instance.instance_id / condition).resolve()
        _safe_replace_directory(worktree, overwrite=config.overwrite)
        _safe_replace_directory(output, overwrite=config.overwrite)
        worktree.parent.mkdir(parents=True, exist_ok=True)
        output.mkdir(parents=True, exist_ok=True)
        shutil.copytree(instance.repo, worktree, symlinks=True)
        copied_hash = tree_hash(worktree)
        if copied_hash != snapshot_hash:
            raise RuntimeError(
                f"copied {condition} snapshot hash differs from source: "
                f"{copied_hash} != {snapshot_hash}"
            )
        agents_path = worktree / "AGENTS.md"
        original_agents = agents_path.read_text() if agents_path.is_file() else None
        prepared[condition] = PreparedCondition(
            condition=condition,
            worktree=worktree,
            output_dir=output,
            copied_snapshot_hash=copied_hash,
            original_agents=original_agents,
        )
    return prepared


def _execute_condition(
    instance: ReplayInstance,
    config: ReplayConfig,
    prepared: PreparedCondition,
    *,
    snapshot_hash: str,
    starting_commit: str | None,
    binary: str,
    version: str | None,
    executor: Executor,
) -> RunResult:
    condition = prepared.condition
    output = prepared.output_dir
    errors: list[str] = []
    preference_path: Path | None = None
    preference_hash: str | None = None
    preference_text: str | None = None
    working_cwd = instance.in_repo_working_directory(prepared.worktree)

    setup = _run_setup(
        _setup_command(instance) if config.run_setup else None,
        working_cwd,
        config.timeout,
    )
    if setup:
        (output / "setup_stdout.log").write_text(setup.stdout)
        (output / "setup_stderr.log").write_text(setup.stderr)
        if setup.error:
            errors.append(setup.error)
        if setup.returncode not in {0}:
            errors.append(f"setup command exited with {setup.returncode}")

    agents_path = prepared.worktree / "AGENTS.md"
    if condition == "preference":
        preference_path = select_preference(instance, config.preference_file)
        if preference_path is None:
            errors.append("preference condition requires a preference file")
        else:
            preference_text = preference_path.read_text()
            preference_hash = sha256_text(preference_text)
            final_agents = compose_agents(prepared.original_agents, preference_text)
            agents_path.write_text(final_agents)
            (output / "injected_AGENTS.md").write_text(final_agents)
    final_agents = agents_path.read_text() if agents_path.is_file() else None
    if prepared.original_agents is not None:
        (output / "original_AGENTS.md").write_text(prepared.original_agents)

    supplied, prompt = serialize_context(instance)
    write_json(output / "supplied_context.json", supplied)
    agent_start_manifest = tree_manifest(prepared.worktree, include_git=False)
    agent_start_hash = manifest_hash(agent_start_manifest)
    started_at = utc_now()
    final_response_path = output / "final_response.txt"
    command = build_codex_command(config, binary, working_cwd, final_response_path)

    if errors:
        outcome = ExecutionOutcome(
            command=command,
            returncode=None,
            stdout="",
            stderr="",
            final_response="",
            error="; ".join(errors),
        )
    else:
        outcome = executor.execute(
            command=command,
            prompt=prompt,
            cwd=working_cwd,
            timeout=config.timeout,
            final_response_path=final_response_path,
        )
        if outcome.error:
            errors.append(outcome.error)
        if outcome.returncode not in {0}:
            errors.append(f"Codex exited with {outcome.returncode}")
    ended_at = utc_now()

    (output / "stdout.log").write_text(outcome.stdout)
    (output / "stderr.log").write_text(outcome.stderr)
    (output / "transcript.jsonl").write_text(outcome.stdout)
    final_response_path.write_text(outcome.final_response)
    patch, git_status = collect_git_artifacts(
        prepared.worktree, starting_commit, instance.repo
    )
    (output / "patch.diff").write_text(patch)
    (output / "git_status.txt").write_text(git_status)
    after_manifest = tree_manifest(prepared.worktree, include_git=False)
    changed = changed_paths(agent_start_manifest, after_manifest)
    source_hash_after = tree_hash(instance.repo)
    if source_hash_after != snapshot_hash:
        errors.append("source repo/ was mutated during replay")

    exit_status = (
        "success"
        if outcome.returncode == 0 and not errors
        else "timed_out" if outcome.timed_out else "failed"
    )
    result_data = {
        "instance_id": instance.instance_id,
        "condition": condition,
        "user_id": instance.metadata.get("user_id"),
        "session_id": instance.metadata.get("session_id"),
        "turn_index": instance.metadata.get("turn_index"),
        "input_metadata": instance.metadata,
        "starting_snapshot_hash": snapshot_hash,
        "copied_snapshot_hash": prepared.copied_snapshot_hash,
        "agent_start_worktree_hash": agent_start_hash,
        "source_snapshot_hash_after_run": source_hash_after,
        "starting_git_commit": starting_commit,
        "model": config.model or "authenticated CLI default",
        "agent": "codex",
        "agent_version": version,
        "execution_configuration": {
            "command": command,
            "working_directory": str(working_cwd),
            "sandbox": "workspace-write",
            "ephemeral": True,
            "ignore_user_config": True,
            "ignore_rules": True,
            "reasoning_effort": config.reasoning_effort,
            "setup_command": _setup_command(instance) if config.run_setup else None,
        },
        "timeout_seconds": config.timeout,
        "start_timestamp": started_at,
        "end_timestamp": ended_at,
        "exit_status": exit_status,
        "returncode": outcome.returncode,
        "timed_out": outcome.timed_out,
        "conversation_context_path": str(output / "supplied_context.json"),
        "current_user_message": instance.conversation.current_user_message,
        "original_agents_md": prepared.original_agents,
        "final_agents_md": final_agents,
        "learned_preferences_injected": condition == "preference" and preference_text is not None,
        "agent_final_response": outcome.final_response,
        "transcript_path": str(output / "transcript.jsonl"),
        "git_diff_path": str(output / "patch.diff"),
        "changed_files": changed,
        "git_status": git_status,
        "preference_file_path": str(preference_path) if preference_path else None,
        "preference_file_hash": preference_hash,
        "errors": errors,
    }
    result_path = output / "result.json"
    write_json(result_path, result_data)
    return RunResult(
        instance_id=instance.instance_id,
        condition=condition,
        exit_status=exit_status,
        returncode=outcome.returncode,
        result_path=result_path,
        errors=errors,
    )


def run_instance(
    instance: ReplayInstance,
    config: ReplayConfig,
    *,
    executor: Executor | None = None,
) -> list[RunResult]:
    """Run one reconstructed turn. This function has no evaluation-data input."""
    if not config.conditions or set(config.conditions) - set(CONDITIONS):
        raise ReplayValidationError(f"conditions must be drawn from {CONDITIONS}")
    source_before = tree_hash(instance.repo)
    starting_commit = git_commit(instance.repo)
    binary = resolve_codex_binary(config.codex_binary)
    version = codex_version(binary)
    prepared = _prepare_conditions(instance, config, source_before)
    # All requested copies exist and have been hash-verified before either run.
    if len({item.copied_snapshot_hash for item in prepared.values()}) != 1:
        raise RuntimeError("condition copies did not begin from one identical snapshot")
    runner = executor or SubprocessExecutor()
    results = [
        _execute_condition(
            instance,
            config,
            prepared[condition],
            snapshot_hash=source_before,
            starting_commit=starting_commit,
            binary=binary,
            version=version,
            executor=runner,
        )
        for condition in config.conditions
    ]
    run_metadata = {
        "instance_id": instance.instance_id,
        "input_metadata": instance.metadata,
        "starting_snapshot_hash": source_before,
        "starting_git_commit": starting_commit,
        "conditions": [asdict(result) | {"result_path": str(result.result_path)} for result in results],
        "source_snapshot_hash_after_all_runs": tree_hash(instance.repo),
    }
    write_json(
        config.results_root.resolve() / instance.instance_id / "run_metadata.json",
        run_metadata,
    )
    return results
