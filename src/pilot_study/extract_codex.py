#!/usr/bin/env python3
"""Run the existing turn-wise preference judge through authenticated Codex CLI."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from src.extraction.preference_judge import JUDGE_RESPONSE_SCHEMA, vectorize_sessions

PUSHBACK_TYPES = {
    "correction", "rejection", "failure_report", "pacing_complaint",
    "takeover", "requirement_change",
}


def seed_pushback_cache(source: Path, destination: Path, data_dir: Path,
                        session_ids: list[str]) -> int:
    """Copy successful, in-sample pushback judgments into an isolated cache."""
    if not source.exists():
        raise FileNotFoundError(source)
    conversations = pd.read_parquet(
        data_dir / "conversations.parquet",
        filters=[("session_id", "in", session_ids)],
        columns=["session_id", "turn_number", "role", "is_conversational", "prompt_pushback"],
    )
    prompts = conversations[
        (conversations["role"] == "user")
        & conversations["is_conversational"].fillna(False)
        & conversations["prompt_pushback"].isin(PUSHBACK_TYPES)
    ]
    wanted = {(str(row.session_id), int(row.turn_number)) for row in prompts.itertuples()}
    selected: dict[tuple[str, int], dict] = {}
    for line in source.read_text().splitlines():
        try:
            record = json.loads(line)
            event = record["event"]
            key = (str(event["session_id"]), int(event["target_raw_turn_number"]))
            if key in wanted and not record.get("llm_call_failed"):
                selected[key] = record
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n"
                                   for record in selected.values()))
    return len(selected)


class CodexResponsesAdapter:
    """Small Responses-like adapter accepted by the existing extractor."""
    def __init__(self, binary: str = "codex", model: str = "default", timeout: int = 900):
        resolved = shutil.which(binary)
        if resolved is None:
            bundled = Path("/Applications/ChatGPT.app/Contents/Resources/codex")
            resolved = str(bundled) if bundled.is_file() else None
        if not resolved:
            raise RuntimeError("Codex CLI was not found")
        self.binary, self.model, self.timeout = resolved, model, timeout
        self.responses = self

    def create(self, *, instructions: str, input: str, **_kwargs):
        prompt = f"{instructions}\n\n---\n\n{input}"
        with tempfile.TemporaryDirectory(prefix="pilot-codex-judge-") as directory:
            root = Path(directory); schema = root / "schema.json"; output = root / "response.json"
            schema.write_text(json.dumps(JUDGE_RESPONSE_SCHEMA))
            command = [self.binary, "exec", "--ephemeral", "--sandbox", "read-only",
                       "--ignore-user-config", "--skip-git-repo-check",
                       "--output-schema", str(schema), "--output-last-message", str(output)]
            if self.model != "default": command += ["--model", self.model]
            command.append(prompt)
            proc = subprocess.run(command, cwd=root, text=True, capture_output=True, timeout=self.timeout)
            if proc.returncode:
                raise RuntimeError((proc.stderr or proc.stdout)[-4000:])
            if not output.exists():
                raise RuntimeError("Codex completed without writing the structured output")
            return SimpleNamespace(output_text=output.read_text(), id=None, usage=None)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--selected-sessions", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--model", default="default", help="Codex model; default uses authenticated CLI default")
    p.add_argument("--session-id", help="Restrict to one selected session for a smoke test")
    p.add_argument("--turn-number", type=int)
    p.add_argument("--pushback-only", action="store_true",
                   help="Judge only canonical SWE-Chat pushback classes")
    p.add_argument("--seed-cache", type=Path,
                   help="Seed an empty output cache from matching successful judgments")
    p.add_argument("--timeout", type=int, default=900)
    a = p.parse_args()
    sessions = pd.read_csv(a.selected_sessions).session_id.astype(str).tolist()
    if a.session_id:
        if a.session_id not in sessions: raise SystemExit("--session-id is not in --selected-sessions")
        sessions = [a.session_id]
    if a.turn_number is not None and len(sessions) != 1: raise SystemExit("--turn-number requires --session-id")
    judgments = a.out_dir / "preference_turn_judgments.jsonl"
    if a.seed_cache:
        if not a.pushback_only:
            raise SystemExit("--seed-cache currently requires --pushback-only")
        if judgments.exists():
            raise SystemExit("refusing to overwrite an existing output judgment cache")
        count = seed_pushback_cache(a.seed_cache, judgments, a.data_dir, sessions)
        print(f"seeded {count} matching pushback judgments")
    client = CodexResponsesAdapter(model=a.model, timeout=a.timeout)
    turn, session, user = vectorize_sessions(
        data_dir=a.data_dir, session_ids=sessions, out_dir=a.out_dir,
        model=f"codex-cli/{a.model}", turn_number=a.turn_number, client=client,
        prompt_pushback_types=PUSHBACK_TYPES if a.pushback_only else None,
    )
    print(f"wrote {len(turn)} turn, {len(session)} session, {len(user)} user vectors")

if __name__ == "__main__":
    main()
