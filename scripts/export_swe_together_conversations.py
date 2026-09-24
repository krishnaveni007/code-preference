#!/usr/bin/env python3
"""Export readable, secret-free conversations from SWE-Together artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def assistant_text(stdout: Path) -> str:
    chunks: list[str] = []
    if not stdout.exists():
        return ""
    for line in stdout.read_text(errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") != "assistant":
            continue
        for item in (event.get("message") or {}).get("content") or []:
            if isinstance(item, dict) and item.get("type") == "text" and item.get("text"):
                chunks.append(item["text"].strip())
    return "\n\n".join(chunks)


def trial_markdown(trial: Path, initial: str, title: str) -> str:
    out = [f"# {title}", "", "## User", "", initial.strip()]
    first = trial / "agent" / "command-0-1" / "stdout.txt"
    if text := assistant_text(first):
        out += ["", "## Assistant", "", text]
    command_dirs = sorted(
        trial.joinpath("agent").glob("command-*-0"),
        key=lambda p: int(p.name.split("-")[1]),
    )
    for command_dir in command_dirs:
        turn = int(command_dir.name.split("-")[1])
        if turn == 0:
            continue
        decision_path = trial / "agent" / f"episode-{turn}" / "user_decision.json"
        if decision_path.exists():
            decision = json.loads(decision_path.read_text())
            if decision.get("has_message"):
                out += ["", "## User simulator", "", str(decision.get("content", "")).strip()]
            else:
                out += ["", "## Harness", "", "_Simulator remained silent; harness asked the agent to continue._"]
        if text := assistant_text(command_dir / "stdout.txt"):
            out += ["", "## Assistant", "", text]
    return "\n".join(out).rstrip() + "\n"


def oracle_markdown(path: Path) -> str:
    out = ["# Original SWE-Chat conversation", ""]
    for line in path.read_text(errors="replace").splitlines():
        row = json.loads(line)
        if row.get("_is_header"):
            continue
        if row.get("user_message"):
            out += ["## User", "", str(row["user_message"]).strip(), ""]
        if row.get("agent_text"):
            out += ["## Assistant", "", str(row["agent_text"]).strip(), ""]
    return "\n".join(out).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("baseline", "preference", "oracle", "instruction", "output_dir"):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    initial = args.instruction.read_text()
    (args.output_dir / "baseline_conversation.md").write_text(
        trial_markdown(args.baseline, initial, "Baseline simulated conversation"))
    (args.output_dir / "preference_conversation.md").write_text(
        trial_markdown(args.preference, initial, "Preference-intervention simulated conversation"))
    (args.output_dir / "original_swe_chat_conversation.md").write_text(
        oracle_markdown(args.oracle))


if __name__ == "__main__":
    main()
