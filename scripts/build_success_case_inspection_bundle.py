#!/usr/bin/env python3
"""Build human-readable baseline/intervention inspection reports.

The reports contain only visible user/agent conversation, injected preference
text, ledger metrics, and verifier outputs. Hidden model reasoning, credentials,
and environment configuration are deliberately excluded.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SWE = ROOT / "third_party" / "SWE-Together"
LEDGER = ROOT / "outputs" / "longitudinal_pilot" / "eval_metrics_ledger.json"
OUT = ROOT / "outputs" / "longitudinal_pilot" / "success_case_inspection"


CASES = [
    {
        "slug": "01-soph-c425e4-codex",
        "title": "Soph · cli-task-c425e4 · Codex",
        "why": "Cleanest example: reward rises from 0.55 to 0.70 while user turns fall from 5 to 2 and corrections fall from 2 to 1.",
        "trials": [
            ("Baseline", "third_party/SWE-Together/trials/longitudinal_pilot/c425e4/codex_baseline_r2/cli-task-c425e4__wdGPcJE"),
            ("Compact directional profile", "third_party/SWE-Together/trials/longitudinal_pilot/c425e4/codex_directional_r1/cli-task-c425e4__a7RnHFH"),
            ("Detailed natural-language profile", "third_party/SWE-Together/trials/longitudinal_pilot/c425e4/codex_detailed_r1/cli-task-c425e4__GCy4jtq"),
        ],
    },
    {
        "slug": "02-khaong-70c88c-opencode",
        "title": "khaong · cli-task-70c88c · OpenCode",
        "why": "Correctness stays at the ceiling (1.00) while user turns fall from 3 to 2 and corrections fall from 1 to 0.",
        "trials": [
            ("Baseline", "third_party/SWE-Together/trials/longitudinal_profiles/cli-task-70c88c/01_baseline/cli-task-70c88c__gURumNn"),
            ("Direction plus judge reasoning", "third_party/SWE-Together/trials/longitudinal_profiles/cli-task-70c88c/05_direction_plus_judge_reasoning/cli-task-70c88c__kvh9KG3"),
        ],
    },
    {
        "slug": "03-nagi-aa88f5-opencode",
        "title": "Nagi-ovo · gemini-voyager-task-aa88f5 · OpenCode",
        "why": "Reward rises from 0.35 to 0.70 and steering turns fall from 3 to 2; baseline corrections were already zero.",
        "trials": [
            ("Baseline", "third_party/SWE-Together/trials/longitudinal_profiles/gemini-voyager-task-aa88f5/01_baseline/gemini-voyager-task-aa88f5__x6pZBus"),
            ("Direction plus judge reasoning", "third_party/SWE-Together/trials/longitudinal_profiles/gemini-voyager-task-aa88f5/05_direction_plus_judge_reasoning/gemini-voyager-task-aa88f5__UgLick4"),
            ("Descriptive, no polarity", "third_party/SWE-Together/trials/longitudinal_profiles/gemini-voyager-task-aa88f5/06_descriptive_no_polarity/gemini-voyager-task-aa88f5__ERRvUbe"),
            ("Personalized OpenCode skill", "third_party/SWE-Together/trials/personalized_skills/gemini-voyager-task-aa88f5/02_personalized_skill/gemini-voyager-task-aa88f5__pRQz8s7"),
        ],
    },
    {
        "slug": "04-soph-4a9dde-opencode",
        "title": "Soph · cli-task-4a9dde · OpenCode",
        "why": "Reward improves from 0.00 to 0.10–0.30 and corrections fall from 6 to 4–5, although all runs reach 15 user turns.",
        "trials": [
            ("Baseline", "third_party/SWE-Together/trials/longitudinal_profiles/cli-task-4a9dde/01_baseline/cli-task-4a9dde__KDEynyc"),
            ("Updated high-confidence", "third_party/SWE-Together/trials/updated_rubric_profiles/cli-task-4a9dde/02_high_confidence_only/cli-task-4a9dde__suQNZBG"),
            ("Updated all-confidence", "third_party/SWE-Together/trials/updated_rubric_profiles/cli-task-4a9dde/03_all_confidence_bands/cli-task-4a9dde__WVHYd7c"),
        ],
    },
    {
        "slug": "05-nagi-16a5c7-codex",
        "title": "Nagi-ovo · gemini-voyager-task-16a5c7 · Codex",
        "why": "Detailed profile raises reward from 0.85 to 1.00 without increasing turns, corrections, or nudges.",
        "trials": [
            ("Baseline", "third_party/SWE-Together/trials/longitudinal_pilot/16a5c7/codex_baseline_smoke_r1/gemini-voyager-task-16a5c7__UK7GEPo"),
            ("Detailed natural-language profile", "third_party/SWE-Together/trials/longitudinal_pilot/16a5c7/codex_detailed_r1/gemini-voyager-task-16a5c7__epBEkvp"),
        ],
    },
    {
        "slug": "06-khaong-70c88c-claude-code",
        "title": "khaong · cli-task-70c88c · Claude Code",
        "why": "Compact profile raises reward from 0.00 to 1.00 without increasing turns, corrections, or nudges.",
        "trials": [
            ("Baseline", "third_party/SWE-Together/trials/longitudinal_pilot/70c88c/baseline_r5/cli-task-70c88c__kJAr2re"),
            ("Compact directional profile", "third_party/SWE-Together/trials/longitudinal_pilot/70c88c/directional_r5/cli-task-70c88c__Cb5Nqxm"),
        ],
    },
]


def json_events(path: Path) -> list[dict]:
    events = []
    if not path.exists():
        return events
    for line in path.read_text(errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def opencode_text(path: Path) -> str:
    chunks = []
    for event in json_events(path):
        if event.get("type") != "text":
            continue
        value = str((event.get("part") or {}).get("text") or "").strip()
        if value:
            chunks.append(value)
    return "\n\n".join(chunks)


def codex_turns(path: Path) -> list[str]:
    turns: list[list[str]] = []
    current: list[str] | None = None
    for event in json_events(path):
        if event.get("type") == "thread.started":
            if current is not None:
                turns.append(current)
            current = []
        elif current is not None and event.get("type") == "item.completed":
            item = event.get("item") or {}
            if item.get("type") == "agent_message" and str(item.get("text") or "").strip():
                current.append(str(item["text"]).strip())
    if current is not None:
        turns.append(current)
    return ["\n\n".join(parts) or "[No visible text response]" for parts in turns]


def claude_turns(path: Path) -> list[str]:
    order: list[str] = []
    chunks: dict[str, list[str]] = {}
    seen: set[str] = set()
    for event in json_events(path):
        if event.get("type") != "assistant":
            continue
        message = event.get("message") or {}
        message_id = str(message.get("id") or "")
        if not message_id or message_id in seen:
            continue
        seen.add(message_id)
        session_id = str(event.get("session_id") or "unknown")
        if session_id not in chunks:
            order.append(session_id)
            chunks[session_id] = []
        for block in message.get("content") or []:
            if block.get("type") == "text" and str(block.get("text") or "").strip():
                chunks[session_id].append(str(block["text"]).strip())
    return ["\n\n".join(chunks[key]) or "[No visible text response]" for key in order]


def infer_task_id(trial: Path) -> str:
    result = json.loads((trial / "result.json").read_text())
    return str(result.get("task_name") or result.get("task_id") or trial.name.split("__", 1)[0])


def transcript(trial: Path, task_id: str) -> list[tuple[str, str, str]]:
    instruction = SWE / "tasks" / task_id / "instruction.md"
    users: list[tuple[str, str]] = [("initial task", instruction.read_text(errors="replace").strip())]
    episodes = sorted((trial / "agent").glob("episode-*"), key=lambda p: int(p.name.split("-")[-1]))
    for episode in episodes:
        try:
            decision = json.loads((episode / "user_decision.json").read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        if decision.get("has_message") and str(decision.get("content") or "").strip():
            users.append((str(decision.get("action") or "simulated user"), str(decision["content"]).strip()))

    if (trial / "agent" / "codex.txt").exists():
        agents = codex_turns(trial / "agent" / "codex.txt")
    elif (trial / "agent" / "claude-code.txt").exists():
        agents = claude_turns(trial / "agent" / "claude-code.txt")
    else:
        paths = sorted((trial / "agent").glob("opencode.txt.turn-*"), key=lambda p: int(p.name.rsplit("-", 1)[-1]))
        agents = [opencode_text(path) or "[No visible text response]" for path in paths]

    output: list[tuple[str, str, str]] = []
    for idx in range(max(len(users), len(agents))):
        if idx < len(users):
            output.append(("user", users[idx][0], users[idx][1]))
        if idx < len(agents):
            output.append(("agent", "visible response", agents[idx]))
    return output


def manifest(task_id: str) -> dict[str, dict]:
    path = SWE / "tasks" / task_id / "tests" / "test_manifest.yaml"
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(errors="replace")) or {}
    return {
        str(gate.get("id")): gate
        for gate in data.get("gates", [])
        if isinstance(gate, dict) and gate.get("id")
    }


def verifier_gates(trial: Path, task_id: str) -> list[dict]:
    path = trial / "verifier" / "gates.json"
    if not path.exists():
        return []
    raw = path.read_text(errors="replace").strip()
    records: list[dict] = []
    try:
        value = json.loads(raw)
        if isinstance(value, list):
            records = value
        elif isinstance(value, dict):
            if "id" in value:
                records = [value]
            else:
                records = [{"id": key, "pass": val} for key, val in value.items()]
    except json.JSONDecodeError:
        for line in raw.splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                records.append(value)
    details = manifest(task_id)
    normalized = []
    for record in records:
        gate_id = str(record.get("id") or "")
        verdict = record.get("verdict")
        passed = record.get("pass")
        if verdict is not None:
            passed = str(verdict).lower() == "pass"
        elif isinstance(passed, str):
            passed = passed.lower() == "true"
        item = dict(details.get(gate_id) or {})
        item.update(record)
        item["id"] = gate_id
        item["passed"] = bool(passed)
        normalized.append(item)
    return normalized


def fence(text: str) -> str:
    marker = "````" if "```" in text else "```"
    return f"{marker}text\n{text.rstrip()}\n{marker}"


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def build() -> None:
    ledger_rows = json.loads(LEDGER.read_text())["rows"]
    by_path = {str(Path(row["trial_path"]).resolve()): row for row in ledger_rows}
    OUT.mkdir(parents=True, exist_ok=True)
    index = [
        "# Preference-intervention success cases: inspection bundle",
        "",
        "These reports reproduce visible conversation and verifier evidence for selected baseline/intervention comparisons. Hidden chain-of-thought and raw tool chatter are excluded. Failed infrastructure attempts are not used as comparison baselines.",
        "",
        "| Case | Why inspect it |",
        "|---|---|",
    ]

    for case in CASES:
        report = [f"# {case['title']}", "", case["why"], "", "## Comparison", ""]
        report += ["| Run | Reward | User turns | Corrections | Nudges | Intent coverage | SWE-Chat |", "|---|---:|---:|---:|---:|---:|---:|"]
        run_data = []
        for name, rel_path in case["trials"]:
            trial = ROOT / rel_path
            row = by_path.get(str(trial.resolve()))
            if row is None:
                raise RuntimeError(f"Ledger row not found for {trial}")
            run_data.append((name, trial, row))
            report.append(
                f"| {name} | {row.get('verifier_reward')} | {row.get('total_user_turns')} | "
                f"{row.get('corrections')} | {row.get('nudges')} | {row.get('intent_coverage')} | "
                f"{row.get('swe_chat_success_score')} |"
            )

        for name, trial, row in run_data:
            task_id = row["task_id"]
            report += ["", f"## {name}", "", f"Trial: `{relative(trial)}`", ""]
            profile = trial / "agent" / "injected_preference.md"
            skill = trial / "agent" / "injected_skill.md"
            profile_path = profile if profile.exists() else skill if skill.exists() else None
            if profile_path:
                report += ["<details>", "<summary>Injected preference context</summary>", "", fence(profile_path.read_text(errors="replace")), "", "</details>", ""]
            else:
                report += ["_No preference context was injected._", ""]

            report += ["### Verifier gates", "", "| Result | Gate | Weight | Description |", "|---|---|---:|---|"]
            gates = verifier_gates(trial, task_id)
            if gates:
                for gate in gates:
                    result = "PASS" if gate["passed"] else "FAIL"
                    description = str(gate.get("description") or "").replace("|", "\\|")
                    report.append(f"| {result} | `{gate['id']}` | {gate.get('weight', '')} | {description} |")
            else:
                report.append("| — | No gate artifact | | |")

            stdout = trial / "verifier" / "test-stdout.txt"
            if stdout.exists() and stdout.read_text(errors="replace").strip():
                report += ["", "<details>", "<summary>Raw verifier output</summary>", "", fence(stdout.read_text(errors="replace")), "", "</details>"]

            report += ["", "### Exact visible conversation", ""]
            for idx, (role, action, content) in enumerate(transcript(trial, task_id), start=1):
                heading = "User" if role == "user" else "Agent"
                report += [f"#### {heading} · {action}", "", fence(content), ""]

        report_path = OUT / f"{case['slug']}.md"
        report_path.write_text("\n".join(report).rstrip() + "\n")
        index.append(f"| [{case['title']}]({report_path.name}) | {case['why']} |")

    (OUT / "README.md").write_text("\n".join(index).rstrip() + "\n")
    print(OUT / "README.md")


if __name__ == "__main__":
    build()
