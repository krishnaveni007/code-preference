#!/usr/bin/env python3
"""Render six auditable CLAUDE.md preference-intervention conditions.

This is a render-only utility. It uses the cached user profiles and turn-level
LLM judgments produced by extract_four_user_preferences.py and makes no API
calls.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.extraction.preference_judge import RUBRICS


DEFAULT_SOURCE = ROOT / "outputs/longitudinal_pilot/four_user_extraction"
DEFAULT_OUTPUT = ROOT / "outputs/longitudinal_pilot/six_condition_interventions"
RUBRIC_BY_ID = {rubric["id"]: rubric for rubric in RUBRICS}
USERS = ("Soph", "khaong", "Nagi-ovo")
TRIALS_ROOT = ROOT / "third_party/SWE-Together/trials/longitudinal_pilot"
BASELINE_RUNS = {
    # Soph and khaong use the same repository. The Codex c425e4 capture
    # preserves the complete root instruction file in a world_state record.
    "Soph": "c425e4/codex_baseline_r2",
    "khaong": "c425e4/codex_baseline_r2",
    # The Voyager Codex capture includes the repository CLAUDE.md verbatim in
    # the turn-zero Repository Configuration Files block.
    "Nagi-ovo": "16a5c7/codex_baseline_smoke_r1",
}
GUARDRAIL = (
    "Apply these preferences only when relevant and when they do not conflict "
    "with the task or repository instructions."
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def compact(text: str) -> str:
    return " ".join(str(text or "").split())


def retained(axis: dict[str, Any]) -> bool:
    confidence = float(axis.get("confidence") or 0)
    return bool(axis.get("direction")) and 0.50 <= confidence < 1.0


def confidence_band(axis: dict[str, Any]) -> str | None:
    confidence = float(axis.get("confidence") or 0)
    if not retained(axis):
        return None
    if confidence >= 0.85:
        return "consistent"
    if confidence >= 0.70:
        return "default"
    return "tentative"


def directional_instruction(axis: dict[str, Any]) -> str:
    rubric = RUBRIC_BY_ID[axis["axis_id"]]
    return compact(rubric[axis["direction"]])


def heading() -> list[str]:
    return ["# User working preferences", "", GUARDRAIL, ""]


def render_band_section(title: str, axes: list[dict[str, Any]]) -> list[str]:
    lines = [f"## {title}", ""]
    if not axes:
        lines.extend(["No preferences met this confidence band.", ""])
        return lines
    for axis in axes:
        lines.append(f"- {directional_instruction(axis)}")
    lines.append("")
    return lines


def render_high_only(profile: dict[str, Any]) -> str:
    axes = [a for a in profile["axes"] if confidence_band(a) == "consistent"]
    axes.sort(key=lambda a: (-a["confidence"], a["axis_id"]))
    lines = heading()
    lines += render_band_section("Follow these consistently:", axes)
    return "\n".join(lines).rstrip() + "\n"


def render_all_bands(profile: dict[str, Any]) -> str:
    axes = [a for a in profile["axes"] if retained(a)]
    axes.sort(key=lambda a: (-a["confidence"], a["axis_id"]))
    grouped = defaultdict(list)
    for axis in axes:
        grouped[confidence_band(axis)].append(axis)
    lines = heading()
    lines += render_band_section("Follow these consistently:", grouped["consistent"])
    lines += render_band_section(
        "Follow these by default, unless the task calls for otherwise:",
        grouped["default"],
    )
    lines += render_band_section(
        "These were seen only once or twice — weigh them, do not treat them as rules:",
        grouped["tentative"],
    )
    return "\n".join(lines).rstrip() + "\n"


def records_by_axis(records: list[dict[str, Any]], user: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("user") != user:
            continue
        if record.get("method1", {}).get("label") not in {"high", "low"}:
            continue
        grouped[record["axis_id"]].append(record)
    for values in grouped.values():
        values.sort(key=lambda r: (str(r["session_id"]), int(r["turn_number"])))
    return grouped


def representative_records(
    records: list[dict[str, Any]], max_per_axis: int = 3,
) -> list[dict[str, Any]]:
    """Keep both poles when present, then fill up to the cap."""
    selected: list[dict[str, Any]] = []
    for label in ("high", "low"):
        match = next((r for r in records if r["method1"]["label"] == label), None)
        if match:
            selected.append(match)
    for record in records:
        if len(selected) >= max_per_axis:
            break
        if record not in selected:
            selected.append(record)
    return selected


def render_evidence_condition(
    profile: dict[str, Any], records: list[dict[str, Any]], *, include_reasoning: bool,
) -> str:
    grouped = records_by_axis(records, profile["user"])
    axes = [a for a in profile["axes"] if retained(a)]
    axes.sort(key=lambda a: (-a["confidence"], a["axis_id"]))
    lines = heading()
    for axis in axes:
        rubric = RUBRIC_BY_ID[axis["axis_id"]]
        lines.extend([
            f"## {axis['axis_name']}", "",
            f"**Preference rubric:** {compact(rubric['description'])}", "",
        ])
        reps = representative_records(grouped.get(axis["axis_id"], []))
        for label in ("high", "low"):
            selected = [r for r in reps if r["method1"]["label"] == label]
            if not selected:
                continue
            lines.extend([
                f"### Prefers {label.title()}: {compact(rubric[label])}", "",
                (
                    "**Contexts — LLM judge rationales:**"
                    if include_reasoning
                    else "**Contexts — representative user messages:**"
                ),
                "",
            ])
            for record in selected:
                if include_reasoning:
                    rationale = compact(record["method1"].get("rationale"))
                    lines.append(f"- {rationale}")
                else:
                    message = compact(record.get("user_message"))
                    lines.append(f"- “{message}”")
            lines.append("")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_descriptive(profile: dict[str, Any]) -> str:
    """Render every useful cached Method-2 description without polarity."""
    lines = heading()
    for axis in profile["axes"]:
        descriptions = [
            compact(item.get("description"))
            for item in axis.get("descriptions", [])
            if compact(item.get("description"))
        ]
        if not descriptions:
            continue
        lines.append(f"## {axis['axis_name']}")
        lines.append("")
        lines.extend(f"- {description}" for description in descriptions)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def baseline_claude(user: str) -> str:
    """Recover the root CLAUDE.md captured at the held-out task checkpoint."""
    run_dir = TRIALS_ROOT / BASELINE_RUNS[user]
    session_files = sorted(run_dir.glob("*/agent/sessions/**/*.jsonl"))
    for path in session_files:
        for line in path.read_text(errors="ignore").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if user in {"Soph", "khaong"} and record.get("type") == "world_state":
                agents_md = record.get("payload", {}).get("state", {}).get("agents_md")
                if isinstance(agents_md, dict) and agents_md.get("text"):
                    return str(agents_md["text"]).rstrip() + "\n"
            if user == "Nagi-ovo" and record.get("type") == "response_item":
                payload = record.get("payload", {})
                if payload.get("role") != "user":
                    continue
                for item in payload.get("content", []):
                    text = item.get("text", "") if isinstance(item, dict) else ""
                    marker = "### CLAUDE.md\n\n"
                    if marker in text:
                        return text.split(marker, 1)[1].rstrip() + "\n"
    raise RuntimeError(f"Could not recover baseline CLAUDE.md for {user}")


def compose(preference_block: str, baseline: str) -> str:
    if not preference_block:
        return baseline
    return (
        preference_block.rstrip()
        + "\n\n<!-- repository-claude-md:start -->\n\n"
        + baseline.lstrip()
    )


def render_readme() -> str:
    return """# Six preference-intervention conditions

Each user directory contains one subdirectory per experimental condition. Every
`CLAUDE.md` is ready to inject: the baseline contains the repository's captured
root `CLAUDE.md`, and conditions 2–6 prepend learned preferences to that same
baseline.

## Confidence bands

- **Tentative:** `0.50 <= confidence < 0.70`
- **Default:** `0.70 <= confidence < 0.85`
- **Consistent:** `0.85 <= confidence < 1.00`

Confidence is the Laplace-smoothed majority agreement:

```text
(max(high_turns, low_turns) + 1) / (high_turns + low_turns + 2)
```

Tied rubrics have no profile direction and are excluded from directional
conditions. The evidence conditions include at most three representative user
messages per retained rubric and preserve both poles when both were observed.
The reasoning condition uses the cached turn-level LLM judge rationale. The
descriptive condition preserves every useful cached Method-2 preference
description, grouped by active rubric and without High/Low annotation.

## Conditions

1. `01_baseline`: repository `CLAUDE.md` only.
2. `02_high_confidence_only`: only consistent directional preferences.
3. `03_all_confidence_bands`: consistent, default, and tentative directions.
4. `04_direction_plus_user_messages`: rubric description, observed High/Low descriptions, and representative user messages for each observed pole.
5. `05_direction_plus_judge_reasoning`: the same rubric and High/Low structure, replacing user messages with the cached judge rationales.
6. `06_descriptive_no_polarity`: all contextual Method-2 descriptions by active rubric.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    profiles = {p["user"]: p for p in read_json(args.source / "user_profiles.json")}
    records = read_json(args.source / "turn_level_extractions.json")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "README.md").write_text(render_readme())

    for user in USERS:
        profile = profiles[user]
        baseline = baseline_claude(user)
        preference_blocks = {
            "01_baseline": "",
            "02_high_confidence_only": render_high_only(profile),
            "03_all_confidence_bands": render_all_bands(profile),
            "04_direction_plus_user_messages": render_evidence_condition(
                profile, records, include_reasoning=False,
            ),
            "05_direction_plus_judge_reasoning": render_evidence_condition(
                profile, records, include_reasoning=True,
            ),
            "06_descriptive_no_polarity": render_descriptive(profile),
        }
        for condition, preference_block in preference_blocks.items():
            condition_dir = args.output / user / condition
            condition_dir.mkdir(parents=True, exist_ok=True)
            (condition_dir / "CLAUDE.md").write_text(
                compose(preference_block, baseline)
            )

    print(f"Rendered {len(USERS)} users x 6 conditions to {args.output}")


if __name__ == "__main__":
    main()
