#!/usr/bin/env python3
"""Plan and run the one-replicate OpenCode preference intervention study.

The design has 41 cells: five conditions for Soph and khaong, and four for
Nagi-ovo (whose high-confidence-only profile is identical to baseline). Tasks
are ordered by historical average duration; conditions are deterministically
shuffled within each task to reduce condition-order drift. ``--smoke`` selects
the shortest task's baseline and descriptive-no-polarity conditions.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SWE = ROOT / "third_party" / "SWE-Together"
PROFILES = ROOT / "outputs" / "longitudinal_pilot" / "six_condition_interventions"
DEFAULT_TRIALS = SWE / "trials" / "longitudinal_profiles"
DEFAULT_RESULTS = SWE / "results" / "longitudinal_profiles"
MODEL = "openai/gpt-5.4-2026-03-05"
SEED = 20260922

TASKS = (
    ("Soph", "cli-task-c425e4", 318),
    ("khaong", "cli-task-70c88c", 350),
    ("Nagi-ovo", "gemini-voyager-task-72a86c", 416),
    ("Soph", "cli-task-30159a", 446),
    ("Nagi-ovo", "gemini-voyager-task-aa88f5", 532),
    ("Nagi-ovo", "gemini-voyager-task-4a5730", 623),
    ("Nagi-ovo", "gemini-voyager-task-c5c01d", 702),
    ("khaong", "cli-task-b3d2dd", 835),
    ("Soph", "cli-task-4a9dde", 898),
)

# 04_direction_plus_user_messages is intentionally excluded.
CONDITIONS = (
    "01_baseline",
    "02_high_confidence_only",
    "03_all_confidence_bands",
    "05_direction_plus_judge_reasoning",
    "06_descriptive_no_polarity",
)

# Nagi-ovo has no high-confidence preference, so condition 02 adds no
# intervention and would merely duplicate baseline for all four tasks.
USER_EXCLUDED_CONDITIONS = {
    "Nagi-ovo": {"02_high_confidence_only"},
}

PASSTHROUGH_ENV = {
    "E2B_API_KEY",
    "GHCR_TOKEN",
    "GHCR_USER",
    "HOME",
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "LANG",
    "LC_ALL",
    "NO_PROXY",
    "OPENAI_API_KEY",
    "PATH",
    "REQUESTS_CA_BUNDLE",
    "SSL_CERT_FILE",
    "TMPDIR",
    "USER",
}


def load_env() -> None:
    for path in (ROOT / ".env", SWE / ".env"):
        if not path.exists():
            continue
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def subprocess_env() -> dict[str, str]:
    """Return only the environment required by the evaluation subprocess."""
    return {key: value for key, value in os.environ.items() if key in PASSTHROUGH_ENV}


def build_plan(smoke: bool, trials_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for user, task, duration in TASKS:
        conditions = [
            condition for condition in CONDITIONS
            if condition not in USER_EXCLUDED_CONDITIONS.get(user, set())
        ]
        random.Random(f"{SEED}:{task}").shuffle(conditions)
        for condition in conditions:
            rows.append({
                "user_id": user,
                "task_id": task,
                "condition": condition,
                "replicate": 1,
                "historical_duration_seconds": duration,
                "profile_path": str(PROFILES / user / condition / "CLAUDE.md"),
                "trials_dir": str(trials_root / task / condition),
                "action_harness": "opencode",
                "provider": "openai",
                "model": MODEL,
                "reasoning_effort": "high",
            })
    if smoke:
        wanted = {"01_baseline", "06_descriptive_no_polarity"}
        rows = [
            row for row in rows
            if row["task_id"] == TASKS[0][1] and row["condition"] in wanted
        ]
        rows.sort(key=lambda row: 0 if row["condition"] == "01_baseline" else 1)
    return rows


def validate_plan(plan: list[dict[str, object]]) -> None:
    missing: list[str] = []
    for row in plan:
        for path in (Path(str(row["profile_path"])), SWE / "tasks" / str(row["task_id"])):
            if not path.exists():
                missing.append(str(path))
    if missing:
        raise SystemExit("Missing plan inputs:\n" + "\n".join(sorted(set(missing))))


def run_command(row: dict[str, object], execute: bool) -> int:
    command = [
        str(SWE / ".venv" / "bin" / "python"),
        str(SWE / "src" / "run_eval.py"),
        "--model", MODEL,
        "--user-model", MODEL,
        "--tag", f"{row.get('tag_prefix', 'profiles')}-{row['task_id']}-{row['condition']}",
        "--agent-type", "opencode",
        "--reasoning-effort", "high",
        "--env-type", "e2b",
        "--workers", "1",
        # E2B Hobby sandboxes expire at 3600s. Leave five minutes for Harbor
        # cleanup, patch capture, verification, and artifact persistence.
        "--agent-timeout", "3300",
        "--tasks", str(row["task_id"]),
        "--trials-dir", str(row["trials_dir"]),
        "--preference-profile", str(row["profile_path"]),
        "--skip-existing",
    ]
    print("$ " + " ".join(command))
    if not execute:
        return 0
    return subprocess.run(command, cwd=SWE, env=subprocess_env()).returncode


def cache_audit(plan: list[dict[str, object]], require_hits: bool) -> dict[str, object]:
    rows = []
    failures = []
    for cell in plan:
        trial_root = Path(str(cell["trials_dir"]))
        transcripts = list(trial_root.glob("**/agent/opencode.txt"))
        if not transcripts:
            failures.append({
                "task_id": cell["task_id"],
                "condition": cell["condition"],
                "error": "missing agent/opencode.txt",
            })
        for transcript in transcripts:
            cached = 0
            prompt = 0
            model_turns = 0
            for line in transcript.read_text(errors="replace").splitlines():
                try:
                    event = json.loads(line)
                except (json.JSONDecodeError, AttributeError):
                    continue
                if event.get("type") != "step_finish":
                    continue
                model_turns += 1
                usage = (event.get("part") or {}).get("tokens") or {}
                cache_read = int((usage.get("cache") or {}).get("read") or 0)
                cached += cache_read
                prompt += int(usage.get("input") or 0) + cache_read
            eligible = model_turns > 1
            ok = not eligible or cached > 0
            row = {
                "task_id": cell["task_id"],
                "condition": cell["condition"],
                "transcript": str(transcript),
                "model_turns": model_turns,
                "prompt_tokens": prompt,
                "cached_input_tokens": cached,
                "cache_hit_rate": (cached / prompt) if prompt else 0.0,
                "cache_required": eligible,
                "cache_ok": ok,
            }
            rows.append(row)
            if not ok:
                failures.append(row)
    report = {"rows": rows, "failures": failures}
    report_path = ROOT / "outputs" / "longitudinal_pilot" / "opencode_cache_audit.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"cache audit: {len(rows)} transcripts, {len(failures)} failures -> {report_path}")
    if require_hits and failures:
        raise SystemExit("Cache audit failed: resumed OpenCode calls reported zero cached tokens")
    return report


def evaluate(
    plan: list[dict[str, object]], results_dir: Path, execute: bool, *,
    skip_correctness: bool = False,
) -> int:
    roots = sorted({str(row["trials_dir"]) for row in plan})
    command = [str(SWE / ".venv" / "bin" / "python"), "-m", "eval.run_eval"]
    for root in roots:
        command += ["--trials-root", root]
    command += [
        "--tasks-root", str(SWE / "tasks"),
        "--output-dir", str(results_dir),
        "--model-tag", "opencode-gpt54-high-profile-study",
        "--correctness-workers", "8",
        "--intent-coverage-workers", "5",
        "--tag-workers", "8",
        "--intent-coverage-model", "openai/gpt-5-mini",
        "--tag-model", "openai/gpt-5-mini",
    ]
    if skip_correctness:
        command.append("--skip-correctness")
    print("$ " + " ".join(command))
    if not execute:
        return 0
    return subprocess.run(command, cwd=SWE, env=subprocess_env()).returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("plan", "run", "audit-cache", "evaluate"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--approved-budget-usd", type=float)
    parser.add_argument(
        "--parallel", type=int, default=1,
        help="Number of independent Harbor/E2B cells to run concurrently",
    )
    parser.add_argument("--trials-root", type=Path, default=DEFAULT_TRIALS)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--require-cache-hit", action="store_true")
    parser.add_argument(
        "--skip-correctness",
        action="store_true",
        help="Run intent coverage and message tagging without the correctness judge",
    )
    args = parser.parse_args()
    if args.parallel < 1:
        raise SystemExit("--parallel must be at least 1")

    load_env()
    plan = build_plan(args.smoke, args.trials_root)
    validate_plan(plan)
    plan_path = ROOT / "outputs" / "longitudinal_pilot" / (
        "opencode_profile_smoke_plan.json" if args.smoke else "opencode_profile_plan.json"
    )
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    print(f"planned {len(plan)} cells -> {plan_path}")

    if args.stage == "plan":
        for row in plan:
            print(f"{row['task_id']}\t{row['condition']}\t{row['profile_path']}")
        return 0
    if args.stage == "run":
        if not args.execute or args.parallel == 1:
            for row in plan:
                rc = run_command(row, args.execute)
                if rc:
                    return rc
            return 0
        failures: list[tuple[str, str, int]] = []
        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            future_rows = {pool.submit(run_command, row, True): row for row in plan}
            for future in as_completed(future_rows):
                row = future_rows[future]
                rc = future.result()
                if rc:
                    failures.append((str(row["task_id"]), str(row["condition"]), rc))
        if failures:
            for task_id, condition, rc in failures:
                print(f"FAILED rc={rc}: {task_id} {condition}", file=sys.stderr)
            return 1
        return 0
    if args.stage == "audit-cache":
        cache_audit(plan, args.require_cache_hit)
        return 0
    return evaluate(
        plan, args.results_dir, args.execute,
        skip_correctness=args.skip_correctness,
    )


if __name__ == "__main__":
    raise SystemExit(main())
