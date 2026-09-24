#!/usr/bin/env python3
"""Run the updated-rubric intervention study without rerunning baseline."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import run_opencode_profile_eval as base


ROOT = base.ROOT
SWE = base.SWE
PROFILES = ROOT / "outputs/longitudinal_pilot/updated_rubric_extraction/interventions"
TRIALS = SWE / "trials/updated_rubric_profiles"
RESULTS = SWE / "results/updated_rubric_profiles"
PLAN_PATH = ROOT / "outputs/longitudinal_pilot/updated_rubric_profile_plan.json"
CONDITIONS = (
    "02_high_confidence_only",
    "03_all_confidence_bands",
    "04_preference_contexts_by_direction",
    "05_descriptive_no_polarity",
)


def build_plan() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for user, task, duration in base.TASKS:
        conditions = [c for c in CONDITIONS if not (
            user == "Nagi-ovo" and c == "02_high_confidence_only"
        )]
        random.Random(f"updated-rubric:{base.SEED}:{task}").shuffle(conditions)
        for condition in conditions:
            rows.append({
                "user_id": user,
                "task_id": task,
                "condition": condition,
                "replicate": 1,
                "historical_duration_seconds": duration,
                "profile_path": str(PROFILES / user / condition / "CLAUDE.md"),
                "trials_dir": str(TRIALS / task / condition),
                "action_harness": "opencode",
                "provider": "openai",
                "model": base.MODEL,
                "reasoning_effort": "high",
                "tag_prefix": "updated-rubric",
            })
    return rows


def evaluation_roots(plan: list[dict[str, object]]) -> list[str]:
    roots = {str(row["trials_dir"]) for row in plan}
    # Reuse the original baseline cohort. The evaluator's validity filter
    # excludes the two failed Nagi baseline attempts automatically.
    old = SWE / "trials/longitudinal_profiles"
    for _, task, _ in base.TASKS:
        roots.add(str(old / task / "01_baseline"))
    return sorted(roots)


def evaluate(
    plan: list[dict[str, object]], execute: bool, *, skip_correctness: bool = False
) -> int:
    command = [str(SWE / ".venv/bin/python"), "-m", "eval.run_eval"]
    for root in evaluation_roots(plan):
        command += ["--trials-root", root]
    command += [
        "--tasks-root", str(SWE / "tasks"),
        "--output-dir", str(RESULTS),
        "--model-tag", "opencode-gpt54-high-updated-rubric",
        "--correctness-workers", "8",
        "--intent-coverage-workers", "5",
        "--tag-workers", "8",
        "--intent-coverage-model", "openai/gpt-5-mini",
        "--tag-model", "openai/gpt-5-mini",
    ]
    if skip_correctness:
        command.append("--skip-correctness")
    print("$ " + " ".join(command), flush=True)
    if not execute:
        return 0
    return subprocess.run(command, cwd=SWE, env=base.subprocess_env()).returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("plan", "run", "audit-cache", "evaluate"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--parallel", type=int, default=3)
    parser.add_argument("--require-cache-hit", action="store_true")
    parser.add_argument(
        "--skip-correctness",
        action="store_true",
        help="Run intent coverage and message tagging without the correctness judge",
    )
    args = parser.parse_args()
    if args.parallel < 1:
        raise SystemExit("--parallel must be at least 1")

    base.load_env()
    plan = build_plan()
    base.validate_plan(plan)
    PLAN_PATH.write_text(json.dumps(plan, indent=2) + "\n")
    print(f"planned {len(plan)} new cells -> {PLAN_PATH}", flush=True)

    if args.stage == "plan":
        for row in plan:
            print(f"{row['task_id']}\t{row['condition']}\t{row['profile_path']}")
        return 0
    if args.stage == "audit-cache":
        base.cache_audit(plan, args.require_cache_hit)
        return 0
    if args.stage == "evaluate":
        return evaluate(plan, args.execute, skip_correctness=args.skip_correctness)
    if not args.execute:
        for row in plan:
            base.run_command(row, False)
        return 0

    failures: list[tuple[str, str, int]] = []
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        future_rows = {pool.submit(base.run_command, row, True): row for row in plan}
        for future in as_completed(future_rows):
            row = future_rows[future]
            rc = future.result()
            if rc:
                failures.append((str(row["task_id"]), str(row["condition"]), rc))
    for task_id, condition, rc in failures:
        print(f"FAILED rc={rc}: {task_id} {condition}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
