#!/usr/bin/env python3
"""Run the OpenCode personalized-skill SWE-Together intervention.

Only the nine personalized-skill cells are new runs. The existing
``longitudinal_profiles/*/01_baseline`` trials remain the control cohort.

AGENTS.md is never used as the intervention channel in this experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import run_opencode_profile_eval as base


ROOT = base.ROOT
SWE = base.SWE
SKILLS = ROOT / "outputs/longitudinal_pilot/personalized_skills"
TRIALS = SWE / "trials/personalized_skills"
RESULTS = SWE / "results/personalized_skills"
PLAN_PATH = ROOT / "outputs/longitudinal_pilot/personalized_skill_plan.json"
AUDIT_PATH = ROOT / "outputs/longitudinal_pilot/personalized_skill_audit.json"
CONDITION = "02_personalized_skill"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_plan() -> list[dict[str, object]]:
    rows = []
    for user, task, duration in base.TASKS:
        skill_path = SKILLS / user / "SKILL.md"
        rows.append({
            "user_id": user,
            "task_id": task,
            "condition": CONDITION,
            "control_condition": "01_baseline",
            "control_trials_dir": str(SWE / "trials/longitudinal_profiles" / task / "01_baseline"),
            "replicate": 1,
            "historical_duration_seconds": duration,
            "skill_path": str(skill_path),
            "skill_sha256": sha256(skill_path),
            "trials_dir": str(TRIALS / task / CONDITION),
            "action_harness": "opencode",
            "provider": "openai",
            "model": base.MODEL,
            "reasoning_effort": "high",
            "agents_intervention": False,
        })
    return rows


def validate_plan(plan: list[dict[str, object]]) -> None:
    missing = []
    for row in plan:
        task = SWE / "tasks" / str(row["task_id"])
        if not task.exists():
            missing.append(str(task))
        if row["skill_path"] and not Path(str(row["skill_path"])).exists():
            missing.append(str(row["skill_path"]))
    if missing:
        raise SystemExit("Missing plan inputs:\n" + "\n".join(sorted(set(missing))))


def run_command(row: dict[str, object], execute: bool) -> int:
    command = [
        str(SWE / ".venv/bin/python"),
        str(SWE / "src/run_eval.py"),
        "--model", base.MODEL,
        "--user-model", base.MODEL,
        "--tag", f"personalized-skill-{row['task_id']}-{row['condition']}",
        "--agent-type", "opencode",
        "--reasoning-effort", "high",
        "--env-type", "e2b",
        "--workers", "1",
        "--agent-timeout", "3300",
        "--tasks", str(row["task_id"]),
        "--trials-dir", str(row["trials_dir"]),
        "--skip-existing",
    ]
    if row["skill_path"]:
        command += ["--personalized-skill", str(row["skill_path"])]
    print("$ " + " ".join(command), flush=True)
    if not execute:
        return 0
    return subprocess.run(command, cwd=SWE, env=base.subprocess_env()).returncode


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def audit(plan: list[dict[str, object]], require_complete: bool) -> dict:
    rows = []
    failures = []
    for cell in plan:
        trial_results = sorted(Path(str(cell["trials_dir"])).glob("*/result.json"))
        if not trial_results:
            failure = {"task_id": cell["task_id"], "condition": cell["condition"], "error": "missing result.json"}
            failures.append(failure)
            rows.append({**cell, **failure})
            continue
        # One replicate is planned. If retries left multiple physical attempts,
        # retain each so failures are never hidden.
        for result_path in trial_results:
            trial = result_path.parent
            result = load_json(result_path)
            install = load_json(trial / "agent/personalized_skill_installation.json")
            invocations = load_json(trial / "agent/personalized_skill_invocations.json")
            rewards = ((result.get("verifier_result") or {}).get("rewards") or {})
            expected_condition = "personalized_skill"
            ok = (
                result.get("exception_info") is None
                and "reward" in rewards
                and install.get("condition") == expected_condition
                and install.get("agents_unchanged") is True
                and install.get("skill_sha256") == cell["skill_sha256"]
                and bool(invocations)
            )
            record = {
                **cell,
                "trial_path": str(trial),
                "status": "valid" if ok else "failed",
                "verifier_reward": rewards.get("reward"),
                "agents_sha256": install.get("agents_after"),
                "installed_skill_sha256": install.get("skill_sha256"),
                "skill_invocation_calls": invocations.get("skill_invocation_calls"),
                "skill_invocation_turns": invocations.get("skill_invocation_turns"),
                "invoked_outer_turn_indices": invocations.get("invoked_outer_turn_indices"),
                "exception_info": result.get("exception_info"),
            }
            rows.append(record)
            if not ok:
                failures.append(record)
    report = {"rows": rows, "failures": failures}
    AUDIT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    print(f"audit: {len(rows)} attempts, {len(failures)} failures -> {AUDIT_PATH}")
    if require_complete and failures:
        raise SystemExit("Personalized-skill audit failed")
    return report


def evaluate(plan: list[dict[str, object]], execute: bool, skip_correctness: bool) -> int:
    roots = sorted({str(row["trials_dir"]) for row in plan})
    # Pair new intervention runs with the already-completed original baseline.
    roots.extend(sorted({str(row["control_trials_dir"]) for row in plan}))
    command = [str(SWE / ".venv/bin/python"), "-m", "eval.run_eval"]
    for root in roots:
        command += ["--trials-root", root]
    command += [
        "--tasks-root", str(SWE / "tasks"),
        "--output-dir", str(RESULTS),
        "--model-tag", "opencode-gpt54-high-personalized-skill",
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
    parser.add_argument("stage", choices=("plan", "run", "audit", "evaluate"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--parallel", type=int, default=3)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--skip-correctness", action="store_true")
    args = parser.parse_args()
    if args.parallel < 1:
        raise SystemExit("--parallel must be at least 1")
    base.load_env()
    plan = build_plan()
    validate_plan(plan)
    PLAN_PATH.write_text(json.dumps(plan, indent=2) + "\n")
    print(f"planned {len(plan)} new personalized-skill cells -> {PLAN_PATH}")
    if args.stage == "plan":
        for row in plan:
            print(f"{row['user_id']}\t{row['task_id']}\t{row['condition']}\t{row['skill_path'] or '-'}")
        return 0
    if args.stage == "audit":
        audit(plan, args.require_complete)
        return 0
    if args.stage == "evaluate":
        return evaluate(plan, args.execute, args.skip_correctness)
    if not args.execute or args.parallel == 1:
        for row in plan:
            rc = run_command(row, args.execute)
            if rc:
                return rc
        return 0
    failures = []
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {pool.submit(run_command, row, True): row for row in plan}
        for future in as_completed(futures):
            row = futures[future]
            rc = future.result()
            if rc:
                failures.append((row["task_id"], row["condition"], rc))
    for task, condition, rc in failures:
        print(f"FAILED rc={rc}: {task} {condition}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
