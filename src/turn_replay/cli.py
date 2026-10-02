from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from src.pilot_study.common import write_json
from .models import ReplayInstance
from .runner import CONDITIONS, ReplayConfig, dry_run_plan, run_instance


def _conditions(value: str | None) -> tuple[str, ...]:
    return CONDITIONS if value in {None, "both"} else (value,)


def _config(args: argparse.Namespace) -> ReplayConfig:
    return ReplayConfig(
        work_root=args.work_dir,
        results_root=args.results_dir,
        conditions=_conditions(args.condition),
        preference_file=args.preference_file,
        codex_binary=args.codex_binary,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        timeout=args.timeout,
        overwrite=args.overwrite,
        run_setup=not args.skip_setup,
    )


def _add_run_options(parser: argparse.ArgumentParser, *, instance: bool) -> None:
    if instance:
        parser.add_argument("--instance", type=Path, required=True)
    parser.add_argument(
        "--condition", choices=("both", *CONDITIONS), default="both"
    )
    parser.add_argument("--preference-file", type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path("work"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--codex-binary", default="codex")
    parser.add_argument("--model")
    parser.add_argument("--reasoning-effort")
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-setup",
        action="store_true",
        help="Do not run metadata.environment.setup_command",
    )


def _matches(instance: ReplayInstance, args: argparse.Namespace) -> bool:
    fields = {
        "user_id": args.user_id,
        "session_id": args.session_id,
        "snapshot_confidence": args.snapshot_confidence,
        "original_agent": args.agent_type,
    }
    return all(
        expected is None or str(instance.metadata.get(name)) == expected
        for name, expected in fields.items()
    )


def _run(args: argparse.Namespace) -> int:
    instance = ReplayInstance.load(args.instance)
    config = _config(args)
    if args.dry_run:
        print(json.dumps(dry_run_plan(instance, config), indent=2))
        return 0
    results = run_instance(instance, config)
    for result in results:
        print(
            f"{result.instance_id}\t{result.condition}\t{result.exit_status}\t"
            f"{result.result_path}"
        )
    return 0 if all(result.exit_status == "success" for result in results) else 1


def _batch(args: argparse.Namespace) -> int:
    config = _config(args)
    roots = sorted(
        path for path in args.instances_dir.iterdir()
        if path.is_dir() and (path / "repo").is_dir()
    )
    report: list[dict[str, Any]] = []
    failures = 0
    for root in roots:
        try:
            instance = ReplayInstance.load(root)
            if not _matches(instance, args):
                continue
            if args.dry_run:
                row = dry_run_plan(instance, config)
                print(json.dumps(row, indent=2))
                report.append(row)
                continue
            results = run_instance(instance, config)
            row = {
                "instance_id": instance.instance_id,
                "results": [
                    {
                        "condition": result.condition,
                        "exit_status": result.exit_status,
                        "result_path": str(result.result_path),
                        "errors": result.errors,
                    }
                    for result in results
                ],
            }
            if any(result.exit_status != "success" for result in results):
                failures += 1
            report.append(row)
        except Exception as exc:
            failures += 1
            row = {"instance_path": str(root), "error": f"{type(exc).__name__}: {exc}"}
            report.append(row)
            print(json.dumps(row), file=sys.stderr)
            if args.fail_fast:
                break
    if not args.dry_run:
        write_json(args.results_dir.resolve() / "batch_results.json", report)
    print(f"batch instances={len(report)} failures={failures}")
    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.turn_replay",
        description="Replay one real SWE Chat user turn under paired conditions.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="Run one instance")
    _add_run_options(run, instance=True)
    run.set_defaults(function=_run)

    batch = subparsers.add_parser("batch", help="Run a directory of instances")
    batch.add_argument("--instances-dir", type=Path, required=True)
    _add_run_options(batch, instance=False)
    batch.add_argument("--user-id")
    batch.add_argument("--session-id")
    batch.add_argument("--snapshot-confidence")
    batch.add_argument("--agent-type")
    batch.add_argument("--fail-fast", action="store_true")
    batch.set_defaults(function=_batch)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.timeout <= 0:
        raise SystemExit("--timeout must be positive")
    if getattr(args, "instances_dir", None) is not None and not args.instances_dir.is_dir():
        raise SystemExit(f"--instances-dir is not a directory: {args.instances_dir}")
    return args.function(args)
