#!/usr/bin/env python3
"""Build a small, human-auditable Phase-0 preference-event sample.

This script deliberately does not label an implementation preference.  It
uses inexpensive metadata to find *candidate* events in chronologically held
out sessions, verifies that the following edit has an exact/strong
turn-to-commit mapping, and exports enough local context for manual review.

The existing chat rubric scores are used only as axis hints for reviewers;
they are never used as the future outcome or to decide which pole won.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "mapping"))
from build_turn_commit_map import (  # noqa: E402
    assign_actions_to_prompts,
    build_map,
)


CODE_AXES = {
    "R01": "Solution Scope",
    "R02": "Abstraction Level",
    "R03": "Dependency Posture",
    "R04": "Correctness Guarantees",
    "R05": "Robustness Philosophy",
    "R06": "Testing Rigor",
    "R07": "Performance Sensitivity",
    "R08": "Security Posture",
    "R09": "Refactoring Aggressiveness",
    "R10": "Documentation Richness",
    "R13": "Code Clarity",
    "R14": "Code Conciseness",
}

HIGH_VALUE_PUSHBACK = {
    "correction": 4,
    "rejection": 5,
    "takeover": 5,
    "requirement_change": 3,
    "failure_report": 2,
    "pacing_complaint": 1,
}


def text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value)


def clip(value: Any, limit: int = 1400) -> str:
    value = text(value).strip()
    return value if len(value) <= limit else value[:limit] + "\n...[truncated]"


def axis_hints(row: pd.Series) -> list[str]:
    hints = []
    for rid, name in CODE_AXES.items():
        value = row.get(f"score_{rid}", 0)
        if pd.notna(value) and float(value) != 0:
            hints.append(f"{rid} {name} ({int(value):+d})")
    return hints


def context_window(conversation: pd.DataFrame, turn_number: int) -> str:
    conversational = conversation[conversation["is_conversational"].fillna(False)].copy()
    before = conversational[conversational.turn_number < turn_number].tail(2)
    current = conversational[conversational.turn_number == turn_number]
    after = conversational[conversational.turn_number > turn_number].head(2)
    rows = pd.concat([before, current, after]).drop_duplicates("turn_number")
    parts = []
    for _, row in rows.iterrows():
        marker = "TARGET" if int(row.turn_number) == turn_number else "context"
        parts.append(
            f"[{marker} turn {int(row.turn_number)} | {row.role}]\n{clip(row.content)}"
        )
    return "\n\n".join(parts)


def action_summary(actions: list[dict[str, Any]]) -> tuple[str, str, str]:
    paths, added, deleted = [], [], []
    for action in actions:
        paths.append(action["file_path"])
        if action["added_lines"]:
            added.append(
                f"# {action['file_path']} (tool turn {action['tool_turn_number']})\n"
                + "\n".join(action["added_lines"])
            )
        if action["deleted_lines"]:
            deleted.append(
                f"# {action['file_path']} (tool turn {action['tool_turn_number']})\n"
                + "\n".join(action["deleted_lines"])
            )
    return (
        ";".join(sorted(set(paths))),
        clip("\n\n".join(added), 3000),
        clip("\n\n".join(deleted), 3000),
    )


def chronological_split(group: pd.DataFrame, min_past: int, future_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    group = group.sort_values(["created_at", "session_id"]).reset_index(drop=True)
    n_future = max(1, math.ceil(len(group) * future_fraction))
    n_future = min(n_future, len(group) - min_past)
    return group.iloc[:-n_future], group.iloc[-n_future:]


def candidate_rows_for_session(
    data_dir: Path,
    session_id: str,
    user_id: str,
    scores: pd.DataFrame,
    past_session_ids: list[str],
) -> list[dict[str, Any]]:
    session_scores = scores[
        (scores.session_id == session_id) & (scores.user_id == user_id)
    ].copy()
    session_scores = session_scores[
        session_scores.prompt_pushback.fillna("").isin(HIGH_VALUE_PUSHBACK)
    ]
    if session_scores.empty:
        return []

    conversation = pd.read_parquet(
        data_dir / "conversations.parquet",
        filters=[("session_id", "==", session_id)],
        columns=[
            "session_id", "turn_number", "conversation_turn_number", "role",
            "turn_type", "is_conversational", "content", "timestamp",
            "tool_name", "file_path", "tool_input_json",
        ],
    ).sort_values("turn_number")
    actions = assign_actions_to_prompts(conversation)
    actions_by_prompt: dict[int, list[dict[str, Any]]] = {}
    for action in actions:
        actions_by_prompt.setdefault(int(action["user_turn_number"]), []).append(action)

    edges, _ = build_map(data_dir, session_id)
    if edges.empty:
        return []
    good = edges[edges.confidence.isin(["exact", "strong"])].copy()
    if good.empty:
        return []

    results = []
    for _, score_row in session_scores.iterrows():
        turn = int(score_row.turn_number)
        prompt_actions = actions_by_prompt.get(turn, [])
        prompt_edges = good[good.user_turn_number == turn]
        if not prompt_actions or prompt_edges.empty:
            continue
        paths, added, deleted = action_summary(prompt_actions)
        hints = axis_hints(score_row)
        pushback = text(score_row.prompt_pushback)
        best_recall = float(prompt_edges.survival_recall.max())
        best_confidence = "exact" if (prompt_edges.confidence == "exact").any() else "strong"
        rank_score = (
            HIGH_VALUE_PUSHBACK.get(pushback, 0) * 10
            + min(len(prompt_actions), 5) * 2
            + min(len(hints), 4)
            + best_recall
        )
        prompt_content = conversation.loc[
            conversation.turn_number == turn, "content"
        ]
        results.append({
            "event_id": f"{session_id[:8]}_t{turn}",
            "user_id": user_id,
            "future_session_id": session_id,
            "future_turn_number": turn,
            "past_session_ids": json.dumps(past_session_ids),
            "n_past_sessions": len(past_session_ids),
            "prompt_pushback": pushback,
            "prompt_content": clip(prompt_content.iloc[0] if len(prompt_content) else ""),
            "context_window": context_window(conversation, turn),
            "axis_hints_not_labels": "; ".join(hints),
            "n_following_write_actions": len(prompt_actions),
            "edited_files": paths,
            "edit_added_lines": added,
            "edit_deleted_lines": deleted,
            "commit_shas": ";".join(sorted(prompt_edges.commit_sha.unique())),
            "mapping_confidence": best_confidence,
            "max_survival_recall": round(best_recall, 4),
            "candidate_rank_score": round(rank_score, 4),
            # Empty human-audit fields are the actual Phase-0 deliverable.
            "audit_usable": "",
            "audit_target_axis": "",
            "audit_outcome_direction": "",
            "audit_outcome_evidence": "",
            "audit_before_state_reconstructable": "",
            "audit_two_valid_implementations": "",
            "audit_task_forced_choice": "",
            "audit_notes": "",
        })
    return results


def render_report(events: pd.DataFrame, users: pd.DataFrame) -> str:
    lines = [
        "# Phase 0 candidate-event audit",
        "",
        "These events were selected with metadata and exact/strong edit-survival evidence. "
        "They are **not preference labels**. Review each event and complete the `audit_*` "
        "columns in `candidate_events.csv`.",
        "",
        "## Review checklist",
        "",
        "- Is the user's eventual choice explicit in the trajectory?",
        "- Could two comparably correct implementations satisfy the task?",
        "- Is the choice discretionary rather than forced by a bug, test, or compiler?",
        "- Is the before-state reconstructable?",
        "- Is one code-observable rubric axis the primary difference?",
        "",
        "## Selected users",
        "",
        "| user | past sessions | future sessions | candidates | repositories |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in users.iterrows():
        lines.append(
            f"| {row.user_id} | {row.n_past_sessions} | {row.n_future_sessions} | "
            f"{row.n_candidate_events} | {row.n_repositories} |"
        )
    for _, row in events.iterrows():
        lines.extend([
            "",
            f"## {row.event_id}: {row.user_id}",
            "",
            f"- Pushback: `{row.prompt_pushback}`",
            f"- Axis hints (not labels): {row.axis_hints_not_labels or 'none'}",
            f"- Mapping: `{row.mapping_confidence}`, survival `{row.max_survival_recall}`",
            f"- Files: {row.edited_files}",
            "",
            "### Conversation window",
            "",
            row.context_window,
            "",
            "### Lines added by following edit actions",
            "",
            "```text",
            row.edit_added_lines or "(none)",
            "```",
            "",
            "### Lines deleted by following edit actions",
            "",
            "```text",
            row.edit_deleted_lines or "(none)",
            "```",
        ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/swechat_data"))
    parser.add_argument("--selected-sessions", type=Path, default=Path("outputs/source1_sample/selected_sessions.csv"))
    parser.add_argument("--clean-sessions", type=Path, default=Path("outputs/scan_out/clean_sessions.csv"))
    parser.add_argument("--turn-scores", type=Path, default=Path("outputs/chat_vectors/chat_turn_vectors_full100.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/phase0_candidate_audit"))
    parser.add_argument("--min-past-sessions", type=int, default=3)
    parser.add_argument("--future-fraction", type=float, default=0.30)
    parser.add_argument("--target-users", type=int, default=10)
    parser.add_argument("--events-per-user", type=int, default=2)
    args = parser.parse_args()

    selected = pd.read_csv(args.selected_sessions, usecols=["user_id", "session_id", "arm"])
    clean_ids = set(pd.read_csv(args.clean_sessions).session_id.astype(str))
    sessions = pd.read_parquet(
        args.data_dir / "sessions.parquet",
        columns=["session_id", "user_id", "repo_id", "created_at"],
    )
    sessions["created_at"] = pd.to_datetime(sessions.created_at, utc=True, errors="coerce")
    cohort = selected.merge(sessions, on=["user_id", "session_id"], how="inner")
    cohort = cohort[cohort.session_id.astype(str).isin(clean_ids)]

    eligible: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for user_id, group in cohort.groupby("user_id"):
        if len(group) < args.min_past_sessions + 1:
            continue
        past, future = chronological_split(group, args.min_past_sessions, args.future_fraction)
        eligible[user_id] = (past, future)

    score_columns = [
        "user_id", "session_id", "turn_number", "prompt_pushback",
        *[f"score_{rid}" for rid in CODE_AXES],
    ]
    scores = pd.read_csv(args.turn_scores, usecols=score_columns)

    # Rank before touching the large commit parquet.  This is only an I/O
    # optimization: the rank uses pushback category counts, not preference
    # directions or agreement with either profile condition.
    scan_order = []
    for user_id, (_, future) in eligible.items():
        future_ids = set(future.session_id.astype(str))
        future_scores = scores[
            (scores.user_id == user_id) & scores.session_id.isin(future_ids)
        ]
        categories = future_scores.prompt_pushback.fillna("")
        metadata_rank = sum(HIGH_VALUE_PUSHBACK.get(value, 0) for value in categories)
        if metadata_rank:
            scan_order.append((metadata_rank, user_id))
    scan_order.sort(reverse=True)

    all_events: list[dict[str, Any]] = []
    user_meta: dict[str, dict[str, Any]] = {}
    for _, user_id in scan_order:
        past, future = eligible[user_id]
        past_ids = past.session_id.astype(str).tolist()
        user_events = []
        for session_id in future.session_id.astype(str):
            user_events.extend(candidate_rows_for_session(
                args.data_dir, session_id, user_id, scores, past_ids
            ))
        if not user_events:
            continue
        user_events.sort(key=lambda row: row["candidate_rank_score"], reverse=True)
        all_events.extend(user_events[: args.events_per_user])
        user_meta[user_id] = {
            "user_id": user_id,
            "n_past_sessions": len(past),
            "n_future_sessions": len(future),
            "n_candidate_events": len(user_events),
            "n_repositories": pd.concat([past, future]).repo_id.nunique(),
            "best_candidate_score": user_events[0]["candidate_rank_score"],
        }
        # Phase 0 needs an auditable small sample, not an exhaustive corpus
        # census. Stop once enough users have survived the expensive mapping
        # filter; candidate rank still orders events within these users.
        if len(user_meta) >= args.target_users:
            break

    if not all_events:
        raise SystemExit("No Phase-0 candidates passed the filters")
    users = pd.DataFrame(user_meta.values()).sort_values(
        ["best_candidate_score", "n_candidate_events"], ascending=False
    ).head(args.target_users)
    chosen = set(users.user_id)
    events = pd.DataFrame([row for row in all_events if row["user_id"] in chosen])
    events = events.sort_values(
        ["user_id", "candidate_rank_score"], ascending=[True, False]
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    events.to_csv(args.out_dir / "candidate_events.csv", index=False)
    users.to_csv(args.out_dir / "selected_users.csv", index=False)
    (args.out_dir / "audit_report.md").write_text(render_report(events, users))
    config = {
        "min_past_sessions": args.min_past_sessions,
        "future_fraction": args.future_fraction,
        "target_users": args.target_users,
        "events_per_user": args.events_per_user,
        "eligible_users_before_event_filter": len(eligible),
        "users_with_candidate_events": len(user_meta),
        "selected_users": len(users),
        "selected_events": len(events),
        "note": "Chat axes are reviewer hints only, never outcome labels.",
    }
    (args.out_dir / "run_summary.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps(config, indent=2))
    print(f"wrote {args.out_dir / 'candidate_events.csv'}")
    print(f"wrote {args.out_dir / 'audit_report.md'}")


if __name__ == "__main__":
    main()
