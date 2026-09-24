#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.pilot_study.common import nested_find, pseudonym, write_json


SESSION_KEYS = {"session_id", "sessionid", "original_session_id", "source_session_id"}
TASK_KEYS = {"task_id", "taskid", "instance_id", "instanceid"}


def discover_tasks(root: Path) -> list[dict]:
    tasks = []
    for path in sorted(root.rglob("original_session.json")):
        try:
            raw = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            tasks.append({"path": str(path), "error": f"invalid_json: {error}"})
            continue
        tasks.append({
            "path": str(path),
            "task_id": nested_find(raw, TASK_KEYS) or path.parent.name,
            "held_out_session_id": nested_find(raw, SESSION_KEYS),
        })
    return tasks


def build_manifest(sessions_path: Path, together_root: Path, salt: str,
                   conversation_ids: set[str] | None = None) -> list[dict]:
    sessions = pd.read_parquet(sessions_path)
    required = {"session_id", "user_id", "repo_id", "created_at"}
    missing = required - set(sessions.columns)
    if missing:
        raise ValueError(f"sessions parquet missing columns: {sorted(missing)}")
    sessions = sessions.copy()
    sessions["session_id"] = sessions.session_id.astype(str)
    sessions["created_at"] = pd.to_datetime(sessions.created_at, utc=True, errors="coerce")
    counts = sessions.groupby("session_id", dropna=False).size()
    rows = []
    for task in discover_tasks(together_root):
        base = {"task_id": task.get("task_id"), "held_out_session_id": task.get("held_out_session_id"),
                "original_session_path": task["path"]}
        sid = task.get("held_out_session_id")
        reason = task.get("error")
        match = sessions[sessions.session_id == str(sid)] if sid else sessions.iloc[0:0]
        if not reason and not sid: reason = "missing_session_id"
        if not reason and counts.get(str(sid), 0) != 1: reason = "missing_or_ambiguous_swechat_session"
        if not reason:
            held = match.iloc[0]
            if pd.isna(held.user_id): reason = "missing_user_id"
            elif pd.isna(held.created_at): reason = "missing_held_out_timestamp"
        if reason:
            rows.append({**base, "eligible": False, "exclusion_reason": reason})
            continue
        earlier = sessions[(sessions.user_id == held.user_id) & (sessions.created_at < held.created_at)]
        earlier = earlier.sort_values(["created_at", "session_id"])
        prior = [{"session_id": r.session_id, "repository": r.repo_id,
                  "timestamp": r.created_at.isoformat(),
                  "extractable": conversation_ids is None or r.session_id in conversation_ids}
                 for r in earlier.itertuples()]
        rows.append({**base, "eligible": len(prior) >= 5, "exclusion_reason": None if len(prior) >= 5 else "fewer_than_5_prior_sessions",
                     "source_user_id": str(held.user_id),
                     "pseudonymous_user_id": pseudonym(str(held.user_id), salt),
                     "repository": held.repo_id, "held_out_timestamp": held.created_at.isoformat(),
                     "prior_session_ids": [p["session_id"] for p in prior],
                     "prior_sessions": prior, "n_prior_sessions": len(prior)})
    return rows


def select(rows: list[dict], n_users: int, seed: int, user_ids: list[str] | None = None,
           prior_limit: int | None = None) -> list[dict]:
    eligible = [r for r in rows if r.get("eligible") and (not user_ids or r.get("source_user_id") in user_ids)]
    # Latest held-out task per user; seeded hash is only a deterministic final tie-break.
    eligible.sort(key=lambda r: (r["pseudonymous_user_id"], r["held_out_timestamp"], r["task_id"]), reverse=True)
    latest = {}
    for row in eligible: latest.setdefault(row["pseudonymous_user_id"], row)
    candidates = list(latest.values())
    # Prefer repository diversity, then prior-session volume, then deterministic seed ordering.
    if user_ids:
        order = {user_id: index for index, user_id in enumerate(user_ids)}
        candidates.sort(key=lambda r: order[r["source_user_id"]])
    else:
        candidates.sort(key=lambda r: (-r["n_prior_sessions"], pseudonym(r["task_id"], str(seed))))
    chosen, repos, chosen_users = [], set(), set()
    for distinct in (True, False):
        for row in candidates:
            if row["pseudonymous_user_id"] in chosen_users or (distinct and row["repository"] in repos): continue
            selected = dict(row)
            selected["n_prior_sessions_available"] = row["n_prior_sessions"]
            if prior_limit:
                extractable = [p for p in row["prior_sessions"] if p.get("extractable", True)]
                selected["excluded_unextractable_prior_session_ids"] = [p["session_id"] for p in row["prior_sessions"] if not p.get("extractable", True)]
                selected["prior_sessions"] = extractable[-prior_limit:]
                selected["prior_session_ids"] = [p["session_id"] for p in selected["prior_sessions"]]
                selected["n_prior_sessions"] = len(selected["prior_session_ids"])
            chosen.append(selected); repos.add(row["repository"]); chosen_users.add(row["pseudonymous_user_id"])
            if len(chosen) == n_users: return chosen
    return chosen


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sessions", type=Path, required=True); p.add_argument("--swe-together", type=Path, required=True)
    p.add_argument("--conversations", type=Path, help="Validate that sampled prior sessions have conversation rows")
    p.add_argument("--out", type=Path, required=True); p.add_argument("--selected-out", type=Path, required=True)
    p.add_argument("--prior-sessions-out", type=Path)
    p.add_argument("--users", type=int, default=5); p.add_argument("--seed", type=int, default=20260907)
    p.add_argument("--user-ids", help="Comma-separated exact SWE-Chat user IDs")
    p.add_argument("--prior-limit", type=int, help="Use only this many sessions closest before the held-out task")
    p.add_argument("--pseudonym-salt", default="swe-together-pilot-v1")
    a = p.parse_args()
    conversation_ids = None
    if a.conversations:
        conversation_ids = set(pd.read_parquet(a.conversations, columns=["session_id"])["session_id"].astype(str))
    rows = build_manifest(a.sessions, a.swe_together, a.pseudonym_salt, conversation_ids)
    requested = a.user_ids.split(",") if a.user_ids else None
    chosen = select(rows, a.users, a.seed, requested, a.prior_limit)
    if requested:
        missing = set(requested) - {row.get("source_user_id") for row in chosen}
        if missing: raise SystemExit(f"requested users without an eligible task: {sorted(missing)}")
    write_json(a.out, rows); write_json(a.selected_out, chosen)
    if a.prior_sessions_out:
        a.prior_sessions_out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"session_id": sorted({sid for row in chosen for sid in row["prior_session_ids"]})}).to_csv(a.prior_sessions_out, index=False)
    print(f"discovered={len(rows)} eligible={sum(bool(r.get('eligible')) for r in rows)} selected={len(chosen)}")

if __name__ == "__main__": main()
