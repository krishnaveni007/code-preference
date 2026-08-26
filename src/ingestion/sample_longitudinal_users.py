#!/usr/bin/env python3
"""Select a natural longitudinal SWE-Chat cohort for preference learning curves.

Unlike the earlier signal-enriched sample, this selection does not use
pushback labels or preference scores. It finds users with enough sessions and
takes a contiguous chronological tail for each user. The final sessions are
marked as a held-out validation block.
"""

import argparse
import os

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--sessions-per-user", type=int, default=30)
    parser.add_argument("--validation-sessions", type=int, default=10)
    parser.add_argument(
        "--max-users", type=int, default=None,
        help="optional deterministic cap after ranking by available session count",
    )
    args = parser.parse_args()

    if not 0 < args.validation_sessions < args.sessions_per_user:
        raise SystemExit("validation-sessions must be between 1 and sessions-per-user - 1")

    sessions = pd.read_parquet(
        os.path.join(args.data_dir, "sessions.parquet"),
        columns=[
            "session_id", "user_id", "repo_id", "created_at", "turn_count",
            "prompt_count",
        ],
    ).dropna(subset=["user_id", "created_at"])
    sessions["created_at"] = pd.to_datetime(sessions["created_at"], utc=True)

    available = sessions.groupby("user_id").size().rename("n_sessions_available")
    eligible = available[available >= args.sessions_per_user].sort_values(
        ascending=False, kind="stable"
    )
    if args.max_users is not None:
        eligible = eligible.head(args.max_users)

    selected = sessions[sessions["user_id"].isin(eligible.index)].copy()
    selected = selected.sort_values(["user_id", "created_at", "session_id"])
    selected = selected.groupby("user_id", group_keys=False).tail(args.sessions_per_user)
    selected["session_ordinal"] = selected.groupby("user_id").cumcount() + 1
    cutoff = args.sessions_per_user - args.validation_sessions
    selected["split"] = selected["session_ordinal"].gt(cutoff).map(
        {False: "estimation", True: "validation"}
    )
    # Kept for compatibility with the existing vectorizer's selection schema.
    selected["arm"] = "natural_longitudinal"
    selected = selected.merge(available, on="user_id", how="left")

    os.makedirs(args.out_dir, exist_ok=True)
    selected_path = os.path.join(args.out_dir, "selected_sessions.csv")
    selected.to_csv(selected_path, index=False)

    summary = selected.groupby("user_id").agg(
        n_sessions_selected=("session_id", "nunique"),
        n_sessions_available=("n_sessions_available", "first"),
        n_repos=("repo_id", "nunique"),
        n_prompts=("prompt_count", "sum"),
        first_session=("created_at", "min"),
        last_session=("created_at", "max"),
    ).reset_index()
    summary["span_days"] = (
        summary["last_session"] - summary["first_session"]
    ).dt.total_seconds() / 86400
    summary.to_csv(os.path.join(args.out_dir, "cohort_summary.csv"), index=False)

    print(
        f"Selected {selected['user_id'].nunique()} users, {len(selected)} sessions, "
        f"and {int(selected['prompt_count'].fillna(0).sum())} user prompts"
    )
    print(
        f"Per user: first {cutoff} selected sessions available for estimation; "
        f"last {args.validation_sessions} held out for validation"
    )
    print(f"Wrote {selected_path} and cohort_summary.csv")


if __name__ == "__main__":
    main()
