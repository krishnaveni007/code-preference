#!/usr/bin/env python3
"""
clean_cohort_users.py

Given clean_sessions.csv from clean_session_scan.py, report how the clean
sessions distribute across developers: sessions per user, and how many users
survive a minimum-sessions threshold.

Usage
  python3 clean_cohort_users.py --data-dir ./swechat_data \
      --clean ./scan_out/clean_sessions.csv --out-dir ./scan_out
  python3 clean_cohort_users.py --data-dir ./swechat_data \
      --clean ./scan_out/clean_sessions.csv --min-sessions 5
"""

import argparse
import os

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./swechat_data")
    ap.add_argument("--clean", default="./scan_out/clean_sessions.csv")
    ap.add_argument("--out-dir", default="./scan_out")
    ap.add_argument("--min-sessions", type=int, default=3,
                    help="threshold highlighted in the summary")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    clean = set(pd.read_csv(args.clean).session_id)

    cols = ["session_id", "repo_id", "user_id", "turn_count",
            "files_touched_count", "agent_percentage", "duration_seconds"]
    sessions = pd.read_parquet(f"{args.data_dir}/sessions.parquet")
    sessions = sessions[[c for c in cols if c in sessions.columns]]

    # user_id on sessions can be null; fall back to the commit author
    if sessions.user_id.isna().any():
        commits = pd.read_parquet(
            f"{args.data_dir}/commits.parquet",
            columns=["checkpoint_pk", "user_id"]).dropna()
        print(f"note: {int(sessions.user_id.isna().sum())} sessions have a "
              f"null user_id and are dropped from per-user counts")

    cl = sessions[sessions.session_id.isin(clean) & sessions.user_id.notna()]
    all_s = sessions[sessions.user_id.notna()]

    n_sess, n_users = len(cl), cl.user_id.nunique()
    print(f"\nclean sessions        {n_sess:,}")
    print(f"distinct developers   {n_users:,}")
    print(f"mean sessions/user    {n_sess / max(n_users, 1):.2f}")
    print(f"median sessions/user  {cl.groupby('user_id').size().median():.0f}")

    per_user = (cl.groupby("user_id")
                  .agg(sessions=("session_id", "size"),
                       repos=("repo_id", "nunique"),
                       med_turns=("turn_count", "median"),
                       med_files=("files_touched_count", "median"),
                       med_agent_pct=("agent_percentage", "median"))
                  .sort_values("sessions", ascending=False))
    per_user.to_csv(f"{args.out_dir}/clean_sessions_per_user.csv")

    print("\nusers surviving a minimum-sessions threshold:")
    print(f"  {'min sessions':>13}  {'users':>7}  {'sessions kept':>14}")
    for k in (1, 2, 3, 5, 10, 20):
        sub = per_user[per_user.sessions >= k]
        flag = "   <-- need >= 100 users" if len(sub) >= 100 else ""
        print(f"  {k:>13}  {len(sub):>7,}  {int(sub.sessions.sum()):>14,}{flag}")

    print(f"\ndistribution of sessions per user:")
    print(per_user.sessions.describe()[["min", "25%", "50%", "75%", "max"]]
          .astype(int).to_string())

    print(f"\ntop 10 developers by clean sessions:")
    print(per_user.head(10).round(1).to_string())

    # concentration: is the cohort a few heavy users or many light ones?
    top10_share = per_user.sessions.head(10).sum() / max(n_sess, 1)
    print(f"\ntop 10 developers hold {top10_share:.0%} of all clean sessions")

    # is the clean cohort representative?
    print("\nclean cohort vs. everything else:")
    other = all_s[~all_s.session_id.isin(clean)]
    for col in ("turn_count", "files_touched_count", "agent_percentage",
                "duration_seconds"):
        if col in cl.columns:
            print(f"  {col:<22} clean median {cl[col].median():>9.1f}   "
                  f"rest median {other[col].median():>9.1f}")

    print(f"\nwrote {args.out_dir}/clean_sessions_per_user.csv")


if __name__ == "__main__":
    main()