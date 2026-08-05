#!/usr/bin/env python3
"""
Check whether a session's checkpoints are genuinely exclusive to it, or
shared with other sessions via the many-to-many checkpoints<->sessions
relationship (checkpoints.session_pks). Answers: is a commit that shows
up in this session's timeline actually "from" this session, or does it
belong to (or is shared with) a different session entirely?

Also checks whether a commit appearing twice under one session's
checkpoint list (as a46f6c26ae did) is a genuine dual-checkpoint
association or a duplicate row in commits.parquet.

Usage:
    python check_checkpoint_sharing.py --data-dir ./swechat_data \
        --session-id <session_id>
"""

import argparse
import json

import numpy as np
import pandas as pd


def maybe_json(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (str, bytes)):
        try:
            v = json.loads(x)
            return v if isinstance(v, list) else [v]
        except Exception:
            return []
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--session-id", required=True)
    args = ap.parse_args()

    sessions = pd.read_parquet(f"{args.data_dir}/sessions.parquet",
                               columns=["session_id", "checkpoint_ids", "created_at"])
    checkpoints = pd.read_parquet(f"{args.data_dir}/checkpoints.parquet",
                                  columns=["checkpoint_pk", "session_pks",
                                          "session_count", "commit_shas", "commit_count"])
    commits = pd.read_parquet(f"{args.data_dir}/commits.parquet",
                              columns=["commit_sha", "checkpoint_pk", "commit_date",
                                      "commit_message"])

    this_sess = sessions[sessions["session_id"] == args.session_id]
    if this_sess.empty:
        raise SystemExit(f"session {args.session_id} not found")
    my_checkpoints = maybe_json(this_sess.iloc[0]["checkpoint_ids"])
    print(f"Session {args.session_id[:12]}: {len(my_checkpoints)} checkpoints "
         f"{my_checkpoints}\n")

    for ckpt_id in my_checkpoints:
        row = checkpoints[checkpoints["checkpoint_pk"] == ckpt_id]
        if row.empty:
            print(f"  {ckpt_id}: NOT FOUND in checkpoints.parquet")
            continue
        r = row.iloc[0]
        sessions_on_ckpt = maybe_json(r["session_pks"])
        print(f"  {ckpt_id}:")
        print(f"    session_count (checkpoints.parquet field): {r['session_count']}")
        print(f"    session_pks: {sessions_on_ckpt}")
        other_sessions = [s for s in sessions_on_ckpt if s != args.session_id]
        if other_sessions:
            print(f"    >>> SHARED with {len(other_sessions)} other session(s): "
                 f"{other_sessions}")
            other_meta = sessions[sessions["session_id"].isin(other_sessions)]
            for _, o in other_meta.iterrows():
                print(f"        other session {o['session_id'][:12]}, "
                     f"created_at={o['created_at']}")
        else:
            print(f"    exclusive to this session")

        commit_shas = maybe_json(r["commit_shas"])
        print(f"    commit_count (checkpoints.parquet field): {r['commit_count']}")
        print(f"    commit_shas: {commit_shas}")

    print(f"\n--- checking for duplicate commit rows ---")
    my_commits = commits[commits["checkpoint_pk"].isin(my_checkpoints)]
    dupe_shas = my_commits["commit_sha"].value_counts()
    dupes = dupe_shas[dupe_shas > 1]
    if len(dupes):
        print(f"  {len(dupes)} commit SHA(s) appear more than once in "
             f"commits.parquet under this session's checkpoints:")
        for sha, n in dupes.items():
            rows = my_commits[my_commits["commit_sha"] == sha]
            print(f"    {sha}: {n} rows, checkpoint_pk values = "
                 f"{rows['checkpoint_pk'].tolist()}")
            print(f"      -> {'genuine dual-checkpoint association' if rows['checkpoint_pk'].nunique() > 1 else 'true duplicate row (same checkpoint_pk twice) -- likely a data quality issue'}")
    else:
        print("  no duplicate commit rows found")


if __name__ == "__main__":
    main()