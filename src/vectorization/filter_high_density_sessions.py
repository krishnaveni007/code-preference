#!/usr/bin/env python3
"""
Filter to sessions where at least --threshold% of turns are
preference-active (>=1 of 14 axes fired), producing a filtered
session-vectors CSV (and matching turn-vectors CSV) ready to feed
directly into cluster_user_profiles.py and convergence_analysis_heldout.py
via their existing --session-vectors / --turn-vectors flags.

Usage:
    python filter_high_density_sessions.py \
        --session-vectors chat_session_vectors_full100.csv \
        --turn-vectors chat_turn_vectors_full100.csv \
        --out-dir ./high_density \
        --threshold 60
"""

import argparse
import os

import numpy as np
import pandas as pd

RUBRIC_IDS = [f"R{str(i).zfill(2)}" for i in range(1, 15)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session-vectors", required=True)
    ap.add_argument("--turn-vectors", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--threshold", type=float, default=60,
                    help="minimum %% of a session's turns that must be "
                        "preference-active to keep it")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    sessions = pd.concat([pd.read_csv(p) for p in args.session_vectors.split(",")],
                        ignore_index=True)
    turns = pd.concat([pd.read_csv(p) for p in args.turn_vectors.split(",")],
                      ignore_index=True)
    score_cols = [f"score_{rid}" for rid in RUBRIC_IDS]
    turns["any_fired"] = (turns[score_cols] != 0).any(axis=1)

    density = turns.groupby("session_id")["any_fired"].agg(n_turns="size", n_active="sum")
    density["pct_active"] = 100 * density["n_active"] / density["n_turns"]

    sessions = sessions.merge(density[["pct_active"]], on="session_id", how="left")
    kept = sessions[sessions["pct_active"] >= args.threshold].copy()
    kept_turns = turns[turns["session_id"].isin(kept["session_id"])]

    n_sessions_total = sessions["user_id"].count() and len(sessions)
    n_users_total = sessions["user_id"].nunique()
    n_users_kept = kept["user_id"].nunique()

    print(f"Sessions: {len(sessions)} total -> {len(kept)} kept "
         f"({len(kept)/len(sessions):.0%}) at >={args.threshold}% density")
    print(f"Users: {n_users_total} total -> {n_users_kept} have >=1 kept session "
         f"({n_users_kept/n_users_total:.0%})")

    if "arm" in kept.columns:
        print(f"\narm composition of KEPT sessions: "
             f"{kept.groupby('arm')['user_id'].nunique().to_dict()}")
        print(f"arm composition of ALL sessions: "
             f"{sessions.groupby('arm')['user_id'].nunique().to_dict()}")
        print("(if the kept-session arm split looks meaningfully more "
             "high-signal-heavy than the full split, this filter is not a "
             "neutral subsample -- worth stating in the report rather than "
             "treating cluster/convergence results on this subset as "
             "representative of the full 100-user cohort)")

    per_user_kept = kept.groupby("user_id").size()
    print(f"\nkept sessions per user (among the {n_users_kept} users with any): "
         f"min={per_user_kept.min()}, median={per_user_kept.median():.0f}, "
         f"max={per_user_kept.max()}")
    n_enough_for_convergence = int((per_user_kept >= 5).sum())
    print(f"users with >=5 kept sessions (needed for the held-out convergence "
         f"analysis at default --holdout 2 --min-train-sessions 3): "
         f"{n_enough_for_convergence}")

    sess_path = os.path.join(args.out_dir, "session_vectors_highdensity.csv")
    turn_path = os.path.join(args.out_dir, "turn_vectors_highdensity.csv")
    kept.to_csv(sess_path, index=False)
    kept_turns.to_csv(turn_path, index=False)
    print(f"\nwrote {sess_path} ({len(kept)} sessions)")
    print(f"wrote {turn_path} ({len(kept_turns)} turns)")
    print(f"\nNext steps:")
    print(f"  python cluster_user_profiles.py --session-vectors {sess_path} "
         f"--turn-vectors {turn_path} --out-dir ./cluster_highdensity --center")
    print(f"  python convergence_analysis_heldout.py --data-dir ./swechat_data "
         f"--session-vectors {sess_path} --out-dir ./convergence_highdensity "
         f"--holdout 2 --min-train-sessions 3")


if __name__ == "__main__":
    main()