#!/usr/bin/env python3
"""
Pick the case-study candidate from the sampled pool: the user whose
session count and average turns/session land closest to (10, 10), among
the high-signal arm (since the case study is meant to showcase strong
preference signal, not the control arm's null case).

"Turns" here = n_user_prompts, consistent with how eligibility was
defined in sample_users_sessions.py.

Usage:
    python select_case_study_user.py --selected-sessions ./source1_sample/selected_sessions.csv
"""

import argparse
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selected-sessions", required=True)
    ap.add_argument("--target-sessions", type=int, default=10)
    ap.add_argument("--target-turns", type=int, default=10)
    ap.add_argument("--arm", default="high_signal",
                    choices=["high_signal", "random_control", "any"])
    ap.add_argument("--top-n", type=int, default=10,
                    help="show the top N candidates, not just the winner")
    args = ap.parse_args()

    sel = pd.read_csv(args.selected_sessions)
    if args.arm != "any":
        sel = sel[sel["arm"] == args.arm]

    per_user = sel.groupby("user_id").agg(
        n_sessions=("session_id", "nunique"),
        avg_turns=("n_user_prompts", "mean"),
        min_turns=("n_user_prompts", "min"),
        max_turns=("n_user_prompts", "max"),
        total_pushback=("n_pushback", "sum"),
        pushback_rate=("nitpicker_pushback_rate", "first"),
    ).reset_index()

    # distance in (sessions, avg_turns) space, normalized so neither axis
    # dominates just because sessions is a smaller-range integer
    per_user["dist"] = (
        ((per_user["n_sessions"] - args.target_sessions) / args.target_sessions) ** 2 +
        ((per_user["avg_turns"] - args.target_turns) / args.target_turns) ** 2
    ) ** 0.5
    per_user = per_user.sort_values("dist")

    print(f"Target: {args.target_sessions} sessions x {args.target_turns} avg turns/session, "
         f"arm={args.arm}\n")
    print(per_user.head(args.top_n).to_string(index=False))

    winner = per_user.iloc[0]
    print(f"\n--> Best match: {winner['user_id']} "
         f"({int(winner['n_sessions'])} sessions, "
         f"avg {winner['avg_turns']:.1f} turns/session, "
         f"pushback_rate={winner['pushback_rate']:.3f})")
    print(f"\nTo run the case study, use:")
    print(f"  --selected-sessions can stay as-is; pass "
         f"--user-id {winner['user_id']} to chat_pref_vectorise_v2.py")


if __name__ == "__main__":
    main()