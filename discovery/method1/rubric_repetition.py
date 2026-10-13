#!/usr/bin/env python3
"""Do users bring up the same rubric again in later sessions?

For each rubric: users who express it in >= 2 (and >= 3) distinct sessions, as a share of users
who express it at all. Baseline: shuffle rubric labels among each user's own turns (keeping their
sessions and turn counts), so repetition that arises only from having many sessions is removed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

V = Path("outputs/discovery/method1/final")


def load_turn_rubrics() -> pd.DataFrame:
    final = json.loads((V / "rubrics_final.json").read_text())["rubrics"]
    clusters = json.loads((V / "hdbscan/clusters_mcs100_leaf_ms10.json").read_text())
    ordered = sorted(clusters["clusters"], key=lambda c: -c["n_turns"])
    rank_to_cid = {i + 1: c["cluster"] for i, c in enumerate(ordered)}
    cid_to_rubric = {rank_to_cid[r]: rb["name"] for rb in final for r in rb["clusters"]}
    L = np.asarray(clusters["assignments"])
    rows = [json.loads(l) for l in (V / "statements_flat.jsonl").open()]
    recs = [(r["user_id"], r["session_id"], r["turn_number"], cid_to_rubric[L[i]])
            for i, r in enumerate(rows) if L[i] >= 0 and L[i] in cid_to_rubric]
    return pd.DataFrame(recs, columns=["user_id", "session_id", "turn_number", "rubric"]).drop_duplicates()


def repetition(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["rubric", "user_id"])["session_id"].nunique().reset_index(name="n_sessions")
    out = g.groupby("rubric").agg(users=("user_id", "size"),
                                  users_2plus=("n_sessions", lambda s: int((s >= 2).sum())),
                                  users_3plus=("n_sessions", lambda s: int((s >= 3).sum())))
    out["share_2plus"] = out.users_2plus / out.users
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--shuffles", type=int, default=200)
    p.add_argument("--min-sessions", type=int, default=2, help="only users with at least this many sessions can repeat")
    args = p.parse_args()

    df = load_turn_rubrics()
    sess_per_user = df.groupby("user_id")["session_id"].nunique()
    eligible = sess_per_user[sess_per_user >= args.min_sessions].index
    print(f"{df.turn_number.size} (turn, rubric) pairs, {df.user_id.nunique()} users, "
          f"{len(eligible)} users with >= {args.min_sessions} sessions containing rubric turns")
    d = df[df.user_id.isin(eligible)].copy()

    real = repetition(d)
    rng = np.random.default_rng(0)
    base = []
    for _ in range(args.shuffles):
        s = d.copy()
        s["rubric"] = s.groupby("user_id")["rubric"].transform(lambda x: rng.permutation(x.to_numpy()))
        base.append(repetition(s.drop_duplicates())["share_2plus"])
    base = pd.concat(base, axis=1)
    real["shuffled_share_2plus"] = base.mean(axis=1).reindex(real.index)
    real["shuffled_95pct"] = base.quantile(0.95, axis=1).reindex(real.index)
    real["above_chance"] = real.share_2plus > real.shuffled_95pct
    real = real.sort_values("share_2plus", ascending=False)
    real.to_csv(V / "user_profiles" / "rubric_repetition.csv")

    pd.set_option("display.width", 200)
    print(f"\nAmong users with >= {args.min_sessions} sessions: share of a rubric's users who express it in >= 2 sessions\n")
    print(f"{'rubric':26s} {'users':>5} {'>=2 sess':>8} {'>=3 sess':>8} {'share':>6} {'shuffled':>9} {'95%':>6}  above chance")
    for r, row in real.iterrows():
        print(f"{r:26s} {row.users:>5} {row.users_2plus:>8} {row.users_3plus:>8} {row.share_2plus:>6.2f} "
              f"{row.shuffled_share_2plus:>9.2f} {row.shuffled_95pct:>6.2f}  {'yes' if row.above_chance else ''}")
    tot = real.users_2plus.sum() / real.users.sum()
    print(f"\noverall: {tot:.2f} of (rubric, user) pairs repeat across sessions; shuffled {base.mean().mean():.2f}")

    # per user: of the rubrics they ever express, how many they repeat
    per_user = d.groupby(["user_id", "rubric"])["session_id"].nunique().reset_index(name="n")
    pu = per_user.groupby("user_id").agg(rubrics=("rubric", "size"), repeated=("n", lambda s: int((s >= 2).sum())))
    pu["sessions"] = sess_per_user.reindex(pu.index)
    print(f"\nper user (n={len(pu)}): rubrics expressed median {pu.rubrics.median():.0f}, repeated across sessions median {pu.repeated.median():.0f}; "
          f"{(pu.repeated >= 1).mean():.0%} of users repeat at least one rubric")
    pu.sort_values("repeated", ascending=False).to_csv(V / "user_profiles" / "user_repetition.csv")


if __name__ == "__main__":
    main()
