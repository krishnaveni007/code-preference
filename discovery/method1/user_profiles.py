#!/usr/bin/env python3
"""User x rubric matrix: for each user, the number of distinct turns whose statements fall in
each final rubric's clusters. Then: how many users clear a minimum, rubric co-occurrence across
users vs. a shuffled baseline, and a first look at user clustering."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

V = Path("outputs/discovery/method1/v2")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--min-turns", type=int, default=10)
    p.add_argument("--out-dir", type=Path, default=V / "user_profiles")
    args = p.parse_args()
    args.out_dir.mkdir(exist_ok=True)

    final = json.loads((V / "rubrics_final.json").read_text())["rubrics"]
    clusters = json.loads((V / "cluster_hdbscan/clusters_mcs100_leaf_ms10.json").read_text())
    ordered = sorted(clusters["clusters"], key=lambda c: -c["n_turns"])
    rank_to_cid = {i + 1: c["cluster"] for i, c in enumerate(ordered)}
    cid_to_rubric = {rank_to_cid[r]: rb["id"] for rb in final for r in rb["clusters"]}
    rubric_ids = [rb["id"] for rb in final]

    L = np.asarray(clusters["assignments"])
    rows = [json.loads(l) for l in (V / "statements_flat.jsonl").open()]
    recs = [(r["user_id"], f"{r['session_id']}:{r['turn_number']}", cid_to_rubric[L[i]])
            for i, r in enumerate(rows) if L[i] >= 0 and L[i] in cid_to_rubric]
    df = pd.DataFrame(recs, columns=["user_id", "turn", "rubric"]).drop_duplicates()
    counts = df.pivot_table(index="user_id", columns="rubric", values="turn", aggfunc="nunique", fill_value=0)
    counts = counts.reindex(columns=rubric_ids, fill_value=0)
    counts["total_turns"] = df.groupby("user_id")["turn"].nunique()
    counts.to_csv(args.out_dir / "user_rubric_counts.csv")

    print(f"statements in final-rubric clusters: {len(df)} (turn,rubric) pairs from {df.turn.nunique()} turns, {counts.shape[0]} users")
    print(f"turns per user: median {counts.total_turns.median():.0f}, p75 {counts.total_turns.quantile(.75):.0f}, max {counts.total_turns.max()}")
    keep = counts[counts.total_turns >= args.min_turns]
    print(f"users with >= {args.min_turns} turns: {len(keep)}  (they hold {keep.total_turns.sum()/counts.total_turns.sum():.0%} of turns)")

    X = keep[rubric_ids].to_numpy(dtype=float)
    P = X / X.sum(axis=1, keepdims=True)
    pd.DataFrame(P, index=keep.index, columns=rubric_ids).to_csv(args.out_dir / "user_rubric_proportions.csv")

    # how concentrated is each user? share of their turns in their top-3 rubrics, vs. shuffled baseline
    top3 = np.sort(P, axis=1)[:, -3:].sum(axis=1)
    rng = np.random.default_rng(0)
    base = []
    pool = df[df.user_id.isin(keep.index)]
    for _ in range(20):
        sh = pool.copy(); sh["rubric"] = rng.permutation(sh["rubric"].to_numpy())
        c = sh.pivot_table(index="user_id", columns="rubric", values="turn", aggfunc="nunique", fill_value=0).reindex(columns=rubric_ids, fill_value=0).to_numpy(float)
        c = c / c.sum(1, keepdims=True); base.append(np.sort(c, 1)[:, -3:].sum(1).mean())
    print(f"\nmean share of a user's turns in their top-3 rubrics: real {top3.mean():.2f}, shuffled {np.mean(base):.2f}")

    # rubric co-occurrence across users (Spearman on proportions), real vs shuffled
    C = pd.DataFrame(P, columns=rubric_ids).corr(method="spearman")
    iu = np.triu_indices(len(rubric_ids), 1)
    print(f"|spearman| between rubrics across users: mean {np.abs(C.to_numpy()[iu]).mean():.2f}, max {np.abs(C.to_numpy()[iu]).max():.2f}")
    pairs = sorted(((C.iloc[i, j], rubric_ids[i], rubric_ids[j]) for i, j in zip(*iu)), key=lambda t: -abs(t[0]))
    print("strongest pairs:")
    for v, a, b in pairs[:6]:
        print(f"  {v:+.2f}  {a} — {b}")
    C.to_csv(args.out_dir / "rubric_correlation_across_users.csv")
    print(f"\n-> {args.out_dir}")


if __name__ == "__main__":
    main()
