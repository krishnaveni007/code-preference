#!/usr/bin/env python3
"""
Two diagnostics on top of evaluate_full_run.py's raw discriminability
table, aimed at answering one question: are the low ICC values real (no
preference structure), or a sparsity/discretization artifact?

1. CONDITIONAL discriminability: same one-way ICC, but computed only on
   sessions where the axis actually fired (nonzero), per user. This
   directly tests the zero-inflation hypothesis -- most sessions score 0
   for most axes (median firing rate is 0% for 10/14 axes in the full
   run), and those zeros dominate the raw ICC's variance decomposition
   regardless of whether the nonzero cases are person-specific. If
   conditional ICC is much higher than raw ICC, sparsity was suppressing
   the signal, not erasing it.

   Caveat this doesn't fix: dropping zeros also drops users who never
   fired the axis at all, shrinking k (fewer groups) and n_i (fewer
   observations per group) -- so a HIGHER conditional ICC with a much
   SMALLER k is not directly comparable to the raw number; both are
   reported so you can see the tradeoff, not just the headline value.

2. BOOTSTRAP confidence interval on the all-100 raw ICC (resample users
   with replacement, recompute, repeat), to give a sense of how stable
   each point estimate actually is -- important before reading small
   between-arm differences (80 vs 20 users) as meaningful.

Usage:
    python discriminability_conditional.py \
        --session-vectors chat_session_vectors_full100.csv \
        --out-dir ./eval_out --method recent
"""

import argparse
import os

import numpy as np
import pandas as pd

RUBRIC_IDS = [f"R{str(i).zfill(2)}" for i in range(1, 15)]
RUBRIC_NAMES = {
    "R01": "Solution Scope", "R02": "Abstraction Level", "R03": "Dependency Posture",
    "R04": "Correctness Guarantees", "R05": "Robustness Philosophy", "R06": "Testing Rigor",
    "R07": "Performance Sensitivity", "R08": "Security Posture", "R09": "Refactoring Aggressiveness",
    "R10": "Documentation Richness", "R11": "Explanation Verbosity", "R12": "Interaction Autonomy",
    "R13": "Code Clarity", "R14": "Code Conciseness",
}


def one_way_icc(values_by_group: list) -> dict:
    groups = [np.asarray(g, dtype=float) for g in values_by_group if len(g) > 0]
    k = len(groups)
    if k < 2:
        return {"icc": None, "reason": "fewer than 2 users with data"}
    n_i = np.array([len(g) for g in groups])
    N = n_i.sum()
    grand_mean = np.concatenate(groups).mean()
    group_means = np.array([g.mean() for g in groups])
    ss_between = np.sum(n_i * (group_means - grand_mean) ** 2)
    ss_within = np.sum([np.sum((g - g.mean()) ** 2) for g in groups])
    df_between, df_within = k - 1, N - k
    if df_within <= 0:
        return {"icc": None, "reason": "no within-user degrees of freedom"}
    msb = ss_between / df_between
    msw = ss_within / df_within if df_within > 0 else 0.0
    n0 = (N - (np.sum(n_i ** 2) / N)) / df_between
    denom = msb + (n0 - 1) * msw
    if denom == 0:
        return {"icc": None, "reason": "zero total variance"}
    return {"icc": round(float((msb - msw) / denom), 4), "k": k, "N": int(N), "reason": None}


def bootstrap_ci(groups: list, n_boot: int = 500, seed: int = 0) -> tuple:
    """Resample users (groups) with replacement, recompute ICC each time.
    Returns (p5, p50, p95) of the bootstrap distribution, or (None,None,None)
    if too few groups to resample meaningfully."""
    groups = [g for g in groups if len(g) > 0]
    if len(groups) < 5:
        return None, None, None
    rng = np.random.default_rng(seed)
    vals = []
    idx = np.arange(len(groups))
    for _ in range(n_boot):
        sample_idx = rng.choice(idx, size=len(idx), replace=True)
        sample = [groups[i] for i in sample_idx]
        r = one_way_icc(sample)
        if r["icc"] is not None:
            vals.append(r["icc"])
    if not vals:
        return None, None, None
    return (round(float(np.percentile(vals, 5)), 4),
            round(float(np.percentile(vals, 50)), 4),
            round(float(np.percentile(vals, 95)), 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session-vectors", required=True)
    ap.add_argument("--method", default="recent", choices=["recent", "maxtie", "mean"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-boot", type=int, default=500)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    sessions = pd.concat([pd.read_csv(p) for p in args.session_vectors.split(",")],
                        ignore_index=True)
    print(f"Loaded {len(sessions)} sessions, {sessions['user_id'].nunique()} users, "
         f"method={args.method}\n")

    rows = []
    for rid in RUBRIC_IDS:
        col = f"score_{args.method}_{rid}"
        if col not in sessions.columns:
            continue

        # raw: all sessions, including the zeros
        raw_groups = [g[col].values for _, g in sessions.groupby("user_id")]
        raw = one_way_icc(raw_groups)

        # conditional: only sessions where this axis actually fired
        fired = sessions[sessions[col] != 0]
        cond_groups = [g[col].values for _, g in fired.groupby("user_id")]
        cond = one_way_icc(cond_groups)

        # bootstrap CI on the RAW estimate (the one going in the headline table)
        p5, p50, p95 = bootstrap_ci(raw_groups, n_boot=args.n_boot)

        rows.append({
            "axis": rid, "name": RUBRIC_NAMES[rid],
            "icc_raw": raw["icc"], "icc_raw_ci_p5": p5, "icc_raw_ci_p50": p50,
            "icc_raw_ci_p95": p95,
            "icc_conditional": cond["icc"],
            "k_users_raw": raw.get("k"), "k_users_conditional": cond.get("k"),
            "n_sessions_conditional": cond.get("N"),
        })

    out = pd.DataFrame(rows).sort_values("icc_conditional", ascending=False, na_position="last")
    out.to_csv(os.path.join(args.out_dir, f"discriminability_conditional_{args.method}.csv"),
              index=False)
    pd.set_option("display.width", 160)
    print(out.to_string(index=False))

    print("\nReading guide:")
    print("  icc_raw            -- what evaluate_full_run.py reported (includes zeros)")
    print("  icc_raw_ci_p5/p95  -- 90% bootstrap interval on icc_raw; if this is wide, "
         "the point estimate alone is not trustworthy")
    print("  icc_conditional    -- ICC restricted to sessions where the axis actually "
         "fired -- tests whether zero-inflation was suppressing icc_raw")
    print("  k_users_conditional / n_sessions_conditional -- sample size backing "
         "icc_conditional; a high conditional ICC on very few users/sessions is "
         "suggestive, not conclusive")


if __name__ == "__main__":
    main()