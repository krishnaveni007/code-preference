#!/usr/bin/env python3
"""
The simple, unbounded discriminability metric from the original 50-user
sanity check: discriminability = sigma^2_between / sigma^2_within, with
NO ANOVA mean-square weighting and NO n0 unequal-group-size correction
(that's what evaluate_full_run.py / discriminability_conditional.py
compute instead, via one_way_icc -- ICC(1) is a related but different,
bounded statistic).

Definitions used here (the "naive" version, matching a straightforward
groupby-mean / groupby-var computation rather than an ANOVA):
  sigma^2_between = variance, across the k users, of each user's own
                    mean session-level score for this axis
  sigma^2_within  = each user's own variance around their own mean,
                    averaged across users (simple mean of per-user
                    variances -- NOT pooled sum-of-squares/df)
  discriminability = sigma^2_between / sigma^2_within

This is unbounded above (unlike ICC) and, having no correction for
unequal sessions-per-user or small samples, is MORE exposed to noise
and zero-inflation than ICC is -- so the same raw-vs-conditional split
matters here too, arguably more.

Usage:
    python discriminability_variance_ratio.py \
        --session-vectors chat_session_vectors_full100.csv \
        --out-dir ./eval_out --method mean
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


def variance_ratio(values_by_group: list, ddof: int = 1) -> dict:
    """values_by_group: list of arrays, one per user.
    Users with only 1 observation contribute a within-user variance of
    NaN (can't estimate spread from a single point) and are excluded
    from the within-user average, but DO still contribute to the
    between-user mean-of-means. This matches how the earlier sanity
    check would have handled single-observation users."""
    groups = [np.asarray(g, dtype=float) for g in values_by_group if len(g) > 0]
    k = len(groups)
    if k < 2:
        return {"sigma2_between": None, "sigma2_within": None,
               "discriminability": None, "reason": "fewer than 2 users with data", "k": k}

    user_means = np.array([g.mean() for g in groups])
    sigma2_between = float(np.var(user_means, ddof=1))

    per_user_var = [float(np.var(g, ddof=ddof)) for g in groups if len(g) > ddof]
    if not per_user_var:
        return {"sigma2_between": round(sigma2_between, 6), "sigma2_within": None,
               "discriminability": None,
               "reason": "no user has enough observations to estimate within-user variance",
               "k": k}
    sigma2_within = float(np.mean(per_user_var))

    if sigma2_within == 0:
        return {"sigma2_between": round(sigma2_between, 6), "sigma2_within": 0.0,
               "discriminability": None,
               "reason": "zero within-user variance (every user's sessions identical) "
                        "-- ratio undefined (would be infinite)",
               "k": k}

    return {"sigma2_between": round(sigma2_between, 6), "sigma2_within": round(sigma2_within, 6),
            "discriminability": round(sigma2_between / sigma2_within, 4),
            "reason": None, "k": k}


def bootstrap_ci(groups: list, n_boot: int = 500, seed: int = 0) -> tuple:
    groups = [g for g in groups if len(g) > 0]
    if len(groups) < 5:
        return None, None, None
    rng = np.random.default_rng(seed)
    idx = np.arange(len(groups))
    vals = []
    for _ in range(n_boot):
        sample = [groups[i] for i in rng.choice(idx, size=len(idx), replace=True)]
        r = variance_ratio(sample)
        if r["discriminability"] is not None:
            vals.append(r["discriminability"])
    if not vals:
        return None, None, None
    return (round(float(np.percentile(vals, 5)), 4),
            round(float(np.percentile(vals, 50)), 4),
            round(float(np.percentile(vals, 95)), 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session-vectors", required=True)
    ap.add_argument("--method", default="mean", choices=["recent", "maxtie", "mean"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--conditional", action="store_true",
                    help="restrict to sessions where the axis fired (nonzero) "
                        "rather than all sessions")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    sessions = pd.concat([pd.read_csv(p) for p in args.session_vectors.split(",")],
                        ignore_index=True)
    n_users = sessions["user_id"].nunique()
    print(f"Loaded {len(sessions)} sessions, {n_users} users, method={args.method}, "
         f"{'CONDITIONAL (nonzero only)' if args.conditional else 'RAW (all sessions)'}\n")

    rows = []
    for rid in RUBRIC_IDS:
        col = f"score_{args.method}_{rid}"
        if col not in sessions.columns:
            continue
        data = sessions[sessions[col] != 0] if args.conditional else sessions
        groups = [g[col].values for _, g in data.groupby("user_id")]
        r = variance_ratio(groups)
        p5, p50, p95 = bootstrap_ci(groups, n_boot=args.n_boot)
        rows.append({
            "axis": rid, "name": RUBRIC_NAMES[rid],
            "sigma2_within": r["sigma2_within"], "sigma2_between": r["sigma2_between"],
            "discriminability": r["discriminability"],
            "ci_p5": p5, "ci_p50": p50, "ci_p95": p95,
            "k_users": r["k"], "reason": r["reason"],
        })

    out = pd.DataFrame(rows).sort_values("discriminability", ascending=False, na_position="last")
    suffix = "conditional" if args.conditional else "raw"
    out_path = os.path.join(args.out_dir,
                            f"discriminability_varratio_{args.method}_{suffix}.csv")
    out.to_csv(out_path, index=False)
    pd.set_option("display.width", 160)
    print(out[["axis", "name", "sigma2_within", "sigma2_between", "discriminability",
              "ci_p5", "ci_p95", "k_users"]].to_string(index=False))
    print(f"\nwrote {out_path}")
    print("\nNote: unbounded above, unlike ICC -- values >1 are expected and normal "
         "for a genuinely discriminating axis, not an error.")


if __name__ == "__main__":
    main()