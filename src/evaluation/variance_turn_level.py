#!/usr/bin/env python3
"""
Same discriminability = sigma^2_between / sigma^2_within metric as
discriminability_variance_ratio.py, but computed on RAW TURN-LEVEL
scores (chat_turn_vectors.csv) instead of session-level aggregated
vectors -- i.e. every scored user prompt is its own observation, pooled
directly by user_id across ALL their sessions, with no session-level
aggregation step in between.

Two differences to expect relative to the session-level version, both
worth checking against rather than assuming:
  - sigma2_within is likely LARGER here. Session-level aggregation
    (recent/maxtie/mean) already smooths away turn-to-turn noise within
    a session before a user's variance is ever computed; at the turn
    level that noise is still in the data.
  - zero-inflation is likely WORSE here. A session's aggregated score is
    only 0 if NO turn in it fired; an individual turn is far more often
    0 even in a session that has real signal overall (median firing
    rate is 0% for 10/14 axes per-turn). The raw/conditional split
    matters at least as much here as it did at the session level.

There is no "method" flag (recent/maxtie/mean) at this level -- there's
only the raw per-turn ternary score, since aggregation methods are a
session-level concept.

Note on independence: turns within the same session are not independent
observations (they're about the same conversation/task), so pooling all
of a user's turns together, ignoring session boundaries, treats them as
if they were. This is a real simplification, not a hidden assumption --
worth stating alongside the numbers, not just in this docstring.

Usage:
    python discriminability_variance_ratio_turnlevel.py \
        --turn-vectors chat_turn_vectors_full100.csv \
        --out-dir ./eval_out
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
    ap.add_argument("--turn-vectors", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--conditional", action="store_true",
                    help="restrict to turns where the axis fired (nonzero) "
                        "rather than all turns")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    turns = pd.concat([pd.read_csv(p) for p in args.turn_vectors.split(",")],
                      ignore_index=True)
    n_users = turns["user_id"].nunique()
    print(f"Loaded {len(turns)} turns, {n_users} users, TURN-LEVEL (no session "
         f"aggregation), {'CONDITIONAL (nonzero only)' if args.conditional else 'RAW (all turns)'}\n")

    rows = []
    for rid in RUBRIC_IDS:
        col = f"score_{rid}"
        if col not in turns.columns:
            continue
        data = turns[turns[col] != 0] if args.conditional else turns
        groups = [g[col].values for _, g in data.groupby("user_id")]
        r = variance_ratio(groups)
        p5, p50, p95 = bootstrap_ci(groups, n_boot=args.n_boot)
        rows.append({
            "axis": rid, "name": RUBRIC_NAMES[rid],
            "sigma2_within": r["sigma2_within"], "sigma2_between": r["sigma2_between"],
            "discriminability": r["discriminability"],
            "ci_p5": p5, "ci_p95": p95,
            "k_users": r["k"], "n_turns_used": int(sum(len(g) for g in groups)),
            "reason": r["reason"],
        })

    out = pd.DataFrame(rows).sort_values("discriminability", ascending=False, na_position="last")
    suffix = "conditional" if args.conditional else "raw"
    out_path = os.path.join(args.out_dir, f"discriminability_varratio_turnlevel_{suffix}.csv")
    out.to_csv(out_path, index=False)
    pd.set_option("display.width", 160)
    print(out[["axis", "name", "sigma2_within", "sigma2_between", "discriminability",
              "ci_p5", "ci_p95", "k_users", "n_turns_used"]].to_string(index=False))
    print(f"\nwrote {out_path}")
    print("\nNote: unbounded above, like the session-level version. Compare "
         "sigma2_within here against the session-level table's sigma2_within per "
         "axis -- if turn-level within-user variance is consistently larger, "
         "that confirms session-level aggregation was smoothing out real "
         "turn-to-turn noise before it ever reached the between/within split.")


if __name__ == "__main__":
    main()