#!/usr/bin/env python3
"""
For each user, order their sessions chronologically and build a running
(cumulative) preference vector after each one -- the mean of
score_mean_<axis> over sessions 1..k, for k=1..N. Compare each running
vector to the user's own FINAL vector (all N sessions) via cosine
similarity, and find the point at which it stabilizes.

Cosine similarity, not Euclidean distance, is the primary metric here on
purpose: it's scale-invariant, which matters given the earlier finding
that overall intensity (how positive a user's scores are on average)
correlates with session count as a likely artifact -- a distance metric
sensitive to magnitude would conflate "hasn't converged yet" with "has
a lower baseline intensity," which are different things.

"Converged" = the smallest k such that cosine similarity to the final
vector is >= --threshold AND stays >= threshold for every session after
that (a one-time lucky spike early on doesn't count -- it has to hold).

Caveat worth keeping in view, not hidden: early cumulative vectors are
built from very little data (1-2 sessions, many axes still at exactly
0 because they haven't fired yet), so part of what "low early
similarity" measures is genuinely having too little data yet, not a
separate "instability" phenomenon -- that's the point of the analysis,
not a confound to explain away.

Usage:
    python convergence_analysis.py \
        --data-dir ./swechat_data \
        --session-vectors chat_session_vectors_full100.csv \
        --out-dir ./convergence_out \
        --min-sessions 5 --threshold 0.9
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUBRIC_IDS = [f"R{str(i).zfill(2)}" for i in range(1, 15)]


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return np.nan
    return float(np.dot(a, b) / (na * nb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="for sessions.parquet (created_at)")
    ap.add_argument("--session-vectors", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--min-sessions", type=int, default=5,
                    help="only analyze users with at least this many scored sessions")
    ap.add_argument("--threshold", type=float, default=0.9,
                    help="cosine similarity to final vector counted as 'converged'")
    ap.add_argument("--max-k", type=int, default=10,
                    help="cap the convergence curve x-axis (sessions-per-user cap "
                        "was 10 in the sampling design)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    sessions_vec = pd.concat([pd.read_csv(p) for p in args.session_vectors.split(",")],
                             ignore_index=True)
    sess_meta = pd.read_parquet(os.path.join(args.data_dir, "sessions.parquet"),
                                columns=["session_id", "created_at"])
    sessions_vec = sessions_vec.merge(sess_meta, on="session_id", how="left")
    sessions_vec["created_at"] = pd.to_datetime(sessions_vec["created_at"], utc=True)

    score_cols = [f"score_mean_{rid}" for rid in RUBRIC_IDS]
    missing = [c for c in score_cols if c not in sessions_vec.columns]
    if missing:
        raise SystemExit(f"missing columns: {missing} -- need score_mean_* "
                         f"(the continuous aggregation method)")

    convergence_rows = []
    curve_rows = []  # for the pooled convergence-curve plot

    for uid, g in sessions_vec.groupby("user_id"):
        g = g.dropna(subset=["created_at"]).sort_values("created_at")
        n = len(g)
        if n < args.min_sessions:
            continue

        mat = g[score_cols].values  # n_sessions x 14
        final_vec = mat.mean(axis=0)

        sims = []
        for k in range(1, n + 1):
            running_vec = mat[:k].mean(axis=0)
            sim = cosine_sim(running_vec, final_vec)
            sims.append(sim)
            curve_rows.append({"user_id": uid, "k": k, "n_total": n, "cosine_sim": sim})

        sims = np.array(sims)
        # first k such that sim >= threshold for ALL k' >= k (stable convergence)
        converged_k = None
        for k in range(len(sims)):
            tail = sims[k:]
            if np.all(~np.isnan(tail)) and np.all(tail >= args.threshold):
                converged_k = k + 1  # 1-indexed
                break
        convergence_rows.append({
            "user_id": uid, "n_sessions": n,
            "converged_at_session": converged_k,
            "converged": converged_k is not None,
            "final_sim_at_n_minus_1": round(float(sims[-2]), 4) if n >= 2 else None,
        })

    conv_df = pd.DataFrame(convergence_rows)
    conv_df.to_csv(os.path.join(args.out_dir, "convergence_per_user.csv"), index=False)

    n_users = len(conv_df)
    n_converged = int(conv_df["converged"].sum())
    print(f"{n_users} users with >= {args.min_sessions} sessions analyzed")
    print(f"{n_converged}/{n_users} ({n_converged/n_users:.0%}) converged (cosine "
         f">= {args.threshold} and held) within their observed sessions")
    if n_converged:
        conv_only = conv_df[conv_df["converged"]]["converged_at_session"]
        print(f"  converged_at_session: mean={conv_only.mean():.2f}, "
             f"median={conv_only.median():.0f}, "
             f"min={conv_only.min()}, max={conv_only.max()}")
    n_not = n_users - n_converged
    if n_not:
        print(f"  {n_not} users never stably reached threshold={args.threshold} "
             f"within their observed session count -- either they need more "
             f"sessions than we have, or their preference genuinely keeps "
             f"moving (task-conditional, as found for R01 in the case study) "
             f"rather than settling to one fixed vector at all.")

    # ---------------------------------------------------------- pooled curve
    curve_df = pd.DataFrame(curve_rows)
    curve_df = curve_df[curve_df["k"] <= args.max_k]
    pooled = curve_df.groupby("k")["cosine_sim"].agg(["mean", "median", "std", "count"])
    pooled.to_csv(os.path.join(args.out_dir, "convergence_curve.csv"))
    print(f"\n=== pooled convergence curve (mean cosine sim to own final vector, "
         f"by sessions observed so far) ===")
    print(pooled.to_string())

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(pooled.index, pooled["mean"], marker="o", color="#3B6FA0", label="mean")
    ax.fill_between(pooled.index,
                    pooled["mean"] - pooled["std"] / np.sqrt(pooled["count"]),
                    pooled["mean"] + pooled["std"] / np.sqrt(pooled["count"]),
                    alpha=0.2, color="#3B6FA0", label="±1 SE")
    ax.axhline(args.threshold, color="#A6373D", linestyle="--", linewidth=1,
              label=f"threshold ({args.threshold})")
    ax.set_xlabel("Sessions observed so far")
    ax.set_ylabel("Cosine similarity to user's own final vector")
    ax.set_title(f"Preference vector convergence (n={n_users} users with "
                f">={args.min_sessions} sessions)")
    ax.set_xticks(range(1, args.max_k + 1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "convergence_curve.png"),
               bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"\nwrote convergence_per_user.csv, convergence_curve.csv, convergence_curve.png")


if __name__ == "__main__":
    main()