#!/usr/bin/env python3
"""
Two things:
  1. Firing rate: of all turns in a session, what fraction trigger each
     axis? Reported as a distribution across sessions (not "does this
     axis ever fire" -- that's coverage.csv from evaluate_full_run.py --
     but "how DENSELY does it fire within a session, when it's active").
  2. User profile heatmap: each user's session vectors averaged into one
     row (14 axes), all 100 users stacked into one heatmap, rows ordered
     by hierarchical clustering so similar profiles group together --
     in random order, 100 rows of a 14-column heatmap look like noise
     even if real between-user structure exists.

Needs BOTH chat_turn_vectors.csv (for firing rate, which is a per-turn
question) and chat_session_vectors.csv (for the profile heatmap, which
is a per-session question).

Usage:
    python build_profiles_and_heatmap.py \
        --turn-vectors chat_turn_vectors_full100.csv \
        --session-vectors chat_session_vectors_full100.csv \
        --out-dir ./profile_out
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list

RUBRIC_IDS = [f"R{str(i).zfill(2)}" for i in range(1, 15)]
RUBRIC_NAMES = {
    "R01": "Solution Scope", "R02": "Abstraction Level", "R03": "Dependency Posture",
    "R04": "Correctness Guarantees", "R05": "Robustness Philosophy", "R06": "Testing Rigor",
    "R07": "Performance Sensitivity", "R08": "Security Posture", "R09": "Refactoring Aggressiveness",
    "R10": "Documentation Richness", "R11": "Explanation Verbosity", "R12": "Interaction Autonomy",
    "R13": "Code Clarity", "R14": "Code Conciseness",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turn-vectors", required=True,
                    help="comma-separated if multiple files")
    ap.add_argument("--session-vectors", required=True,
                    help="comma-separated if multiple files")
    ap.add_argument("--method", default="mean", choices=["mean", "recent", "maxtie"],
                    help="which session-level aggregation to average across a "
                        "user's sessions for the profile heatmap. 'mean' "
                        "(continuous) is the natural default here since we're "
                        "averaging again on top of it; 'recent'/'maxtie' are "
                        "ternary and get averaged into a continuous value too.")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    turns = pd.concat([pd.read_csv(p) for p in args.turn_vectors.split(",")],
                      ignore_index=True)
    sessions = pd.concat([pd.read_csv(p) for p in args.session_vectors.split(",")],
                        ignore_index=True)
    print(f"turns: {len(turns)} rows, sessions: {len(sessions)} rows, "
         f"{sessions['user_id'].nunique()} users")

    # ==================================================== 1. FIRING RATE
    print("\n=== Firing rate: fraction of turns in a session that trigger "
         "each axis (distribution across sessions) ===")
    rate_rows = []
    for rid in RUBRIC_IDS:
        col = f"score_{rid}"
        if col not in turns.columns:
            continue
        per_session_rate = (turns.assign(_fired=turns[col] != 0)
                            .groupby("session_id")["_fired"].mean())
        rate_rows.append({
            "axis": rid, "name": RUBRIC_NAMES[rid],
            "mean_pct_of_turns": round(float(per_session_rate.mean()) * 100, 2),
            "median_pct_of_turns": round(float(per_session_rate.median()) * 100, 2),
            "p90_pct_of_turns": round(float(per_session_rate.quantile(0.9)) * 100, 2),
            "max_pct_of_turns": round(float(per_session_rate.max()) * 100, 2),
        })
    rates = pd.DataFrame(rate_rows).sort_values("mean_pct_of_turns", ascending=False)
    rates.to_csv(os.path.join(args.out_dir, "firing_rate_per_session.csv"), index=False)
    print(rates.to_string(index=False))

    # bar chart: mean firing rate per axis, sorted
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(rates["axis"], rates["mean_pct_of_turns"], color="#3B6FA0")
    ax.set_ylabel("Mean % of turns in a session that fire this axis")
    ax.set_xlabel("Rubric axis")
    ax.set_title("Firing rate per axis (mean across sessions)")
    for i, v in enumerate(rates["mean_pct_of_turns"]):
        ax.annotate(f"{v:.1f}%", (i, v), textcoords="offset points",
                   xytext=(0, 3), ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "firing_rate_bar.png"),
               bbox_inches="tight", dpi=200)
    plt.close(fig)

    # ==================================================== 2. PROFILE HEATMAP
    print(f"\n=== User profile heatmap (method={args.method}, averaged "
         f"across each user's sessions) ===")
    col_prefix = f"score_{args.method}_"
    profile = sessions.groupby("user_id")[[f"{col_prefix}{rid}" for rid in RUBRIC_IDS]].mean()
    profile.columns = RUBRIC_IDS
    profile.to_csv(os.path.join(args.out_dir, "user_profiles.csv"))

    # hierarchical clustering on the 100 users so similar profiles sit
    # near each other -- in user_id order this would just look like noise
    mat = profile.values
    mat_filled = np.nan_to_num(mat, nan=0.0)
    order = leaves_list(linkage(mat_filled, method="average", metric="euclidean"))
    ordered_users = profile.index[order]
    ordered_mat = mat_filled[order]

    fig, ax = plt.subplots(figsize=(8, max(10, 0.13 * len(ordered_users))))
    im = ax.imshow(ordered_mat, cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(14))
    ax.set_xticklabels(RUBRIC_IDS, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(ordered_users)))
    ax.set_yticklabels(ordered_users, fontsize=5)
    ax.set_xlabel("Rubric axis")
    ax.set_title(f"User profiles ({args.method}, averaged across sessions), "
                "clustered by similarity")
    cbar = fig.colorbar(im, ax=ax, ticks=[-1, 0, 1], shrink=0.4)
    cbar.ax.set_yticklabels(["low (-1)", "no signal (0)", "high (+1)"])
    fig.tight_layout()
    path = os.path.join(args.out_dir, "user_profile_heatmap.png")
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"wrote {path} ({len(ordered_users)} users x 14 axes, clustered)")

    # a coarser companion: how much between-user spread is there per axis,
    # as a quick numeric companion to eyeballing the heatmap
    print("\n=== per-axis spread across the 100 user profiles ===")
    spread = profile.std().sort_values(ascending=False)
    for rid, v in spread.items():
        print(f"  {rid} ({RUBRIC_NAMES[rid]}): std={v:.3f}")


if __name__ == "__main__":
    main()