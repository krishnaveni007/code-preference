#!/usr/bin/env python3
"""
Bar chart: x = percentage of a session's turns that are preference-active
(at least one of the 14 axes fired), y = number of sessions in that
percentage band.

Usage:
    python plot_sessions_vs_active_turns.py \
        --turn-vectors chat_turn_vectors_full100.csv \
        --out-dir ./eval_out
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUBRIC_IDS = [f"R{str(i).zfill(2)}" for i in range(1, 15)]
BAR_COLOR = "#3B6FA0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turn-vectors", required=True,
                    help="comma-separated if multiple files")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--bin-width", type=int, default=10,
                    help="percentage points per bin, e.g. 10 -> 0-10%, 10-20%, ...")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    turns = pd.concat([pd.read_csv(p) for p in args.turn_vectors.split(",")],
                      ignore_index=True)
    score_cols = [f"score_{rid}" for rid in RUBRIC_IDS]
    turns["any_fired"] = (turns[score_cols] != 0).any(axis=1)

    per_session = turns.groupby("session_id")["any_fired"].agg(
        n_turns="size", n_active="sum")
    per_session["pct_active"] = 100 * per_session["n_active"] / per_session["n_turns"]
    per_session.to_csv(os.path.join(args.out_dir, "active_turns_per_session.csv"))

    vals = per_session["pct_active"].values
    n_sessions = len(vals)
    n_zero = int((vals == 0).sum())
    print(f"{n_sessions} sessions")
    print(f"%% turns active per session: mean={vals.mean():.1f}%, "
         f"median={np.median(vals):.1f}%, min={vals.min():.1f}%, max={vals.max():.1f}%")
    print(f"sessions with 0%% active turns: {n_zero} ({n_zero/n_sessions:.1%})")

    # 0% gets its own bin (it's a qualitatively different case -- "no
    # signal at all" vs. "some signal, just not much"), then fixed-width
    # bins from just-above-0 to 100
    bw = args.bin_width
    edges = [0, 0.01] + list(np.arange(bw, 100 + bw, bw))
    edges[-1] = 100.0001  # ensure exactly 100% falls in the last bin
    counts, bin_edges = np.histogram(vals, bins=edges)

    labels = ["0%"]
    for i in range(1, len(bin_edges) - 1):
        lo, hi = bin_edges[i], min(bin_edges[i + 1], 100)
        labels.append(f"{lo:.0f}-{hi:.0f}%")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(range(len(counts)), counts, color=BAR_COLOR, width=0.85)
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("% of session's turns that are preference-active")
    ax.set_ylabel("Sessions")
    ax.set_title(f"Sessions vs. %% preference-active turns "
                f"(n={n_sessions}, median={np.median(vals):.0f}%%)")
    for i, c in enumerate(counts):
        if c > 0:
            ax.annotate(str(int(c)), (i, c), textcoords="offset points",
                       xytext=(0, 4), ha="center", fontsize=8)
    fig.tight_layout()
    path = os.path.join(args.out_dir, "sessions_vs_pct_active_turns.png")
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)

    print(f"\nwrote {path}")
    print(f"wrote {os.path.join(args.out_dir, 'active_turns_per_session.csv')}")


if __name__ == "__main__":
    main()