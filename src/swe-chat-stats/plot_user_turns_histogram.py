#!/usr/bin/env python3
"""Plot the distribution of user conversational turns per SWE-Chat session.

Usage:
    python src/swe-chat-stats/plot_user_turns_histogram.py \
        --turn-vectors outputs/chat_vectors/chat_turn_vectors_full100.csv \
        --output outputs/swe-chat-stats/section15_out/figures/user_turns_per_session.png
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BAR_COLOR = "#3B6FA0"

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
})


def user_turns_per_session(data_dir: str) -> pd.Series:
    conversations = pd.read_parquet(
        os.path.join(data_dir, "conversations.parquet"),
        columns=["session_id", "role", "is_conversational"],
    )
    sessions = pd.read_parquet(
        os.path.join(data_dir, "sessions.parquet"),
        columns=["session_id"],
    )

    user_turns = conversations[
        conversations["role"].eq("user")
        & conversations["is_conversational"].fillna(False)
    ].groupby("session_id").size()

    # Include sessions with no user conversational rows as zero-turn sessions.
    return (
        sessions.set_index("session_id")
        .join(user_turns.rename("n_user_turns"))
        ["n_user_turns"]
        .fillna(0)
        .astype(int)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/swechat_data")
    parser.add_argument(
        "--turn-vectors",
        help=(
            "Optional scored turn-vector CSV. When supplied, plot only the "
            "sessions and user turns present in this scoring cohort."
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "outputs/swe-chat-stats/section15_out/figures/"
            "hist_user_turns_per_session.png"
        ),
    )
    args = parser.parse_args()

    if args.turn_vectors:
        scored_turns = pd.read_csv(
            args.turn_vectors, usecols=["session_id", "turn_number"]
        )
        values = scored_turns.groupby("session_id").size()
        # The 100-user scoring cohort has a compact 3-40 turn range, so show
        # one bar per exact integer rather than combining values into bins.
        bin_edges = np.arange(values.min(), values.max() + 2)
        cohort_label = "100-user scoring cohort"
    else:
        values = user_turns_per_session(args.data_dir)
        # Irregular bins preserve detail near the median while keeping the
        # full dataset's long tail (maximum 523 turns) readable.
        bin_edges = np.array(
            [0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 524]
        )
        cohort_label = "all SWE-Chat sessions"

    counts, _ = np.histogram(values, bins=bin_edges)
    labels = []
    for low, high in zip(bin_edges[:-1], bin_edges[1:]):
        labels.append(str(low) if high - low == 1 else f"{low}-{high - 1}")

    fig, ax = plt.subplots(figsize=(12, 5.2))
    bars = ax.bar(range(len(counts)), counts, color=BAR_COLOR, width=0.85)
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("User conversational turns per session")
    ax.set_ylabel("Sessions")
    ax.set_title(
        f"SWE-Chat user turns per session — {cohort_label} "
        f"(n={len(values):,}, median={values.median():.0f})"
    )

    for bar, count in zip(bars, counts):
        if count:
            ax.annotate(
                f"{count:,}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                textcoords="offset points",
                xytext=(0, 4),
                ha="center",
                fontsize=8,
            )

    ax.margins(x=0.015)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)

    print(f"sessions={len(values)}")
    print(f"median={values.median():.0f}")
    print(f"mean={values.mean():.2f}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
