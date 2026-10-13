#!/usr/bin/env python3
"""Per rubric: how many users mention it once, twice, ... (mentions = distinct user turns).
Horizontal stacked bars, one sequential hue for the ordered bins."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rubric_repetition import load_turn_rubrics, V  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BINS = [(1, 1, "1"), (2, 2, "2"), (3, 4, "3–4"), (5, 9, "5–9"), (10, 10**9, "10+")]
RAMP = ["#cde2fb", "#9ec5f4", "#5598e7", "#256abf", "#104281"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--unit", choices=["turns", "sessions"], default="turns", help="what counts as one mention")
    p.add_argument("--out", type=Path, default=V / "user_profiles" / "rubric_mention_counts.png")
    args = p.parse_args()

    df = load_turn_rubrics()
    key = "turn_number" if args.unit == "turns" else "session_id"
    per = df.groupby(["rubric", "user_id"])[key if args.unit == "sessions" else ["session_id", "turn_number"]].nunique() \
        if args.unit == "sessions" else df.groupby(["rubric", "user_id"]).size()
    per = per.rename("n").reset_index() if isinstance(per, pd.Series) else per.reset_index()
    table = pd.DataFrame({lab: per[(per.n >= lo) & (per.n <= hi)].groupby("rubric").size() for lo, hi, lab in BINS}).fillna(0).astype(int)
    table = table.loc[table.sum(axis=1).sort_values().index]  # smallest total at the bottom -> largest on top
    table.to_csv(args.out.with_suffix(".csv"))

    fig, ax = plt.subplots(figsize=(11, 7.2), facecolor=SURFACE); ax.set_facecolor(SURFACE)
    left = np.zeros(len(table))
    for (lo, hi, lab), colour in zip(BINS, RAMP):
        vals = table[lab].to_numpy()
        ax.barh(table.index, vals, left=left, color=colour, edgecolor=SURFACE, linewidth=2, height=0.72, label=lab)
        for y, (l, v) in enumerate(zip(left, vals)):
            if v >= 4:
                ax.text(l + v / 2, y, str(v), ha="center", va="center", fontsize=8.5,
                        color=INK if colour in RAMP[:2] else SURFACE)
        left += vals
    for y, tot in enumerate(left):
        ax.text(tot + 1, y, str(int(tot)), va="center", fontsize=9, color=INK2)
    ax.set_xlabel(f"users", color=INK2, fontsize=10)
    ax.set_xlim(0, left.max() * 1.08)
    ax.tick_params(colors=INK2, length=0); ax.tick_params(axis="y", labelsize=10, labelcolor=INK)
    ax.xaxis.grid(True, color=GRID, linewidth=1); ax.set_axisbelow(True)
    for s in ax.spines.values(): s.set_visible(False)
    unit = "turns" if args.unit == "turns" else "sessions"
    ax.set_title(f"How many users mention each rubric, by number of {unit} in which they mention it", loc="left",
                 fontsize=12.5, color=INK, pad=22)
    ax.text(0, 1.01, f"segment = users who mention the rubric in that many {unit}; number at the end = all users who mention it",
            transform=ax.transAxes, fontsize=9, color=INK2, va="bottom")
    leg = ax.legend(title=f"{unit} per user", loc="lower right", frameon=False, fontsize=9, title_fontsize=9, ncol=5,
                    handlelength=1.2, columnspacing=1.0)
    for t in leg.get_texts(): t.set_color(INK2)
    leg.get_title().set_color(INK2)
    fig.tight_layout()
    fig.savefig(args.out, dpi=170, facecolor=SURFACE); fig.savefig(args.out.with_suffix(".pdf"), facecolor=SURFACE)
    print(f"-> {args.out} (+ .pdf, .csv)\n"); print(table.iloc[::-1].to_string())


if __name__ == "__main__":
    main()
