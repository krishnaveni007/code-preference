#!/usr/bin/env python3
"""Map of the preference statements: 2-D UMAP for position, HDBSCAN (10-D) for membership.

Colour carries only assigned-vs-noise (two categories); cluster identity comes from the
number printed at each cluster's 2-D median and the list on the right."""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, NOISE = "#2a78d6", "#c3c2b7"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--clusters", type=Path, default=Path("outputs/discovery/method1/final/hdbscan/clusters_mcs100_leaf_ms10.json"))
    p.add_argument("--umap2d", type=Path, default=Path("outputs/discovery/method1/final/hdbscan/umap_2d_s0.npy"))
    p.add_argument("--out", type=Path, default=Path("outputs/discovery/method1/final/hdbscan/map_mcs100.png"))
    p.add_argument("--rubrics", type=Path, default=None,
                   help="consolidated rubrics json; labels then show rubric number (+A/B for merged clusters) and short names")
    args = p.parse_args()

    data = json.loads(args.clusters.read_text())
    L = np.asarray(data["assignments"]); Z = np.load(args.umap2d)
    ordered = sorted(data["clusters"], key=lambda c: -c["n_turns"])
    rank = {c["cluster"]: i + 1 for i, c in enumerate(ordered)}
    n_noise = int((L < 0).sum()); n_all = len(L)

    # optional: relabel clusters by rubric
    label = {c["cluster"]: str(rank[c["cluster"]]) for c in ordered}
    side_rows = None
    if args.rubrics:
        rub = json.loads(args.rubrics.read_text())["rubrics"]
        by_rank = {rank[c["cluster"]]: c["cluster"] for c in ordered}
        for k, rb in enumerate(rub, 1):
            two_sided = bool(rb["side_b_clusters"])
            for r in rb["side_a_clusters"]:
                label[by_rank[r]] = f"{k}A" if two_sided else str(k)
            for r in rb["side_b_clusters"]:
                label[by_rank[r]] = f"{k}B"
        side_rows = [(str(k), rb["short_name"],
                      f"{rb['turns_a']} / {rb['turns_b']}" if rb["side_b_clusters"] else str(rb["turns_a"]))
                     for k, rb in enumerate(rub, 1)]
        for c in ordered:  # clusters not absorbed by any rubric (e.g. uniform requests) keep their rank
            if label[c["cluster"]] == str(rank[c["cluster"]]) and not any(
                    rank[c["cluster"]] in rb["side_a_clusters"] + rb["side_b_clusters"] for rb in rub):
                label[c["cluster"]] = f"c{rank[c['cluster']]}"

    plt.rcParams["font.family"] = ["DejaVu Sans", "sans-serif"]
    fig = plt.figure(figsize=(17, 9.5), facecolor=SURFACE)
    ax = fig.add_axes([0.02, 0.05, 0.54, 0.86]); ax.set_facecolor(SURFACE)
    side = fig.add_axes([0.575, 0.05, 0.415, 0.86]); side.axis("off")

    noise = L < 0
    ax.scatter(Z[noise, 0], Z[noise, 1], s=4, c=NOISE, alpha=0.5, linewidths=0, rasterized=True)
    ax.scatter(Z[~noise, 0], Z[~noise, 1], s=5, c=BLUE, alpha=0.45, linewidths=0, rasterized=True)
    # label positions: 2-D median per cluster, then push apart any pair closer than min_gap
    pos = np.array([[np.median(Z[L == c["cluster"], 0]), np.median(Z[L == c["cluster"], 1])] for c in ordered])
    span = max(np.ptp(Z[:, 0]), np.ptp(Z[:, 1])); min_gap = 0.045 * span
    for _ in range(200):
        moved = False
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                d = pos[j] - pos[i]; dist = np.hypot(*d)
                if dist < min_gap:
                    push = (d / (dist or 1e-9)) * (min_gap - dist) / 2
                    pos[i] -= push; pos[j] += push; moved = True
        if not moved:
            break
    for c, (x, y) in zip(ordered, pos):
        ax.text(x, y, label[c["cluster"]], ha="center", va="center", fontsize=10.5, fontweight="bold", color=INK,
                bbox=dict(boxstyle="round,pad=0.28,rounding_size=0.9", fc=SURFACE, ec=INK, lw=0.8, alpha=0.95), zorder=5)
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Preference statements from SWE-Chat user turns", loc="left", fontsize=15, color=INK, pad=14)
    ax.text(0, 1.005, f"{n_all:,} statements · UMAP 2-D for position · HDBSCAN (min_cluster_size=100) in 10-D for membership · "
            f"blue = in one of {len(ordered)} clusters ({1-n_noise/n_all:.0%}) · gray = unassigned ({n_noise/n_all:.0%})",
            transform=ax.transAxes, fontsize=9.5, color=INK2, va="bottom")

    side.set_xlim(0, 1); side.set_ylim(0, 1)
    if side_rows is None:
        side_rows = [(str(rank[c["cluster"]]), textwrap.shorten(c.get("short_name", c["name"]), width=78, placeholder="…"),
                      str(c["n_turns"])) for c in ordered]
        heading = "Clusters, by number of user turns"
    else:
        heading = "Rubrics (A / B turns where two-sided)"
    side.text(0, 1.0, heading, fontsize=12, color=INK, fontweight="bold", va="top")
    side.text(1.0, 1.0, "turns", fontsize=9.5, color=MUTED, va="top", ha="right")
    y = 0.955; step = 0.955 / max(len(side_rows), 1)
    for num, name, turns_txt in side_rows:
        side.text(0.0, y, f"{num:>2}", fontsize=10, color=INK, fontweight="bold", va="top", family="DejaVu Sans Mono")
        side.text(0.05, y, name, fontsize=10, color=INK, va="top")
        side.text(1.0, y, turns_txt, fontsize=9.5, color=MUTED, va="top", ha="right", family="DejaVu Sans Mono")
        y -= step
    side.text(0, 0.0, "Positions are a 2-D projection; distances are approximate. Identity is given by the label, not by colour.\n"
                      "A/B mark the two sides of one rubric; c# = cluster not turned into a rubric.",
              fontsize=8.5, color=MUTED, va="bottom")
    fig.savefig(args.out, dpi=170, facecolor=SURFACE)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
