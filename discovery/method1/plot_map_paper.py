#!/usr/bin/env python3
"""Paper-style map: scatter only, one colour per final rubric, rubric names written at the
cluster positions, noise and non-rubric clusters in gray. No title, legend or notes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

V = Path("outputs/discovery/method1/final")
SURFACE, INK, NOISE, DROPPED_GRAY = "#ffffff", "#0b0b0b", "#c8c7c1", "#7a7975"
# 15 distinguishable hues (tab20 saturated set first, then its lighter partners); tab20's gray is
# replaced by gold so gray stays reserved for clusters that were not kept as rubrics
PALETTE = [plt.cm.tab20(i) for i in (0, 2, 4, 6, 8, 10, 12)] + ["#e6ab02"] + [plt.cm.tab20(i) for i in (16, 18, 1, 3, 5, 7, 9)]
# clusters (by turn-count rank) that were examined but not kept as rubrics
DROPPED = {8: "Session State", 9: "Commit Granularity", 11: "UI Layout", 13: "Review Gating",
           16: "Quality Gates", 19: "UI Color", 20: "Naming"}


def densest(pts: np.ndarray, k: int = 20) -> np.ndarray:
    """Point of the cluster with the most neighbours within a small radius (2-D)."""
    if len(pts) <= k:
        return np.median(pts, axis=0)
    d, _ = cKDTree(pts).query(pts, k=k)
    return pts[np.argmin(d[:, -1])]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=V / "hdbscan/map_rubrics_color.pdf")
    p.add_argument("--fontsize", type=float, default=11.5)
    p.add_argument("--width", type=float, default=16.0, help="figure width in inches")
    p.add_argument("--height", type=float, default=6.5, help="figure height in inches")
    p.add_argument("--label-anchor", choices=["median", "densest"], default="median",
                   help="where a label sits: coordinate-wise median of the cluster (default) or its densest spot")
    p.add_argument("--repel", choices=["vertical", "both"], default="vertical",
                   help="how overlapping labels are pushed apart: vertically only (default) or along either axis")
    p.add_argument("--point-scale", type=float, default=1.0, help="multiply all marker sizes by this factor")
    p.add_argument("--hide-dropped", action="store_true",
                   help="draw clusters that were not kept as rubrics like unassigned points (light gray, no label)")
    p.add_argument("--no-rotate", action="store_true", help="keep UMAP axes as they are (default: PCA-rotate so the long axis is horizontal)")
    args = p.parse_args()

    clusters = json.loads((V / "hdbscan/clusters_mcs100_leaf_ms10.json").read_text())
    L = np.asarray(clusters["assignments"]); Z = np.load(V / "hdbscan/umap_2d_s0.npy")
    if not args.no_rotate:  # rotate so the direction of largest spread is horizontal
        Zc = Z - Z.mean(axis=0)
        _, _, vt = np.linalg.svd(Zc, full_matrices=False)
        Z = Zc @ vt.T
    ordered = sorted(clusters["clusters"], key=lambda c: -c["n_turns"])
    rank_to_cid = {i + 1: c["cluster"] for i, c in enumerate(ordered)}
    rubrics = json.loads((V / "rubrics_final.json").read_text())["rubrics"]

    anchor_of = (lambda P: densest(P)) if args.label_anchor == "densest" else (lambda P: np.median(P, axis=0))
    fig, ax = plt.subplots(figsize=(args.width, args.height), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    in_rubric = np.zeros(len(L), bool)
    labels = []  # (x, y, text, colour)
    for k, rb in enumerate(rubrics):
        colour = PALETTE[k % len(PALETTE)]
        for r in rb["clusters"]:
            idx = np.where(L == rank_to_cid[r])[0]
            in_rubric[idx] = True
            ax.scatter(Z[idx, 0], Z[idx, 1], s=6 * args.point_scale ** 2, color=colour, alpha=0.75, linewidths=0, rasterized=True, zorder=3)
            labels.append([*anchor_of(Z[idx]), rb["name"], colour])
    dropped_labels = []
    for r, name in ([] if args.hide_dropped else DROPPED.items()):
        idx = np.where(L == rank_to_cid[r])[0]
        in_rubric[idx] = True
        ax.scatter(Z[idx, 0], Z[idx, 1], s=6 * args.point_scale ** 2, color=DROPPED_GRAY, alpha=0.7, linewidths=0, rasterized=True, zorder=2)
        dropped_labels.append([*anchor_of(Z[idx]), name, None])
    labels += dropped_labels
    rest = ~in_rubric
    ax.scatter(Z[rest, 0], Z[rest, 1], s=4 * args.point_scale ** 2, color=NOISE, alpha=0.5, linewidths=0, rasterized=True, zorder=1)

    # push overlapping labels apart, measured in inches on the final canvas
    pad = 0.03
    x0, x1 = Z[:, 0].min(), Z[:, 0].max(); y0, y1 = Z[:, 1].min(), Z[:, 1].max()
    ax.set_xlim(x0 - pad * (x1 - x0), x1 + pad * (x1 - x0)); ax.set_ylim(y0 - pad * (y1 - y0), y1 + pad * (y1 - y0))
    sx = args.width * 0.98 / (ax.get_xlim()[1] - ax.get_xlim()[0]); sy = args.height * 0.98 / (ax.get_ylim()[1] - ax.get_ylim()[0])
    pos = np.array([[l[0] * sx, l[1] * sy] for l in labels])
    char_in = args.fontsize / 72 * 0.62  # approx. width of one bold character
    widths = np.array([len(l[2]) * char_in + 0.35 for l in labels]); height_in = args.fontsize / 72 * 1.9
    anchor = pos.copy()
    for _ in range(600):
        moved = False
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                dx, dy = pos[j] - pos[i]
                need_x, need_y = (widths[i] + widths[j]) / 2, height_in
                if abs(dx) < need_x and abs(dy) < need_y:
                    # move each label a small step away from the other; vertical only, or the cheaper axis
                    if args.repel == "vertical" or (need_y - abs(dy)) * need_x < (need_x - abs(dx)) * need_y:
                        step = np.array([0.0, (need_y - abs(dy)) / 2 * (1 if dy >= 0 else -1)])
                    else:
                        step = np.array([(need_x - abs(dx)) / 2 * (1 if dx >= 0 else -1), 0.0])
                    pos[i] -= step * 0.5; pos[j] += step * 0.5; moved = True
        if not moved:
            break
    # keep every label fully inside the canvas
    x_lo = ax.get_xlim()[0] * sx; x_hi = ax.get_xlim()[1] * sx
    y_lo = ax.get_ylim()[0] * sy; y_hi = ax.get_ylim()[1] * sy
    for i in range(len(pos)):
        pos[i, 0] = np.clip(pos[i, 0], x_lo + widths[i] / 2 + 0.05, x_hi - widths[i] / 2 - 0.05)
        pos[i, 1] = np.clip(pos[i, 1], y_lo + height_in / 2 + 0.05, y_hi - height_in / 2 - 0.05)
    pos = pos / np.array([sx, sy])
    for (x, y), (_, _, text, colour) in zip(pos, labels):
        if colour is None:  # dropped cluster: gray box, white text
            ax.text(x, y, text, ha="center", va="center", fontsize=args.fontsize, fontweight="bold", color=SURFACE, zorder=5,
                    bbox=dict(boxstyle="round,pad=0.25,rounding_size=0.6", fc=DROPPED_GRAY, ec=DROPPED_GRAY, lw=1.2, alpha=0.95))
        else:
            ax.text(x, y, text, ha="center", va="center", fontsize=args.fontsize, fontweight="bold", color=INK, zorder=5,
                    bbox=dict(boxstyle="round,pad=0.25,rounding_size=0.6", fc=SURFACE, ec=colour, lw=1.2, alpha=0.95))

    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    fig.subplots_adjust(0.01, 0.01, 0.99, 0.99)
    fig.savefig(args.out, facecolor=SURFACE)
    fig.savefig(args.out.with_suffix(".png"), dpi=170, facecolor=SURFACE)
    print(f"-> {args.out} (+ .png)")


if __name__ == "__main__":
    main()
