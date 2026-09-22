#!/usr/bin/env python3
"""Do users group by which rubrics they express? Hierarchical clustering of user rubric
proportions, silhouette vs. a shuffled baseline, and a heatmap ordered by the clustering."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.metrics import silhouette_score

V = Path("outputs/discovery/method1/v2/user_profiles")
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUES = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ks", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    p.add_argument("--shuffles", type=int, default=200)
    args = p.parse_args()

    P = pd.read_csv(V / "user_rubric_proportions.csv", index_col=0)
    counts = pd.read_csv(V / "user_rubric_counts.csv", index_col=0).loc[P.index]
    X = P.to_numpy(); n, d = X.shape
    Z = linkage(X, method="ward")
    rng = np.random.default_rng(0)

    print(f"{n} users x {d} rubrics (proportions). Ward linkage. Silhouette (euclidean), real vs. shuffled-rubric baseline:")
    print(f"{'k':>2} {'real':>6} {'shuffled mean':>14} {'shuffled 95%':>13}")
    best = None
    for k in args.ks:
        lab = fcluster(Z, k, criterion="maxclust")
        real = silhouette_score(X, lab)
        base = []
        for _ in range(args.shuffles):
            Xs = np.vstack([rng.permutation(row) for row in X])  # shuffle rubric identities within each user
            base.append(silhouette_score(Xs, fcluster(linkage(Xs, "ward"), k, criterion="maxclust")))
        print(f"{k:>2} {real:>6.3f} {np.mean(base):>14.3f} {np.quantile(base, .95):>13.3f}")
        if best is None or real > best[1]:
            best = (k, real, lab)
    k, sil, lab = best

    print(f"\nbest k = {k} (silhouette {sil:.3f}). Cluster profiles: mean proportion per rubric, top-3 rubrics per cluster")
    prof = pd.DataFrame(X, index=P.index, columns=P.columns).groupby(lab).mean()
    for c, row in prof.iterrows():
        members = (lab == c).sum(); turns = counts.total_turns[lab == c].sum()
        top = row.sort_values(ascending=False).head(3)
        print(f"  cluster {c}: {members:2d} users, {turns:4d} turns | " + ", ".join(f"{r} {v:.2f}" for r, v in top.items()))
    pd.DataFrame({"cluster": lab, "total_turns": counts.total_turns}, index=P.index).to_csv(V / "user_clusters.csv")

    # heatmap ordered by dendrogram leaves
    order = dendrogram(Z, no_plot=True)["leaves"]
    fig = plt.figure(figsize=(13, 0.32 * n + 4.5), facecolor=SURFACE)
    ax = fig.add_axes([0.26, 0.04, 0.58, 0.80]); ax.set_facecolor(SURFACE)
    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    im = ax.imshow(X[order], aspect="auto", cmap=cmap, vmin=0, vmax=max(0.5, X.max()))
    ax.set_xticks(range(d)); ax.set_xticklabels([c.replace("_", " ") for c in P.columns], rotation=45, ha="left", fontsize=9, color=INK2)
    ax.xaxis.tick_top()
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"{u[:14]}  ({int(counts.total_turns[u])})" for u in P.index[order]], fontsize=7.5, color=INK2, family="DejaVu Sans Mono")
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=0)
    # cluster boundaries
    lab_ordered = lab[order]
    for i in range(1, n):
        if lab_ordered[i] != lab_ordered[i - 1]:
            ax.axhline(i - 0.5, color=INK, lw=1.2)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02); cb.outline.set_visible(False)
    cb.set_label("share of the user's turns", color=INK2, fontsize=9); cb.ax.tick_params(labelsize=8, colors=INK2)
    fig.text(0.26, 0.985, f"Which rubrics does each user express?  {n} users with >= 10 turns in rubric clusters", fontsize=13, color=INK, va="top")
    fig.text(0.26, 0.968, f"Rows: users ordered by Ward clustering (lines = the {k}-cluster cut); label = user id (turns). "
                          f"Cell: share of that user's turns falling in the rubric.", fontsize=9, color=INK2, va="top")
    fig.savefig(V / "user_rubric_heatmap.png", dpi=170, facecolor=SURFACE)
    print(f"\n-> {V / 'user_rubric_heatmap.png'}")


if __name__ == "__main__":
    main()
