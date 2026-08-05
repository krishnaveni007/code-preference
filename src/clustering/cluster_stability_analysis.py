#!/usr/bin/env python3
"""
How few users would you need to recover ~the same k=6 centroids found on
the full sample?

Method: cluster the FULL sample once as the reference. Then, for a range
of smaller subsample sizes, repeatedly draw random subsamples (without
replacement), re-run the SAME clustering (same k, same preprocessing),
match each subsample cluster to its nearest reference cluster (k-means
labels are arbitrary between runs -- a matching step via the Hungarian
algorithm is required before any "how close" comparison means anything),
and measure the distance between matched centroids.

Reports, per subsample size:
  - mean/median matched-centroid distance to the reference (lower = more
    stable / more similar to what the full sample found)
  - the fraction of draws in which EVERY reference cluster got matched to
    some subsample cluster with at least 1 member (a reference cluster
    that's small in the full sample, like a 3-user cluster, can simply
    fail to appear in a subsample by chance -- this is tracked
    separately from centroid distance, since "no member sampled" is a
    different failure mode than "centroid drifted")

Usage:
    python cluster_stability_analysis.py \
        --session-vectors chat_session_vectors_full100.csv \
        --out-dir ./stability_out \
        --k 6 --row-zscore \
        --sample-sizes 20,30,40,50,60,70,80,94 --n-draws 50
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

RUBRIC_IDS = [f"R{str(i).zfill(2)}" for i in range(1, 15)]


def build_profiles(sessions: pd.DataFrame) -> pd.DataFrame:
    cols = [f"score_mean_{rid}" for rid in RUBRIC_IDS]
    profile = sessions.groupby("user_id")[cols].mean()
    profile.columns = RUBRIC_IDS
    return profile


def prep(profile: pd.DataFrame, row_zscore: bool) -> pd.DataFrame:
    if row_zscore:
        row_std = profile.std(axis=1).replace(0, np.nan)
        out = profile.sub(profile.mean(axis=1), axis=0).div(row_std, axis=0)
        return out.dropna()
    return profile.sub(profile.mean(axis=1), axis=0)  # at least center


def match_clusters(ref_centroids: np.ndarray, other_centroids: np.ndarray):
    """Hungarian matching between two sets of centroids (in the same
    standardized space) by Euclidean distance. Returns (row_ind, col_ind,
    dists) -- row_ind indexes ref_centroids, col_ind indexes
    other_centroids, dists is the matched-pair distance for each."""
    k_ref, k_other = len(ref_centroids), len(other_centroids)
    cost = np.zeros((k_ref, k_other))
    for i in range(k_ref):
        for j in range(k_other):
            cost[i, j] = np.linalg.norm(ref_centroids[i] - other_centroids[j])
    row_ind, col_ind = linear_sum_assignment(cost)
    dists = cost[row_ind, col_ind]
    return row_ind, col_ind, dists


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session-vectors", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--row-zscore", action="store_true")
    ap.add_argument("--sample-sizes", default="20,30,40,50,60,70,80,94")
    ap.add_argument("--n-draws", type=int, default=50,
                    help="random subsamples per sample size")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    sessions = pd.concat([pd.read_csv(p) for p in args.session_vectors.split(",")],
                        ignore_index=True)
    profile = build_profiles(sessions)
    prepped = prep(profile, args.row_zscore)
    n_total = len(prepped)
    print(f"{n_total} users available after preprocessing "
         f"(row_zscore={args.row_zscore})")

    scaler = StandardScaler()
    X_full = scaler.fit_transform(prepped.values)
    user_ids = prepped.index.to_numpy()

    # ---------------------------------------------------------- reference
    ref_km = KMeans(n_clusters=args.k, n_init=10, random_state=args.seed).fit(X_full)
    ref_centroids = ref_km.cluster_centers_
    ref_sizes = pd.Series(ref_km.labels_).value_counts().sort_index()
    print(f"\nReference clustering (all {n_total} users), cluster sizes:")
    print(ref_sizes.to_string())
    smallest_cluster_frac = ref_sizes.min() / n_total
    print(f"smallest reference cluster: {ref_sizes.min()} users "
         f"({smallest_cluster_frac:.1%} of the sample)")

    # ---------------------------------------------------------- subsampling
    sizes = [int(s) for s in args.sample_sizes.split(",")]
    rng = np.random.default_rng(args.seed)
    rows = []

    for n in sizes:
        if n > n_total:
            print(f"skipping n={n} (exceeds available {n_total} users)")
            continue
        for draw in range(args.n_draws):
            idx = rng.choice(n_total, size=n, replace=False)
            X_sub = X_full[idx]
            if n <= args.k:
                continue  # can't fit k clusters with fewer points than k
            sub_km = KMeans(n_clusters=args.k, n_init=10,
                           random_state=args.seed + draw).fit(X_sub)
            sub_centroids = sub_km.cluster_centers_

            row_ind, col_ind, dists = match_clusters(ref_centroids, sub_centroids)
            # did every reference cluster get a matched subsample cluster
            # with actual members drawn from it? (all k always get SOME
            # match via Hungarian -- what matters is whether the ref
            # cluster's own members were even present in this subsample)
            ref_labels_full = ref_km.labels_
            sampled_ref_labels = set(ref_labels_full[idx])
            all_ref_clusters_represented = len(sampled_ref_labels) == args.k

            rows.append({
                "n": n, "draw": draw,
                "mean_matched_dist": float(dists.mean()),
                "max_matched_dist": float(dists.max()),
                "all_ref_clusters_represented": all_ref_clusters_represented,
            })

    result = pd.DataFrame(rows)
    result.to_csv(os.path.join(args.out_dir, "stability_per_draw.csv"), index=False)

    summary = result.groupby("n").agg(
        mean_dist_avg=("mean_matched_dist", "mean"),
        mean_dist_std=("mean_matched_dist", "std"),
        max_dist_avg=("max_matched_dist", "mean"),
        pct_draws_all_clusters_represented=("all_ref_clusters_represented", "mean"),
        n_draws=("draw", "count"),
    )
    summary.to_csv(os.path.join(args.out_dir, "stability_summary.csv"))
    pd.set_option("display.width", 160)
    print(f"\n=== stability summary (n_draws={args.n_draws} per size) ===")
    print(summary.to_string())

    print(f"\nNote: 'pct_draws_all_clusters_represented' is the fraction of "
         f"draws where every one of the {args.k} reference clusters had at "
         f"least one of ITS OWN members included in the subsample -- distinct "
         f"from centroid distance. A reference cluster with "
         f"{ref_sizes.min()} members ({smallest_cluster_frac:.1%} of the "
         f"sample) will be missing from many small subsamples purely by "
         f"chance, regardless of how real or fake that cluster is.")

    # ---------------------------------------------------------- plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].errorbar(summary.index, summary["mean_dist_avg"],
                     yerr=summary["mean_dist_std"], marker="o", color="#3B6FA0",
                     capsize=3)
    axes[0].set_xlabel("Subsample size (users)")
    axes[0].set_ylabel("Mean matched-centroid distance to reference")
    axes[0].set_title("Centroid stability vs. sample size")

    axes[1].plot(summary.index, summary["pct_draws_all_clusters_represented"] * 100,
                marker="o", color="#A6373D")
    axes[1].set_xlabel("Subsample size (users)")
    axes[1].set_ylabel("%% draws where all reference clusters had >=1 member")
    axes[1].set_title("Small-cluster representation vs. sample size")
    axes[1].set_ylim(-5, 105)

    fig.tight_layout()
    path = os.path.join(args.out_dir, "stability_curves.png")
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"\nwrote stability_per_draw.csv, stability_summary.csv, {path}")


if __name__ == "__main__":
    main()