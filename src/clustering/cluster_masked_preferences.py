#!/usr/bin/env python3
"""Cluster preference means while masking axes with low directional confidence."""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd


RUBRIC_IDS = [f"R{i:02d}" for i in range(1, 15)]


def masked_kmeans(x: np.ndarray, k: int, seed: int, n_init: int = 30):
    rng_master = np.random.default_rng(seed)
    best = None
    for _ in range(n_init):
        rng = np.random.default_rng(rng_master.integers(2**32))
        seeds = rng.choice(len(x), size=k, replace=False)
        centroids = np.where(np.isfinite(x[seeds]), x[seeds], 0.0)
        labels = np.zeros(len(x), dtype=int)
        for _ in range(300):
            distances = np.array([
                np.nanmean((x - centroid) ** 2, axis=1) for centroid in centroids
            ]).T
            new_labels = distances.argmin(axis=1)
            new_centroids = centroids.copy()
            for cluster in range(k):
                members = x[new_labels == cluster]
                if len(members):
                    observed = np.isfinite(members).any(axis=0)
                    for axis in np.flatnonzero(observed):
                        new_centroids[cluster, axis] = np.nanmean(members[:, axis])
            if np.array_equal(labels, new_labels) and np.allclose(centroids, new_centroids):
                labels, centroids = new_labels, new_centroids
                break
            labels, centroids = new_labels, new_centroids
        inertia = float(sum(np.nanmean((x[i] - centroids[labels[i]]) ** 2) for i in range(len(x))))
        if best is None or inertia < best[2]:
            best = labels.copy(), centroids.copy(), inertia
    return best


def masked_pairwise(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(x)
    distances = np.full((n, n), np.nan)
    overlap = np.zeros((n, n), dtype=int)
    for i in range(n):
        common = np.isfinite(x[i]) & np.isfinite(x)
        overlap[i] = common.sum(axis=1)
        valid = overlap[i] > 0
        distances[i, valid] = np.sqrt(np.nansum((x[valid] - x[i]) ** 2, axis=1) / overlap[i, valid])
    return distances, overlap


def masked_silhouette(distances: np.ndarray, labels: np.ndarray) -> float:
    scores = []
    for i, label in enumerate(labels):
        same = (labels == label) & (np.arange(len(labels)) != i)
        within = distances[i, same]
        within = within[np.isfinite(within)]
        if not len(within):
            continue
        other_means = []
        for other in np.unique(labels[labels != label]):
            values = distances[i, labels == other]
            values = values[np.isfinite(values)]
            if len(values):
                other_means.append(values.mean())
        if not other_means:
            continue
        a, b = within.mean(), min(other_means)
        scores.append((b - a) / max(a, b) if max(a, b) else 0.0)
    return float(np.mean(scores)) if scores else np.nan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-vectors", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--min-support", type=int, default=0)
    parser.add_argument("--min-retained-axes", type=int, default=2)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--row-zscore", action="store_true",
        help="Within each user, center and scale retained preferences so clustering uses relative pattern rather than overall level/spread",
    )
    args = parser.parse_args()

    sessions = pd.read_csv(args.session_vectors)
    rows = []
    for user, group in sessions.groupby("user_id", sort=True):
        row = {"user_id": user}
        for axis in RUBRIC_IDS:
            high = group[f"high_turns_{axis}"].sum()
            low = group[f"low_turns_{axis}"].sum()
            support = high + low
            confidence = (max(high, low) + 1) / (support + 2)
            preference = group[f"score_mean_{axis}"].mean()
            row[axis] = (
                preference
                if confidence >= args.threshold and support >= args.min_support
                else np.nan
            )
            row[f"confidence_{axis}"] = confidence
            row[f"support_{axis}"] = support
        rows.append(row)
    full = pd.DataFrame(rows).set_index("user_id")
    retained = full[RUBRIC_IDS].notna().sum(axis=1)
    eligible = retained[retained >= args.min_retained_axes].index
    preferences = full.loc[eligible, RUBRIC_IDS]

    if args.row_zscore:
        row_mean = preferences.mean(axis=1)
        row_std = preferences.std(axis=1, ddof=0).replace(0, 1)
        transformed = preferences.sub(row_mean, axis=0).div(row_std, axis=0)
        x = transformed.to_numpy()
        axis_mean = pd.Series(0.0, index=RUBRIC_IDS)
        axis_std = pd.Series(1.0, index=RUBRIC_IDS)
    else:
        # Standardize each axis using only reliable observations on that axis.
        axis_mean = preferences.mean(axis=0)
        axis_std = preferences.std(axis=0, ddof=0).replace(0, 1)
        transformed = preferences.sub(axis_mean).div(axis_std)
        x = transformed.to_numpy()
    pairwise, overlap = masked_pairwise(x)
    sweep_rows = []
    models = {}
    for k in range(2, 9):
        labels, centroids, inertia = masked_kmeans(x, k, args.seed)
        sil = masked_silhouette(pairwise, labels)
        sweep_rows.append({"k": k, "silhouette": sil, "inertia": inertia})
        models[k] = labels, centroids, inertia
    sweep = pd.DataFrame(sweep_rows)
    chosen_k = args.k if args.k is not None else int(sweep.loc[sweep.silhouette.idxmax(), "k"])
    labels, centroids_z, inertia = models.get(chosen_k, masked_kmeans(x, chosen_k, args.seed))
    centroids_output = centroids_z * axis_std.to_numpy() + axis_mean.to_numpy()

    assignments = preferences.copy()
    assignments.insert(0, "n_retained_axes", retained.loc[eligible])
    assignments.insert(0, "cluster", labels)
    centroid_frame = pd.DataFrame(centroids_output, columns=RUBRIC_IDS)
    sizes = assignments.groupby("cluster").size()
    diagnostics = assignments.groupby("cluster").n_retained_axes.agg(["size", "mean", "median"])
    summary = pd.DataFrame([{
        "n_users_total": len(full), "n_users_clustered": len(eligible),
        "n_users_excluded": len(full) - len(eligible), "threshold": args.threshold,
        "min_support": args.min_support,
        "min_retained_axes": args.min_retained_axes, "k": chosen_k,
        "row_zscore": args.row_zscore,
        "silhouette": masked_silhouette(pairwise, labels),
        "median_pairwise_shared_axes": float(np.median(overlap[np.triu_indices(len(overlap), 1)])),
        "cluster_sizes": ";".join(f"{i}:{int(n)}" for i, n in sizes.items()),
    }])

    os.makedirs(args.out_dir, exist_ok=True)
    summary.to_csv(os.path.join(args.out_dir, "cluster_summary.csv"), index=False)
    sweep.to_csv(os.path.join(args.out_dir, "k_selection.csv"), index=False)
    assignments.to_csv(os.path.join(args.out_dir, "user_clusters.csv"))
    centroid_frame.to_csv(os.path.join(args.out_dir, "cluster_centroids.csv"), index_label="cluster")
    full.to_csv(os.path.join(args.out_dir, "masked_user_preferences.csv"))
    diagnostics.to_csv(os.path.join(args.out_dir, "cluster_diagnostics.csv"))
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nK selection")
    print(sweep.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nCluster retained-axis diagnostics")
    print(diagnostics.to_string(float_format=lambda value: f"{value:.2f}"))
    print("\nCentroids (within-user relative z-scores)" if args.row_zscore else "\nCentroids (original preference-score scale)")
    print(centroid_frame.to_string(float_format=lambda value: f"{value:+.3f}"))


if __name__ == "__main__":
    main()
