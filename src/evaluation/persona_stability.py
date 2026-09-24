#!/usr/bin/env python3
"""Test whether future user vectors return to their historical persona cluster."""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from user_identifiability import RUBRIC_IDS, split_profiles, wilson_interval


def fit_kmeans(x: np.ndarray, k: int, seed: int, n_init: int = 20) -> tuple[np.ndarray, np.ndarray, float]:
    """Small deterministic NumPy k-means implementation."""
    best = None
    for initialization in range(n_init):
        rng = np.random.default_rng(seed + initialization)
        centroids = x[rng.choice(len(x), size=k, replace=False)].copy()
        labels = np.zeros(len(x), dtype=int)
        for _ in range(300):
            distances = np.linalg.norm(x[:, None, :] - centroids[None, :, :], axis=2)
            new_labels = distances.argmin(axis=1)
            new_centroids = np.vstack([
                x[new_labels == cluster].mean(axis=0)
                if np.any(new_labels == cluster) else x[rng.integers(len(x))]
                for cluster in range(k)
            ])
            if np.array_equal(labels, new_labels) and np.allclose(centroids, new_centroids):
                labels, centroids = new_labels, new_centroids
                break
            labels, centroids = new_labels, new_centroids
        inertia = float(np.sum((x - centroids[labels]) ** 2))
        if best is None or inertia < best[2]:
            best = labels.copy(), centroids.copy(), inertia
    return best


def silhouette(x: np.ndarray, labels: np.ndarray) -> float:
    pairwise = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)
    values = []
    for i, label in enumerate(labels):
        same = labels == label
        same[i] = False
        if not same.any():
            values.append(0.0)
            continue
        a = float(pairwise[i, same].mean())
        b = min(float(pairwise[i, labels == other].mean()) for other in np.unique(labels) if other != label)
        values.append((b - a) / max(a, b) if max(a, b) else 0.0)
    return float(np.mean(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-vectors", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--method", choices=["mean", "recent", "maxtie"], default="mean")
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-permutations", type=int, default=10000)
    args = parser.parse_args()

    score_cols = [f"score_{args.method}_{rid}" for rid in RUBRIC_IDS]
    vectors = pd.read_csv(args.session_vectors)
    sessions = pd.read_parquet(
        os.path.join(args.data_dir, "sessions.parquet"),
        columns=["session_id", "created_at"],
    )
    vectors = vectors.merge(sessions, on="session_id", how="left")
    vectors["created_at"] = pd.to_datetime(vectors.created_at, utc=True)
    vectors = vectors.dropna(subset=["created_at"])
    users, references, queries, n_reference, n_query = split_profiles(
        vectors, score_cols, args.train_fraction
    )

    # Fit every transformation and cluster using historical data only.
    center = references.mean(axis=0)
    scale = references.std(axis=0)
    scale[scale == 0] = 1.0
    reference_z = (references - center) / scale
    query_z = (queries - center) / scale
    sweep_rows = []
    max_k = min(args.k_max, len(users) - 1)
    for k in range(args.k_min, max_k + 1):
        labels, centroids, inertia = fit_kmeans(reference_z, k, args.seed)
        sweep_rows.append({
            "k": k,
            "silhouette": silhouette(reference_z, labels),
            "inertia": inertia,
        })
    sweep = pd.DataFrame(sweep_rows)
    chosen_k = args.k if args.k is not None else int(sweep.loc[sweep.silhouette.idxmax(), "k"])
    if chosen_k < 2 or chosen_k >= len(users):
        raise SystemExit("k must be at least 2 and smaller than the number of users")

    historical_cluster, centroids, _ = fit_kmeans(reference_z, chosen_k, args.seed)
    distances = np.linalg.norm(query_z[:, None, :] - centroids[None, :, :], axis=2)
    future_cluster = distances.argmin(axis=1)
    stable = future_cluster == historical_cluster

    rows = []
    for i, user in enumerate(users):
        own = int(historical_cluster[i])
        own_distance = float(distances[i, own])
        other_distance = float(np.delete(distances[i], own).min())
        rows.append({
            "user_id": user,
            "n_reference_sessions": int(n_reference[i]),
            "n_future_sessions": int(n_query[i]),
            "historical_cluster": own,
            "future_nearest_cluster": int(future_cluster[i]),
            "same_cluster": bool(stable[i]),
            "own_centroid_distance": own_distance,
            "nearest_other_centroid_distance": other_distance,
            "centroid_margin": other_distance - own_distance,
        })
    per_user = pd.DataFrame(rows)

    successes = int(stable.sum())
    low, high = wilson_interval(successes, len(users))
    historical_props = np.bincount(historical_cluster, minlength=chosen_k) / len(users)
    future_props = np.bincount(future_cluster, minlength=chosen_k) / len(users)
    chance_agreement = float(np.dot(historical_props, future_props))
    rng = np.random.default_rng(args.seed)
    permuted_successes = np.array([
        np.sum(rng.permutation(historical_cluster) == future_cluster)
        for _ in range(args.n_permutations)
    ])
    p_value = float((1 + np.sum(permuted_successes >= successes)) / (args.n_permutations + 1))
    summary = pd.DataFrame([{
        "n_users": len(users),
        "k": chosen_k,
        "historical_silhouette": silhouette(reference_z, historical_cluster),
        "same_cluster_users": successes,
        "same_cluster_accuracy": successes / len(users),
        "same_cluster_ci95_low": low,
        "same_cluster_ci95_high": high,
        "chance_agreement_from_cluster_sizes": chance_agreement,
        "permutation_p_value": p_value,
        "median_centroid_margin": per_user.centroid_margin.median(),
    }])

    os.makedirs(args.out_dir, exist_ok=True)
    sweep.to_csv(os.path.join(args.out_dir, "k_selection_historical.csv"), index=False)
    per_user.to_csv(os.path.join(args.out_dir, "persona_stability_per_user.csv"), index=False)
    summary.to_csv(os.path.join(args.out_dir, "persona_stability_summary.csv"), index=False)
    print("Future-vector persona stability")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nCluster sizes (historical):")
    print(pd.Series(historical_cluster).value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
