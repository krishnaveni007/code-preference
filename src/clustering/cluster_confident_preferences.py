#!/usr/bin/env python3
"""Cluster users using only directionally confident preference axes."""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

EVALUATION_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "evaluation"))
if EVALUATION_DIR not in sys.path:
    sys.path.insert(0, EVALUATION_DIR)
from persona_stability import fit_kmeans, silhouette  # noqa: E402
from user_identifiability import RUBRIC_IDS  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-vectors", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    sessions = pd.read_csv(args.session_vectors)
    users = sorted(sessions.user_id.unique())
    vectors, detail_rows = [], []
    for user in users:
        group = sessions[sessions.user_id == user]
        values = []
        for axis in RUBRIC_IDS:
            high = int(group[f"high_turns_{axis}"].sum())
            low = int(group[f"low_turns_{axis}"].sum())
            support = high + low
            confidence = (max(high, low) + 1) / (support + 2)
            direction = int(np.sign(high - low))
            retained = confidence > args.threshold and direction != 0
            value = direction * confidence if retained else 0.0
            values.append(value)
            detail_rows.append({
                "user_id": user, "axis": axis, "high_count": high,
                "low_count": low, "support": support, "confidence": confidence,
                "direction": direction, "retained": retained, "cluster_value": value,
            })
        vectors.append(values)
    x = np.asarray(vectors)
    detail = pd.DataFrame(detail_rows)
    active_counts = (x != 0).sum(axis=1)

    sweep_rows = []
    for k in range(args.k_min, min(args.k_max, len(users) - 1) + 1):
        labels, centroids, inertia = fit_kmeans(x, k, args.seed)
        sweep_rows.append({"k": k, "silhouette": silhouette(x, labels), "inertia": inertia})
    sweep = pd.DataFrame(sweep_rows)
    chosen_k = args.k if args.k is not None else int(sweep.loc[sweep.silhouette.idxmax(), "k"])
    labels, centroids, inertia = fit_kmeans(x, chosen_k, args.seed)

    assignments = pd.DataFrame(x, index=pd.Index(users, name="user_id"), columns=RUBRIC_IDS)
    assignments.insert(0, "n_retained_axes", active_counts)
    assignments.insert(0, "cluster", labels)
    centroid_frame = pd.DataFrame(
        centroids, index=pd.Index(range(chosen_k), name="cluster"), columns=RUBRIC_IDS
    )
    cluster_diagnostics = assignments.groupby("cluster").agg(
        n_users=("n_retained_axes", "size"),
        mean_retained_axes=("n_retained_axes", "mean"),
        median_retained_axes=("n_retained_axes", "median"),
    )
    summary = pd.DataFrame([{
        "n_users": len(users), "threshold": args.threshold, "k": chosen_k,
        "silhouette": silhouette(x, labels), "inertia": inertia,
        "users_with_zero_retained_axes": int((active_counts == 0).sum()),
        "mean_retained_axes": float(active_counts.mean()),
        "median_retained_axes": float(np.median(active_counts)),
    }])

    os.makedirs(args.out_dir, exist_ok=True)
    summary.to_csv(os.path.join(args.out_dir, "cluster_summary.csv"), index=False)
    sweep.to_csv(os.path.join(args.out_dir, "k_selection.csv"), index=False)
    assignments.to_csv(os.path.join(args.out_dir, "user_clusters.csv"))
    centroid_frame.to_csv(os.path.join(args.out_dir, "cluster_centroids.csv"))
    detail.to_csv(os.path.join(args.out_dir, "user_axis_confidence.csv"), index=False)
    cluster_diagnostics.to_csv(os.path.join(args.out_dir, "cluster_diagnostics.csv"))

    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nK selection")
    print(sweep.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nCluster diagnostics")
    print(cluster_diagnostics.to_string(float_format=lambda value: f"{value:.2f}"))
    print("\nRetention by axis")
    print(detail.groupby("axis").retained.agg(["sum", "mean"]).to_string(float_format=lambda value: f"{value:.2%}"))


if __name__ == "__main__":
    main()
