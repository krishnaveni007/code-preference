#!/usr/bin/env python3
"""Cluster users with an exact, common number of sessions."""

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
    parser.add_argument("--exact-sessions", type=int, default=10)
    parser.add_argument("--method", choices=["mean", "recent", "maxtie"], default="mean")
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    score_cols = [f"score_{args.method}_{rid}" for rid in RUBRIC_IDS]
    sessions = pd.read_csv(args.session_vectors)
    counts = sessions.groupby("user_id").size()
    eligible = counts[counts == args.exact_sessions].index
    sessions = sessions[sessions.user_id.isin(eligible)]
    profiles = sessions.groupby("user_id")[score_cols].mean().fillna(0)
    profiles.columns = RUBRIC_IDS

    center = profiles.mean(axis=0).to_numpy(copy=True)
    scale = profiles.std(axis=0, ddof=0).to_numpy(copy=True)
    scale[scale == 0] = 1.0
    x = (profiles.to_numpy() - center) / scale

    rows = []
    for k in range(args.k_min, min(args.k_max, len(profiles) - 1) + 1):
        labels, centroids, inertia = fit_kmeans(x, k, args.seed)
        rows.append({"k": k, "silhouette": silhouette(x, labels), "inertia": inertia})
    sweep = pd.DataFrame(rows)
    chosen_k = args.k if args.k is not None else int(sweep.loc[sweep.silhouette.idxmax(), "k"])
    labels, centroids_z, inertia = fit_kmeans(x, chosen_k, args.seed)

    assignments = profiles.copy()
    assignments.insert(0, "n_sessions", args.exact_sessions)
    assignments.insert(0, "cluster", labels)
    centroids_raw = pd.DataFrame(
        centroids_z * scale + center,
        columns=RUBRIC_IDS,
        index=pd.Index(range(chosen_k), name="cluster"),
    )
    sizes = assignments.groupby("cluster").size().rename("n_users")
    centroid_z_frame = pd.DataFrame(centroids_z, columns=RUBRIC_IDS)
    defining_rows = []
    for cluster in range(chosen_k):
        ordered = centroid_z_frame.loc[cluster].sort_values(key=np.abs, ascending=False)
        for rank, (axis, value) in enumerate(ordered.head(5).items(), start=1):
            defining_rows.append({
                "cluster": cluster,
                "n_users": int(sizes.loc[cluster]),
                "rank": rank,
                "axis": axis,
                "centroid_z": value,
                "centroid_raw": centroids_raw.loc[cluster, axis],
            })

    os.makedirs(args.out_dir, exist_ok=True)
    sweep.to_csv(os.path.join(args.out_dir, "k_selection.csv"), index=False)
    assignments.to_csv(os.path.join(args.out_dir, "user_clusters.csv"))
    centroids_raw.to_csv(os.path.join(args.out_dir, "cluster_centroids.csv"))
    pd.DataFrame(defining_rows).to_csv(os.path.join(args.out_dir, "defining_axes.csv"), index=False)
    summary = pd.DataFrame([{
        "n_users": len(profiles),
        "sessions_per_user": args.exact_sessions,
        "k": chosen_k,
        "silhouette": silhouette(x, labels),
        "inertia": inertia,
        "cluster_sizes": ";".join(f"{c}:{int(n)}" for c, n in sizes.items()),
    }])
    summary.to_csv(os.path.join(args.out_dir, "cluster_summary.csv"), index=False)

    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nK selection")
    print(sweep.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nDefining axes (largest standardized centroid deviations)")
    print(pd.DataFrame(defining_rows).to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()
