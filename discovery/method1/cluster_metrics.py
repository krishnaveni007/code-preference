#!/usr/bin/env python3
"""Per-cluster and overall quality metrics for a k-means run produced by cluster_messages.py.

Per cluster:
  cohesion    mean cosine similarity of members to their own centroid (higher = tighter)
  separation  cosine distance from this centroid to the nearest other centroid (higher = more isolated)
  silhouette  mean silhouette of members, cosine metric (in [-1, 1]; ~0 = no better than a neighbour)
  top_user    share of the cluster contributed by its single most frequent user
Overall: silhouette, Davies-Bouldin, Calinski-Harabasz, plus a random-label baseline for silhouette.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                             silhouette_samples, silhouette_score)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/discovery/method1/archive/raw_message_kmeans/cluster_raw_dedup"))
    parser.add_argument("--positives", type=Path, default=Path("outputs/discovery/method1/gate/positives_dedup.jsonl"))
    parser.add_argument("--k", type=int, nargs="+", default=[20, 40, 80])
    parser.add_argument("--detail-k", type=int, default=40)
    args = parser.parse_args()

    X = np.load(args.out_dir / "embeddings.npy")
    rows = [json.loads(line) for line in args.positives.open()]
    users = np.array([r["user_id"] for r in rows])
    rng = np.random.default_rng(0)

    print(f"{'k':>3} {'silhouette':>11} {'random-label sil':>17} {'davies-bouldin':>15} {'calinski-harabasz':>18}")
    for k in args.k:
        data = json.loads((args.out_dir / f"clusters_k{k}.json").read_text())
        labels = np.asarray(data["assignments"])
        sil = silhouette_score(X, labels, metric="cosine")
        rand = silhouette_score(X, rng.permutation(labels), metric="cosine")
        db = davies_bouldin_score(X, labels)
        ch = calinski_harabasz_score(X, labels)
        print(f"{k:>3} {sil:>11.4f} {rand:>17.4f} {db:>15.3f} {ch:>18.1f}")

    k = args.detail_k
    data = json.loads((args.out_dir / f"clusters_k{k}.json").read_text())
    labels = np.asarray(data["assignments"])
    names = {c["cluster"]: c.get("name", "") for c in data["clusters"]}
    sil_s = silhouette_samples(X, labels, metric="cosine")
    cents = np.vstack([X[labels == c].mean(axis=0) for c in range(k)])
    cents /= np.linalg.norm(cents, axis=1, keepdims=True)
    cc = cents @ cents.T
    np.fill_diagonal(cc, -1)
    recs = []
    for c in range(k):
        idx = labels == c
        member_users = users[idx]
        top_share = pd.Series(member_users).value_counts().iloc[0] / idx.sum()
        recs.append({
            "cluster": c, "size": int(idx.sum()), "n_users": int(len(set(member_users))),
            "top_user": round(float(top_share), 2),
            "cohesion": round(float((X[idx] @ cents[c]).mean()), 3),
            "separation": round(float(1 - cc[c].max()), 3),
            "silhouette": round(float(sil_s[idx].mean()), 3),
            "name": names[c][:90],
        })
    df = pd.DataFrame(recs).sort_values("silhouette", ascending=False)
    df.to_csv(args.out_dir / f"metrics_k{k}.csv", index=False)
    # Baseline for cohesion: mean cosine of a random point to a random centroid-sized group's mean.
    rand_lab = rng.permutation(labels)
    rand_coh = np.mean([(X[rand_lab == c] @ (X[rand_lab == c].mean(0) / np.linalg.norm(X[rand_lab == c].mean(0)))).mean()
                        for c in range(k)])
    print(f"\nk={k} per cluster (sorted by silhouette). Random-label cohesion baseline = {rand_coh:.3f}\n")
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 90)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
