#!/usr/bin/env python3
"""Does each k-means cluster reappear when the random seed changes?

Refits k-means with extra seeds on the same embeddings, matches clusters to the
reference run (seed 0) one-to-one by centroid cosine (Hungarian assignment), and
reports for every reference cluster the member overlap (Jaccard) with its match.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


def centroids(X: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    C = np.vstack([X[labels == c].mean(axis=0) for c in range(k)])
    return C / np.linalg.norm(C, axis=1, keepdims=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/discovery/method1/archive/v1_prefers_x_over_y/cluster_statements"))
    parser.add_argument("--positives", type=Path, default=Path("outputs/discovery/method1/archive/v1_prefers_x_over_y/statements_flat.jsonl"))
    parser.add_argument("--k", type=int, default=40)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--stable-jaccard", type=float, default=0.5)
    args = parser.parse_args()

    X = np.load(args.out_dir / "embeddings.npy")
    ref = json.loads((args.out_dir / f"clusters_k{args.k}.json").read_text())
    ref_labels = np.asarray(ref["assignments"])
    names = {c["cluster"]: c.get("name", "") for c in ref["clusters"]}
    users = np.array([json.loads(l)["user_id"] for l in args.positives.open()])
    k = args.k
    ref_C = centroids(X, ref_labels, k)

    table = pd.DataFrame({
        "cluster": range(k),
        "size": [int((ref_labels == c).sum()) for c in range(k)],
        "n_users": [len(set(users[ref_labels == c])) for c in range(k)],
    })
    for seed in args.seeds:
        labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
        C = centroids(X, labels, k)
        sim = ref_C @ C.T
        rows, cols = linear_sum_assignment(-sim)
        match = dict(zip(rows, cols))
        jac, cos = [], []
        for c in range(k):
            a = set(np.where(ref_labels == c)[0]); b = set(np.where(labels == match[c])[0])
            jac.append(len(a & b) / len(a | b)); cos.append(float(sim[c, match[c]]))
        table[f"jaccard_s{seed}"] = np.round(jac, 2)
        table[f"cos_s{seed}"] = np.round(cos, 3)
        print(f"seed {seed} vs seed 0: adjusted Rand index = {adjusted_rand_score(ref_labels, labels):.3f}, "
              f"mean Jaccard of matched clusters = {np.mean(jac):.2f}")
    jac_cols = [f"jaccard_s{s}" for s in args.seeds]
    table["min_jaccard"] = table[jac_cols].min(axis=1)
    table["stable"] = table["min_jaccard"] >= args.stable_jaccard
    table["name"] = [names[c][:80] for c in range(k)]
    table = table.sort_values("min_jaccard", ascending=False)
    table.to_csv(args.out_dir / f"stability_k{k}.csv", index=False)
    print(f"\n{int(table['stable'].sum())}/{k} clusters have Jaccard >= {args.stable_jaccard} with their match in every seed\n")
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 80)
    print(table.drop(columns=[c for c in table.columns if c.startswith("cos_")]).to_string(index=False))


if __name__ == "__main__":
    main()
