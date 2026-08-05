#!/usr/bin/env python3
"""
Prediction Strength (Tibshirani & Walther, 2005) -- tests whether a
clustering structure GENERALIZES to unseen users, not just whether it's
stable under resampling from the same pool (that's cluster_stability_
analysis.py; this is a different, stronger question).

Method, repeated over many random train/test splits and swept across a
range of k:
  1. Split users into TRAIN and TEST halves.
  2. Cluster TRAIN -> get train centroids.
  3. Cluster TEST *independently* on its own -> the test set's own
     "ground truth" grouping.
  4. ALSO assign every test user to their nearest TRAIN centroid (this
     is what "the train clustering predicts for new users" means).
  5. For every pair of test users that the test set's OWN clustering put
     in the same group, check whether the TRAIN-based assignment also
     put them in the same group. The fraction that agree, taking the
     WORST (minimum) rate across the test set's own clusters, is the
     prediction strength for that split.

Published interpretation: prediction strength >= 0.8-0.9 means that k is
well-supported by out-of-sample evidence; below that, the clustering at
that k does not reliably generalize, however good it looked in-sample
(e.g. via silhouette on the full data).

Usage:
    python cluster_prediction_strength.py \
        --session-vectors chat_session_vectors_full100.csv \
        --out-dir ./prediction_strength_out \
        --row-zscore --k-min 2 --k-max 8 --n-splits 50
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
    return profile.sub(profile.mean(axis=1), axis=0)


def prediction_strength(X_train, X_test, k, seed):
    if len(X_train) <= k or len(X_test) <= k:
        return np.nan
    km_train = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X_train)
    km_test = KMeans(n_clusters=k, n_init=10, random_state=seed + 1).fit(X_test)

    test_labels_via_train = km_train.predict(X_test)
    test_labels_own = km_test.labels_

    ps_per_cluster = []
    for c in range(k):
        members = np.where(test_labels_own == c)[0]
        m = len(members)
        if m < 2:
            continue  # a singleton or empty test-cluster contributes no pairs
        # vectorized pairwise co-membership check under the train-based assignment
        labels_via_train_members = test_labels_via_train[members]
        same = labels_via_train_members[:, None] == labels_via_train_members[None, :]
        n_pairs = m * (m - 1) / 2
        n_agree = (same.sum() - m) / 2  # subtract diagonal, halve for unordered pairs
        ps_per_cluster.append(n_agree / n_pairs)

    return min(ps_per_cluster) if ps_per_cluster else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session-vectors", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--row-zscore", action="store_true")
    ap.add_argument("--k-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=8)
    ap.add_argument("--n-splits", type=int, default=50,
                    help="random train/test splits per k")
    ap.add_argument("--train-frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    sessions = pd.concat([pd.read_csv(p) for p in args.session_vectors.split(",")],
                        ignore_index=True)
    profile = build_profiles(sessions)
    prepped = prep(profile, args.row_zscore)
    n_total = len(prepped)
    print(f"{n_total} users available (row_zscore={args.row_zscore})")

    scaler = StandardScaler()
    X = scaler.fit_transform(prepped.values)

    rng = np.random.default_rng(args.seed)
    rows = []
    for k in range(args.k_min, args.k_max + 1):
        for split in range(args.n_splits):
            idx = rng.permutation(n_total)
            n_train = int(n_total * args.train_frac)
            train_idx, test_idx = idx[:n_train], idx[n_train:]
            ps = prediction_strength(X[train_idx], X[test_idx], k, args.seed + split)
            rows.append({"k": k, "split": split, "prediction_strength": ps})

    result = pd.DataFrame(rows)
    result.to_csv(os.path.join(args.out_dir, "prediction_strength_per_split.csv"), index=False)

    summary = result.groupby("k")["prediction_strength"].agg(
        ["mean", "median", "std", "count"])
    summary.to_csv(os.path.join(args.out_dir, "prediction_strength_summary.csv"))
    print(f"\n=== Prediction strength by k (n_splits={args.n_splits}) ===")
    print(summary.to_string())

    well_supported = summary[summary["mean"] >= 0.8]
    print(f"\nk values with mean prediction strength >= 0.8 (published threshold "
         f"for 'generalizes well'): {well_supported.index.tolist() or 'NONE'}")
    if well_supported.empty:
        print("No k in the tested range clears the standard 0.8 threshold. This "
             "means the clustering does not reliably predict held-out users at "
             "any of these k, regardless of how it looked via in-sample "
             "silhouette on the full data -- a real, informative result, not "
             "a failed analysis.")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(summary.index, summary["mean"], yerr=summary["std"],
               marker="o", color="#3B6FA0", capsize=3)
    ax.axhline(0.8, color="#A6373D", linestyle="--", linewidth=1, label="0.8 (well-supported)")
    ax.axhline(0.9, color="#D9A441", linestyle="--", linewidth=1, label="0.9 (strongly supported)")
    ax.set_xlabel("k (number of clusters)")
    ax.set_ylabel("Prediction strength (held-out, higher = generalizes better)")
    ax.set_title(f"Cluster generalization by k (n={n_total} users, "
                f"{args.n_splits} random train/test splits each)")
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    path = os.path.join(args.out_dir, "prediction_strength_by_k.png")
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"\nwrote prediction_strength_per_split.csv, prediction_strength_summary.csv, {path}")


if __name__ == "__main__":
    main()