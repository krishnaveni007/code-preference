#!/usr/bin/env python3
"""Multivariate user discriminability for SWE-chat session vectors.

For each session, compare its distance to another session from the same user
with its distance to a session from a different user. The statistic is the
probability that the same-user distance is smaller, averaged so every anchor
user and every comparison user receive equal weight.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd


RUBRIC_IDS = [f"R{i:02d}" for i in range(1, 15)]


def calculate(distances: np.ndarray, labels: np.ndarray) -> tuple[float, np.ndarray]:
    """Return the equally user-weighted statistic and per-user values."""
    unique, inverse = np.unique(labels, return_inverse=True)
    counts = np.bincount(inverse)
    user_d = []
    for group, user in enumerate(unique):
        indices = np.flatnonzero(inverse == group)
        if len(indices) < 2:
            continue
        different = inverse != group
        # Each comparison user has total weight 1/(N-1), regardless of how
        # many sessions that user contributes.
        between_weights = 1.0 / (len(unique) - 1) / counts[inverse[different]]
        session_d = []
        for index in indices:
            same_indices = indices[indices != index]
            within = distances[index, same_indices]
            between = distances[index, different]
            comparisons = within[:, None] - between[None, :]
            wins_by_between_session = np.mean(
                (comparisons < 0) + 0.5 * (comparisons == 0), axis=0
            )
            session_d.append(float(wins_by_between_session @ between_weights))
        user_d.append(float(np.mean(session_d)))
    return float(np.mean(user_d)), np.asarray(user_d)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-vectors", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--method", choices=["mean", "recent", "maxtie"], default="mean")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--n-permutations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--exact-sessions", type=int, default=None,
        help="Restrict to users having exactly this many sessions",
    )
    args = parser.parse_args()

    columns = [f"score_{args.method}_{rid}" for rid in RUBRIC_IDS]
    frame = pd.read_csv(args.session_vectors)
    missing = [column for column in ["user_id", *columns] if column not in frame]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")
    frame = frame.dropna(subset=["user_id"]).copy()
    eligible = frame.groupby("user_id").size()
    if args.exact_sessions is not None:
        eligible = eligible[eligible == args.exact_sessions].index
    else:
        eligible = eligible[eligible >= 2].index
    frame = frame[frame.user_id.isin(eligible)].reset_index(drop=True)
    if len(eligible) < 2:
        raise SystemExit("Need at least two eligible users")

    raw = frame[columns].fillna(0).to_numpy(float)
    center = raw.mean(axis=0)
    scale = raw.std(axis=0)
    scale[scale == 0] = 1.0
    x = (raw - center) / scale
    labels = frame.user_id.astype(str).to_numpy()
    distances = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)
    statistic, per_user_d = calculate(distances, labels)

    rng = np.random.default_rng(args.seed)
    boot_d = np.array([
        rng.choice(per_user_d, size=len(per_user_d), replace=True).mean()
        for _ in range(args.n_bootstrap)
    ])
    permuted = np.array([
        calculate(distances, rng.permutation(labels))[0]
        for _ in range(args.n_permutations)
    ])
    permutation_p = (1 + np.sum(permuted >= statistic)) / (args.n_permutations + 1)

    users = np.unique(labels)
    per_user = pd.DataFrame({
        "user_id": users,
        "n_sessions": [np.sum(labels == user) for user in users],
        "discriminability": per_user_d,
    })
    summary = pd.DataFrame([{
        "n_users": len(users),
        "n_sessions": len(frame),
        "n_dimensions": len(columns),
        "method": args.method,
        "exact_sessions_filter": args.exact_sessions,
        "multivariate_discriminability": statistic,
        "discriminability_ci95_low": np.percentile(boot_d, 2.5),
        "discriminability_ci95_high": np.percentile(boot_d, 97.5),
        "chance_discriminability": 0.5,
        "permutation_mean": permuted.mean(),
        "permutation_p_value": permutation_p,
    }])

    os.makedirs(args.out_dir, exist_ok=True)
    per_user.to_csv(os.path.join(args.out_dir, "multivariate_discriminability_per_user.csv"), index=False)
    summary.to_csv(os.path.join(args.out_dir, "multivariate_discriminability_summary.csv"), index=False)
    pd.DataFrame({"permuted_discriminability": permuted}).to_csv(
        os.path.join(args.out_dir, "multivariate_discriminability_null.csv"), index=False
    )
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nD=0.5 means chance; D=1 means every same-user session distance is")
    print("smaller than every different-user session distance.")


if __name__ == "__main__":
    main()
