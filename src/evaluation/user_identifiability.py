#!/usr/bin/env python3
"""Measure whether held-out preference vectors retrieve the correct user.

For every user, the chronologically early sessions form a reference profile and
the later sessions form an independent query profile.  The query is compared
with every user's reference profile.  A user is a top-1 match when their own
reference is closer than all competing references.

The primary distance is standardized Euclidean distance.  Each rubric axis is
scaled by its standard deviation across the reference profiles so that a
high-variance axis does not dominate merely because of its numerical scale.
"""

from __future__ import annotations

import argparse
import math
import os

import numpy as np
import pandas as pd


RUBRIC_IDS = [f"R{i:02d}" for i in range(1, 15)]


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return np.nan, np.nan
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def binomial_survival(successes: int, total: int, probability: float) -> float:
    """Exact P[X >= successes] for X ~ Binomial(total, probability)."""
    return float(sum(
        math.comb(total, k) * probability**k * (1 - probability) ** (total - k)
        for k in range(successes, total + 1)
    ))


def split_profiles(
    frame: pd.DataFrame, score_cols: list[str], train_fraction: float
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    users, references, queries, n_reference, n_query = [], [], [], [], []
    for user_id, group in frame.groupby("user_id", sort=True):
        group = group.sort_values(["created_at", "session_id"])
        if len(group) < 2:
            continue
        cut = int(np.floor(len(group) * train_fraction))
        cut = min(max(cut, 1), len(group) - 1)
        users.append(user_id)
        references.append(group.iloc[:cut][score_cols].fillna(0).mean().to_numpy(float))
        queries.append(group.iloc[cut:][score_cols].fillna(0).mean().to_numpy(float))
        n_reference.append(cut)
        n_query.append(len(group) - cut)
    return (
        users,
        np.vstack(references),
        np.vstack(queries),
        np.asarray(n_reference),
        np.asarray(n_query),
    )


def retrieval_table(
    users: list[str], references: np.ndarray, queries: np.ndarray,
    n_reference: np.ndarray, n_query: np.ndarray,
) -> pd.DataFrame:
    scale = references.std(axis=0, ddof=1)
    scale[~np.isfinite(scale) | (scale == 0)] = 1.0
    distances = np.sqrt((((queries[:, None, :] - references[None, :, :]) / scale) ** 2).mean(axis=2))

    rows = []
    for i, user_id in enumerate(users):
        order = np.argsort(distances[i], kind="stable")
        rank = int(np.flatnonzero(order == i)[0] + 1)
        own = float(distances[i, i])
        impostor_distances = np.delete(distances[i], i)
        nearest_other = float(impostor_distances.min())
        nearest_other_index = int(np.arange(len(users))[np.arange(len(users)) != i][np.argmin(impostor_distances)])
        rows.append({
            "user_id": user_id,
            "n_reference_sessions": int(n_reference[i]),
            "n_query_sessions": int(n_query[i]),
            "own_distance": own,
            "nearest_other_distance": nearest_other,
            "distance_margin": nearest_other - own,
            "distance_ratio": own / nearest_other if nearest_other > 0 else np.nan,
            "rank": rank,
            "top1_match": rank == 1,
            "top5_match": rank <= 5,
            "reciprocal_rank": 1.0 / rank,
            "rank_percentile": 1.0 - (rank - 1) / max(len(users) - 1, 1),
            "nearest_other_user": users[nearest_other_index],
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-vectors", required=True)
    parser.add_argument("--data-dir", required=True, help="Directory containing sessions.parquet")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--method", choices=["mean", "recent", "maxtie"], default="mean")
    parser.add_argument("--train-fraction", type=float, default=0.7)
    args = parser.parse_args()
    if not 0 < args.train_fraction < 1:
        raise SystemExit("--train-fraction must be between 0 and 1")

    score_cols = [f"score_{args.method}_{rid}" for rid in RUBRIC_IDS]
    vectors = pd.read_csv(args.session_vectors)
    missing = [column for column in ["user_id", "session_id", *score_cols] if column not in vectors]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")
    sessions = pd.read_parquet(
        os.path.join(args.data_dir, "sessions.parquet"), columns=["session_id", "created_at"]
    )
    vectors = vectors.merge(sessions, on="session_id", how="left")
    vectors["created_at"] = pd.to_datetime(vectors["created_at"], utc=True)
    vectors = vectors.dropna(subset=["created_at"])

    users, references, queries, n_reference, n_query = split_profiles(
        vectors, score_cols, args.train_fraction
    )
    if len(users) < 2:
        raise SystemExit("Need at least two users with at least two dated sessions")
    per_user = retrieval_table(users, references, queries, n_reference, n_query)

    n = len(per_user)
    top1 = int(per_user.top1_match.sum())
    top5 = int(per_user.top5_match.sum())
    top1_low, top1_high = wilson_interval(top1, n)
    summary = pd.DataFrame([{
        "n_users": n,
        "n_dimensions": len(score_cols),
        "method": args.method,
        "train_fraction": args.train_fraction,
        "chance_top1": 1 / n,
        "top1_matches": top1,
        "top1_accuracy": top1 / n,
        "top1_ci95_low": top1_low,
        "top1_ci95_high": top1_high,
        "top1_chance_p_value": binomial_survival(top1, n, 1 / n),
        "chance_top5": min(5 / n, 1.0),
        "top5_matches": top5,
        "top5_accuracy": top5 / n,
        "top5_chance_p_value": binomial_survival(top5, n, min(5 / n, 1.0)),
        "mean_reciprocal_rank": per_user.reciprocal_rank.mean(),
        "median_rank": per_user["rank"].median(),
        "median_own_distance": per_user.own_distance.median(),
        "median_nearest_other_distance": per_user.nearest_other_distance.median(),
    }])

    os.makedirs(args.out_dir, exist_ok=True)
    per_user.to_csv(os.path.join(args.out_dir, "identifiability_per_user.csv"), index=False)
    summary.to_csv(os.path.join(args.out_dir, "identifiability_summary.csv"), index=False)
    print("User retrieval from held-out future sessions")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nInterpretation: top1_match means the user's future vector was closest to")
    print("their own earlier profile. It is evidence of discriminability, not proof of")
    print("real-world identity or a permanent property of that person.")


if __name__ == "__main__":
    main()
