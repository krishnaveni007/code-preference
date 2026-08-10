#!/usr/bin/env python3
"""Validate preference vectors against independent future active turns.

For each user, preference-active turns are ordered by session creation time and
turn number.  At each requested k, the estimate is the mean vector over the
first k active turns and the validation target is the mean vector over the next
fixed number of active turns.  The two vectors never share observations.

This is deliberately different from comparing a k-turn vector with a final
vector that contains those same k turns: overlap mechanically inflates
agreement.
"""

import argparse
import os

import numpy as np
import pandas as pd


RUBRIC_IDS = [f"R{i:02d}" for i in range(1, 15)]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0:
        return np.nan
    return float(np.dot(a, b) / denominator)


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if total == 0:
        return np.nan, np.nan
    p = successes / total
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    half_width = (
        z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator
    )
    return center - half_width, center + half_width


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--turn-vectors", required=True)
    parser.add_argument("--data-dir", required=True, help="directory containing sessions.parquet")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--k-values", default="5,10,15,20,30,40,50,60",
        help="comma-separated active-turn counts used to build the estimate",
    )
    parser.add_argument(
        "--validation-turns", type=int, default=20,
        help="number of immediately following active turns in the independent target",
    )
    parser.add_argument(
        "--max-distance", type=float, default=0.1,
        help="largest cosine distance counted as agreement",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    k_values = sorted({int(value) for value in args.k_values.split(",")})
    score_cols = [f"score_{rubric_id}" for rubric_id in RUBRIC_IDS]

    turns = pd.read_csv(args.turn_vectors)
    missing = [column for column in score_cols if column not in turns.columns]
    if missing:
        raise SystemExit(f"missing score columns: {missing}")

    sessions = pd.read_parquet(
        os.path.join(args.data_dir, "sessions.parquet"),
        columns=["session_id", "created_at"],
    )
    turns = turns.merge(sessions, on="session_id", how="left")
    turns[score_cols] = turns[score_cols].fillna(0)
    turns["preference_active"] = turns[score_cols].ne(0).any(axis=1)
    active = turns[turns["preference_active"]].copy()
    active["created_at"] = pd.to_datetime(active["created_at"], utc=True)
    active = active.dropna(subset=["created_at"]).sort_values(
        ["user_id", "created_at", "session_id", "turn_number"]
    )

    rows = []
    for user_id, user_turns in active.groupby("user_id"):
        matrix = user_turns[score_cols].to_numpy(dtype=float)
        for k in k_values:
            validation_end = k + args.validation_turns
            if len(matrix) < validation_end:
                continue
            estimate = matrix[:k].mean(axis=0)
            future = matrix[k:validation_end].mean(axis=0)
            similarity = cosine_similarity(estimate, future)
            rows.append(
                {
                    "user_id": user_id,
                    "k_estimation_turns": k,
                    "n_validation_turns": args.validation_turns,
                    "n_active_turns_available": len(matrix),
                    "cosine_similarity": similarity,
                    "cosine_distance": 1 - similarity if not np.isnan(similarity) else np.nan,
                    "within_distance": (
                        1 - similarity <= args.max_distance
                        if not np.isnan(similarity)
                        else False
                    ),
                }
            )

    per_user = pd.DataFrame(rows)
    per_user.to_csv(os.path.join(args.out_dir, "future_turn_validation_per_user.csv"), index=False)

    summary_rows = []
    for k in k_values:
        group = per_user[
            (per_user["k_estimation_turns"] == k)
            & per_user["cosine_similarity"].notna()
        ]
        n_users = len(group)
        successes = int(group["within_distance"].sum())
        low, high = wilson_interval(successes, n_users)
        summary_rows.append(
            {
                "k_estimation_turns": k,
                "n_validation_turns": args.validation_turns,
                "n_users_eligible": n_users,
                "mean_cosine_similarity": group["cosine_similarity"].mean(),
                "median_cosine_similarity": group["cosine_similarity"].median(),
                "pct_within_distance": successes / n_users if n_users else np.nan,
                "ci95_low": low,
                "ci95_high": high,
                "max_cosine_distance": args.max_distance,
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(args.out_dir, "future_turn_validation_summary.csv"), index=False)
    print(f"Preference-active turns: {len(active):,} across {active['user_id'].nunique()} users")
    print(f"Independent validation block: next {args.validation_turns} active turns")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nWrote results to {args.out_dir}")


if __name__ == "__main__":
    main()
