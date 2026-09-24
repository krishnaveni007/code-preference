#!/usr/bin/env python3
"""Describe support/confidence and validate thresholds on future sessions."""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd


RUBRIC_IDS = [f"R{i:02d}" for i in range(1, 15)]


def aggregate(group: pd.DataFrame, axis: str) -> tuple[int, int, int, float, int]:
    high = int(group[f"high_turns_{axis}"].sum())
    low = int(group[f"low_turns_{axis}"].sum())
    support = high + low
    confidence = (max(high, low) + 1) / (support + 2)
    return high, low, support, confidence, int(np.sign(high - low))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-vectors", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    args = parser.parse_args()

    vectors = pd.read_csv(args.session_vectors)
    dates = pd.read_parquet(
        os.path.join(args.data_dir, "sessions.parquet"),
        columns=["session_id", "created_at"],
    )
    vectors = vectors.merge(dates, on="session_id", how="left")
    vectors["created_at"] = pd.to_datetime(vectors.created_at, utc=True)
    vectors = vectors.dropna(subset=["created_at"])

    all_rows, validation_rows = [], []
    for user, group in vectors.groupby("user_id"):
        group = group.sort_values(["created_at", "session_id"])
        cut = min(max(int(np.floor(len(group) * args.train_fraction)), 1), len(group) - 1)
        train, test = group.iloc[:cut], group.iloc[cut:]
        for axis in RUBRIC_IDS:
            ah, al, ass, ac, ad = aggregate(group, axis)
            all_rows.append({
                "user_id": user, "axis": axis, "high": ah, "low": al,
                "support": ass, "confidence": ac, "direction": ad,
            })
            h, low, support, confidence, direction = aggregate(train, axis)
            th, tl, test_support, _, test_direction = aggregate(test, axis)
            validation_rows.append({
                "user_id": user, "axis": axis, "train_high": h, "train_low": low,
                "train_support": support, "train_confidence": confidence,
                "train_direction": direction, "test_high": th, "test_low": tl,
                "test_support": test_support, "test_direction": test_direction,
                "direction_match": direction == test_direction if direction and test_direction else np.nan,
            })
    all_values = pd.DataFrame(all_rows)
    validation = pd.DataFrame(validation_rows)

    supports = [1, 2, 3, 5, 8, 10, 15, 20]
    confidences = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    grid_rows = []
    evaluable = validation.test_direction.ne(0)
    for minimum_support in supports:
        for minimum_confidence in confidences:
            selected = (
                (validation.train_support >= minimum_support)
                & (validation.train_confidence >= minimum_confidence)
                & validation.train_direction.ne(0)
            )
            tested = validation[selected & evaluable]
            selected_users = validation.loc[selected, "user_id"].nunique()
            grid_rows.append({
                "min_support": minimum_support,
                "min_confidence": minimum_confidence,
                "selected_user_axes": int(selected.sum()),
                "selected_users": selected_users,
                "mean_axes_per_selected_user": selected.sum() / selected_users if selected_users else np.nan,
                "future_evaluable_user_axes": len(tested),
                "future_direction_accuracy": tested.direction_match.mean(),
                "future_coverage_of_all_evaluable": len(tested) / evaluable.sum(),
            })
    grid = pd.DataFrame(grid_rows)

    per_axis = all_values.groupby("axis").agg(
        zero_support=("support", lambda values: int((values == 0).sum())),
        median_support=("support", "median"),
        support_p75=("support", lambda values: values.quantile(0.75)),
        median_confidence=("confidence", "median"),
        confidence_p75=("confidence", lambda values: values.quantile(0.75)),
    ).reset_index()

    os.makedirs(args.out_dir, exist_ok=True)
    all_values.to_csv(os.path.join(args.out_dir, "all_user_axis_support_confidence.csv"), index=False)
    validation.to_csv(os.path.join(args.out_dir, "chronological_direction_validation.csv"), index=False)
    grid.to_csv(os.path.join(args.out_dir, "threshold_grid.csv"), index=False)
    per_axis.to_csv(os.path.join(args.out_dir, "support_confidence_by_axis.csv"), index=False)
    print("All-data support/confidence")
    print(all_values[["support", "confidence"]].describe(
        percentiles=[.1, .25, .5, .75, .9, .95, .99]
    ).to_string(float_format=lambda value: f"{value:.3f}"))
    print("\nThreshold grid")
    print(grid.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()
