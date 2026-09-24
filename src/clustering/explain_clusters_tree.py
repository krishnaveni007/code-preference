#!/usr/bin/env python3
"""Fit a small CART-style surrogate tree to explain cluster assignments."""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd


RUBRIC_IDS = [f"R{i:02d}" for i in range(1, 15)]


def gini(y: np.ndarray) -> float:
    if not len(y):
        return 0.0
    proportions = np.bincount(y) / len(y)
    return float(1 - np.sum(proportions**2))


def build_tree(x, y, indices, depth, max_depth, min_leaf):
    counts = np.bincount(y[indices], minlength=int(y.max()) + 1)
    prediction = int(counts.argmax())
    node = {"n": len(indices), "counts": counts.tolist(), "prediction": prediction}
    if depth >= max_depth or np.count_nonzero(counts) == 1 or len(indices) < 2 * min_leaf:
        return node
    parent_impurity = gini(y[indices])
    best = None
    for feature in range(x.shape[1]):
        values = x[indices, feature]
        observed = np.unique(values[np.isfinite(values)])
        if len(observed) < 2:
            continue
        thresholds = (observed[:-1] + observed[1:]) / 2
        for threshold in thresholds:
            for missing_left in (True, False):
                missing = ~np.isfinite(values)
                left_mask = (values <= threshold) | (missing if missing_left else False)
                left, right = indices[left_mask], indices[~left_mask]
                if len(left) < min_leaf or len(right) < min_leaf:
                    continue
                impurity = (len(left) * gini(y[left]) + len(right) * gini(y[right])) / len(indices)
                gain = parent_impurity - impurity
                if best is None or gain > best[0]:
                    best = gain, feature, float(threshold), missing_left, left, right
    if best is None or best[0] <= 0:
        return node
    gain, feature, threshold, missing_left, left, right = best
    node.update({
        "axis": RUBRIC_IDS[feature], "threshold": threshold,
        "missing_goes": "left" if missing_left else "right", "gini_gain": gain,
        "left": build_tree(x, y, left, depth + 1, max_depth, min_leaf),
        "right": build_tree(x, y, right, depth + 1, max_depth, min_leaf),
    })
    return node


def predict_one(row, node):
    while "axis" in node:
        value = row[RUBRIC_IDS.index(node["axis"])]
        go_left = node["missing_goes"] == "left" if not np.isfinite(value) else value <= node["threshold"]
        node = node["left"] if go_left else node["right"]
    return node["prediction"]


def tree_lines(node, prefix=""):
    if "axis" not in node:
        return [f"{prefix}Predict cluster {node['prediction']} (n={node['n']}, counts={node['counts']})"]
    missing = node["missing_goes"]
    lines = [f"{prefix}{node['axis']} <= {node['threshold']:+.3f} (missing -> {missing})"]
    lines += tree_lines(node["left"], prefix + "  yes: ")
    lines += tree_lines(node["right"], prefix + "  no:  ")
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user-clusters", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--min-leaf", type=int, default=5)
    args = parser.parse_args()
    frame = pd.read_csv(args.user_clusters)
    preferences = frame[RUBRIC_IDS]
    row_mean = preferences.mean(axis=1)
    row_std = preferences.std(axis=1, ddof=0).replace(0, 1)
    x = preferences.sub(row_mean, axis=0).div(row_std, axis=0).to_numpy()
    y = frame.cluster.to_numpy(dtype=int)
    tree = build_tree(x, y, np.arange(len(y)), 0, args.max_depth, args.min_leaf)
    predicted = np.array([predict_one(row, tree) for row in x])
    result = frame[["user_id", "cluster"]].copy()
    result["tree_prediction"] = predicted
    result["correct"] = predicted == y
    summary = pd.DataFrame([{
        "n_users": len(frame), "n_clusters": len(np.unique(y)),
        "max_depth": args.max_depth, "min_leaf": args.min_leaf,
        "training_accuracy": float((predicted == y).mean()),
    }])
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "tree.json"), "w") as handle:
        json.dump(tree, handle, indent=2)
    with open(os.path.join(args.out_dir, "tree.txt"), "w") as handle:
        handle.write("\n".join(tree_lines(tree)) + "\n")
    result.to_csv(os.path.join(args.out_dir, "tree_predictions.csv"), index=False)
    summary.to_csv(os.path.join(args.out_dir, "tree_summary.csv"), index=False)
    print(summary.to_string(index=False))
    print("\n" + "\n".join(tree_lines(tree)))


if __name__ == "__main__":
    main()
