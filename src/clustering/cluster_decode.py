#!/usr/bin/env python3
"""
Cluster users from a scored_pairs.jsonl file (a different dataset/rubric
than SWE-chat -- for code COMPLETION rather than chat-based coding, but
the same TERNARY {-1, 0, +1} scoring convention -- aggregating
hierarchically: checkpoint (turn-equivalent) -> outcome_id
(session-equivalent) -> user.

Each JSONL line looks like:
  {"user": "...", "outcome_id": "...", "checkpoint": 10,
   "pair_id": "...", "vector": {"Instruction Fidelity": 0, ...}}

Axes are read directly from the "vector" dict keys, not hardcoded, so
this works for any rubric with any number of axes.

SCALE: values are -1/0/+1, the same convention as the SWE-chat chat
scoring -- 0 most likely means "no signal / not evaluated" rather than
"this quality is genuinely absent," so the SAME zero-inflation and
coverage concerns from that pipeline apply here: an axis's raw mean can
be pulled toward 0 just because it rarely fires, independent of whether
its DIRECTION is consistent when it does. This script reports coverage
(fraction of checkpoints/outcomes/users with a nonzero rating) per axis
for exactly that reason -- check it before trusting a low centroid
value as "this cluster doesn't care about this axis" rather than "this
axis rarely fires for anyone."

Hierarchical averaging (NOT a flat mean over every checkpoint):
  1. average checkpoints within each (user, outcome_id) -> one vector
     per user-outcome pair
  2. average those outcome-level vectors within each user -> one final
     profile per user
  This gives every outcome equal weight regardless of how many
  checkpoints it has, rather than letting outcomes with more checkpoints
  dominate a flat mean the way a naive groupby(user).mean() would.

No evidence text exists in this dataset, unlike the SWE-chat pipeline.
In its place, for each cluster's defining axes, this script lists the
(user, outcome_id, checkpoint, pair_id) of a few of that cluster's most
extreme raw rows, so specific examples can still be looked up later if
needed.

Usage:
    python cluster_scored_pairs.py \
        --input scored_pairs.jsonl \
        --out-dir ./cluster_scored_pairs_out \
        --row-zscore
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def load_jsonl(path: str) -> pd.DataFrame:
    rows = []
    axes_seen = set()
    with open(path) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  WARNING: skipping unparseable line {line_no}: {e}")
                continue
            vec = d.get("vector", {})
            axes_seen.update(vec.keys())
            row = {"user": d.get("user"), "outcome_id": d.get("outcome_id"),
                  "checkpoint": d.get("checkpoint"), "pair_id": d.get("pair_id")}
            row.update(vec)
            rows.append(row)
    df = pd.DataFrame(rows)
    return df, sorted(axes_seen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="scored_pairs.jsonl")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--row-zscore", action="store_true",
                    help="remove each user's own mean AND std across axes "
                        "before clustering (shape only, not level/spread). "
                        "Recommended -- see cluster_user_profiles.py's "
                        "notes on why level/spread confounds cluster "
                        "structure if left in.")
    ap.add_argument("--k-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=8)
    ap.add_argument("--k", type=int, default=None, help="force a specific k")
    ap.add_argument("--z-threshold", type=float, default=0.3)
    ap.add_argument("--max-defining-axes", type=int, default=6)
    ap.add_argument("--n-examples", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"loading {args.input} ...")
    raw, axes = load_jsonl(args.input)
    print(f"{len(raw)} rows (checkpoint-level), {raw['user'].nunique()} users, "
         f"{raw['outcome_id'].nunique()} outcomes, {len(axes)} axes: {axes}")

    # sanity check on the value scale, and coverage (nonzero rate) per axis --
    # the same zero-inflation concern from the SWE-chat pipeline applies here
    print(f"\nvalue distribution per axis (checking this is really -1/0/1, "
         f"not something else):")
    unexpected_values = set()
    for ax in axes:
        vc = raw[ax].value_counts(normalize=True).sort_index()
        unexpected_values.update(v for v in vc.index if v not in (-1, 0, 1))
        print(f"  {ax}: {dict(vc.round(3))}")
    if unexpected_values:
        print(f"  WARNING: values outside {{-1,0,1}} found: {unexpected_values} "
             f"-- check the rubric encoding before trusting anything below.")

    print(f"\ncoverage per axis (fraction of checkpoint-level rows with a "
         f"NONZERO rating -- low coverage means this axis rarely fires for "
         f"anyone, and its centroid values should be read with that in mind):")
    coverage = (raw[axes] != 0).mean().sort_values()
    for ax, c in coverage.items():
        flag = " <-- LOW COVERAGE" if c < 0.4 else ""
        print(f"  {ax}: {c:.1%}{flag}")

    # ---------------------------------------------------- hierarchical averaging
    # step 1: checkpoint -> outcome
    outcome_level = raw.groupby(["user", "outcome_id"])[axes].mean().reset_index()
    n_checkpoints_per_outcome = raw.groupby(["user", "outcome_id"]).size()
    print(f"\nstep 1: {len(raw)} checkpoint rows -> {len(outcome_level)} "
         f"user-outcome rows (checkpoints/outcome: mean="
         f"{n_checkpoints_per_outcome.mean():.2f}, "
         f"median={n_checkpoints_per_outcome.median():.0f}, "
         f"max={n_checkpoints_per_outcome.max()})")

    # step 2: outcome -> user
    profile = outcome_level.groupby("user")[axes].mean()
    n_outcomes_per_user = outcome_level.groupby("user").size()
    print(f"step 2: {len(outcome_level)} user-outcome rows -> "
         f"{len(profile)} user profiles (outcomes/user: mean="
         f"{n_outcomes_per_user.mean():.2f}, "
         f"median={n_outcomes_per_user.median():.0f}, "
         f"max={n_outcomes_per_user.max()})")
    profile.to_csv(os.path.join(args.out_dir, "user_profiles.csv"))

    # -------------------------------------------------------- intensity diagnostic
    intensity = profile.mean(axis=1)
    intensity_df = pd.DataFrame({"intensity": intensity, "n_outcomes": n_outcomes_per_user})
    corr = intensity_df["intensity"].corr(intensity_df["n_outcomes"])
    print(f"\ncorrelation(per-user mean score across axes, n_outcomes) = {corr:.3f}")
    print("(a strong correlation here is the same data-volume/level confound "
         "found in the SWE-chat clustering -- worth checking before trusting "
         "an unstandardized run)")

    # ------------------------------------------------------------- preprocessing
    if args.row_zscore:
        row_std = profile.std(axis=1).replace(0, np.nan)
        n_zero_var = int(row_std.isna().sum())
        prepped = profile.sub(profile.mean(axis=1), axis=0).div(row_std, axis=0)
        if n_zero_var:
            print(f"\nWARNING: {n_zero_var} users have zero variance across "
                 f"their own axes -- dropping them (can't z-score).")
            prepped = prepped.dropna()
        print(f"ROW Z-SCORED: level and spread removed, shape only.")
    else:
        prepped = profile.sub(profile.mean(axis=1), axis=0)
        print(f"ROW-CENTERED only (pass --row-zscore to also remove spread).")

    n_users = len(prepped)
    scaler = StandardScaler()
    X = scaler.fit_transform(prepped.values)

    # ------------------------------------------------------------------- k selection
    if args.k is None:
        rows = []
        for k in range(args.k_min, min(args.k_max, n_users - 1) + 1):
            km = KMeans(n_clusters=k, n_init=10, random_state=args.seed).fit(X)
            sil = silhouette_score(X, km.labels_)
            rows.append({"k": k, "silhouette": round(float(sil), 4),
                        "inertia": round(float(km.inertia_), 2)})
        sweep = pd.DataFrame(rows)
        sweep.to_csv(os.path.join(args.out_dir, "k_selection.csv"), index=False)
        print(f"\n=== k selection ===")
        print(sweep.to_string(index=False))
        best_k = int(sweep.loc[sweep["silhouette"].idxmax(), "k"])
        print(f"best k by silhouette: {best_k} "
             f"(reminder: silhouette < ~0.25 conventionally means 'no "
             f"substantial structure found', not just 'this k beat the "
             f"others' -- check the magnitude, not just the ranking)")
    else:
        best_k = args.k
        print(f"\nusing forced k={best_k}")

    km = KMeans(n_clusters=best_k, n_init=10, random_state=args.seed).fit(X)
    prepped = prepped.assign(cluster=km.labels_)
    prepped[["cluster"]].to_csv(os.path.join(args.out_dir, "user_clusters.csv"))
    print(f"\ncluster sizes:")
    print(prepped["cluster"].value_counts().sort_index().to_string())

    # ------------------------------------------------------------- characterization
    profile_c = profile.loc[prepped.index].assign(cluster=km.labels_)
    grand_mean, grand_std = profile.mean(), profile.std()
    centroid_raw = profile_c.groupby("cluster")[axes].mean()
    centroid_z = (centroid_raw - grand_mean) / grand_std
    centroid_raw.to_csv(os.path.join(args.out_dir, "cluster_centroids.csv"))

    defining = {}
    print(f"\n=== cluster characterization (threshold mode, z>={args.z_threshold}) ===")
    for c in sorted(profile_c["cluster"].unique()):
        n = (profile_c["cluster"] == c).sum()
        z = centroid_z.loc[c].sort_values(key=lambda s: s.abs(), ascending=False)
        top = z[z.abs() >= args.z_threshold].head(args.max_defining_axes)
        if len(top) == 0:
            top = z.head(1)
        defining[c] = top
        print(f"\nCluster {c} (n={n} users, {len(top)} defining axes):")
        for ax, zval in top.items():
            direction = "high" if zval > 0 else "low"
            print(f"  {ax}: z={zval:+.2f} ({direction}, relative to the population), "
                 f"raw mean={centroid_raw.loc[c, ax]:+.3f}")

    # ------------------------------------------------------------------ heatmap
    fig, ax = plt.subplots(figsize=(max(8, len(axes) * 0.7), max(3, 0.6 * best_k + 1)))
    im = ax.imshow(centroid_raw.values, cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(axes)))
    ax.set_xticklabels(axes, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(best_k))
    ax.set_yticklabels([f"Cluster {c} (n={(profile_c['cluster']==c).sum()})"
                        for c in sorted(profile_c["cluster"].unique())], fontsize=9)
    for i in range(centroid_raw.shape[0]):
        for j in range(centroid_raw.shape[1]):
            v = centroid_raw.values[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7,
                   color="white" if abs(v) > 0.5 else "black")
    ax.set_title(f"Cluster centroids ({best_k} clusters), mean score per axis")
    cbar = fig.colorbar(im, ax=ax, ticks=[-1, 0, 1], shrink=0.6)
    cbar.ax.set_yticklabels(["low (-1)", "no signal (0)", "high (+1)"])
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "cluster_centroids_heatmap.png"),
               bbox_inches="tight", dpi=200)
    plt.close(fig)

    # --------------------------------------------------------------------- PCA
    pca = PCA(n_components=2, random_state=args.seed)
    coords = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = plt.get_cmap("tab10")
    for c in sorted(prepped["cluster"].unique()):
        mask = prepped["cluster"].values == c
        ax.scatter(coords[mask, 0], coords[mask, 1], label=f"Cluster {c}",
                  color=cmap(c % 10), s=50, edgecolor="black", linewidth=0.3)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%})")
    ax.set_title("User profiles, PCA projection colored by cluster")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "cluster_pca.png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"\nPCA: PC1+PC2 explain {sum(pca.explained_variance_ratio_[:2]):.0%} of variance")

    # ---------------------------------------------------- example raw rows (no text evidence)
    report_lines = [f"# Scored-pairs user clusters (k={best_k})\n"]
    print(f"\n=== example raw rows per cluster's defining axes (no evidence "
         f"text in this dataset -- pair_id lets you look up the source later) ===")
    for c in sorted(profile_c["cluster"].unique()):
        cluster_users = profile_c[profile_c["cluster"] == c].index.tolist()
        n = len(cluster_users)
        report_lines.append(f"## Cluster {c} (n={n} users)\n")
        print(f"\n--- Cluster {c} (n={n}) ---")
        for ax, zval in defining[c].items():
            direction_val = 1 if zval > 0 else -1
            report_lines.append(f"**{ax}**: z={zval:+.2f} (relative to population), "
                               f"raw mean={centroid_raw.loc[c, ax]:+.3f}, "
                               f"direction={direction_val:+d}\n")
            print(f"  {ax} (z={zval:+.2f}, direction={direction_val:+d}):")
            examples = raw[raw["user"].isin(cluster_users) & (raw[ax] == direction_val)]
            examples = examples[["user", "outcome_id", "checkpoint", "pair_id"]] \
                .sample(min(args.n_examples, len(examples)), random_state=args.seed) \
                if len(examples) else examples
            for _, row in examples.iterrows():
                print(f"    - {row['pair_id']}")
                report_lines.append(f"- `{row['pair_id']}`\n")
        report_lines.append("")

    report_path = os.path.join(args.out_dir, "cluster_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"\nwrote {report_path}, cluster_centroids_heatmap.png, cluster_pca.png, "
         f"user_clusters.csv, cluster_centroids.csv, user_profiles.csv")


if __name__ == "__main__":
    main()