#!/usr/bin/env python3
"""
Compare cluster "personas" between SWE-chat (chat-driven coding) and
DECODE (code completion) -- two datasets with NO overlapping users
(confirmed), different rubrics (14 axes vs 10 axes), and independently
chosen k. The only honest comparison possible is at the level of
PROFILE SHAPE: do similar combinations of traits recur across two
independent populations, using a rough conceptual mapping between the
two rubrics -- NOT "does the same person cluster the same way," which
is impossible to test without overlapping users.

AXIS MAPPING is a judgment call, not a rigorous correspondence -- the
two rubrics were built independently for different task types. Only the
5 pairs below were rated "high confidence" when reviewed; the rest of
each rubric (about half of each) has no defensible partner and is
excluded rather than force-matched.

Requires the RAW per-user profile + cluster assignment for each dataset
(user_profiles.csv + user_clusters.csv, which both cluster_user_profiles.py
and cluster_scored_pairs.py already write) -- NOT the saved
cluster_centroids.csv, which only has the K cluster means and can't be
used to correctly reconstruct population-level z-scores.

Usage:
    python compare_cluster_personas.py \
        --swe-profiles ./cluster_highdensity_zscore/user_profiles.csv \
        --swe-clusters ./cluster_highdensity_zscore/user_clusters.csv \
        --decode-profiles ./cluster_scored_pairs_min3/user_profiles.csv \
        --decode-clusters ./cluster_scored_pairs_min3/user_clusters.csv \
        --out-dir ./cross_dataset_compare
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# (SWE-chat axis id, DECODE axis name, canonical display name, sign_flip)
# sign_flip=True means the SWE-chat axis's +1 pole and the DECODE axis's +1
# pole are OPPOSITE concepts, so the SWE-chat value must be negated before
# comparison. Caught this for R14: SWE-chat's Code Conciseness high(+1) =
# "prefers VERBOSE, spelled-out code" (see the rubric table), while
# DECODE's Brevity and Conciseness high(+1) almost certainly means MORE
# terse the natural way the name reads. Without this flip, every reading
# involving conciseness would be silently inverted.
AXIS_MAPPING = [
    ("R05", "Robustness and Error Handling", "Robustness", False),
    ("R04", "Correctness and Precision", "Correctness", False),
    ("R14", "Brevity and Conciseness", "Conciseness", True),
    ("R02", "Modularity and Abstraction", "Abstraction", False),
    ("R13", "Explicitness and Clarity", "Clarity", False),
]


def row_zscore(profile: pd.DataFrame) -> pd.DataFrame:
    row_std = profile.std(axis=1).replace(0, np.nan)
    out = profile.sub(profile.mean(axis=1), axis=0).div(row_std, axis=0)
    return out.dropna()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--swe-profiles", required=True)
    ap.add_argument("--swe-clusters", required=True)
    ap.add_argument("--decode-profiles", required=True)
    ap.add_argument("--decode-clusters", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    swe_ids = [m[0] for m in AXIS_MAPPING]
    decode_ids = [m[1] for m in AXIS_MAPPING]
    canonical = [m[2] for m in AXIS_MAPPING]
    flips = [m[3] for m in AXIS_MAPPING]
    print(f"Comparing on {len(canonical)} mapped axes: {canonical}")
    if any(flips):
        flipped_names = [c for c, f in zip(canonical, flips) if f]
        print(f"Sign-flipped before comparison (SWE-chat's +1 pole is the "
             f"OPPOSITE concept from DECODE's +1 pole): {flipped_names}")
    print("(this is a judgment-call mapping between two independently-built "
         "rubrics -- treat as suggestive, not rigorous)\n")

    # ------------------------------------------------------------- SWE-chat
    swe_profile = pd.read_csv(args.swe_profiles, index_col=0)
    swe_clusters = pd.read_csv(args.swe_clusters, index_col=0)
    missing = [c for c in swe_ids if c not in swe_profile.columns]
    if missing:
        raise SystemExit(f"SWE-chat profile is missing expected columns: {missing}")
    swe_z = row_zscore(swe_profile[swe_ids])
    swe_z.columns = canonical
    for c, flip in zip(canonical, flips):
        if flip:
            swe_z[c] = -swe_z[c]
    swe_z = swe_z.join(swe_clusters, how="inner")
    swe_centroids = swe_z.groupby("cluster")[canonical].mean()
    swe_sizes = swe_z.groupby("cluster").size()
    print(f"SWE-chat: {len(swe_z)} users, {len(swe_centroids)} clusters, "
         f"sizes {swe_sizes.to_dict()}")

    # -------------------------------------------------------------- DECODE
    decode_profile = pd.read_csv(args.decode_profiles, index_col=0)
    decode_clusters = pd.read_csv(args.decode_clusters, index_col=0)
    missing = [c for c in decode_ids if c not in decode_profile.columns]
    if missing:
        raise SystemExit(f"DECODE profile is missing expected columns: {missing}")
    decode_z = row_zscore(decode_profile[decode_ids])
    decode_z.columns = canonical
    decode_z = decode_z.join(decode_clusters, how="inner")
    decode_centroids = decode_z.groupby("cluster")[canonical].mean()
    decode_sizes = decode_z.groupby("cluster").size()
    print(f"DECODE: {len(decode_z)} users, {len(decode_centroids)} clusters, "
         f"sizes {decode_sizes.to_dict()}")

    # ---------------------------------------------------------- cross-distance
    n_swe, n_dec = len(swe_centroids), len(decode_centroids)
    dist = np.zeros((n_swe, n_dec))
    cos_sim = np.zeros((n_swe, n_dec))
    for i, (sc, srow) in enumerate(swe_centroids.iterrows()):
        for j, (dc, drow) in enumerate(decode_centroids.iterrows()):
            s, d = srow.values, drow.values
            dist[i, j] = np.linalg.norm(s - d)
            ns, nd = np.linalg.norm(s), np.linalg.norm(d)
            cos_sim[i, j] = float(np.dot(s, d) / (ns * nd)) if ns > 0 and nd > 0 else np.nan
    dist_df = pd.DataFrame(dist, index=[f"SWE-{c}" for c in swe_centroids.index],
                           columns=[f"DEC-{c}" for c in decode_centroids.index])
    cos_df = pd.DataFrame(cos_sim, index=dist_df.index, columns=dist_df.columns)
    dist_df.to_csv(os.path.join(args.out_dir, "cross_dataset_distances.csv"))
    cos_df.to_csv(os.path.join(args.out_dir, "cross_dataset_cosine_similarity.csv"))

    print(f"\n=== cross-dataset centroid distance (5-axis mapped subspace, "
         f"lower = more similar profile shape) ===")
    print(dist_df.round(2).to_string())

    print(f"\n=== cross-dataset COSINE SIMILARITY (same subspace, -1 to +1; "
         f"scale-invariant -- measures whether two profiles point the same "
         f"DIRECTION regardless of how extreme either one is, which raw "
         f"distance conflates with magnitude) ===")
    print(cos_df.round(2).to_string())
    print("(near 0 = unrelated directions; near +1 = same shape of "
         "emphasis, regardless of intensity; near -1 = opposite emphasis, "
         "e.g. one cluster's high-Robustness/low-Brevity is the other's "
         "low-Robustness/high-Brevity)")

    print(f"\n=== nearest DECODE cluster for each SWE-chat cluster (by cosine "
         f"similarity) ===")
    for sc in swe_centroids.index:
        row = cos_df.loc[f"SWE-{sc}"]
        nearest = row.idxmax()
        print(f"  SWE-chat cluster {sc} (n={swe_sizes[sc]}) -> nearest {nearest} "
             f"(cosine={row.max():.2f}); full row: {dict(row.round(2))}")

    print(f"\n=== nearest SWE-chat cluster for each DECODE cluster (by cosine "
         f"similarity) ===")
    for dc in decode_centroids.index:
        col = cos_df[f"DEC-{dc}"]
        nearest = col.idxmax()
        print(f"  DECODE cluster {dc} (n={decode_sizes[dc]}) -> nearest {nearest} "
             f"(cosine={col.max():.2f})")

    # mutual nearest-neighbor by COSINE similarity, not distance -- direction
    # match is the more meaningful "same persona shape" criterion here
    mutual = []
    for sc in swe_centroids.index:
        nearest_dec = cos_df.loc[f"SWE-{sc}"].idxmax()
        dc = int(nearest_dec.replace("DEC-", ""))
        nearest_swe_back = cos_df[nearest_dec].idxmax()
        if nearest_swe_back == f"SWE-{sc}":
            mutual.append((sc, dc, cos_df.loc[f"SWE-{sc}", nearest_dec]))
    print(f"\n=== mutual nearest-neighbor pairs by cosine similarity (both "
         f"directions agree -- the closest thing to a real cross-dataset "
         f"match this method can show) ===")
    if mutual:
        for sc, dc, cs in mutual:
            print(f"  SWE-chat cluster {sc} <-> DECODE cluster {dc}  "
                 f"(cosine similarity={cs:.2f})")
            print(f"    SWE-chat {sc}: {swe_centroids.loc[sc].round(2).to_dict()}")
            print(f"    DECODE {dc}:   {decode_centroids.loc[dc].round(2).to_dict()}")
            print(f"    NOTE: high cosine similarity with two SMALL/near-zero "
                 f"centroids means 'both are generically flat,' not 'same "
                 f"persona' -- check the magnitude of BOTH vectors above, "
                 f"not just this similarity score, before treating this as "
                 f"a real match.")
    else:
        print("  none -- no SWE-chat/DECODE cluster pair is each other's "
             "nearest neighbor by direction, which is itself informative: "
             "on this 5-axis mapped subspace, no strong cross-dataset "
             "persona correspondence is showing up.")

    # ------------------------------------------------------------------ heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(max(12, n_dec * 2.4), max(4, n_swe * 0.8)))

    im0 = axes[0].imshow(dist_df.values, cmap="RdBu_r", aspect="auto")
    axes[0].set_xticks(range(n_dec)); axes[0].set_xticklabels(dist_df.columns, rotation=45, ha="right")
    axes[0].set_yticks(range(n_swe)); axes[0].set_yticklabels(dist_df.index)
    for i in range(n_swe):
        for j in range(n_dec):
            axes[0].text(j, i, f"{dist_df.values[i,j]:.2f}", ha="center", va="center", fontsize=8)
    axes[0].set_title("Euclidean distance\n(lower = closer)")
    fig.colorbar(im0, ax=axes[0], shrink=0.7)

    im1 = axes[1].imshow(cos_df.values, cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
    axes[1].set_xticks(range(n_dec)); axes[1].set_xticklabels(cos_df.columns, rotation=45, ha="right")
    axes[1].set_yticks(range(n_swe)); axes[1].set_yticklabels(cos_df.index)
    for i in range(n_swe):
        for j in range(n_dec):
            axes[1].text(j, i, f"{cos_df.values[i,j]:.2f}", ha="center", va="center", fontsize=8)
    axes[1].set_title("Cosine similarity\n(scale-invariant; +1 = same direction)")
    fig.colorbar(im1, ax=axes[1], shrink=0.7)

    fig.suptitle("Cross-dataset cluster comparison (5-axis mapped subspace)")
    fig.tight_layout()
    path = os.path.join(args.out_dir, "cross_dataset_heatmap.png")
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"\nwrote cross_dataset_distances.csv, cross_dataset_cosine_similarity.csv, {path}")


if __name__ == "__main__":
    main()