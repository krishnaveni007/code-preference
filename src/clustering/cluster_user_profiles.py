#!/usr/bin/env python3
"""
Cluster the 100 developers by their averaged preference profile
(user_id x 14 axes, from score_mean averaged across each user's
sessions), characterize each cluster (defining axes, centroid), and
pull real evidence quotes from the turn-level data for each cluster's
most-defining axes.

Design choices, stated rather than silently assumed:
  - Standardized (z-scored) before clustering. Axes have very different
    variances (R12 std~0.36 vs R07 std~0.13 in the full-run coverage
    check); unstandardized k-means would just group users by their R12
    score and ignore everything else.
  - k is chosen by sweeping a range and reporting silhouette scores, not
    picked arbitrarily. Override with --k if you want a specific value
    anyway (e.g. to match a hypothesis-driven number of personas).
  - Coverage caveat: a user who never fired an axis has
    score_mean_<axis> = 0, which conflates "genuinely neutral" with "no
    data for this user on this axis." This isn't fixed here -- it's
    flagged per-axis in the output so a cluster whose defining axis is
    low-coverage (R07, R08, R14, R03, R13 per the corpus-wide coverage
    check) gets read with appropriate caution.

Usage:
    python cluster_user_profiles.py \
        --session-vectors chat_session_vectors_full100.csv \
        --turn-vectors chat_turn_vectors_full100.csv \
        --out-dir ./cluster_out
    # or --k 4 to force a specific number of clusters
"""

import argparse
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

RUBRIC_IDS = [f"R{str(i).zfill(2)}" for i in range(1, 15)]
RUBRIC_NAMES = {
    "R01": "Solution Scope", "R02": "Abstraction Level", "R03": "Dependency Posture",
    "R04": "Correctness Guarantees", "R05": "Robustness Philosophy", "R06": "Testing Rigor",
    "R07": "Performance Sensitivity", "R08": "Security Posture", "R09": "Refactoring Aggressiveness",
    "R10": "Documentation Richness", "R11": "Explanation Verbosity", "R12": "Interaction Autonomy",
    "R13": "Code Clarity", "R14": "Code Conciseness",
}


def build_profiles(sessions: pd.DataFrame) -> pd.DataFrame:
    cols = [f"score_mean_{rid}" for rid in RUBRIC_IDS]
    profile = sessions.groupby("user_id")[cols].mean()
    profile.columns = RUBRIC_IDS
    return profile


def choose_k(X_scaled: np.ndarray, k_min: int, k_max: int, seed: int) -> pd.DataFrame:
    rows = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X_scaled)
        sil = silhouette_score(X_scaled, km.labels_)
        rows.append({"k": k, "silhouette": round(float(sil), 4), "inertia": round(float(km.inertia_), 2)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session-vectors", required=True)
    ap.add_argument("--turn-vectors", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--k", type=int, default=None,
                    help="force this many clusters; default sweeps 2-8 and "
                        "picks the best silhouette score")
    ap.add_argument("--k-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=8)
    ap.add_argument("--n-examples", type=int, default=3,
                    help="evidence quotes per cluster per defining axis")
    ap.add_argument("--defining-axis-mode", default="threshold",
                    choices=["threshold", "top-n"],
                    help="'threshold' (default): a cluster's defining axes are "
                        "whichever clear --z-threshold -- different clusters "
                        "can and should end up with different NUMBERS of "
                        "defining axes, not just different identities. "
                        "'top-n': always exactly --n-defining-axes per "
                        "cluster, for comparison/backward compatibility.")
    ap.add_argument("--z-threshold", type=float, default=0.3,
                    help="minimum |z-score| for an axis to count as defining "
                        "a cluster, in threshold mode")
    ap.add_argument("--n-defining-axes", type=int, default=3,
                    help="axes per cluster in top-n mode; also used as a cap "
                        "in threshold mode (--max-defining-axes)")
    ap.add_argument("--max-defining-axes", type=int, default=6,
                    help="upper cap on defining axes per cluster in "
                        "threshold mode, so a cluster with many axes above "
                        "threshold doesn't produce an unreadably long list")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--center", action="store_true",
                    help="row-center each user's profile (subtract their own "
                        "mean across all 14 axes) before clustering. Removes "
                        "a per-user 'answers everything more positively' "
                        "LEVEL effect. Does NOT remove differences in how "
                        "much a user's own profile varies (spread) -- if "
                        "clusters after --center still show the same top "
                        "axes at different magnitudes rather than different "
                        "axes, try --row-zscore instead.")
    ap.add_argument("--row-zscore", action="store_true",
                    help="full row standardization: subtract each user's own "
                        "mean AND divide by their own std, not just center. "
                        "Removes both per-user level and per-user spread, "
                        "leaving only the relative SHAPE of a user's profile "
                        "(which axes rank higher than others for them) to "
                        "drive clustering. Takes precedence over --center.")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    sessions = pd.concat([pd.read_csv(p) for p in args.session_vectors.split(",")],
                        ignore_index=True)
    turns = pd.concat([pd.read_csv(p) for p in args.turn_vectors.split(",")],
                      ignore_index=True)

    profile = build_profiles(sessions)
    n_users = len(profile)
    print(f"{n_users} users, profile shape {profile.shape}")

    # -------------------------------------------------- intensity diagnostic
    # Each user's own mean across all 14 axes. If this correlates strongly
    # with session count or arm, the earlier finding that clusters differ
    # mostly in magnitude (not shape) is likely explained by a mundane
    # data-volume or sampling confound rather than a genuine trait.
    intensity = profile[RUBRIC_IDS].mean(axis=1)
    n_sessions_per_user = sessions.groupby("user_id").size()
    intensity_df = pd.DataFrame({"intensity": intensity,
                                 "n_sessions": n_sessions_per_user})
    if "arm" in sessions.columns:
        arm_per_user = sessions.groupby("user_id")["arm"].first()
        intensity_df["arm"] = arm_per_user
    intensity_df.to_csv(os.path.join(args.out_dir, "intensity_diagnostic.csv"))
    corr_sessions = intensity_df["intensity"].corr(intensity_df["n_sessions"])
    print(f"\n=== Intensity diagnostic ===")
    print(f"  correlation(per-user mean score across all axes, n_sessions) = "
         f"{corr_sessions:.3f}")
    if "arm" in intensity_df.columns:
        print(intensity_df.groupby("arm")["intensity"].describe()[["mean", "std", "count"]]
             .to_string())
    print("  (a strong correlation here means the earlier magnitude-only "
         "clustering was likely tracking a data-volume or sampling artifact, "
         "not a real trait)")

    if args.row_zscore:
        row_mean = profile[RUBRIC_IDS].mean(axis=1)
        row_std = profile[RUBRIC_IDS].std(axis=1).replace(0, np.nan)
        profile_for_clustering = profile[RUBRIC_IDS].sub(row_mean, axis=0).div(row_std, axis=0)
        n_dropped = profile_for_clustering.isna().any(axis=1).sum()
        if n_dropped:
            print(f"\nWARNING: {n_dropped} users have zero variance across their "
                 f"own 14 axes (identical value everywhere, often all-zero) -- "
                 f"row std is 0, can't z-score. Dropping them from clustering.")
            profile_for_clustering = profile_for_clustering.dropna()
            profile = profile.loc[profile_for_clustering.index]
        print(f"\nROW Z-SCORED: each user's own mean AND std across all 14 "
             f"axes have been removed before clustering (shape only).")
    elif args.center:
        profile_for_clustering = profile[RUBRIC_IDS].sub(profile[RUBRIC_IDS].mean(axis=1), axis=0)
        print(f"\nROW-CENTERED: each user's own mean across all 14 axes has "
             f"been subtracted before clustering.")
    else:
        profile_for_clustering = profile[RUBRIC_IDS]

    # coverage per axis, for the caveat
    coverage = {}
    for rid in RUBRIC_IDS:
        col = f"support_{rid}"
        if col in sessions.columns:
            users_with_signal = sessions[sessions[col] > 0]["user_id"].nunique()
            coverage[rid] = round(users_with_signal / n_users, 3)
    print("\nPer-axis coverage (fraction of users with ANY signal) -- low-coverage "
         "axes deserve more caution if they turn out to define a cluster:")
    for rid, c in sorted(coverage.items(), key=lambda x: x[1]):
        flag = " <-- LOW COVERAGE" if c < 0.7 else ""
        print(f"  {rid} ({RUBRIC_NAMES[rid]}): {c:.0%}{flag}")

    # ---------------------------------------------------------- clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(profile_for_clustering.values)

    if args.k is None:
        sweep = choose_k(X_scaled, args.k_min, args.k_max, args.seed)
        sweep.to_csv(os.path.join(args.out_dir, "k_selection.csv"), index=False)
        print(f"\n=== k selection (silhouette score, higher is better) ===")
        print(sweep.to_string(index=False))
        best_k = int(sweep.loc[sweep["silhouette"].idxmax(), "k"])
        print(f"\nbest k by silhouette: {best_k}")
    else:
        best_k = args.k
        print(f"\nusing forced k={best_k}")

    km = KMeans(n_clusters=best_k, n_init=10, random_state=args.seed).fit(X_scaled)
    profile = profile.assign(cluster=km.labels_)
    profile.to_csv(os.path.join(args.out_dir, "user_clusters.csv"))

    print(f"\n=== cluster sizes ===")
    print(profile["cluster"].value_counts().sort_index().to_string())

    # ---------------------------------------------------------- characterize
    grand_mean = profile[RUBRIC_IDS].mean()
    grand_std = profile[RUBRIC_IDS].std()
    centroid_raw = profile.groupby("cluster")[RUBRIC_IDS].mean()
    centroid_z = (centroid_raw - grand_mean) / grand_std
    centroid_raw.to_csv(os.path.join(args.out_dir, "cluster_centroids.csv"))

    defining = {}
    print(f"\n=== cluster characterization ({args.defining_axis_mode} mode) ===")
    for c in sorted(profile["cluster"].unique()):
        n = (profile["cluster"] == c).sum()
        z = centroid_z.loc[c].sort_values(key=lambda s: s.abs(), ascending=False)
        if args.defining_axis_mode == "top-n":
            top = z.head(args.n_defining_axes)
        else:
            above_threshold = z[z.abs() >= args.z_threshold]
            top = above_threshold.head(args.max_defining_axes)
            if len(top) == 0:
                # nothing clears the bar -- still report the single strongest
                # axis so the cluster isn't left with an empty description,
                # but flag explicitly that it's below threshold
                top = z.head(1)
        defining[c] = top
        print(f"\nCluster {c} (n={n} users, {len(top)} defining axes):")
        for rid, zval in top.items():
            direction = "high" if centroid_raw.loc[c, rid] > 0 else "low"
            cov_flag = " [LOW COVERAGE]" if coverage.get(rid, 1) < 0.7 else ""
            below_flag = (" [below threshold -- shown as the single "
                          "strongest axis only]"
                         if args.defining_axis_mode == "threshold" and
                         abs(zval) < args.z_threshold else "")
            print(f"  {rid} ({RUBRIC_NAMES[rid]}): z={zval:+.2f}, "
                 f"mean={centroid_raw.loc[c, rid]:+.3f} ({direction}){cov_flag}{below_flag}")

    # ---------------------------------------------------------- centroid heatmap
    fig, ax = plt.subplots(figsize=(9, max(3, 0.6 * best_k + 1)))
    im = ax.imshow(centroid_raw.values, cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(14))
    ax.set_xticklabels(RUBRIC_IDS, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(best_k))
    ax.set_yticklabels([f"Cluster {c} (n={(profile['cluster']==c).sum()})"
                        for c in sorted(profile["cluster"].unique())], fontsize=9)
    for i in range(centroid_raw.shape[0]):
        for j in range(centroid_raw.shape[1]):
            v = centroid_raw.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                   color="white" if abs(v) > 0.5 else "black")
    ax.set_title(f"Cluster centroids ({best_k} clusters)")
    cbar = fig.colorbar(im, ax=ax, ticks=[-1, 0, 1], shrink=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "cluster_centroids_heatmap.png"),
               bbox_inches="tight", dpi=200)
    plt.close(fig)

    # ---------------------------------------------------------- PCA scatter
    pca = PCA(n_components=2, random_state=args.seed)
    coords = pca.fit_transform(X_scaled)
    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = plt.get_cmap("tab10")
    for c in sorted(profile["cluster"].unique()):
        mask = profile["cluster"].values == c
        ax.scatter(coords[mask, 0], coords[mask, 1], label=f"Cluster {c}",
                  color=cmap(c % 10), s=40, edgecolor="black", linewidth=0.3)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%} of variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%} of variance)")
    ax.set_title("User profiles, PCA projection colored by cluster")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "cluster_pca.png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"\nPCA: PC1+PC2 explain {sum(pca.explained_variance_ratio_[:2]):.0%} "
         f"of total variance -- if this is low, the 2D picture is a lossy view "
         f"of a higher-dimensional separation, not the whole story.")

    # ---------------------------------------------------------- evidence per cluster
    print(f"\n=== evidence quotes per cluster's defining axes ===")
    report_lines = [f"# User preference clusters (k={best_k})\n"]
    for c in sorted(profile["cluster"].unique()):
        cluster_users = profile[profile["cluster"] == c].index.tolist()
        n = len(cluster_users)
        report_lines.append(f"## Cluster {c} (n={n} users)\n")
        print(f"\n--- Cluster {c} (n={n}) ---")
        for rid, zval in defining[c].items():
            direction_val = 1 if centroid_raw.loc[c, rid] > 0 else -1
            direction_label = "high" if direction_val == 1 else "low"
            cov_flag = " (LOW COVERAGE AXIS -- read with caution)" if coverage.get(rid, 1) < 0.7 else ""
            report_lines.append(f"**{rid} ({RUBRIC_NAMES[rid]})**: {direction_label}"
                               f" (z={zval:+.2f}){cov_flag}\n")
            print(f"  {rid} ({RUBRIC_NAMES[rid]}) -- {direction_label}{cov_flag}")

            ev_col = f"evidence_{rid}"
            score_col = f"score_{rid}"
            if ev_col not in turns.columns:
                continue
            cluster_turns = turns[turns["user_id"].isin(cluster_users) &
                                  (turns[score_col] == direction_val)]
            examples = cluster_turns[ev_col].dropna()
            examples = examples[examples.str.len() > 0].sample(
                min(args.n_examples, len(examples)), random_state=args.seed) \
                if len(examples) else examples
            for ex in examples:
                print(f"    - {ex[:150]}")
                report_lines.append(f"- {ex}\n")
        report_lines.append("")

    report_path = os.path.join(args.out_dir, "cluster_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"\nwrote {report_path}, cluster_centroids_heatmap.png, cluster_pca.png, "
         f"user_clusters.csv, cluster_centroids.csv")


if __name__ == "__main__":
    main()