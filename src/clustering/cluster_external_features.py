#!/usr/bin/env python3
"""
Characterize clusters by features EXTERNAL to the preference vector
itself -- programming language, repo domain/audience, coding agent used,
natural language of prompts, session success, agent-authored code %.

This is a validity check as much as a description: if clusters that were
built purely from preference scores also differ systematically on
things like "which programming language" or "which repo domain,"
that's evidence the clusters track something structurally real (e.g.
different languages have different idiomatic style norms that could
genuinely shape preference) rather than noise. If nothing external
correlates with cluster membership, that's worth knowing too.

Sources:
  - repositories.parquet: repo_type_domain, repo_type_audience directly;
    programming language extracted from repo_github_metadata (the raw
    GitHub API repo response JSON, which standardly has a top-level
    "language" key -- parsed defensively since the exact structure
    wasn't independently re-verified against live data this pass).
  - sessions.parquet: agent, session_success, agent_percentage,
    duration_seconds, joined via user_id -> which repos/sessions.
  - conversations.parquet: language (detected NATURAL language of the
    prompt, e.g. English/Japanese/Korean -- distinct from repo
    programming language), aggregated to a per-user modal value.

For each feature, reports a simple association check: chi-square for
categorical features, one-way ANOVA F-test for numeric ones -- so
"this looks different across clusters" is backed by a number, not just
eyeballed percentages.

Usage:
    python cluster_external_features.py \
        --data-dir ./swechat_data \
        --user-clusters ./cluster_k10/user_clusters.csv \
        --out-dir ./cluster_k10/features
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, f_oneway


def maybe_json(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return None
    return None


def extract_language(meta_raw):
    d = maybe_json(meta_raw)
    if not d:
        return None
    # standard GitHub REST API repo response has a top-level "language" key;
    # fall back to a couple of plausible alternate nestings defensively
    for key_path in [("language",), ("repo", "language"), ("data", "language")]:
        cur = d
        ok = True
        for k in key_path:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                ok = False
                break
        if ok and cur:
            return cur
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--user-clusters", required=True,
                    help="user_clusters.csv from cluster_user_profiles.py")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    clusters = pd.read_csv(args.user_clusters)
    if "user_id" not in clusters.columns:
        clusters = clusters.rename(columns={clusters.columns[0]: "user_id"})
    clusters = clusters[["user_id", "cluster"]]
    n_clusters = clusters["cluster"].nunique()
    print(f"{len(clusters)} users, {n_clusters} clusters")

    sessions = pd.read_parquet(
        os.path.join(args.data_dir, "sessions.parquet"),
        columns=["session_id", "user_id", "repo_id", "agent", "session_success",
                "agent_percentage", "duration_seconds"])
    repos = pd.read_parquet(
        os.path.join(args.data_dir, "repositories.parquet"),
        columns=["repo_id", "repo_type_domain", "repo_type_audience",
                "repo_github_metadata"])
    repos["prog_language"] = repos["repo_github_metadata"].map(extract_language)
    lang_extract_rate = repos["prog_language"].notna().mean()
    print(f"programming language extracted for {lang_extract_rate:.0%} of repos "
         f"(check this isn't near-0%, which would mean the JSON structure "
         f"assumption above didn't match)")

    sessions = sessions.merge(
        repos[["repo_id", "repo_type_domain", "repo_type_audience", "prog_language"]],
        on="repo_id", how="left")

    # natural language of prompts, per user (modal value)
    conv = pd.read_parquet(
        os.path.join(args.data_dir, "conversations.parquet"),
        columns=["session_id", "user_id", "language", "turn_type"])
    conv_up = conv[conv["turn_type"] == "user_prompt"]
    nat_lang_mode = (conv_up.dropna(subset=["language"])
                    .groupby("user_id")["language"]
                    .agg(lambda s: s.value_counts().idxmax() if len(s) else None))

    # ---------------------------------------------- per-user feature aggregation
    def modal(s):
        s = s.dropna()
        return s.value_counts().idxmax() if len(s) else None

    per_user = sessions.groupby("user_id").agg(
        n_sessions=("session_id", "nunique"),
        primary_agent=("agent", modal),
        primary_prog_language=("prog_language", modal),
        primary_repo_domain=("repo_type_domain", modal),
        primary_repo_audience=("repo_type_audience", modal),
        mean_session_success=("session_success",
                              lambda s: pd.to_numeric(s, errors="coerce").mean()),
        mean_agent_percentage=("agent_percentage", "mean"),
        mean_duration_seconds=("duration_seconds", "mean"),
    ).reset_index()
    per_user = per_user.merge(nat_lang_mode.rename("primary_nat_language"),
                              on="user_id", how="left")

    merged = per_user.merge(clusters, on="user_id", how="inner")
    merged.to_csv(os.path.join(args.out_dir, "user_features.csv"), index=False)
    print(f"\n{len(merged)} users matched to a cluster with feature data")

    categorical = ["primary_agent", "primary_prog_language", "primary_repo_domain",
                   "primary_repo_audience", "primary_nat_language"]
    numeric = ["n_sessions", "mean_session_success", "mean_agent_percentage",
              "mean_duration_seconds"]

    print(f"\n=== Categorical features by cluster (top value + % per cluster) ===")
    for feat in categorical:
        print(f"\n--- {feat} ---")
        ct = pd.crosstab(merged["cluster"], merged[feat])
        if ct.shape[1] < 2 or ct.shape[0] < 2:
            print("  not enough distinct values/clusters to test")
            continue
        # top category per cluster, as a fraction of that cluster
        row_pct = ct.div(ct.sum(axis=1), axis=0)
        for c in sorted(merged["cluster"].unique()):
            if c not in row_pct.index:
                continue
            top = row_pct.loc[c].sort_values(ascending=False).head(2)
            print(f"  cluster {c}: " + ", ".join(f"{v}={p:.0%}" for v, p in top.items()))
        try:
            chi2, p, dof, _ = chi2_contingency(ct)
            print(f"  chi-square: chi2={chi2:.2f}, dof={dof}, p={p:.4f}"
                 f"{'  <-- significant at p<0.05' if p < 0.05 else ''}")
        except ValueError as e:
            print(f"  chi-square not computable: {e}")

    print(f"\n=== Numeric features by cluster (mean +/- std, ANOVA F-test) ===")
    for feat in numeric:
        print(f"\n--- {feat} ---")
        groups = [g[feat].dropna().values for _, g in merged.groupby("cluster")]
        for c, g in zip(sorted(merged["cluster"].unique()), groups):
            if len(g):
                print(f"  cluster {c}: mean={g.mean():.2f}, std={g.std():.2f}, n={len(g)}")
        groups = [g for g in groups if len(g) >= 2]
        if len(groups) >= 2:
            try:
                fstat, p = f_oneway(*groups)
                print(f"  ANOVA: F={fstat:.2f}, p={p:.4f}"
                     f"{'  <-- significant at p<0.05' if p < 0.05 else ''}")
            except ValueError as e:
                print(f"  ANOVA not computable: {e}")

    print(f"\nwrote {os.path.join(args.out_dir, 'user_features.csv')}")
    print("\nNote: with 10 clusters and only 100 users, some clusters will be "
         "small (single digits) -- chi-square and ANOVA are both unreliable "
         "at very small group sizes, so treat p-values from tiny clusters as "
         "suggestive, not confirmatory.")


if __name__ == "__main__":
    main()