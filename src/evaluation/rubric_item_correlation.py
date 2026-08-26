#!/usr/bin/env python3
"""Evaluate whether rubric items trigger on the same turns.

The primary estimand is the phi correlation between binary per-turn trigger
indicators (a rubric triggers when its score is nonzero). Uncertainty is
estimated by resampling users, preserving the dependence among turns from the
same developer. Jaccard overlap, lift, odds ratios, Fisher exact p-values with
Benjamini-Hochberg correction, arm-specific correlations, and direction
agreement are reported as complementary diagnostics.

Usage:
    python3 src/evaluation/rubric_item_correlation.py \
        --turn-vectors outputs/chat_vectors/chat_turn_vectors_full100.csv \
        --out-dir outputs/eval_out/rubric_correlations
"""

import argparse
import os
from itertools import combinations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import fisher_exact


RUBRIC_NAMES = {
    "R01": "Solution Scope", "R02": "Abstraction Level",
    "R03": "Dependency Posture", "R04": "Correctness Guarantees",
    "R05": "Robustness Philosophy", "R06": "Testing Rigor",
    "R07": "Performance Sensitivity", "R08": "Security Posture",
    "R09": "Refactoring Aggressiveness", "R10": "Documentation Richness",
    "R11": "Explanation Verbosity", "R12": "Interaction Autonomy",
    "R13": "Code Clarity", "R14": "Code Conciseness",
}
RUBRIC_IDS = list(RUBRIC_NAMES)


def phi_from_counts(n11, n10, n01, n00):
    """Phi coefficient for a 2x2 table; accepts scalars or arrays."""
    numerator = n11 * n00 - n10 * n01
    denominator = np.sqrt(
        (n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)
    )
    return np.divide(
        numerator, denominator, out=np.full_like(numerator, np.nan, dtype=float),
        where=denominator != 0,
    )


def bh_adjust(p_values):
    """Benjamini-Hochberg false-discovery-rate adjusted p-values."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def table_counts(a, b):
    return np.array([
        np.sum(a & b), np.sum(a & ~b), np.sum(~a & b), np.sum(~a & ~b)
    ], dtype=int)


def bootstrap_phi_by_user(turns, a_col, b_col, n_bootstrap, rng):
    """Cluster bootstrap: sample users and retain every turn for each draw."""
    per_user = []
    for _, group in turns.groupby("user_id", sort=False):
        a = group[a_col].to_numpy(dtype=bool)
        b = group[b_col].to_numpy(dtype=bool)
        per_user.append(table_counts(a, b))
    counts = np.asarray(per_user)
    n_users = len(counts)
    draw_indices = rng.integers(0, n_users, size=(n_bootstrap, n_users))
    sampled = counts[draw_indices].sum(axis=1)
    values = phi_from_counts(sampled[:, 0], sampled[:, 1], sampled[:, 2], sampled[:, 3])
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan
    return tuple(np.quantile(values, [0.025, 0.975]))


def pair_stats(turns, a, b, n_bootstrap, rng):
    ac, bc = f"trigger_{a}", f"trigger_{b}"
    av = turns[ac].to_numpy(dtype=bool)
    bv = turns[bc].to_numpy(dtype=bool)
    n11, n10, n01, n00 = table_counts(av, bv)
    n = len(turns)
    phi = float(phi_from_counts(
        np.array(n11), np.array(n10), np.array(n01), np.array(n00)
    ))
    ci_low, ci_high = bootstrap_phi_by_user(turns, ac, bc, n_bootstrap, rng)
    fisher_or, p_value = fisher_exact([[n11, n10], [n01, n00]])
    union = n11 + n10 + n01
    expected_both = (n11 + n10) * (n11 + n01) / n

    both = turns[av & bv]
    same_direction = (
        (both[f"score_{a}"] == both[f"score_{b}"]).mean() if len(both) else np.nan
    )

    row = {
        "rubric_a": a, "name_a": RUBRIC_NAMES[a],
        "rubric_b": b, "name_b": RUBRIC_NAMES[b],
        "n_turns": n, "a_triggered": int(n11 + n10),
        "b_triggered": int(n11 + n01), "both_triggered": int(n11),
        "expected_both_independent": expected_both,
        "phi": phi, "phi_ci95_low": ci_low, "phi_ci95_high": ci_high,
        "jaccard": n11 / union if union else np.nan,
        "lift": n11 / expected_both if expected_both else np.nan,
        "odds_ratio": fisher_or, "fisher_p": p_value,
        "same_direction_when_both": same_direction,
    }

    for arm in sorted(turns["arm"].dropna().unique()) if "arm" in turns else []:
        arm_data = turns[turns["arm"] == arm]
        aa = arm_data[ac].to_numpy(dtype=bool)
        bb = arm_data[bc].to_numpy(dtype=bool)
        c = table_counts(aa, bb)
        row[f"phi_{arm}"] = float(phi_from_counts(*[np.array(x) for x in c]))
        row[f"both_{arm}"] = int(c[0])
    return row


def write_heatmap(results, out_path):
    matrix = pd.DataFrame(np.eye(len(RUBRIC_IDS)), index=RUBRIC_IDS, columns=RUBRIC_IDS)
    for row in results.itertuples():
        matrix.loc[row.rubric_a, row.rubric_b] = row.phi
        matrix.loc[row.rubric_b, row.rubric_a] = row.phi
    labels = [f"{rid}\n{RUBRIC_NAMES[rid]}" for rid in RUBRIC_IDS]
    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(
        matrix, cmap="vlag", center=0, vmin=-1.0, vmax=1.0, square=True,
        linewidths=0.4, xticklabels=labels, yticklabels=labels, ax=ax,
        cbar_kws={"label": "Phi correlation of turn-level triggers"},
    )
    ax.set_title("Rubric item trigger correlations")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_report(results, turns, args, out_path):
    meaningful = results[
        (results["both_triggered"] >= args.min_overlap)
        & (results["phi"].abs() >= args.min_abs_phi)
        & ((results["phi_ci95_low"] > 0) | (results["phi_ci95_high"] < 0))
        & (results["fdr_q"] < 0.05)
    ].copy()
    positive = meaningful.sort_values("phi", ascending=False)
    negative = meaningful.sort_values("phi").query("phi < 0")

    lines = [
        "# Correlated rubric-item evaluation", "",
        f"Analyzed **{len(turns):,} turns**, **{turns['session_id'].nunique():,} sessions**, "
        f"and **{turns['user_id'].nunique():,} users**.", "",
        "A rubric item is triggered when its turn-level score is nonzero. The primary "
        "metric is phi correlation. Confidence intervals use a user-cluster bootstrap; "
        "Fisher exact p-values are Benjamini-Hochberg corrected across all 91 pairs.", "",
        f"A pair is called practically meaningful here when |phi| >= {args.min_abs_phi:g}, "
        f"overlap >= {args.min_overlap}, the 95% clustered-bootstrap interval excludes zero, "
        "and FDR q < 0.05.", "",
        f"**{len(meaningful)} of {len(results)} pairs met all criteria.**", "",
        "## Strongest positive relationships", "",
        "| Pair | Both | Phi (95% CI) | Lift | Jaccard | Same direction | FDR q |", 
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in positive.head(15).itertuples():
        lines.append(
            f"| {row.rubric_a} {row.name_a} / {row.rubric_b} {row.name_b} "
            f"| {row.both_triggered} | {row.phi:.3f} ({row.phi_ci95_low:.3f}, "
            f"{row.phi_ci95_high:.3f}) | {row.lift:.2f} | {row.jaccard:.3f} "
            f"| {row.same_direction_when_both:.1%} | {row.fdr_q:.2g} |"
        )
    if negative.empty:
        lines += ["", "## Negative relationships", "", "No negative pair met all criteria."]
    else:
        lines += [
            "", "## Negative relationships", "",
            "| Pair | Both | Phi (95% CI) | Lift | FDR q |",
            "|---|---:|---:|---:|---:|",
        ]
        for row in negative.head(10).itertuples():
            lines.append(
                f"| {row.rubric_a} {row.name_a} / {row.rubric_b} {row.name_b} "
                f"| {row.both_triggered} | {row.phi:.3f} ({row.phi_ci95_low:.3f}, "
                f"{row.phi_ci95_high:.3f}) | {row.lift:.2f} | {row.fdr_q:.2g} |"
            )
    lines += [
        "", "## Interpretation notes", "",
        "- Phi measures whether two items fire on the same turns; it does not establish that "
        "one rubric item causes the other.",
        "- Lift is useful for rare items but can look large when the absolute overlap is tiny; "
        "the minimum-overlap rule guards against that.",
        "- High phi suggests possible conceptual overlap or a shared task context. Review the "
        "paired evidence text before merging or rewriting rubric items.",
        "- Arm-specific columns in `rubric_pair_correlations.csv` help detect relationships "
        "created by the high-signal sampling strategy.", "",
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--turn-vectors", required=True,
                        help="one or more turn-vector CSV paths, comma-separated")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000,
                        help="number of user-cluster bootstrap draws")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-abs-phi", type=float, default=0.10)
    parser.add_argument("--min-overlap", type=int, default=20)
    args = parser.parse_args()
    if args.bootstrap < 100:
        raise SystemExit("--bootstrap must be at least 100")
    os.makedirs(args.out_dir, exist_ok=True)

    paths = [path.strip() for path in args.turn_vectors.split(",")]
    turns = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    required = ["user_id", "session_id"] + [f"score_{rid}" for rid in RUBRIC_IDS]
    missing = [column for column in required if column not in turns]
    if missing:
        raise SystemExit(f"missing required columns: {missing}")
    for rid in RUBRIC_IDS:
        turns[f"score_{rid}"] = pd.to_numeric(turns[f"score_{rid}"], errors="coerce").fillna(0)
        turns[f"trigger_{rid}"] = turns[f"score_{rid}"].ne(0)

    rng = np.random.default_rng(args.seed)
    rows = [pair_stats(turns, a, b, args.bootstrap, rng)
            for a, b in combinations(RUBRIC_IDS, 2)]
    results = pd.DataFrame(rows)
    results["fdr_q"] = bh_adjust(results["fisher_p"])
    results["bootstrap_excludes_zero"] = (
        (results["phi_ci95_low"] > 0) | (results["phi_ci95_high"] < 0)
    )
    results = results.sort_values(["phi", "both_triggered"], ascending=[False, False])

    csv_path = os.path.join(args.out_dir, "rubric_pair_correlations.csv")
    report_path = os.path.join(args.out_dir, "rubric_correlation_report.md")
    heatmap_path = os.path.join(args.out_dir, "rubric_trigger_phi_heatmap.png")
    results.to_csv(csv_path, index=False)
    write_report(results, turns, args, report_path)
    write_heatmap(results, heatmap_path)

    selected = results[
        (results["both_triggered"] >= args.min_overlap)
        & (results["phi"].abs() >= args.min_abs_phi)
        & results["bootstrap_excludes_zero"] & (results["fdr_q"] < 0.05)
    ]
    print(f"Loaded {len(turns):,} turns, {turns['session_id'].nunique():,} sessions, "
          f"{turns['user_id'].nunique():,} users")
    print(f"Meaningful correlated pairs: {len(selected)}/{len(results)}")
    columns = ["rubric_a", "name_a", "rubric_b", "name_b", "both_triggered",
               "phi", "phi_ci95_low", "phi_ci95_high", "lift", "fdr_q"]
    print(selected[columns].head(15).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nWrote {csv_path}\nWrote {report_path}\nWrote {heatmap_path}")


if __name__ == "__main__":
    main()
