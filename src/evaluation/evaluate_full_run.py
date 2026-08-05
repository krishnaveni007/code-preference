#!/usr/bin/env python3
"""
Evaluation of the full 100-user Source 1 run: discriminability, per-arm
comparison, method reliability at scale, and coverage.

This is the real discriminability computation the project has been
building toward -- sessions nested in USERS (not turns nested in
sessions, which is what the single-user case study computed as a
within-user diagnostic). At n=100 users this actually answers "does
this axis distinguish developers," which n=1 never could.

Computes, per axis and per aggregation method (recent / maxtie / mean):
  - discriminability = one-way random-effects ICC, sessions nested in
    users (same machinery as case_study_variance.py's one_way_icc, one
    level up: users are now the groups, sessions are the observations)
  - the SAME computation restricted to the high-signal arm only, and to
    the random-control arm only -- directly tests whether filtering by
    pushback rate before measuring discriminability inflates it, which
    was the whole reason the two-arm design exists

Also reports, since a report needs more than one table:
  - coverage: what fraction of users / sessions ever fired each axis
  - recent-vs-maxtie agreement at full scale, per axis (extends the
    3-user pilot check that showed R12 improving 71%->90%)
  - data quality: any_llm_call_failed rate

Usage:
    python evaluate_full_run.py \
        --session-vectors chat_session_vectors_full100.csv \
        --out-dir ./eval_out
    # or, if you haven't merged the pilot + remaining-97 runs yet:
    python evaluate_full_run.py \
        --session-vectors source1_pilot_v2/chat_session_vectors.csv,source1_remaining97/chat_session_vectors.csv \
        --out-dir ./eval_out
"""

import argparse
import os

import numpy as np
import pandas as pd

RUBRIC_IDS = [f"R{str(i).zfill(2)}" for i in range(1, 15)]
RUBRIC_NAMES = {
    "R01": "Solution Scope", "R02": "Abstraction Level", "R03": "Dependency Posture",
    "R04": "Correctness Guarantees", "R05": "Robustness Philosophy", "R06": "Testing Rigor",
    "R07": "Performance Sensitivity", "R08": "Security Posture", "R09": "Refactoring Aggressiveness",
    "R10": "Documentation Richness", "R11": "Explanation Verbosity", "R12": "Interaction Autonomy",
    "R13": "Code Clarity", "R14": "Code Conciseness",
}
METHODS = ["recent", "maxtie", "mean"]


def one_way_icc(values_by_group: list) -> dict:
    """One-way random-effects ICC(1), unequal-group-size correction
    (Shrout & Fleiss 1979 / Donner 1986 n0 formula). Validated against
    hand-computable edge cases in case_study_variance.py; identical here,
    just applied with users as groups instead of sessions."""
    groups = [np.asarray(g, dtype=float) for g in values_by_group if len(g) > 0]
    k = len(groups)
    if k < 2:
        return {"icc": None, "reason": "fewer than 2 users with data", "msb": None, "msw": None}

    n_i = np.array([len(g) for g in groups])
    N = n_i.sum()
    grand_mean = np.concatenate(groups).mean()
    group_means = np.array([g.mean() for g in groups])

    ss_between = np.sum(n_i * (group_means - grand_mean) ** 2)
    ss_within = np.sum([np.sum((g - g.mean()) ** 2) for g in groups])

    df_between = k - 1
    df_within = N - k
    if df_within <= 0:
        return {"icc": None, "reason": "no within-user degrees of freedom "
                                       "(every user has exactly 1 session)",
               "msb": None, "msw": None}

    msb = ss_between / df_between
    msw = ss_within / df_within if df_within > 0 else 0.0
    n0 = (N - (np.sum(n_i ** 2) / N)) / df_between

    denom = msb + (n0 - 1) * msw
    if denom == 0:
        return {"icc": None, "reason": "zero total variance (axis constant "
                                       "across every session)",
               "msb": round(float(msb), 6), "msw": round(float(msw), 6)}

    icc = (msb - msw) / denom
    return {"icc": round(float(icc), 4), "msb": round(float(msb), 6),
            "msw": round(float(msw), 6), "k_users": k, "n_sessions": int(N),
            "reason": None}


def discriminability_table(sessions: pd.DataFrame, method: str, label: str) -> pd.DataFrame:
    rows = []
    for rid in RUBRIC_IDS:
        col = f"score_{method}_{rid}"
        if col not in sessions.columns:
            continue
        groups = [g[col].values for _, g in sessions.groupby("user_id")]
        result = one_way_icc(groups)
        rows.append({
            "axis": rid, "name": RUBRIC_NAMES[rid], "arm": label, "method": method,
            "icc": result["icc"], "msb": result["msb"], "msw": result["msw"],
            "reason": result["reason"],
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session-vectors", required=True,
                    help="one or more chat_session_vectors.csv paths, comma-separated")
    ap.add_argument("--turn-vectors", default=None,
                    help="optional: chat_turn_vectors.csv path(s), comma-separated, "
                        "for data-quality checks")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    paths = [p.strip() for p in args.session_vectors.split(",")]
    sessions = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    n_users = sessions["user_id"].nunique()
    n_sessions = len(sessions)
    print(f"Loaded {n_sessions} sessions, {n_users} users, from {len(paths)} file(s)")
    if "arm" not in sessions.columns:
        raise SystemExit("no 'arm' column found -- can't split high_signal vs "
                         "random_control without it")
    print(f"  arms: {sessions.groupby('arm')['user_id'].nunique().to_dict()}")

    # dedupe: if the same user_id/session_id appears in more than one input
    # file (e.g. pilot + full run overlap by accident), keep the first and
    # warn rather than silently double-count a session
    dupes = sessions.duplicated(subset=["user_id", "session_id"], keep="first")
    if dupes.any():
        print(f"  WARNING: {dupes.sum()} duplicate (user_id, session_id) rows "
             f"found across input files -- keeping first occurrence only. "
             f"Check whether the pilot and full-run outputs overlap.")
        sessions = sessions[~dupes]

    # ============================================================ 1. DISCRIMINABILITY
    print("\n=== Discriminability (sessions nested in users, one-way ICC) ===")
    all_results = []
    for method in METHODS:
        all_results.append(discriminability_table(sessions, method, "all_100"))
        hs = sessions[sessions["arm"] == "high_signal"]
        ctrl = sessions[sessions["arm"] == "random_control"]
        all_results.append(discriminability_table(hs, method, "high_signal_only"))
        all_results.append(discriminability_table(ctrl, method, "random_control_only"))
    disc = pd.concat(all_results, ignore_index=True)
    disc.to_csv(os.path.join(args.out_dir, "discriminability.csv"), index=False)

    # console summary: 'recent' method, all three arms, side by side
    pivot = disc[disc["method"] == "recent"].pivot(index=["axis", "name"],
                                                    columns="arm", values="icc")
    print(pivot.to_string())
    print("\n(NaN = not enough data to compute for that axis/arm combination, "
         "not zero discriminability -- see the 'reason' column in discriminability.csv)")

    # ============================================================ 2. COVERAGE
    print("\n=== Coverage: what fraction of users/sessions fired each axis ===")
    cov_rows = []
    for rid in RUBRIC_IDS:
        support_col = f"support_{rid}"
        if support_col not in sessions.columns:
            continue
        fired_sessions = sessions[support_col] > 0
        users_with_signal = sessions[fired_sessions]["user_id"].nunique()
        cov_rows.append({
            "axis": rid, "name": RUBRIC_NAMES[rid],
            "sessions_with_signal": int(fired_sessions.sum()),
            "pct_sessions": round(float(fired_sessions.mean()), 4),
            "users_with_signal": users_with_signal,
            "pct_users": round(users_with_signal / n_users, 4),
        })
    cov = pd.DataFrame(cov_rows).sort_values("pct_users", ascending=False)
    cov.to_csv(os.path.join(args.out_dir, "coverage.csv"), index=False)
    print(cov.to_string(index=False))

    # ============================================================ 3. METHOD AGREEMENT
    print("\n=== recent vs maxtie agreement, per axis, full sample ===")
    agree_rows = []
    for rid in RUBRIC_IDS:
        rc, mc = f"score_recent_{rid}", f"score_maxtie_{rid}"
        if rc not in sessions.columns or mc not in sessions.columns:
            continue
        either = (sessions[rc] != 0) | (sessions[mc] != 0)
        if either.sum() == 0:
            continue
        agree = ((sessions[rc] == sessions[mc]) & either).sum()
        agree_rows.append({"axis": rid, "name": RUBRIC_NAMES[rid],
                          "n_compared": int(either.sum()),
                          "agreement_rate": round(float(agree / either.sum()), 4)})
    agreement = pd.DataFrame(agree_rows).sort_values("agreement_rate")
    agreement.to_csv(os.path.join(args.out_dir, "method_agreement.csv"), index=False)
    print(agreement.to_string(index=False))

    # ============================================================ 4. DATA QUALITY
    print("\n=== Data quality ===")
    if "any_llm_call_failed" in sessions.columns:
        fail_rate = sessions["any_llm_call_failed"].mean()
        print(f"  sessions with >=1 failed LLM call: "
             f"{int(sessions['any_llm_call_failed'].sum())}/{n_sessions} "
             f"({fail_rate:.1%})")
    sessions_per_user = sessions.groupby("user_id").size()
    print(f"  sessions per user: min={sessions_per_user.min()}, "
         f"median={sessions_per_user.median():.0f}, max={sessions_per_user.max()}")

    print(f"\nAll outputs written to {args.out_dir}/")


if __name__ == "__main__":
    main()