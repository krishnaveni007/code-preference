
#!/usr/bin/env python3
"""
Single-user variance decomposition for the case study.

IMPORTANT SCOPE NOTE: this is NOT the between-user discriminability metric
the full study needs. That requires >=2 users (does an axis distinguish
person A from person B?) and will be computed later across the 80-user
sample, with SESSIONS NESTED IN USERS.

What this script computes instead is the one-level-down analog: TURNS
NESTED IN SESSIONS, for a single user. It uses the identical statistical
machinery (one-way random-effects ICC, Shrout & Fleiss / McGraw & Wong
formula for unequal group sizes) so the case study doubles as a working
test of the method, but the number it produces answers a different
question: "how stable is this one person's expressed preference on this
axis, session to session?" -- not "does this axis distinguish people."

Per axis, reports:
  - support: how many turns / sessions actually carried signal at all
  - turn-level variance (pooled, raw ternary score including zeros)
  - within-session variance (MSW) / between-session variance (MSB)
  - icc_session_level: the one-way ICC(1) ratio from MSB/MSW -- this is
    the within-user, turn-in-session version of the eventual metric
  - sign_consistency_turn / sign_consistency_session: among turns/sessions
    that DID fire, what fraction share the majority sign (1.0 = always
    the same direction when it fires; 0.5 = coin flip)
  - recent_vs_maxtie_agreement: fraction of this user's sessions where
    the two aggregation methods agree on this axis (a data point for the
    R12-style inconsistency issue found in the case study review)

Usage:
    python case_study_variance.py \
        --turn-vectors ./source1_case_study/chat_turn_vectors.csv \
        --session-vectors ./source1_case_study/chat_session_vectors.csv \
        --out-dir ./source1_case_study/variance
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


def one_way_icc(values_by_group: list) -> dict:
    """Standard one-way random-effects ICC(1) with the unequal-group-size
    correction (Shrout & Fleiss 1979 / Donner 1986 n0 formula).

    values_by_group: list of arrays, one per session, of that session's
    raw per-turn scores (including zeros) for one axis.

    Returns dict with msb, msw, icc, n0 -- or a dict with icc=None and a
    reason if there isn't enough data to compute it (fewer than 2 groups
    with data, or zero total variance).
    """
    groups = [np.asarray(g, dtype=float) for g in values_by_group if len(g) > 0]
    k = len(groups)
    if k < 2:
        return {"icc": None, "reason": "fewer than 2 sessions with turns", "msb": None, "msw": None}

    n_i = np.array([len(g) for g in groups])
    N = n_i.sum()
    grand_mean = np.concatenate(groups).mean()
    group_means = np.array([g.mean() for g in groups])

    ss_between = np.sum(n_i * (group_means - grand_mean) ** 2)
    ss_within = np.sum([np.sum((g - g.mean()) ** 2) for g in groups])

    df_between = k - 1
    df_within = N - k
    if df_within <= 0:
        return {"icc": None, "reason": "no within-session degrees of freedom "
                                       "(every session has exactly 1 turn)",
               "msb": None, "msw": None}

    msb = ss_between / df_between
    msw = ss_within / df_within if df_within > 0 else 0.0

    n0 = (N - (np.sum(n_i ** 2) / N)) / df_between

    denom = msb + (n0 - 1) * msw
    if denom == 0:
        # zero total variance -- axis never varies at all (e.g. always 0)
        return {"icc": None, "reason": "zero total variance (axis constant "
                                       "across every turn)",
               "msb": round(float(msb), 6), "msw": round(float(msw), 6)}

    icc = (msb - msw) / denom
    return {"icc": round(float(icc), 4), "msb": round(float(msb), 6),
            "msw": round(float(msw), 6), "n0": round(float(n0), 3),
            "reason": None}


def sign_consistency(values) -> dict:
    """Among nonzero values, what fraction share the majority sign?
    Returns None if there are no nonzero values at all."""
    vals = np.asarray(values)
    nonzero = vals[vals != 0]
    if len(nonzero) == 0:
        return {"n_nonzero": 0, "consistency": None}
    pos = int((nonzero == 1).sum())
    neg = int((nonzero == -1).sum())
    majority = max(pos, neg)
    return {"n_nonzero": len(nonzero), "consistency": round(majority / len(nonzero), 4),
            "n_pos": pos, "n_neg": neg}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turn-vectors", required=True)
    ap.add_argument("--session-vectors", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    turns = pd.read_csv(args.turn_vectors)
    sess = pd.read_csv(args.session_vectors)
    user_id = turns["user_id"].iloc[0]
    n_sessions_total = sess["session_id"].nunique()
    n_turns_total = len(turns)

    rows = []
    for rid in RUBRIC_IDS:
        col = f"score_{rid}"
        if col not in turns.columns:
            continue

        # turn-level pooled variance (raw ternary, including zeros)
        turn_var = float(turns[col].var(ddof=0))

        # within/between-session decomposition
        groups = [g[col].values for _, g in turns.groupby("session_id")]
        icc_result = one_way_icc(groups)

        # support
        n_turns_fired = int((turns[col] != 0).sum())
        sessions_with_signal = turns[turns[col] != 0]["session_id"].nunique()

        # sign consistency at turn level (raw per-turn scores)
        turn_sign = sign_consistency(turns[col].values)

        # sign consistency at session level (aggregated 'recent' vector)
        recent_col = f"score_recent_{rid}"
        session_sign = sign_consistency(sess[recent_col].values) if recent_col in sess.columns else {"n_nonzero": 0, "consistency": None}

        # recent vs maxtie agreement, this axis only
        maxtie_col = f"score_maxtie_{rid}"
        if recent_col in sess.columns and maxtie_col in sess.columns:
            either_fired = (sess[recent_col] != 0) | (sess[maxtie_col] != 0)
            if either_fired.sum() > 0:
                agree = (sess[recent_col] == sess[maxtie_col]) & either_fired
                agreement_rate = round(float(agree.sum() / either_fired.sum()), 4)
            else:
                agreement_rate = None
        else:
            agreement_rate = None

        rows.append({
            "axis": rid, "name": RUBRIC_NAMES[rid],
            "n_turns_fired": n_turns_fired,
            "pct_turns_fired": round(n_turns_fired / n_turns_total, 4),
            "n_sessions_with_signal": sessions_with_signal,
            "pct_sessions_with_signal": round(sessions_with_signal / n_sessions_total, 4),
            "turn_level_variance": round(turn_var, 6),
            "msb_between_session": icc_result["msb"],
            "msw_within_session": icc_result["msw"],
            "icc_session_level": icc_result["icc"],
            "icc_na_reason": icc_result["reason"],
            "sign_consistency_turn": turn_sign["consistency"],
            "n_turn_pos": turn_sign.get("n_pos"),
            "n_turn_neg": turn_sign.get("n_neg"),
            "sign_consistency_session": session_sign["consistency"],
            "recent_vs_maxtie_agreement": agreement_rate,
        })

    out = pd.DataFrame(rows)
    csv_path = os.path.join(args.out_dir, "case_study_variance.csv")
    out.to_csv(csv_path, index=False)
    print(f"wrote {csv_path}\n")

    # console summary, most-to-least stable by ICC (NA rows last)
    display = out.copy()
    display["_sort"] = display["icc_session_level"].fillna(-999)
    display = display.sort_values("_sort", ascending=False).drop(columns="_sort")
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    print(display[["axis", "name", "n_turns_fired", "n_sessions_with_signal",
                   "icc_session_level", "sign_consistency_turn",
                   "sign_consistency_session", "recent_vs_maxtie_agreement"]]
         .to_string(index=False))

    print(f"\nuser: {user_id}, {n_sessions_total} sessions, {n_turns_total} turns total")
    print("\nNOTE: icc_session_level is a WITHIN-USER (turns-in-sessions) "
         "statistic, not between-user discriminability. See module docstring.")


if __name__ == "__main__":
    main()