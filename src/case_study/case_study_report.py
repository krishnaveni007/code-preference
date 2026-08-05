#!/usr/bin/env python3
"""
Build the Source 1 case study for one user: turn-by-turn fired-axis
evidence within each session, the session-level aggregated vector under
both methods (recent / maxtie), and a cross-session heatmap ordered
chronologically so within-user preference stability (or drift) across
sessions is visible at a glance.

Inputs: the outputs of chat_pref_vectorise_v2.py run with --user-id, plus
sessions.parquet for chronological ordering (session-level vectors have no
timestamp of their own).

Usage:
    python build_case_study_report.py \
        --data-dir ./swechat_data \
        --turn-vectors ./source1_case_study/chat_turn_vectors.csv \
        --session-vectors ./source1_case_study/chat_session_vectors.csv \
        --out-dir ./source1_case_study/report
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


def compact_vector(row, prefix="score_"):
    return [int(row.get(f"{prefix}{rid}", 0)) for rid in RUBRIC_IDS]


def fmt_vector(vec):
    return "[" + ", ".join(f"{v:+d}" if v != 0 else " 0" for v in vec) + "]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--turn-vectors", required=True)
    ap.add_argument("--session-vectors", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--user-id", default=None,
                    help="if the CSVs contain more than one user, pick which")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    turns = pd.read_csv(args.turn_vectors)
    sessions_vec = pd.read_csv(args.session_vectors)

    if args.user_id:
        turns = turns[turns["user_id"] == args.user_id]
        sessions_vec = sessions_vec[sessions_vec["user_id"] == args.user_id]
    user_id = turns["user_id"].iloc[0]

    # chronological order comes from sessions.parquet -- session-level
    # vectors carry no timestamp of their own
    sess_meta = pd.read_parquet(os.path.join(args.data_dir, "sessions.parquet"),
                                columns=["session_id", "created_at"])
    sessions_vec = sessions_vec.merge(sess_meta, on="session_id", how="left")
    sessions_vec = sessions_vec.sort_values("created_at").reset_index(drop=True)
    session_order = sessions_vec["session_id"].tolist()

    # ---------------------------------------------------------- text report
    lines = [f"# Case study: {user_id}", "",
            f"{len(session_order)} sessions, chronologically ordered.", ""]

    for sess_rank, sid in enumerate(session_order, start=1):
        srow = sessions_vec[sessions_vec["session_id"] == sid].iloc[0]
        sess_turns = turns[turns["session_id"] == sid].sort_values("turn_number")
        lines.append(f"## Session {sess_rank}/{len(session_order)}: {sid}")
        lines.append(f"({srow['created_at']}, {len(sess_turns)} turns)")
        lines.append("")

        for _, t in sess_turns.iterrows():
            vec = compact_vector(t)
            lines.append(f"**Turn {int(t['turn_number'])}**: {fmt_vector(vec)}")
            if t.get("screen_analysis"):
                lines.append(f"  - analysis: {t['screen_analysis']}")
            fired = [rid for rid in RUBRIC_IDS if t.get(f"score_{rid}", 0) != 0]
            for rid in fired:
                score = int(t[f"score_{rid}"])
                ev = t.get(f"evidence_{rid}", "")
                pole = "high" if score == 1 else "low"
                lines.append(f"  - {rid} ({RUBRIC_NAMES[rid]}) -> {pole}: {ev}")
            if not fired:
                lines.append("  - (no axes fired)")
            lines.append("")

        recent_vec = compact_vector(srow, prefix="score_recent_")
        maxtie_vec = compact_vector(srow, prefix="score_maxtie_")
        lines.append(f"**Session {sess_rank} aggregated vector:**")
        lines.append(f"  - recent: {fmt_vector(recent_vec)}")
        lines.append(f"  - maxtie: {fmt_vector(maxtie_vec)}")
        disagree = [RUBRIC_IDS[i] for i in range(14) if recent_vec[i] != maxtie_vec[i]]
        if disagree:
            lines.append(f"  - methods disagree on: {', '.join(disagree)}")
        lines.append("")

    report_path = os.path.join(args.out_dir, "case_study_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {report_path}")

    # ---------------------------------------------------------- heatmaps
    def build_heatmap(method_prefix, title, path):
        mat = np.array([compact_vector(r, prefix=method_prefix)
                        for _, r in sessions_vec.iterrows()])
        fig, ax = plt.subplots(figsize=(10, max(3, 0.5 * len(session_order) + 1)))
        im = ax.imshow(mat, cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(14))
        ax.set_xticklabels(RUBRIC_IDS, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(session_order)))
        ax.set_yticklabels([f"S{i+1}" for i in range(len(session_order))], fontsize=9)
        ax.set_xlabel("Rubric axis")
        ax.set_ylabel("Session (chronological)")
        ax.set_title(title)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if v != 0:
                    ax.text(j, i, f"{v:+d}", ha="center", va="center",
                           fontsize=8, color="white")
        cbar = fig.colorbar(im, ax=ax, ticks=[-1, 0, 1], shrink=0.6)
        cbar.ax.set_yticklabels(["low (-1)", "no signal (0)", "high (+1)"])
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight", dpi=200)
        plt.close(fig)
        print(f"wrote {path}")

    build_heatmap("score_recent_",
                  f"{user_id}: session preference vectors (method=recent)",
                  os.path.join(args.out_dir, "heatmap_recent.png"))
    build_heatmap("score_maxtie_",
                  f"{user_id}: session preference vectors (method=maxtie)",
                  os.path.join(args.out_dir, "heatmap_maxtie.png"))

    # ---------------------------------------------------------- agreement summary
    recent_mat = np.array([compact_vector(r, "score_recent_") for _, r in sessions_vec.iterrows()])
    maxtie_mat = np.array([compact_vector(r, "score_maxtie_") for _, r in sessions_vec.iterrows()])
    n_nonzero_either = ((recent_mat != 0) | (maxtie_mat != 0)).sum()
    n_agree = ((recent_mat == maxtie_mat) & ((recent_mat != 0) | (maxtie_mat != 0))).sum()
    print(f"\nmethod agreement (recent vs maxtie), among cells where either "
         f"method fired: {n_agree}/{n_nonzero_either} "
         f"({n_agree/max(n_nonzero_either,1):.1%})")


if __name__ == "__main__":
    main()