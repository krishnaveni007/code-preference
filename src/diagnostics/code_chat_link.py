#!/usr/bin/env python3
"""
Turn-to-code alignment for one session.

STEP 0 (verification, not assumed): confirms conversations.checkpoint_pk
is constant across all turns in the session -- i.e. it really is the
session-level canonical value and cannot be used to link a specific turn
to a specific commit. Prints a clear PASS/FAIL rather than silently
proceeding on an assumption.

STEP 1: gets the session's REAL checkpoint list from
sessions.checkpoint_ids (not conversations.checkpoint_pk), dates each
checkpoint by its earliest commit (commits.commit_date), and bins
conversation turns into the resulting intervals -- the same
reconstruction method used in section0c_segment.py, applied here to one
session instead of the whole corpus.

STEP 2: for each checkpoint, pulls its commit(s) and their diff stats
(files changed, additions, deletions).

STEP 3: builds a code-churn bar chart with the SAME turn labels on the
y-axis as turn_heatmap.py's chat heatmap, so the two figures can sit
side by side and be read against each other directly.

Usage:
    python session_code_alignment.py --data-dir ./swechat_data \
        --session-id <session_id> --out-dir ./session4_code
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def maybe_json(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (str, bytes)):
        try:
            v = json.loads(x)
            return v if isinstance(v, list) else [v]
        except Exception:
            return []
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--session-id", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    sessions = pd.read_parquet(os.path.join(args.data_dir, "sessions.parquet"))
    conv = pd.read_parquet(
        os.path.join(args.data_dir, "conversations.parquet"),
        columns=["session_id", "turn_number", "turn_type", "timestamp",
                 "checkpoint_pk", "content"])
    commits = pd.read_parquet(
        os.path.join(args.data_dir, "commits.parquet"),
        columns=["commit_sha", "checkpoint_pk", "commit_date", "author_date",
                 "files_changed_count", "total_additions", "total_deletions",
                 "commit_message"])

    sess_conv = conv[conv["session_id"] == args.session_id].sort_values("turn_number")
    if sess_conv.empty:
        raise SystemExit(f"no conversation rows found for session {args.session_id}")

    # ---------------------------------------------------------- STEP 0
    distinct_ckpt = sess_conv["checkpoint_pk"].dropna().unique()
    print("=== STEP 0: is conversations.checkpoint_pk unique per turn? ===")
    print(f"  distinct checkpoint_pk values across {len(sess_conv)} turns: "
         f"{len(distinct_ckpt)}")
    if len(distinct_ckpt) <= 1:
        print(f"  CONFIRMED: constant at {distinct_ckpt.tolist()} -- this is the "
             f"session-level canonical value, NOT a per-turn link. "
             f"Do not join on this column for turn-level code linkage.")
    else:
        print(f"  UNEXPECTED: {len(distinct_ckpt)} distinct values found -- "
             f"this contradicts the earlier corpus-wide finding. Worth "
             f"double-checking before trusting either result: {distinct_ckpt.tolist()}")

    # ---------------------------------------------------------- STEP 1
    sess_row = sessions[sessions["session_id"] == args.session_id]
    if sess_row.empty:
        raise SystemExit(f"session {args.session_id} not found in sessions.parquet")
    real_checkpoint_ids = maybe_json(sess_row.iloc[0]["checkpoint_ids"])
    print(f"\n=== STEP 1: real checkpoints for this session "
         f"(sessions.checkpoint_ids) ===")
    print(f"  {len(real_checkpoint_ids)} checkpoints: {real_checkpoint_ids}")

    ck_commits = commits[commits["checkpoint_pk"].isin(real_checkpoint_ids)].copy()
    ck_commits["_t"] = ck_commits["commit_date"].fillna(ck_commits["author_date"])
    ck_time = ck_commits.groupby("checkpoint_pk").agg(
        ckpt_time=("_t", "min"),
        n_commits=("commit_sha", "nunique"),
        additions=("total_additions", "sum"),
        deletions=("total_deletions", "sum"),
        files_changed=("files_changed_count", "sum"))
    ck_time = ck_time.sort_values("ckpt_time")
    n_dateable = ck_time["ckpt_time"].notna().sum()
    print(f"  {n_dateable}/{len(real_checkpoint_ids)} checkpoints have a "
         f"commit date and can be placed in time")
    if n_dateable == 0:
        raise SystemExit("no checkpoints in this session have a dateable commit -- "
                         "cannot align turns to code without at least one")

    # ---------------------------------------------- bin turns (two granularities)
    user_turns = sess_conv[sess_conv["turn_type"] == "user_prompt"].copy()
    user_turns["timestamp"] = pd.to_datetime(user_turns["timestamp"], utc=True)
    dated_ck = ck_time.dropna(subset=["ckpt_time"])
    boundaries = dated_ck["ckpt_time"].tolist()
    ck_ids_ordered = dated_ck.index.tolist()

    def bin_turns(boundary_times, boundary_labels, boundary_meta):
        """boundary_meta: list of dicts, one per boundary, with the stats
        to attach to any turn that falls before it."""
        idx = np.searchsorted(
            np.array([pd.Timestamp(b).tz_localize(None) if pd.Timestamp(b).tzinfo is None
                     else pd.Timestamp(b).tz_convert(None) for b in boundary_times],
                    dtype="datetime64[ns]"),
            user_turns["timestamp"].dt.tz_localize(None).values.astype("datetime64[ns]"),
            side="left")
        rows = []
        for k, (_, t) in enumerate(user_turns.iterrows()):
            i = idx[k]
            if i < len(boundary_labels):
                rows.append({"turn_number": int(t["turn_number"]),
                            "next_boundary": boundary_labels[i],
                            "boundary_time": boundary_times[i], **boundary_meta[i]})
            else:
                rows.append({"turn_number": int(t["turn_number"]),
                            "next_boundary": None, "boundary_time": None,
                            "n_commits": 0, "additions": 0, "deletions": 0, "files_changed": 0})
        return pd.DataFrame(rows)

    # checkpoint-level (coarser; what STEP 1 above dates and reports)
    ck_meta = [{"n_commits": int(r["n_commits"]), "additions": int(r["additions"]),
               "deletions": int(r["deletions"]), "files_changed": int(r["files_changed"])}
              for _, r in dated_ck.iterrows()]
    align = bin_turns(boundaries, ck_ids_ordered, ck_meta)
    align.to_csv(os.path.join(args.out_dir, "turn_code_alignment.csv"), index=False)
    print(f"\n=== STEP 2/3a: turn -> next CHECKPOINT -> commit stats (coarse) ===")
    print(align.to_string(index=False))

    # commit-level (finer; a checkpoint can bundle multiple commits, and
    # binning by checkpoint alone hides which of those commits a turn
    # actually preceded -- worth checking whenever n_commits > 1 above)
    per_commit_sorted = ck_commits.dropna(subset=["_t"]).sort_values("_t")
    commit_times = per_commit_sorted["_t"].tolist()
    commit_labels = per_commit_sorted["commit_sha"].tolist()
    commit_meta = [{"n_commits": 1, "additions": int(r["total_additions"]),
                   "deletions": int(r["total_deletions"]),
                   "files_changed": int(r["files_changed_count"]),
                   "commit_message": r["commit_message"]}
                  for _, r in per_commit_sorted.iterrows()]
    align_commit = bin_turns(commit_times, commit_labels, commit_meta)
    align_commit.to_csv(os.path.join(args.out_dir, "turn_code_alignment_by_commit.csv"), index=False)
    print(f"\n=== STEP 2/3b: turn -> next COMMIT -> diff stats (fine-grained) ===")
    print(align_commit[["turn_number", "next_boundary", "boundary_time",
                       "additions", "deletions", "files_changed",
                       "commit_message"]].to_string(index=False))

    # ---------------------------------------------- full chronological picture
    # "next commit after this turn" silently drops any commit that lands
    # BEFORE the first turn (it can never be anyone's "next" boundary) or
    # in a gap after the last scored turn. That's exactly where a commit
    # can hide if the conversation window being scored starts mid-task.
    # Print every commit for this session's real checkpoints against every
    # turn's timestamp, interleaved, so nothing is invisible.
    print(f"\n=== full chronological picture: every turn and every commit "
         f"for this session's checkpoints ===")
    timeline = []
    for _, t in user_turns.iterrows():
        timeline.append({"time": t["timestamp"], "kind": "turn",
                        "label": f"turn {int(t['turn_number'])}", "detail": ""})
    for _, r in per_commit_sorted.iterrows():
        msg_first_line = str(r["commit_message"]).split("\\n")[0][:80]
        timeline.append({"time": r["_t"], "kind": "COMMIT",
                        "label": r["commit_sha"][:10],
                        "detail": f"+{int(r['total_additions'])}/-{int(r['total_deletions'])} "
                                 f"{int(r['files_changed_count'])}f -- {msg_first_line}"})
    tl = pd.DataFrame(timeline).sort_values("time")
    tl.to_csv(os.path.join(args.out_dir, "full_timeline.csv"), index=False)
    for _, row in tl.iterrows():
        marker = ">>>" if row["kind"] == "COMMIT" else "   "
        print(f"{marker} {row['time']}  {row['kind']:>5}  {row['label']:<12} {row['detail']}")
    n_commits_before_first_turn = int(
        (per_commit_sorted["_t"] < user_turns["timestamp"].min()).sum())
    if n_commits_before_first_turn:
        print(f"\n  NOTE: {n_commits_before_first_turn} commit(s) for this session's "
             f"checkpoints landed BEFORE the first scored turn -- these never "
             f"appear as anyone's 'next commit' above and would otherwise be "
             f"invisible. See the full timeline for what they were.")

    # ---------------------------------------------------------- STEP 3: chart
    # Uses the commit-level (finer) alignment, since that's the one worth
    # looking at when a checkpoint bundles more than one commit.
    fig, axes = plt.subplots(1, 2, figsize=(9, max(2.5, 0.45 * len(align_commit) + 1)),
                             sharey=True, gridspec_kw={"width_ratios": [1, 1]})
    y = range(len(align_commit))
    labels = [f"turn {t}" for t in align_commit["turn_number"]]

    axes[0].barh(y, align_commit["additions"], color="#3B8C5A", label="additions")
    axes[0].barh(y, -align_commit["deletions"], color="#A6373D", label="deletions")
    axes[0].set_yticks(list(y))
    axes[0].set_yticklabels(labels, fontsize=9)
    axes[0].invert_yaxis()
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_xlabel("Lines (+/-) in the NEXT commit after this turn")
    axes[0].legend(fontsize=8, loc="lower right")
    axes[0].set_title("Code churn (per commit)")

    axes[1].barh(y, align_commit["files_changed"], color="#3B6FA0")
    axes[1].set_xlabel("Files changed in next commit")
    axes[1].set_title("Files touched")

    fig.suptitle(f"Session {args.session_id[:12]}: turns aligned to "
                "next-commit code churn")
    fig.tight_layout()
    path = os.path.join(args.out_dir, "turn_code_alignment.png")
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"\nwrote {path}")
    print(f"wrote {os.path.join(args.out_dir, 'turn_code_alignment.csv')}")


if __name__ == "__main__":
    main()