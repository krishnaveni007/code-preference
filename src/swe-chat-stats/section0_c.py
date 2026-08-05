#!/usr/bin/env python3
"""
Section 0c — reconstruct subsessions by timestamp alignment.

WHY: conversations.checkpoint_pk is the session's CANONICAL checkpoint,
denormalized onto every turn, so it is constant within a session and cannot
segment it. Real per-session checkpoints live in sessions.checkpoint_ids.
Checkpoints have no timestamp column of their own, so we date them through
their commits and bin conversation turns into the resulting intervals.

DEFINITION USED HERE
    For a session with checkpoints c_1..c_k ordered by their earliest commit
    time t_1 < ... < t_k, subsession i is the set of turns with
    t_{i-1} <= turn.timestamp < t_i  (t_0 = session start).
    So subsession i is "the work that led up to checkpoint i".

Reports, per session and aggregated:
  - how many checkpoints are datable (have >=1 commit with a date)
  - how many subsessions have >=1 user_prompt
  - how many have a code delta on BOTH ends (i.e. a previous checkpoint to
    diff against) -- this is the real source-2 denominator
  - the users x usable-subsessions feasibility grid

Usage:
    python3 section0_c.py --data-dir ../swechat_data --out-dir ./section0c_out
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd


def find_parquet(data_dir, name):
    for p in [os.path.join(data_dir, f"{name}.parquet"),
              os.path.join(data_dir, "data", f"{name}.parquet")]:
        if os.path.exists(p):
            return p
    hits = glob.glob(os.path.join(data_dir, "**", f"{name}*.parquet"), recursive=True)
    if not hits:
        raise FileNotFoundError(name)
    return hits[0]


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


def dist(s):
    s = pd.to_numeric(pd.Series(s), errors="coerce").dropna()
    if not len(s):
        return {}
    return {"n": int(len(s)), "mean": round(float(s.mean()), 2),
            "median": round(float(s.median()), 2),
            "p25": round(float(s.quantile(.25)), 2),
            "p75": round(float(s.quantile(.75)), 2),
            "p90": round(float(s.quantile(.90)), 2),
            "max": round(float(s.max()), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", default="./section0c_out")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    out = {}

    print("loading ...", flush=True)
    sessions = pd.read_parquet(find_parquet(args.data_dir, "sessions"))
    checkpoints = pd.read_parquet(find_parquet(args.data_dir, "checkpoints"))
    commits = pd.read_parquet(find_parquet(args.data_dir, "commits"),
                              columns=["commit_sha", "checkpoint_pk", "user_id",
                                       "commit_date", "author_date",
                                       "total_additions", "total_deletions",
                                       "files_changed_count"])

    # ---------------------------------------------------------------- dating
    # A checkpoint's time = earliest commit date among its commits.
    ck_time = (commits.assign(_t=commits["commit_date"].fillna(commits["author_date"]))
               .groupby("checkpoint_pk")
               .agg(ckpt_time=("_t", "min"),
                    n_commits=("commit_sha", "nunique"),
                    additions=("total_additions", "sum"),
                    deletions=("total_deletions", "sum"),
                    files=("files_changed_count", "sum")))

    out["dating"] = {
        "n_checkpoints_total": int(len(checkpoints)),
        "n_checkpoints_datable": int(ck_time["ckpt_time"].notna().sum()),
        "frac_datable": round(float(ck_time["ckpt_time"].notna().mean()), 4),
    }

    # -------------------------------------------------- session -> checkpoints
    rows = []
    for _, s in sessions.iterrows():
        for cid in maybe_json(s.get("checkpoint_ids")):
            rows.append((s["session_id"], str(cid)))
    s2c = pd.DataFrame(rows, columns=["session_id", "checkpoint_pk"]).drop_duplicates()
    s2c = s2c.merge(ck_time, left_on="checkpoint_pk", right_index=True, how="left")

    per_sess_ck = s2c.groupby("session_id").agg(
        n_ckpts=("checkpoint_pk", "nunique"),
        n_datable=("ckpt_time", lambda x: int(x.notna().sum())))
    out["checkpoints_per_session"] = {
        "all": dist(per_sess_ck["n_ckpts"]),
        "datable_only": dist(per_sess_ck["n_datable"]),
        "sessions_with_ge2_ckpts": int((per_sess_ck["n_ckpts"] >= 2).sum()),
        "sessions_with_ge2_datable": int((per_sess_ck["n_datable"] >= 2).sum()),
        "n_sessions_with_any_ckpt": int(len(per_sess_ck)),
    }
    print("  sessions with >=2 checkpoints:",
          out["checkpoints_per_session"]["sessions_with_ge2_ckpts"])

    # ------------------------------------------------------------ turn binning
    print("loading conversation turns ...", flush=True)
    conv = pd.read_parquet(
        find_parquet(args.data_dir, "conversations"),
        columns=["session_id", "user_id", "turn_number", "turn_type",
                 "timestamp", "prompt_pushback", "word_count"])
    up = conv[conv["turn_type"] == "user_prompt"].dropna(subset=["timestamp"])
    print(f"  {len(up):,} user_prompt turns with timestamps")

    # only bother with sessions that actually have >=2 datable checkpoints
    multi = per_sess_ck[per_sess_ck["n_datable"] >= 2].index
    bounds = (s2c[s2c["session_id"].isin(multi) & s2c["ckpt_time"].notna()]
              .sort_values(["session_id", "ckpt_time"]))

    real_pb = {"correction", "failure_report", "rejection", "takeover",
               "requirement_change", "pacing_complaint"}

    seg_rows = []
    up_by_sess = {k: v for k, v in up.groupby("session_id")}
    for sid, grp in bounds.groupby("session_id"):
        turns = up_by_sess.get(sid)
        if turns is None or not len(turns):
            continue
        times = grp["ckpt_time"].tolist()
        cks = grp["checkpoint_pk"].tolist()
        adds = grp["additions"].tolist()
        dels = grp["deletions"].tolist()
        edges = [pd.Timestamp.min.tz_localize("UTC")] + times
        idx = np.searchsorted(np.array(times, dtype="datetime64[ns]"),
                              turns["timestamp"].values.astype("datetime64[ns]"),
                              side="left")
        turns = turns.assign(_seg=idx)
        for i, ck in enumerate(cks):
            t = turns[turns["_seg"] == i]
            seg_rows.append({
                "session_id": sid,
                "seg_index": i,
                "checkpoint_pk": ck,
                "prev_checkpoint_pk": cks[i - 1] if i > 0 else None,
                "has_prev": i > 0,
                "n_user_prompts": int(len(t)),
                "n_pushback": int(t["prompt_pushback"].astype(str).isin(real_pb).sum()),
                "additions": adds[i],
                "deletions": dels[i],
                "user_id": turns["user_id"].dropna().iloc[0]
                if turns["user_id"].notna().any() else None,
            })

    segs = pd.DataFrame(seg_rows)
    segs.to_csv(os.path.join(args.out_dir, "subsessions.csv"), index=False)

    if len(segs):
        usable = segs[(segs["n_user_prompts"] >= 1) & segs["has_prev"] &
                      ((segs["additions"].fillna(0) + segs["deletions"].fillna(0)) > 0)]
        out["subsessions"] = {
            "n_segments_total": int(len(segs)),
            "n_with_user_prompt": int((segs["n_user_prompts"] >= 1).sum()),
            "n_with_prev_checkpoint": int(segs["has_prev"].sum()),
            "n_usable": int(len(usable)),
            "user_prompts_per_segment": dist(segs["n_user_prompts"]),
            "pushback_per_segment": dist(segs["n_pushback"]),
            "empty_segment_frac": round(float((segs["n_user_prompts"] == 0).mean()), 4),
        }
        pu = usable.dropna(subset=["user_id"]).groupby("user_id").size()
        out["feasibility"] = {
            "n_users_with_usable_subsessions": int(len(pu)),
            "usable_per_user": dist(pu),
            "users_clearing": {str(k): int((pu >= k).sum())
                               for k in (1, 3, 5, 10, 15, 20)},
            "unbalanced_units_cap20_min3": int(
                np.minimum(pu[pu >= 3], 20).sum()) if (pu >= 3).any() else 0,
        }
        pu.sort_values(ascending=False).to_csv(
            os.path.join(args.out_dir, "usable_per_user.csv"))
    else:
        out["subsessions"] = {"error": "no segments built"}

    with open(os.path.join(args.out_dir, "section0c_stats.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)

    print("\n=== RESULTS ===")
    print(json.dumps(out, indent=2, default=str)[:4000])
    print(f"\nwrote {args.out_dir}/")


if __name__ == "__main__":
    main()