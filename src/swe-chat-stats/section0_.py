#!/usr/bin/env python3
"""
Section 0: SWE-chat dataset overview + annotation feasibility check.

Run locally against the downloaded parquet tables. Computes every number we
need for the "dataset overview" section of the doc, plus a feasibility table
telling us the real achievable (n_users x n_subsessions) budget BEFORE we
spend anything on LLM annotation.

Usage:
    python3 dataset_stats.py --data-dir ./swechat_data --out-dir ./section0_out

Design notes:
  - Schema-defensive. Every table is introspected first; stats are computed
    only for columns that actually exist. A column-name mismatch degrades to
    a warning instead of a crash.
  - Writes both a machine-readable JSON blob (for later figure scripts) and a
    human-readable report.md we can paste straight into the doc draft.
  - No LLM calls, no network. Pure pandas over local parquet.
"""

import argparse
import json
import os
import sys
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Candidate column names. The dataset card and the actual parquet have drifted
# from each other before (user_id documented on conversations but absent), so
# we probe a list of plausible names and use whichever is present.
# ---------------------------------------------------------------------------
CANDIDATES = {
    "session_id":      ["session_id", "session_pk"],
    "repo_id":         ["repo_id", "repository_id"],
    "user_id":         ["user_id", "author", "git_author", "author_login", "username"],
    "checkpoint_pk":   ["checkpoint_pk", "checkpoint_id"],
    "canonical_ckpt":  ["canonical_checkpoint_pk", "canonical_checkpoint_id"],
    "commit_sha":      ["commit_sha", "sha", "commit_id"],
    "turn_number":     ["conversation_turn_number", "turn_number", "turn_idx"],
    "turn_type":       ["turn_type", "role", "message_type"],
    "pushback":        ["prompt_pushback", "pushback", "pushback_class"],
    "intent":          ["prompt_intent", "intent"],
    "persona":         ["user_persona", "persona"],
    "success":         ["session_success", "success_score"],
    "agent":           ["agent", "agent_name", "model", "model_name"],
    "content":         ["content", "text", "message", "target_content"],
    "attribution":     ["file_attribution", "attribution"],
    "is_agent_author": ["is_agent_author", "agent_authored"],
    "timestamp":       ["timestamp", "created_at", "ts", "commit_time"],
    "patch":           ["patch", "diff"],
    "file_path":       ["file_path", "path", "filename"],
}

TABLES = ["repositories", "sessions", "session_logs", "checkpoints",
          "commits", "conversations"]


def find_col(df, key):
    """Return the first candidate column name for `key` present in df, else None."""
    for c in CANDIDATES.get(key, [key]):
        if c in df.columns:
            return c
    return None


def load_tables(data_dir):
    """Load whichever of the six parquet tables are present."""
    tables = OrderedDict()
    for name in TABLES:
        # tolerate both flat files and HF-style subdirectories
        candidates = [
            os.path.join(data_dir, f"{name}.parquet"),
            os.path.join(data_dir, name, "train-00000-of-00001.parquet"),
            os.path.join(data_dir, "data", f"{name}.parquet"),
        ]
        path = next((p for p in candidates if os.path.exists(p)), None)
        if path is None:
            # glob fallback: any parquet under a dir named `name`
            import glob
            hits = glob.glob(os.path.join(data_dir, "**", name, "*.parquet"),
                             recursive=True)
            hits += glob.glob(os.path.join(data_dir, "**", f"{name}*.parquet"),
                              recursive=True)
            path = hits[0] if hits else None
        if path is None:
            print(f"  [warn] table not found: {name}")
            continue
        print(f"  loading {name} from {path} ...", flush=True)
        tables[name] = pd.read_parquet(path)
        print(f"    -> {tables[name].shape[0]:,} rows x "
              f"{tables[name].shape[1]} cols")
    return tables


def schema_report(tables):
    """Dump the real schema so we stop guessing column names."""
    out = {}
    for name, df in tables.items():
        out[name] = {
            "n_rows": int(df.shape[0]),
            "columns": {c: str(df[c].dtype) for c in df.columns},
            "resolved": {k: find_col(df, k) for k in CANDIDATES
                         if find_col(df, k) is not None},
        }
    return out


# ---------------------------------------------------------------------------
# User identity resolution.
# user_id is not reliably on `conversations`. The documented resolution chain
# is session -> checkpoint -> commit -> git author. We try direct columns
# first, then fall back to the join chain, and REPORT which path was used and
# how much coverage each gives, because that coverage number is itself a
# finding for the doc (how many sessions are attributable to a named user).
# ---------------------------------------------------------------------------
def resolve_session_users(tables):
    sessions = tables.get("sessions")
    if sessions is None:
        return None, {"method": "unavailable"}

    s_sid = find_col(sessions, "session_id")
    meta = {"n_sessions": int(len(sessions))}

    # Path A: direct user column on sessions
    s_uid = find_col(sessions, "user_id")
    if s_uid is not None:
        m = sessions[[s_sid, s_uid]].rename(
            columns={s_sid: "session_id", s_uid: "user_id"}).dropna()
        m = m[m["user_id"].astype(str).str.strip() != ""]
        meta.update(method="sessions.user_id direct",
                    coverage=float(len(m) / max(len(sessions), 1)))
        return m.drop_duplicates(), meta

    # Path B: sessions -> canonical checkpoint -> commits -> author
    commits = tables.get("commits")
    if commits is None:
        return None, {"method": "no commits table"}

    c_ckpt = find_col(commits, "checkpoint_pk")
    c_uid = find_col(commits, "user_id")
    s_ckpt = find_col(sessions, "canonical_ckpt") or find_col(sessions, "checkpoint_pk")

    if not (c_ckpt and c_uid and s_ckpt):
        return None, {"method": "join columns missing",
                      "have": {"commits.ckpt": c_ckpt, "commits.user": c_uid,
                               "sessions.ckpt": s_ckpt}}

    # human-authored commits only, if we can tell
    cm = commits.copy()
    agent_col = find_col(cm, "is_agent_author")
    if agent_col is not None:
        cm = cm[~cm[agent_col].fillna(False).astype(bool)]

    ck2user = (cm[[c_ckpt, c_uid]].dropna()
               .rename(columns={c_ckpt: "_ckpt", c_uid: "user_id"}))
    ck2user = ck2user[ck2user["user_id"].astype(str).str.strip() != ""]

    # a checkpoint can have multiple authors -> keep the modal author, and
    # record how often that ambiguity occurs
    multi = ck2user.groupby("_ckpt")["user_id"].nunique()
    meta["multi_author_checkpoint_frac"] = float((multi > 1).mean()) if len(multi) else 0.0
    ck2user = (ck2user.groupby("_ckpt")["user_id"]
               .agg(lambda s: s.value_counts().index[0]).reset_index())

    m = (sessions[[s_sid, s_ckpt]]
         .rename(columns={s_sid: "session_id", s_ckpt: "_ckpt"})
         .merge(ck2user, on="_ckpt", how="left")[["session_id", "user_id"]])
    resolved = m.dropna(subset=["user_id"])
    meta.update(method="sessions -> checkpoint -> human commit author",
                coverage=float(len(resolved) / max(len(sessions), 1)))
    return resolved.drop_duplicates(), meta


# ---------------------------------------------------------------------------
# Core descriptive stats
# ---------------------------------------------------------------------------
def dist(series):
    """Compact distribution summary for the doc tables."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return {}
    return {
        "n": int(len(s)),
        "mean": round(float(s.mean()), 2),
        "median": round(float(s.median()), 2),
        "p25": round(float(s.quantile(.25)), 2),
        "p75": round(float(s.quantile(.75)), 2),
        "p90": round(float(s.quantile(.90)), 2),
        "max": round(float(s.max()), 2),
    }


def overview_stats(tables, sess_users):
    st = {}

    # --- top-level counts -------------------------------------------------
    st["counts"] = {name: int(len(df)) for name, df in tables.items()}

    convs = tables.get("conversations")
    sessions = tables.get("sessions")
    repos = tables.get("repositories")
    commits = tables.get("commits")
    ckpts = tables.get("checkpoints")

    if repos is not None:
        st["counts"]["distinct_repos"] = int(repos.shape[0])

    # --- users ------------------------------------------------------------
    if sess_users is not None and len(sess_users):
        st["users"] = {
            "n_distinct_users": int(sess_users["user_id"].nunique()),
            "n_sessions_attributed": int(sess_users["session_id"].nunique()),
        }
        per_user = sess_users.groupby("user_id")["session_id"].nunique()
        st["users"]["sessions_per_user"] = dist(per_user)
        st["users"]["users_with_ge_N_sessions"] = {
            str(k): int((per_user >= k).sum()) for k in (1, 2, 3, 5, 10, 20, 30)
        }

    # --- sessions ---------------------------------------------------------
    if convs is not None and sessions is not None:
        c_sid = find_col(convs, "session_id")
        if c_sid:
            tps = convs.groupby(c_sid).size()
            st["sessions"] = {"turns_per_session": dist(tps)}

        for key, label in [("agent", "agent"), ("persona", "user_persona"),
                           ("success", "session_success")]:
            col = find_col(sessions, key)
            if col is None:
                continue
            if key == "success":
                st.setdefault("sessions", {})["session_success"] = dist(sessions[col])
            else:
                vc = sessions[col].fillna("<null>").astype(str).value_counts()
                st.setdefault("sessions", {})[label] = {
                    k: int(v) for k, v in vc.head(25).items()}

    # --- conversation-level annotations ----------------------------------
    if convs is not None:
        st["conversations"] = {}
        for key, label in [("turn_type", "turn_type"), ("pushback", "prompt_pushback"),
                           ("intent", "prompt_intent")]:
            col = find_col(convs, key)
            if col is None:
                continue
            vc = convs[col].fillna("<null>").astype(str).value_counts()
            st["conversations"][label] = {k: int(v) for k, v in vc.head(30).items()}

        # pushback turns per session / per user
        pb_col = find_col(convs, "pushback")
        c_sid = find_col(convs, "session_id")
        if pb_col and c_sid:
            is_pb = (~convs[pb_col].isna()) & \
                    (~convs[pb_col].astype(str).isin(["non_pushback", "", "<null>", "None"]))
            pb = convs[is_pb]
            st["conversations"]["n_pushback_turns"] = int(is_pb.sum())
            st["conversations"]["pushback_rate"] = round(float(is_pb.mean()), 4)
            st["conversations"]["pushback_per_session"] = dist(
                pb.groupby(c_sid).size())
            st["conversations"]["sessions_with_any_pushback"] = int(
                pb[c_sid].nunique())
            if sess_users is not None and len(sess_users):
                pbu = (pb[[c_sid]].rename(columns={c_sid: "session_id"})
                       .merge(sess_users, on="session_id", how="inner"))
                st["conversations"]["pushback_per_user"] = dist(
                    pbu.groupby("user_id").size())

    # --- commits / checkpoints -------------------------------------------
    if commits is not None:
        st["commits"] = {}
        agent_col = find_col(commits, "is_agent_author")
        if agent_col:
            vc = commits[agent_col].fillna("<null>").astype(str).value_counts()
            st["commits"]["is_agent_author"] = {k: int(v) for k, v in vc.items()}
        ck = find_col(commits, "checkpoint_pk")
        if ck:
            st["commits"]["commits_per_checkpoint"] = dist(
                commits.groupby(ck).size())

    if ckpts is not None and sessions is not None:
        s_ckpt = find_col(sessions, "canonical_ckpt") or find_col(sessions, "checkpoint_pk")
        s_sid = find_col(sessions, "session_id")
        if s_ckpt and s_sid:
            st.setdefault("checkpoints", {})["checkpoints_per_session_canonical"] = dist(
                sessions.groupby(s_ckpt)[s_sid].nunique())

    return st


# ---------------------------------------------------------------------------
# Feasibility: how many (user, subsession) units can we actually annotate?
#
# A "subsession" is defined here as a checkpoint-delimited segment within a
# session: checkpoint_t -> conversation turns -> checkpoint_{t+1}. We count
# a subsession as USABLE if it has (a) at least one user turn, and (b) a
# resolvable code delta on both ends. Both criteria are reported separately
# so we can see which one is the binding constraint.
# ---------------------------------------------------------------------------
def feasibility(tables, sess_users, target_users=100, target_subsessions=20):
    convs = tables.get("conversations")
    sessions = tables.get("sessions")
    commits = tables.get("commits")
    if convs is None or sessions is None or sess_users is None:
        return {"error": "missing tables for feasibility"}

    c_sid = find_col(convs, "session_id")
    c_ckpt = find_col(convs, "checkpoint_pk")
    c_type = find_col(convs, "turn_type")

    res = {"definition": "subsession = (session_id, checkpoint_pk) segment",
           "target": {"users": target_users, "subsessions_per_user": target_subsessions}}

    if not (c_sid and c_ckpt):
        res["error"] = "conversations lacks session_id/checkpoint_pk; " \
                       "cannot segment into subsessions"
        return res

    df = convs[[c_sid, c_ckpt] + ([c_type] if c_type else [])].copy()
    df.columns = ["session_id", "checkpoint_pk"] + (["turn_type"] if c_type else [])

    # user turns only, if we can tell them apart
    if c_type:
        user_like = df["turn_type"].astype(str).str.lower().isin(
            ["user", "human", "prompt", "user_message"])
        res["turn_type_values_seen"] = {
            k: int(v) for k, v in df["turn_type"].astype(str).value_counts().head(15).items()}
        if user_like.any():
            df = df[user_like]

    seg = (df.groupby(["session_id", "checkpoint_pk"]).size()
           .reset_index(name="n_user_turns"))
    seg = seg.merge(sess_users, on="session_id", how="left")

    res["n_subsessions_total"] = int(len(seg))
    res["n_subsessions_attributed"] = int(seg["user_id"].notna().sum())
    res["user_turns_per_subsession"] = dist(seg["n_user_turns"])

    seg_att = seg.dropna(subset=["user_id"])

    # code-delta availability per subsession
    if commits is not None:
        cm_ckpt = find_col(commits, "checkpoint_pk")
        if cm_ckpt:
            has_commit = set(commits[cm_ckpt].dropna().unique())
            seg_att = seg_att.assign(
                has_code=seg_att["checkpoint_pk"].isin(has_commit))
            res["subsessions_with_code_delta"] = int(seg_att["has_code"].sum())
            res["frac_subsessions_with_code_delta"] = round(
                float(seg_att["has_code"].mean()), 4)

    # feasibility curve: for each minimum-subsession threshold, how many users
    # clear it? This is the table that decides the annotation budget.
    per_user = seg_att.groupby("user_id").size()
    res["subsessions_per_user"] = dist(per_user)
    res["users_clearing_threshold"] = {
        str(k): int((per_user >= k).sum()) for k in (1, 5, 10, 15, 20, 30, 50)
    }
    res["n_users_total"] = int(per_user.shape[0])

    k = target_subsessions
    clearing = int((per_user >= k).sum())
    res["verdict"] = {
        "users_with_ge_target_subsessions": clearing,
        "target_met": bool(clearing >= target_users),
        "max_units_at_target": int(min(clearing, target_users) * k),
    }

    # same curve but restricted to subsessions with a code delta
    if "has_code" in seg_att.columns:
        pu_code = seg_att[seg_att["has_code"]].groupby("user_id").size()
        res["users_clearing_threshold_code_only"] = {
            str(kk): int((pu_code >= kk).sum()) for kk in (1, 5, 10, 15, 20, 30, 50)
        }
        res["subsessions_per_user_code_only"] = dist(pu_code)

    return res


# ---------------------------------------------------------------------------
def write_report(out_dir, schema, stats, feas, user_meta):
    lines = ["# Section 0 — SWE-chat dataset overview (recomputed)", ""]

    lines += ["## Table sizes", ""]
    for k, v in stats.get("counts", {}).items():
        lines.append(f"- `{k}`: {v:,}")
    lines.append("")

    lines += ["## User resolution", "",
              f"- method: `{user_meta.get('method')}`",
              f"- session coverage: {user_meta.get('coverage')}"]
    if "multi_author_checkpoint_frac" in user_meta:
        lines.append("- checkpoints with >1 human author: "
                     f"{user_meta['multi_author_checkpoint_frac']:.3f}")
    u = stats.get("users", {})
    if u:
        lines += [f"- distinct users: **{u.get('n_distinct_users')}**",
                  f"- sessions attributed: {u.get('n_sessions_attributed'):,}",
                  f"- sessions per user: {u.get('sessions_per_user')}",
                  f"- users with >= N sessions: {u.get('users_with_ge_N_sessions')}"]
    lines.append("")

    lines += ["## Conversations", ""]
    for k, v in stats.get("conversations", {}).items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")

    lines += ["## Sessions", ""]
    for k, v in stats.get("sessions", {}).items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")

    lines += ["## Commits / checkpoints", ""]
    for sec in ("commits", "checkpoints"):
        for k, v in stats.get(sec, {}).items():
            lines.append(f"- **{k}**: {v}")
    lines.append("")

    lines += ["## Feasibility for the 100 users x 20 subsessions plan", ""]
    for k, v in feas.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")

    lines += ["## Resolved schema", ""]
    for t, info in schema.items():
        lines.append(f"### {t} ({info['n_rows']:,} rows)")
        lines.append("```")
        for c, dt in info["columns"].items():
            lines.append(f"{c}: {dt}")
        lines.append("```")
        lines.append("")

    path = os.path.join(out_dir, "report.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True,
                    help="directory containing the SWE-chat parquet tables")
    ap.add_argument("--out-dir", default="./section0_out")
    ap.add_argument("--target-users", type=int, default=100)
    ap.add_argument("--target-subsessions", type=int, default=20)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading tables ...")
    tables = load_tables(args.data_dir)
    if not tables:
        sys.exit("No parquet tables found under --data-dir")

    print("\nIntrospecting schema ...")
    schema = schema_report(tables)

    print("Resolving users ...")
    sess_users, user_meta = resolve_session_users(tables)
    print(f"  method: {user_meta.get('method')}, "
          f"coverage: {user_meta.get('coverage')}")
    if sess_users is not None:
        sess_users.to_csv(os.path.join(args.out_dir, "session_user_map.csv"),
                          index=False)

    print("Computing overview stats ...")
    stats = overview_stats(tables, sess_users)

    print("Computing feasibility ...")
    feas = feasibility(tables, sess_users,
                       args.target_users, args.target_subsessions)

    blob = {"schema": schema, "user_resolution": user_meta,
            "stats": stats, "feasibility": feas}
    with open(os.path.join(args.out_dir, "section0_stats.json"), "w") as f:
        json.dump(blob, f, indent=2, default=str)

    rp = write_report(args.out_dir, schema, stats, feas, user_meta)

    print(f"\nWrote:\n  {os.path.join(args.out_dir, 'section0_stats.json')}"
          f"\n  {rp}"
          f"\n  {os.path.join(args.out_dir, 'session_user_map.csv')}")
    print("\n--- FEASIBILITY VERDICT ---")
    print(json.dumps(feas.get("verdict", feas), indent=2, default=str))


if __name__ == "__main__":
    main()