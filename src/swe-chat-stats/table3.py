#!/usr/bin/env python3
"""Verify / recompute every figure in the intro section tables.
Usage: python verify_intro_tables.py --data-dir ./swechat_data --out-dir ./intro_out
"""
import argparse, glob, json, os
import numpy as np, pandas as pd

def find_parquet(data_dir, name):
    for p in [os.path.join(data_dir, f"{name}.parquet"),
              os.path.join(data_dir, "data", f"{name}.parquet")]:
        if os.path.exists(p):
            return p
    hits = glob.glob(os.path.join(data_dir, "**", f"{name}*.parquet"), recursive=True)
    if not hits:
        raise FileNotFoundError(name)
    return hits[0]

AGENT_NORM = {"claude code": "Claude Code", "claude-code": "Claude Code",
              "opencode": "OpenCode", "codex": "Codex", "gemini cli": "Gemini CLI",
              "cursor": "Cursor", "copilot cli": "Copilot CLI"}

def norm_agent(x):
    if pd.isna(x):
        return "unknown"
    return AGENT_NORM.get(str(x).strip().lower(), str(x).strip())

def med(s):
    s = pd.to_numeric(pd.Series(s), errors="coerce").dropna()
    if not len(s):
        return None
    return {"n": int(len(s)), "mean": round(float(s.mean()), 2),
            "median": round(float(s.median()), 1),
            "p75": round(float(s.quantile(.75)), 1),
            "p90": round(float(s.quantile(.90)), 1),
            "max": round(float(s.max()), 1)}

def jlen(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return 0
    if isinstance(x, (list, np.ndarray)):
        return len(x)
    try:
        v = json.loads(x)
        return len(v) if isinstance(v, list) else 1
    except Exception:
        return 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", default="./intro_out")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    R = {}

    sessions = pd.read_parquet(find_parquet(args.data_dir, "sessions"))
    repos = pd.read_parquet(find_parquet(args.data_dir, "repositories"))
    sessions["_agent"] = sessions["agent"].map(norm_agent)
    is_cc = sessions["_agent"] == "Claude Code"

    # temporal span
    ca = pd.to_datetime(sessions["created_at"], utc=True, errors="coerce")
    R["temporal"] = {"min": str(ca.min()), "max": str(ca.max()),
                     "n_dated": int(ca.notna().sum()),
                     "span_days": int((ca.max() - ca.min()).days) if ca.notna().any() else None,
                     "by_month": {str(k): int(v) for k, v in
                                  ca.dt.to_period("M").value_counts().sort_index().items()}}

    # Table 2: units
    conv = pd.read_parquet(find_parquet(args.data_dir, "conversations"),
                           columns=["session_id", "turn_type", "is_conversational",
                                    "prompt_pushback", "prompt_intent"])
    n_entries = len(conv)
    n_conv_direct = int(conv["is_conversational"].fillna(False).sum())
    n_up = int((conv["turn_type"] == "user_prompt").sum())
    n_ar = int((conv["turn_type"] == "assistant_response").sum())
    R["units"] = {
        "transcript_entries": n_entries,
        "conversational_turns_direct": n_conv_direct,
        "conversational_turns_derived": n_up + n_ar,
        "derived_matches_direct": bool(n_conv_direct == n_up + n_ar),
        "user_prompts": n_up, "assistant_responses": n_ar,
        "user_prompt_share_of_entries": round(n_up / n_entries, 4),
        "per_session": {
            "transcript_entries": med(conv.groupby("session_id").size()),
            "conversational_turns": med(conv[conv["is_conversational"].fillna(False)]
                                        .groupby("session_id").size()),
            "user_prompts": med(conv[conv["turn_type"] == "user_prompt"]
                                .groupby("session_id").size())},
        "sessions_turn_count_col": med(sessions.get("turn_count")),
        "sessions_prompt_count_col": med(sessions.get("prompt_count"))}

    # Table 3: developers
    uid = sessions["user_id"]
    has_uid = uid.notna() & (uid.astype(str).str.strip() != "")
    per_user = sessions[has_uid].groupby("user_id")["session_id"].nunique()
    R["developers"] = {"n_distinct": int(per_user.shape[0]),
                       "sessions_attributed": int(has_uid.sum()),
                       "attribution_coverage": round(float(has_uid.mean()), 4),
                       "sessions_per_user": med(per_user),
                       "top5_share": round(float(per_user.nlargest(5).sum() / per_user.sum()), 4)}

    # Table 3: agents
    vc = sessions["_agent"].value_counts()
    R["agents"] = {"counts": {k: int(v) for k, v in vc.items()},
                   "shares": {k: round(float(v / len(sessions)), 4) for k, v in vc.items()},
                   "n_distinct_after_norm": int(vc.shape[0]),
                   "users_per_agent": {k: int(v) for k, v in
                                       sessions[has_uid].groupby("_agent")["user_id"].nunique().items()}}

    # Table 3: checkpoints
    ck_n = sessions["checkpoint_ids"].map(jlen)
    R["checkpoints"] = {"per_session": med(ck_n),
                        "frac_sessions_multi": round(float((ck_n > 1).mean()), 4),
                        "n_sessions_multi": int((ck_n > 1).sum()),
                        "checkpoints_count_col": med(sessions.get("checkpoints_count"))}

    # Table 3: repositories
    R["repositories"] = {
        "n": int(len(repos)),
        "domain": {str(k): int(v) for k, v in repos["repo_type_domain"].fillna("<null>").value_counts().items()},
        "audience": {str(k): int(v) for k, v in repos["repo_type_audience"].fillna("<null>").value_counts().items()},
        "is_fork": {str(k): int(v) for k, v in repos["is_fork"].value_counts().items()}}

    # Table 3: intent
    R["prompt_intent"] = {str(k): int(v) for k, v in
                          conv["prompt_intent"].dropna().value_counts().items()}

    # Table 3: personas, pooled vs split
    pv = sessions["user_persona"].fillna("<null>")
    R["personas"] = {
        "pooled": {str(k): int(v) for k, v in pv.value_counts().items()},
        "claude_code": {str(k): int(v) for k, v in pv[is_cc].value_counts().items()},
        "other_agents": {str(k): int(v) for k, v in pv[~is_cc].value_counts().items()},
        "other_rate_claude_code": round(int(((pv == "Other") & is_cc).sum()) / max(int(is_cc.sum()), 1), 4),
        "other_rate_non_claude_code": round(int(((pv == "Other") & ~is_cc).sum()) / max(int((~is_cc).sum()), 1), 4)}

    # pushback, pooled
    lab = conv[conv["prompt_pushback"].notna()]
    real_pb = {"correction", "failure_report", "rejection", "takeover",
               "requirement_change", "pacing_complaint"}
    pb = lab["prompt_pushback"].astype(str).isin(real_pb)
    R["pushback"] = {"n_labeled": int(len(lab)), "n_pushback": int(pb.sum()),
                     "rate_pooled": round(float(pb.mean()), 4),
                     "class_counts": {str(k): int(v) for k, v in lab["prompt_pushback"].value_counts().items()},
                     "labels_only_on_user_prompts": bool((lab["turn_type"] == "user_prompt").all())}

    # session_success: flat finding, or blunt instrument?
    ss = pd.to_numeric(sessions["session_success"], errors="coerce")
    R["session_success"] = {
        "n_scored": int(ss.notna().sum()), "summary": med(ss),
        "n_distinct_values": int(ss.dropna().nunique()),
        "top15_values": {str(int(k)): int(v) for k, v in ss.dropna().value_counts().head(15).items()},
        "share_at_mode": round(float(ss.value_counts(normalize=True).iloc[0]), 4) if ss.notna().any() else None}

    tc = pd.to_numeric(sessions.get("turn_count"), errors="coerce")
    grp = pd.cut(tc, bins=[0, 2, 5, 10, 20, 50, np.inf],
                 labels=["1-2", "3-5", "6-10", "11-20", "21-50", "51+"])
    agg = (sessions.assign(_g=grp, _s=ss).dropna(subset=["_s"])
           .groupby("_g", observed=True)["_s"].agg(n="size", mean="mean", median="median"))
    R["success_by_length_pooled"] = {str(k): {"n": int(r["n"]), "mean": round(float(r["mean"]), 1),
                                              "median": round(float(r["median"]), 1)}
                                     for k, r in agg.iterrows()}

    with open(os.path.join(args.out_dir, "intro_numbers.json"), "w") as f:
        json.dump(R, f, indent=2, default=str)

    u = R["units"]
    print("=" * 66)
    print(f"SPAN  {R['temporal']['min']} -> {R['temporal']['max']}  ({R['temporal']['span_days']} days)")
    print("\nTABLE 2 — UNITS")
    print(f"  transcript entries   {u['transcript_entries']:>10,}  median/session {u['per_session']['transcript_entries']['median']}")
    print(f"  conversational turns {u['conversational_turns_direct']:>10,}  median/session {u['per_session']['conversational_turns']['median']}"
          f"  (derived={u['conversational_turns_derived']:,} match={u['derived_matches_direct']})")
    print(f"  user prompts         {u['user_prompts']:>10,}  median/session {u['per_session']['user_prompts']['median']}")
    print(f"  sessions.turn_count median={u['sessions_turn_count_col']['median']}  prompt_count median={u['sessions_prompt_count_col']['median']}")
    print("\nTABLE 3 — COMPOSITION")
    d = R["developers"]
    print(f"  developers {d['n_distinct']}, coverage {d['attribution_coverage']}, median {d['sessions_per_user']['median']}, max {d['sessions_per_user']['max']}, top5 share {d['top5_share']}")
    print(f"  agents {R['agents']['counts']}")
    print(f"  checkpoints/session median {R['checkpoints']['per_session']['median']}, frac multi {R['checkpoints']['frac_sessions_multi']}")
    print(f"  repo domain {R['repositories']['domain']}")
    print(f"  repo audience {R['repositories']['audience']}")
    print(f"  intent {R['prompt_intent']}")
    print(f"  personas pooled {R['personas']['pooled']}")
    print(f"  'Other' rate  CC={R['personas']['other_rate_claude_code']}  non-CC={R['personas']['other_rate_non_claude_code']}")
    print(f"  pushback {R['pushback']['rate_pooled']} ({R['pushback']['n_pushback']:,}/{R['pushback']['n_labeled']:,})")
    print("\nSESSION_SUCCESS")
    print(f"  distinct values {R['session_success']['n_distinct_values']}, share at mode {R['session_success']['share_at_mode']}")
    print(f"  top values {R['session_success']['top15_values']}")
    print(f"  by length {R['success_by_length_pooled']}")
    print("=" * 66)

if __name__ == "__main__":
    main()