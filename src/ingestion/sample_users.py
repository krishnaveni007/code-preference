#!/usr/bin/env python3
"""
Source 1 sampling: select users and sessions for chat preference-vector
scoring, BEFORE spending any LLM budget.

Design (per plan discussion):
  - "Nitpicker" candidacy is NOT read from sessions.user_persona. That
    label is ~82% "Other" outside Claude Code (unreliable there), so
    gating sampling on it would silently make the sample Claude-Code-only
    without saying so. Instead we compute each developer's own pushback
    rate across their own labeled user prompts as a data-driven proxy --
    the same underlying signal "Expert Nitpicker" was approximating, but
    computed directly rather than trusted from an LLM label.
  - Session eligibility: user-prompt count in [3, 40]. (History: [8,30]
    only cleared 47/189 developers; [5,30] got to 65; widened to [3,40]
    with min_eligible_sessions=5 to clear the 80-developer target at 87.)
  - Within a user's eligible sessions, prefer higher pushback-turn count.
  - TWO output arms, not one:
      arm=high_signal  -> top N_USERS by nitpicker score (your original ask)
      arm=random_control -> a random sample of users, same session-selection
                             rules, for an unbiased discriminability check
    Selecting only on signal and then measuring discriminability on that
    same selection inflates between-user variance by construction; the
    control arm exists so the doc can report a number that isn't
    circular.

Usage:
    python sample_users_sessions.py --data-dir ./swechat_data \
        --out-dir ./source1_sample --n-users 80 --n-control 20 \
        --sessions-per-user-min 5 --sessions-per-user-max 10
"""

import argparse, glob, json, os
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


REAL_PUSHBACK = {"correction", "failure_report", "rejection", "takeover",
                 "requirement_change", "pacing_complaint"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", default="./source1_sample")
    ap.add_argument("--n-users", type=int, default=80)
    ap.add_argument("--n-control", type=int, default=20,
                    help="size of the unbiased random-sample control arm")
    ap.add_argument("--sessions-per-user-min", type=int, default=5)
    ap.add_argument("--sessions-per-user-max", type=int, default=10)
    ap.add_argument("--min-user-prompts", type=int, default=3)
    ap.add_argument("--max-user-prompts", type=int, default=40)
    ap.add_argument("--min-eligible-sessions", type=int, default=5,
                    help="a user must have at least this many eligible "
                         "sessions to be selectable for the HIGH-SIGNAL arm")
    ap.add_argument("--control-min-eligible-sessions", type=int, default=3,
                    help="looser floor for the CONTROL arm -- it doesn't "
                         "need to be nitpicker-selected, just needs enough "
                         "sessions to score, and drawing it from a separate "
                         "(laxer) pool means it isn't limited to whatever "
                         "the high-signal arm didn't already take")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print("loading sessions ...")
    sessions = pd.read_parquet(find_parquet(args.data_dir, "sessions"))
    uid = sessions["user_id"]
    has_uid = uid.notna() & (uid.astype(str).str.strip() != "")

    print("loading conversations (user_prompt rows only) ...")
    conv = pd.read_parquet(
        find_parquet(args.data_dir, "conversations"),
        columns=["session_id", "turn_type", "prompt_pushback"])
    up = conv[conv["turn_type"] == "user_prompt"].copy()
    up["is_pushback"] = up["prompt_pushback"].astype(str).isin(REAL_PUSHBACK)
    up["is_labeled"] = up["prompt_pushback"].notna()

    # ---------------------------------------------- per-session eligibility
    per_sess = up.groupby("session_id").agg(
        n_user_prompts=("prompt_pushback", "size"),
        n_pushback=("is_pushback", "sum"),
        n_labeled=("is_labeled", "sum"))
    per_sess = per_sess.merge(
        sessions[["session_id", "user_id", "agent", "user_persona"]]
        .rename(columns={"user_id": "session_user_id"}),
        on="session_id", how="left")
    per_sess = per_sess[per_sess["session_user_id"].notna() &
                        (per_sess["session_user_id"].astype(str).str.strip() != "")]

    eligible = per_sess[
        (per_sess["n_user_prompts"] >= args.min_user_prompts) &
        (per_sess["n_user_prompts"] <= args.max_user_prompts)
    ].copy()

    print(f"sessions total: {len(sessions)}, developer-attributed: "
         f"{int(has_uid.sum())}, eligible ({args.min_user_prompts}-"
         f"{args.max_user_prompts} user prompts): {len(eligible)}")

    # ---------------------------------------------- per-developer nitpicker proxy
    # Computed over ALL of a developer's labeled prompts corpus-wide, not
    # just their eligible sessions, so the score isn't circular with the
    # session filter itself.
    dev_pb = up[up["is_labeled"]].merge(
        sessions[["session_id", "user_id"]], on="session_id", how="left")
    dev_pb = dev_pb[dev_pb["user_id"].notna()]
    nitpicker = dev_pb.groupby("user_id").agg(
        n_labeled_prompts=("is_pushback", "size"),
        pushback_rate=("is_pushback", "mean"))
    # require a minimum labeled-prompt count so the rate isn't noise from 1-2 prompts
    nitpicker = nitpicker[nitpicker["n_labeled_prompts"] >= 10]

    # persona cross-tab, reported but NOT used for selection
    persona_mix = (sessions[has_uid].groupby("user_id")["user_persona"]
                   .agg(lambda s: s.value_counts().idxmax()
                        if s.notna().any() else "<null>"))
    nitpicker = nitpicker.join(persona_mix.rename("modal_persona"), how="left")

    # ---------------------------------------------- eligible-session counts per user
    elig_per_user = eligible.groupby("session_user_id").size()
    nitpicker = nitpicker.join(elig_per_user.rename("n_eligible_sessions"), how="left")
    nitpicker["n_eligible_sessions"] = nitpicker["n_eligible_sessions"].fillna(0).astype(int)

    selectable = nitpicker[nitpicker["n_eligible_sessions"] >= args.min_eligible_sessions]
    print(f"\ndevelopers with >={args.min_eligible_sessions} eligible sessions "
         f"and >=10 labeled prompts: {len(selectable)}")

    # ---------------------------------------------- feasibility diagnostics
    # Two curves, computed ONCE so parameter tuning doesn't require a
    # fresh full rerun each time: (a) how the selectable-developer count
    # responds to loosening the min-eligible-sessions floor at the CURRENT
    # band, and (b) how it responds to widening the user-prompt band
    # itself, holding the session-count floor fixed. The band is usually
    # the bigger lever -- it determines the eligible-session pool that
    # everything else draws from.
    print("\n--- feasibility curve: min_eligible_sessions @ band "
         f"[{args.min_user_prompts},{args.max_user_prompts}] ---")
    curve_floor = {}
    for k in (1, 2, 3, 4, 5, 6, 8, 10):
        n = int((nitpicker["n_eligible_sessions"] >= k).sum())
        curve_floor[k] = n
        print(f"  min_eligible_sessions={k:>3}: {n:>4} developers")

    print("\n--- feasibility curve: user-prompt band width "
         f"(min_eligible_sessions={args.min_eligible_sessions}) ---")
    band_candidates = [(8, 30), (6, 30), (5, 30), (4, 35), (3, 40), (5, 999)]
    curve_band = {}
    for lo, hi in band_candidates:
        elig_band = per_sess[(per_sess["n_user_prompts"] >= lo) &
                             (per_sess["n_user_prompts"] <= hi)]
        elig_per_user_band = elig_band.groupby("session_user_id").size()
        n_selectable = int((nitpicker.index.map(
            lambda u: elig_per_user_band.get(u, 0)) >= args.min_eligible_sessions).sum())
        curve_band[f"{lo}-{hi}"] = {
            "eligible_sessions": int(len(elig_band)),
            "selectable_developers": n_selectable,
        }
        print(f"  band [{lo:>3},{hi:>3}]: {len(elig_band):>5} eligible sessions, "
             f"{n_selectable:>4} developers clear min_eligible_sessions="
             f"{args.min_eligible_sessions}")

    if len(selectable) < args.n_users:
        print(f"\n** WARNING: only {len(selectable)} developers are selectable, "
             f"short of the requested {args.n_users}. See the curves above "
             f"for what --min-eligible-sessions or a wider band would buy you. **")

    # ---------------------------------------------- arm 1: high signal
    high_signal_users = (selectable.sort_values("pushback_rate", ascending=False)
                         .head(args.n_users))

    # ---------------------------------------------- arm 2: random control
    # Drawn from its OWN pool (looser session-count floor via
    # --control-min-eligible-sessions), not from whatever high-signal left
    # behind. With a small overall selectable pool, "leftovers" can easily
    # be empty even though a genuinely separate, less-restrictive pool
    # exists. Still excludes anyone already placed in the high-signal arm,
    # so the two arms never overlap.
    control_pool = nitpicker[
        (nitpicker["n_eligible_sessions"] >= args.control_min_eligible_sessions) &
        (~nitpicker.index.isin(high_signal_users.index))
    ]
    n_ctrl = min(args.n_control, len(control_pool))
    if n_ctrl < args.n_control:
        print(f"** WARNING: only {n_ctrl} developers available for the "
             f"control arm at control_min_eligible_sessions="
             f"{args.control_min_eligible_sessions} (requested "
             f"{args.n_control}). **")
    control_idx = (rng.choice(control_pool.index, size=n_ctrl, replace=False)
                  if n_ctrl else [])
    control_users = control_pool.loc[control_idx]

    def select_sessions_for_user(user_id):
        u_sessions = eligible[eligible["session_user_id"] == user_id].sort_values(
            "n_pushback", ascending=False)
        n = min(max(len(u_sessions), args.sessions_per_user_min),
               args.sessions_per_user_max)
        n = min(n, len(u_sessions))
        n = min(n, args.sessions_per_user_max)
        n = max(n, min(args.sessions_per_user_min, len(u_sessions)))
        return u_sessions.head(n)

    def build_arm(users_df, arm_name):
        rows = []
        for uid_ in users_df.index:
            sess = select_sessions_for_user(uid_)
            for _, r in sess.iterrows():
                rows.append({
                    "arm": arm_name, "user_id": uid_,
                    "session_id": r["session_id"],
                    "n_user_prompts": int(r["n_user_prompts"]),
                    "n_pushback": int(r["n_pushback"]),
                    "agent": r["agent"],
                    "nitpicker_pushback_rate": round(
                        float(users_df.loc[uid_, "pushback_rate"]), 4),
                    "modal_persona": users_df.loc[uid_, "modal_persona"],
                })
        return pd.DataFrame(rows)

    high_arm = build_arm(high_signal_users, "high_signal")
    ctrl_arm = build_arm(control_users, "random_control")
    selection = pd.concat([high_arm, ctrl_arm], ignore_index=True)
    selection.to_csv(os.path.join(args.out_dir, "selected_sessions.csv"), index=False)

    # ---------------------------------------------- summary
    summary = {
        "eligible_sessions_total": int(len(eligible)),
        "selectable_developers": int(len(selectable)),
        "feasibility_curve_min_eligible_sessions": curve_floor,
        "feasibility_curve_band_width": curve_band,
        "high_signal_arm": {
            "n_users_requested": args.n_users,
            "n_users_selected": int(len(high_signal_users)),
            "n_sessions_selected": int(len(high_arm)),
            "sessions_per_user": high_arm.groupby("user_id").size().describe()
            .round(2).to_dict() if len(high_arm) else {},
            "pushback_rate_range": [
                round(float(high_signal_users["pushback_rate"].min()), 4),
                round(float(high_signal_users["pushback_rate"].max()), 4)],
        },
        "random_control_arm": {
            "n_users_requested": args.n_control,
            "n_users_selected": int(len(control_users)),
            "n_sessions_selected": int(len(ctrl_arm)),
            "pushback_rate_range": [
                round(float(control_users["pushback_rate"].min()), 4),
                round(float(control_users["pushback_rate"].max()), 4)]
            if len(control_users) else None,
        },
        "persona_agreement_check": {
            "note": "modal_persona reported for comparison only; NOT used "
                    "for selection. High overlap with 'Expert Nitpicker' "
                    "among the high-signal arm would be a nice sanity check, "
                    "not a requirement.",
            "high_signal_persona_counts": high_signal_users["modal_persona"]
            .value_counts().to_dict() if len(high_signal_users) else {},
        },
        "estimated_llm_calls": {
            "note": "1 screen call + ~1-3 direction calls per triggered "
                    "turn, per user prompt scored",
            "total_user_prompts_to_score": int(selection["n_user_prompts"].sum()),
            "rough_call_estimate_low": int(selection["n_user_prompts"].sum() * 2),
            "rough_call_estimate_high": int(selection["n_user_prompts"].sum() * 4),
        },
    }
    with open(os.path.join(args.out_dir, "sampling_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n" + json.dumps(summary, indent=2, default=str))
    print(f"\nwrote {args.out_dir}/selected_sessions.csv, sampling_summary.json")


if __name__ == "__main__":
    main()