#!/usr/bin/env python3
"""
commit_report.py -- SWE-chat

Answers two separate questions and shows they give different answers:

  (a) CHECKPOINT-WISE : which commits does conversations.checkpoint_pk link
      this session to?
  (b) TIME-WISE       : if you instead walk commits.author_date around the
      session window, which commits do you land on?

Then deep-dives any commit you name: message, per-file numstat, attribution
class counts, whether agent_version is populated, and which checkpoint /
sessions it belongs to.

Usage
  python3 commit_report.py --data-dir ./swechat_data \
      --session-id 3dded5ac-a667-436b-a093-ad8efbdf0e31 \
      --commits 1f780fac,89ad18a3,a46f6c26 --pad-hours 6
"""

import argparse
import json
import os

import pandas as pd


def jloads(x, default=None):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return default
    if isinstance(x, (list, dict)):
        return x
    try:
        return json.loads(x)
    except (json.JSONDecodeError, TypeError):
        return default


def parse_numstat(raw):
    """commits.numstat is CSV with header: insertions,deletions,path"""
    out = {}
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return out
    for i, line in enumerate(str(raw).splitlines()):
        line = line.strip()
        if not line:
            continue
        if i == 0 and line.lower().startswith(("insertions", "additions")):
            continue                                   # header row
        parts = line.split(",", 2) if "," in line else line.split("\t", 2)
        if len(parts) < 3:
            continue
        a, d, path = parts[0].strip(), parts[1].strip(), parts[2].strip()
        try:
            out[path] = [0 if a == "-" else int(a), 0 if d == "-" else int(d)]
        except ValueError:
            continue
    return out


def attribution_detail(raw):
    """-> {path: (class, has_agent_version, has_committed_version)}"""
    obj = jloads(raw, {})
    out = {}
    if isinstance(obj, dict):
        for path, v in obj.items():
            if isinstance(v, dict):
                out[path] = (v.get("attribution", "?"),
                             v.get("agent_version") is not None,
                             v.get("committed_version") is not None)
            else:
                out[path] = (str(v), False, False)
    return out


def to_utc(ts):
    """Normalise to a tz-aware UTC Timestamp whether naive or already aware."""
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def short(x, n=8):
    return str(x)[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./swechat_data")
    ap.add_argument("--session-id", required=True)
    ap.add_argument("--commits", default="",
                    help="comma-separated sha prefixes to deep-dive")
    ap.add_argument("--pad-hours", type=float, default=6.0)
    ap.add_argument("--out-dir", default="./commit_report")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    conv = pd.read_parquet(f"{args.data_dir}/conversations.parquet")
    commits = pd.read_parquet(f"{args.data_dir}/commits.parquet")
    try:
        checkpoints = pd.read_parquet(f"{args.data_dir}/checkpoints.parquet")
    except FileNotFoundError:
        checkpoints = None

    s = conv[conv.session_id == args.session_id].sort_values("turn_number")
    if s.empty:
        raise SystemExit("session not found in conversations")

    cps = list(s.checkpoint_pk.dropna().unique())
    repo = s.repo_id.dropna().iloc[0] if s.repo_id.notna().any() else None
    user = s.user_id.dropna().iloc[0] if s.user_id.notna().any() else None
    t0, t1 = s.timestamp.min(), s.timestamp.max()

    print(f"session      {args.session_id}")
    print(f"repo         {repo}")
    print(f"user_id      {user}")
    print(f"turns        {len(s):,}  (turn_number {s.turn_number.min()}–{s.turn_number.max()})")
    print(f"window       {t0}  ->  {t1}")
    print(f"checkpoint   {cps}")

    # ---------- (a) checkpoint-wise -------------------------------------
    ck = commits[commits.checkpoint_pk.isin(cps)].drop_duplicates("commit_sha")
    ck = ck.sort_values("author_date")
    print(f"\n(a) CHECKPOINT-WISE -- commits linked via checkpoint_pk: {len(ck)}")
    for _, c in ck.iterrows():
        print(f"    {short(c.commit_sha)}  {str(c.author_date)[:19]}  "
              f"{c.files_changed_count:>3} files  "
              f"+{c.total_additions:<6}-{c.total_deletions:<6}  "
              f"{str(c.commit_message).splitlines()[0][:58]}")

    # ---------- (b) time-wise -------------------------------------------
    pad = pd.Timedelta(hours=args.pad_hours)
    tw = commits.copy()
    if repo is not None and "repo_id" in tw.columns:
        tw = tw[tw.repo_id == repo]
    tw["author_date"] = pd.to_datetime(tw.author_date, utc=True, errors="coerce")
    lo = to_utc(t0) - pad
    hi = to_utc(t1) + pad
    tw = tw[(tw.author_date >= lo) & (tw.author_date <= hi)]
    tw = tw.drop_duplicates("commit_sha").sort_values("author_date")

    linked = set(ck.commit_sha)
    print(f"\n(b) TIME-WISE -- commits in {repo} within ±{args.pad_hours}h "
          f"of the session window: {len(tw)}")
    print(f"    {'sha':<10}{'author_date':<26}{'in ckpt':<9}{'author':<18}files  churn")
    for _, c in tw.iterrows():
        mark = "YES" if c.commit_sha in linked else "-"
        inwin = "in" if lo + pad <= c.author_date <= hi - pad else "out"
        print(f"    {short(c.commit_sha):<10}{str(c.author_date)[:19]:<26}"
              f"{mark:<9}{str(c.get('user_id'))[:16]:<18}"
              f"{c.files_changed_count:>4}   +{c.total_additions}/-{c.total_deletions}"
              f"   [{inwin} window]")

    only_time = set(tw.commit_sha) - linked
    only_ckpt = linked - set(tw.commit_sha)
    print(f"\n    reachable by time but NOT by checkpoint: {len(only_time)}"
          + (f"  {[short(x) for x in sorted(only_time)]}" if only_time else ""))
    print(f"    reachable by checkpoint but NOT in time window: {len(only_ckpt)}"
          + (f"  {[short(x) for x in sorted(only_ckpt)]}" if only_ckpt else ""))

    tw.assign(checkpoint_linked=tw.commit_sha.isin(linked))[
        ["commit_sha", "author_date", "user_id", "checkpoint_pk",
         "files_changed_count", "total_additions", "total_deletions",
         "commit_message", "checkpoint_linked"]
    ].to_csv(f"{args.out_dir}/commits_timewise.csv", index=False)

    # ---------- (c) deep dive -------------------------------------------
    wanted = [x.strip() for x in args.commits.split(",") if x.strip()]
    rows = []
    for pref in wanted:
        m = commits[commits.commit_sha.astype(str).str.startswith(pref)]
        print(f"\n{'=' * 74}\n=== {pref} ===")
        if m.empty:
            print("    NOT FOUND in commits.parquet")
            continue
        c = m.iloc[0]
        msg = str(c.commit_message)
        print(f"    sha            {c.commit_sha}")
        print(f"    author_date    {c.author_date}")
        print(f"    user_id        {c.get('user_id')}   "
              f"github={c.get('github_username')}  is_agent={c.get('is_agent_author')}")
        print(f"    branch         {c.get('branch')}")
        print(f"    checkpoint_pk  {c.checkpoint_pk}"
              + ("   <-- THIS SESSION" if c.checkpoint_pk in cps else
                 "   <-- different checkpoint"))
        print(f"    commit_index   {c.get('commit_index')} of {c.get('num_commits')}")
        print(f"    files/churn    {c.files_changed_count} files, "
              f"+{c.total_additions}/-{c.total_deletions}")
        print(f"    message        {msg.splitlines()[0][:70]}")
        if len(msg.splitlines()) > 1:
            body = " ".join(msg.splitlines()[1:]).strip()
            print(f"                   {body[:140]}")
        if msg.lstrip().startswith("*") or "\n* " in msg or "(#" in msg.splitlines()[0]:
            print("    ! looks like a squashed PR merge")

        # which sessions share this commit's checkpoint
        if checkpoints is not None:
            cp = checkpoints[checkpoints.checkpoint_pk == c.checkpoint_pk]
            if not cp.empty:
                sess = jloads(cp.iloc[0].get("session_pks"), []) or []
                print(f"    checkpoint has {len(sess)} session(s); "
                      f"this session {'IS' if args.session_id in sess else 'is NOT'} among them")
                print(f"    checkpoint commit_count = {cp.iloc[0].get('commit_count')}")

        nstat = parse_numstat(c.numstat)
        print(f"    numstat parsed: {len(nstat)} files")
        for p, (a, d) in sorted(nstat.items(), key=lambda kv: -kv[1][0])[:8]:
            print(f"        +{a:<6}-{d:<6} {p}")

        att = attribution_detail(c.file_attribution)
        classes = pd.Series([v[0] for v in att.values()]).value_counts().to_dict() if att else {}
        n_agent_ver = sum(1 for v in att.values() if v[1])
        n_comm_ver = sum(1 for v in att.values() if v[2])
        print(f"    file_attribution: {len(att)} files  classes={classes}")
        print(f"        agent_version populated:     {n_agent_ver}/{len(att)}")
        print(f"        committed_version populated: {n_comm_ver}/{len(att)}")

        ac = jloads(c.agent_changes, []) or []
        print(f"    agent_changes: {len(ac)} entries"
              + ("   <-- EMPTY: attribution had nothing to match against"
                 if len(ac) == 0 else ""))
        if ac and isinstance(ac[0], dict):
            print(f"        keys: {sorted(ac[0].keys())}")

        rows.append({
            "sha": c.commit_sha, "author_date": c.author_date,
            "checkpoint_pk": c.checkpoint_pk,
            "in_this_session_checkpoint": c.checkpoint_pk in cps,
            "files": c.files_changed_count,
            "additions": c.total_additions, "deletions": c.total_deletions,
            "message": msg.splitlines()[0],
            "attribution_classes": json.dumps(classes),
            "agent_version_populated": n_agent_ver,
            "n_agent_changes": len(ac),
        })

    if rows:
        pd.DataFrame(rows).to_csv(f"{args.out_dir}/commit_deepdive.csv", index=False)
    print(f"\nwrote {args.out_dir}/commits_timewise.csv"
          + (" and commit_deepdive.csv" if rows else ""))


if __name__ == "__main__":
    main()