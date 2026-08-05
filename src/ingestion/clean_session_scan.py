#!/usr/bin/env python3
"""
clean_session_scan.py  --  whole-dataset scan of SWE-chat

Counts sessions usable for session-level preference -> code attribution, i.e.
sessions where the committed code can be credited to this session alone.

A (session, commit) pair is CLEAN when all of:
  1. the commit's checkpoint references exactly one session
  2. no OTHER session in the same repo was active in the window
     (previous commit in repo, this commit] AND touched any of the same files
  3. the commit is not a squashed PR merge (those fold several sessions'
     work into one commit)

Reports how many sessions survive each filter, and writes the clean list.

Usage
  python3 clean_session_scan.py --data-dir ./swechat_data --out-dir ./scan_out
  python3 clean_session_scan.py --data-dir ./swechat_data --sample-repos 20
"""

import argparse
import json
import os
import re
from collections import defaultdict

import pandas as pd

SQUASH = re.compile(r"\(#\d+\)\s*$")


def jloads(x, default=None):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return default
    if isinstance(x, (list, dict)):
        return x
    try:
        return json.loads(x)
    except (json.JSONDecodeError, TypeError):
        return default


def to_utc(s):
    s = pd.to_datetime(s, utc=True, errors="coerce")
    return s


def commit_paths(row):
    """Repo-relative paths from files_changed (name-status) or numstat."""
    out = set()
    raw = row.get("files_changed")
    for line in str(raw or "").splitlines():
        parts = line.split("\t") if "\t" in line else line.split(None, 1)
        if len(parts) >= 2:
            out.add(parts[-1].strip())
    if out:
        return out
    for i, line in enumerate(str(row.get("numstat") or "").splitlines()):
        line = line.strip()
        if not line or (i == 0 and line.lower().startswith(("insertions", "additions"))):
            continue
        parts = line.split(",", 2) if "," in line else line.split("\t", 2)
        if len(parts) >= 3:
            out.add(parts[2].strip())
    return out


def is_squash(msg):
    first = str(msg or "").splitlines()[0] if str(msg or "").strip() else ""
    body = str(msg or "")
    return bool(SQUASH.search(first)) or "\n* " in body


def basenames(paths):
    return {os.path.basename(str(p)) for p in paths if p}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./swechat_data")
    ap.add_argument("--out-dir", default="./scan_out")
    ap.add_argument("--sample-repos", type=int, default=0,
                    help="limit to N repos for a quick run")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    sessions = pd.read_parquet(
        f"{args.data_dir}/sessions.parquet",
        columns=["session_id", "repo_id", "files_touched", "files_touched_count",
                 "created_at", "duration_seconds", "canonical_checkpoint_pk",
                 "checkpoint_ids"])
    commits = pd.read_parquet(
        f"{args.data_dir}/commits.parquet",
        columns=["commit_sha", "checkpoint_pk", "repo_id", "author_date",
                 "user_id", "commit_message", "files_changed", "numstat",
                 "files_changed_count", "num_commits"])
    checkpoints = pd.read_parquet(
        f"{args.data_dir}/checkpoints.parquet",
        columns=["checkpoint_pk", "session_pks", "session_count", "commit_count"])

    print(f"sessions {len(sessions):,}  commits {len(commits):,}  "
          f"checkpoints {len(checkpoints):,}")

    if args.sample_repos:
        keep = sessions.repo_id.dropna().unique()[:args.sample_repos]
        sessions = sessions[sessions.repo_id.isin(keep)]
        commits = commits[commits.repo_id.isin(keep)]
        print(f"sampled to {len(keep)} repos: {len(sessions):,} sessions")

    sessions["start"] = to_utc(sessions.created_at)
    sessions["end"] = sessions.start + pd.to_timedelta(
        sessions.duration_seconds.fillna(0), unit="s")
    sessions["files"] = sessions.files_touched.map(
        lambda x: basenames(jloads(x, []) or []))

    commits["adate"] = to_utc(commits.author_date)
    commits = commits.dropna(subset=["adate"]).sort_values(["repo_id", "adate"])
    commits["prev_adate"] = commits.groupby("repo_id").adate.shift(1)
    commits["squash"] = commits.commit_message.map(is_squash)

    ck_sessions = {r.checkpoint_pk: (jloads(r.session_pks, []) or [])
                   for r in checkpoints.itertuples()}
    ck_count = dict(zip(checkpoints.checkpoint_pk, checkpoints.session_count))

    files_by_sid = dict(zip(sessions.session_id, sessions.files))

    sess_by_repo = defaultdict(list)
    for r in sessions.itertuples():
        sess_by_repo[r.repo_id].append(r)

    rows = []
    for c in commits.itertuples():
        sess_ids = ck_sessions.get(c.checkpoint_pk, [])
        n_sess = ck_count.get(c.checkpoint_pk, len(sess_ids))
        if not sess_ids:
            continue

        single = (n_sess == 1)
        lo = c.prev_adate if pd.notna(c.prev_adate) else c.adate - pd.Timedelta(days=1)
        cfiles = basenames(commit_paths(c._asdict()))

        for sid in sess_ids:
            sfiles = files_by_sid.get(sid, set())

            # other sessions in this repo overlapping the commit window
            collide, n_concurrent = set(), 0
            for o in sess_by_repo.get(c.repo_id, []):
                if o.session_id == sid or pd.isna(o.start):
                    continue
                if o.end < lo or o.start > c.adate:
                    continue
                n_concurrent += 1
                collide |= (o.files & (sfiles | cfiles))

            rows.append(dict(
                session_id=sid, commit_sha=c.commit_sha, repo_id=c.repo_id,
                checkpoint_pk=c.checkpoint_pk, checkpoint_sessions=n_sess,
                single_session_checkpoint=single,
                is_squash=bool(c.squash),
                concurrent_sessions=n_concurrent,
                colliding_files=len(collide),
                commit_files=len(cfiles),
                clean=bool(single and not c.squash and not collide),
            ))

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("no (session, commit) pairs built")
    df.to_csv(f"{args.out_dir}/session_commit_pairs.csv", index=False)

    n_pairs = len(df)
    n_sess = df.session_id.nunique()
    print(f"\n(session, commit) pairs: {n_pairs:,}   distinct sessions: {n_sess:,}")

    print("\nfilter cascade (pairs):")
    step = df
    for label, mask in [
        ("all pairs with a linked commit", pd.Series(True, index=df.index)),
        ("checkpoint has exactly 1 session", df.single_session_checkpoint),
        ("  ... and not a squashed PR merge", df.single_session_checkpoint & ~df.is_squash),
        ("  ... and no concurrent session touched the same files",
         df.single_session_checkpoint & ~df.is_squash & df.colliding_files.eq(0)),
    ]:
        sub = df[mask]
        print(f"  {label:<56} {len(sub):>7,} pairs  "
              f"{sub.session_id.nunique():>6,} sessions  "
              f"({len(sub) / n_pairs:>5.1%})")

    clean_sessions = set(df[df.clean].session_id)
    print(f"\nCLEAN sessions: {len(clean_sessions):,} of {n_sess:,} "
          f"({len(clean_sessions) / n_sess:.1%})")

    print("\nwhy pairs are excluded:")
    ex = df[~df.clean]
    print(f"  multi-session checkpoint      {(~ex.single_session_checkpoint).sum():>7,}")
    print(f"  squashed PR merge             {ex.is_squash.sum():>7,}")
    print(f"  concurrent session file clash {ex.colliding_files.gt(0).sum():>7,}")

    print(f"\nsquash rate overall: {df.is_squash.mean():.1%} of pairs")
    print(f"median concurrent sessions per commit window: "
          f"{df.concurrent_sessions.median():.0f}")

    pd.DataFrame(sorted(clean_sessions), columns=["session_id"]).to_csv(
        f"{args.out_dir}/clean_sessions.csv", index=False)
    print(f"\nwrote {args.out_dir}/clean_sessions.csv and session_commit_pairs.csv")


if __name__ == "__main__":
    main()