#!/usr/bin/env python3
"""
original_pipeline.py -- runs the three-stage approach end to end on one session
and reports where each stage fails.

  Stage 1  nearest-subsequent-commit: for every agent write-turn, find the
           first commit by that author in that repo after the turn timestamp.
  Stage 2  line-mapping survival: intersect the agent's edited file with the
           commit's files, then check how many of the agent's written lines
           appear in the commit (both in the patch's added lines and in
           file_attribution.committed_version).
  Stage 3  code-state reconstruction: seed each file from its first Read
           tool_result, replay every agent edit in turn order, and record the
           state at each turn plus every point where the replay diverges.

Outputs (in --out-dir)
  stage1_turn_to_commit.csv
  stage2_line_survival.csv
  stage3_replay_states.csv
  stage3_replay_failures.csv

Usage
  python3 original_pipeline.py --data-dir ./swechat_data \
      --session-id 3dded5ac-a667-436b-a093-ad8efbdf0e31 --out-dir ./pipeline_out
"""

import argparse
import hashlib
import json
import os
import re
from collections import defaultdict

import pandas as pd

WRITE_TOOLS = {"write", "edit", "multiedit", "notebookedit",
               "write_file", "edit_file", "replace", "str_replace"}
READ_TOOLS = {"read", "read_file", "view"}

READ_PREFIX = re.compile(r"^\s*\d+\s*(?:→|->|\t)")


# ------------------------------------------------------------------ utils
def jloads(x, default=None):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return default
    if isinstance(x, (list, dict)):
        return x
    try:
        return json.loads(x)
    except (json.JSONDecodeError, TypeError):
        return default


def to_utc(ts):
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def norm_lines(text):
    """Comparable lines: strip trailing space, drop blanks and lone braces."""
    out = []
    for ln in str(text or "").splitlines():
        t = ln.rstrip()
        if not t.strip():
            continue
        if t.strip() in {"{", "}", "(", ")", "[", "]", "};", ");"}:
            continue
        out.append(t.strip())
    return out


def strip_read_prefix(text):
    """Claude Code Read returns '   12→line'; strip that to get raw content."""
    lines = str(text or "").splitlines()
    if not lines:
        return ""
    hits = sum(1 for ln in lines[:20] if READ_PREFIX.match(ln))
    if hits < max(3, len(lines[:20]) // 2):
        return str(text or "")
    return "\n".join(READ_PREFIX.sub("", ln) for ln in lines)


def parse_numstat(raw):
    out = {}
    for i, line in enumerate(str(raw or "").splitlines()):
        line = line.strip()
        if not line:
            continue
        if i == 0 and line.lower().startswith(("insertions", "additions")):
            continue
        parts = line.split(",", 2) if "," in line else line.split("\t", 2)
        if len(parts) < 3:
            continue
        try:
            out[parts[2].strip()] = [int(parts[0]), int(parts[1])]
        except ValueError:
            continue
    return out


def patch_added_lines(patch):
    """unified diff -> {path: set(normalised added lines)}"""
    out, cur = defaultdict(set), None
    for ln in str(patch or "").splitlines():
        if ln.startswith("+++ b/"):
            cur = ln[6:].strip()
        elif ln.startswith("diff --git"):
            cur = None
        elif cur and ln.startswith("+") and not ln.startswith("+++"):
            body = ln[1:].rstrip()
            if body.strip():
                out[cur].add(body.strip())
    return out


def committed_versions(file_attribution):
    obj = jloads(file_attribution, {}) or {}
    out = {}
    if isinstance(obj, dict):
        for path, v in obj.items():
            if isinstance(v, dict) and v.get("committed_version"):
                out[path] = v["committed_version"]
    return out


def attribution_classes(file_attribution):
    obj = jloads(file_attribution, {}) or {}
    return {p: (v.get("attribution") if isinstance(v, dict) else str(v))
            for p, v in obj.items()} if isinstance(obj, dict) else {}


def suffix_match(abs_path, candidates):
    """Absolute agent path -> repo-relative commit path."""
    a = str(abs_path).replace("\\", "/")
    for c in sorted(candidates, key=len, reverse=True):
        if a.endswith("/" + c) or a == c:
            return c
    base = os.path.basename(a)
    hits = [c for c in candidates if os.path.basename(c) == base]
    return hits[0] if len(hits) == 1 else None


def edit_payload(tool_input_json):
    """-> list of (old_string, new_string); Write returns ('', content)."""
    args = jloads(tool_input_json, {}) or {}
    if not isinstance(args, dict):
        return []
    subs = args.get("edits") or args.get("replacements")
    if isinstance(subs, list) and subs:
        return [(e.get("old_string") or e.get("old_str") or "",
                 e.get("new_string") or e.get("new_str") or "")
                for e in subs if isinstance(e, dict)]
    if args.get("content") is not None or args.get("file_text") is not None:
        return [("", args.get("content") or args.get("file_text") or "")]
    old = args.get("old_string") or args.get("old_str")
    new = args.get("new_string") or args.get("new_str")
    if old is not None or new is not None:
        return [(old or "", new or "")]
    return []


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./swechat_data")
    ap.add_argument("--session-id", required=True)
    ap.add_argument("--out-dir", default="./pipeline_out")
    ap.add_argument("--max-gap-hours", type=float, default=24.0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    conv = pd.read_parquet(f"{args.data_dir}/conversations.parquet")
    commits = pd.read_parquet(f"{args.data_dir}/commits.parquet")

    s = conv[conv.session_id == args.session_id].sort_values("turn_number")
    if s.empty:
        raise SystemExit("session not found")
    repo = s.repo_id.dropna().iloc[0]
    user = s.user_id.dropna().iloc[0]
    print(f"session {args.session_id}\nrepo {repo}   user {user}   "
          f"{len(s):,} turns\n")

    s = s.copy()
    s["ts"] = s.timestamp.map(to_utc)
    writes = s[s.turn_type.eq("tool_use")
               & s.tool_name.astype(str).str.lower().isin(WRITE_TOOLS)
               & s.file_path.notna()].copy()
    print(f"agent write-turns: {len(writes)} across "
          f"{writes.file_path.nunique()} file(s)")

    # candidate commits: same repo, same author, ordered by time
    cand = commits[(commits.repo_id == repo) & (commits.user_id == user)].copy()
    cand["adate"] = cand.author_date.map(to_utc)
    cand = cand.drop_duplicates("commit_sha").sort_values("adate").reset_index(drop=True)
    print(f"candidate commits by this author in repo: {len(cand)}\n")

    ckset = set(s.checkpoint_pk.dropna().unique())

    # ---------------- STAGE 1: nearest subsequent commit -----------------
    print("=" * 70)
    print("STAGE 1  nearest subsequent commit per agent write-turn")
    gap_cap = pd.Timedelta(hours=args.max_gap_hours)
    stage1 = []
    for _, w in writes.iterrows():
        later = cand[cand.adate >= w.ts]
        if later.empty:
            stage1.append(dict(turn_number=w.turn_number, file_path=w.file_path,
                               commit_sha=None, reason="no later commit"))
            continue
        c = later.iloc[0]
        if c.adate - w.ts > gap_cap:
            stage1.append(dict(turn_number=w.turn_number, file_path=w.file_path,
                               commit_sha=None, reason="gap exceeds cap"))
            continue
        nstat = parse_numstat(c.numstat)
        rel = suffix_match(w.file_path, nstat.keys())
        stage1.append(dict(
            turn_number=int(w.turn_number), turn_ts=w.ts,
            file_path=w.file_path, rel_path=rel,
            commit_sha=c.commit_sha, commit_ts=c.adate,
            gap_min=round((c.adate - w.ts).total_seconds() / 60, 1),
            commit_message=str(c.commit_message).splitlines()[0][:70],
            commit_files=int(c.files_changed_count),
            file_in_commit=rel is not None,
            checkpoint_linked=c.checkpoint_pk in ckset,
            reason="",
        ))
    df1 = pd.DataFrame(stage1)
    df1.to_csv(f"{args.out_dir}/stage1_turn_to_commit.csv", index=False)

    ok = df1[df1.commit_sha.notna()]
    hit = int(ok.file_in_commit.sum()) if len(ok) else 0
    print(f"  turns mapped to a commit : {len(ok)}/{len(df1)}")
    print(f"  edited file present in that commit : {hit}/{len(ok)}"
          f"  ({hit / max(len(ok), 1):.0%})")
    print(f"  -> {len(ok) - hit} turns map to a commit that does not contain "
          f"the file the agent just edited")
    if len(ok):
        print("\n  commits the heuristic lands on:")
        for sha, g in ok.groupby("commit_sha"):
            print(f"    {sha[:8]}  {len(g):>3} turns  "
                  f"file present in {int(g.file_in_commit.sum())}/{len(g)}"
                  f"  ckpt-linked={bool(g.checkpoint_linked.iloc[0])}"
                  f"  | {g.commit_message.iloc[0][:44]}")

    # ---------------- STAGE 2: line-mapping survival ---------------------
    print("\n" + "=" * 70)
    print("STAGE 2  line-mapping the agent's written lines into the commit")
    cache = {}
    stage2 = []
    for _, r in ok.iterrows():
        sha = r.commit_sha
        if sha not in cache:
            c = cand[cand.commit_sha == sha].iloc[0]
            cache[sha] = (patch_added_lines(c.patch),
                          committed_versions(c.file_attribution),
                          attribution_classes(c.file_attribution))
        added, comm_ver, att = cache[sha]
        turn = s[s.turn_number == r.turn_number].iloc[0]
        pairs = edit_payload(turn.tool_input_json)
        new_lines = [ln for _, new in pairs for ln in norm_lines(new)]
        if not new_lines:
            continue
        rel = r.rel_path or suffix_match(r.file_path, set(added) | set(comm_ver))
        in_patch = sum(1 for ln in new_lines if rel and ln in added.get(rel, set()))
        ctext = set(norm_lines(comm_ver.get(rel, ""))) if rel else set()
        in_committed = sum(1 for ln in new_lines if ln in ctext)
        stage2.append(dict(
            turn_number=int(r.turn_number), commit_sha=sha[:8], rel_path=rel,
            agent_lines_written=len(new_lines),
            lines_in_patch_added=in_patch,
            lines_in_committed_version=in_committed,
            survival_vs_patch=round(in_patch / len(new_lines), 3),
            survival_vs_committed=round(in_committed / len(new_lines), 3),
            has_committed_version=bool(ctext),
            attribution=att.get(rel, ""),
        ))
    df2 = pd.DataFrame(stage2)
    df2.to_csv(f"{args.out_dir}/stage2_line_survival.csv", index=False)
    if len(df2):
        tot = int(df2.agent_lines_written.sum())
        print(f"  agent lines written across mapped turns : {tot}")
        print(f"  found in the commit's added lines       : "
              f"{int(df2.lines_in_patch_added.sum())} "
              f"({df2.lines_in_patch_added.sum() / max(tot,1):.0%})")
        print(f"  found in committed_version              : "
              f"{int(df2.lines_in_committed_version.sum())} "
              f"({df2.lines_in_committed_version.sum() / max(tot,1):.0%})")
        print(f"  turns with 0% survival by line-mapping  : "
              f"{int((df2.survival_vs_patch == 0).sum())}/{len(df2)}")
        print(f"  turns with no committed_version to check: "
              f"{int((~df2.has_committed_version).sum())}/{len(df2)}")
        if df2.attribution.astype(bool).any():
            print(f"  attribution of those files: "
                  f"{df2[df2.attribution.astype(bool)].attribution.value_counts().to_dict()}")
    else:
        print("  no turns produced comparable lines")

    # ---------------- STAGE 3: replay reconstruction ---------------------
    print("\n" + "=" * 70)
    print("STAGE 3  replaying agent edits to reconstruct per-turn code state")
    reads = s[s.turn_type.eq("tool_result")].set_index("tool_call_id", drop=False) \
        if "tool_call_id" in s.columns else None
    read_calls = s[s.turn_type.eq("tool_use")
                   & s.tool_name.astype(str).str.lower().isin(READ_TOOLS)
                   & s.file_path.notna()]

    seeds = {}
    for _, rc in read_calls.iterrows():
        if rc.file_path in seeds or reads is None:
            continue
        tid = rc.get("tool_call_id")
        if tid is not None and tid in reads.index:
            res = reads.loc[tid]
            if isinstance(res, pd.DataFrame):
                res = res.iloc[0]
            seeds[rc.file_path] = strip_read_prefix(res.content)
    print(f"  files seeded from a Read tool_result: {len(seeds)}"
          f" / {writes.file_path.nunique()} edited")

    buf = {p: v for p, v in seeds.items()}
    states, failures = [], []
    for _, w in writes.iterrows():
        path = w.file_path
        pairs = edit_payload(w.tool_input_json)
        if not pairs:
            failures.append(dict(turn_number=int(w.turn_number), file_path=path,
                                 reason="no parsable edit payload"))
            continue
        for old, new in pairs:
            if old == "":                       # Write / create
                buf[path] = new
                continue
            if path not in buf:
                failures.append(dict(turn_number=int(w.turn_number),
                                     file_path=path,
                                     reason="no seed: file never Read first"))
                buf[path] = new                 # optimistic seed, flagged
                continue
            if old not in buf[path]:
                failures.append(dict(turn_number=int(w.turn_number),
                                     file_path=path,
                                     reason="old_string not found in "
                                            "reconstructed buffer"))
                continue
            buf[path] = buf[path].replace(old, new, 1)
        states.append(dict(
            turn_number=int(w.turn_number), file_path=path,
            n_lines=len(str(buf.get(path, "")).splitlines()),
            sha1=hashlib.sha1(str(buf.get(path, "")).encode()).hexdigest()[:12],
        ))

    df3 = pd.DataFrame(states)
    df3f = pd.DataFrame(failures)
    df3.to_csv(f"{args.out_dir}/stage3_replay_states.csv", index=False)
    df3f.to_csv(f"{args.out_dir}/stage3_replay_failures.csv", index=False)

    n_edits = sum(len(edit_payload(w.tool_input_json)) for _, w in writes.iterrows())
    print(f"  edit operations replayed : {n_edits}")
    print(f"  divergences              : {len(df3f)}"
          f"  ({len(df3f) / max(n_edits, 1):.0%})")
    if len(df3f):
        for reason, g in df3f.groupby("reason"):
            print(f"      {len(g):>4}  {reason}")
    print(f"  turn transitions recorded: {max(len(df3) - 1, 0)}")

    print(f"\nwrote 4 CSVs to {args.out_dir}/")


if __name__ == "__main__":
    main()