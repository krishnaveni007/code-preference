#!/usr/bin/env python3
"""
claude_md_timeline.py  --  where in a repo's observed commit sequence does
CLAUDE.md first show up, and what happens to it afterwards?

The question this answers: is CLAUDE.md typically introduced mid-history — N
commits go by, then it appears, then it gets touched now and then?

Method: for each repo, order every commit the dataset has for it by author_date,
then locate the first commit that touches a CLAUDE.md. Two very different cases
fall out and must not be mixed:

  observed creation   the first CLAUDE.md event has status 'A'. The file was born
                      inside the observation window, so the "before" count is a
                      real count of commits that had no CLAUDE.md.
  pre-existing        the first event is 'M' (or 'D'). The file already existed
                      when observation started, so the "before" count is
                      meaningless — we simply never saw its birth.

Only the first group can answer the question. The second group is reported
separately so the split is visible rather than hidden.

Caveat that limits every number here: the dataset holds ~7.3% of these repos'
commits over a ~2.5 month window, so "position in the observed sequence" is not
position in the repo's real history.

Run from the repo root:
  python memory_explore/claude_md_timeline.py
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
from pathlib import Path

import pyarrow.parquet as pq


def load_repo_commits(path: str) -> dict[str, list[tuple[dt.datetime, str]]]:
    """repo_id -> [(author_date, commit_sha), ...] sorted, deduped."""
    pf = pq.ParquetFile(path)
    seen: dict[str, dict[str, dt.datetime]] = collections.defaultdict(dict)
    for b in pf.iter_batches(batch_size=2000,
                             columns=["repo_id", "commit_sha", "author_date"]):
        d = b.to_pydict()
        for repo, sha, ad in zip(d["repo_id"], d["commit_sha"], d["author_date"]):
            if repo is None or sha is None or ad is None:
                continue
            seen[repo][sha] = ad
    return {r: sorted(((ad, sha) for sha, ad in m.items()), key=lambda x: x[0])
            for r, m in seen.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--commits", default="data/swechat_data/commits.parquet")
    ap.add_argument("--snapshots", default="memory_explore/outputs/claude_md_snapshots.jsonl")
    ap.add_argument("--out", default="memory_explore/outputs/claude_md_timeline.md")
    ap.add_argument("--basename", default="CLAUDE.md")
    args = ap.parse_args()

    repo_commits = load_repo_commits(args.commits)

    snaps = [json.loads(l) for l in open(args.snapshots)]
    snaps = [s for s in snaps if s["basename"] == args.basename and s["author_date"]]

    # per repo: the set of commits that touched any CLAUDE.md, and the events
    events_by_repo: dict[str, list[dict]] = collections.defaultdict(list)
    for s in snaps:
        events_by_repo[s["repo_id"]].append(s)
    for v in events_by_repo.values():
        v.sort(key=lambda x: x["author_date"])

    born, pre = [], []
    for repo, evs in events_by_repo.items():
        seq = repo_commits.get(repo, [])
        if not seq:
            continue
        pos = {sha: i for i, (_, sha) in enumerate(seq)}
        touching = {e["commit_sha"] for e in evs}
        first = evs[0]
        idx = pos.get(first["commit_sha"])
        if idx is None:
            continue
        after_idx = sorted({pos[c] for c in touching if c in pos})
        rec = {
            "repo": repo,
            "n_commits_observed": len(seq),
            "first_event_status": first["status"],
            "first_event_date": first["author_date"][:10],
            "position": idx + 1,                       # 1-based
            "commits_before": idx,
            "commits_after": len(seq) - idx - 1,
            "n_touching_commits": len(after_idx),
            "n_events": len(evs),
            "touch_positions": after_idx,
            "first_date": seq[0][0].isoformat()[:10],
            "last_date": seq[-1][0].isoformat()[:10],
            "attribution_first": first["attribution"],
        }
        (born if first["status"] == "A" else pre).append(rec)

    born.sort(key=lambda r: -r["n_commits_observed"])
    pre.sort(key=lambda r: -r["n_commits_observed"])

    L = [
        f"# {args.basename} 在仓库提交序列里出现的位置",
        "",
        "对每个仓库，把数据集里该仓库的所有提交按时间排序，再定位第一次动 "
        f"{args.basename} 的提交在第几位。",
        "",
        "两种情况必须分开看：",
        "",
        f"- **观测到诞生**（首个事件是 `A`）：文件在观测期内被创建，"
        f"所以“之前有几次提交”是真实的、确实没有 {args.basename} 的提交数。共 **{len(born)}** 个仓库。",
        f"- **早于观测期**（首个事件是 `M`/`D`）：观测开始时文件已存在，"
        f"“之前有几次提交”没有意义 —— 只是没看到它诞生。共 **{len(pre)}** 个仓库。",
        "",
        "**贯穿性限制**：数据集只有这些仓库约 7.3% 的提交、约 2.5 个月的窗口，"
        "所以“观测序列里的位置”不等于仓库真实历史里的位置。",
        "",
        "## 观测到诞生的仓库",
        "",
        "`位置` = 第一次出现在观测序列的第几个提交；`之前/之后` = 该位置前后的提交数；",
        "`动过的提交` = 观测期内一共有几个提交动过它；`分布` = 这些提交在序列中的位次。",
        "",
        "| 仓库 | 观测提交数 | 位置 | 之前 | 之后 | 动过的提交 | 诞生日 | 首次归属 | 分布 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in born:
        tp = ",".join(str(p + 1) for p in r["touch_positions"][:14])
        if len(r["touch_positions"]) > 14:
            tp += ",…"
        L.append(
            f"| {r['repo']} | {r['n_commits_observed']} | {r['position']} | "
            f"{r['commits_before']} | {r['commits_after']} | {r['n_touching_commits']} | "
            f"{r['first_event_date']} | {r['attribution_first']} | {tp} |"
        )

    L += [
        "",
        "## 文件早于观测期的仓库",
        "",
        "这些只能看到修改，看不到诞生。",
        "",
        "| 仓库 | 观测提交数 | 首次动它的位置 | 动过的提交 | 首个事件 | 日期 |",
        "|---|---|---|---|---|---|",
    ]
    for r in pre:
        L.append(
            f"| {r['repo']} | {r['n_commits_observed']} | {r['position']} | "
            f"{r['n_touching_commits']} | {r['first_event_status']} | "
            f"{r['first_event_date']} |"
        )

    # aggregate stats over the "born" group only
    def q(xs, p):
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None

    before = [r["commits_before"] for r in born]
    after = [r["commits_after"] for r in born]
    touch = [r["n_touching_commits"] for r in born]
    midway = [r for r in born if r["commits_before"] > 0 and r["commits_after"] > 0]

    L += [
        "",
        "## 汇总（只统计观测到诞生的那组）",
        "",
        "| 指标 | 值 |",
        "|---|---|",
        f"| 仓库数 | {len(born)} |",
        f"| 诞生前有提交、之后也有提交（真正“出现在中间”） | {len(midway)} |",
        f"| 诞生就在观测序列第 1 位（之前没有提交） | {sum(1 for r in born if r['commits_before'] == 0)} |",
        f"| 诞生后再没有提交 | {sum(1 for r in born if r['commits_after'] == 0)} |",
        f"| 诞生前提交数 中位/p90/最大 | {q(before, .5)} / {q(before, .9)} / {max(before) if before else '-'} |",
        f"| 诞生后提交数 中位/p90/最大 | {q(after, .5)} / {q(after, .9)} / {max(after) if after else '-'} |",
        f"| 动过它的提交数 中位/最大 | {q(touch, .5)} / {max(touch) if touch else '-'} |",
        f"| 诞生后再也没被改过的仓库 | {sum(1 for r in born if r['n_touching_commits'] == 1)} |",
    ]

    Path(args.out).write_text("\n".join(L) + "\n")

    print(f"观测到诞生: {len(born)} 个仓库 | 文件早于观测期: {len(pre)} 个仓库")
    print(f"真正“出现在中间”（前后都有提交）: {len(midway)}")
    print(f"诞生就在第 1 位: {sum(1 for r in born if r['commits_before'] == 0)}")
    print(f"诞生后再没被改过: {sum(1 for r in born if r['n_touching_commits'] == 1)}")
    print(f"诞生前提交数 中位 {q(before, .5)} / 最大 {max(before) if before else '-'}")
    print(f"写入 {args.out}")


if __name__ == "__main__":
    main()
