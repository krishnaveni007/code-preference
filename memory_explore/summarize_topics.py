#!/usr/bin/env python3
"""
summarize_topics.py  --  keyword statistics over the markdown headings of every
recovered CLAUDE.md.

No interpretation, no categories. Just counts:
  1. headings by level
  2. the most frequent headings verbatim (normalized: lowercased, punctuation
     and numbering stripped)
  3. the most frequent keywords appearing inside headings, with plurals folded
     onto the singular and stopwords dropped

Headings inside fenced code blocks are skipped. Files whose body is
byte-identical to an already-seen file are dropped, so mirrored repos
(marcus-sa/brain vs osabiohq/osabio, camkeith vs ckeith26) don't double-count.

Run from the repo root:
  python memory_explore/summarize_topics.py
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")

STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "with", "by",
    "at", "from", "as", "is", "are", "be", "this", "that", "it", "its", "you",
    "your", "we", "our", "if", "when", "how", "what", "why", "all", "any",
    "not", "no", "do", "does", "can", "will", "should", "must", "may", "via",
    "using", "use", "used", "into", "out", "up", "down", "over", "per", "vs",
    "e", "g", "i", "ie", "eg", "etc", "&", "-", "–", "—", "'s",
}

# fold common variants onto one form so counts aren't split
FOLD = {
    "commands": "command", "tests": "test", "testing": "test", "files": "file",
    "rules": "rule", "conventions": "convention", "dependencies": "dependency",
    "deps": "dependency", "scripts": "script", "modules": "module",
    "components": "component", "endpoints": "endpoint", "variables": "variable",
    "patterns": "pattern", "principles": "principle", "guidelines": "guideline",
    "standards": "standard", "notes": "note", "docs": "documentation",
    "commits": "commit", "branches": "branch", "packages": "package",
    "libraries": "library", "libs": "library", "tables": "table",
    "migrations": "migration", "models": "model", "routes": "route",
    "gotchas": "gotcha", "practices": "practice", "workflows": "workflow",
    "structures": "structure", "layers": "layer", "paths": "path",
    "keys": "key", "types": "type", "errors": "error", "checks": "check",
    "steps": "step", "tasks": "task", "tools": "tool", "agents": "agent",
    "skills": "skill", "hooks": "hook", "commandes": "command",
}


def normalize_heading(title: str) -> str:
    t = re.sub(r"[`*_\[\]]", "", title).strip()
    t = re.sub(r"^[\d.\)\s]+", "", t)          # leading "1. " / "2) "
    t = re.sub(r"\s+", " ", t).strip(" :.")
    return t


def headings_of(body: str) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    in_code = False
    for line in body.split("\n"):
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        m = HEADING_RE.match(line)
        if not m:
            continue
        t = normalize_heading(m.group(2))
        if t:
            out.append((len(m.group(1)), t))
    return out


def tokens(title: str) -> list[str]:
    words = re.findall(r"[a-z0-9][a-z0-9.+#/-]*", title.lower())
    out = []
    for w in words:
        w = w.strip(".-/")
        if not w or w in STOP or len(w) == 1:
            continue
        out.append(FOLD.get(w, w))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="memory_explore/claude_md_files")
    ap.add_argument("--out", default="memory_explore/claude_md_files/TOPICS.md")
    ap.add_argument("--top-headings", type=int, default=60)
    ap.add_argument("--top-keywords", type=int, default=80)
    args = ap.parse_args()

    d = Path(args.dir)
    idx = json.loads((d / "index.json").read_text())

    seen: set[int] = set()
    files: list[tuple[dict, str]] = []
    for k in idx["kept"]:
        body = (d / k["file"]).read_text().split("-->", 1)[-1]
        h = hash(body.strip())
        if h in seen:
            continue
        seen.add(h)
        files.append((k, body))

    by_level: collections.Counter = collections.Counter()
    exact: collections.Counter = collections.Counter()
    exact_files: dict[str, set[str]] = collections.defaultdict(set)
    kw: collections.Counter = collections.Counter()
    kw_files: dict[str, set[str]] = collections.defaultdict(set)
    kw_examples: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter)
    n = 0

    for k, body in files:
        for lvl, title in headings_of(body):
            n += 1
            by_level[lvl] += 1
            low = title.lower()
            exact[low] += 1
            exact_files[low].add(k["file"])
            for w in set(tokens(title)):
                kw[w] += 1
                kw_files[w].add(k["file"])
                kw_examples[w][low] += 1

    L = [
        "# CLAUDE.md 小标题关键词统计",
        "",
        f"统计对象：**{len(files)}** 个内容不重复的 CLAUDE.md"
        f"（导出 {len(idx['kept'])} 个，去掉 {len(idx['kept']) - len(files)} 个镜像仓库的重复）。",
        f"标题总数 **{n}**，不同标题 **{len(exact)}** 个。代码块内的 `#` 不计入。",
        "",
        "## 按标题层级",
        "",
        "| 层级 | 数量 |",
        "|---|---|",
    ]
    for lvl in sorted(by_level):
        L.append(f"| {'#' * lvl} | {by_level[lvl]} |")

    L += [
        "",
        f"## 最常见的标题（原文，前 {args.top_headings}）",
        "",
        "| 次数 | 出现在几个文件 | 标题 |",
        "|---|---|---|",
    ]
    for t, c in exact.most_common(args.top_headings):
        L.append(f"| {c} | {len(exact_files[t])} | {t.replace('|', '/')} |")

    L += [
        "",
        f"## 关键词统计（前 {args.top_keywords}）",
        "",
        "同一标题里同一词只计一次；复数并入单数（commands→command，testing→test 等）；"
        "冠词、介词等停用词已剔除。",
        "",
        "| 次数 | 文件数 | 关键词 | 该词出现的代表标题 |",
        "|---|---|---|---|",
    ]
    for w, c in kw.most_common(args.top_keywords):
        ex = ", ".join(t.replace("|", "/") for t, _ in kw_examples[w].most_common(4))
        L.append(f"| {c} | {len(kw_files[w])} | **{w}** | {ex} |")

    L += [
        "",
        "## 每个文件的标题数",
        "",
        "| 文件 | 标题数 |",
        "|---|---|",
    ]
    counts = sorted(((k["file"], len(headings_of(b))) for k, b in files),
                    key=lambda x: -x[1])
    for fn, c in counts:
        L.append(f"| [{fn}]({fn}) | {c} |")

    Path(args.out).write_text("\n".join(L) + "\n")

    print(f"{len(files)} 个不重复文件，{n} 个标题，{len(exact)} 个不同标题")
    print(f"按层级: {dict(sorted(by_level.items()))}")
    print()
    print("关键词前 25:")
    for w, c in kw.most_common(25):
        print(f"  {c:>4}  {w}")
    print()
    print(f"写入 {args.out}")


if __name__ == "__main__":
    main()
