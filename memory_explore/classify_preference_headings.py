#!/usr/bin/env python3
"""
classify_preference_headings.py  --  which CLAUDE.md headings express a
*preference* (how the author wants work done) rather than describing the project.

Deliberately conservative and auditable:

  * matching is on WORD BOUNDARIES, not substrings. An earlier substring version
    silently matched "pr " inside "project overview" and "claude" inside the
    boilerplate "# CLAUDE.md" heading, inflating the preference count.
  * every heading assigned to a bucket is printed in the report, so the
    classification can be checked by eye rather than trusted.
  * a heading matching several buckets goes to the first one listed; the report
    notes multi-matches so borderline cases are visible.

Anything not matched is left in 未归类 — not silently folded into a bucket.

Run from the repo root:
  python memory_explore/classify_preference_headings.py
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")

# Boilerplate that /init emits — carries no topic signal, excluded up front.
BOILERPLATE = {"claude.md", "agents.md", "gemini.md"}

# (bucket, [terms]) — terms are matched as whole words/phrases.
# Ordered: the sharper buckets are tried before the vaguer ones.
BUCKETS: list[tuple[str, list[str]]] = [
    ("硬性禁止 / 必须做", [
        "do not", "don't", "never", "always", "forbidden", "prohibited",
        "hard rule", "hard rules", "core rule", "core rules", "golden rule",
        "no exceptions", "mandatory", "critical", "must", "required", "do", "dont",
    ]),
    ("代码风格 / 命名 / 格式", [
        "code style", "coding style", "style guide", "style", "styling",
        "naming", "name", "filename", "format", "formatting", "formatter",
        "lint", "linting", "linter", "eslint", "prettier", "gofmt", "swiftlint",
        "swiftformat", "black", "indentation", "typing", "type hints",
        "docstring", "docstrings", "comments", "comment",
    ]),
    ("测试要求", [
        "test", "tests", "testing", "coverage", "tdd", "assertions",
        "test coverage", "unit tests", "integration tests", "fixtures",
    ]),
    ("Git / 提交 / 分支 / PR", [
        "git", "commit", "commits", "branch", "branches", "pr", "prs",
        "pull request", "pull requests", "merge", "rebase", "worktree",
        "worktrees", "changelog", "release", "versioning", "trailer",
        "trailers", "pre-commit",
    ]),
    ("流程 / 检查清单 / 质量门", [
        "workflow", "workflows", "checklist", "process", "quality gate",
        "quality gates", "quality", "code quality", "review", "ci", "pipeline",
        "preflight", "definition of done", "before you start", "before every commit",
    ]),
    ("设计原则 / 模式偏好", [
        "principle", "principles", "philosophy", "pattern", "patterns",
        "anti-pattern", "anti-patterns", "best practice", "best practices",
        "idiom", "idioms", "paradigm", "design decision", "design decisions",
        "guideline", "guidelines", "approach",
    ]),
    ("对 agent 的行为指令", [
        "for ai agents", "for agents", "for ai", "agent instruction",
        "agent instructions", "instruction manual", "behavioral guidelines",
        "behavioural guidelines", "behavior", "behaviour", "work rules",
        "how to work", "ai voice", "voice", "collaboration", "communication",
        "multi-session behavior",
    ]),
    ("笼统的约定 / 规则 / 坑", [
        "convention", "conventions", "rule", "rules", "policy", "policies",
        "constraint", "constraints", "requirement", "requirements",
        "expectation", "expectations", "gotcha", "gotchas", "pitfall",
        "pitfalls", "caveat", "caveats", "standard", "standards",
    ]),
]


def normalize_heading(title: str) -> str:
    t = re.sub(r"[`*_\[\]]", "", title).strip()
    t = re.sub(r"^[\d.\)\s]+", "", t)
    t = re.sub(r"\s+", " ", t).strip(" :.")
    return t


def headings_of(body: str) -> list[str]:
    out, in_code = [], False
    for line in body.split("\n"):
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        m = HEADING_RE.match(line)
        if m:
            t = normalize_heading(m.group(2))
            if t:
                out.append(t)
    return out


def compile_bucket(terms: list[str]) -> re.Pattern:
    # longest first so "pull request" wins over "pr"; \b keeps "pr" out of "project"
    parts = sorted({re.escape(t) for t in terms}, key=len, reverse=True)
    return re.compile(r"(?<![\w-])(?:" + "|".join(parts) + r")(?![\w-])", re.I)


COMPILED = [(name, compile_bucket(terms)) for name, terms in BUCKETS]


def buckets_of(title: str) -> list[str]:
    return [name for name, rx in COMPILED if rx.search(title)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="memory_explore/claude_md_files")
    ap.add_argument("--out", default="memory_explore/claude_md_files/PREFERENCE_HEADINGS.md")
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

    assigned: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    files_of: dict[str, set[str]] = collections.defaultdict(set)
    multi: collections.Counter = collections.Counter()
    unmatched: collections.Counter = collections.Counter()
    n = n_boiler = n_pref = 0
    pref_per_file: collections.Counter = collections.Counter()

    for k, body in files:
        for title in headings_of(body):
            n += 1
            low = title.lower()
            if low in BOILERPLATE:
                n_boiler += 1
                continue
            bs = buckets_of(title)
            if not bs:
                unmatched[low] += 1
                continue
            n_pref += 1
            assigned[bs[0]][low] += 1
            files_of[bs[0]].add(k["file"])
            pref_per_file[k["file"]] += 1
            if len(bs) > 1:
                multi[" + ".join(bs)] += 1

    order = [name for name, _ in BUCKETS]
    L = [
        "# CLAUDE.md 里哪些小标题是关于 preference 的",
        "",
        f"统计对象：**{len(files)}** 个内容不重复的 CLAUDE.md，共 **{n}** 个标题。",
        f"其中 **{n_boiler}** 个是 `/init` 生成的样板标题（`# CLAUDE.md` 等），已剔除。",
        "",
        f"归入 preference 的：**{n_pref}**（占非样板标题的 "
        f"{100*n_pref/max(1, n-n_boiler):.0f}%）。"
        f"未归类：**{sum(unmatched.values())}**。",
        "",
        "匹配按**词边界**进行，不是子串 —— 否则 `pr` 会匹配到 `project`。"
        "每个被归类的标题都在下面列出，请自行核对。",
        "",
        "## 各类数量",
        "",
        "| 类别 | 标题数 | 出现在几个文件 |",
        "|---|---|---|",
    ]
    for name in order:
        if name in assigned:
            L.append(f"| {name} | {sum(assigned[name].values())} | {len(files_of[name])} |")

    for name in order:
        if name not in assigned:
            continue
        L += ["", f"## {name}", "",
              f"共 {sum(assigned[name].values())} 个标题，"
              f"{len(assigned[name])} 个不同，出现在 {len(files_of[name])} 个文件。", "",
              "| 次数 | 标题 |", "|---|---|"]
        for t, c in assigned[name].most_common():
            L.append(f"| {c} | {t.replace('|', '/')} |")

    if multi:
        L += ["", "## 同时命中多类的标题（已归入第一类，此处仅供核对）", "",
              "| 次数 | 命中的类别组合 |", "|---|---|"]
        for k2, v in multi.most_common():
            L.append(f"| {v} | {k2} |")

    L += ["", "## 每个文件的 preference 标题数", "",
          "| 文件 | preference 标题数 |", "|---|---|"]
    for fn, c in pref_per_file.most_common():
        L.append(f"| [{fn}]({fn}) | {c} |")
    zero = [k["file"] for k, _ in files if pref_per_file[k["file"]] == 0]
    if zero:
        L += ["", f"完全没有 preference 标题的文件（{len(zero)} 个）：", ""]
        L += [f"- [{f}]({f})" for f in zero]

    L += ["", f"## 未归类的标题（{sum(unmatched.values())} 个，前 120）", "",
          "这些多是项目专有内容，或需要读正文才能判断。", "",
          "| 次数 | 标题 |", "|---|---|"]
    for t, c in unmatched.most_common(120):
        L.append(f"| {c} | {t.replace('|', '/')} |")

    Path(args.out).write_text("\n".join(L) + "\n")

    print(f"{len(files)} 个文件，{n} 个标题（样板 {n_boiler} 已剔除）")
    print(f"preference: {n_pref}  未归类: {sum(unmatched.values())}")
    print()
    for name in order:
        if name in assigned:
            print(f"  {sum(assigned[name].values()):>4}  {name}"
                  f"   ({len(files_of[name])} 个文件)")
    print()
    print(f"有 preference 标题的文件: {len(pref_per_file)}/{len(files)}")
    print(f"写入 {args.out}")


if __name__ == "__main__":
    main()
