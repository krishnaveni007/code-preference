"""Prepare one turn-level replay: sealed repo at the turn's state, truncated transcript to resume, prompt file.

Input is a case JSON (see cases/*.json). Output dir gets:
  repo/                      clone holding only history up to `sha`, on the original branch name, no remote
  config/projects/<slug>/<session_id>.jsonl   transcript lines [0, cut_line) with the original cwd rewritten
  prompt.txt                 the prompt at cut_line (rewritten)
  prompt_ablated.txt         `prompt_override` (rewritten), for a U case's ablated arms
  state_check.json           consistency of the transcript's last file observations with repo/ (see check_state)

The state is a commit, not a replay of edits: a case is only valid when the transcript shows a clean tree at that
commit before cut_line (no Edit/Write after the commit), which check_state verifies from the transcript itself.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import shutil
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import common  # noqa: E402

LINE_RE = re.compile(r"^\s*(\d+)[→\t](.*)$")


def git(repo, *args, check=True):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=check).stdout


def sealed_clone(src: pathlib.Path, sha: str, branch: str, dst: pathlib.Path):
    """Clone, keep one branch at `sha`, drop every other ref, remote and reflog so no later commit is reachable."""
    # a local clone copies refs/heads only: pin the sha (it may sit on a remote-tracking or no branch in src)
    full = git(src, "rev-parse", f"{sha}^{{commit}}").strip()
    git(src, "update-ref", f"refs/heads/_replay_{full[:12]}", full)
    subprocess.run(["git", "clone", "-q", "--no-local", "--no-checkout", str(src), str(dst)], check=True)
    git(dst, "checkout", "-q", "-B", branch, sha)
    git(dst, "remote", "remove", "origin")
    for ref in git(dst, "for-each-ref", "--format=%(refname)").split():
        if ref != f"refs/heads/{branch}":
            git(dst, "update-ref", "-d", ref)
    git(dst, "reflog", "expire", "--expire=now", "--all")
    git(dst, "gc", "-q", "--prune=now")
    later = git(dst, "rev-list", "--all", "--not", sha).strip()
    assert not later, f"commits after {sha} still reachable"
    assert not git(dst, "status", "--porcelain").strip()


def rewriter(orig_re: str, new_cwd: str):
    pat = re.compile(orig_re + r"(?P<rest>(?:[\\/]+[\w.\-]+)*)")

    def sub(m):
        return new_cwd + m.group("rest").replace("\\", "/").replace("//", "/")

    def walk(x):
        if isinstance(x, str):
            return pat.sub(sub, x)
        if isinstance(x, list):
            return [walk(v) for v in x]
        if isinstance(x, dict):
            return {k: walk(v) for k, v in x.items()}
        return x

    return walk


def prompt_text(entry) -> str:
    c = entry["message"]["content"]
    return c if isinstance(c, str) else "\n".join(b.get("text", "") for b in c if b.get("type") == "text")


def check_state(context: list[dict], repo: pathlib.Path, cwd: str, anchor_line: int = -1) -> dict:
    """For every repo file the context touched, compare its LAST observation with repo/.

    Read result (no later edit) -> each numbered line must equal the file's line; Write -> whole content;
    Edit -> new_string must be present. Mismatch means the commit is not the state the agent last saw.
    """
    results = {}
    for e in context:
        if e.get("type") == "user" and isinstance(e["message"]["content"], list):
            for b in e["message"]["content"]:
                if b.get("type") == "tool_result":
                    results[b["tool_use_id"]] = b
    last = {}
    for i, e in enumerate(context):
        if e.get("type") != "assistant" or e.get("isSidechain"):
            continue
        for b in e["message"].get("content", []):
            if not (isinstance(b, dict) and b.get("type") == "tool_use"):
                continue
            fp = b["input"].get("file_path", "")
            if b["name"] in ("Read", "Edit", "Write", "MultiEdit") and fp.startswith(cwd + "/"):
                last[fp[len(cwd) + 1:]] = (b["name"], b["input"], results.get(b["id"]), i)
    report = {}
    for rel, (name, inp, res, i) in sorted(last.items()):
        p = repo / rel
        tag = "after anchor" if i > anchor_line else "before anchor"  # only "after anchor" contradicts the commit
        if p.is_dir():
            continue
        if not p.exists():
            report[rel] = f"{name}: file missing ({tag})"
            continue
        text = p.read_text(errors="replace")
        if name == "Write":
            report[rel] = "ok" if text == inp["content"] else f"Write: content differs ({tag})"
        elif name in ("Edit", "MultiEdit"):
            news = [inp["new_string"]] if name == "Edit" else [x["new_string"] for x in inp["edits"]]
            report[rel] = "ok" if all(n in text for n in news) else f"Edit: new_string absent ({tag})"
        else:
            body = res.get("content") if res else None
            if isinstance(body, list):
                body = "\n".join(x.get("text", "") for x in body if x.get("type") == "text")
            if not isinstance(body, str) or res.get("is_error"):
                report[rel] = f"Read: no text result ({tag})"
                continue
            lines, bad, n = text.split("\n"), 0, 0
            for ln in body.split("\n"):
                m = LINE_RE.match(ln)
                if not m or len(m.group(2)) >= 2000:
                    continue
                n += 1
                i = int(m.group(1)) - 1
                if i >= len(lines) or lines[i].rstrip("\r") != m.group(2).rstrip("\r"):
                    bad += 1
            report[rel] = "ok" if n and not bad else f"Read: {bad}/{n} lines differ ({tag})"
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("--clone", required=True, help="local full clone of the repo")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    case = json.loads(pathlib.Path(a.case).read_text())
    out = pathlib.Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    sealed_clone(pathlib.Path(a.clone), case["sha"], case["branch"], out / "repo")

    lines = [json.loads(l) for l in open(common.transcript(case["transcript"])) if l.strip()]
    cut = case["cut_line"]
    target = lines[cut]
    assert target["type"] == "user", "cut_line must be the target prompt"
    walk = rewriter(case["orig_cwd_regex"], case["container_cwd"])
    context = [walk(e) for e in lines[:cut]]
    for e in context:
        if "cwd" in e:
            e["cwd"] = case["container_cwd"]
    slug = re.sub(r"[^A-Za-z0-9]", "-", case["container_cwd"])
    tdir = out / "config" / "projects" / slug
    tdir.mkdir(parents=True)
    with open(tdir / f"{case['session_id']}.jsonl", "w") as f:
        for e in context:
            f.write(json.dumps(e) + "\n")

    prompt = walk(prompt_text(target))
    (out / "prompt.txt").write_text(prompt)
    if case.get("prompt_override"):  # ablated U prompt, run with run_turn.py --prompt-file prep/prompt_ablated.txt
        (out / "prompt_ablated.txt").write_text(walk(case["prompt_override"]))
    state = check_state(context, out / "repo", case["container_cwd"], case.get("anchor_line", -1))
    (out / "state_check.json").write_text(json.dumps(state, indent=1))
    bad = {k: v for k, v in state.items() if v != "ok"}
    after = sum("after anchor" in v for v in bad.values())
    print(f"context lines {len(context)}, prompt {len(prompt)} chars, files checked {len(state)}, mismatches {len(bad)} "
          f"({after} after the anchor commit)")
    for k, v in bad.items():
        print("  ", k, v)


if __name__ == "__main__":
    main()
