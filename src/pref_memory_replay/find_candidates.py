"""Find replayable C (correction) and U (unprompted) turns among the repeated-preference trajectories.

Input: the trajectories JSON (extracted from trajectories.html). For each C/U statement that is a REPEAT (the same
(user, mode, dim, side) was stated in an earlier session, so memory could carry it), locate the prompt in the Claude
Code transcript by its quote and decide replayability:

  replayed prompt   C: the real prompt before the correction (interrupt markers skipped); U: the statement's prompt
  state anchor      a commit sha the transcript shows as HEAD after the LAST possible file change before the replayed
                    prompt: `git commit` output "[branch sha] ...", or the first line of `git log` output when nothing
                    changed files earlier in the session (a log does not show a clean tree). No anchor -> rejected.
  mutation          Edit/Write/MultiEdit/NotebookEdit, a state-changing Bash command, or a non-read-only subagent
  context tokens    of the last assistant call before the cut (resume cost and 1M-window check)

Writes candidates.json (all located turns with their reject reason, if any).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from inspect_transcript import is_prompt, text_of  # noqa: E402

import common  # noqa: E402

SNAP = common.SNAPSHOT
EDIT_TOOLS = {"Edit", "Write", "MultiEdit", "NotebookEdit"}
READONLY_AGENTS = {"Explore", "Plan", "claude-code-guide", "statusline-setup"}
MUT_RE = re.compile(r"(sed -i|perl -p?i|\bnode -e|python3? -c .*(open|write)|\bmv |\brm |\bcp |\btouch |\bmkdir |git (checkout|reset|apply|stash|merge|rebase|"
                    r"pull|cherry-pick|restore|switch|revert|am)\b|\b(npm|pnpm|yarn|bun) (i|install|add|run build|run "
                    r"format)|pip install|uv (add|sync|pip)|go generate|cargo (fix|fmt)|\bprettier\b.*--write|ruff "
                    r"(format|check --fix)|black |cat <<|\btee |\bpatch\b|(^|[^0-9&>])>\s*[\w./~$])")
COMMIT_RE = re.compile(r"^\[([\w./\-]+)(?: \(root-commit\))? ([0-9a-f]{7,40})\]", re.M)
LOG_RE = re.compile(r"^(?:commit )?([0-9a-f]{7,40})\b", re.M)
INTERRUPT = ("[Request interrupted", "<task-notification>", "<local-command", "<command-name>", "Caveat:")


def norm(s):
    return re.sub(r"\s+", " ", s or "").strip()


def transcript_path(sid):
    for d in ("transcripts", "transcripts1"):
        p = SNAP / d / f"{sid}.jsonl"
        if p.exists():
            return p
    return None


_CACHE = {}


def load_lines(p):
    """JSONL lines; a file with unparsable lines (some transcripts are not line-delimited) is returned as None."""
    if p not in _CACHE:
        try:
            _CACHE.clear()
            _CACHE[p] = [json.loads(l) for l in open(p) if l.strip()]
        except json.JSONDecodeError:
            _CACHE[p] = None
    return _CACHE[p]


def real_prompt(e):
    return is_prompt(e) and not text_of(e).lstrip().startswith(INTERRUPT)


def results_by_id(lines):
    res = {}
    for e in lines:
        if e.get("type") == "user" and isinstance(e.get("message", {}).get("content"), list):
            for b in e["message"]["content"]:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    c = b.get("content")
                    if isinstance(c, list):
                        c = "\n".join(x.get("text", "") for x in c if isinstance(x, dict))
                    res[b["tool_use_id"]] = c if isinstance(c, str) else ""
    return res


def anchor_before(lines, cut):
    """(sha, branch, line) of the HEAD shown after the last mutation before cut, or a reason string."""
    res = results_by_id(lines[:cut])
    last_mut, anchor = -1, None
    for i, e in enumerate(lines[:cut]):
        if e.get("type") != "assistant" or e.get("isSidechain"):
            continue
        for b in e["message"].get("content", []):
            if not (isinstance(b, dict) and b.get("type") == "tool_use"):
                continue
            name, inp = b["name"], b.get("input", {})
            cmd = inp.get("command", "") if name == "Bash" else ""
            out = res.get(b["id"], "")
            if name == "Bash" and re.search(r"\bgit commit\b", cmd) and COMMIT_RE.search(out):
                m = COMMIT_RE.search(out)
                anchor = (m.group(2), m.group(1), i)
                continue
            if name == "Bash" and re.search(r"\bgit log\b", cmd) and not re.search(r"\.\.|--all|\s[\w/]+/[\w-]+\s*$", cmd):
                m = LOG_RE.search(out or "")
                if m and last_mut < 0:  # `git log` shows HEAD, not a clean tree: valid only before any change
                    anchor = (m.group(1), None, i)
                continue
            mut = name in EDIT_TOOLS or (name == "Bash" and MUT_RE.search(cmd)) or \
                (name in ("Agent", "Task") and inp.get("subagent_type") not in READONLY_AGENTS)
            if mut:
                last_mut = i
    if anchor is None:
        return "no commit/log anchor before the turn"
    if anchor[2] < last_mut:
        return "files changed after the last anchor"
    return anchor


def context_tokens(lines, cut):
    for e in reversed(lines[:cut]):
        u = e.get("message", {}).get("usage") if e.get("type") == "assistant" else None
        if u:
            return u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0) + u.get("cache_creation_input_tokens", 0)
    return 0


def load_traj(path):
    """The trajectory groups: a JSON list, or trajectories.html (its <script id="data" type="application/json">)."""
    text = pathlib.Path(path).read_text()
    if path.endswith(".html"):
        text = re.search(r'<script id="data" type="application/json">(.*?)</script>', text, re.S).group(1)
    return json.loads(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True, help="trajectories.html or its extracted JSON list")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import pyarrow.parquet as pq
    sessions = {s["session_id"][:8]: s for s in pq.read_table(
        SNAP / "sessions.parquet", columns=["session_id", "repo_id", "agent"]).to_pylist()}
    groups = load_traj(a.traj)
    out = []
    for gi, g in enumerate(groups):
        first_sess = g["msgs"][0]["s"]
        for j, x in enumerate(g["msgs"]):
            if x["k"] not in "CU" or x["s"] == first_sess:
                continue  # only repeats: an earlier session already carries this preference
            rec = {"group": gi, "user": g["user"], "mode": g["mode"], "dim": g["dim"], "side": g["side"], "type": g["type"],
                   "k": x["k"], "t": x["t"], "s": x["s"], "quote": x["q"], "manner": x["m"], "corrects": x["c"],
                   "repo": x["repo"]}
            s = sessions.get(x["s"])
            p = transcript_path(s["session_id"]) if s else None
            if not p:
                out.append({**rec, "reject": "no transcript"})
                continue
            lines = load_lines(p)
            if lines is None:
                out.append({**rec, "reject": "transcript not line-delimited JSON"})
                continue
            if not any(e.get("type") == "assistant" and "message" in e for e in lines[:50]):
                out.append({**rec, "reject": "not a Claude Code transcript"})
                continue
            segs = [norm(q) for q in re.split(r"\.\.\.|…", x["q"]) if len(norm(q)) > 3]
            idx = [i for i, e in enumerate(lines) if real_prompt(e) and all(sg in norm(text_of(e)) for sg in segs)]
            if not idx:
                out.append({**rec, "reject": "quote not found"})
                continue
            stmt = idx[0]
            if x["k"] == "C":
                prev = [i for i in range(stmt) if real_prompt(lines[i])]
                if not prev:
                    out.append({**rec, "reject": "no prompt before the correction"})
                    continue
                cut = prev[-1]
            else:
                cut = stmt
            anc = anchor_before(lines, cut)
            rec.update({"session_id": s["session_id"], "transcript": str(p), "cut_line": cut, "stmt_line": stmt,
                        "cwd": next((e["cwd"] for e in lines if e.get("cwd")), None),
                        "branch_at_cut": lines[cut].get("gitBranch"), "ts_cut": lines[cut].get("timestamp"),
                        "ctx_tokens": context_tokens(lines, cut), "prompt_words": len(text_of(lines[cut]).split()),
                        "prompt": text_of(lines[cut])[:3000]})
            if isinstance(anc, str):
                rec["reject"] = anc
            else:
                rec.update({"sha": anc[0], "anchor_branch": anc[1], "anchor_line": anc[2]})
            out.append(rec)
    json.dump(out, open(a.out, "w"), indent=1)
    ok = [r for r in out if "reject" not in r]
    from collections import Counter
    print(len(out), "repeat C/U turns;", Counter(r.get("reject", "ok") for r in out).most_common())
    print("ok by kind", Counter(r["k"] for r in ok), "users", len({r["user"] for r in ok}))


if __name__ == "__main__":
    main()
