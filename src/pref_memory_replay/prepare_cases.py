"""Prepare every input a case's arms need, so run_all.py can start: repo clone, sealed state, both memory arms.

Per case (all under $TURN_REPLAY_DATA/<case_id>/):
  clone     $TURN_REPLAY_CLONES/<owner>__<name>, cloned from GitHub once; the case sha is fetched if it is not on a
            branch (GitHub serves unreachable commits by full sha; the short sha is resolved via the REST API)
  prep/     prepare_turn.py: sealed repo at the sha, truncated transcript, prompt.txt / prompt_ablated.txt,
            state_check.json. A case whose state check has mismatches AFTER the anchor commit should be dropped.
  mem_real/ snapshot_store.py: the user's recovered memory store as of the cut, from the session's own project
            folder only (only when it holds a MEMORY.md at the cut; otherwise the real-memory arm is skipped)
  mem_c2p/  render_c2p.py: C2 preference notes from statements in sessions that started before
            the target session (cutoff = the transcript's first timestamp, the log clock)
A step whose output already exists is skipped; --force redoes it.

  python src/pref_memory_replay/prepare_cases.py [--cases a,b] [--force] [--no-c2p]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import shutil
import subprocess
import sys
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import common  # noqa: E402

PY = sys.executable
HERE = common.MODULE


def sh(cmd, log, **kw):
    with open(log, "a") as f:
        f.write("$ " + " ".join(map(str, cmd)) + "\n")
        f.flush()
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, **kw).returncode


def has_commit(clone, sha):
    return subprocess.run(["git", "-C", str(clone), "cat-file", "-e", f"{sha}^{{commit}}"],
                          capture_output=True).returncode == 0


def ensure_clone(repo: str, sha: str, log) -> pathlib.Path:
    clone = common.CLONES / repo.replace("/", "__")
    if not clone.exists():
        common.CLONES.mkdir(parents=True, exist_ok=True)
        sh(["git", "clone", "-q", f"https://github.com/{repo}.git", str(clone)], log, check=True)
    if not has_commit(clone, sha):
        full = sha
        if len(sha) < 40:
            with urllib.request.urlopen(f"https://api.github.com/repos/{repo}/commits/{sha}", timeout=30) as r:
                full = json.load(r)["sha"]
        sh(["git", "-C", str(clone), "fetch", "-q", "origin", full], log, check=True)
    return clone


def project_slug(transcript: pathlib.Path) -> str:
    """Claude Code's project folder name for the session's cwd (the memory store lives under it)."""
    for line in open(transcript):
        cwd = json.loads(line).get("cwd") if line.strip() else None
        if cwd:
            return re.sub(r"[^A-Za-z0-9]", "-", cwd)
    raise ValueError(f"no cwd in {transcript}")


def first_timestamp(transcript: str) -> str:
    for line in open(transcript):
        if line.strip():
            t = json.loads(line).get("timestamp")
            if t:
                return t
    raise ValueError(f"no timestamp in {transcript}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", help="comma-separated case ids (default: every case in $TURN_REPLAY_CASES)")
    ap.add_argument("--force", action="store_true", help="redo steps whose output exists")
    ap.add_argument("--no-c2p", action="store_true", help="skip the C2 memory render (needs SWE-chat + explorer data)")
    ap.add_argument("--cc-bin", default=common.JUDGE_CC, help="claude >= 2.1.280 for the C2 'How to apply' lines")
    a = ap.parse_args()
    paths = sorted(common.CASES.glob("*.json"))
    if a.cases:
        paths = [p for p in paths if p.stem in a.cases.split(",")]
    for p in paths:
        case = json.loads(p.read_text())
        cid, cdir = case["case_id"], common.DATA / case["case_id"]
        cdir.mkdir(parents=True, exist_ok=True)
        log = cdir / "prepare.log"
        try:
            if a.force or not (cdir / "prep" / "prompt.txt").exists():
                clone = ensure_clone(case["repo"], case["sha"], log)
                sh([PY, HERE / "prepare_turn.py", p, "--clone", clone, "--out", cdir / "prep"], log, check=True)
            state = json.loads((cdir / "prep" / "state_check.json").read_text())
            after = [k for k, v in state.items() if "after anchor" in v]

            store = common.STORES / "stores" / case["repo"].replace("/", "__")
            if store.exists() and (a.force or not (cdir / "mem_real").exists()):
                sh([PY, HERE / "snapshot_store.py", "--repo", case["repo"], "--at", case["ts_cut"],
                    "--slug=" + project_slug(common.transcript(case["transcript"])), "--out", cdir / "mem_real"], log, check=True)
                if not (cdir / "mem_real" / "MEMORY.md").exists():
                    shutil.rmtree(cdir / "mem_real")  # no index at the cut: drop the real-memory arm

            if not a.no_c2p and (a.force or not (cdir / "mem_c2p" / "MEMORY.md").exists()):
                if not a.cc_bin:
                    raise SystemExit("--cc-bin (or $TURN_REPLAY_JUDGE_CC) is needed for the C2 render")
                sh([PY, HERE / "render_c2p.py", "--user", case["user"],
                    "--cutoff", first_timestamp(common.transcript(case["transcript"])), "--out", cdir / "mem_c2p", "--cc-bin", a.cc_bin],
                   log, check=True)
            print(cid, "ok", f"real={'yes' if (cdir / 'mem_real').exists() else 'no'}",
                  f"c2p={'yes' if (cdir / 'mem_c2p' / 'MEMORY.md').exists() else 'no'}",
                  f"STATE MISMATCH after anchor: {after}" if after else "", flush=True)
        except Exception as e:  # keep going; the log has the details
            print(cid, "FAILED", repr(e), f"(see {log})", flush=True)


if __name__ == "__main__":
    main()
