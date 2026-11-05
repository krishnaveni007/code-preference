"""Run one replayed turn in the turn-replay sandbox (Docker) by resuming the truncated transcript.

Copies prep/ to a fresh run dir (repo + config), optionally installs repo dependencies, then runs
`claude -p --resume <session_id> --fork-session` with prompt.txt (or --prompt). Writes:
  stream.jsonl   stream-json events of the replayed turn
  diff.patch     `git diff` of the repo after the turn (incl. untracked files)
  run.json       command metadata, exit code, wall time
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import common  # noqa: E402

IMAGE = common.IMAGE
HOME_CFG = "/home/agent/.claude-config"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("--prep", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=common.AGENT_MODEL)
    ap.add_argument("--prompt", help="override prompt.txt (e.g. a resume probe)")
    ap.add_argument("--max-turns", type=int)
    ap.add_argument("--setup", default="", help="shell run in the repo before the agent (e.g. dependency install)")
    ap.add_argument("--memory-dir", help="memory folder to mount as the project's auto-memory (memory arm)")
    ap.add_argument("--timeout", type=int, default=2400)
    a = ap.parse_args()
    case = json.loads(pathlib.Path(a.case).read_text())
    prep, out = pathlib.Path(a.prep), pathlib.Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    shutil.copytree(prep / "repo", out / "repo", symlinks=True)
    shutil.copytree(prep / "config", out / "config")
    slug = next((out / "config" / "projects").iterdir()).name
    if a.memory_dir:
        shutil.copytree(a.memory_dir, out / "config" / "projects" / slug / "memory")
    prompt = a.prompt or (prep / "prompt.txt").read_text()
    (out / "prompt.txt").write_text(prompt)

    cwd = case["container_cwd"]
    claude = ["claude", "-p", "--resume", case["session_id"], "--fork-session", "--output-format", "stream-json",
              "--verbose", "--model", a.model, "--permission-mode", "bypassPermissions"]
    if a.max_turns:
        claude += ["--max-turns", str(a.max_turns)]
    setup = f"( {a.setup} ) > /tmp/setup.log 2>&1 ; tail -3 /tmp/setup.log >&2 ; " if a.setup else ""
    script = setup + " ".join(claude) + " < /run/prompt.txt"
    env = dict(os.environ)
    env["CLAUDE_CODE_OAUTH_TOKEN"] = common.oauth_token()
    cmd = ["docker", "run", "--rm", "-e", "CLAUDE_CODE_OAUTH_TOKEN", "-e", f"CLAUDE_CONFIG_DIR={HOME_CFG}",
           "-v", f"{(out / 'repo').resolve()}:{cwd}", "-v", f"{(out / 'config').resolve()}:{HOME_CFG}",
           "-v", f"{(out / 'prompt.txt').resolve()}:/run/prompt.txt:ro", "-w", cwd, IMAGE, "bash", "-lc", script]
    t0 = time.time()
    with open(out / "stream.jsonl", "w") as f, open(out / "stderr.txt", "w") as ferr:
        try:
            rc = subprocess.run(cmd, stdout=f, stderr=ferr, env=env, timeout=a.timeout).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
    subprocess.run(["git", "-C", str(out / "repo"), "add", "-A", "-N"], check=False)
    diff = subprocess.run(["git", "-C", str(out / "repo"), "diff"], capture_output=True, text=True).stdout
    (out / "diff.patch").write_text(diff)
    meta = {"case": case["case_id"], "model": a.model, "image": IMAGE, "script": script, "rc": rc,
            "seconds": round(time.time() - t0), "memory_dir": a.memory_dir, "diff_lines": diff.count("\n")}
    (out / "run.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
