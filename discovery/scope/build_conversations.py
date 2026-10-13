#!/usr/bin/env python3
"""SWE-chat v2 -> conversations table in method1's format (no model calls).

method1/gate.py reads `conversations.parquet` (session_id, user_id, turn_number, turn_type, word_count,
prompt_pushback, content). SWE-chat v2 ships native transcripts instead, so every human session is parsed
with the native Claude Code / Codex / OpenCode parsers of the tf_eval pipeline, and each human turn becomes
one row. Extra columns for the scope analysis: agent, repo_id, time, kind (genuine / slash / ...),
prev_agent_reply (the agent's reply to the previous human turn, for context when labelling direction).
`content` keeps only what the developer typed: Codex's injected context before "## My request for Codex:"
is cut (the full text stays in `raw_content`).

Sessions: all human sessions of the cleaned corpus (tf_eval split v2, any role), subagent threads excluded.
user_id is SWE-chat's own user_id (the commit author); sessions whose author is a bot or an agent identity
(`[bot]`, `claude`, `Copilot`, `claude@...`, `agent@...`) are dropped. Output: data/swechat_v2/conversations.parquet (+ build log).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
import sys
from pathlib import Path

import pandas as pd

TF = Path("/data/user_data/zheyuan4/research/simulator/tf_eval")
sys.path.insert(0, str(TF / "swechat"))
sys.path.insert(0, str(TF))
import build_swechat as BS  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "swechat_v2"
PREV_CHARS = 2000
# commit "authors" that are bots or agent identities: they merge different people into one user_id
NON_HUMAN = re.compile(r"\[bot\]$|^(claude|copilot|agent@.*|claude@.*)$", re.I)
CODEX_REQUEST = "## My request for Codex:"   # Codex prepends IDE / browser / attachment context before this line


def user_text(text: str) -> str:
    """The part of a human turn the developer typed (drops Codex's injected context block)."""
    i = text.find(CODEX_REQUEST)
    return text[i + len(CODEX_REQUEST):].strip() if i >= 0 else text


def parse_session(s: dict) -> tuple[str, list[dict] | str]:
    tp = os.path.join(BS.SC, "transcripts", os.path.basename(s["transcript"]))
    try:
        fmt = BS.detect_format(tp)
        if fmt is None:
            return s["session_id"], "unknown format"
        if fmt == "opencode":
            _, hturns, _ = BS.parse_opencode(tp, merge_children=False)
        else:
            _, hturns, _ = (BS.PARSERS[fmt] or BS.parse_transcript)(tp)
    except FileNotFoundError:
        return s["session_id"], "transcript missing"
    except Exception as e:  # noqa: BLE001
        return s["session_id"], "parse error: %r" % (e,)
    rows, prev = [], ""
    for k, h in enumerate(hturns):
        raw = h.get("user") or ""
        text = user_text(raw)
        rows.append(dict(session_id=s["session_id"], turn_number=k, turn_type="user_prompt",
                         word_count=len(text.split()), prompt_pushback=None, content=text, raw_content=raw,
                         kind=h.get("kind"), time=str(h.get("time") or ""), agent=s["agent"],
                         repo_id=s["repo_id"], prev_agent_reply=(prev or "")[-PREV_CHARS:],
                         n_tools=int(h.get("n_tools") or 0)))
        prev = h.get("agent_text") or ""
    return s["session_id"], rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--jobs", type=int, default=16)
    p.add_argument("--limit", type=int, default=0, help="first N sessions only (pilot)")
    args = p.parse_args()
    split = [json.loads(l) for l in open(TF / "data/swechat_v2/split_v2/sessions_split.jsonl")]
    sessions = [s for s in split if not s.get("subagent") and s.get("reason") != "subagent thread"]
    if args.limit:
        sessions = sessions[:args.limit]
    users = pd.read_parquet(os.path.join(BS.SC, "sessions.parquet"), columns=["session_id", "user_id"])
    uid = dict(zip(users.session_id, users.user_id.astype(str)))
    rows, log = [], {}
    with mp.Pool(args.jobs, maxtasksperchild=50) as pool:
        for k, (sid, res) in enumerate(pool.imap_unordered(parse_session, sessions, chunksize=4)):
            if isinstance(res, str):
                log[sid] = res
            else:
                rows += res
            if k % 500 == 0:
                print("sessions %d/%d, rows %d, failed %d" % (k, len(sessions), len(rows), len(log)), flush=True)
    df = pd.DataFrame(rows)
    df["user_id"] = df.session_id.map(uid)
    n0 = df.session_id.nunique()
    df = df[~df.user_id.fillna("").str.match(NON_HUMAN)]
    print("dropped %d sessions authored by bot / agent identities" % (n0 - df.session_id.nunique()), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    name = "conversations%s.parquet" % ("_pilot" if args.limit else "")
    df.to_parquet(OUT / name, index=False)
    json.dump(dict(sessions=len(sessions), parsed=len(sessions) - len(log), failed=log, rows=len(df),
                   by_agent=df.groupby("agent").size().to_dict(), by_kind=df.kind.value_counts().to_dict()),
              open(OUT / name.replace(".parquet", "_build.json"), "w"), indent=1)
    print("done: %d rows from %d sessions (%d failed) -> %s" % (len(df), len(sessions) - len(log), len(log), OUT / name))


if __name__ == "__main__":
    main()
