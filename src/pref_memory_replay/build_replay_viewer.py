#!/usr/bin/env python3
"""Build a standalone HTML viewer of the turn-level replays: per case (tab C / tab U) every arm's memory index and the
memory notes the agent actually opened, the agent's turn (prose, tool calls, truncated results, diff) and the judge's
verdict with evidence. Output ($TURN_REPLAY_DATA/viewer/turn_replay.html) embeds all data; serve that folder locally.

  python src/pref_memory_replay/build_replay_viewer.py
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from judge_turn import judge, recorded_events, render  # noqa: E402

DATA, CASES = common.DATA, common.CASES
OUT = DATA / "viewer" / "turn_replay.html"
TRUNC_RES, TRUNC_ARG, TRUNC_DIFF = 600, 240, 60_000
LINE_RE = re.compile(r"^\s*\d+[→\t]", re.M)

# run dir -> (label, prompt variant, memory dir name); "recorded" is the SWE-chat turn itself. Order = column order.
ARM_DEFS = [
    ("recorded", "Original (SWE-chat recording)", "original", None),
    ("full_c0_r1", "Full prompt · no memory", "original", None),
    ("c0_r1", "No memory", "original", None),
    ("ablated_c0_r1", "Ablated · no memory", "ablated", None),
    ("mem_r1", "Real memory (user's store at the time)", "original", "mem_real"),
    ("ablated_mem_r1", "Ablated · real memory", "ablated", "mem_real"),
    ("c2p_r1", "C2 preference memory", "original", "mem_c2p"),
    ("ablated_c2p_r1", "Ablated · C2 preference memory", "ablated", "mem_c2p"),
]


def arms_on_disk(cdir):
    """The arms of a case that have a judge verdict (recorded: judge_recorded.json; runs: <run>/judge.json)."""
    out = []
    for run, label, variant, mem in ARM_DEFS:
        done = (cdir / "judge_recorded.json").exists() if run == "recorded" else (cdir / run / "judge.json").exists()
        if done:
            out.append((None if run == "recorded" else run, label, variant, mem))
    return out


def clip(s, n):
    s = s if isinstance(s, str) else json.dumps(s, ensure_ascii=False)
    return s if len(s) <= n else s[:n] + f"… [+{len(s) - n} chars]"


def result_text(b):
    c = b.get("content")
    if isinstance(c, list):
        c = "\n".join(x.get("text", "") if x.get("type") == "text" else f"[{x.get('type')}]" for x in c)
    return c if isinstance(c, str) else json.dumps(c)


def tool_arg(name, inp):
    if name in ("Edit", "MultiEdit"):
        return f"{inp.get('file_path')}\n{clip(inp.get('new_string', inp.get('edits', '')), TRUNC_ARG)}"
    if name == "Write":
        return f"{inp.get('file_path')} ({len(inp.get('content', ''))} chars)"
    if name == "ExitPlanMode":
        return inp.get("plan", "")
    for k in ("command", "file_path", "pattern", "url", "query", "description", "prompt"):
        if k in inp:
            return clip(str(inp[k]), TRUNC_ARG)
    return clip(inp, TRUNC_ARG)


def is_memory_path(p):
    p = p.replace("\\", "/")
    return "/.claude" in p and "/memory/" in p


def steps_of(events):
    """Ordered turn steps plus the memory notes the agent opened (path, text from the tool result)."""
    steps, results, mem_reads = [], {}, []
    for e in events:
        if e.get("type") == "user" and isinstance(e.get("message", {}).get("content"), list):
            for b in e["message"]["content"]:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    results[b["tool_use_id"]] = b
    for e in events:
        if e.get("isSidechain") or e.get("type") not in ("assistant", "user"):
            continue
        content = e.get("message", {}).get("content")
        if not isinstance(content, list):
            continue
        for b in content:
            if not isinstance(b, dict):
                continue
            if e["type"] == "assistant" and b.get("type") == "text" and b["text"].strip():
                steps.append({"k": "text", "t": b["text"].strip()})
            elif e["type"] == "assistant" and b.get("type") == "tool_use":
                inp = b.get("input", {})
                r = results.get(b["id"])
                res = result_text(r) if r else ""
                mem = b["name"] == "Read" and is_memory_path(inp.get("file_path", ""))
                if mem:
                    mem_reads.append({"path": inp["file_path"], "text": LINE_RE.sub("", res)})
                steps.append({"k": "tool", "name": b["name"], "arg": tool_arg(b["name"], inp), "mem": mem,
                              "res": clip(res, TRUNC_RES), "err": bool(r and r.get("is_error"))})
    return steps, mem_reads


def load_json(p):
    return json.loads(p.read_text()) if p.exists() else None


def load_arm(case, cdir, run, label, variant, memname):
    if run is None:
        events = recorded_events(case)
        prompt = (cdir / "prep/prompt.txt").read_text()
        judge = load_json(cdir / "judge_recorded.json")
        diff, cost, seconds = "", None, None
    else:
        rdir = cdir / run
        events = [json.loads(l) for l in open(rdir / "stream.jsonl") if l.strip().startswith("{")]
        prompt = (rdir / "prompt.txt").read_text()
        judge = load_json(rdir / "judge.json")
        diff = clip((rdir / "diff.patch").read_text(), TRUNC_DIFF)
        res = next((e for e in events if e.get("type") == "result"), {})
        cost, seconds = res.get("total_cost_usd"), load_json(rdir / "run.json").get("seconds")
    steps, mem_reads = steps_of(events)
    index = (cdir / memname / "MEMORY.md").read_text() if memname else None
    verdict = (judge or {}).get("verdict") or {}
    return {"label": label, "variant": variant, "run": run or "recorded", "prompt": prompt, "steps": steps,
            "mem_index": index, "mem_name": memname, "mem_reads": mem_reads, "diff": diff, "cost": cost,
            "seconds": seconds, "judge": verdict, "counters": (judge or {}).get("counters", {}),
            "judge_model": (judge or {}).get("judge_model"), "_events": events}


SUMMARY_SCHEMA = {"type": "object", "required": ["agent", "judge"],
                  "properties": {"agent": {"type": "string"}, "judge": {"type": "string"}}}
SUMMARY_PROMPT = """Summarise one replayed coding-agent turn for a comparison table. Plain prose, no markdown.
agent: 2-3 sentences on what the agent did in this turn (what it investigated, whether it changed code or stopped at a
plan or question, what fix or position it took). Facts from the turn only.
judge: 1-2 sentences giving the judge's main reason for its verdict ({vlabel}: {verdict}).

<agent_turn>
{turn}
</agent_turn>

<judge_rationale>
{rationale}
</judge_rationale>"""


def summarise(arm, kind, cache, model, cc_bin):
    """One tool-less model call per arm, cached by a hash of its inputs in the case's summaries.json."""
    turn, _ = render(arm.pop("_events"))
    vlabel, verdict = ("correction warranted", arm["judge"].get("correction_warranted")) if kind == "C" else \
        ("follows preference", arm["judge"].get("follows_preference"))
    prompt = SUMMARY_PROMPT.format(vlabel=vlabel, verdict=verdict, turn=clip(turn, 60_000),
                                   rationale=arm["judge"].get("rationale", ""))
    key = hashlib.sha256(prompt.encode()).hexdigest()[:16]
    if cache.get(arm["run"], {}).get("key") != key:
        if not cc_bin:
            return {"agent": "", "judge": ""}
        cache[arm["run"]] = {"key": key, **judge(prompt, SUMMARY_SCHEMA, model, cc_bin)["verdict"]}
    return {k: cache[arm["run"]][k] for k in ("agent", "judge")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cc-bin", default=common.JUDGE_CC, help="claude >= 2.1.280 for the summary calls; without it only cached summaries are used")
    ap.add_argument("--model", default=common.JUDGE_MODEL)
    a = ap.parse_args()
    cases = []
    for path in sorted(CASES.glob("*.json")):
        case = json.loads(path.read_text())
        cid, cdir = case["case_id"], DATA / case["case_id"]
        arms = arms_on_disk(cdir)
        if len(arms) < 2:
            continue
        loaded = [load_arm(case, cdir, *x) for x in arms]
        cache_p = cdir / "summaries.json"
        cache = json.loads(cache_p.read_text()) if cache_p.exists() else {}
        for arm in loaded:
            arm["summary"] = summarise(arm, case["kind"], cache, a.model, a.cc_bin)
        cache_p.write_text(json.dumps(cache, indent=1))
        cases.append({
            "id": cid, "kind": case["kind"], "user": case["user"], "group": case["group"], "note": case["note"],
            "session": case["session_id"], "sha": case["sha"], "branch": case["branch"],
            "correction": case.get("judge", {}).get("correction"), "span": case.get("preference_span"),
            "dim": case.get("judge", {}).get("dim"), "side": case.get("judge", {}).get("side"),
            "repo": case.get("repo"), "arms": loaded})
    html = TEMPLATE.replace("/*DATA*/", json.dumps(cases, ensure_ascii=False).replace("</", "<\\/"))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html)
    print(OUT, len(html) // 1024, "KB")


TEMPLATE = Path(__file__).with_name("replay_viewer_template.html").read_text()
if __name__ == "__main__":
    main()
