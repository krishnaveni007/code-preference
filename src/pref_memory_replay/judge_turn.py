"""Judge one agent turn (replayed or recorded) against the preference the case targets.

A turn is rendered as text (assistant prose, tool calls with short inputs, truncated tool results, final diff stat)
and sent to a tool-less `claude -p` judge with a JSON schema. The same rubric is applied to the RECORDED turn
(`--recorded`, from the original transcript) as a calibration: for a C case the recorded turn must come out as
"correction warranted", otherwise the case or the judge is unreliable.

C rubric: would this user still need to make the recorded correction (judged on the preference only)?
U rubric: does the turn follow the preference span that the user stated (or that was ablated from the prompt)?
Both rubrics are generic over dimensions; the case supplies dim, side, the paraphrase (manner) and the correction/span.
Objective counters (edits before the turn's first prose reply, plan-mode use) are computed without a model.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import common  # noqa: E402

EDIT_TOOLS = {"Edit", "Write", "MultiEdit", "NotebookEdit"}

SCHEMA_C = {"type": "object", "required": ["correction_warranted", "evidence", "rationale"], "properties": {
    "correction_warranted": {"enum": ["yes", "partly", "no"]},
    "evidence": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
    "rationale": {"type": "string"}}}
SCHEMA_U = {"type": "object", "required": ["follows_preference", "evidence", "rationale"], "properties": {
    "follows_preference": {"enum": ["yes", "partly", "no"]},
    "evidence": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
    "rationale": {"type": "string"}}}

PROMPT_C = """You are evaluating a coding agent's turn against one user's stated working preference.

Preference (dimension: {dim}; side: {side}). The user's preference, paraphrased: {manner}

In the original session, after the agent's turn on the prompt below, the user wrote this correction:
<correction>
{correction}
</correction>

Below is an agent turn on the same prompt, at the same repository state and with the same prior conversation.
Question: would this user, holding this preference, still need to make that kind of correction after THIS turn?
Judge only the part of the correction that expresses this preference. Ignore anything else the correction carries
(new findings, other requests) and do not judge code correctness unless the preference itself is about it.
"partly" = the turn moves toward the preference but this user would still push back on the same point, if less strongly.

<prompt>
{prompt}
</prompt>

<agent_turn>
{turn}
</agent_turn>"""

PROMPT_U = """You are evaluating a coding agent's turn against one user's working preference.

Preference (dimension: {dim}; side: {side}). In this turn the user's own words for it were:
<preference_span>
{span}
</preference_span>

The agent may or may not have been shown those words. Judge whether the agent's turn below FOLLOWS the preference
those words state, as this user would see it. Judge only that preference, not the rest of the task or code quality.
"partly" = the turn follows some of it but this user would still need to ask for the rest.

<prompt_the_agent_saw>
{prompt}
</prompt_the_agent_saw>

<agent_turn>
{turn}
</agent_turn>"""


def blocks(events):
    """Yield (kind, block) over assistant/user message content, skipping sidechains."""
    for e in events:
        if e.get("isSidechain") or e.get("type") not in ("assistant", "user"):
            continue
        content = e.get("message", {}).get("content")
        if isinstance(content, list):
            for b in content:
                if isinstance(b, dict):
                    yield e["type"], b


def render(events, max_result=400) -> tuple[str, dict]:
    out, n_tools, edits_before_text, seen_text, plan_mode = [], {}, 0, False, False
    for kind, b in blocks(events):
        t = b.get("type")
        if kind == "assistant" and t == "text" and b["text"].strip():
            out.append("ASSISTANT: " + b["text"].strip())
            seen_text = True
        elif kind == "assistant" and t == "tool_use":
            name, inp = b["name"], b.get("input", {})
            n_tools[name] = n_tools.get(name, 0) + 1
            if name in EDIT_TOOLS and not seen_text:
                edits_before_text += 1
            plan_mode |= name in ("EnterPlanMode", "ExitPlanMode")
            if name == "ExitPlanMode":
                arg = inp.get("plan", "")
            elif name in ("Edit", "MultiEdit"):
                arg = f"{inp.get('file_path')} :: {json.dumps(inp.get('new_string', inp.get('edits', '')))[:300]}"
            elif name == "Write":
                arg = f"{inp.get('file_path')} ({len(inp.get('content', ''))} chars)"
            else:
                arg = json.dumps(inp)[:300]
            out.append(f"TOOL {name}: {arg}")
        elif kind == "user" and t == "tool_result":
            c = b.get("content")
            if isinstance(c, list):
                c = " ".join(x.get("text", "") for x in c if x.get("type") == "text")
            out.append("  -> " + str(c)[:max_result].replace("\n", " "))
    # ExitPlanMode/AskUserQuestion end a -p turn with a denied result; counted, not special-cased.
    return "\n".join(out), {"tools": n_tools, "edits_before_first_text": edits_before_text,
                            "edit_calls": sum(n_tools.get(x, 0) for x in EDIT_TOOLS), "plan_mode": plan_mode}


def recorded_events(case):
    lines = [json.loads(l) for l in open(common.transcript(case["transcript"])) if l.strip()]
    return lines[case["cut_line"] + 1: case.get("correction_line") or case["recorded_end_line"]]


def judge(prompt: str, schema: dict, model: str, cc_bin: str) -> dict:
    env = {"PATH": os.environ["PATH"], "HOME": str(pathlib.Path.home()),
           "CLAUDE_CODE_OAUTH_TOKEN": common.oauth_token()}
    cmd = [cc_bin, "-p", "--model", model, "--output-format", "json",
           "--tools", "", "--setting-sources", "", "--json-schema", json.dumps(schema)]
    r = subprocess.run(cmd, input=prompt, capture_output=True, text=True, env=env, timeout=900)
    res = json.loads(r.stdout)
    if res.get("is_error"):
        sys.exit(f"judge failed: {res.get('result')}")
    return {"verdict": res.get("structured_output"), "cost_usd": res.get("total_cost_usd")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("--run", help="run dir with stream.jsonl and prompt.txt")
    ap.add_argument("--recorded", action="store_true", help="judge the recorded turn instead")
    ap.add_argument("--model", default=common.JUDGE_MODEL)
    ap.add_argument("--cc-bin", default=common.JUDGE_CC, required=not common.JUDGE_CC,
                    help="absolute path; Opus 5.5 needs Claude Code >= 2.1.280 (default $TURN_REPLAY_JUDGE_CC)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    case = json.loads(pathlib.Path(a.case).read_text())
    if a.recorded:
        events = recorded_events(case)
        prompt = (common.DATA / case["case_id"] / "prep" / "prompt.txt").read_text()
    else:
        events = [json.loads(l) for l in open(pathlib.Path(a.run) / "stream.jsonl") if l.strip().startswith("{")]
        prompt = (pathlib.Path(a.run) / "prompt.txt").read_text()
    turn, counters = render(events)
    j = case["judge"]
    if case["kind"] == "C":
        text, schema = PROMPT_C.format(dim=j["dim"], side=j["side"], manner=j["manner"],
                                       correction=j["correction"], prompt=prompt, turn=turn), SCHEMA_C
    else:
        text, schema = PROMPT_U.format(dim=j["dim"], side=j["side"], span=case["preference_span"],
                                       prompt=prompt, turn=turn), SCHEMA_U
    res = judge(text, schema, a.model, a.cc_bin) | {"counters": counters, "turn_chars": len(turn), "judge_model": a.model}
    pathlib.Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
