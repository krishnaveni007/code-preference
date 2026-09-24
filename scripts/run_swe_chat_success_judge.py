#!/usr/bin/env python3
"""Run the SWE-Chat overall-success judge on valid longitudinal trials.

The visible conversation is reconstructed from the task instruction, simulator
decisions, and OpenCode text events. Hidden reasoning and raw tool outputs are
not included. Results are cached in each trial as
``swe_chat_success_verdict.json``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SWE = ROOT / "third_party" / "SWE-Together"
LEDGER = ROOT / "outputs" / "longitudinal_pilot" / "eval_metrics_ledger.json"
DEFAULT_MODEL = "openai/gpt-5-mini"
OUT_NAME = "swe_chat_success_verdict.json"

if str(SWE / "external" / "harbor" / "src") not in sys.path:
    sys.path.insert(0, str(SWE / "external" / "harbor" / "src"))

from harbor.llms.lite_llm import LiteLLM  # noqa: E402


SYSTEM_PROMPT = """You are scoring the overall success of an interactive coding session between a human
user and an AI coding agent.
You will receive:
1. The full conversation transcript (user messages and agent responses).
2. A summary of tool calls made during the session (category counts, top tools).
3. Commit information, if any (commit messages and diff summaries).

## Evaluation procedure
First, analyze the session along five dimensions. For each, note concrete evidence from
the transcript.
1. **Goal completion**: Did the agent fulfill what the user asked for? Identify every
distinct user request or task. For each, judge whether it was fully resolved,
partially resolved, or unresolved.
2. **Final session state**: How did the session end? A natural conclusion (user
confirms satisfaction, moves on to a new topic, or signs off) is positive. An
abrupt stop (user abandons mid-task, expresses frustration, or silently disengages
after an unresolved error) is negative. Weigh the ending heavily - a session that
goes well for many turns but ends in an unresolved failure should score
substantially lower than one that resolves cleanly.
3. **Agent efficiency**: Did the agent make steady progress, or did it spin? Negative
signals include: the user repeating the same instruction or correction, the agent
retrying a failed approach without changing strategy, and unnecessary tool calls
that do not advance the task. Positive signals include: appropriate use of
research tools before acting, surfacing uncertainty and offering the user choices,
and recovering from errors autonomously.
4. **Code and commit quality**: If code was produced, does it appear correct and
complete based on the available evidence (test results, diff content, user
reactions)? Were changes committed? Commits are a strong positive signal for task-
oriented sessions but are not required for short advisory or exploratory sessions
where no code change was the expected outcome.
5. **User experience**: Did the user have to fight the agent, or did the interaction
flow naturally? Look for signs of satisfaction (thanks, approval, moving to the
next task) and dissatisfaction (re-explaining, correcting, expressing frustration).

## Scoring rubric
After analyzing the five dimensions, assign a single integer score from 0 to 100.
- **90-100**: Every user request was fully resolved. The final session state is clean
(code works, commits are sound, no dangling errors). The user's last messages
indicate satisfaction or natural conclusion.
- **70-89**: The core user goal was met, but minor issues remain (e.g., an edge case
unhandled, a small follow-up the user would still need to do). No major unresolved
errors. The session ended on a reasonable note.
- **50-69**: Meaningful progress was made on the primary goal, but at least one
significant sub-task is incomplete or the solution has a known defect. The user
may have had to redirect the agent more than once on the same point.
- **30-49**: The agent produced some relevant output, but the primary user goal was not
achieved. The codebase may be in a worse or broken state, or the user visibly
gave up on the main task.
- **10-29**: Very little useful work was accomplished. The agent struggled repeatedly,
required heavy user correction, or went down unproductive paths for most of the
session.
- **0-9**: Nothing of value was produced. The session ended with no progress toward any
user goal.
When a session is ambiguous (e.g., short, exploratory, no clear deliverable), score
based on whether the agent's responses were the most helpful thing it could have
provided given the input. Do not penalize the agent for the user's request being
vague, but do penalize it if it failed to make productive use of whatever
information was available.

## Output format
Respond with valid JSON only. Do not include any text outside the JSON object.
{
  "score": <integer 0-100>,
  "reason": "<2-4 sentences: what was accomplished, what issues were present, and why this score.>"
}
"""


def load_dotenv() -> None:
    path = SWE / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def json_events(path: Path) -> list[dict]:
    events = []
    if not path.exists():
        return events
    for line in path.read_text(errors="replace").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            events.append(value)
    return events


def visible_agent_text(path: Path) -> str:
    chunks = []
    for event in json_events(path):
        if event.get("type") != "text":
            continue
        text = str((event.get("part") or {}).get("text") or "").strip()
        if text:
            chunks.append(text)
    return "\n\n".join(chunks)


def visible_codex_turns(path: Path) -> list[str]:
    turns: list[list[str]] = []
    current: list[str] | None = None
    for event in json_events(path):
        if event.get("type") == "thread.started":
            if current is not None:
                turns.append(current)
            current = []
            continue
        if current is None or event.get("type") != "item.completed":
            continue
        item = event.get("item") or {}
        if item.get("type") == "agent_message" and str(item.get("text") or "").strip():
            current.append(str(item["text"]).strip())
    if current is not None:
        turns.append(current)
    return ["\n\n".join(turn) for turn in turns]


def visible_claude_turns(path: Path) -> list[str]:
    order: list[str] = []
    chunks: dict[str, list[str]] = {}
    seen_messages: set[str] = set()
    for event in json_events(path):
        if event.get("type") != "assistant":
            continue
        message = event.get("message") or {}
        message_id = str(message.get("id") or "")
        if not message_id or message_id in seen_messages:
            continue
        seen_messages.add(message_id)
        session_id = str(event.get("session_id") or "unknown")
        if session_id not in chunks:
            order.append(session_id)
            chunks[session_id] = []
        for block in message.get("content") or []:
            if block.get("type") == "text" and str(block.get("text") or "").strip():
                chunks[session_id].append(str(block["text"]).strip())
    return ["\n\n".join(chunks[session_id]) for session_id in order]


def build_transcript(trial: Path, task_id: str) -> str:
    instruction = SWE / "tasks" / task_id / "instruction.md"
    users = [instruction.read_text(errors="replace").strip() if instruction.exists() else ""]
    for ep in sorted((trial / "agent").glob("episode-*"), key=lambda p: int(p.name.split("-")[-1])):
        try:
            decision = json.loads((ep / "user_decision.json").read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        if decision.get("has_message") and str(decision.get("content") or "").strip():
            users.append(str(decision["content"]).strip())

    codex = trial / "agent" / "codex.txt"
    claude = trial / "agent" / "claude-code.txt"
    if codex.exists():
        agents = visible_codex_turns(codex)
    elif claude.exists():
        agents = visible_claude_turns(claude)
    else:
        turns = sorted(
            (trial / "agent").glob("opencode.txt.turn-*"),
            key=lambda p: int(p.name.rsplit("-", 1)[-1]),
        )
        agents = [visible_agent_text(path) for path in turns]
    parts = []
    for idx in range(max(len(users), len(agents))):
        if idx < len(users):
            parts.append(f"USER TURN {idx}:\n{users[idx]}")
        if idx < len(agents):
            parts.append(f"AGENT TURN {idx}:\n{agents[idx] or '[No visible text response]'}")
    return "\n\n".join(parts)


def tool_and_commit_summary(trial: Path) -> tuple[str, str]:
    counts: Counter[str] = Counter()
    commit_commands = []
    codex = trial / "agent" / "codex.txt"
    claude = trial / "agent" / "claude-code.txt"
    if codex.exists():
        for event in json_events(codex):
            if event.get("type") != "item.completed":
                continue
            item = event.get("item") or {}
            tool = str(item.get("type") or "unknown")
            if tool in {"agent_message", "reasoning"}:
                continue
            counts[tool] += 1
            command = str(item.get("command") or "")
            if re.search(r"\bgit\s+commit\b", command):
                commit_commands.append(re.sub(r"\s+", " ", command).strip()[:500])
    elif claude.exists():
        seen_messages: set[str] = set()
        for event in json_events(claude):
            if event.get("type") != "assistant":
                continue
            message = event.get("message") or {}
            message_id = str(message.get("id") or "")
            if not message_id or message_id in seen_messages:
                continue
            seen_messages.add(message_id)
            for block in message.get("content") or []:
                if block.get("type") != "tool_use":
                    continue
                tool = str(block.get("name") or "unknown")
                counts[tool] += 1
                command = str((block.get("input") or {}).get("command") or "")
                if re.search(r"\bgit\s+commit\b", command):
                    commit_commands.append(re.sub(r"\s+", " ", command).strip()[:500])
    else:
        for event in json_events(trial / "agent" / "opencode.txt"):
            if event.get("type") != "tool_use":
                continue
            part = event.get("part") or {}
            tool = str(part.get("tool") or "unknown")
            counts[tool] += 1
            command = str((((part.get("state") or {}).get("input") or {}).get("command") or ""))
            if re.search(r"\bgit\s+commit\b", command):
                commit_commands.append(re.sub(r"\s+", " ", command).strip()[:500])
    total = sum(counts.values())
    tools = ", ".join(f"{name}: {count}" for name, count in counts.most_common()) or "none"
    tool_summary = f"Total tool calls: {total}. Tool counts: {tools}."

    patch = trial / "agent" / "final.patch"
    patch_text = patch.read_text(errors="replace") if patch.exists() else ""
    files = re.findall(r"^diff --git a/(.+?) b/", patch_text, flags=re.MULTILINE)
    added = sum(1 for line in patch_text.splitlines() if line.startswith("+") and not line.startswith("+++"))
    removed = sum(1 for line in patch_text.splitlines() if line.startswith("-") and not line.startswith("---"))
    commit_part = "; ".join(commit_commands) if commit_commands else "No git commit command observed."
    diff_part = (
        f"Final diff: {len(files)} files changed, {added} lines added, {removed} lines removed. "
        f"Files: {', '.join(files)}."
        if patch_text else "No final patch was recorded."
    )
    return tool_summary, f"{commit_part} {diff_part}"


def parse_verdict(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("judge response contained no JSON object")
    value = json.loads(match.group(0))
    score = value.get("score")
    reason = value.get("reason")
    if not isinstance(score, int) or not 0 <= score <= 100:
        raise ValueError(f"invalid score: {score!r}")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("missing reason")
    return {"score": score, "reason": reason.strip()}


async def judge_one(row: dict, llm: LiteLLM, sem: asyncio.Semaphore, model: str, force: bool) -> dict:
    trial = Path(row["trial_path"])
    output = trial / OUT_NAME
    if output.exists() and not force:
        try:
            cached = json.loads(output.read_text())
            if isinstance(cached.get("score"), int):
                return {"trial_path": str(trial), "status": "cached", **cached}
        except (json.JSONDecodeError, OSError):
            pass
    transcript = build_transcript(trial, row["task_id"])
    tools, commits = tool_and_commit_summary(trial)
    user_prompt = (
        "## Session data\n\n"
        "### Full conversation transcript\n"
        f"{transcript}\n\n"
        "### Tool-call summary\n"
        f"{tools}\n\n"
        "### Commit and diff information\n"
        f"{commits}"
    )
    async with sem:
        started = time.monotonic()
        try:
            response = await llm.call(
                prompt=user_prompt,
                message_history=[{"role": "system", "content": SYSTEM_PROMPT}],
            )
            verdict = parse_verdict(response.content)
            verdict.update({
                "judge": "SWE-Chat overall-success rubric",
                "judge_model": model,
                "elapsed_seconds": round(time.monotonic() - started, 3),
            })
            output.write_text(json.dumps(verdict, indent=2, ensure_ascii=False) + "\n")
            return {"trial_path": str(trial), "status": "ok", **verdict}
        except Exception as exc:
            return {
                "trial_path": str(trial),
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }


async def amain() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    load_dotenv()
    payload = json.loads(LEDGER.read_text())
    rows = [row for row in payload["rows"] if row.get("run_status") == "Valid"]
    if args.limit is not None:
        rows = rows[: args.limit]
    llm = LiteLLM(model_name=args.model, temperature=0)
    sem = asyncio.Semaphore(args.workers)
    results = await asyncio.gather(*(judge_one(row, llm, sem, args.model, args.force) for row in rows))
    summary = {
        "judge_model": args.model,
        "count": len(results),
        "status_counts": dict(Counter(row["status"] for row in results)),
        "results": results,
    }
    summary_path = ROOT / "outputs" / "longitudinal_pilot" / "swe_chat_success_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: summary[k] for k in ("judge_model", "count", "status_counts")}, indent=2))
    return 1 if summary["status_counts"].get("error") else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(amain()))
