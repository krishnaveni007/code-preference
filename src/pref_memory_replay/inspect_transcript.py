"""List the real user prompts of a SWE-chat Claude Code transcript with the agent activity between them.

A "prompt" is a non-sidechain user line whose content is text (not a tool_result, not a meta/command echo).
Usage: inspect_transcript.py <transcript.jsonl> [--around N]
"""
import argparse
import json
from collections import Counter


def load(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def is_prompt(e):
    if e.get("type") != "user" or e.get("isSidechain") or e.get("isMeta"):
        return False
    c = e.get("message", {}).get("content")
    if isinstance(c, str):
        return not c.startswith("<local-command") and not c.startswith("<command-name>/clear")
    if isinstance(c, list):
        return any(b.get("type") == "text" for b in c) and not any(b.get("type") == "tool_result" for b in c)
    return False


def text_of(e):
    c = e["message"]["content"]
    if isinstance(c, str):
        return c
    return "\n".join(b.get("text", "") for b in c if b.get("type") == "text")


def prompts(lines):
    return [i for i, e in enumerate(lines) if is_prompt(e)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--around", type=int)
    a = ap.parse_args()
    lines = load(a.path)
    ps = prompts(lines)
    rng = range(len(ps)) if a.around is None else range(max(0, a.around - 2), min(len(ps), a.around + 2))
    for k in rng:
        i = ps[k]
        j = ps[k + 1] if k + 1 < len(ps) else len(lines)
        tools = Counter()
        for e in lines[i:j]:
            if e.get("type") == "assistant" and not e.get("isSidechain"):
                for b in e["message"].get("content", []):
                    if isinstance(b, dict) and b.get("type") == "tool_use":
                        tools[b["name"]] += 1
        print(f"#{k} line {i} {lines[i].get('timestamp', '')[:16]} | {text_of(lines[i])[:160]!r}")
        print(f"     tools: {dict(tools)}")


if __name__ == "__main__":
    main()
