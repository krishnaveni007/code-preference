#!/usr/bin/env python3
"""Have the model read all per-cluster rubric drafts at once and consolidate them into a
smaller set of rubrics: merge clusters that are two answers to one question or the same
question twice; keep different questions apart even when they share a topic."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import litellm
from dotenv import load_dotenv

litellm.suppress_debug_info = True
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")
API_BASE = os.getenv("LITELLM_API_BASE", "https://ai-gateway.andrew.cmu.edu/")
MODEL = os.getenv("NAME_MODEL", "openai/wine-claude-sonnet-4-6")

INSTRUCTIONS = """You are given draft axes, one per cluster of developer preference statements. \
Each has a question, side A (what that cluster's developers want), side B (the opposite), and \
whether side B has any real statements behind it.

Consolidate them into a final list of rubrics. A rubric is one question about how an AI coding \
agent should work or how code should be written, with two answers that reasonable developers \
could each want.

Rules:
- If two drafts are opposite answers to one question, they become ONE rubric; put each in its side.
- If two drafts ask the same question, merge them.
- If two drafts share a topic but ask different questions (e.g. "should tests be run continuously" \
vs "how should tests be structured"), keep them separate.
- If a draft's side B is not something a developer would want (e.g. "commit secrets to git"), the \
draft is not a two-sided preference. Either rewrite it as a matter of degree ("how much / how strict") \
with two reasonable answers, or list it under "uniform_requests" (things developers agree on).
- Every draft number must appear exactly once: in one rubric's side_a_clusters or side_b_clusters, \
or in uniform_requests.
- Wording: plain, specific, no project names. name 3-8 words; question one sentence; each side one sentence."""

SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "rubrics": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "properties": {"name": {"type": "string"}, "question": {"type": "string"},
                           "side_a": {"type": "string"}, "side_b": {"type": "string"},
                           "side_a_clusters": {"type": "array", "items": {"type": "integer"}},
                           "side_b_clusters": {"type": "array", "items": {"type": "integer"}},
                           "note": {"type": "string"}},
            "required": ["name", "question", "side_a", "side_b", "side_a_clusters", "side_b_clusters", "note"]}},
        "uniform_requests": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "properties": {"name": {"type": "string"}, "clusters": {"type": "array", "items": {"type": "integer"}}},
            "required": ["name", "clusters"]}},
    },
    "required": ["rubrics", "uniform_requests"],
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--drafts", type=Path, default=Path("outputs/discovery/method1/final/rubrics_draft.json"))
    p.add_argument("--out", type=Path, default=Path("outputs/discovery/method1/final/rubrics"))
    args = p.parse_args()
    drafts = json.loads(args.drafts.read_text())
    by_rank = {d["rank"]: d for d in drafts}

    body = "\n\n".join(
        f"[{d['rank']}] {d['name']}  ({d['n_turns']} turns)\n  question: {d['question']}\n  side A: {d['pole_a']}\n"
        f"  side B: {d['pole_b']}\n  side B evidence: {d['pole_b_source']}" for d in drafts)
    r = litellm.completion(api_key=os.environ["LITELLM_API_KEY"], base_url=API_BASE, model=MODEL,
                           messages=[{"role": "system", "content": INSTRUCTIONS}, {"role": "user", "content": body}],
                           response_format={"type": "json_schema", "json_schema": {"name": "rubrics", "strict": True, "schema": SCHEMA}},
                           max_tokens=8000)
    out = json.loads(r.choices[0].message.content)

    used = [c for rb in out["rubrics"] for c in rb["side_a_clusters"] + rb["side_b_clusters"]] + \
           [c for u in out["uniform_requests"] for c in u["clusters"]]
    missing = sorted(set(by_rank) - set(used)); dup = sorted({c for c in used if used.count(c) > 1})
    for rb in out["rubrics"]:
        rb["turns_a"] = sum(by_rank[c]["n_turns"] for c in rb["side_a_clusters"] if c in by_rank)
        rb["turns_b"] = sum(by_rank[c]["n_turns"] for c in rb["side_b_clusters"] if c in by_rank)
        rb["side_b_has_data"] = bool(rb["side_b_clusters"]) or any(by_rank[c]["pole_b_source"] != "none" for c in rb["side_a_clusters"] if c in by_rank)
    for u in out["uniform_requests"]:
        u["turns"] = sum(by_rank[c]["n_turns"] for c in u["clusters"] if c in by_rank)
    out["check"] = {"missing_clusters": missing, "duplicated_clusters": dup}
    args.out.with_suffix(".json").write_text(json.dumps(out, ensure_ascii=False, indent=1))

    lines = [f"# Consolidated rubrics — {len(out['rubrics'])} rubrics + {len(out['uniform_requests'])} uniform requests",
             "", "Built by the model from the 25 per-cluster drafts. Turn counts are sums over the absorbed clusters. "
             "\"side B has data\" = some real statements take side B (in an opposite cluster or inside a merged cluster).", ""]
    for i, rb in enumerate(out["rubrics"], 1):
        lines += [f"## {i}. {rb['name']}", "", f"*{rb['question']}*", "",
                  f"- **A** ({rb['turns_a']} turns, clusters {rb['side_a_clusters']}): {rb['side_a']}",
                  f"- **B** ({rb['turns_b']} turns, clusters {rb['side_b_clusters']}): {rb['side_b']}"
                  + ("" if rb["side_b_has_data"] else "  — **no data for B**"),
                  *( [f"- note: {rb['note']}"] if rb["note"] else [] ), ""]
    if out["uniform_requests"]:
        lines += ["## Uniform requests (no disagreement in the data)", ""]
        lines += [f"- **{u['name']}** — {u['turns']} turns, clusters {u['clusters']}" for u in out["uniform_requests"]]
    (args.out.with_suffix(".md")).write_text("\n".join(lines))

    print(f"-> {args.out.with_suffix('.md')}   check: missing={missing} duplicated={dup}\n")
    for i, rb in enumerate(out["rubrics"], 1):
        flag = "" if rb["side_b_has_data"] else "   [B: no data]"
        print(f"{i:2d}. {rb['name']}  (A {rb['turns_a']} / B {rb['turns_b']} turns; clusters A{rb['side_a_clusters']} B{rb['side_b_clusters']}){flag}")
        print(f"     Q: {rb['question']}\n     A: {rb['side_a']}\n     B: {rb['side_b']}")
        if rb["note"]: print(f"     note: {rb['note']}")
    print("\nuniform requests:")
    for u in out["uniform_requests"]:
        print(f"  - {u['name']} ({u['turns']} turns, clusters {u['clusters']})")


if __name__ == "__main__":
    main()
