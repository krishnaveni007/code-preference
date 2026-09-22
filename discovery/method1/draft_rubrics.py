#!/usr/bin/env python3
"""Draft one rubric per HDBSCAN cluster.

A rubric is an axis: the question it asks, pole A (what this cluster's developers want),
pole B (the opposite). Pole B is marked by where its evidence comes from:
  in_cluster    dissenting statements inside this cluster (quoted)
  other_cluster another cluster is the opposite pole (given via --opposites)
  none          no statement supports it; written as the logical opposite only
Model drafts the text from 30 random members + the cluster description; counts are real."""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import time
from pathlib import Path

import litellm
import numpy as np
from dotenv import load_dotenv

litellm.suppress_debug_info = True
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")
API_BASE = os.getenv("LITELLM_API_BASE", "https://ai-gateway.andrew.cmu.edu/")
MODEL = os.getenv("NAME_MODEL", "openai/wine-claude-sonnet-4-6")

INSTRUCTIONS = """You are turning one cluster of developer preference statements into a rubric axis.

You get the cluster's name and description, and 30 statements sampled from it. Each statement \
was rewritten from a message a developer sent to an AI coding agent.

Write:
- name: 3 to 8 words naming the axis (the question), not one side of it. \
"Ask before acting vs. proceed autonomously", not "Ask before acting".
- question: one sentence, the choice this axis is about, phrased so that either answer is reasonable.
- pole_a: what the developers in THIS cluster want, one sentence, in their terms.
- pole_b: the opposite position, one sentence. Write it as something a reasonable developer could \
want, not as a strawman.
- pole_b_quotes: statements FROM THE 30 GIVEN that actually take the pole_b position, verbatim, \
up to 3. Empty list if none do.
- pole_a_quotes: 3 statements from the 30 that best show pole_a, verbatim.

Do not use project-specific names. If the 30 statements pull in two directions on the same \
question, that is a two-sided axis: put one side in pole_a and the other in pole_b and quote both."""

SCHEMA = {"type": "object", "additionalProperties": False,
          "properties": {k: {"type": "string"} for k in ("name", "question", "pole_a", "pole_b")}
          | {"pole_a_quotes": {"type": "array", "items": {"type": "string"}},
             "pole_b_quotes": {"type": "array", "items": {"type": "string"}}},
          "required": ["name", "question", "pole_a", "pole_b", "pole_a_quotes", "pole_b_quotes"]}


def draft(cluster: dict, sample: list[str], retries: int = 3) -> dict:
    body = (f"<cluster name>{cluster['short_name']}</cluster name>\n<description>{cluster['description']}</description>\n\n"
            "<statements>\n" + "\n".join(f"- {s}" for s in sample) + "\n</statements>")
    last = None
    for attempt in range(retries):
        try:
            r = litellm.completion(api_key=os.environ["LITELLM_API_KEY"], base_url=API_BASE, model=MODEL,
                                   messages=[{"role": "system", "content": INSTRUCTIONS}, {"role": "user", "content": body}],
                                   response_format={"type": "json_schema", "json_schema": {"name": "rubric", "strict": True, "schema": SCHEMA}})
            return json.loads(r.choices[0].message.content)
        except Exception as e:  # noqa: BLE001
            last = e; time.sleep(2 ** attempt)
    return {"name": "(failed)", "question": str(last)[:200], "pole_a": "", "pole_b": "", "pole_a_quotes": [], "pole_b_quotes": []}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--clusters", type=Path, default=Path("outputs/discovery/method1/v2/cluster_hdbscan/clusters_mcs100_leaf_ms10.json"))
    p.add_argument("--positives", type=Path, default=Path("outputs/discovery/method1/v2/statements_flat.jsonl"))
    p.add_argument("--out", type=Path, default=Path("outputs/discovery/method1/v2/rubrics_draft"))
    p.add_argument("--opposites", default="", help="rank pairs whose clusters are opposite poles, e.g. 6:14,7:14,13:14")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    data = json.loads(args.clusters.read_text()); L = np.asarray(data["assignments"])
    rows = [json.loads(l) for l in args.positives.open()]
    turns = np.array([f"{r['session_id']}:{r['turn_number']}" for r in rows])
    ordered = sorted(data["clusters"], key=lambda c: -c["n_turns"])
    rank_of = {c["cluster"]: i + 1 for i, c in enumerate(ordered)}
    opp = {}
    for pair in filter(None, args.opposites.split(",")):
        a, b = map(int, pair.split(":")); opp[a] = b
    rng = np.random.default_rng(args.seed)

    def job(c):
        idx = np.where(L == c["cluster"])[0]
        sample = [rows[i]["content"] for i in rng.choice(idx, size=min(30, len(idx)), replace=False)]
        return draft(c, sample)
    with cf.ThreadPoolExecutor(8) as pool:
        drafts = list(pool.map(job, ordered))

    rubrics = []
    for c, d in zip(ordered, drafts):
        r = rank_of[c["cluster"]]
        if d["pole_b_quotes"]:
            src = "in_cluster"
        elif r in opp:
            src = f"other_cluster (#{opp[r]})"
        else:
            src = "none"
        rubrics.append({"rank": r, "cluster": c["cluster"], "n_turns": c["n_turns"], "n_statements": c["size"],
                        "seed_stability_J": c["min_jaccard"], **d, "pole_b_source": src})
    args.out.with_suffix(".json").write_text(json.dumps(rubrics, ensure_ascii=False, indent=1))

    n_two = sum(1 for r in rubrics if r["pole_b_source"] != "none")
    lines = [f"# Rubric draft — {len(rubrics)} axes from HDBSCAN clusters (v2 statements)", "",
             f"Pole B has data support in {n_two}/{len(rubrics)} axes (dissenting statements inside the cluster, or an opposite cluster). "
             "The rest are written as the logical opposite and marked **no data**. Text is model-drafted; counts are real.", ""]
    for r in rubrics:
        flag = "" if r["pole_b_source"] != "none" else " — **pole B: no data**"
        lines += [f"## {r['rank']}. {r['name']}", "",
                  f"*{r['question']}*", "",
                  f"- **Turns:** {r['n_turns']} ({r['n_statements']} statements) · seed stability J = {r['seed_stability_J']:.2f}",
                  f"- **Pole A:** {r['pole_a']}",
                  f"- **Pole B:** {r['pole_b']} ({r['pole_b_source']}){flag}", "",
                  "Pole A examples:"] + [f"- {q}" for q in r["pole_a_quotes"]]
        if r["pole_b_quotes"]:
            lines += ["", "Pole B examples (from this cluster):"] + [f"- {q}" for q in r["pole_b_quotes"]]
        lines.append("")
    args.out.with_suffix(".md").write_text("\n".join(lines))
    print(f"-> {args.out.with_suffix('.md')}\n")
    for r in rubrics:
        print(f"{r['rank']:2d}. [{r['n_turns']:3d} turns, J={r['seed_stability_J']:.2f}, pole B: {r['pole_b_source']}] {r['name']}")
        print(f"      Q: {r['question']}")
        print(f"      A: {r['pole_a']}")
        print(f"      B: {r['pole_b']}")


if __name__ == "__main__":
    main()
