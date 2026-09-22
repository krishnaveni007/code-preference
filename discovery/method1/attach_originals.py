#!/usr/bin/env python3
"""Attach the original user message to every quoted statement in the rubric draft."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rubrics", type=Path, default=Path("outputs/discovery/method1/v2/rubrics_draft.json"))
    p.add_argument("--clusters", type=Path, default=Path("outputs/discovery/method1/v2/cluster_hdbscan/clusters_mcs100_leaf_ms10.json"))
    p.add_argument("--statements", type=Path, default=Path("outputs/discovery/method1/v2/statements_flat.jsonl"))
    p.add_argument("--originals", type=Path, default=Path("outputs/discovery/method1/positives_dedup.jsonl"))
    args = p.parse_args()

    norm = lambda s: re.sub(r"\W+", " ", s.lower()).strip()
    rows = [json.loads(l) for l in args.statements.open()]
    L = np.asarray(json.loads(args.clusters.read_text())["assignments"])
    orig = {(r["session_id"], r["turn_number"]): r["content"] for r in map(json.loads, args.originals.open())}
    rubrics = json.loads(args.rubrics.read_text())

    def resolve(quote: str, cluster: int) -> dict:
        key = norm(quote)
        for i in np.where(L == cluster)[0]:
            if norm(rows[i]["content"]) == key:
                r = rows[i]
                return {"statement": r["content"], "original": orig.get((r["session_id"], r["turn_number"]), ""),
                        "session_id": r["session_id"], "turn_number": r["turn_number"], "user_id": r["user_id"]}
        return {"statement": quote, "original": "(not found)", "session_id": None, "turn_number": None, "user_id": None}

    for rb in rubrics:
        rb["pole_a_quotes"] = [resolve(q, rb["cluster"]) if isinstance(q, str) else q for q in rb["pole_a_quotes"]]
        rb["pole_b_quotes"] = [resolve(q, rb["cluster"]) if isinstance(q, str) else q for q in rb["pole_b_quotes"]]
    args.rubrics.write_text(json.dumps(rubrics, ensure_ascii=False, indent=1))

    def block(quotes: list[dict]) -> list[str]:
        out = []
        for q in quotes:
            o = " ".join(q["original"].split())
            out += [f"- **{q['statement']}**", f"  - 原文: {o[:500]}{'…' if len(o) > 500 else ''}"]
        return out

    n_two = sum(1 for r in rubrics if r["pole_b_source"] != "none")
    lines = [f"# Rubric draft — {len(rubrics)} axes from HDBSCAN clusters (v2 statements)", "",
             f"Pole B has data support in {n_two}/{len(rubrics)} axes. Each quote shows the rewritten statement in bold "
             "and the original user message beneath it.", ""]
    for r in rubrics:
        flag = "" if r["pole_b_source"] != "none" else " — **pole B: no data**"
        lines += [f"## {r['rank']}. {r['name']}", "", f"*{r['question']}*", "",
                  f"- **Turns:** {r['n_turns']} ({r['n_statements']} statements) · seed stability J = {r['seed_stability_J']:.2f}",
                  f"- **Pole A:** {r['pole_a']}", f"- **Pole B:** {r['pole_b']} ({r['pole_b_source']}){flag}", "",
                  "Pole A examples:"] + block(r["pole_a_quotes"])
        if r["pole_b_quotes"]:
            lines += ["", "Pole B examples (from this cluster):"] + block(r["pole_b_quotes"])
        lines.append("")
    args.rubrics.with_suffix(".md").write_text("\n".join(lines))
    missing = sum(1 for r in rubrics for q in r["pole_a_quotes"] + r["pole_b_quotes"] if q["original"] == "(not found)")
    print(f"attached originals; unresolved quotes: {missing}")
    r = next(x for x in rubrics if x["rank"] == 17)
    print(f"\n例：#{r['rank']} {r['name']}\n")
    for q in r["pole_a_quotes"][:2] + r["pole_b_quotes"][:2]:
        print(f"改写: {q['statement']}\n原文: {' '.join(q['original'].split())[:300]}\n")


if __name__ == "__main__":
    main()
