#!/usr/bin/env python3
"""Give each k-means cluster a short name and a one-sentence description, Clio-style:
the model sees random members AND the nearest non-members, and must name what
distinguishes the cluster from its neighbours. Writes a final category table."""

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

INSTRUCTIONS = """You are naming one group of short preference statements. Each statement was \
rewritten from a message a developer sent to an AI coding agent and says how that developer wants \
code written or how they want the agent to behave. The statements were grouped by a clustering \
algorithm.

You get two lists: statements INSIDE the group, and the nearest statements OUTSIDE the group. \
Write a name and a description that fit the inside statements and would NOT fit the outside ones.

- name: at most 10 words. Name the preference, not the topic. "Ask before acting on ambiguous \
tasks" is a name; "Agent behaviour" is not.
- description: one sentence saying what these developers want, specific enough to exclude the \
outside statements.
- If the inside statements pull in two opposite directions on the same question, name the \
question and say so in the description."""

SCHEMA = {"type": "object", "additionalProperties": False,
          "properties": {"name": {"type": "string"}, "description": {"type": "string"}},
          "required": ["name", "description"]}


def name_one(inside: list[str], outside: list[str], retries: int = 3) -> dict:
    body = ("<inside>\n" + "\n".join(f"- {s}" for s in inside) + "\n</inside>\n\n"
            "<outside>\n" + "\n".join(f"- {s}" for s in outside) + "\n</outside>")
    last = None
    for attempt in range(retries):
        try:
            r = litellm.completion(api_key=os.environ["LITELLM_API_KEY"], base_url=API_BASE, model=MODEL,
                                   messages=[{"role": "system", "content": INSTRUCTIONS}, {"role": "user", "content": body}],
                                   response_format={"type": "json_schema", "json_schema": {"name": "n", "strict": True, "schema": SCHEMA}})
            return json.loads(r.choices[0].message.content)
        except Exception as e:  # noqa: BLE001
            last = e; time.sleep(2 ** attempt)
    return {"name": "(naming failed)", "description": str(last)[:200]}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=Path("outputs/discovery/method1/cluster_statements"))
    p.add_argument("--positives", type=Path, default=Path("outputs/discovery/method1/statements_flat.jsonl"))
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--clusters-file", type=Path, default=None,
                   help="cluster json to name (default clusters_k{k}.json); noise label -1 is allowed")
    p.add_argument("--emb-dir", type=Path, default=None, help="where embeddings.npy lives (default --out-dir)")
    p.add_argument("--tag", default=None, help="output name: categories_{tag}.md (default k{k})")
    p.add_argument("--n-inside", type=int, default=30)
    p.add_argument("--n-outside", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    X = np.load((args.emb_dir or args.out_dir) / "embeddings.npy")
    rows = [json.loads(l) for l in args.positives.open()]
    texts = [r["content"] for r in rows]
    turns = np.array([f"{r['session_id']}:{r['turn_number']}" for r in rows])
    cfile = args.clusters_file or (args.out_dir / f"clusters_k{args.k}.json")
    data = json.loads(cfile.read_text())
    L = np.asarray(data["assignments"]); k = int(L.max()) + 1
    tag = args.tag or f"k{args.k}"
    rng = np.random.default_rng(args.seed)
    C = np.vstack([X[L == c].mean(0) for c in range(k)]); C /= np.linalg.norm(C, axis=1, keepdims=True)

    jobs = []
    for c in range(k):
        idx = np.where(L == c)[0]
        inside = [texts[i] for i in rng.choice(idx, size=min(args.n_inside, len(idx)), replace=False)]
        out_idx = np.where(L != c)[0]
        nearest_out = out_idx[np.argsort(-(X[out_idx] @ C[c]))[:args.n_outside]]
        jobs.append((c, inside, [texts[i] for i in nearest_out]))
    with cf.ThreadPoolExecutor(8) as pool:
        names = list(pool.map(lambda j: name_one(j[1], j[2]), jobs))

    by_c = {cl["cluster"]: cl for cl in data["clusters"]}
    for (c, _, _), nm in zip(jobs, names):
        idx = np.where(L == c)[0]
        by_c[c].update({"short_name": nm["name"], "description": nm["description"],
                        "n_turns": int(len(set(turns[idx])))})
    cfile.write_text(json.dumps(data, ensure_ascii=False))

    ordered = sorted(data["clusters"], key=lambda cl: -cl["n_turns"])
    noise = int((L < 0).sum())
    lines = [f"# Preference categories — {tag}: {k} clusters on {len(rows)} preference statements"
             + (f", {noise} ({noise/len(rows):.0%}) unassigned (noise)" if noise else ""), "",
             "Names and descriptions are model-written from 30 random members and the 30 nearest non-members. "
             "turns = distinct user turns whose statements fall in the cluster.", "",
             "| # | category | turns | stmts | description |", "|---|---|---|---|---|"]
    for i, cl in enumerate(ordered, 1):
        lines.append(f"| {i} | **{cl['short_name']}** | {cl['n_turns']} | {cl['size']} | {cl['description']} |")
    lines += ["", "## Examples (nearest to centroid)", ""]
    for i, cl in enumerate(ordered, 1):
        lines += [f"### {i}. {cl['short_name']}", ""] + [f"- {m['text'][:200]}" for m in cl["nearest"][:5]] + [""]
    (args.out_dir / f"categories_{tag}.md").write_text("\n".join(lines))
    print(f"-> {args.out_dir / f'categories_{tag}.md'}\n")
    for i, cl in enumerate(ordered, 1):
        print(f"{i:2d}. [{cl['n_turns']:4d} turns] {cl['short_name']}\n      {cl['description']}")


if __name__ == "__main__":
    main()
