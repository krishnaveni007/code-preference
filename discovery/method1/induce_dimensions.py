#!/usr/bin/env python3
"""Induce preference dimensions from the 1b statements by reading them in batches.

  induce  statements_flat.jsonl -> induction/candidates.jsonl   (one API call per batch of 300)
  merge   candidates.jsonl      -> induction/dimensions.json/.md (embed, group, one call per group)

Every candidate remembers which batch proposed it, so a merged dimension carries
"batch support": in how many independent batches it was proposed.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import litellm
import numpy as np
from dotenv import load_dotenv

litellm.suppress_debug_info = True
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")
API_BASE = os.getenv("LITELLM_API_BASE", "https://ai-gateway.andrew.cmu.edu/")
INDUCE_MODEL = os.getenv("INDUCE_MODEL", "openai/wine-claude-sonnet-4-6")
EMBED_MODEL = os.getenv("EMBED_MODEL", "openai/wine-gemini-embedding-001")

INDUCE_INSTRUCTIONS = """You are given a batch of short preference statements. Each was rewritten \
from a message a developer sent to an AI coding agent, and says how that developer wants code \
written or how they want the agent to behave.

Find the preference DIMENSIONS that recur in this batch. A dimension is an axis along which \
different developers could reasonably want different things, e.g. "how much to document" or \
"ask before acting vs. act autonomously". It is not a topic (UI, git, testing) and not a single \
person's request.

For each dimension give:
- name: 2 to 6 words
- description: one sentence, what choice this dimension is about
- high: what a developer at one end wants
- low: what a developer at the other end wants
- n_statements: roughly how many statements in this batch fall on this dimension

Only include dimensions supported by at least 5 statements in this batch. Order by n_statements, \
largest first. Do not pad the list."""

INDUCE_INSTRUCTIONS_NOPOLES = """You are given a batch of short preference statements. Each was \
rewritten from a message a developer sent to an AI coding agent, and says how that developer wants \
code written or how they want the agent to behave.

Find the preference DIMENSIONS that recur in this batch. A dimension is a question on which \
different developers could reasonably want different things, e.g. "how much to document" or \
"whether to ask before acting". It is not a topic (UI, git, testing) and not a single person's request.

For each dimension give:
- name: 2 to 6 words
- description: one sentence saying what choice this dimension is about
- n_statements: roughly how many statements in this batch fall on this dimension (your estimate)

Only include dimensions supported by at least 5 statements in this batch. Order by n_statements, \
largest first. Do not pad the list."""

INDUCE_SCHEMA_NOPOLES = {
    "type": "object", "additionalProperties": False,
    "properties": {"dimensions": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "properties": {"name": {"type": "string"}, "description": {"type": "string"},
                       "n_statements": {"type": "integer"}},
        "required": ["name", "description", "n_statements"]}}},
    "required": ["dimensions"],
}

INDUCE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {"dimensions": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "properties": {"name": {"type": "string"}, "description": {"type": "string"},
                       "high": {"type": "string"}, "low": {"type": "string"},
                       "n_statements": {"type": "integer"}},
        "required": ["name", "description", "high", "low", "n_statements"]}}},
    "required": ["dimensions"],
}

MERGE_INSTRUCTIONS = """Below are candidate preference dimensions that were proposed independently \
from different batches of developer preference statements, and that an embedding model judged \
similar. Decide whether they are the same dimension. If they are, write one merged definition. \
If the group actually contains two or more distinct dimensions, write one definition for each.

For each output dimension give name (2-6 words), description (one sentence), high, low, and \
the list of input candidate indices it absorbs. Every input index must appear in exactly one \
output dimension."""

MERGE_INSTRUCTIONS_NOPOLES = """Below are candidate preference dimensions that were proposed \
independently from different batches of developer preference statements, and that an embedding \
model judged similar. Decide whether they are the same dimension. If they are, write one merged \
definition. If the group actually contains two or more distinct dimensions, write one definition \
for each.

For each output dimension give name (2-6 words), description (one sentence), and the list of input \
candidate indices it absorbs. Every input index must appear in exactly one output dimension."""

MERGE_SCHEMA_NOPOLES = {
    "type": "object", "additionalProperties": False,
    "properties": {"dimensions": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "properties": {"name": {"type": "string"}, "description": {"type": "string"},
                       "absorbs": {"type": "array", "items": {"type": "integer"}}},
        "required": ["name", "description", "absorbs"]}}},
    "required": ["dimensions"],
}

POLES = True

MERGE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {"dimensions": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "properties": {"name": {"type": "string"}, "description": {"type": "string"},
                       "high": {"type": "string"}, "low": {"type": "string"},
                       "absorbs": {"type": "array", "items": {"type": "integer"}}},
        "required": ["name", "description", "high", "low", "absorbs"]}}},
    "required": ["dimensions"],
}


def _call(model: str, system: str, user: str, schema: dict[str, Any], name: str, retries: int = 3) -> dict[str, Any]:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            response = litellm.completion(
                api_key=os.environ["LITELLM_API_KEY"], base_url=API_BASE, model=model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                response_format={"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}},
                max_tokens=8000,
            )
            return json.loads(response.choices[0].message.content)
        except Exception as error:  # noqa: BLE001
            last = error
            time.sleep(2 ** attempt)
    raise RuntimeError(f"{name} failed: {last}") from last


def stage_induce(statements: Path, out_dir: Path, batch_size: int, seed: int, model: str, workers: int) -> None:
    rows = [json.loads(line) for line in statements.open()]
    random.Random(seed).shuffle(rows)
    batches = [rows[i:i + batch_size] for i in range(0, len(rows), batch_size)]
    print(f"{len(rows)} statements -> {len(batches)} batches of {batch_size}, model {model}")

    def work(item: tuple[int, list[dict[str, Any]]]) -> dict[str, Any]:
        b, batch = item
        text = "\n".join(f"- {r['content']}" for r in batch)
        rec = {"batch": b, "seed": seed, "n_statements": len(batch), "model": model, "dimensions": [], "error": ""}
        try:
            instr, schema = (INDUCE_INSTRUCTIONS, INDUCE_SCHEMA) if POLES else (INDUCE_INSTRUCTIONS_NOPOLES, INDUCE_SCHEMA_NOPOLES)
            rec["dimensions"] = _call(model, instr, text, schema, "dimensions")["dimensions"]
        except Exception as error:  # noqa: BLE001
            rec["error"] = str(error)[:300]
        return rec

    out = out_dir / "candidates.jsonl"
    with out.open("w") as handle, cf.ThreadPoolExecutor(workers) as pool:
        for rec in pool.map(work, enumerate(batches)):
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"  batch {rec['batch']:2d}: {len(rec['dimensions'])} dimensions"
                  + (f"  ERROR {rec['error'][:80]}" if rec["error"] else ""), flush=True)
    print(f"-> {out}")


def _embed(texts: list[str]) -> np.ndarray:
    vectors: list[list[float]] = []
    for i in range(0, len(texts), 64):
        response = litellm.embedding(api_key=os.environ["LITELLM_API_KEY"], api_base=API_BASE,
                                     model=EMBED_MODEL, input=texts[i:i + 64])
        vectors += [d["embedding"] for d in sorted(response.data, key=lambda d: d["index"])]
    matrix = np.asarray(vectors, dtype=np.float32)
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def stage_merge(out_dir: Path, distance: float, model: str, workers: int) -> None:
    from sklearn.cluster import AgglomerativeClustering

    cands: list[dict[str, Any]] = []
    for line in (out_dir / "candidates.jsonl").open():
        rec = json.loads(line)
        for d in rec["dimensions"]:
            cands.append({**d, "batch": rec["batch"]})
    for i, c in enumerate(cands):
        c["idx"] = i
    print(f"{len(cands)} candidates from {len({c['batch'] for c in cands})} batches")

    texts = [f"{c['name']}: {c['description']}" + (f" One end: {c['high']} Other end: {c['low']}" if POLES else "")
             for c in cands]
    X = _embed(texts)
    groups = AgglomerativeClustering(n_clusters=None, distance_threshold=distance, metric="cosine",
                                     linkage="average").fit_predict(X)
    n_groups = groups.max() + 1
    print(f"agglomerative grouping at cosine distance {distance}: {n_groups} groups")

    def work(g: int) -> list[dict[str, Any]]:
        members = [c for c in cands if groups[c["idx"]] == g]
        if len(members) == 1:
            c = members[0]
            return [{**{k: c[k] for k in ("name", "description", "high", "low") if k in c}, "absorbs": [c["idx"]]}]
        text = "\n\n".join(f"[{c['idx']}] {c['name']}\n  {c['description']}"
                           + (f"\n  high: {c['high']}\n  low: {c['low']}" if POLES else "") for c in members)
        instr, schema = (MERGE_INSTRUCTIONS, MERGE_SCHEMA) if POLES else (MERGE_INSTRUCTIONS_NOPOLES, MERGE_SCHEMA_NOPOLES)
        try:
            return _call(model, instr, text, schema, "merged")["dimensions"]
        except Exception as error:  # noqa: BLE001
            return [{**{k: members[0][k] for k in ("name", "description", "high", "low") if k in members[0]},
                     "absorbs": [c["idx"] for c in members], "error": str(error)[:200]}]

    with cf.ThreadPoolExecutor(workers) as pool:
        merged = [d for out in pool.map(work, range(n_groups)) for d in out]

    by_idx = {c["idx"]: c for c in cands}
    for d in merged:
        absorbed = [by_idx[i] for i in d["absorbs"] if i in by_idx]
        d["n_candidates"] = len(absorbed)
        d["batch_support"] = len({c["batch"] for c in absorbed})
        d["statement_estimate"] = int(sum(c["n_statements"] for c in absorbed))
        d["candidate_names"] = [c["name"] for c in absorbed]
    merged.sort(key=lambda d: (-d["batch_support"], -d["statement_estimate"]))
    (out_dir / "dimensions.json").write_text(json.dumps(merged, ensure_ascii=False, indent=1))

    n_batches = len({c["batch"] for c in cands})
    lines = [f"# Induced preference dimensions", "",
             f"{len(cands)} candidates from {n_batches} batches -> {len(merged)} dimensions. "
             f"batch_support = how many of the {n_batches} batches independently proposed it.", "",
             "| # | dimension | batches | est. statements (model) | description |" + (" high | low |" if POLES else ""),
             "|---|---|---|---|---|" + ("---|---|" if POLES else "")]
    for i, d in enumerate(merged, 1):
        lines.append(f"| {i} | **{d['name']}** | {d['batch_support']}/{n_batches} | {d['statement_estimate']} | "
                     f"{d['description']} |" + (f" {d['high']} | {d['low']} |" if POLES else ""))
    (out_dir / "dimensions.md").write_text("\n".join(lines))
    print(f"-> {len(merged)} dimensions, {out_dir / 'dimensions.md'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["induce", "merge"])
    parser.add_argument("--statements", type=Path, default=Path("outputs/discovery/method1/statements_flat.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/discovery/method1/induction"))
    parser.add_argument("--batch-size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default=INDUCE_MODEL)
    parser.add_argument("--distance", type=float, default=0.2, help="merge: cosine distance threshold for grouping")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--no-poles", action="store_true", help="dimensions carry only name + description")
    args = parser.parse_args()
    global POLES
    POLES = not args.no_poles
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "induce":
        stage_induce(args.statements, args.out_dir, args.batch_size, args.seed, args.model, args.workers)
    else:
        stage_merge(args.out_dir, args.distance, args.model, args.workers)


if __name__ == "__main__":
    main()
