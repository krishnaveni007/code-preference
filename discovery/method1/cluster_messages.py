#!/usr/bin/env python3
"""Method 1a: cluster preference-related user messages on their raw text.

Stages, each cached on disk under --out-dir:
  embed   positives.jsonl -> embeddings.npy (+ ids.json)      one API call per batch
  cluster embeddings.npy  -> clusters_k{K}.json                k-means, no API calls
  name    clusters_k{K}.json -> clusters_k{K}.md               one API call per cluster
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
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
EMBED_MODEL = os.getenv("EMBED_MODEL", "openai/wine-gemini-embedding-001")
NAME_MODEL = os.getenv("PREFERENCE_JUDGE_MODEL", "openai/wine-claude-haiku-4-5")
MAX_WORDS = 300
EMBED_BATCH = 64


def _truncate(text: str) -> str:
    words = str(text).split()
    return " ".join(words[:MAX_WORDS])


def _embed_batch(texts: list[str], max_retries: int = 4) -> list[list[float]]:
    last: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = litellm.embedding(
                api_key=os.environ["LITELLM_API_KEY"], api_base=API_BASE,
                model=EMBED_MODEL, input=texts,
            )
            vectors = sorted(response.data, key=lambda d: d["index"])
            return [d["embedding"] for d in vectors]
        except Exception as error:  # noqa: BLE001
            last = error
            time.sleep(2 ** attempt)
    raise RuntimeError(f"embedding failed: {last}") from last


def stage_embed(positives: Path, out_dir: Path, workers: int) -> None:
    rows = [json.loads(line) for line in positives.open()]
    ids = [{"session_id": r["session_id"], "turn_number": r["turn_number"],
            "user_id": r["user_id"]} for r in rows]
    texts = [_truncate(r["content"]) for r in rows]
    batches = [texts[i:i + EMBED_BATCH] for i in range(0, len(texts), EMBED_BATCH)]
    print(f"{len(texts)} messages, {len(batches)} batches")
    vectors: list[list[list[float]]] = [None] * len(batches)  # type: ignore[list-item]
    with cf.ThreadPoolExecutor(workers) as pool:
        for i, vec in zip(range(len(batches)), pool.map(_embed_batch, batches)):
            vectors[i] = vec
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(batches)} batches", flush=True)
    matrix = np.asarray([v for batch in vectors for v in batch], dtype=np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    np.save(out_dir / "embeddings.npy", matrix)
    (out_dir / "ids.json").write_text(json.dumps(ids))
    print(f"saved {matrix.shape} -> {out_dir / 'embeddings.npy'}")


def stage_cluster(positives: Path, out_dir: Path, ks: list[int], seed: int) -> None:
    from sklearn.cluster import KMeans

    matrix = np.load(out_dir / "embeddings.npy")
    rows = [json.loads(line) for line in positives.open()]
    assert len(rows) == matrix.shape[0]
    rng = np.random.default_rng(seed)
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(matrix)
        labels = km.labels_
        clusters = []
        for c in range(k):
            idx = np.where(labels == c)[0]
            dist = np.linalg.norm(matrix[idx] - km.cluster_centers_[c], axis=1)
            nearest = idx[np.argsort(dist)[:20]]
            others = np.setdiff1d(idx, nearest)
            random_pick = rng.choice(others, size=min(5, len(others)), replace=False) if len(others) else []
            clusters.append({
                "cluster": c, "size": int(len(idx)),
                "n_users": int(len({rows[i]["user_id"] for i in idx})),
                "nearest": [_row_view(rows[i]) for i in nearest],
                "random": [_row_view(rows[i]) for i in random_pick],
            })
        clusters.sort(key=lambda c: -c["size"])
        out = out_dir / f"clusters_k{k}.json"
        out.write_text(json.dumps({"k": k, "assignments": labels.tolist(), "clusters": clusters},
                                  ensure_ascii=False))
        sizes = sorted((c["size"] for c in clusters), reverse=True)
        print(f"k={k}: sizes max={sizes[0]} median={sizes[len(sizes)//2]} min={sizes[-1]} -> {out}")


def _row_view(row: dict[str, Any]) -> dict[str, Any]:
    return {"user_id": row["user_id"], "session_id": row["session_id"],
            "turn_number": row["turn_number"], "text": _truncate(row["content"])[:600]}


NAME_INSTRUCTIONS = {
    "messages": """You are shown messages that a clustering algorithm grouped together. \
They were all sent by developers to an AI coding agent and were all judged to express some \
preference about the code or about how the agent should work. Write one sentence saying what \
these messages have in common. If the common thread is a topic or technology rather than a \
preference, say so plainly.""",
    "statements": """You are shown short preference statements that a clustering algorithm grouped \
together. Each was rewritten by a model from a message a developer sent to an AI coding agent. \
Write one sentence saying what preference these statements share. If they share a topic rather \
than a preference, or share nothing, say so plainly.""",
}
KIND = "messages"


def _name_cluster(cluster: dict[str, Any]) -> str:
    sample = "\n\n".join(f"- {m['text'][:400]}" for m in cluster["nearest"])
    for attempt in range(3):
        try:
            response = litellm.completion(
                api_key=os.environ["LITELLM_API_KEY"], base_url=API_BASE, model=NAME_MODEL,
                messages=[{"role": "system", "content": NAME_INSTRUCTIONS[KIND]},
                          {"role": "user", "content": sample}],
                max_tokens=150,
            )
            return response.choices[0].message.content.strip()
        except Exception:  # noqa: BLE001
            if attempt == 2:
                return "(naming failed)"
            time.sleep(2 ** attempt)
    return "(naming failed)"


def stage_name(out_dir: Path, ks: list[int], workers: int) -> None:
    for k in ks:
        data = json.loads((out_dir / f"clusters_k{k}.json").read_text())
        clusters = data["clusters"]
        with cf.ThreadPoolExecutor(workers) as pool:
            names = list(pool.map(_name_cluster, clusters))
        title = "1a — raw-message" if KIND == "messages" else "1b — preference-statement"
        lines = [f"# Method {title} k-means, k={k}", "",
                 f"{sum(c['size'] for c in clusters)} messages. Clusters sorted by size.", ""]
        for c, name in zip(clusters, names):
            c["name"] = name
            lines += [f"## cluster {c['cluster']} — {c['size']} msgs, {c['n_users']} users", "",
                      f"**{name}**", "", "Nearest to centroid:", ""]
            lines += [f"- {m['text'][:300].replace(chr(10), ' ')}" for m in c["nearest"][:10]]
            lines += ["", "Random:", ""]
            lines += [f"- {m['text'][:300].replace(chr(10), ' ')}" for m in c["random"]]
            lines.append("")
        (out_dir / f"clusters_k{k}.json").write_text(json.dumps(data, ensure_ascii=False))
        (out_dir / f"clusters_k{k}.md").write_text("\n".join(lines))
        print(f"k={k}: named {len(clusters)} clusters -> {out_dir / f'clusters_k{k}.md'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["embed", "cluster", "name", "all"])
    parser.add_argument("--positives", type=Path, default=Path("outputs/discovery/method1/positives.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/discovery/method1/cluster_raw"))
    parser.add_argument("--k", type=int, nargs="+", default=[20, 40, 80])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--kind", choices=list(NAME_INSTRUCTIONS), default="messages",
                        help="what the input rows are; picks the naming instructions and report title")
    args = parser.parse_args()
    global KIND
    KIND = args.kind
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.stage in ("embed", "all") and not (args.out_dir / "embeddings.npy").exists():
        stage_embed(args.positives, args.out_dir, args.workers)
    if args.stage in ("cluster", "all"):
        stage_cluster(args.positives, args.out_dir, args.k, args.seed)
    if args.stage in ("name", "all"):
        stage_name(args.out_dir, args.k, args.workers)


if __name__ == "__main__":
    main()
