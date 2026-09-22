#!/usr/bin/env python3
"""UMAP -> HDBSCAN on the 1b statement embeddings.

  sweep   try several min_cluster_size values, print cluster count / noise share / persistence
  run     one min_cluster_size: save clusters (+ turn counts, persistence, core examples),
          name them with the model, and check reappearance across UMAP seeds
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import hdbscan
import numpy as np
import pandas as pd
import umap
from scipy.optimize import linear_sum_assignment


def reduce(X: np.ndarray, seed: int, dims: int, neighbors: int) -> np.ndarray:
    return umap.UMAP(n_components=dims, n_neighbors=neighbors, min_dist=0.0, metric="cosine",
                     random_state=seed).fit_transform(X)


SELECTION = "eom"


def cluster(Z: np.ndarray, mcs: int, min_samples: int | None):
    return hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=min_samples, metric="euclidean",
                           cluster_selection_method=SELECTION).fit(Z)


def summarize(labels: np.ndarray, persistence: np.ndarray) -> str:
    n = labels.max() + 1
    sizes = np.bincount(labels[labels >= 0]) if n > 0 else np.array([])
    noise = (labels < 0).mean()
    return (f"clusters={n:3d}  noise={noise:5.1%}  size: median={int(np.median(sizes)) if n else 0:4d} "
            f"max={int(sizes.max()) if n else 0:4d}  persistence: median={np.median(persistence) if n else 0:.2f} "
            f"max={persistence.max() if n else 0:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["sweep", "run"])
    parser.add_argument("--emb-dir", type=Path, default=Path("outputs/discovery/method1/cluster_statements"))
    parser.add_argument("--positives", type=Path, default=Path("outputs/discovery/method1/statements_flat.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/discovery/method1/cluster_hdbscan"))
    parser.add_argument("--dims", type=int, default=10)
    parser.add_argument("--neighbors", type=int, default=30)
    parser.add_argument("--mcs", type=int, nargs="+", default=[30, 50, 100])
    parser.add_argument("--min-samples", type=int, default=None)
    parser.add_argument("--selection", choices=["eom", "leaf"], default="eom")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stability-seeds", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    global SELECTION
    SELECTION = args.selection

    X = np.load(args.emb_dir / "embeddings.npy")
    rows = [json.loads(line) for line in args.positives.open()]
    turns = np.array([f"{r['session_id']}:{r['turn_number']}" for r in rows])

    cache = args.out_dir / f"umap_d{args.dims}_n{args.neighbors}_s{args.seed}.npy"
    if cache.exists():
        Z = np.load(cache)
    else:
        Z = reduce(X, args.seed, args.dims, args.neighbors)
        np.save(cache, Z)

    if args.stage == "sweep":
        for mcs in args.mcs:
            model = cluster(Z, mcs, args.min_samples)
            print(f"min_cluster_size={mcs:4d}  {summarize(model.labels_, model.cluster_persistence_)}")
        return

    mcs = args.mcs[0]
    model = cluster(Z, mcs, args.min_samples)
    labels, probs, pers = model.labels_, model.probabilities_, model.cluster_persistence_
    print(f"min_cluster_size={mcs}  {summarize(labels, pers)}")

    # reappearance across UMAP seeds: match by centroid cosine in the ORIGINAL embedding space
    n = labels.max() + 1
    def centroids(L: np.ndarray, k: int) -> np.ndarray:
        C = np.vstack([X[L == c].mean(axis=0) for c in range(k)])
        return C / np.linalg.norm(C, axis=1, keepdims=True)
    C0 = centroids(labels, n)
    jacc = {}
    for s in args.stability_seeds:
        Zs = reduce(X, s, args.dims, args.neighbors)
        Ls = cluster(Zs, mcs, args.min_samples).labels_
        ns = Ls.max() + 1
        sim = C0 @ centroids(Ls, ns).T
        r, c = linear_sum_assignment(-sim)
        match = dict(zip(r, c))
        jacc[s] = []
        for k in range(n):
            a = set(np.where(labels == k)[0])
            b = set(np.where(Ls == match[k])[0]) if k in match else set()
            jacc[s].append(len(a & b) / len(a | b) if a | b else 0.0)
        print(f"  seed {s}: {ns} clusters, noise {(Ls < 0).mean():.1%}, mean Jaccard to seed-{args.seed} clusters {np.mean(jacc[s]):.2f}")

    from cluster_messages import _name_cluster, _row_view  # same folder
    import concurrent.futures as cf
    import cluster_messages
    cluster_messages.KIND = "statements"

    clusters = []
    for k in range(n):
        idx = np.where(labels == k)[0]
        core = idx[np.argsort(-probs[idx])[:20]]
        clusters.append({
            "cluster": k, "size": int(len(idx)), "n_turns": int(len(set(turns[idx]))),
            "persistence": round(float(pers[k]), 3),
            "min_jaccard": round(float(min(jacc[s][k] for s in args.stability_seeds)), 2),
            "nearest": [_row_view(rows[i]) for i in core],
            "random": [_row_view(rows[i]) for i in np.random.default_rng(0).choice(idx, size=min(5, len(idx)), replace=False)],
        })
    with cf.ThreadPoolExecutor(args.workers) as pool:
        names = list(pool.map(_name_cluster, clusters))
    for c, name in zip(clusters, names):
        c["name"] = name
    clusters.sort(key=lambda c: -c["n_turns"])

    tag = f"mcs{mcs}_{args.selection}" + (f"_ms{args.min_samples}" if args.min_samples else "")
    (args.out_dir / f"clusters_{tag}.json").write_text(json.dumps(
        {"min_cluster_size": mcs, "dims": args.dims, "neighbors": args.neighbors, "seed": args.seed,
         "noise_share": float((labels < 0).mean()), "assignments": labels.tolist(), "clusters": clusters}, ensure_ascii=False))
    lines = [f"# HDBSCAN on preference statements (UMAP {args.dims}d, min_cluster_size={mcs})", "",
             f"{len(rows)} statements; {n} clusters; {(labels < 0).mean():.1%} noise. "
             f"persistence = HDBSCAN cluster stability; J = min Jaccard with matched cluster under other UMAP seeds "
             f"{args.stability_seeds}. Sorted by number of distinct turns.", ""]
    for c in clusters:
        lines += [f"## cluster {c['cluster']} — {c['n_turns']} turns, {c['size']} stmts, persistence {c['persistence']}, J={c['min_jaccard']}", "",
                  f"**{c['name']}**", "", "Core (highest membership probability):", ""]
        lines += [f"- {m['text'][:300]}" for m in c["nearest"][:10]]
        lines += ["", "Random:", ""] + [f"- {m['text'][:300]}" for m in c["random"]] + [""]
    (args.out_dir / f"clusters_{tag}.md").write_text("\n".join(lines))
    print(f"\n-> {args.out_dir / f'clusters_{tag}.md'}\n")
    for c in clusters:
        print(f"[{c['cluster']:3d}] {c['n_turns']:4d} turns  pers={c['persistence']:.2f}  J={c['min_jaccard']:.2f}  {c['name'][:110]}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
