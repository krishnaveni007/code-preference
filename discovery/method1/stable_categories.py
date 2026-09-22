#!/usr/bin/env python3
"""Report the high-frequency preference categories: clusters that are stable across seeds.

Coarse level = stable clusters at k=20. Fine level = stable clusters at k=40, nested under the
coarse cluster that holds >= --nest of their members; the rest are listed as standalone.
Every row carries real counts (statements, users), the min Jaccard across seeds, and the
statements nearest the centroid as examples.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load(out_dir: Path, k: int):
    data = json.loads((out_dir / f"clusters_k{k}.json").read_text())
    stab = pd.read_csv(out_dir / f"stability_k{k}.csv").set_index("cluster")
    return np.asarray(data["assignments"]), {c["cluster"]: c for c in data["clusters"]}, stab


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/discovery/method1/cluster_statements"))
    parser.add_argument("--positives", type=Path, default=Path("outputs/discovery/method1/statements_flat.jsonl"))
    parser.add_argument("--coarse", type=int, default=20)
    parser.add_argument("--fine", type=int, default=40)
    parser.add_argument("--nest", type=float, default=0.6)
    parser.add_argument("--examples", type=int, default=5)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.positives.open()]
    users = np.array([r["user_id"] for r in rows])
    Lc, Cc, Sc = load(args.out_dir, args.coarse)
    Lf, Cf, Sf = load(args.out_dir, args.fine)
    stable_c = [c for c in Cc if Sc.loc[c, "stable"]]
    stable_f = [c for c in Cf if Sf.loc[c, "stable"]]

    def row(level: int, c: int, L: np.ndarray, C: dict, S: pd.DataFrame) -> dict:
        idx = np.where(L == c)[0]
        return {"level": f"k{level}", "cluster": c, "size": int(len(idx)), "n_users": int(len(set(users[idx]))),
                "min_jaccard": float(S.loc[c, "min_jaccard"]), "name": C[c]["name"],
                "examples": [m["text"] for m in C[c]["nearest"][:args.examples]]}

    # nest fine clusters under the coarse cluster holding >= nest share of their members
    nested: dict[int, list[int]] = {c: [] for c in stable_c}
    standalone: list[int] = []
    for f in stable_f:
        idx = np.where(Lf == f)[0]
        parent, share = pd.Series(Lc[idx]).value_counts(normalize=True).agg(["idxmax", "max"])
        if share >= args.nest and int(parent) in nested:
            nested[int(parent)].append(f)
        else:
            standalone.append(f)

    report = []
    for c in sorted(stable_c, key=lambda c: -len(set(users[Lc == c]))):
        entry = row(args.coarse, c, Lc, Cc, Sc)
        entry["children"] = [row(args.fine, f, Lf, Cf, Sf) for f in nested[c]]
        report.append(entry)
    extra = [row(args.fine, f, Lf, Cf, Sf) for f in standalone]
    (args.out_dir / "stable_categories.json").write_text(json.dumps({"coarse": report, "fine_standalone": extra}, ensure_ascii=False, indent=1))

    lines = ["# High-frequency preference categories (stable across seeds)", "",
             f"Coarse = stable k={args.coarse} clusters ({len(stable_c)}/{args.coarse}); fine = stable k={args.fine} clusters "
             f"({len(stable_f)}/{args.fine}), nested where >= {int(args.nest*100)}% of members fall in one coarse cluster. "
             f"Stable = Jaccard >= 0.5 with the matched cluster under every other seed. Counts are real; names are model-written.", ""]
    for i, e in enumerate(report, 1):
        lines += [f"## {i}. [{e['n_users']} users, {e['size']} stmts, J={e['min_jaccard']:.2f}] {e['name']}", ""]
        lines += [f"- {x}" for x in e["examples"]]
        for ch in e["children"]:
            lines += ["", f"  ### sub: [{ch['n_users']} users, {ch['size']} stmts, J={ch['min_jaccard']:.2f}] {ch['name']}", ""]
            lines += [f"  - {x}" for x in ch["examples"]]
        lines.append("")
    if extra:
        lines += ["## Stable fine clusters not nested in any stable coarse cluster", ""]
        for e in extra:
            lines += [f"### [{e['n_users']} users, {e['size']} stmts, J={e['min_jaccard']:.2f}] {e['name']}", ""]
            lines += [f"- {x}" for x in e["examples"]] + [""]
    (args.out_dir / "stable_categories.md").write_text("\n".join(lines))

    print(f"{len(stable_c)} coarse categories, {sum(len(v) for v in nested.values())} nested fine, {len(extra)} standalone fine")
    print(f"-> {args.out_dir / 'stable_categories.md'}\n")
    for i, e in enumerate(report, 1):
        print(f"{i:2d}. [{e['n_users']:3d} users, {e['size']:4d} stmts, J={e['min_jaccard']:.2f}] {e['name'][:110]}")
        for ch in e["children"]:
            print(f"      - [{ch['n_users']:3d} users, {ch['size']:4d}, J={ch['min_jaccard']:.2f}] {ch['name'][:100]}")
    if extra:
        print("\nstandalone fine:")
        for e in extra:
            print(f"    * [{e['n_users']:3d} users, {e['size']:4d}, J={e['min_jaccard']:.2f}] {e['name'][:100]}")


if __name__ == "__main__":
    main()
