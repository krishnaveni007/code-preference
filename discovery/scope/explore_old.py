#!/usr/bin/env python3
"""Exploration on the old release (no model calls): which preferences recur, and does recurrence decay
with time? Co-occurrence as in pair_cooccurrence.py (cosine >= 0.88). Rubric = method1 cluster -> final
rubric of the EARLIER statement (descriptive only; 43% of statements are in no cluster -> 'unassigned')."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import strict_probe as SP  # noqa: E402
from pair_cooccurrence import NON_HUMAN  # noqa: E402

TAU = 0.88


def rubric_of_statement():
    V = SP.M1
    final = json.loads((V / "rubrics_final.json").read_text())["rubrics"]
    cl = json.loads((V / "hdbscan/clusters_mcs100_leaf_ms10.json").read_text())
    ordered = sorted(cl["clusters"], key=lambda c: -c["n_turns"])
    rank = {i + 1: c["cluster"] for i, c in enumerate(ordered)}
    c2r = {rank[r]: rb["name"] for rb in final for r in rb["clusters"]}
    return [c2r.get(int(c), "unassigned") for c in cl["assignments"]]


def main():
    st = SP.load()
    E = np.load(SP.M1 / "embeddings/embeddings.npy").astype(np.float32)
    st["rubric"] = rubric_of_statement()
    st["ts"] = st.t.astype("int64") / 1e9                 # seconds
    human = st.user_id.notna() & ~st.user_id.fillna("").str.match(NON_HUMAN)
    rows = []
    for _, g in st[human].groupby("user_id"):
        ix = g.index.to_numpy()
        if len(ix) < 2:
            continue
        iu = np.triu_indices(len(ix), 1)
        s = (E[ix] @ E[ix].T)[iu]
        a, b = g.iloc[iu[0]], g.iloc[iu[1]]
        first = np.where(a.ts.to_numpy() <= b.ts.to_numpy(), 0, 1)
        rub = np.where(first == 0, a.rubric.to_numpy(), b.rubric.to_numpy())
        gap = np.abs(b.ts.to_numpy() - a.ts.to_numpy()) / 86400
        rows.append(pd.DataFrame(dict(hit=s >= TAU, rubric=rub, gap_days=gap,
                                      same_session=a.session_id.to_numpy() == b.session_id.to_numpy(),
                                      same_repo=a.repo_id.to_numpy() == b.repo_id.to_numpy())))
    d = pd.concat(rows)
    pop = 0.00016
    print("1) co-occurrence by rubric of the earlier statement (cross-session pairs, same repo)")
    x = d[~d.same_session & d.same_repo]
    t = x.groupby("rubric").hit.agg(["mean", "sum", "size"])
    t["lift"] = t["mean"] / pop
    print(t.sort_values("mean", ascending=False).to_string(float_format=lambda v: "%.4f" % v if v < 1 else "%.0f" % v))
    print("\n2) decay with time between the two statements (cross-session pairs, same repo)")
    bins = [0, 1, 3, 7, 14, 30, 60, 365]
    x = x.assign(b=pd.cut(x.gap_days, bins, right=False))
    t = x.groupby("b", observed=True).hit.agg(["mean", "size"])
    t["lift"] = t["mean"] / pop
    print(t.to_string(float_format=lambda v: "%.4f" % v if v < 1 else "%.0f" % v))


if __name__ == "__main__":
    main()
