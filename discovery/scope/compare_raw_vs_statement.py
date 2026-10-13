#!/usr/bin/env python3
"""Exploration (old release, no model calls): embed the raw preference message vs the rewritten statement?

1. Which similarity predicts the judged relation better? Uses the (A, B) pairs labelled by strict_probe.py
   (statement level); the raw similarity is that of the two source messages. AUC for same-vs-rest and
   same-vs-conflict.
2. Project-vocabulary bias: mean similarity of random pairs at each level (same session / same repo /
   other repo / other agent / other user), raw vs statement. Raw messages share file and project names,
   which should inflate within-repo similarity independent of preferences.
"""
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import strict_probe as SP  # noqa: E402

ROOT = SP.ROOT
OUT = ROOT / "outputs/discovery/scope"


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if not len(pos) or not len(neg):
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def main():
    st = SP.load()
    Es = np.load(SP.M1 / "embeddings/embeddings.npy").astype(np.float32)
    ids = json.loads((OUT / "old_raw_emb/ids.json").read_text())
    Er = np.load(OUT / "old_raw_emb/embeddings.npy").astype(np.float32)
    raw_ix = {(d["session_id"], int(d["turn_number"])): i for i, d in enumerate(ids)}
    st["raw"] = [raw_ix.get((s, int(n))) for s, n in zip(st.session_id, st.turn_number)]
    print("statements with a raw message: %d / %d" % (st.raw.notna().sum(), len(st)))

    pairs = [json.loads(l) for l in open(OUT / "strict_probe/pairs.jsonl")]
    rows = []
    for p in pairs:
        ra, rb = st.raw.iat[p["a"]], st.raw.iat[p["b"]]
        if p["label"] and ra is not None and rb is not None and not np.isnan(ra) and not np.isnan(rb) and ra != rb:
            rows.append((p["label"], p["sim"], float(Er[int(ra)] @ Er[int(rb)])))
    print("\n1. judged pairs usable: %d" % len(rows))
    for name, k in (("statement", 1), ("raw message", 2)):
        same = [r[k] for r in rows if r[0] == "same"]
        rest = [r[k] for r in rows if r[0] != "same"]
        conf = [r[k] for r in rows if r[0] == "conflict"]
        print("  %-12s AUC same-vs-rest %.3f | same-vs-conflict %.3f" % (name, auc(same, rest), auc(same, conf)))

    print("\n2. mean similarity of random pairs by level (20k pairs per level, users known)")
    rng = random.Random(0)
    users = st[st.user_id.notna() & st.raw.notna()].reset_index()
    by_user = users.groupby("user_id").indices
    levels = {k: [] for k in ("same_session", "same_repo", "other_repo", "other_agent", "other_user")}
    U = list(by_user)
    tries = 0
    while min(len(v) for v in levels.values()) < 20000 and tries < 3_000_000:
        tries += 1
        u = rng.choice(U)
        ix = by_user[u]
        if len(ix) < 2:
            continue
        i, j = rng.sample(list(ix), 2)
        a, b = users.iloc[i], users.iloc[j]
        if a.session_id == b.session_id:
            lvl = "same_session"
        elif a.agent != b.agent:
            lvl = "other_agent"
        else:
            lvl = "same_repo" if a.repo_id == b.repo_id else "other_repo"
        if len(levels[lvl]) < 20000:
            levels[lvl].append((a["index"], b["index"], int(a.raw), int(b.raw)))
        v = rng.choice(U)
        if v != u and len(levels["other_user"]) < 20000:
            c = users.iloc[rng.choice(list(by_user[v]))]
            levels["other_user"].append((a["index"], c["index"], int(a.raw), int(c.raw)))
    base_s = base_r = None
    print("  %-13s %7s %10s %8s %10s %8s" % ("level", "pairs", "stmt mean", "lift", "raw mean", "lift"))
    for lvl in ("other_user", "same_session", "same_repo", "other_repo", "other_agent"):
        v = levels[lvl]
        if not v:
            continue
        s = np.mean([Es[a] @ Es[b] for a, b, _, _ in v])
        r = np.mean([Er[ra] @ Er[rb] for _, _, ra, rb in v if ra != rb])
        if lvl == "other_user":
            base_s, base_r = s, r
        print("  %-13s %7d %10.3f %8.3f %10.3f %8.3f" % (lvl, len(v), s, s - base_s, r, r - base_r))


if __name__ == "__main__":
    main()
