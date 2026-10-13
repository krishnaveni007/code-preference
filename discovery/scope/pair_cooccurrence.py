#!/usr/bin/env python3
"""Pair-level co-occurrence of preference statements by level (no model calls).

For every pair of statements of the same user, the level is the closest relation of the two messages:
same session > same repo (other session) > other repo with the same agent > other agent. The population
level is a random sample of cross-user pairs. Co-occurrence at a level = share of its pairs with
cosine >= tau (tau calibrated on judged pairs in strict_probe.py: at 0.88 no pair was a conflict and about
half were strictly the same preference; 0.85 / 0.90 / 0.92 as sensitivity). Each pair counts once, so the
rates need no correction for how many statements a level holds.

Two aggregates: pooled over all pairs (dominated by users with many statements, since pairs grow
quadratically) and per user (mean of each user's own rate, users with >= MIN_PAIRS pairs at the level).
Lift = rate / population rate. Users whose commit identity is a bot or agent are left out of the
same-user levels (their sessions belong to different people) but stay in the population sample.

--data old: method1 statements (old release); --data v2: outputs/discovery/scope (SWE-chat v2).
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import strict_probe as SP  # noqa: E402

ROOT = SP.ROOT
NON_HUMAN = re.compile(r"\[bot\]$|^(claude|copilot|agent@.*|claude@.*)$", re.I)
LEVELS = ["same_session", "same_repo", "other_repo", "other_agent"]
MIN_PAIRS = 10


def load(data: str) -> tuple[pd.DataFrame, np.ndarray]:
    if data == "old":
        return SP.load(), np.load(SP.M1 / "embeddings/embeddings.npy").astype(np.float32)
    O = ROOT / "outputs/discovery/scope"
    st = pd.DataFrame([json.loads(l) for l in open(O / "statements_flat.jsonl")])
    st["t"] = pd.to_datetime(st.time, utc=True, errors="coerce")
    st["session_start"] = st.groupby("session_id").t.transform("min")
    return st, np.load(O / "emb_statements/embeddings.npy").astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", choices=["old", "v2"], default="old")
    p.add_argument("--taus", type=float, nargs="+", default=[0.85, 0.88, 0.90, 0.92])
    p.add_argument("--population-pairs", type=int, default=3_000_000)
    args = p.parse_args()
    st, E = load(args.data)
    assert len(st) == len(E), (len(st), len(E))
    T = np.asarray(args.taus)[:, None]
    human = st.user_id.notna() & ~st.user_id.fillna("").astype(str).str.match(NON_HUMAN)

    pooled = {l: np.zeros(len(args.taus)) for l in LEVELS}
    n = {l: 0 for l in LEVELS}
    per_user = []
    for u, g in st[human].groupby("user_id"):
        ix = g.index.to_numpy()
        if len(ix) < 2:
            continue
        iu = np.triu_indices(len(ix), 1)
        s = (E[ix] @ E[ix].T)[iu]
        a, b = g.iloc[iu[0]], g.iloc[iu[1]]
        same_sess = a.session_id.to_numpy() == b.session_id.to_numpy()
        same_agent = a.agent.to_numpy() == b.agent.to_numpy()
        same_repo = a.repo_id.to_numpy() == b.repo_id.to_numpy()
        lvl = np.where(same_sess, 0, np.where(~same_agent, 3, np.where(same_repo, 1, 2)))
        for k, l in enumerate(LEVELS):
            m = lvl == k
            hits = (s[m][None, :] >= T).sum(1)
            pooled[l] += hits
            n[l] += int(m.sum())
            if m.sum() >= MIN_PAIRS:
                per_user.append(dict(user=u, level=l, pairs=int(m.sum()),
                                     **{"r%.2f" % t: hits[i] / m.sum() for i, t in enumerate(args.taus)}))

    rng = np.random.default_rng(0)
    uid = st.user_id.astype(str).to_numpy()
    known = st.user_id.notna().to_numpy()
    i = rng.integers(0, len(st), args.population_pairs)
    j = rng.integers(0, len(st), args.population_pairs)
    m = (uid[i] != uid[j]) & known[i] & known[j]
    i, j = i[m], j[m]
    hits = np.zeros(len(args.taus))
    for c in range(0, len(i), 100_000):                  # chunks: gathering millions of 3072-d pairs at once would not fit
        s = np.einsum("ij,ij->i", E[i[c:c + 100_000]], E[j[c:c + 100_000]])
        hits += (s[None, :] >= T).sum(1)
    pop = hits / len(i)

    pu = pd.DataFrame(per_user)
    rows = [dict(level="population", pairs=len(i), users=np.nan,
                 **{"pooled@%.2f" % t: pop[k] for k, t in enumerate(args.taus)},
                 **{"per_user@%.2f" % t: pop[k] for k, t in enumerate(args.taus)})]
    for l in LEVELS:
        x = pu[pu.level == l] if len(pu) else pu
        rows.append(dict(level=l, pairs=n[l], users=len(x),
                         **{"pooled@%.2f" % t: pooled[l][k] / max(n[l], 1) for k, t in enumerate(args.taus)},
                         **{"per_user@%.2f" % t: (x["r%.2f" % t].mean() if len(x) else np.nan) for t in args.taus}))
    df = pd.DataFrame(rows).set_index("level")
    for k, t in enumerate(args.taus):
        for agg in ("pooled", "per_user"):
            df["%s_lift@%.2f" % (agg, t)] = df["%s@%.2f" % (agg, t)] / pop[k]
    pd.set_option("display.width", 250)
    print("data %s: statements %d, users %d (same-user levels: %d human users)" % (
        args.data, len(st), st.user_id.nunique(), st[human].user_id.nunique()))
    show = ["pairs", "users", "pooled@0.88", "pooled_lift@0.88", "per_user@0.88", "per_user_lift@0.88"] + \
           ["per_user_lift@%.2f" % t for t in args.taus if t != 0.88]
    print(df[show].to_string(float_format=lambda x: "%.5f" % x if x < 1 else "%.1f" % x))
    out = ROOT / ("outputs/discovery/scope/pair_cooccurrence_%s.csv" % args.data)
    df.to_csv(out)
    pu.to_csv(out.with_name(out.stem + "_per_user.csv"), index=False)


if __name__ == "__main__":
    main()
