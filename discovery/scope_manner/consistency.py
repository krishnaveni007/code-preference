"""Step 5: per-matter consistency. Pairs of statements on the same matter, split into same-user/different-session
and different-user pairs; share with the same stance and with opposite stances; mean over users with bootstrap CI.

Usage: python consistency.py [--inp restated] [--matters matters_final] [--min-users 5]
Output: data/consistency.json and a table on stdout.
"""
import argparse, collections, itertools, json
import numpy as np
import common as C

ap = argparse.ArgumentParser(); ap.add_argument("--inp", default="restated"); ap.add_argument("--matters", default="matters_final")
ap.add_argument("--min-users", type=int, default=5); a = ap.parse_args()
msgs = C.load_messages(); R = C.load_records(C.DATA / f"{a.inp}.jsonl")
S = [o for o in (json.loads(l) for l in open(C.DATA / f"{a.matters}.jsonl")) if "error" not in o and o.get("one_matter")]
rng = np.random.default_rng(0)


def stances(o):
    st = {}
    for g in o.get("stances", []):
        if g.get("label") == "off-matter": continue
        for i in g.get("members", []):
            try:
                i = int(i)
                if i in R: st[i] = g["label"]
            except (TypeError, ValueError): pass
    opp = {frozenset(p) for p in o.get("opposite", []) if isinstance(p, list) and len(p) == 2}
    return st, opp


def agree(pairs):
    byu = collections.defaultdict(list)
    for u, s in pairs: byu[u].append(s)
    if not byu: return (np.nan, np.nan, np.nan, 0)
    vals = np.array([np.mean(v) for v in byu.values()])
    bs = [vals[rng.integers(0, len(vals), len(vals))].mean() for _ in range(300)]
    return (float(vals.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), len(vals))


rows = []
for o in S:
    st, opp = stances(o); within = []; between = []; within_o = []; between_o = []
    for x, y in itertools.combinations(list(st), 2):
        A, B = msgs[x], msgs[y]; same = st[x] == st[y]; io = frozenset([st[x], st[y]]) in opp
        if A["user_id"] == B["user_id"]:
            if A["session_id"] != B["session_id"]: within.append((A["user_id"], same)); within_o.append((A["user_id"], io))
        else:
            k = tuple(sorted([A["user_id"], B["user_id"]])); between.append((k, same)); between_o.append((k, io))
    w, b, wo, bo = agree(within), agree(between), agree(within_o), agree(between_o)
    sess = collections.defaultdict(set)
    for i in st: sess[msgs[i]["user_id"]].add(msgs[i]["session_id"])
    rows.append(dict(matter=o.get("matter", ""), cluster=o["cluster"], n=len(st), users=len(sess),
                     users_multi_session=sum(1 for v in sess.values() if len(v) >= 2),
                     within_same=w[0], within_lo=w[1], within_hi=w[2], between_same=b[0], between_lo=b[1], between_hi=b[2],
                     within_opp=wo[0], between_opp=bo[0], n_within_pairs=len(within), n_between_pairs=len(between)))
json.dump(rows, open(C.DATA / "consistency.json", "w"), indent=1, ensure_ascii=False)
sel = sorted([r for r in rows if r["users_multi_session"] >= a.min_users], key=lambda r: -r["users_multi_session"])
print(f"matters {len(rows)}; with >={a.min_users} users having multiple sessions: {len(sel)}")
print(f"{'matter':45s} {'n':>4s} {'users':>5s} {'uMS':>4s} | within-user same [CI]   between-user same [CI] | opp within / between")
for r in sel:
    print(f"{r['matter'][:45]:45s} {r['n']:4d} {r['users']:5d} {r['users_multi_session']:4d} | {r['within_same']:.2f} [{r['within_lo']:.2f},{r['within_hi']:.2f}]   {r['between_same']:.2f} [{r['between_lo']:.2f},{r['between_hi']:.2f}] | {r['within_opp']:.2f} / {r['between_opp']:.2f}")
