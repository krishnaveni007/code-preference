#!/usr/bin/env python3
"""All within-user pairs of preference messages with raw cosine >= FLOOR (and < RESEND_COS), in random
order so that any judged prefix is an unbiased sample. Templates and bot / agent identities are excluded as
in candidates.py. Output: candidates_floor<FLOOR>.jsonl (kind=own, level, raw)."""
import argparse
import json
import random

import numpy as np
import pandas as pd

import candidates as C


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--floor", type=float, default=0.70)
    args = p.parse_args()
    msg, Er, _, _ = C.load(raw_only=True)
    tpl = C.templates(msg, Er)
    known = (msg.user_id.notna() & ~msg.user_id.fillna("").astype(str).str.match(C.NON_HUMAN)).to_numpy() & ~tpl
    order = msg.order.to_numpy()
    rows = []
    for _, g in msg[known].groupby("user_id"):
        ix = g.index.to_numpy()
        if len(ix) < 2:
            continue
        iu = np.triu_indices(len(ix), 1)
        s = (Er[ix] @ Er[ix].T)[iu]
        m = (s >= args.floor) & (s < C.RESEND_COS)
        for i, j, v in zip(ix[iu[0][m]], ix[iu[1][m]], s[m]):
            a, b = (i, j) if order[i] < order[j] else (j, i)
            A, B = msg.iloc[a], msg.iloc[b]
            lvl = ("same_session" if A.session_id == B.session_id else "other_agent" if A.agent != B.agent
                   else "same_repo" if A.repo_id == B.repo_id else "other_repo")
            rows.append(dict(kind="own", a=int(a), b=int(b), level=lvl, raw=round(float(v), 4), mix=round(float(v), 4)))
    random.Random(0).shuffle(rows)
    out = C.O / ("candidates_floor%03d.jsonl" % round(args.floor * 100))
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("%d pairs -> %s" % (len(rows), out))
    print(pd.Series([r["level"] for r in rows]).value_counts().to_string())


if __name__ == "__main__":
    main()
