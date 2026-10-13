#!/usr/bin/env python3
"""Candidate pairs of preference messages for the LLM judge (no model calls).

Unit: a preference message (gate positive, exact duplicates removed). Two similarity scores per pair:
  raw   cosine of the two raw-message embeddings
  mix   (raw + stmt) / 2, stmt = the highest cosine between the two messages' rewritten statements
        (messages the rewrite left without a statement fall back to raw)
Templates: a message whose raw embedding is >= TEMPLATE_COS to a message of another user, or whose first
PREFIX_WORDS words are shared with another user, is treated as harness- or tool-generated boilerplate
(e.g. "Implement the plan as specified ...", "Fix only the selected review findings ...") and dropped.
Repeats within one user are kept (that is the behaviour being measured), except near-identical re-sends
(raw cosine >= RESEND_COS), which are left out.
Candidates: for each message B,
  own      the K_OWN earlier messages of the same user with the highest score, taken separately for raw and
           mix and merged (so the two scores' recall can be compared afterwards). Users whose commit identity
           is a bot or agent, and messages without a user, get no own candidates.
  others   the K_OTHER messages of other users with the highest score (raw and mix, merged): prevalence.
Level of an own pair: same session > same repo > other repo (same agent) > other agent.
Output: outputs/discovery/scope/candidates.jsonl, templates.json
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
O = ROOT / "outputs/discovery/scope"
NON_HUMAN = re.compile(r"\[bot\]$|^(claude|copilot|agent@.*|claude@.*)$", re.I)
TEMPLATE_COS = 0.95
RESEND_COS = 0.95          # same user, raw cosine >= this: a re-sent message, not a recurring preference
PREFIX_WORDS = 12          # templates with a fixed opening and a variable tail (e.g. tool-generated review prompts)


def load(raw_only=False):
    msg = pd.DataFrame([json.loads(l) for l in open(O / "gate/positives_dedup.jsonl")])
    Er = np.load(O / "emb_raw/embeddings.npy").astype(np.float32)
    if raw_only:                                      # before the rewrite is finished: mix = raw
        st = pd.DataFrame(columns=["session_id", "turn_number"])
        Es = np.zeros((0, Er.shape[1]), np.float32)
    else:
        st = pd.DataFrame([json.loads(l) for l in open(O / "statements_flat.jsonl")])
        Es = np.load(O / "emb_statements/embeddings.npy").astype(np.float32)
    assert len(msg) == len(Er) and len(st) == len(Es)
    key = {(s, int(n)): i for i, (s, n) in enumerate(zip(msg.session_id, msg.turn_number))}
    st["m"] = [key.get((s, int(n)), -1) for s, n in zip(st.session_id, st.turn_number)] if len(st) else []
    msg["t"] = pd.to_datetime(msg.time, utc=True, errors="coerce")
    msg["order"] = msg.t.astype("int64") + msg.turn_number          # ties inside a session broken by turn
    return msg, Er, st, Es


def templates(msg, Er):
    uid = msg.user_id.fillna("?").astype(str).to_numpy()
    tpl = np.zeros(len(msg), bool)
    prefix = msg.content.map(lambda t: " ".join(str(t).lower().split()[:PREFIX_WORDS]))
    long_enough = msg.content.map(lambda t: len(str(t).split()) >= 20).to_numpy()     # short messages may share openings by chance
    users_per_prefix = pd.Series(uid).groupby(prefix.to_numpy()).nunique()
    tpl |= long_enough & (prefix.map(users_per_prefix).to_numpy() >= 2)
    for c in range(0, len(msg), 2000):
        S = Er[c:c + 2000] @ Er.T
        other = uid[c:c + 2000, None] != uid[None, :]
        tpl[c:c + 2000] |= ((S >= TEMPLATE_COS) & other).any(1) & long_enough[c:c + 2000]
    return tpl


def stmt_max(ia, ib, st_of, Es):
    """Highest statement cosine between message ia and each message in ib (nan if a side has none)."""
    out = np.full(len(ib), np.nan, np.float32)
    sa = st_of.get(ia)
    if sa is None:
        return out
    for k, j in enumerate(ib):
        sb = st_of.get(j)
        if sb is not None:
            out[k] = float((Es[sa] @ Es[sb].T).max())
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k-own", type=int, default=5)
    p.add_argument("--k-other", type=int, default=3)
    p.add_argument("--raw-only", action="store_true", help="raw scores only (rewrite not finished yet)")
    args = p.parse_args()
    msg, Er, st, Es = load(args.raw_only)
    tpl = templates(msg, Er)
    json.dump(dict(n_templates=int(tpl.sum()), examples=msg.content[tpl].str[:200].head(30).tolist()),
              open(O / "templates.json", "w"), ensure_ascii=False, indent=1)
    print("messages %d, templates dropped %d" % (len(msg), tpl.sum()), flush=True)
    keep = np.where(~tpl)[0]
    st_of = {m: g.index.to_numpy() for m, g in st[st.m >= 0].groupby("m")} if len(st) else {}
    uid = msg.user_id.astype(str).to_numpy()
    known = (msg.user_id.notna() & ~msg.user_id.fillna("").astype(str).str.match(NON_HUMAN)).to_numpy()
    has_user = msg.user_id.notna().to_numpy()
    rows = []
    by_user = pd.Series(keep).groupby(uid[keep]).apply(np.asarray).to_dict()
    S = Er @ Er.T                                         # all raw-message cosines (~20k x 20k float32, ~1.6 GB)
    order = msg.order.to_numpy()
    for n, b in enumerate(keep):
        if n % 2000 == 0:
            print("  %d/%d" % (n, len(keep)), flush=True)
        # own: earlier messages of the same user
        if known[b]:
            pool = by_user[uid[b]]
            pool = pool[order[pool] < order[b]]
            if len(pool):
                raw = S[b, pool]
                mix = np.where(np.isnan(s := stmt_max(b, pool, st_of, Es)), raw, (raw + s) / 2)
                ok = raw < RESEND_COS                                     # re-sends are not candidates
                pool, raw, mix = pool[ok], raw[ok], mix[ok]
                pick = set(pool[np.argsort(-raw)[:args.k_own]]) | set(pool[np.argsort(-mix)[:args.k_own]])
                for a in pick:
                    k = int(np.where(pool == a)[0][0])
                    A, B = msg.iloc[a], msg.iloc[b]
                    lvl = ("same_session" if A.session_id == B.session_id else "other_agent" if A.agent != B.agent
                           else "same_repo" if A.repo_id == B.repo_id else "other_repo")
                    rows.append(dict(kind="own", a=int(a), b=int(b), level=lvl, raw=round(float(raw[k]), 4),
                                     mix=round(float(mix[k]), 4), rank_raw=int((raw > raw[k]).sum()),
                                     rank_mix=int((mix > mix[k]).sum()), pool=len(pool)))
        # others: prevalence among other users
        if has_user[b]:
            pool = keep[(uid[keep] != uid[b]) & has_user[keep]]
            raw = S[b, pool]
            top = pool[np.argpartition(-raw, 50)[:50]] if len(pool) > 50 else pool   # statement scores only for the raw top 50
            r50 = S[b, top]
            top, r50 = top[np.argsort(-r50)], np.sort(r50)[::-1]
            s50 = stmt_max(b, top, st_of, Es)
            m50 = np.where(np.isnan(s50), r50, (r50 + s50) / 2)
            pick = set(top[:args.k_other]) | set(top[np.argsort(-m50)[:args.k_other]])
            for a in pick:
                k = int(np.where(top == a)[0][0])
                rows.append(dict(kind="other", a=int(a), b=int(b), level="other_user", raw=round(float(r50[k]), 4),
                                 mix=round(float(m50[k]), 4), pool=len(pool)))
    with open(O / ("candidates_raw.jsonl" if args.raw_only else "candidates.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    d = pd.DataFrame(rows)
    print("candidates %d" % len(d))
    print(d.groupby(["kind", "level"]).size().to_string())


if __name__ == "__main__":
    main()
