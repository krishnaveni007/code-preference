#!/usr/bin/env python3
"""Quick probe, before any clustering: how often has a user already stated (strictly) the same preference?

Data: method1's rewritten statements (old SWE-chat release, 9,150 statements) and their embeddings.
For a random sample of later statements B, take the most similar EARLIER statement A of the same user
at each level, and the most similar statement of any OTHER user (population baseline):
  same_session     earlier turn of the same session
  same_repo        earlier session, same repository
  other_repo       earlier session, different repository, same agent
  other_agent      earlier session with a different agent
  other_user       one random other user (with >= 20 statements): pool comparable to a user's own history
  any_other_user   all other users' statements: how common the preference is in the population
A model then labels each (A, B) pair:
  same       an agent that follows A would thereby do what B asks (strict co-occurrence)
  related    same direction / topic, but following A would not by itself satisfy B
  conflict   following A would violate B
  unrelated  otherwise
Note: the candidate pools differ in size across levels (larger pools make a close top-1 likelier); pool
sizes are recorded so rates can be normalised later. Output: outputs/discovery/scope/strict_probe/.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import litellm
import numpy as np
import pandas as pd
from dotenv import load_dotenv

litellm.suppress_debug_info = True
ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
API_BASE = os.getenv("LITELLM_API_BASE", "https://ai-gateway.andrew.cmu.edu/")
M1 = ROOT / "outputs/discovery/method1/final"
OUT = ROOT / "outputs/discovery/scope/strict_probe"
LEVELS = ["same_session", "same_repo", "other_repo", "other_agent", "other_user", "any_other_user"]

PROMPT = """Two preference statements were extracted from messages that developers sent to an AI coding agent.
A was said first. Decide how A relates to B, from the point of view of an agent that has learned A and is
now in the situation where B was said.

same: an agent that follows A would thereby do what B asks; B adds nothing an agent following A would miss.
related: same topic or direction, but following A would not by itself satisfy B (different object, timing, scope or specificity).
conflict: following A would go against what B asks.
unrelated: otherwise.

A: {a}
B: {b}

Answer with JSON: {{"label": "same" | "related" | "conflict" | "unrelated", "why": "<at most 15 words>"}}"""


def judge(a: str, b: str, model: str) -> dict:
    for attempt in range(4):
        try:
            r = litellm.completion(api_key=os.environ["LITELLM_API_KEY"], base_url=API_BASE, model=model,
                                   messages=[{"role": "user", "content": PROMPT.format(a=a, b=b)}],
                                   max_tokens=80, temperature=0)
            txt = r.choices[0].message.content
            return json.loads(txt[txt.find("{"):txt.rfind("}") + 1])
        except Exception as e:  # noqa: BLE001
            err = repr(e)[:200]
            time.sleep(2 ** attempt)
    return {"label": None, "why": err}


def load() -> pd.DataFrame:
    st = pd.DataFrame([json.loads(l) for l in open(M1 / "statements_flat.jsonl")])
    conv = pd.read_parquet(ROOT / "data/swechat_data/conversations.parquet",
                           columns=["session_id", "turn_number", "repo_id", "agent", "timestamp"],
                           filters=[("turn_type", "==", "user_prompt")])
    turn_t = conv.groupby(["session_id", "turn_number"]).timestamp.min()
    sess = conv.groupby("session_id").agg(repo_id=("repo_id", "first"), agent=("agent", "first"),
                                          session_start=("timestamp", "min"))
    st = st.join(sess, on="session_id")
    st["t"] = [turn_t.get((s, n)) for s, n in zip(st.session_id, st.turn_number)]
    st["t"] = pd.to_datetime(st.t, utc=True, errors="coerce").fillna(pd.to_datetime(st.session_start, utc=True))
    st["session_start"] = pd.to_datetime(st.session_start, utc=True)
    return st


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=400, help="sampled later statements B")
    p.add_argument("--model", default="openai/wine-claude-sonnet-4-6")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    st = load()
    E = np.load(M1 / "embeddings/embeddings.npy").astype(np.float32)
    by_user = defaultdict(list)
    for i, u in enumerate(st.user_id):
        if u:                                          # a few statements have no user id: never matched
            by_user[u].append(i)

    big_users = sorted(u for u, v in by_user.items() if len(v) >= 20)
    rng_pick = random.Random(args.seed + 1)

    def candidates(j: int) -> dict[str, tuple[int, float, int]]:
        b = st.iloc[j]
        pools = defaultdict(list)
        for i in by_user[b.user_id]:
            a = st.iloc[i]
            if a.session_id == b.session_id:
                if a.turn_number < b.turn_number:
                    pools["same_session"].append(i)
            elif a.session_start < b.session_start:
                lvl = "other_agent" if a.agent != b.agent else ("same_repo" if a.repo_id == b.repo_id else "other_repo")
                pools[lvl].append(i)
        out = {}
        for lvl, idx in pools.items():
            s = E[idx] @ E[j]
            k = int(np.argmax(s))
            out[lvl] = (idx[k], float(s[k]), len(idx))
        others = np.array([i for i in range(len(st)) if st.user_id.iat[i] and st.user_id.iat[i] != b.user_id])
        s = E[others] @ E[j]
        k = int(np.argmax(s))
        out["any_other_user"] = (int(others[k]), float(s[k]), len(others))
        u = rng_pick.choice([v for v in big_users if v != b.user_id])
        idx = by_user[u]
        s = E[idx] @ E[j]
        k = int(np.argmax(s))
        out["other_user"] = (idx[k], float(s[k]), len(idx))
        return out

    rng = random.Random(args.seed)
    eligible = [j for j in range(len(st)) if st.user_id.iat[j] and len(by_user[st.user_id.iat[j]]) >= 2]
    sample = rng.sample(eligible, min(args.n, len(eligible)))
    jobs = []
    for j in sample:
        for lvl, (i, sim, pool) in candidates(j).items():
            jobs.append(dict(b=j, a=i, level=lvl, sim=round(sim, 4), pool=pool, user=st.user_id.iat[j],
                             A=st.content.iat[i], B=st.content.iat[j]))
    print("sampled %d statements -> %d pairs %s" % (len(sample), len(jobs), dict(Counter(x["level"] for x in jobs))), flush=True)
    with cf.ThreadPoolExecutor(args.workers) as ex:
        res = list(ex.map(lambda x: judge(x["A"], x["B"], args.model), jobs))
    with open(OUT / "pairs.jsonl", "w") as f:
        for x, r in zip(jobs, res):
            f.write(json.dumps(dict(x, label=r.get("label"), why=r.get("why")), ensure_ascii=False) + "\n")
    df = pd.DataFrame([dict(x, label=r.get("label")) for x, r in zip(jobs, res)])
    tab = pd.crosstab(df.level, df.label, normalize="index").round(3).reindex(LEVELS)
    tab["n"] = df.level.value_counts().reindex(LEVELS)
    tab["median_pool"] = df.groupby("level").pool.median().reindex(LEVELS)
    print(tab.to_string())
    tab.to_csv(OUT / "summary_by_level.csv")


if __name__ == "__main__":
    main()
