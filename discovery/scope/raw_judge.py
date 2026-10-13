#!/usr/bin/env python3
"""Re-judge the strict_probe pairs on the ORIGINAL messages (no rewritten statements), then ask which
embedding (statement vs raw message) predicts the raw-text judgment better. Messages are cut to their
first 200 words. Output: outputs/discovery/scope/strict_probe/pairs_raw.jsonl"""
import concurrent.futures as cf
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import strict_probe as SP  # noqa: E402
from compare_raw_vs_statement import auc  # noqa: E402

OUT = SP.ROOT / "outputs/discovery/scope"
PROMPT = """Two messages a developer sent to an AI coding agent, in two different moments. Message A came first.
Suppose an agent remembered whatever general preference message A shows about how the agent should work or
how code should be written, and is now in the situation of message B. Decide:

same: B asks for something an agent following A's preference would already have done; B's preference is A's preference.
related: B's preference is on the same topic or in the same direction, but following A would not by itself satisfy B.
conflict: following A's preference would go against what B asks.
unrelated: B expresses no preference that A speaks to (including when either message shows no general preference).

Message A:
{a}

Message B:
{b}

Answer with JSON: {{"label": "same" | "related" | "conflict" | "unrelated", "why": "<at most 15 words>"}}"""


def cut(t, n=200):
    w = str(t).split()
    return " ".join(w[:n]) + (" ..." if len(w) > n else "")


def judge(a, b, model="openai/wine-claude-sonnet-4-6"):
    import litellm
    err = ""
    for attempt in range(4):
        try:
            r = litellm.completion(api_key=os.environ["LITELLM_API_KEY"], base_url=SP.API_BASE, model=model,
                                   messages=[{"role": "user", "content": PROMPT.format(a=cut(a), b=cut(b))}],
                                   max_tokens=80, temperature=0)
            t = r.choices[0].message.content
            return json.loads(t[t.find("{"):t.rfind("}") + 1])
        except Exception as e:  # noqa: BLE001
            err = repr(e)[:200]
            time.sleep(2 ** attempt)
    return {"label": None, "why": err}


def main():
    st = SP.load()
    ids = json.loads((OUT / "old_raw_emb/ids.json").read_text())
    Er = np.load(OUT / "old_raw_emb/embeddings.npy")
    raw_ix = {(d["session_id"], int(d["turn_number"])): i for i, d in enumerate(ids)}
    text = {}
    for l in open(SP.M1.parent / "gate/positives_dedup.jsonl"):
        r = json.loads(l)
        text[(r["session_id"], int(r["turn_number"]))] = r["content"]
    pairs = [json.loads(l) for l in open(OUT / "strict_probe/pairs.jsonl")]
    jobs = []
    for p in pairs:
        ka = (st.session_id.iat[p["a"]], int(st.turn_number.iat[p["a"]]))
        kb = (st.session_id.iat[p["b"]], int(st.turn_number.iat[p["b"]]))
        if ka != kb and ka in text and kb in text and p["label"]:
            jobs.append(dict(p, raw_a=text[ka], raw_b=text[kb], raw_sim=float(Er[raw_ix[ka]] @ Er[raw_ix[kb]])))
    with cf.ThreadPoolExecutor(8) as ex:
        res = list(ex.map(lambda j: judge(j["raw_a"], j["raw_b"]), jobs))
    with open(OUT / "strict_probe/pairs_raw.jsonl", "w") as f:
        for j, r in zip(jobs, res):
            f.write(json.dumps(dict(j, raw_label=r.get("label"), raw_why=r.get("why")), ensure_ascii=False) + "\n")
    J = [dict(j, raw_label=r.get("label")) for j, r in zip(jobs, res) if r.get("label")]
    from collections import Counter
    print("pairs judged on raw text: %d; labels %s" % (len(J), dict(Counter(j["raw_label"] for j in J))))
    print("agreement with the statement-level label: %.0f%%" % (100 * np.mean([j["raw_label"] == j["label"] for j in J])))
    cm = Counter((j["label"], j["raw_label"]) for j in J)
    print("statement label -> raw label (same row):", {k[1]: v for k, v in cm.items() if k[0] == "same"})
    for name, key in (("statement cos", "sim"), ("raw-message cos", "raw_sim")):
        same = [j[key] for j in J if j["raw_label"] == "same"]
        rest = [j[key] for j in J if j["raw_label"] != "same"]
        conf = [j[key] for j in J if j["raw_label"] == "conflict"]
        print("  %-16s AUC raw-same vs rest %.3f | vs conflict %.3f" % (name, auc(same, rest), auc(same, conf)))


if __name__ == "__main__":
    main()
