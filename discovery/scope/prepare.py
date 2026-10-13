#!/usr/bin/env python3
"""Glue between method1's stages for the SWE-chat v2 run (no model calls).

positives   gate labels + conversations -> gate/positives.jsonl (label 1, with metadata)
            and gate/positives_dedup.jsonl (exact-duplicate texts removed, as in method1)
flatten     statements.jsonl (one row per message) -> statements_flat.jsonl (one row per statement,
            method1's format plus repo_id / agent / time for the scope analysis)
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/discovery/scope"
META = ["repo_id", "agent", "time", "kind"]


def positives() -> None:
    labels = {}
    for line in open(OUT / "gate/gate_labels.jsonl"):
        r = json.loads(line)
        if r.get("label") in (0, 1):
            labels[(r["session_id"], int(r["turn_number"]))] = r["label"]
    conv = pd.read_parquet(ROOT / "data/swechat_v2/conversations.parquet")
    lab = pd.Series([labels.get((s, int(n))) for s, n in zip(conv.session_id, conv.turn_number)], index=conv.index)
    rate = conv[lab.notna()].assign(pos=lab[lab.notna()] == 1).groupby("agent").pos.agg(["mean", "size"])
    print("positive rate by agent:\n" + rate[rate["size"] >= 100].round(3).to_string())
    conv = conv[lab == 1]
    rows = [dict(session_id=r.session_id, user_id=r.user_id, turn_number=int(r.turn_number), shipped_pushback=None,
                 word_count=int(r.word_count), content=r.content, **{k: getattr(r, k) for k in META})
            for r in conv.itertuples()]
    with open(OUT / "gate/positives.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    seen, n_dup = set(), 0
    with open(OUT / "gate/positives_dedup.jsonl", "w") as f:
        for r in rows:
            key = r["content"].strip()
            if key in seen:
                n_dup += 1
                continue
            seen.add(key)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("labelled %d, positive %d (%.1f%%), exact duplicates removed %d -> %d"
          % (len(labels), len(rows), 100 * len(rows) / max(len(labels), 1), n_dup, len(rows) - n_dup))


def flatten() -> None:
    pos = {(r["session_id"], int(r["turn_number"])): r for r in map(json.loads, open(OUT / "gate/positives_dedup.jsonl"))}
    n_msg = n_empty = n_err = 0
    with open(OUT / "statements_flat.jsonl", "w") as f:
        for line in open(OUT / "statements.jsonl"):
            r = json.loads(line)
            n_msg += 1
            if r.get("error"):
                n_err += 1
                continue
            if not r["statements"]:
                n_empty += 1
            meta = pos.get((r["session_id"], int(r["turn_number"])), {})
            for k, s in enumerate(r["statements"]):
                f.write(json.dumps(dict(session_id=r["session_id"], user_id=r["user_id"], turn_number=r["turn_number"],
                                        statement_index=k, shipped_pushback=None, word_count=r["word_count"], content=s,
                                        **{m: meta.get(m) for m in META}), ensure_ascii=False) + "\n")
    print("messages %d (errors %d, no transferable preference %d)" % (n_msg, n_err, n_empty))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=["positives", "flatten"])
    {"positives": positives, "flatten": flatten}[p.parse_args().step]()
