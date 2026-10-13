#!/usr/bin/env python3
"""Evaluate a judge prompt (sonnet-4-6) on the reviewed gold pairs: agreement, P/R for same and conflict,
and the disagreements. Usage: prompt_iter.py <prompt file> [--model ...]. Messages cut at 800 words."""
import argparse
import concurrent.futures as cf
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import litellm

litellm.suppress_debug_info = True
ROOT = Path(__file__).resolve().parents[2]
O = ROOT / "outputs/discovery/scope"
API_BASE = os.getenv("LITELLM_API_BASE", "https://ai-gateway.andrew.cmu.edu/")


def cut(t, n=800):
    w = str(t).split()
    return " ".join(w[:n]) + (" ..." if len(w) > n else "")


def judge(prompt, a, b, model):
    err = ""
    for attempt in range(5):
        try:
            r = litellm.completion(api_key=os.environ["LITELLM_API_KEY"], base_url=API_BASE, model=model,
                                   messages=[{"role": "user", "content": prompt.format(a=cut(a), b=cut(b))}],
                                   max_tokens=120, temperature=0)
            t = r.choices[0].message.content
            return json.loads(t[t.find("{"):t.rfind("}") + 1])
        except Exception as e:  # noqa: BLE001
            err = repr(e)[:200]
            time.sleep(2 ** attempt)
    return {"label": None, "why": err}


def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    p = argparse.ArgumentParser()
    p.add_argument("prompt")
    p.add_argument("--model", default="openai/wine-claude-sonnet-4-6")
    p.add_argument("--gold", default=str(O / "prompt_iter/gold.json"))
    args = p.parse_args()
    prompt = open(args.prompt).read()
    msgs = [json.loads(l)["content"] for l in open(O / "gate/positives_dedup.jsonl")]
    gold = json.load(open(args.gold))
    with cf.ThreadPoolExecutor(8) as ex:
        res = list(ex.map(lambda g: judge(prompt, msgs[g["a"]], msgs[g["b"]], args.model), gold))
    out = []
    for g, r in zip(gold, res):
        out.append(dict(g, pred=r.get("label"), why=r.get("why")))
    tag = Path(args.prompt).stem
    json.dump(out, open(O / ("prompt_iter/result_%s.json" % tag), "w"), indent=0, ensure_ascii=False)
    ok = [o for o in out if o["pred"]]
    print("%s: n=%d agreement %.0f%%" % (tag, len(ok), 100 * sum(o["pred"] == o["gold"] for o in ok) / len(ok)))
    for lab in ("same", "conflict", "related", "unrelated"):
        tp = sum(o["pred"] == lab and o["gold"] == lab for o in ok)
        pp = sum(o["pred"] == lab for o in ok)
        gp = sum(o["gold"] == lab for o in ok)
        print("  %-9s precision %.2f (%d/%d)  recall %.2f (%d/%d)" % (lab, tp / max(pp, 1), tp, pp, tp / max(gp, 1), tp, gp))
    print("  confusion gold->pred:", dict(Counter((o["gold"], o["pred"]) for o in ok if o["gold"] != o["pred"])))
    for o in ok:
        if o["gold"] != o["pred"]:
            print("   x %s gold=%s pred=%s | A: %s | B: %s | %s" % (o["src"], o["gold"], o["pred"],
                  msgs[o["a"]][:90].replace("\n", " "), msgs[o["b"]][:90].replace("\n", " "), o["why"]))


if __name__ == "__main__":
    main()
