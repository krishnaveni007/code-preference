#!/usr/bin/env python3
"""LLM judge on raw messages for candidate pairs (candidates.py). Resumable (pairs already judged are
skipped), N threads. Labels (agent that learned A's preference, now in B's situation):
  same / related / conflict / unrelated   (prompt = judge_prompt_v5.txt, first 800 words of each message;
  chosen on 86 reviewed + 40 held-out pairs: same P ~0.82-0.86 R ~0.85, conflict P ~0.92 R ~0.85)
Daily-quota 429s ("Too many tokens per day") pause the run and probe every 10 min.
--sample N: a stratified sample (by kind/level and score decile) instead of all candidates, e.g. for the
sonnet reference set. Output: judgments_<tag>.jsonl
"""
import argparse
import concurrent.futures as cf
import json
import os
import random
import sys
import time
from pathlib import Path

import litellm
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
PROMPT = (Path(__file__).parent / "judge_prompt_v5.txt").read_text()   # chosen after prompt_iter.py (v1-v5)


def cut(t, n=800):
    w = str(t).split()
    return " ".join(w[:n]) + (" ..." if len(w) > n else "")

litellm.suppress_debug_info = True
ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
O = ROOT / "outputs/discovery/scope"
API_BASE = os.getenv("LITELLM_API_BASE", "https://ai-gateway.andrew.cmu.edu/")


def call(a, b, model):
    """-> (label, why, error); waits out daily-quota 429s."""
    err = ""
    for attempt in range(6):
        try:
            kw = {} if "gpt-5" in model else {"temperature": 0}   # gpt-5 reasoning models take no temperature
            r = litellm.completion(api_key=os.environ["LITELLM_API_KEY"], base_url=API_BASE, model=model,
                                   messages=[{"role": "user", "content": PROMPT.format(a=cut(a), b=cut(b))}],
                                   max_tokens=2000, timeout=180, **kw)   # room for thinking models
            t = r.choices[0].message.content
            j = json.loads(t[t.find("{"):t.rfind("}") + 1])
            return j.get("label"), j.get("why"), ""
        except Exception as e:  # noqa: BLE001
            err = repr(e)[:300]
            if "per day" in err:
                print(time.strftime("%H:%M"), "daily quota hit; sleeping 10 min", flush=True)
                time.sleep(600)
                continue
            time.sleep(2 ** attempt)
    return None, None, err


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="openai/wine-claude-sonnet-4-6")
    p.add_argument("--tag", default="sonnet46_v5")
    p.add_argument("--sample", type=int, default=0)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--candidates", default="candidates.jsonl")
    args = p.parse_args()
    msgs = [json.loads(l)["content"] for l in open(O / "gate/positives_dedup.jsonl")]
    cand = [json.loads(l) for l in open(O / args.candidates)]
    if args.sample:
        rng = random.Random(0)
        strata = {}
        for c in cand:
            strata.setdefault((c["kind"], c["level"], min(int(c["mix"] * 20), 19)), []).append(c)
        per = max(1, args.sample // len(strata))
        cand = [x for v in strata.values() for x in rng.sample(v, min(per, len(v)))]
    out = O / ("judgments_%s.jsonl" % args.tag)
    done = set()
    if out.exists():
        for l in open(out):
            r = json.loads(l)
            if r.get("label"):
                done.add((r["a"], r["b"]))
    todo = [c for c in cand if (c["a"], c["b"]) not in done]
    print("candidates %d, done %d, to do %d" % (len(cand), len(done), len(todo)), flush=True)
    with open(out, "a") as f, cf.ThreadPoolExecutor(args.workers) as ex:
        for start in range(0, len(todo), args.workers * 20):
            chunk = todo[start:start + args.workers * 20]
            for c, (lab, why, err) in zip(chunk, ex.map(lambda c: call(msgs[c["a"]], msgs[c["b"]], args.model), chunk)):
                f.write(json.dumps(dict(c, label=lab, why=why, error=err, model=args.model), ensure_ascii=False) + "\n")
            f.flush()
            print("%s %d/%d" % (time.strftime("%H:%M"), start + len(chunk), len(todo)), flush=True)


if __name__ == "__main__":
    main()
