"""Step 3: one LLM call per cluster (>=3 members): same matter?, stance groups, mutually exclusive stance pairs.

Usage: python stance.py [--inp restated] [--thr 0.3] [--model M] [--workers 16] [--chunk 90] [--retry]
Output: data/stance_<thr>.jsonl (append-only, resumable). Clusters larger than --chunk are judged in chunks
(cluster id becomes "<c>_<k>"). --retry re-judges clusters whose previous attempt failed, with lenient parsing.
"""
import argparse, collections, json, threading
from concurrent.futures import ThreadPoolExecutor
import common as C

PROMPT = """You are given short statements, each restating how a developer wants an AI coding agent to work. They were grouped automatically because they look similar. Decide whether they concern one matter, and if so, group them by stance. Output JSON only.

"matter": the matter they concern, as a short noun phrase (or "" if they do not share one).
"one_matter": true if at least most statements are about the same matter.
"stances": a list of groups. Each group is {"label": a short phrase stating the stance, "members": [statement ids]}. Two statements share a stance if following one would satisfy the other. Put statements that are off-matter into a group labelled "off-matter".
"opposite": a list of [label_a, label_b] pairs where following stance a would violate stance b. Only list pairs that are genuinely incompatible, not merely different.
"""


def judge_group(R, mem, model, temps=(0,)):
    txt = "\n".join(f"[{i}] {R[i]['manner']}" for i in mem)
    last = None
    for t in temps:
        try:
            o = C.parse_json(C.chat([{"role": "system", "content": PROMPT}, {"role": "user", "content": txt}], model,
                                    {"type": "json_object"}, temperature=t, timeout=300, max_tokens=32000))
            assert isinstance(o.get("stances"), list); return o
        except Exception as e:  # noqa: BLE001
            last = str(e)[:200]
    raise RuntimeError(last)


def run_jobs(jobs, R, out, model, workers, temps):
    """jobs: list of (job_id, member_ids, note). Appends to `out`; skips ids already successful."""
    done = set(C.load_records(out)) if out.exists() else set()
    lock = threading.Lock(); f = open(out, "a")

    def one(job):
        jid, mem, note = job
        if jid in done: return
        try: o = judge_group(R, mem, model, temps); o.update(cluster=jid, n=len(mem), note=note)
        except Exception as e:  # noqa: BLE001
            o = {"cluster": jid, "error": str(e)[:200]}
        with lock:
            f.write(json.dumps(o, ensure_ascii=False) + "\n"); f.flush()
            print(jid, len(mem), "ok" if "error" not in o else "ERR " + o["error"][:60], flush=True)
    with ThreadPoolExecutor(workers) as ex: list(ex.map(one, jobs))
    f.close()


def chunked(jid, mem, chunk, note):
    if len(mem) <= chunk: return [(str(jid), mem, note)]
    return [(f"{jid}_{k // chunk}", mem[k:k + chunk], note + "/chunk") for k in range(0, len(mem), chunk)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--inp", default="restated"); ap.add_argument("--thr", type=float, default=0.3)
    ap.add_argument("--model", default=C.DEFAULT_CHAT_MODEL); ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=90); ap.add_argument("--retry", action="store_true")
    a = ap.parse_args()
    R = C.load_records(C.DATA / f"{a.inp}.jsonl")
    ids = json.load(open(C.DATA / "manner_ids.json")); lab = json.load(open(C.DATA / f"clusters_{a.thr}.json"))
    clusters = collections.defaultdict(list)
    for i, c in zip(ids, lab): clusters[c].append(i)
    out = C.DATA / f"stance_{a.thr}.jsonl"
    jobs = [j for c, mem in sorted(clusters.items()) if len(mem) >= 3 for j in chunked(c, mem, a.chunk, "cluster")]
    if a.retry:
        ok = set(C.load_records(out)); bad = {json.loads(l)["cluster"] for l in open(out) if '"error"' in l}
        jobs = [j for j in jobs if j[0] in bad - ok]; print("retrying", len(jobs))
    run_jobs(jobs, R, out, a.model, a.workers, temps=(0, 0.3, 0.5) if a.retry else (0,))
    recs = [json.loads(l) for l in open(out)]
    print("clusters", len(recs), "errors", sum(1 for o in recs if "error" in o), "one_matter", sum(1 for o in recs if o.get("one_matter")), flush=True)
