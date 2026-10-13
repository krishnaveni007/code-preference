"""Step 4: merge clusters that name the same matter, re-judge the merged groups, write the final matter file.

Usage: python merge_matters.py [--inp restated] [--thr 0.3] [--merge-thr 0.25] [--model M] [--workers 16] [--chunk 90]
Reads data/stance_<thr>.jsonl; embeds each judged cluster's "matter" name; clusters the names (average linkage,
cosine distance <= merge-thr); groups with >=2 sub-clusters are re-judged on the union of their members
(chunked if large). Output: data/stance_merged_<merge-thr>.jsonl and data/matters_final.jsonl
(= re-judged merged groups + untouched single clusters).
"""
import argparse, collections, json
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import common as C
from stance import run_jobs, chunked

ap = argparse.ArgumentParser(); ap.add_argument("--inp", default="restated"); ap.add_argument("--thr", type=float, default=0.3)
ap.add_argument("--merge-thr", type=float, default=0.25); ap.add_argument("--model", default=C.DEFAULT_CHAT_MODEL)
ap.add_argument("--workers", type=int, default=16); ap.add_argument("--chunk", type=int, default=90)
a = ap.parse_args()
R = C.load_records(C.DATA / f"{a.inp}.jsonl")
ids = json.load(open(C.DATA / "manner_ids.json")); lab = json.load(open(C.DATA / f"clusters_{a.thr}.json"))
raw = collections.defaultdict(list)
for i, c in zip(ids, lab): raw[c].append(i)
S = {o["cluster"]: o for o in C.load_records(C.DATA / f"stance_{a.thr}.jsonl").values()}
good = [c for c in S if S[c].get("one_matter") and S[c].get("matter")]
names = [S[c]["matter"] for c in good]
E = C.embed(names); np.save(C.DATA / "E_matter_names.npy", E); json.dump(good, open(C.DATA / "matter_name_order.json", "w"))
labm = AgglomerativeClustering(n_clusters=None, distance_threshold=a.merge_thr, metric="cosine", linkage="average").fit_predict(E)
groups = collections.defaultdict(list)
for c, l in zip(good, labm): groups[int(l)].append(c)
def members_of(c):  # original cluster id may be "<c>" or "<c>_<k>" (chunk)
    return raw[int(str(c).split("_")[0])] if "_" in str(c) else raw[c]
jobs = []
for g, cs in groups.items():
    if len(cs) < 2: continue
    mem = sorted({i for c in cs for i in members_of(c)})
    jobs += chunked(f"m{g}", mem, a.chunk, "merged")
print("merged groups:", sum(1 for cs in groups.values() if len(cs) > 1), "| jobs:", len(jobs), flush=True)
out = C.DATA / f"stance_merged_{a.merge_thr}.jsonl"
run_jobs(jobs, R, out, a.model, a.workers, temps=(0, 0.3, 0.5))
final = list(C.load_records(out).values()) + [S[c] for cs in groups.values() if len(cs) == 1 for c in cs]
with open(C.DATA / "matters_final.jsonl", "w") as fo:
    for o in final: fo.write(json.dumps(o, ensure_ascii=False) + "\n")
print("matters_final.jsonl:", len(final), "matters")
