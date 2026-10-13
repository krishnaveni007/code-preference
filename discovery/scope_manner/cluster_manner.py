"""Step 2: embed "topic. manner" for every manner/both record and cluster into candidate matters.

Usage: python cluster_manner.py [--inp restated] [--thr 0.3]
Outputs (data/): manner_ids.json (record ids in embedding order), E_manner.npy, clusters_<thr>.json (label per id).
"""
import argparse, collections, json
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import common as C

ap = argparse.ArgumentParser(); ap.add_argument("--inp", default="restated"); ap.add_argument("--thr", type=float, default=0.3); ap.add_argument("--force", action="store_true", help="overwrite an existing clusters file")
a = ap.parse_args()
R = C.load_records(C.DATA / f"{a.inp}.jsonl")
msgs = C.load_messages()
M = [o for o in R.values() if o.get("kind") != "task" and o.get("manner", "").strip()]
M.sort(key=lambda o: o["id"]); ids = [o["id"] for o in M]
print("records", len(R), "manner", len(M), "users", len({msgs[i]["user_id"] for i in ids}), flush=True)
idf, ef = C.DATA / "manner_ids.json", C.DATA / "E_manner.npy"
if idf.exists() and ef.exists() and json.load(open(idf)) == ids:
    E = np.load(ef)
else:
    E = C.embed([o["topic"] + ". " + o["manner"] for o in M]); np.save(ef, E); json.dump(ids, open(idf, "w"))
cf = C.DATA / f"clusters_{a.thr}.json"
if cf.exists() and not a.force:
    raise SystemExit(f"{cf} exists; downstream stance files refer to its cluster numbers. Use --force to recluster.")
lab = AgglomerativeClustering(n_clusters=None, distance_threshold=a.thr, metric="cosine", linkage="average").fit_predict(E)
sizes = collections.Counter(lab)
json.dump([int(x) for x in lab], open(C.DATA / f"clusters_{a.thr}.json", "w"))
print(f"thr {a.thr}: clusters {len(sizes)}, size>=3: {sum(1 for n in sizes.values() if n >= 3)}, max {max(sizes.values())}, singletons {sum(1 for n in sizes.values() if n == 1)}")
