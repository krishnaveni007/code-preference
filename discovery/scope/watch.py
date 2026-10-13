#!/usr/bin/env python3
"""Progress summary of the main judgments: labels by level so far (random order -> unbiased prefix)."""
import json
from collections import Counter, defaultdict
from pathlib import Path

O = Path(__file__).resolve().parents[2] / "outputs/discovery/scope"
rows = [json.loads(l) for l in open(O / "judgments_sonnet46_v5.jsonl")]
ok = [r for r in rows if r.get("label")]
err = Counter((r.get("error") or "")[:60] for r in rows if not r.get("label"))
print("judged %d (failed %d) %s" % (len(ok), len(rows) - len(ok), dict(err.most_common(2)) if err else ""))
by = defaultdict(Counter)
for r in ok:
    by[r["level"]][r["label"]] += 1
print("%-13s %6s %7s %8s %9s %9s" % ("level", "n", "same", "related", "conflict", "unrelated"))
for lvl in ("same_session", "same_repo", "other_repo", "other_agent"):
    c = by[lvl]; n = sum(c.values())
    if n:
        print("%-13s %6d %6.1f%% %7.1f%% %8.1f%% %8.1f%%" % (lvl, n, *(100 * c[k] / n for k in ("same", "related", "conflict", "unrelated"))))
bins = [0.70, 0.75, 0.80, 0.85, 0.95]
print("same rate by raw score:", ", ".join("%.2f-%.2f %.0f%% (n=%d)" % (lo, hi, 100 * sum(r["label"] == "same" for r in b) / max(len(b), 1), len(b))
                                         for lo, hi in zip(bins, bins[1:]) for b in [[r for r in ok if lo <= r["raw"] < hi]]))
