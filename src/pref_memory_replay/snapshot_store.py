"""Materialise a user's recovered Claude Code memory store as it stood at a given time (the "real memory" arm).

Source: swechat_memory_stores/observations.jsonl (only real unbounded Read/Write observations). For each note the
latest observation at or before --at is taken; nothing is interpolated. Writes the notes to --out and a report of
  stale_days       how old each note's evidence is at --at (the note may have changed in the unobserved gap)
  index_unbacked   notes the chosen MEMORY.md links but whose body has no observation at or before --at
  after_only       notes first observed after --at (may or may not have existed then; excluded)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import re
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import common  # noqa: E402

STORES = common.STORES
LINK_RE = re.compile(r"\]\(([^)]+\.md)\)")


def ts(s: str) -> dt.datetime:
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--at", required=True, help="ISO timestamp, e.g. the target prompt's timestamp")
    ap.add_argument("--out", required=True)
    ap.add_argument("--slug", help="Claude Code project folder of the target session (its cwd with every "
                    "non-alphanumeric char as '-'); keeps only that project's memory when several share a repo")
    a = ap.parse_args()
    at = ts(a.at)
    rows = [json.loads(l) for l in open(STORES / "observations.jsonl")]
    rows = [r for r in rows if r["repo"] == a.repo]
    if a.slug:
        rows = [r for r in rows if f"/projects/{a.slug}/memory/" in r["path"].replace("\\", "/")]
    latest, after = {}, set()
    for r in rows:
        if ts(r["ts"]) <= at:
            if r["note"] not in latest or ts(r["ts"]) >= ts(latest[r["note"]]["ts"]):
                latest[r["note"]] = r
        else:
            after.add(r["note"])
    out = pathlib.Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for note, r in latest.items():
        (out / note).write_text((STORES / r["state_file"]).read_text())
    index = (out / "MEMORY.md").read_text() if (out / "MEMORY.md").exists() else ""
    linked = set(LINK_RE.findall(index))
    report = {"repo": a.repo, "slug": a.slug, "at": a.at, "notes": len(latest),
              "stale_days": {n: round((at - ts(r["ts"])).total_seconds() / 86400, 2) for n, r in sorted(latest.items())},
              "index_unbacked": sorted(linked - set(latest)),
              "after_only": sorted(after - set(latest))}
    (out.parent / f"{out.name}_report.json").write_text(json.dumps(report, indent=1))
    print(f"{len(latest)} notes, index links {len(linked)}, unbacked {len(report['index_unbacked'])}, "
          f"after-only {len(report['after_only'])}, MEMORY.md {report['stale_days'].get('MEMORY.md')} days old")


if __name__ == "__main__":
    main()
