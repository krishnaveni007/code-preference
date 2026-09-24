#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.extraction.preference_judge import RUBRICS
from src.pilot_study.common import read_json, write_json

EXCLUDED_FROM_INTERVENTION = {"specification_granularity"}


def profiles(manifest: list[dict], judgments: Path, threshold: float, min_sessions: int) -> list[dict]:
    allowed = {sid for row in manifest for sid in row["prior_session_ids"]}
    records = [json.loads(line) for line in judgments.read_text().splitlines() if line.strip()]
    by_user = []
    rubric = {r["id"]: r for r in RUBRICS}
    for row in manifest:
        timestamps = {item["session_id"]: item.get("timestamp") for item in row.get("prior_sessions", [])}
        axes = []
        for axis_id, meta in rubric.items():
            if axis_id in EXCLUDED_FROM_INTERVENTION:
                continue
            evidence = []
            for rec in records:
                event, axis = rec.get("event", {}), rec.get("axes", {}).get(axis_id, {})
                sid = str(event.get("session_id", ""))
                if sid in allowed and sid in row["prior_session_ids"] and axis.get("label") in ("high", "low"):
                    evidence.append((sid, timestamps.get(sid), axis))
            # Match preftool: every supported turn is an observation; direction
            # is the sign of the mean and confidence is Laplace-smoothed
            # agreement (agreeing + 1) / (support + 2).
            high = sum(axis["label"] == "high" for _, _, axis in evidence)
            low = sum(axis["label"] == "low" for _, _, axis in evidence)
            if high == low:
                continue
            direction = "high" if high > low else "low"
            support = high + low
            agreeing = max(high, low)
            confidence = (agreeing + 1) / (support + 2) if support else 0.0
            supporting = sorted({sid for sid, _, axis in evidence if axis["label"] == direction})
            contradicting = sorted({sid for sid, _, axis in evidence if axis["label"] != direction})
            if len(supporting) >= min_sessions and confidence >= threshold:
                axes.append({"dimension": axis_id, "normalized_preference": meta[direction], "direction": direction,
                             "confidence": round(confidence, 4), "scope": "persistent_user",
                             "supporting_sessions": supporting, "contradicting_evidence": contradicting,
                             "supporting_turns": agreeing, "contradicting_turns": min(high, low),
                             "last_observed_timestamp": max((ts for _, ts, _ in evidence if ts), default=None)})
        by_user.append({"pseudonymous_user_id": row["pseudonymous_user_id"], "task_id": row["task_id"], "preferences": axes})
    return by_user


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--manifest",type=Path,required=True); p.add_argument("--judgments",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True); p.add_argument("--threshold",type=float,default=.75); p.add_argument("--min-sessions",type=int,default=2)
    p.add_argument("--user-id", help="Restrict output to one pseudonymous user")
    p.add_argument("--task-id", help="Restrict output to one held-out task")
    a=p.parse_args(); manifest=read_json(a.manifest)
    if a.user_id:
        manifest=[row for row in manifest if row["pseudonymous_user_id"] == a.user_id]
        if not manifest:
            raise SystemExit(f"user not found in manifest: {a.user_id}")
    if a.task_id:
        manifest=[row for row in manifest if row["task_id"] == a.task_id]
        if not manifest:
            raise SystemExit(f"task not found in manifest: {a.task_id}")
    write_json(a.out, profiles(manifest,a.judgments,a.threshold,a.min_sessions))
if __name__ == "__main__": main()
