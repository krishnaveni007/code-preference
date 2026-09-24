import json
from pathlib import Path
import pandas as pd
from src.pilot_study.build_longitudinal import build_manifest, select
from src.pilot_study.interventions import render, render_audit, START, END
from src.pilot_study.build_profiles import profiles
from src.pilot_study.extract_codex import PUSHBACK_TYPES, seed_pushback_cache

def test_temporal_mapping_and_selection(tmp_path):
    times=pd.date_range("2025-01-01",periods=7,tz="UTC")
    rows=[{"session_id":f"s{i}","user_id":"real-user","repo_id":"r","created_at":times[i]} for i in range(7)]
    pd.DataFrame(rows).to_parquet(tmp_path/"sessions.parquet")
    task=tmp_path/"task-a";task.mkdir();(task/"original_session.json").write_text(json.dumps({"session_id":"s6","task_id":"t1"}))
    manifest=build_manifest(tmp_path/"sessions.parquet",tmp_path,"salt")
    assert manifest[0]["n_prior_sessions"]==6
    assert "s6" not in manifest[0]["prior_session_ids"]
    assert select(manifest,1,1)[0]["task_id"]=="t1"

def test_intervention_is_bounded_and_marked():
    profile={"preferences":[{"dimension":"b","confidence":.8,"normalized_preference":"Be concise."},{"dimension":"a","confidence":.9,"normalized_preference":"Test broadly."}]}
    text=render(profile,300)
    assert text.startswith(START) and text.endswith(END+"\n") and len(text)<=300
    profile.update({"pseudonymous_user_id": "u", "task_id": "t"})
    for pref in profile["preferences"]:
        pref.update({"direction": "high", "supporting_turns": 3,
                     "contradicting_turns": 0, "supporting_sessions": ["s1", "s2"],
                     "contradicting_evidence": []})
    audit=render_audit(profile,text)
    assert "# Extracted working preferences" in audit
    assert "canonical SWE-Chat pushback turns only" in audit

def test_profile_matches_preftool_confidence_and_drops_descriptive_axis(tmp_path):
    judgments = tmp_path / "judgments.jsonl"
    records = []
    for sid, label in (("s1", "high"), ("s2", "high"), ("s3", "low")):
        records.append({"event": {"session_id": sid}, "axes": {
            "agent_autonomy": {"label": label},
            "specification_granularity": {"label": "high"},
        }})
    judgments.write_text("\n".join(json.dumps(r) for r in records))
    manifest = [{"pseudonymous_user_id": "u", "task_id": "t",
                 "prior_session_ids": ["s1", "s2", "s3"],
                 "prior_sessions": [{"session_id": s, "timestamp": f"2025-01-0{i}T00:00:00Z"}
                                    for i, s in enumerate(("s1", "s2", "s3"), 1)]}]
    result = profiles(manifest, judgments, threshold=.59, min_sessions=2)[0]["preferences"]
    autonomy = next(p for p in result if p["dimension"] == "agent_autonomy")
    assert autonomy["confidence"] == .6  # (2 agreeing + 1) / (3 support + 2)
    assert all(p["dimension"] != "specification_granularity" for p in result)

def test_seed_pushback_cache_excludes_non_pushback_and_failed(tmp_path):
    data = tmp_path / "data"; data.mkdir()
    pd.DataFrame([
        {"session_id": "s1", "turn_number": 1, "role": "user", "is_conversational": True,
         "prompt_pushback": "correction"},
        {"session_id": "s1", "turn_number": 2, "role": "user", "is_conversational": True,
         "prompt_pushback": "non_pushback"},
        {"session_id": "s1", "turn_number": 3, "role": "user", "is_conversational": True,
         "prompt_pushback": "failure_report"},
    ]).to_parquet(data / "conversations.parquet")
    source = tmp_path / "source.jsonl"
    records = [
        {"event": {"session_id": "s1", "target_raw_turn_number": 1}},
        {"event": {"session_id": "s1", "target_raw_turn_number": 2}},
        {"event": {"session_id": "s1", "target_raw_turn_number": 3}, "llm_call_failed": True},
    ]
    source.write_text("\n".join(json.dumps(record) for record in records))
    destination = tmp_path / "out" / "judgments.jsonl"
    assert seed_pushback_cache(source, destination, data, ["s1"]) == 1
    seeded = [json.loads(line) for line in destination.read_text().splitlines()]
    assert seeded[0]["event"]["target_raw_turn_number"] == 1
    assert "correction" in PUSHBACK_TYPES
