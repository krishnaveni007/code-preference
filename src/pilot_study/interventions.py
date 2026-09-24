#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from src.pilot_study.common import read_json, write_json

START = "<!-- longitudinal-preference-profile:start -->"
END = "<!-- longitudinal-preference-profile:end -->"

def render(profile: dict, max_chars: int) -> str:
    prefs=sorted(profile["preferences"], key=lambda p:(-p["confidence"],p["dimension"]))
    lines=[START,"## User working preferences","Apply these only when relevant and when they do not conflict with task or repository instructions."]
    for p in prefs:
        candidate=f"- {p['normalized_preference']}"
        if len("\n".join(lines+[candidate,END])) > max_chars: break
        lines.append(candidate)
    return "\n".join(lines+[END])+"\n"

def render_audit(profile: dict, intervention: str) -> str:
    prefs=sorted(profile["preferences"], key=lambda p:(-p["confidence"],p["dimension"]))
    session_ids={sid for pref in prefs for sid in
                 pref["supporting_sessions"] + pref["contradicting_evidence"]}
    lines=[
        "# Extracted working preferences",
        "",
        f"User: `{profile['pseudonymous_user_id']}`  ",
        f"Held-out task: `{profile['task_id']}`  ",
        "Evidence filter: canonical SWE-Chat pushback turns only  ",
        f"Sessions contributing threshold-passing evidence: {len(session_ids)}",
        "",
        "## Agent intervention",
        "",
        intervention.strip(),
        "",
        "## Evidence audit",
        "",
        "| Dimension | Direction | Confidence | Supporting turns | Contradicting turns | Supporting sessions |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for pref in prefs:
        lines.append(
            f"| `{pref['dimension']}` | {pref['direction']} | {pref['confidence']:.4f} | "
            f"{pref['supporting_turns']} | {pref['contradicting_turns']} | "
            f"{len(pref['supporting_sessions'])} |"
        )
    if not prefs:
        lines.append("| _No preferences passed the fixed thresholds._ | | | | | |")
    lines += ["", "The descriptive `specification_granularity` axis is excluded from intervention.", ""]
    return "\n".join(lines)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--profiles",type=Path,required=True);p.add_argument("--out-dir",type=Path,required=True);p.add_argument("--max-chars",type=int,default=1800)
    a=p.parse_args(); index=[]; swe_profiles={}
    for prof in read_json(a.profiles):
        path=a.out_dir/prof["task_id"] / "AGENTS.md"; path.parent.mkdir(parents=True,exist_ok=True); text=render(prof,a.max_chars);path.write_text(text)
        (path.parent / "preference.md").write_text(render_audit(prof, text))
        index.append({"task_id":prof["task_id"],"path":str(path),"content":text})
        swe_profiles[prof["task_id"]] = text
    write_json(a.out_dir/"index.json",index)
    write_json(a.out_dir/"swe_together_profiles.json",swe_profiles)
if __name__=="__main__": main()
