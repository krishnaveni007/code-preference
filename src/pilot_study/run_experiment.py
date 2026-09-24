#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, shlex, subprocess, time
from pathlib import Path
from src.pilot_study.common import read_json, write_json

DEFAULT_PASSTHROUGH_ENV = ("PATH", "HOME", "TMPDIR", "LANG", "LC_ALL")

def main():
    p=argparse.ArgumentParser();p.add_argument("--manifest",type=Path,required=True);p.add_argument("--interventions",type=Path,required=True)
    p.add_argument("--out-dir",type=Path,required=True);p.add_argument("--replicates",type=int,default=3);p.add_argument("--seed",type=int,default=20260907)
    p.add_argument("--command",help="Adapter command; receives PILOT_* environment variables");p.add_argument("--smoke-test",action="store_true")
    p.add_argument("--execute",action="store_true");p.add_argument("--skip-existing",action="store_true");p.add_argument("--approved-budget-usd",type=float)
    p.add_argument("--pass-env", action="append", default=[], help="Environment variable to expose to the adapter; repeat as needed")
    a=p.parse_args(); rows=read_json(a.manifest); interventions={r["task_id"]:r for r in read_json(a.interventions)}
    if a.smoke_test: rows=rows[:1]; a.replicates=1
    plan=[]
    for row in rows:
      for rep in range(a.replicates):
       for condition in ("baseline","intervention"):
        run_id=f"{row['task_id']}__r{rep+1}__{condition}"; out=a.out_dir/run_id
        plan.append({"run_id":run_id,"task_id":row["task_id"],"pseudonymous_user_id":row["pseudonymous_user_id"],"condition":condition,"replicate":rep+1,"seed":a.seed+rep,"output_dir":str(out),
                     "profile_path":interventions.get(row["task_id"],{}).get("path") if condition=="intervention" else None})
    write_json(a.out_dir/"run_plan.json",plan);print(f"planned_runs={len(plan)} paired_tasks={len(rows)*a.replicates}")
    if not a.execute: print("dry-run only; pass --execute with --command and --approved-budget-usd");return
    if not a.command or a.approved_budget_usd is None: raise SystemExit("execution requires --command and --approved-budget-usd")
    for run in plan:
      out=Path(run["output_dir"]); metrics=out/"metrics.json"
      if a.skip_existing and metrics.exists(): continue
      out.mkdir(parents=True,exist_ok=True)
      allowed_env = set(DEFAULT_PASSTHROUGH_ENV) | set(a.pass_env)
      env = {key: value for key, value in os.environ.items() if key in allowed_env}
      env.update({f"PILOT_{k.upper()}":str(v or "") for k,v in run.items()})
      started=time.time(); proc=subprocess.run(shlex.split(a.command),env=env,text=True,capture_output=True)
      (out/"stdout.log").write_text(proc.stdout);(out/"stderr.log").write_text(proc.stderr)
      write_json(out/"run_metadata.json",{**run,"started_at":started,"wall_clock_seconds":time.time()-started,"returncode":proc.returncode})
      if proc.returncode: raise SystemExit(f"run failed: {run['run_id']}")

if __name__=="__main__":main()
