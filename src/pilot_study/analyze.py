#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd

METRICS=["judge_score","passed","user_correction","intent_coverage","interaction_turns","simulator_messages","input_tokens","output_tokens","reasoning_tokens","estimated_cost_usd","wall_clock_seconds","unresolved_requirements","preference_adherence"]
def main():
 p=argparse.ArgumentParser();p.add_argument("--runs",type=Path,required=True);p.add_argument("--out-dir",type=Path,required=True);a=p.parse_args();rows=[]
 for path in a.runs.glob("*/metrics.json"):
  raw=json.loads(path.read_text()); meta=json.loads((path.parent/"run_metadata.json").read_text()) if (path.parent/"run_metadata.json").exists() else {}
  raw["user_correction"]=raw.get("corrections",0)+.2*raw.get("nudges",0);rows.append({**meta,**raw})
 if not rows: raise SystemExit("no metrics.json files found; no experimental findings can be reported")
 df=pd.DataFrame(rows);a.out_dir.mkdir(parents=True,exist_ok=True);df.to_csv(a.out_dir/"per_run.csv",index=False)
 keys=["pseudonymous_user_id","task_id","replicate"]
 duplicated=df.duplicated(keys+["condition"])
 if duplicated.any(): raise SystemExit("duplicate condition within a user/task/replicate pair")
 counts=df.groupby(keys).condition.nunique()
 if (counts != 2).any(): raise SystemExit("incomplete baseline/intervention pairs found")
 wide=df.pivot(index=keys,columns="condition",values=[m for m in METRICS if m in df]).reset_index()
 effects=[]
 for _,r in wide.iterrows():
  item={k:r[(k,"")] for k in keys}
  for m in [x for x in METRICS if x in df]:
   b=r[(m,"baseline")];i=r[(m,"intervention")];item[f"{m}_absolute_change"]=i-b;item[f"{m}_percent_change"]=None if b==0 else 100*(i-b)/b
  effects.append(item)
 pd.DataFrame(effects).to_csv(a.out_dir/"paired_effects.csv",index=False)
 effect_df=pd.DataFrame(effects)
 per_user=effect_df.groupby("pseudonymous_user_id",as_index=False).mean(numeric_only=True);per_user.to_csv(a.out_dir/"per_user.csv",index=False)
 summary=per_user.mean(numeric_only=True).to_dict();(a.out_dir/"summary.json").write_text(json.dumps({"exploratory":True,"equal_weight_unit":"user","n_users":len(per_user),"mean_user_effects":summary},indent=2)+"\n")
 focus=[m for m in ["judge_score","passed","user_correction","intent_coverage","interaction_turns","input_tokens","output_tokens","reasoning_tokens","estimated_cost_usd"] if f"{m}_absolute_change" in summary]
 lines=["# Exploratory longitudinal preference pilot","",f"Users analyzed: {len(per_user)}. Effects are intervention minus baseline and are averaged with equal weight per user.","","| Metric | Mean absolute change | Mean percent change |","|---|---:|---:|"]
 for m in focus:
  absolute=summary.get(f"{m}_absolute_change",float("nan"));percent=summary.get(f"{m}_percent_change",float("nan"))
  lines.append(f"| {m} | {absolute:.4g} | {percent:.2f}% |")
 lines += ["","Do not interpret this exploratory result as confirmatory evidence. Review run failures, simulator/model drift, leakage, unresolved requirements, and preference applicability before drawing conclusions."]
 (a.out_dir/"pilot_report.md").write_text("\n".join(lines)+"\n")
if __name__=="__main__":main()
