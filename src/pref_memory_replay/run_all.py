"""Run every arm of every case (run_turn.py) and judge it (judge_turn.py), N at a time; resumable.

Arms per case: C -> c0_r1 (no memory), c2p_r1 (C2), mem_r1 (real store, only if mem_real/ exists);
U -> full_c0_r1, ablated_c0_r1, ablated_c2p_r1, ablated_mem_r1 (only if mem_real/ exists). The recorded turn is
judged once per case (judge_recorded.json). A finished arm (judge.json present) is skipped; --rejudge re-runs the
judge only (e.g. after a rubric change).

  python src/pref_memory_replay/run_all.py --cc-bin /abs/claude-for-judge [--cases a,b] [--jobs 4] [--rejudge]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import common  # noqa: E402

ROOT = common.MODULE
PY = sys.executable
DATA = common.DATA
MODEL = common.AGENT_MODEL


def arms(case):
    cdir = DATA / case["case_id"]
    real = (cdir / "mem_real" / "MEMORY.md").exists()
    if case["kind"] == "C":
        out = [("c0_r1", None, None), ("c2p_r1", "mem_c2p", None)] + ([("mem_r1", "mem_real", None)] if real else [])
    else:
        abl = cdir / "prep" / "prompt_ablated.txt"
        out = [("full_c0_r1", None, None), ("ablated_c0_r1", None, abl), ("ablated_c2p_r1", "mem_c2p", abl)] + \
            ([("ablated_mem_r1", "mem_real", abl)] if real else [])
    return out


def sh(cmd, log):
    with open(log, "a") as f:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT).returncode


def job(case_path, run, mem, prompt_file, cc_bin, rejudge):
    case = json.loads(pathlib.Path(case_path).read_text())
    cdir = DATA / case["case_id"]
    log = cdir / f"{run}.log"
    if not (cdir / run / "run.json").exists():
        cmd = [PY, str(ROOT / "run_turn.py"), case_path, "--prep", str(cdir / "prep"), "--out", str(cdir / run),
               "--model", MODEL, "--timeout", "2700"]
        if mem:
            cmd += ["--memory-dir", str(cdir / mem)]
        if prompt_file:
            cmd += ["--prompt", prompt_file.read_text()]
        if case.get("setup"):
            cmd += ["--setup", case["setup"]]
        sh(cmd, log)
    if rejudge or not (cdir / run / "judge.json").exists():
        sh([PY, str(ROOT / "judge_turn.py"), case_path, "--run", str(cdir / run), "--cc-bin", cc_bin,
            "--out", str(cdir / run / "judge.json")], log)
    j = json.loads((cdir / run / "judge.json").read_text()) if (cdir / run / "judge.json").exists() else {}
    v = j.get("verdict") or {}
    return case["case_id"], run, v.get("correction_warranted") or v.get("follows_preference") or "FAILED"


def recorded(case_path, cc_bin, rejudge):
    case = json.loads(pathlib.Path(case_path).read_text())
    cdir = DATA / case["case_id"]
    out = cdir / "judge_recorded.json"
    if rejudge or not out.exists():
        sh([PY, str(ROOT / "judge_turn.py"), case_path, "--recorded", "--cc-bin", cc_bin, "--out", str(out)],
           cdir / "recorded.log")
    v = (json.loads(out.read_text()).get("verdict") or {}) if out.exists() else {}
    return case["case_id"], "recorded", v.get("correction_warranted") or v.get("follows_preference") or "FAILED"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cc-bin", default=common.JUDGE_CC, required=not common.JUDGE_CC)
    ap.add_argument("--cases", help="comma-separated case ids (default: every case in $TURN_REPLAY_CASES)")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--rejudge", action="store_true")
    a = ap.parse_args()
    paths = sorted(common.CASES.glob("*.json"))
    if a.cases:
        paths = [p for p in paths if p.stem in a.cases.split(",")]
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        futs = [ex.submit(recorded, str(p), a.cc_bin, a.rejudge) for p in paths]
        for p in paths:
            case = json.loads(p.read_text())
            futs += [ex.submit(job, str(p), run, mem, pf, a.cc_bin, a.rejudge) for run, mem, pf in arms(case)]
        for f in cf.as_completed(futs):
            try:
                print(*f.result(), flush=True)
            except Exception as e:  # keep going; the arm's log has the details
                print("ERROR", repr(e), flush=True)


if __name__ == "__main__":
    main()
