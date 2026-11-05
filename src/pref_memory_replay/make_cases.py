"""Turn selected candidates (find_candidates.py output, filtered to a selection) into case files for prepare_turn.py.

C case: replay the prompt before the correction; the judge gets the correction text and the trajectory paraphrase.
U case: replay the statement's prompt in full and ablated. The ablated prompt is a minimal edit that removes only the
preference (one tool-less model call, cached in the case file once written); review every ablation by hand.

  python src/pref_memory_replay/make_cases.py --selection $TURN_REPLAY_DATA/selection.json --tier pick --cc-bin /abs/claude
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from find_candidates import load_lines, real_prompt  # noqa: E402
from inspect_transcript import text_of  # noqa: E402
from judge_turn import judge  # noqa: E402
import common  # noqa: E402

CASES = common.CASES
ABLATE_SCHEMA = {"type": "object", "required": ["ablated", "removed"],
                 "properties": {"ablated": {"type": "string"}, "removed": {"type": "string"}}}
ABLATE_PROMPT = """A developer wrote the message below to a coding agent. Part of it states a working preference
(dimension: {dim}; side: {side}; paraphrase: {manner}). The span that carries it:
<span>
{span}
</span>

Rewrite the message with that preference removed: delete the words that state it and words that only serve it, keep
the task and every other word verbatim, fix only the grammar the deletion breaks. Do not add anything. If the
preference IS the whole request, return the task as it would read without it (shortest neutral wording).
Return JSON {{"ablated": "<message>", "removed": "<what you deleted>"}}.

<message>
{message}
</message>"""


def cwd_regex(cwd: str) -> str:
    """Regex matching the original cwd with either slash style (Windows transcripts mix them)."""
    parts = [re.escape(p) for p in re.split(r"[\\/]+", cwd) if p]
    drive = re.match(r"^([A-Za-z]):", cwd)
    if drive:
        parts[0] = f"[{drive.group(1).upper()}{drive.group(1).lower()}]:"
        return r"[\\/]+".join(parts)
    return r"[\\/]+" + r"[\\/]+".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", required=True)
    ap.add_argument("--tier", default="pick")
    ap.add_argument("--cc-bin", default=common.JUDGE_CC, required=not common.JUDGE_CC)
    ap.add_argument("--model", default=common.JUDGE_MODEL)
    a = ap.parse_args()
    for r in json.load(open(a.selection)):
        if r["tier"] != a.tier:
            continue
        cid = f"{re.sub(r'[^A-Za-z0-9]', '', r['user'])[:16]}_{r['k']}_{r['s']}"
        CASES.mkdir(parents=True, exist_ok=True)
        path = CASES / f"{cid}.json"
        old = json.loads(path.read_text()) if path.exists() else {}
        lines = load_lines(pathlib.Path(r["transcript"]))
        name = re.split(r"[\\/]+", r["cwd"].rstrip("\\/"))[-1]
        case = {
            "case_id": cid, "kind": r["k"], "user": r["user"],
            "group": f"{r['mode']} / {r['dim']} / {r['side']}",
            "session_id": r["session_id"], "transcript": common.relative_transcript(r["transcript"]), "cut_line": r["cut_line"],
            "repo_url": f"https://github.com/{r['repo']}", "repo": r["repo"], "sha": r["sha"],
            "branch": r["branch_at_cut"] or r["anchor_branch"] or "replay",
            "orig_cwd_regex": cwd_regex(r["cwd"]), "container_cwd": f"/workspace/{name}",
            "ts_cut": r["ts_cut"], "ctx_tokens": r["ctx_tokens"], "anchor_line": r["anchor_line"],
            "judge": {"dim": r["dim"], "side": r["side"], "manner": r["manner"]},
        }
        nxt = next((i for i in range(r["cut_line"] + 1, len(lines)) if real_prompt(lines[i])), len(lines))
        if r["k"] == "C":
            case["correction_line"] = r["stmt_line"]
            case["judge"]["correction"] = text_of(lines[r["stmt_line"]])
            case["note"] = (f"Replay the agent's response to the prompt before the correction. Correction ({r['corrects']}): "
                            f"{r['quote'][:300]}. State = commit {r['sha']} (last HEAD the transcript shows, no file "
                            f"change after it before the turn).")
            if nxt != r["stmt_line"]:
                case["note"] += f" NOTE: {r['stmt_line'] - nxt} prompts between the replayed one and the correction."
        else:
            case["recorded_end_line"] = nxt
            case["preference_span"] = r["quote"]
            case["note"] = (f"Unprompted statement: {r['manner']} State = commit {r['sha']} (last HEAD the transcript "
                            f"shows, no file change after it before the turn).")
            if old.get("prompt_override"):
                case["prompt_override"], case["ablation_removed"] = old["prompt_override"], old.get("ablation_removed")
                if old.get("ablation_hand_edited"):
                    case["ablation_hand_edited"] = True
            else:
                msg = text_of(lines[r["cut_line"]])
                out = judge(ABLATE_PROMPT.format(dim=r["dim"], side=r["side"], manner=r["manner"], span=r["quote"],
                                                 message=msg), ABLATE_SCHEMA, a.model, a.cc_bin)["verdict"]
                case["prompt_override"], case["ablation_removed"] = out["ablated"], out["removed"]
            case["ablated_variant"] = True
        path.write_text(json.dumps(case, indent=1, ensure_ascii=False))
        print(cid, r["sha"], case["container_cwd"])


if __name__ == "__main__":
    main()
