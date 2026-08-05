"""
CodePref-Bench: CHAT preference vector pipeline (separate from code scoring).

v2 CHANGES from the subsession-based version:
  - Aggregation is at the SESSION level, not per checkpoint_window. We are
    deliberately not using the subsession/checkpoint logic for this pass.
  - Context window is 5 conversational turns total (mixed user+assistant,
    ending with the current user turn), not 4.
  - TWO aggregation methods are computed and reported side by side, not one:
      "recent"   : value = sign of the most recent turn that triggered
                   this axis in the session. Ties can't occur (there's
                   always exactly one most-recent triggering turn).
      "maxtie"   : value = the majority sign across all turns that
                   triggered this axis (count of +1 turns vs count of -1
                   turns); if tied, break by recency (same rule as above,
                   restricted to turns carrying the tied-for-first sign).
    Neither method gives special weight to pushback-type turns; both are
    pure functions of the ternary per-turn scores and turn order. (The
    original per-subsession aggregation preferred pushback-sourced turns
    as the tiebreak signal -- that rule is NOT carried over here per the
    updated plan. Re-add it if that's still wanted.)

Only USER turns are scored - the developer's own words are the preference
signal, not the agent's unprompted choices. Each user turn is scored using
a WINDOWED cumulative_context (last 5 conversational turns, mixed roles,
ending with the current turn) - still includes the preceding assistant
turns so a correction has the thing it's correcting available, just
bounded instead of growing across the whole session.

Two-stage, ternary, per turn (see prior discussion):
  STAGE 1: screen which of the 14 axes have real evidence in this turn's
           windowed context.
  STAGE 2: for each triggered axis, one separate call -> direction (+1/-1)
           + one-sentence evidence.

Outputs:
  chat_turn_vectors.csv    -> raw per-turn ternary scores + evidence
  chat_session_vectors.csv -> aggregated per-session chat vector, with
                               BOTH aggregation methods, support counts,
                               and a `revised` flag per axis

Requires: pip install openai pandas
"""

import argparse
import json
import os
import time

import pandas as pd
import numpy as np
from openai import OpenAI
from tqdm import tqdm

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
MODEL = "gpt-5.4-mini"

# GPT-5.x reasoning-tier models reject an explicit temperature and only
# support their default (1). Classic models (gpt-4o and earlier) support
# temperature=0 for more deterministic output. Detected by prefix so
# switching MODEL doesn't require touching the API call below.
def _supports_temperature(model: str) -> bool:
    return not model.startswith("gpt-5")

PUSHBACK_TYPES = {"correction", "rejection", "failure_report", "pacing_complaint",
                   "takeover", "requirement_change"}

# ---------------------------------------------------------------------
# Rubric definitions + one calibration example per pole. These examples
# are embedded directly in the prompts as few-shot anchors.
# ---------------------------------------------------------------------

RUBRICS = [
    {"id": "R01", "name": "Solution Scope",
     "upper": "prefers proactive improvements, related refactors, and anticipating future needs beyond the explicit request",
     "lower": "prefers minimal, localized changes strictly limited to what was explicitly asked",
     "upper_ex": "\"While you're in there, can you also clean up the other functions in this file with the same issue?\"",
     "lower_ex": "\"Just fix this one bug, don't touch anything else.\""},
    {"id": "R02", "name": "Abstraction Level",
     "upper": "prefers reusable abstractions, helper functions, generic components, and DRY design",
     "lower": "prefers concrete, task-specific implementations without generalization",
     "upper_ex": "\"Can you extract this into a reusable helper so we don't repeat it in three places?\"",
     "lower_ex": "\"Don't bother making this generic, just hardcode it for this one case.\""},
    {"id": "R03", "name": "Dependency Posture",
     "upper": "prefers leveraging mature third-party packages for productivity and maintainability",
     "lower": "prefers minimizing dependencies and using the standard library or custom code",
     "upper_ex": "\"Let's just use lodash for this instead of writing our own deep-clone.\"",
     "lower_ex": "\"I'd rather not add a new dependency for this, can we do it with plain JS?\""},
    {"id": "R04", "name": "Correctness Guarantees",
     "upper": "favors strong type systems, assertions, contracts, exhaustive checks, and static analysis",
     "lower": "prioritizes rapid implementation with lightweight validation",
     "upper_ex": "\"Add proper type annotations and validate this at the boundary.\"",
     "lower_ex": "\"Don't worry about strict typing here, let's just get it working.\""},
    {"id": "R05", "name": "Robustness Philosophy",
     "upper": "prefers defensive programming, input validation, graceful degradation, retries, and comprehensive error handling",
     "lower": "prioritizes fail-fast behavior and the common execution path",
     "upper_ex": "\"What happens if the API call fails? Add a retry and fallback.\"",
     "lower_ex": "\"Don't worry about error handling here, just assume the happy path.\""},
    {"id": "R06", "name": "Testing Rigor",
     "upper": "expects comprehensive unit, integration, or property-based tests validating functionality and edge cases",
     "lower": "accepts minimal or manual testing alongside code changes",
     "upper_ex": "\"Add unit tests covering the edge cases before we move on.\"",
     "lower_ex": "\"No need for tests on this, it's a quick throwaway script.\""},
    {"id": "R07", "name": "Performance Sensitivity",
     "upper": "emphasizes algorithmic efficiency, memory optimization, and performance-conscious design",
     "lower": "prioritizes readability and maintainability over runtime performance",
     "upper_ex": "\"This loop runs on every request, can you optimize away the O(n^2) behavior?\"",
     "lower_ex": "\"Performance isn't a concern here, just make it readable.\""},
    {"id": "R08", "name": "Security Posture",
     "upper": "consistently favors secure defaults, input sanitization, auth checks, secret management, and adversarial protection",
     "lower": "assumes trusted inputs and prioritizes functionality over proactive security",
     "upper_ex": "\"Make sure user input is sanitized before it hits the query.\"",
     "lower_ex": "\"This is internal-only, don't worry about auth on this endpoint for now.\""},
    {"id": "R09", "name": "Refactoring Aggressiveness",
     "upper": "favors broader architectural improvements, cleanup, and modernization when opportunities arise",
     "lower": "favors localized edits that minimize disruption to surrounding code",
     "upper_ex": "\"Since we're touching this file, let's also restructure the module layout.\"",
     "lower_ex": "\"Please don't restructure anything else, just make the smallest change possible.\""},
    {"id": "R10", "name": "Documentation Richness",
     "upper": "expects detailed comments, docstrings, API docs, and README updates",
     "lower": "relies on self-documenting code with minimal comments",
     "upper_ex": "\"Add a docstring and update the README to reflect this.\"",
     "lower_ex": "\"No need to document this, the code speaks for itself.\""},
    {"id": "R11", "name": "Explanation Verbosity",
     "upper": "values detailed rationales, implementation decisions, design tradeoffs, and educational explanations",
     "lower": "prefers concise, code-focused outputs with little discussion",
     "upper_ex": "\"Walk me through why you chose this approach over the alternatives.\"",
     "lower_ex": "\"Just show me the code, skip the explanation.\""},
    {"id": "R12", "name": "Interaction Autonomy",
     "upper": "prefers the agent to proactively execute changes with minimal user intervention",
     "lower": "prefers the agent to seek confirmation before making significant decisions or assumptions",
     "upper_ex": "\"Just go ahead and make the change, you don't need to check with me first.\"",
     "lower_ex": "\"Before you touch anything, check with me on the approach.\""},
    {"id": "R13", "name": "Code Clarity",
     "upper": "prefers explicit, self-explanatory naming and structure, favoring readability over brevity",
     "lower": "accepts terse, implicit code where meaning must be inferred from context",
     "upper_ex": "\"Use descriptive variable names, not single letters.\"",
     "lower_ex": "\"It's fine to keep it terse, I can follow the shorthand.\""},
    {"id": "R14", "name": "Code Conciseness",
     "upper": "prefers verbose, spelled-out implementations even at the cost of length",
     "lower": "prefers terse, compact code that minimizes line count",
     "upper_ex": "\"Spell it out fully rather than using a clever one-liner.\"",
     "lower_ex": "\"Can you condense this into fewer lines?\""},
]
RUBRIC_BY_ID = {r["id"]: r for r in RUBRICS}
ALL_IDS = [r["id"] for r in RUBRICS]

# ---------------------------------------------------------------------
# Raw LLM call logging. Every attempt (successful or not) is appended
# here BEFORE parsing, so you have a full audit trail for later
# LLM-as-judge re-verification, disagreement debugging, or swapping
# models and comparing raw outputs - not just the final flattened score.
# ---------------------------------------------------------------------

RAW_LOG_PATH = "chat_llm_raw_log.jsonl"


def _log_raw(record: dict):
    with open(RAW_LOG_PATH, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def _call_json(system_prompt: str, user_content: str, log_context: dict, max_retries: int = 3) -> dict:
    for attempt in range(max_retries):
        record = {
            "timestamp": time.time(),
            "model": MODEL,
            "attempt": attempt + 1,
            **log_context,
        }
        try:
            kwargs = dict(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content[:40000]},
                ],
                response_format={"type": "json_object"},
            )
            if _supports_temperature(MODEL):
                kwargs["temperature"] = 0
            resp = client.chat.completions.create(**kwargs)
            raw_text = resp.choices[0].message.content
            record["raw_response"] = raw_text
            parsed = json.loads(raw_text)
            record["parsed_ok"] = True
            _log_raw(record)
            return parsed
        except Exception as e:
            record["parsed_ok"] = False
            record["error"] = str(e)
            _log_raw(record)
            print(f"    retry {attempt+1}/{max_retries} after error: {e}")
            time.sleep(2 ** attempt)
    # All retries exhausted - return a sentinel distinct from a legitimate
    # "no signal" response, so callers can flag this turn rather than
    # silently recording it as all-zero.
    return {"_call_failed": True}


def _rubric_block():
    lines = []
    for r in RUBRICS:
        lines.append(
            f"{r['id']}: {r['name']}\n"
            f"  +1 example: {r['upper_ex']}\n"
            f"  -1 example: {r['lower_ex']}"
        )
    return "\n".join(lines)


MAX_TRIGGERED_AXES = 4  # hard cap: force the model to prioritize strongest
                        # evidence rather than flagging half the rubric on
                        # thin signal. Enforced both in the prompt and,
                        # defensively, in code in case the model ignores it.


def screen_triggered_rubrics(text: str, log_context: dict) -> tuple:
    """Returns (triggered_ids, analysis_text, call_failed).

    The model is required to write a short analysis of what the turn is
    actually expressing BEFORE listing triggered axes, and is capped at
    MAX_TRIGGERED_AXES. Both are aimed at the same failure mode: a
    screening call that pattern-matches keywords and over-triggers axes on
    weak or inferred evidence, which then inflates `support` counts and
    contaminates the aggregated preference vector with noise."""
    system_prompt = f"""You are screening a developer's conversation with an AI coding agent to \
determine which preference axes have STRONG, EXPLICIT evidence present in it. Do not guess at all \
14, and do not flag an axis on vague or inferred signal - only flag axes where a second reader would \
agree without needing your reasoning explained to them. A turn doesn't need to be a correction to \
count: a developer directly asking for something ("add tests for this") is just as valid evidence as \
a correction - but it must be a direct, explicit ask, not a stretch.

The input below is split into two labeled blocks: PRIOR CONTEXT and CURRENT TURN. Score ONLY what \
the CURRENT TURN itself expresses. PRIOR CONTEXT exists so a correction or reference has its \
antecedent available to you - it is background, not evidence. If something was only said in PRIOR \
CONTEXT (by either the developer or the agent) and the CURRENT TURN does not itself restate, \
confirm, or react to it, that is not grounds to trigger an axis.

Before answering, briefly analyze what the CURRENT TURN is actually asking for or reacting to. Then \
decide which axes, if any, have genuinely clear evidence in the CURRENT TURN specifically.

Axes (with example utterances that would trigger each pole):
{_rubric_block()}

Return AT MOST {MAX_TRIGGERED_AXES} axes. If more seem to apply, keep only the \
{MAX_TRIGGERED_AXES} with the strongest, most explicit evidence and drop the rest - do not pad the \
list to reach the cap, and do not include an axis just because it's plausible.

Respond with ONLY a JSON object: {{"analysis": "1-2 sentence analysis of what the CURRENT TURN \
actually expresses", "triggered": ["R01", "R05", ...]}}. If nothing is triggered, return \
{{"analysis": "...", "triggered": []}}. No other text."""
    result = _call_json(system_prompt, text, {**log_context, "stage": "screen", "rubric_id": None})
    if result.get("_call_failed"):
        return [], "", True
    triggered = [rid for rid in result.get("triggered", []) if rid in ALL_IDS]
    if len(triggered) > MAX_TRIGGERED_AXES:
        tqdm.write(f"    WARNING: screen returned {len(triggered)} axes "
             f"(cap is {MAX_TRIGGERED_AXES}); truncating, model ignored the cap")
        triggered = triggered[:MAX_TRIGGERED_AXES]
    return triggered, result.get("analysis", ""), False


def label_direction(text: str, rubric_id: str, log_context: dict) -> dict:
    r = RUBRIC_BY_ID[rubric_id]
    system_prompt = f"""You are judging a single preference axis, based ONLY on the CURRENT TURN in \
the input below (the input is split into PRIOR CONTEXT and CURRENT TURN blocks). PRIOR CONTEXT is \
background so a correction or reference has its antecedent available - it is not itself evidence. \
Your evidence quote must come from the CURRENT TURN, not from PRIOR CONTEXT, unless the CURRENT \
TURN explicitly restates or reacts to something said earlier.

Axis: {r['name']}
+1 (upper pole) = {r['upper']}
  example: {r['upper_ex']}
-1 (lower pole) = {r['lower']}
  example: {r['lower_ex']}

Based on the CURRENT TURN, does the evidence lean toward +1 or -1? Respond with ONLY a JSON \
object: {{"direction": 1 or -1, "evidence": "short quote or paraphrase from the CURRENT TURN, one sentence"}}"""
    result = _call_json(system_prompt, text, {**log_context, "stage": "direction", "rubric_id": rubric_id})
    if result.get("_call_failed"):
        return {"direction": None, "evidence": "", "call_failed": True}
    direction = result.get("direction")
    if direction not in (1, -1, 1.0, -1.0):
        direction = None
    return {"direction": direction, "evidence": result.get("evidence", ""), "call_failed": False}


def score_turn(text: str, log_context: dict) -> tuple:
    """Returns (scores_dict, screen_analysis, llm_call_failed) -
    llm_call_failed is True if ANY call for this turn (screening or any
    triggered axis's direction call) exhausted its retries. That turn's
    zeros may be real "no signal" OR may be masking a dropped call - this
    flag tells you which."""
    triggered, screen_analysis, screen_failed = screen_triggered_rubrics(text, log_context)
    any_failed = screen_failed
    out = {rid: {"score": 0, "evidence": ""} for rid in ALL_IDS}
    for rid in triggered:
        labeled = label_direction(text, rid, log_context)
        if labeled.get("call_failed"):
            any_failed = True
        if labeled["direction"] is not None:
            out[rid] = {"score": int(labeled["direction"]), "evidence": labeled["evidence"]}
    return out, screen_analysis, any_failed


def _aggregate_recent(nonzero: pd.DataFrame, col: str, evidence_col: str) -> tuple:
    """Method 'recent': value = sign of the most recent triggering turn.
    Returns (value, evidence_text)."""
    last = nonzero.sort_values("turn_number").iloc[-1]
    return int(last[col]), last[evidence_col]


def _aggregate_maxtie(nonzero: pd.DataFrame, col: str, evidence_col: str) -> tuple:
    """Method 'maxtie': value = majority sign by count; ties broken by
    recency, restricted to turns carrying the tied-for-first sign (so the
    reported evidence always genuinely supports the reported value).
    Returns (value, evidence_text)."""
    pos = int((nonzero[col] == 1).sum())
    neg = int((nonzero[col] == -1).sum())
    if pos > neg:
        winning_sign = 1
    elif neg > pos:
        winning_sign = -1
    else:
        # tie -> whichever sign the single most recent triggering turn has
        last = nonzero.sort_values("turn_number").iloc[-1]
        winning_sign = int(last[col])
    supporting = nonzero[nonzero[col] == winning_sign]
    most_recent_supporting = supporting.sort_values("turn_number").iloc[-1]
    return winning_sign, most_recent_supporting[evidence_col]


def _aggregate_mean(nonzero: pd.DataFrame, col: str) -> tuple:
    """Method 'mean': the raw average of every firing instance's score,
    e.g. 5 turns at +1 and 5 turns at -1 -> (5*1 + 5*-1)/10 = 0.0.
    Continuous in [-1, 1], not ternary -- unlike 'recent' and 'maxtie'
    this collapses the ternary structure the rest of the pipeline uses,
    so treat it as a supplementary summary statistic, not a drop-in
    replacement. There's no single turn this value "comes from" (it's an
    average across all of them), so there's no evidence quote to attach
    to it the way recent/maxtie have one.
    Returns (value, n_pos, n_neg)."""
    pos = int((nonzero[col] == 1).sum())
    neg = int((nonzero[col] == -1).sum())
    return round((pos - neg) / (pos + neg), 4), pos, neg


def aggregate_session(turn_rows: pd.DataFrame) -> dict:
    """turn_rows: all scored user turns within ONE session, with
    score_R01.. and evidence_R01.. columns.

    Per axis, for EACH of three methods:
      score_recent_<rid> / score_maxtie_<rid>  = ternary, with an
        evidence_<method>_<rid> quote from whichever turn produced it
      score_mean_<rid>   = continuous in [-1, 1], the raw average across
        every firing instance (no single evidence quote -- see
        _aggregate_mean)
    Shared across all methods (doesn't depend on which one you use):
      high_turns_<rid> / low_turns_<rid> / support_<rid> = raw counts
      revised_<rid> = True if the session has BOTH +1 and -1 turns for
                      this axis (the developer's stance appeared to flip
                      somewhere within the session)
    """
    result = {}
    for rid in ALL_IDS:
        col = f"score_{rid}"
        ev_col = f"evidence_{rid}"
        high_turns = int((turn_rows[col] == 1).sum())
        low_turns = int((turn_rows[col] == -1).sum())
        support = high_turns + low_turns

        result[f"high_turns_{rid}"] = high_turns
        result[f"low_turns_{rid}"] = low_turns
        result[f"support_{rid}"] = support
        result[f"revised_{rid}"] = high_turns > 0 and low_turns > 0

        if support == 0:
            result[f"score_recent_{rid}"] = 0
            result[f"evidence_recent_{rid}"] = ""
            result[f"score_maxtie_{rid}"] = 0
            result[f"evidence_maxtie_{rid}"] = ""
            result[f"score_mean_{rid}"] = 0.0
            continue

        nonzero = turn_rows[turn_rows[col] != 0]
        v_recent, e_recent = _aggregate_recent(nonzero, col, ev_col)
        v_maxtie, e_maxtie = _aggregate_maxtie(nonzero, col, ev_col)
        v_mean, _, _ = _aggregate_mean(nonzero, col)
        result[f"score_recent_{rid}"] = v_recent
        result[f"evidence_recent_{rid}"] = e_recent
        result[f"score_maxtie_{rid}"] = v_maxtie
        result[f"evidence_maxtie_{rid}"] = e_maxtie
        result[f"score_mean_{rid}"] = v_mean
    return result


def build_context_windows(conv: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Build cumulative_context for every user_prompt row: the last
    `window` conversational turns (mixed user+assistant, is_conversational
    == True), ending with the current turn itself. So a window of 5 means
    up to 4 preceding turns plus the current one.

    The window text is EXPLICITLY split into a "PRIOR CONTEXT" block and
    a "CURRENT TURN" block, rather than one flat concatenation. This
    fixes a real failure mode found in case-study review: with an
    undifferentiated window, the judge sometimes scored a turn based on
    something an earlier (often the agent's) turn said, not anything the
    current turn itself expressed. Structurally marking which part is
    "the thing to score" vs. "background so a correction has its
    antecedent" removes the ambiguity the prompt alone couldn't fully
    close.

    conv must already be filtered to ONE session and sorted by
    turn_number (ascending) before calling this.
    """
    conv_only = conv[conv["is_conversational"]].sort_values("turn_number").reset_index(drop=True)
    texts = conv_only["content"].fillna("").tolist()
    roles = conv_only["role"].fillna("").tolist()
    turn_numbers = conv_only["turn_number"].tolist()

    contexts = {}
    for i, tn in enumerate(turn_numbers):
        lo = max(0, i - window + 1)
        prior_chunk = [f"[{roles[j]}] {texts[j]}" for j in range(lo, i)]
        current_line = f"[{roles[i]}] {texts[i]}"
        prior_block = "\n\n".join(prior_chunk) if prior_chunk else "(none -- this is the first turn in the available window)"
        contexts[tn] = (
            "=== PRIOR CONTEXT (background only -- do NOT score based on "
            "this alone; it exists so a correction or reference has its "
            "antecedent available) ===\n"
            f"{prior_block}\n\n"
            "=== CURRENT TURN (score ONLY what this turn itself expresses) ===\n"
            f"{current_line}"
        )
    return conv_only.assign(
        cumulative_context=conv_only["turn_number"].map(contexts))


def select_pilot_users(selected: pd.DataFrame, n: int, seed: int = 0) -> list:
    """Pick n users for a pilot run, preferring at least one control-arm
    user (if n >= 2) so the pilot can sanity-check contrast between arms,
    not just validate the pipeline runs without erroring."""
    rng = np.random.default_rng(seed)
    users = selected[["user_id", "arm"]].drop_duplicates()
    hs = users[users["arm"] == "high_signal"]["user_id"].tolist()
    ctrl = users[users["arm"] == "random_control"]["user_id"].tolist()
    picked = []
    remaining = n
    if remaining >= 2 and ctrl:
        picked.append(rng.choice(ctrl))
        remaining -= 1
    if hs and remaining > 0:
        picked += list(rng.choice(hs, size=min(remaining, len(hs)), replace=False))
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True,
                    help="dir with sessions.parquet / conversations.parquet")
    ap.add_argument("--selected-sessions", default="selected_sessions.csv",
                    help="output of sample_users_sessions.py")
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--pilot-n-users", type=int, default=None,
                    help="if set, restrict to this many users (prefers >=1 "
                         "control-arm user) before scoring -- use this for "
                         "a cheap sanity-check run before the full pass")
    ap.add_argument("--pilot-seed", type=int, default=0)
    ap.add_argument("--user-id", default=None,
                    help="restrict to exactly this one user (e.g. for a "
                         "case study). Takes precedence over --pilot-n-users.")
    ap.add_argument("--exclude-users", default=None,
                    help="comma-separated user_ids to SKIP -- for resuming "
                         "a run without re-scoring (and re-paying for) users "
                         "already scored in an earlier pass, e.g. a pilot.")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    selected = pd.read_csv(args.selected_sessions)

    if args.exclude_users:
        exclude = [u.strip() for u in args.exclude_users.split(",")]
        before_users = selected["user_id"].nunique()
        before_sessions = len(selected)
        selected = selected[~selected["user_id"].isin(exclude)]
        print(f"EXCLUDED {len(exclude)} already-scored users: {exclude}")
        print(f"  users: {before_users} -> {selected['user_id'].nunique()}, "
             f"sessions: {before_sessions} -> {len(selected)}")
        missing = set(exclude) - set(pd.read_csv(args.selected_sessions)["user_id"])
        if missing:
            print(f"  NOTE: {missing} were not found in the selection file at "
                 f"all (typo, or already outside this sample) -- nothing to "
                 f"exclude for those.")

    if args.user_id:
        selected = selected[selected["user_id"] == args.user_id]
        if selected.empty:
            raise SystemExit(f"--user-id {args.user_id!r} not found in "
                             f"{args.selected_sessions}")
        print(f"SINGLE-USER MODE: {args.user_id} "
             f"({selected['session_id'].nunique()} sessions)")
    elif args.pilot_n_users:
        pilot_users = select_pilot_users(selected, args.pilot_n_users, args.pilot_seed)
        selected = selected[selected["user_id"].isin(pilot_users)]
        print(f"PILOT MODE: restricted to {len(pilot_users)} users: {pilot_users}")

    session_ids = set(selected["session_id"])
    print(f"scoring {len(session_ids)} sessions "
         f"({selected['arm'].value_counts().to_dict()})")

    print("loading conversations for selected sessions ...")
    conv_all = pd.read_parquet(
        os.path.join(args.data_dir, "conversations.parquet"),
        columns=["session_id", "turn_number", "role", "turn_type",
                 "content", "is_conversational", "prompt_intent",
                 "prompt_pushback"])
    conv_all = conv_all[conv_all["session_id"].isin(session_ids)]

    id_to_user = dict(zip(selected["session_id"], selected["user_id"]))
    id_to_arm = dict(zip(selected["session_id"], selected["arm"]))

    # pre-compute total user prompts across the whole run, for an
    # overall turn-level bar that doesn't reset per session
    total_prompts = int((conv_all["turn_type"] == "user_prompt").sum())
    session_groups = list(conv_all.groupby("session_id"))

    turn_rows = []
    session_bar = tqdm(session_groups, desc="Sessions", position=0,
                       unit="session")
    overall_turn_bar = tqdm(total=total_prompts, desc="Turns (overall)",
                            position=1, unit="turn")
    for sid, group in session_bar:
        windowed = build_context_windows(group, window=args.window)
        user_turns = windowed[windowed["turn_type"] == "user_prompt"]
        n = len(user_turns)
        session_bar.set_postfix_str(f"{sid[:8]} ({id_to_user.get(sid, '?')})")
        turn_bar = tqdm(total=n, desc=f"  {sid[:8]}", position=2, leave=False,
                       unit="turn")
        for i, (_, row) in enumerate(user_turns.iterrows()):
            log_context = {
                "user_id": id_to_user.get(sid), "session_id": sid,
                "turn_number": int(row["turn_number"]),
            }
            scored, screen_analysis, llm_call_failed = score_turn(
                row["cumulative_context"], log_context)
            flat = {
                "user_id": id_to_user.get(sid), "session_id": sid,
                "arm": id_to_arm.get(sid),
                "turn_number": row["turn_number"],
                "screen_analysis": screen_analysis,
                "prompt_intent": row.get("prompt_intent"),
                "prompt_pushback": row.get("prompt_pushback"),
                "llm_call_failed": llm_call_failed,
            }
            if llm_call_failed:
                tqdm.write(f"    WARNING [{sid[:8]} turn {int(row['turn_number'])}]: "
                          f"one or more calls exhausted retries - scores may be "
                          f"incomplete, not confirmed zero")
            for rid, v in scored.items():
                flat[f"score_{rid}"] = v["score"]
                flat[f"evidence_{rid}"] = v["evidence"]
            turn_rows.append(flat)
            turn_bar.update(1)
            overall_turn_bar.update(1)
        turn_bar.close()
    overall_turn_bar.close()
    session_bar.close()

    turn_df = pd.DataFrame(turn_rows)
    turn_df.to_csv(os.path.join(args.out_dir, "chat_turn_vectors.csv"), index=False)
    print(f"\nWrote chat_turn_vectors.csv ({len(turn_df)} rows)")

    session_rows = []
    for sid, group in turn_df.groupby("session_id"):
        agg = aggregate_session(group)
        agg["user_id"] = group.iloc[0]["user_id"]
        agg["session_id"] = sid
        agg["arm"] = group.iloc[0]["arm"]
        agg["n_turns"] = len(group)
        agg["any_llm_call_failed"] = bool(group["llm_call_failed"].any())
        session_rows.append(agg)

    session_df = pd.DataFrame(session_rows)
    id_cols = ["user_id", "session_id", "arm", "n_turns"]
    session_df = session_df[id_cols + [c for c in session_df.columns if c not in id_cols]]
    session_df = session_df.sort_values(["user_id", "session_id"])
    session_df.to_csv(os.path.join(args.out_dir, "chat_session_vectors.csv"), index=False)
    print(f"Wrote chat_session_vectors.csv ({len(session_df)} rows, "
         f"{session_df['user_id'].nunique()} users)")


if __name__ == "__main__":
    main()