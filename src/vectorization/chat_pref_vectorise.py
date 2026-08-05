"""
CodePref-Bench: CHAT preference vector pipeline (separate from code scoring).

Only USER turns are scored - the developer's own words are the preference
signal, not the agent's unprompted choices. Each user turn is scored using
a WINDOWED cumulative_context (last 4 turns, built in the extraction stage)
rather than full conversation history from turn 0 - still includes the
preceding assistant turns so a correction has the thing it's correcting
available, just bounded instead of growing across the whole session.

Two-stage, ternary, per turn (see prior discussion):
  STAGE 1: screen which of the 14 axes have real evidence in this turn's
           windowed context.
  STAGE 2: for each triggered axis, one separate call -> direction (+1/-1)
           + one-sentence evidence.

Then an explicit, auditable AGGREGATION step turns per-turn scores into one
session-level chat vector per axis:
  - value    = sign from the most recent triggering turn, preferring
               pushback-type turns over plain turns when both exist
               (pushback is contrastive evidence, plain requests are
               assertive evidence - contrastive wins when available)
  - support  = how many turns triggered this axis at all
  - revised  = True if an earlier triggering turn disagreed in sign with
               the final value used
  - confidence = 'high' (pushback-backed) / 'medium' (plain-turn-backed
               only) / 'none' (never triggered)

Outputs:
  chat_turn_vectors.csv    -> raw per-turn ternary scores + evidence
  chat_session_vectors.csv -> aggregated per-session chat vector, with
                               confidence/support/revised metadata per axis

Requires: pip install openai pandas
"""

import json
import os
import time

import pandas as pd
from openai import OpenAI

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


def screen_triggered_rubrics(text: str, log_context: dict) -> tuple:
    """Returns (triggered_ids, call_failed)."""
    system_prompt = f"""You are screening a developer's conversation with an AI coding agent to \
determine which preference axes have actual evidence present in it — not guessing at all 14, only \
flagging the ones with real signal. A turn doesn't need to be a correction to count: a developer \
directly asking for something ("add tests for this") is just as valid evidence as a correction.

Axes (with example utterances that would trigger each pole):
{_rubric_block()}

Respond with ONLY a JSON object: {{"triggered": ["R01", "R05", ...]}}. If nothing is triggered, \
return {{"triggered": []}}. No other text."""
    result = _call_json(system_prompt, text, {**log_context, "stage": "screen", "rubric_id": None})
    if result.get("_call_failed"):
        return [], True
    triggered = result.get("triggered", [])
    return [rid for rid in triggered if rid in ALL_IDS], False


def label_direction(text: str, rubric_id: str, log_context: dict) -> dict:
    r = RUBRIC_BY_ID[rubric_id]
    system_prompt = f"""You are judging a single preference axis in this conversation.

Axis: {r['name']}
+1 (upper pole) = {r['upper']}
  example: {r['upper_ex']}
-1 (lower pole) = {r['lower']}
  example: {r['lower_ex']}

Based only on this conversation, does the evidence lean toward +1 or -1? Respond with ONLY a JSON \
object: {{"direction": 1 or -1, "evidence": "short quote or paraphrase, one sentence"}}"""
    result = _call_json(system_prompt, text, {**log_context, "stage": "direction", "rubric_id": rubric_id})
    if result.get("_call_failed"):
        return {"direction": None, "evidence": "", "call_failed": True}
    direction = result.get("direction")
    if direction not in (1, -1, 1.0, -1.0):
        direction = None
    return {"direction": direction, "evidence": result.get("evidence", ""), "call_failed": False}


def score_turn(text: str, log_context: dict) -> tuple:
    """Returns (scores_dict, llm_call_failed) - llm_call_failed is True if
    ANY call for this turn (screening or any triggered axis's direction
    call) exhausted its retries. That turn's zeros may be real "no signal"
    OR may be masking a dropped call - this flag tells you which."""
    triggered, screen_failed = screen_triggered_rubrics(text, log_context)
    any_failed = screen_failed
    out = {rid: {"score": 0, "evidence": ""} for rid in ALL_IDS}
    for rid in triggered:
        labeled = label_direction(text, rid, log_context)
        if labeled.get("call_failed"):
            any_failed = True
        if labeled["direction"] is not None:
            out[rid] = {"score": int(labeled["direction"]), "evidence": labeled["evidence"]}
    return out, any_failed


def aggregate_subsession(turn_rows: pd.DataFrame) -> dict:
    """turn_rows: all scored user turns within ONE subsession (i.e. one
    checkpoint_window - the stretch of turns between t0/previous commit and
    the next commit), with score_R01.. and evidence_R01.. columns plus
    prompt_pushback.

    Per axis:
      score      = most recent nonzero score in this subsession (+1/-1/0),
                   preferring a pushback-type turn as the source when one
                   triggered this axis (contrastive evidence > assertive)
      high_turns = count of turns in this subsession scored +1 on this axis
      low_turns  = count of turns in this subsession scored -1 on this axis
      support    = high_turns + low_turns (total turns that triggered it)
      revised    = True if this subsession has BOTH +1 and -1 turns for
                   this axis (i.e. the developer's stance flipped within
                   this single subsession, not just across the whole session)
    """
    result = {}
    for rid in ALL_IDS:
        col = f"score_{rid}"
        high_turns = int((turn_rows[col] == 1).sum())
        low_turns = int((turn_rows[col] == -1).sum())
        support = high_turns + low_turns

        if support == 0:
            result[f"score_{rid}"] = 0
            result[f"high_turns_{rid}"] = 0
            result[f"low_turns_{rid}"] = 0
            result[f"support_{rid}"] = 0
            result[f"revised_{rid}"] = False
            result[f"evidence_{rid}"] = ""
            continue

        nonzero = turn_rows[turn_rows[col] != 0]
        pushback_rows = nonzero[nonzero["prompt_pushback"].isin(PUSHBACK_TYPES)]
        source = pushback_rows if not pushback_rows.empty else nonzero
        last = source.sort_values("turn_number").iloc[-1]

        result[f"score_{rid}"] = int(last[col])
        result[f"high_turns_{rid}"] = high_turns
        result[f"low_turns_{rid}"] = low_turns
        result[f"support_{rid}"] = support
        result[f"revised_{rid}"] = high_turns > 0 and low_turns > 0
        result[f"evidence_{rid}"] = last[f"evidence_{rid}"]
    return result


def main():
    turns = pd.read_csv("codepref_pipeline_export/turns.csv")

    turn_rows = []
    for sid, group in turns.groupby("session_id"):
        group = group.sort_values("turn_number")
        # Only score user turns - the developer's own words are the
        # preference signal. Assistant turns still contribute to the
        # windowed cumulative_context (built in the extraction stage) so a
        # user's reaction to a specific agent response has that response
        # available, but we never ask "what does the developer prefer"
        # immediately after an assistant turn with nothing new from the
        # developer yet.
        user_turns = group[group["role"] == "user"]
        n = len(user_turns)
        for i, (_, row) in enumerate(user_turns.iterrows()):
            print(f"[{sid[:8]}] user turn {i+1}/{n}")
            log_context = {
                "user_id": row["user_id"], "session_id": sid,
                "turn_number": int(row["turn_number"]),
                "conversation_turn_number": row["conversation_turn_number"],
                "checkpoint_window": int(row["checkpoint_window"]),
            }
            scored, llm_call_failed = score_turn(row["cumulative_context"], log_context)
            flat = {
                "user_id": row["user_id"], "session_id": sid,
                "turn_number": row["turn_number"],
                "conversation_turn_number": row["conversation_turn_number"],
                "checkpoint_window": row["checkpoint_window"],
                "prompt_intent": row.get("prompt_intent"),
                "prompt_pushback": row.get("prompt_pushback"),
                "llm_call_failed": llm_call_failed,
            }
            if llm_call_failed:
                print(f"    WARNING: one or more calls for this turn exhausted retries - "
                      f"scores below may be incomplete, not confirmed zero")
            for rid, v in scored.items():
                flat[f"score_{rid}"] = v["score"]
                flat[f"evidence_{rid}"] = v["evidence"]
            turn_rows.append(flat)

    turn_df = pd.DataFrame(turn_rows)
    turn_df.to_csv("chat_turn_vectors.csv", index=False)
    print(f"\nWrote chat_turn_vectors.csv ({len(turn_df)} rows)")

    subsession_rows = []
    for (sid, window), group in turn_df.groupby(["session_id", "checkpoint_window"]):
        agg = aggregate_subsession(group)
        agg["user_id"] = group.iloc[0]["user_id"]
        agg["session_id"] = sid
        agg["checkpoint_window"] = window
        agg["n_turns"] = len(group)
        agg["any_llm_call_failed"] = bool(group["llm_call_failed"].any())
        subsession_rows.append(agg)

    subsession_df = pd.DataFrame(subsession_rows)
    # reorder so identifying columns come first
    id_cols = ["user_id", "session_id", "checkpoint_window", "n_turns"]
    subsession_df = subsession_df[id_cols + [c for c in subsession_df.columns if c not in id_cols]]
    subsession_df = subsession_df.sort_values(["session_id", "checkpoint_window"])
    subsession_df.to_csv("chat_subsession_vectors.csv", index=False)
    print(f"Wrote chat_subsession_vectors.csv ({len(subsession_df)} rows)")


if __name__ == "__main__":
    main()
    