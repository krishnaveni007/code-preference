#!/usr/bin/env python3
"""Method 1b: rewrite each preference-related message as one or more short, general
preference statements, stripped of project detail. Output feeds the same clustering as 1a.

  python3 discovery/method1/summarize.py --limit 20            # pilot, prints results
  python3 discovery/method1/summarize.py --all                 # full run, resumable JSONL
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import time
from pathlib import Path
from typing import Any

import litellm
from dotenv import load_dotenv

litellm.suppress_debug_info = True
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")
API_BASE = os.getenv("LITELLM_API_BASE", "https://ai-gateway.andrew.cmu.edu/")
MODEL = os.getenv("PREFERENCE_JUDGE_MODEL", "openai/wine-claude-haiku-4-5")
MAX_WORDS = 300

INSTRUCTIONS = """A developer sent this message to an AI coding agent. Rewrite the preference \
it expresses as a general rule the developer would want followed on any project, not just this one.

Rules for each statement:
- One sentence, at most 20 words, starting with "Wants the agent to ..." or "Asks the agent to ...". \
State only what the developer asked for. Do not invent an alternative they rejected unless they \
named it themselves.
- No names of files, functions, variables, libraries, frameworks, products, commands, or people. \
Use generic words instead: "the config file", "a helper function", "an external library".
- State the disposition, not the task. "Wants the login bug fixed" is a task; "Wants bugs fixed \
one at a time with confirmation in between" is a disposition.
- Specific enough that two reasonable developers could disagree with it. "Wants good code" is \
too vague to keep.

Most messages carry one preference; write up to three if there are clearly several. If the \
message carries no preference that would transfer to another project, return an empty list."""

SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {"statements": {"type": "array", "items": {"type": "string"}}},
    "required": ["statements"],
}


def summarize(text: str, *, model: str = MODEL, max_retries: int = 3) -> list[str]:
    words = str(text).split()
    payload = " ".join(words[:MAX_WORDS]) + (" ...[TRUNCATED]" if len(words) > MAX_WORDS else "")
    last: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = litellm.completion(
                api_key=os.environ["LITELLM_API_KEY"], base_url=API_BASE, model=model,
                messages=[{"role": "system", "content": INSTRUCTIONS},
                          {"role": "user", "content": "<message>\n" + payload + "\n</message>"}],
                response_format={"type": "json_schema", "json_schema": {
                    "name": "preference_statements", "strict": True, "schema": SCHEMA}},
            )
            return [s.strip() for s in json.loads(response.choices[0].message.content)["statements"]]
        except Exception as error:  # noqa: BLE001
            last = error
            if attempt + 1 < max_retries:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"summarize failed: {last}") from last


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positives", type=Path,
                        default=Path("outputs/discovery/method1/gate/positives_dedup.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("outputs/discovery/method1/final/statements.jsonl"))
    parser.add_argument("--model", default=MODEL)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--limit", type=int, default=20)
    mode.add_argument("--all", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.positives.open()]
    if not args.all:
        import random
        random.Random(args.seed).shuffle(rows)
        for r in rows[:args.limit]:
            statements = summarize(r["content"], model=args.model)
            print(f"--- ({r['shipped_pushback']}, {r['word_count']}w) "
                  f"{' '.join(r['content'].split()[:40])[:260]}")
            for s in statements:
                print(f"    -> {s}")
            if not statements:
                print("    -> (none)")
            print()
        return

    done: set[tuple[str, int]] = set()
    if args.out.exists():
        for line in args.out.open():
            try:
                rec = json.loads(line)
                if rec.get("error", "") == "":
                    done.add((rec["session_id"], int(rec["turn_number"])))
            except (KeyError, ValueError):
                continue
    todo = [r for r in rows if (r["session_id"], int(r["turn_number"])) not in done]
    print(f"{len(rows)} messages, {len(done)} done, {len(todo)} to do")

    def work(r: dict[str, Any]) -> dict[str, Any]:
        rec = {k: r[k] for k in ("session_id", "user_id", "turn_number", "shipped_pushback", "word_count")}
        rec.update({"model": args.model, "statements": [], "error": ""})
        try:
            rec["statements"] = summarize(r["content"], model=args.model)
        except Exception as error:  # noqa: BLE001
            rec["error"] = str(error)[:300]
        return rec

    args.out.parent.mkdir(parents=True, exist_ok=True)
    batch = args.workers * 8
    n_err = 0
    with args.out.open("a") as handle, cf.ThreadPoolExecutor(args.workers) as pool:
        for start in range(0, len(todo), batch):
            for rec in pool.map(work, todo[start:start + batch]):
                handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_err += rec["error"] != ""
            handle.flush()
            print(f"{min(start + batch, len(todo))}/{len(todo)}  err={n_err}", flush=True)
    print(f"done -> {args.out}")


if __name__ == "__main__":
    main()
