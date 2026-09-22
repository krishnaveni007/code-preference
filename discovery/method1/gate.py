#!/usr/bin/env python3
"""Binary preference-relevance classifier for SWE-Chat user turns.

Stage 1 of the rubric-discovery pipeline. Runs over every conversational user
turn, drops harness-generated and contentless messages deterministically, then
asks a model for a single 0/1 label: does this message express a preference, or
is it about something being wrong?
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import gc
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import litellm

litellm.suppress_debug_info = True
import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

API_BASE = os.getenv("LITELLM_API_BASE", "https://ai-gateway.andrew.cmu.edu/")
DEFAULT_MODEL = os.getenv("PREFERENCE_JUDGE_MODEL", "openai/wine-claude-haiku-4-5")

MAX_WORDS = 50
MIN_WORDS = 5

# Messages the harness injected rather than the developer typing them.
LEADING_TAG = re.compile(r"^\s*<[a-zA-Z][a-zA-Z0-9_\-]{1,40}>")
HARNESS_TAG = re.compile(
    r"</?(task-notification|system_instruction|system-instruction|command-message"
    r"|command-name|command-args|local-command-std\w+|bash-input|bash-stdout"
    r"|ide_selection|ide_opened_file)>"
)
HARNESS_BLOCK = re.compile(
    r"<usage>\s*<total_tokens>"
    r"|Full transcript available at:"
    r"|^Base directory for this skill:",
    re.MULTILINE,
)
HARNESS_EXACT = re.compile(
    r"^\s*(\[Request interrupted[^\]]*\]|\[Image[^\]]*\]|tool loaded\.?)\s*$",
    re.IGNORECASE,
)
HARNESS_PREFIX = re.compile(
    r"^\s*(This session is being continued from a previous conversation"
    r"|Caveat: The messages below were generated)",
)
SLASH_COMMAND = re.compile(r"^\s*/[a-zA-Z][\w\-:]*\s*$")


def prefilter_reason(content: Any, word_count: Any) -> str | None:
    """Return why a turn is dropped before reaching the model, or None to keep."""
    text = "" if content is None else str(content)
    if not text.strip():
        return "empty"
    if LEADING_TAG.match(text):
        return "harness_tag"
    if HARNESS_TAG.search(text):
        return "harness_tag_inline"
    if HARNESS_BLOCK.search(text):
        return "harness_block"
    if HARNESS_EXACT.match(text):
        return "harness_marker"
    if HARNESS_PREFIX.match(text):
        return "context_summary"
    if SLASH_COMMAND.match(text):
        return "slash_command"
    if int(word_count or 0) < MIN_WORDS:
        return "too_short"
    return None


INSTRUCTIONS = """You are labeling single messages a developer sent to an AI coding agent. You are labeling whether this message shows any preference beyond code details, e.g., not concrete debugging messages, but indicating a more general preference on both codes and interactions. Output 1 if it's preference related message, output 0 for everything else.

Much of the input is not written by the developer: pasted tool output, review findings, CI logs, terminal dumps, and configuration or skill files injected by the agent harness. Output 0 for all of it, even when it is full of preference language."""

RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"label": {"type": "integer", "enum": [0, 1]}},
    "required": ["label"],
}


def classify(text: str, *, model: str, max_retries: int = 3) -> int:
    words = text.split()
    payload = text if len(words) <= MAX_WORDS else " ".join(words[:MAX_WORDS]) + " ...[TRUNCATED]"
    last: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = litellm.completion(
                api_key=os.environ["LITELLM_API_KEY"],
                base_url=API_BASE,
                model=model,
                messages=[
                    {"role": "system", "content": INSTRUCTIONS},
                    {"role": "user", "content": "<message>\n" + payload + "\n</message>"},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "preference_relevance",
                        "strict": True,
                        "schema": RESPONSE_SCHEMA,
                    },
                },
            )
            return int(json.loads(response.choices[0].message.content)["label"])
        except Exception as error:  # noqa: BLE001
            last = error
            if attempt + 1 < max_retries:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"classification failed: {last}") from last


def load_turns(data_dir: Path) -> pd.DataFrame:
    turns = pd.read_parquet(
        data_dir / "conversations.parquet",
        columns=[
            "session_id", "user_id", "turn_number", "turn_type",
            "word_count", "prompt_pushback", "content",
        ],
        filters=[("turn_type", "==", "user_prompt")],
    )
    turns["drop_reason"] = [
        prefilter_reason(c, w) for c, w in zip(turns["content"], turns["word_count"])
    ]
    return turns


def _rss_gb() -> float:
    with open("/proc/self/status") as handle:
        for line in handle:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1_048_576
    return float("nan")


def _load_done(path: Path) -> set[tuple[str, int]]:
    done: set[tuple[str, int]] = set()
    if not path.exists():
        return done
    for line in path.read_text().splitlines():
        try:
            rec = json.loads(line)
            if rec.get("label") in (0, 1):
                done.add((str(rec["session_id"]), int(rec["turn_number"])))
        except (KeyError, ValueError, json.JSONDecodeError):
            continue
    return done


def run_all(turns: pd.DataFrame, *, model: str, out_dir: Path, workers: int) -> None:
    """Label every kept turn; append to JSONL; skip turns already labelled."""
    out_dir.mkdir(parents=True, exist_ok=True)
    turns[["session_id", "user_id", "turn_number", "prompt_pushback", "word_count", "drop_reason"]] \
        .to_csv(out_dir / "gate_prefilter.csv", index=False)
    print(f"user turns: {len(turns)}")
    print(turns["drop_reason"].value_counts(dropna=False).to_string())
    labels_path = out_dir / "gate_labels.jsonl"
    done = _load_done(labels_path)
    todo = turns[turns["drop_reason"].isna()]
    todo = todo[[(s, int(n)) not in done for s, n in zip(todo["session_id"], todo["turn_number"])]]
    print(f"kept {int(turns['drop_reason'].isna().sum())}, already labelled {len(done)}, to do {len(todo)}")

    def work(row: dict[str, Any]) -> dict[str, Any]:
        rec = {
            "session_id": row["session_id"], "user_id": row["user_id"],
            "turn_number": int(row["turn_number"]), "shipped_pushback": row["prompt_pushback"],
            "word_count": int(row["word_count"]), "model": model, "label": None, "error": "",
        }
        try:
            rec["label"] = classify(str(row["content"]), model=model)
        except Exception as error:  # noqa: BLE001
            rec["error"] = str(error)[:300]
        return rec

    # Keep only what the workers need, then drop the full corpus from memory.
    todo = todo[["session_id", "user_id", "turn_number", "prompt_pushback", "word_count", "content"]]
    records = todo.to_dict("records")
    del turns, todo
    gc.collect()

    n_ok = n_err = 0
    batch = workers * 8
    with labels_path.open("a") as handle, cf.ThreadPoolExecutor(workers) as pool:
        for start in range(0, len(records), batch):
            chunk = records[start:start + batch]
            for rec in pool.map(work, chunk):
                handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_ok += rec["label"] is not None
                n_err += rec["label"] is None
            handle.flush()
            done_n = start + len(chunk)
            if done_n % 500 < batch:
                print(f"{done_n}/{len(records)}  ok={n_ok} err={n_err}  rss={_rss_gb():.2f}GB", flush=True)
    print(f"done: ok={n_ok} err={n_err} -> {labels_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/swechat_data"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--limit", type=int, default=10, help="sample this many kept turns")
    mode.add_argument("--all", action="store_true", help="label every kept turn (resumable)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true", help="show the sample, make no calls")
    parser.add_argument("--out", type=Path, default=None,
                        help="sample mode: JSON file; --all mode: output directory")
    args = parser.parse_args()

    if args.all:
        run_all(load_turns(args.data_dir), model=args.model,
                out_dir=args.out or Path("outputs/discovery/method1/gate"), workers=args.workers)
        return

    turns = load_turns(args.data_dir)
    print(f"user turns: {len(turns)}")
    print(turns["drop_reason"].value_counts(dropna=False).to_string())
    kept = turns[turns["drop_reason"].isna()]
    sample = kept.sample(args.limit, random_state=args.seed)
    rows = []
    for i, (_, row) in enumerate(sample.iterrows(), 1):
        label = None if args.dry_run else classify(str(row["content"]), model=args.model)
        rows.append({
            "session_id": row["session_id"], "turn_number": int(row["turn_number"]),
            "shipped_pushback": row["prompt_pushback"], "word_count": int(row["word_count"]),
            "label": label, "content": str(row["content"]),
        })
        print(f"--- [{i}] label={label} shipped={row['prompt_pushback']} "
              f"words={int(row['word_count'])} ---\n{rows[-1]['content'][:900]}\n")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
