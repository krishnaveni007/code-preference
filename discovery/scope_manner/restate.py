"""Step 1: structured restatement of every gated message (one LLM call each).

Usage: python restate.py [--model M] [--workers N] [--out NAME]
Output: outputs/discovery/scope_manner/data/<out>.jsonl (append-only, resumable; last successful line per id wins).
Fields: topic, asked, corrects, condition, trigger, kind, manner, quote (see prompt_v4.txt).
"""
import argparse, json, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import common as C

KEYS = ["topic", "asked", "corrects", "condition", "trigger", "kind", "manner", "quote"]
SCHEMA = {"type": "object", "additionalProperties": False,
          "properties": {**{k: {"type": "string"} for k in KEYS}, "kind": {"type": "string", "enum": ["task", "manner", "both"]}},
          "required": KEYS}
PROMPT = (Path(__file__).parent / "prompt_v4.txt").read_text()


def build_input(msg, prev):
    p = (prev.get((msg["session_id"], msg["turn_number"])) or "")[-1500:]
    body = " ".join(msg["content"].split()[:300])
    return f"<agent_previous_reply>\n{p}\n</agent_previous_reply>\n<developer_message>\n{body}\n</developer_message>"


def restate_one(user_text, model, temperature=0, strict=True):
    rf = {"type": "json_schema", "json_schema": {"name": "restate", "strict": True, "schema": SCHEMA}} if strict else {"type": "json_object"}
    o = C.parse_json(C.chat([{"role": "system", "content": PROMPT}, {"role": "user", "content": user_text}], model, rf,
                            temperature=temperature, timeout=90))
    assert all(k in o for k in KEYS), "missing keys"
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=C.DEFAULT_CHAT_MODEL)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default="restated")
    ap.add_argument("--limit", type=int, default=0, help="only the first N ids (for tests)")
    a = ap.parse_args()
    C.DATA.mkdir(parents=True, exist_ok=True)
    out = C.DATA / f"{a.out}.jsonl"
    msgs = C.load_messages(); prev = C.load_prev_replies()
    done = set(C.load_records(out)) if out.exists() else set()
    todo = [i for i in range(len(msgs)) if i not in done][: a.limit or None]
    print(f"total {len(msgs)} done {len(done)} todo {len(todo)}", flush=True)
    lock = threading.Lock(); f = open(out, "a"); n = [0]

    def one(i):
        try:
            o = C.with_retries(lambda: restate_one(build_input(msgs[i], prev), a.model), tries=5)
        except Exception as e:  # noqa: BLE001
            o = {"error": str(e)[:200]}
        o["id"] = i
        with lock:
            f.write(json.dumps(o, ensure_ascii=False) + "\n"); f.flush(); n[0] += 1
            if n[0] % 500 == 0: print("progress", n[0], flush=True)

    with ThreadPoolExecutor(a.workers) as ex: list(ex.map(one, todo))
    print("finished", flush=True)


if __name__ == "__main__":
    main()
