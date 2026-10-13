"""Step 1b: re-run ids whose restatement failed (malformed JSON), with lenient parsing and rising temperature.

Usage: python repair_records.py [--model M] [--out NAME]
"""
import argparse, json
import common as C
from restate import build_input, restate_one

ap = argparse.ArgumentParser(); ap.add_argument("--model", default=C.DEFAULT_CHAT_MODEL); ap.add_argument("--out", default="restated")
a = ap.parse_args(); path = C.DATA / f"{a.out}.jsonl"
ok = set(C.load_records(path)); bad = {json.loads(l)["id"] for l in open(path) if '"error"' in l}
todo = sorted(bad - ok); print("unresolved ids:", len(todo))
msgs = C.load_messages(); prev = C.load_prev_replies(); f = open(path, "a"); fixed = 0
for i in todo:
    o = None
    for temp, strict in [(0, True), (0.3, False), (0.6, False)]:
        try:
            o = restate_one(build_input(msgs[i], prev), a.model, temperature=temp, strict=strict); break
        except Exception as e:  # noqa: BLE001
            last = str(e)[:120]
    if o: o["id"] = i; f.write(json.dumps(o, ensure_ascii=False) + "\n"); fixed += 1
    else: print("still failing", i, last)
print(f"repaired {fixed} of {len(todo)}")
