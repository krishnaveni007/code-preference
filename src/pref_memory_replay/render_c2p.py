#!/usr/bin/env python3
"""Write a c2p (pointer-format) memory directly from the explorer's coded preference statements, no Claude Code build.

Layout, stances, counts, dates and examples are deterministic; only `How to apply:` and the `Conflicts:` scope are
written by a tool-less model call per note, from that note's evidence alone. Leakage guard: a statement is used only
if its session started (log clock, conversations.parquet) before --cutoff, and only if its quote is a verbatim
substring of one of the user's prompts in that session (or text typed in a tool rejection); otherwise it is dropped
and reported. A stance is kept at n >= 2 statements or one correction/rejection, as in render_c2_sample.py.

Writes to --out: MEMORY.md + one note per dimension (the mountable memory), and next to it <out>_ledger/
(observations.md in the lint's ledger format, drops.json, apply.json, lint.txt). The ledger is kept out of the memory
dir because the pointer format does not index it.

  python src/pref_memory_replay/render_c2p.py --user <swe-chat user_id> --cutoff <target session start, ISO> \
      --out $TURN_REPLAY_DATA/<case>/mem_c2p --cc-bin /abs/path/to/claude
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import shutil
import subprocess
import sys
from collections import defaultdict

import pandas as pd
import pyarrow.dataset as ds

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import c2p_format as bm  # noqa: E402
import common  # noqa: E402

CONV = common.SNAPSHOT / "conversations.parquet"
EXPLORER = common.EXPLORER
MIN_N, MAX_EXAMPLES, MAX_WORDS = 2, 3, 25
CORRECTION = {"correction", "rejection", "tool rejection"}
OPPOSED = [{"ask or wait first", "proceed on its own"}, {"tell me more", "keep it short"},
           {"stick to the request", "improve what you find"}, {"plan or research deeply first", "keep planning light"},
           {"test thoroughly", "test lightly or by hand"}, {"commit in small steps", "few big commits"},
           {"fail fast and loud", "handle gracefully"}, {"step by step", "all at once"}]

APPLY_PROMPT = """You write the `How to apply:` lines of one preference note for a coding agent's memory.
For each stance below, write one or two imperative sentences the developer could have said, saying WHEN (the mode's
trigger) and WHAT to do. Use only what the quotes and paraphrases show; no new facts, no project details, no quotes.
For each conflict, write a short scope: when the conflicting side applies instead, from its own evidence.

Dimension: {dim}
{blocks}

Return JSON: {{"apply": {{"<mode>|<side>": "<sentences>"}}, "scope": {{"<mode>|<side>": "<scope>"}}}}"""
APPLY_SCHEMA = {"type": "object", "required": ["apply", "scope"],
                "properties": {"apply": {"type": "object", "additionalProperties": {"type": "string"}},
                               "scope": {"type": "object", "additionalProperties": {"type": "string"}}}}


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def prompts_before(user: str, cutoff: pd.Timestamp) -> tuple[pd.DataFrame, pd.Series]:
    conv = ds.dataset(str(CONV))
    f = ds.field("user_id") == user
    starts = conv.to_table(columns=["session_id", "timestamp"], filter=f).to_pandas().groupby("session_id").timestamp.min()
    prior = starts[starts < cutoff]
    p = conv.to_table(columns=["session_id", "timestamp", "turn_type", "content", "prompt_pushback"],
                      filter=f & ds.field("turn_type").isin(["user_prompt", "tool_result"])).to_pandas()
    p = p[p.session_id.isin(prior.index) & p.content.notna()].copy()
    rej = p.turn_type == "tool_result"  # text typed while rejecting a tool call
    p.loc[rej, "content"] = p.loc[rej, "content"].str.extract(r"the user said:\n(.*)", flags=re.S)[0]
    p.loc[rej, "prompt_pushback"] = "tool rejection"
    p = p[p.content.notna()]
    p["n"] = p.content.map(norm)
    return p, prior


def cut_quote(q: str) -> str:
    """First MAX_WORDS words, keeping the explorer's own elisions as "…" (each segment stays verbatim)."""
    segs = [norm(x) for x in re.split(r"\.\.\.|…", q) if norm(x)]
    out, n = [], 0
    for seg in segs:
        w = seg.split(" ")[:MAX_WORDS - n]
        out.append(" ".join(w))
        n += len(w)
        if n >= MAX_WORDS:
            break
    total = sum(len(x.split(" ")) for x in segs)
    return " … ".join(out) + (" …" if n < total else "")


def call_model(prompt: str, model: str, cc_bin: str) -> dict:
    env = {"PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin", "HOME": str(pathlib.Path.home()),
           "CLAUDE_CODE_OAUTH_TOKEN": common.oauth_token()}
    cmd = [cc_bin, "-p", "--model", model, "--output-format", "json", "--tools", "", "--setting-sources", "",
           "--json-schema", json.dumps(APPLY_SCHEMA)]
    r = json.loads(subprocess.run(cmd, input=prompt, capture_output=True, text=True, env=env, timeout=900).stdout)
    if r.get("is_error"):
        sys.exit(f"model call failed: {r.get('result')}")
    return r["structured_output"]


def render_note(code, d, modes, stances, conflicts, gen, max_examples) -> str:
    lines = ["---", f"name: {d['file'][:-3]}", f"description: {d['name']}", "metadata:", "  type: feedback", "---",
             f"Phase: {d['phase']}", f"Dimension: {code} · {d['name']}", ""]
    for m in modes:
        lines.append(f"## {m} — {d['triggers'][m]}")
        for side, ev in stances[m]:
            ds_ = sorted(r["date"] for r in ev)
            ns = len({r["sid"] for r in ev})
            lines.append(f"Stance: {side} (n={len(ev)} in {ns} session{'s' * (ns > 1)}, {ds_[0]} → {ds_[-1]})")
            lines.append(f"How to apply: {gen['apply'].get(f'{m}|{side}', '').strip()}")
            lines.append("Examples:")
            seen, shown = set(), []
            for r in sorted(ev, key=lambda r: (len(r["quote"]), r["date"])):  # shortest quotes = densest wording
                q = cut_quote(r["quote"])
                if q not in seen:
                    seen.add(q)
                    shown.append((r["date"], q))
                if len(shown) == max_examples:
                    break
            lines += [f"- {dt}: \"{q}\"" for dt, q in sorted(shown)]
            lines.append("")
    for m, side, ev, rival in conflicts:
        ds_ = sorted(r["date"] for r in ev)
        lines.append(f"Conflicts: {side} (n={len(ev)}, {ds_[0]} → {ds_[-1]}). "
                     f"Scope: {gen['scope'].get(f'{m}|{side}', '').strip()}")
    return "\n".join(lines).rstrip() + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--cutoff", required=True, help="start of the target session (log clock, ISO)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cc-bin", default=common.JUDGE_CC, required=not common.JUDGE_CC)
    ap.add_argument("--model", default=common.JUDGE_MODEL)
    a = ap.parse_args()
    cutoff = pd.Timestamp(a.cutoff)
    spec = bm.pointer_spec()
    dims = spec["dims"]

    prompts, prior = prompts_before(a.user, cutoff)
    sid_of = {s[:8]: s for s in prior.index}
    stmts = [dict(s, dim=d["key"]) for d in json.loads(EXPLORER.read_text())["dims"] for s in d["statements"]
             if s["user"] == a.user and s["label"] == "preference"]
    rows, drops = [], []
    for s in stmts:
        sid = sid_of.get(s["session"])
        if sid is None:
            drops.append({**s, "why": "session not before cutoff"})
            continue
        # the explorer elides with "..." / "…": every segment must be in the same prompt
        segs = [norm(x) for x in re.split(r"\.\.\.|…", s["quote"]) if norm(x)]
        hit = prompts[prompts.session_id == sid]
        for seg in segs:
            hit = hit[hit.n.str.contains(seg, regex=False)]
        if hit.empty or not segs:
            drops.append({**s, "why": "quote not verbatim in that session's prompts"})
            continue
        if s["side"] not in dims[s["dim"]]["sides"]:
            drops.append({**s, "why": "side not in vocabulary"})
            continue
        rows.append({**s, "date": str(prior[sid])[:10], "sid": s["session"], "pb": hit.iloc[0].prompt_pushback})

    out = pathlib.Path(a.out)
    led_dir = out.parent / f"{out.name}_ledger"
    for d in (out, led_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    ledger = [f"{r['date']} | {r['dim']} | {r['mode']} | {r['side']} | {cut_quote(r['manner'])[:90]} | "
              f"{'yes' if r['pb'] in CORRECTION else 'no'}" for r in sorted(rows, key=lambda r: r["date"])]
    (led_dir / "observations.md").write_text("date | DIM | mode | side | gist | correction?\n\n" + "\n".join(ledger) + "\n")
    (led_dir / "drops.json").write_text(json.dumps(drops, indent=1))

    cells = defaultdict(list)
    for r in rows:
        cells[(r["dim"], r["mode"], r["side"])].append(r)
    kept = {k: v for k, v in cells.items() if len(v) >= MIN_N or any(r["pb"] in CORRECTION for r in v)}

    index, applies = defaultdict(list), {}
    for code, d in dims.items():
        dc = {k: v for k, v in kept.items() if k[0] == code}
        if not dc:
            continue
        # per mode: stances by evidence; a side opposed to a stronger one in the same mode goes to Conflicts
        modes = [m for m in bm.MODES if any(k[1] == m for k in dc)]
        stances, conflicts = {}, []
        for m in modes:
            ms = sorted(((k[2], v) for k, v in dc.items() if k[1] == m), key=lambda kv: -len(kv[1]))
            keep = []
            for side, ev in ms:
                rival = next((s for s, _ in keep if {s, side} in OPPOSED), None)
                (conflicts.append((m, side, ev, rival)) if rival else keep.append((side, ev)))
            stances[m] = keep
        blocks = []
        for m in modes:
            for side, ev in stances[m]:
                blocks.append(f"[stance {m}|{side}] mode trigger: {d['triggers'][m]}\n" + "\n".join(
                    f"  - quote: {cut_quote(r['quote'])}  | paraphrase: {r['manner']}" for r in ev[-8:]))
        for m, side, ev, rival in conflicts:
            blocks.append(f"[conflict {m}|{side}] (opposes '{rival}')\n" + "\n".join(
                f"  - quote: {cut_quote(r['quote'])}  | paraphrase: {r['manner']}" for r in ev[-8:]))
        gen = call_model(APPLY_PROMPT.format(dim=f"{code} · {d['name']}", blocks="\n\n".join(blocks)), a.model, a.cc_bin)
        applies[code] = gen
        for k in range(MAX_EXAMPLES, 0, -1):  # fewer examples per stance until the note fits the token cap
            text = render_note(code, d, modes, stances, conflicts, gen, k)
            if bm.estimate_tokens(text) <= bm.C2P_MAX_NOTE_TOKENS:
                break
        (out / d["file"]).write_text(text)
        top = sorted(modes, key=lambda m: -sum(len(ev) for _, ev in stances[m]))[:bm.C2P_MAX_MODES]
        index[d["phase"]].append(f"- [{code} · {d['name']}]({d['file']}) — "
                                 + "; ".join(d["triggers"][m] for m in modes if m in top))
    idx = ["# Memory", "", bm.C2P_INTRO, ""]
    for ph in spec["phases"]:
        if index[ph]:
            idx += [f"## {ph}", *index[ph], ""]
    (out / "MEMORY.md").write_text("\n".join(idx).rstrip() + "\n")
    (led_dir / "apply.json").write_text(json.dumps(applies, indent=1))

    # lint against the ledger and the prior prompts as corpus
    tmp = led_dir / "lint_dir"
    shutil.copytree(out, tmp)
    shutil.copy(led_dir / "observations.md", tmp / "observations.md")
    warns = bm.lint_c2p(tmp, sorted(tmp.glob("*.md")), str(cutoff)[:10], "\n".join(prompts.content))
    shutil.rmtree(tmp)
    (led_dir / "lint.txt").write_text("\n".join(warns) + "\n")
    print(f"{len(stmts)} statements, {len(rows)} used, {len(drops)} dropped "
          f"({sum(x['why'].startswith('session') for x in drops)} after cutoff); {len(prior)} prior sessions; "
          f"{len(kept)} stances kept in {sum(map(len, index.values()))} notes; lint warnings {len(warns)}")
    for w in warns:
        print("  lint:", w)


if __name__ == "__main__":
    main()
