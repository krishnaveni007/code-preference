"""The c2p pointer memory format: its spec (parsed from spec/c2_remember_pointer.md, the single source of truth) and
its lint. Vendored from the pref-memory workspace's memory builder so this package runs on its own."""
from __future__ import annotations

import pathlib
import re

SPEC = pathlib.Path(__file__).resolve().parent / "spec" / "c2_remember_pointer.md"
DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
QUOTE_LINE_RE = re.compile(r"^\s*-\s*(\d{4}-\d{2}-\d{2})\b[^\"“]*[\"“](.+)[\"”]")  # greedy: first to last quote
V2_MAX_QUOTE_WORDS = 25


def read_text(path: pathlib.Path) -> str:
    return path.read_text().strip()


def estimate_tokens(text: str) -> int:
    """Heuristic token estimate: chars / 4."""
    return (len(text) + 3) // 4


def norm_ws(s: str) -> str:
    return " ".join(s.split())


def parse_frontmatter(text: str) -> dict | None:
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    if not m:
        return None
    out = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out



MODES = ["discuss", "plan", "build", "verify", "ship", "configure"]
C2P_INTRO = ("Preferences of this developer, one note per topic. Open a note before acting when its line matches "
             "what you are about to do.")
C2P_PROJECT_LINE = "- [Project context](project-context.md) — before changing code in this repository"
C2P_MAX_MODES, C2P_MAX_EXAMPLES, C2P_MAX_NOTE_TOKENS = 3, 3, 900
C2P_HEAD_RE = re.compile(r"^## (\w+) — (.+)$", re.M)
LEDGER_GIST_RE = re.compile(r"^\s*-\s*\d{4}-\d{2}-\d{2} \(ledger gist\): \S")


def pointer_spec() -> dict:
    """Parsed from the c2p instruction (the single source of truth): phases, and per dimension code its name, file,
    phase, sides and trigger phrase per mode."""
    text = read_text(SPEC)
    block = text.split("## Phases and dimensions", 1)[1].split("\n## ", 1)[0]
    dims, phases, phase = {}, [], None
    for ln in block.splitlines():
        if ln.startswith("### "):
            phase = ln[4:].strip()
            phases.append(phase)
        m = re.match(r"^- (\w+) · ([^·]+) · (\S+\.md): (.+)$", ln)
        if m:
            dims[m.group(1)] = {"name": m.group(2).strip(), "file": m.group(3), "phase": phase,
                                "sides": [x.strip() for x in m.group(4).split("|")], "triggers": {}}
    table = text.split("## Modes and their trigger phrases", 1)[1].split("\n## ", 1)[0]
    for ln in table.splitlines():
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if cells and cells[0] in dims:
            dims[cells[0]]["triggers"] = dict(zip(MODES, cells[1:]))
    return {"phases": phases, "dims": dims}


def ledger_counts(memdir: pathlib.Path) -> tuple[dict, set]:
    """(DIM, mode, side) -> ledger lines, and the set of such keys with a correction."""
    counts, corrected = {}, set()
    f = memdir / "observations.md"
    if f.exists():
        for ln in f.read_text(errors="replace").splitlines():
            cells = [c.strip() for c in ln.split("|")]
            if len(cells) >= 6 and DATE_RE.fullmatch(cells[0] or "-"):
                key = (cells[1].upper(), cells[2].lower(), cells[3].lower())
                counts[key] = counts.get(key, 0) + 1
                if cells[-1].lower().startswith("yes"):
                    corrected.add(key)
    return counts, corrected


def lint_c2p(memdir: pathlib.Path, md: list[pathlib.Path], session_date: str, corpus: str) -> list[str]:
    spec = pointer_spec()
    dims = spec["dims"]
    by_file = {d["file"]: k for k, d in dims.items()}
    fixed = set(by_file) | {"MEMORY.md", "observations.md", "project-context.md", "other.md"}
    w: list[str] = []
    corpus_n = norm_ws(corpus)
    counts, corrected = ledger_counts(memdir)
    sides_all = {x.lower() for d in dims.values() for x in d["sides"]}
    note_modes: dict[str, list[str]] = {}
    in_notes: set = set()
    conflicts: dict[str, str] = {}

    for p in md:
        if p.name not in fixed:
            w.append(f"{p.name}: file name not in the fixed set")
            continue
        if p.name not in by_file:
            continue
        code, d = by_file[p.name], dims[by_file[p.name]]
        text = p.read_text(errors="replace")
        fm = parse_frontmatter(text) or {}
        if fm.get("description") != d["name"]:
            w.append(f"{p.name}: description is not the dimension name")
        for fld in (f"Phase: {d['phase']}", f"Dimension: {code} · {d['name']}"):
            if fld not in text:
                w.append(f"{p.name}: missing line '{fld}'")
        toks = estimate_tokens(text)
        if toks > C2P_MAX_NOTE_TOKENS:
            w.append(f"{p.name}: {toks} tokens (> {C2P_MAX_NOTE_TOKENS})")
        heads = list(C2P_HEAD_RE.finditer(text))
        modes = [h.group(1) for h in heads]
        note_modes[p.name] = modes
        if not heads:
            w.append(f"{p.name}: no mode section")
        if modes != sorted(modes, key=lambda m: MODES.index(m) if m in MODES else 99):
            w.append(f"{p.name}: mode sections not in mode order")
        for i, h in enumerate(heads):
            mode, phrase = h.group(1), h.group(2).strip()
            body = text[h.end(): heads[i + 1].start() if i + 1 < len(heads) else len(text)]
            body = body.split("\nConflicts", 1)[0]
            if mode not in MODES:
                w.append(f"{p.name}: unknown mode '{mode}'")
                continue
            if phrase != d["triggers"].get(mode):
                w.append(f"{p.name}: {mode} heading is not the table's trigger phrase")
            blocks = re.split(r"^(?=Stance(?: \([^)\n]*\))?: )", body, flags=re.M)[1:]  # tolerate "Stance (second side): "
            if not blocks:
                w.append(f"{p.name}: {mode} has no 'Stance: <side> (n=…' line")
            for blk in blocks:
                m = re.match(r"Stance(?: \([^)\n]*\))?: (.+?) \(n=(\d+)", blk)
                if not m:
                    w.append(f"{p.name}: {mode} has a Stance line without '(n=…'")
                    continue
                side = m.group(1).strip().lower()
                if side not in {x.lower() for x in d["sides"]}:
                    w.append(f"{p.name}: {mode} side not in vocabulary: {side}")
                key = (code, mode, side)
                in_notes.add(key)
                if counts.get(key, 0) < 2 and key not in corrected:
                    w.append(f"{p.name}: {mode}/{side} kept with {counts.get(key, 0)} ledger lines and no correction")
                if not re.search(r"^How to apply: \S", blk, re.M):
                    w.append(f"{p.name}: {mode}/{side} has no 'How to apply:'")
                quotes = [ln for ln in blk.split("Examples:", 1)[1].splitlines() if ln.strip().startswith("-")] \
                    if "Examples:" in blk else []
                if not quotes:
                    w.append(f"{p.name}: {mode}/{side} has no examples")
                if len(quotes) > C2P_MAX_EXAMPLES:
                    w.append(f"{p.name}: {mode}/{side} has {len(quotes)} examples (> {C2P_MAX_EXAMPLES})")
                for ln in quotes:
                    if LEDGER_GIST_RE.match(ln):
                        continue
                    qm = QUOTE_LINE_RE.match(ln)
                    if not qm:
                        w.append(f"{p.name}: unparseable example: {ln.strip()[:60]}")
                        continue
                    date, q = qm.groups()
                    q = q.replace('\\"', '"')
                    if date > session_date:
                        w.append(f"{p.name}: example dated {date} after the session")
                    segs = [x for x in re.split(r"…|\.\.\.", q) if norm_ws(x)]
                    if sum(len(x.split()) for x in segs) > V2_MAX_QUOTE_WORDS:
                        w.append(f"{p.name}: example over {V2_MAX_QUOTE_WORDS} words: {q[:40]}")
                    if not segs or not all(norm_ws(x) in corpus_n for x in segs):
                        w.append(f"{p.name}: example not in the recordings: {q[:60]}")
        conflicts[code] = " ".join(re.findall(r"^Conflicts.*$", text, re.M)).lower()

    # every kept ledger stance is in its note, as a stance block or in Conflicts
    for key in sorted(k for k, n in counts.items() if (n >= 2 or k in corrected) and k[0] in dims):
        code, mode, side = key
        if key not in in_notes and side not in conflicts.get(code, "") and side in {x.lower() for x in dims[code]["sides"]}:
            w.append(f"{dims[code]['file']}: kept stance missing: {mode}/{side} (n={counts[key]})")

    index = memdir / "MEMORY.md"
    if not index.exists():
        return sorted(set(w))
    lines = [ln for ln in index.read_text().splitlines() if ln.strip()]
    if lines[:2] != ["# Memory", C2P_INTRO]:
        w.append("MEMORY.md: does not start with the fixed title and intro")
    indexed, phase, last_phase_i = set(), None, -1
    for ln in lines[2:]:
        if ln.startswith("## "):
            phase = ln[3:].strip()
            if phase == "Project":
                continue
            if phase not in spec["phases"]:
                w.append(f"MEMORY.md: unknown heading '{phase}'")
            elif spec["phases"].index(phase) <= last_phase_i:
                w.append(f"MEMORY.md: phase '{phase}' out of order or repeated")
            else:
                last_phase_i = spec["phases"].index(phase)
            continue
        if phase == "Project":
            if ln != C2P_PROJECT_LINE:
                w.append("MEMORY.md: project line differs from the fixed one")
            continue
        m = re.match(r"^- \[(\w+) · ([^\]]+)\]\(([^)]+)\) — (.+)$", ln)
        if not m:
            w.append(f"MEMORY.md: line not in '- [KEY · Name](file.md) — trigger' form: {ln[:50]}")
            continue
        code, name, file, trig = m.groups()
        d = dims.get(code)
        if not d or d["name"] != name or d["file"] != file:
            w.append(f"MEMORY.md: {code} name or file differs from the list")
            continue
        if d["phase"] != phase:
            w.append(f"MEMORY.md: {code} under '{phase}', belongs under '{d['phase']}'")
        indexed.add(file)
        modes = note_modes.get(file)
        if modes is None:
            w.append(f"MEMORY.md: {code} points to a missing note")
            continue
        parts = trig.split("; ")
        allowed = [d["triggers"][x] for x in modes if x in d["triggers"]]
        if len(parts) > C2P_MAX_MODES:
            w.append(f"MEMORY.md: {code} has {len(parts)} trigger phrases (> {C2P_MAX_MODES})")
        if any(x not in allowed for x in parts) or parts != [x for x in allowed if x in parts]:
            w.append(f"MEMORY.md: {code} trigger is not the note's mode phrases in mode order")
        if len(parts) < min(len(allowed), C2P_MAX_MODES):
            w.append(f"MEMORY.md: {code} trigger leaves out a mode the note has")
    text_l = index.read_text().lower()
    leaked = sorted(x for x in sides_all if re.search(rf"(?<![\w-]){re.escape(x)}(?![\w-])", text_l))
    if leaked:
        w.append(f"MEMORY.md: stance words in the index: {', '.join(leaked)}")
    if re.search(r"[\"“”]", index.read_text()):
        w.append("MEMORY.md: quote marks in the index")
    for f in note_modes:
        if f not in indexed:
            w.append(f"MEMORY.md: no line for {f}")
    if (memdir / "project-context.md").exists() and C2P_PROJECT_LINE not in lines:
        w.append("MEMORY.md: project line missing")
    return sorted(set(w))


