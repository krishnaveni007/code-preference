#!/usr/bin/env python3
"""
band_vs_commit.py  --  SWE-chat (SALT-NLP/SWE-chat) real schema

For one session, compare what the AGENT wrote during a band of turns against
what actually landed in each commit linked to that session's checkpoint.

Uses the dataset's own columns rather than reconstructing anything:
  conversations : turn_type=='tool_use', tool_name, file_path, turn_number
  commits       : checkpoint_pk, numstat, file_attribution, agent_changes

Session -> commit does NOT need a timestamp heuristic:
  conversations.checkpoint_pk  ->  commits.checkpoint_pk
The timestamp heuristic is only needed for TURN -> commit, which the schema
does not provide. That is the actual gap.

Usage
  python3 band_vs_commit.py --data-dir ./swechat_data \
      --session-id 3dded5ac-a667-436b-a093-ad8efbdf0e31 --out-dir ./band_analysis
"""

import argparse
import json
import os
from collections import defaultdict

import pandas as pd

WRITE_TOOLS = {  # file-modifying tools, Claude + Gemini naming
    "write", "edit", "multiedit", "notebookedit", "applypatch",
    "write_file", "edit_file", "replace",
}

# label, turn_lo, turn_hi, note   (turn_number, i.e. across ALL rows)
BANDS = [
    ("turns 11–99",   11,  99, "schema migration discussed"),
    ("turns 115–395", 115, 395, "registration status + filtering"),
]


# ---------------------------------------------------------------- helpers
def jloads(x, default=None):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return default
    if isinstance(x, (list, dict)):
        return x
    try:
        return json.loads(x)
    except (json.JSONDecodeError, TypeError):
        return default


def parse_numstat(raw):
    """git numstat -> {path: [added, deleted]}.

    The column may be raw text (tab- or space-separated) or JSON. Handles all.
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return {}
    obj = jloads(raw)
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if isinstance(v, dict):
                out[k] = [int(v.get("additions", v.get("added", 0)) or 0),
                          int(v.get("deletions", v.get("deleted", 0)) or 0)]
            elif isinstance(v, (list, tuple)) and len(v) >= 2:
                out[k] = [int(v[0] or 0), int(v[1] or 0)]
        if out:
            return out
    if isinstance(obj, list):
        out = {}
        for e in obj:
            if isinstance(e, dict):
                path = e.get("path") or e.get("file") or e.get("filename")
                if path:
                    out[path] = [int(e.get("additions", e.get("added", 0)) or 0),
                                 int(e.get("deletions", e.get("deleted", 0)) or 0)]
        if out:
            return out

    out = {}
    for line in str(raw).splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t") if "\t" in line else line.split(None, 2)
        if len(parts) >= 3:
            a, d, path = parts[0], parts[1], parts[-1].strip()
            try:
                out[path] = [0 if a == "-" else int(a), 0 if d == "-" else int(d)]
            except ValueError:
                continue
    return out


def parse_name_status(raw):
    """git name-status -> {path: [0, 0]} (paths only, no line counts)."""
    out = {}
    for line in str(raw or "").splitlines():
        parts = line.split("\t") if "\t" in line else line.split(None, 1)
        if len(parts) >= 2:
            out[parts[-1].strip()] = [0, 0]
    return out


def parse_patch_paths(raw):
    """Unified diff -> {path: [added, deleted]} counted from +/- lines."""
    out, cur = {}, None
    for line in str(raw or "").splitlines():
        if line.startswith("+++ b/"):
            cur = line[6:].strip()
            out.setdefault(cur, [0, 0])
        elif line.startswith("--- a/") and cur is None:
            cur = line[6:].strip()
            out.setdefault(cur, [0, 0])
        elif cur and line.startswith("+") and not line.startswith("+++"):
            out[cur][0] += 1
        elif cur and line.startswith("-") and not line.startswith("---"):
            out[cur][1] += 1
        elif line.startswith("diff --git"):
            cur = None
    return out


def commit_files(c):
    """Best available per-file stats for a commit row, with source label."""
    for col, fn in (("numstat", parse_numstat),
                    ("patch", parse_patch_paths),
                    ("files_changed", parse_name_status)):
        if col in c.index:
            d = fn(c.get(col))
            if d:
                return d, col
    return {}, "none"


def parse_attribution(raw):
    """file_attribution JSON -> {path: 'agent_only'|'human_only'|'mixed'}"""
    obj = jloads(raw, {})
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            out[k] = v if isinstance(v, str) else (
                v.get("attribution") or v.get("class") or str(v))
        return out
    if isinstance(obj, list):
        out = {}
        for e in obj:
            if isinstance(e, dict):
                pth = e.get("path") or e.get("file") or e.get("filename")
                if pth:
                    out[pth] = (e.get("attribution") or e.get("class")
                                or e.get("type") or "?")
        return out
    return {}


def canonicalise(agent_paths, commit_paths):
    """Map absolute agent paths onto repo-relative commit paths by suffix.

    conversations.file_path is an absolute path on the developer's machine;
    commit paths are repo-relative. Match by longest path-suffix, falling
    back to basename.
    """
    cpaths = sorted(commit_paths, key=len, reverse=True)
    by_base = {}
    for c in commit_paths:
        by_base.setdefault(os.path.basename(c), []).append(c)

    mapping, unmatched = {}, []
    for a in agent_paths:
        a_norm = str(a).replace("\\", "/")
        hit = next((c for c in cpaths
                    if a_norm.endswith("/" + c) or a_norm == c), None)
        if hit is None:
            cands = by_base.get(os.path.basename(a_norm), [])
            hit = cands[0] if len(cands) == 1 else None
        if hit is None:
            unmatched.append(a)
        mapping[a] = hit
    return mapping, unmatched


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./swechat_data")
    ap.add_argument("--session-id", required=True)
    ap.add_argument("--out-dir", default="./band_analysis")
    ap.add_argument("--debug-commit", default=None,
                    help="dump raw columns for this commit_sha prefix and exit")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    conv = pd.read_parquet(f"{args.data_dir}/conversations.parquet")
    commits = pd.read_parquet(f"{args.data_dir}/commits.parquet")
    try:
        sessions = pd.read_parquet(f"{args.data_dir}/sessions.parquet")
    except FileNotFoundError:
        sessions = None

    if args.debug_commit:
        m = commits[commits.commit_sha.astype(str).str.startswith(args.debug_commit)]
        if m.empty:
            raise SystemExit(f"no commit starting {args.debug_commit}")
        c = m.iloc[0]
        for col in ("numstat", "files_changed", "file_attribution",
                    "agent_changes", "patch"):
            if col in c.index:
                v = str(c[col])
                print(f"\n--- {col} (len {len(v)}) ---")
                print(repr(v[:900]))
        return

    s = conv[conv.session_id == args.session_id].sort_values("turn_number")
    if s.empty:
        raise SystemExit(f"no conversations rows for session {args.session_id}")

    cp = s.checkpoint_pk.dropna().unique()
    print(f"session {args.session_id}")
    print(f"  {len(s):,} conversation rows, turn_number {s.turn_number.min()}"
          f"–{s.turn_number.max()}")
    print(f"  checkpoint_pk values on those rows: {list(cp)}")
    print("  -> conversations.checkpoint_pk is session-level; it cannot "
          "separate turns within a session")

    if sessions is not None:
        srow = sessions[sessions.session_id == args.session_id]
        if not srow.empty:
            r = srow.iloc[0]
            print(f"  session attribution: agent_lines={r.get('agent_lines')}, "
                  f"human_added={r.get('human_added')}, "
                  f"human_modified={r.get('human_modified')}, "
                  f"agent_percentage={r.get('agent_percentage')}")
            print(f"  canonical_checkpoint_pk={r.get('canonical_checkpoint_pk')}, "
                  f"checkpoint_ids={r.get('checkpoint_ids')}")

    # ---- every commit reachable from this session's checkpoint(s) --------
    cm = commits[commits.checkpoint_pk.isin(cp)].copy()
    cm = cm.drop_duplicates(subset="commit_sha").sort_values("author_date")
    print(f"\n{len(cm)} commit(s) linked via checkpoint_pk "
          f"(no timestamp heuristic needed):")
    for _, c in cm.iterrows():
        print(f"  {c.commit_sha[:8]}  {str(c.author_date)[:16]}  "
              f"+{c.total_additions}/-{c.total_deletions}  "
              f"{c.files_changed_count} files  {str(c.commit_message).splitlines()[0][:60]}")

    # ---- agent edits per band -------------------------------------------
    tools = s[s.turn_type == "tool_use"].copy()
    tools["tool_lc"] = tools.tool_name.astype(str).str.lower()
    writes = tools[tools.tool_lc.isin(WRITE_TOOLS) & tools.file_path.notna()]

    rows, summary = [], []
    for label, lo, hi, note in BANDS:
        band = writes[(writes.turn_number >= lo) & (writes.turn_number <= hi)]
        per_file = defaultdict(int)
        for _, t in band.iterrows():
            per_file[t.file_path] += 1

        print(f"\n=== {label} ({note}) ===")
        print(f"  {len(band)} file-writing tool calls across "
              f"{len(per_file)} file(s)")
        for p, n in sorted(per_file.items(), key=lambda kv: -kv[1]):
            print(f"    {p:<62} {n} edit call(s)")

        # pushback / intent context, already annotated in the dataset
        prompts = s[(s.turn_type == "user_prompt")
                    & (s.turn_number >= lo) & (s.turn_number <= hi)]
        if "prompt_pushback" in prompts.columns:
            pb = prompts.prompt_pushback.dropna()
            pb = pb[pb != "non_pushback"]
            if len(pb):
                print(f"  pushback in band: {dict(pb.value_counts())}")
        if "prompt_intent" in prompts.columns:
            it = prompts.prompt_intent.dropna()
            if len(it):
                print(f"  intent in band:   {dict(it.value_counts())}")

        # compare against every linked commit
        for _, c in cm.iterrows():
            nstat, src = commit_files(c)
            attrib = parse_attribution(c.file_attribution)
            cmap, unmatched = canonicalise(per_file.keys(), nstat.keys())
            shared = {cmap[a] for a in per_file if cmap.get(a)}
            classes = {p: attrib.get(p, "?") for p in shared}

            print(f"  -> {c.commit_sha[:8]} [{len(nstat)} files via {src}]: "
                  f"{len(shared)}/{len(per_file)} agent files present")
            for a in sorted(per_file):
                m = cmap.get(a)
                tag = f"{attrib.get(m, '?')}" if m else "NOT IN COMMIT"
                print(f"       {os.path.basename(a):<40} -> "
                      f"{m or '—':<48} {tag}")

            agent_by_repo = {}
            for a, n in per_file.items():
                agent_by_repo[cmap.get(a) or a] = n
            for p in sorted(set(agent_by_repo) | set(nstat)):
                rows.append({
                    "band": label, "commit_sha": c.commit_sha,
                    "file": p,
                    "agent_edit_calls": agent_by_repo.get(p, 0),
                    "commit_added": nstat.get(p, [0, 0])[0],
                    "commit_deleted": nstat.get(p, [0, 0])[1],
                    "file_attribution": attrib.get(p, ""),
                    "in_both": p in shared,
                })

            summary.append({
                "band": label, "note": note,
                "agent_files": len(per_file),
                "agent_edit_calls": len(band),
                "commit_sha": c.commit_sha[:8],
                "commit_message": str(c.commit_message).splitlines()[0],
                "commit_added": int(c.total_additions),
                "commit_deleted": int(c.total_deletions),
                "commit_files": int(c.files_changed_count),
                "shared_files": len(shared),
                "overlap_frac": len(shared) / len(per_file) if per_file else 0.0,
                "attribution_of_shared": ";".join(
                    f"{os.path.basename(p)}={v}" for p, v in sorted(classes.items())),
                "agent_top_files": "\n".join(
                    f"· {os.path.basename(p)} ({n})"
                    for p, n in sorted(per_file.items(), key=lambda kv: -kv[1])[:2]),
                "commit_top_files": "\n".join(
                    f"· {os.path.basename(p)} +{v[0]}/−{v[1]}"
                    for p, v in sorted(nstat.items(), key=lambda kv: -kv[1][0])[:2]),
                "unmatched_agent_files": len(unmatched),
            })

    pd.DataFrame(rows).to_csv(f"{args.out_dir}/band_vs_commit_files.csv", index=False)
    sm = pd.DataFrame(summary)
    sm.drop(columns=["agent_top_files", "commit_top_files"]).to_csv(
        f"{args.out_dir}/band_summary.csv", index=False)
    print(f"\nwrote CSVs to {args.out_dir}/")
    plot(summary, f"{args.out_dir}/band_vs_commit.png")


# ---------------------------------------------------------------- plot
def plot(summary, out_png):
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, Rectangle

    C_AGENT = "#2166ac"
    C_MATCH, C_PART, C_MISS = "#2e8b57", "#c8a020", "#b2182b"

    n = len(summary)
    fig, ax = plt.subplots(figsize=(13.5, 2.95 * n + 2.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.5, n * 2.95)
    ax.axis("off")

    ax.text(2.0, n * 2.95 - 0.4, "AGENT WROTE (these turns)", ha="center",
            fontsize=11, weight="bold", color=C_AGENT)
    ax.text(7.9, n * 2.95 - 0.4, "COMMIT CONTAINS", ha="center",
            fontsize=11, weight="bold", color="#8a6410")

    for i, r in enumerate(summary):
        y = (n - 1 - i) * 2.95 + 0.35
        ov = r["overlap_frac"]
        col = C_MATCH if ov >= 0.5 else (C_PART if ov > 0 else C_MISS)
        verdict = ("files overlap" if ov >= 0.5 else
                   "partial overlap" if ov > 0 else "no file overlap")

        ax.add_patch(Rectangle((0.15, y), 3.7, 2.1, facecolor="#eef3f9",
                               edgecolor=C_AGENT, lw=1.2))
        ax.text(0.35, y + 1.80, r["band"], fontsize=11, weight="bold", color=C_AGENT)
        ax.text(0.35, y + 1.44, r["note"], fontsize=9, style="italic", color="#555")
        ax.text(0.35, y + 1.06,
                f"{r['agent_files']} files · {r['agent_edit_calls']} edit calls",
                fontsize=10)
        ax.text(0.35, y + 0.70, r["agent_top_files"], fontsize=8.5,
                color="#444", va="top", linespacing=1.5)

        ax.add_patch(Rectangle((6.1, y), 3.75, 2.1, facecolor="#fdf6e6",
                               edgecolor="#c9a227", lw=1.2))
        ax.text(6.3, y + 1.80, r["commit_sha"], fontsize=11, weight="bold",
                color="#8a6410")
        ax.text(6.3, y + 1.44, r["commit_message"][:50], fontsize=9,
                style="italic", color="#555")
        ax.text(6.3, y + 1.06,
                f"{r['commit_files']} files · +{r['commit_added']:,}/−{r['commit_deleted']:,}",
                fontsize=10)
        ax.text(6.3, y + 0.70, r["commit_top_files"], fontsize=8.5,
                color="#444", va="top", linespacing=1.5)

        ax.add_patch(FancyArrowPatch((3.95, y + 1.15), (6.0, y + 1.15),
                                     arrowstyle="-|>", mutation_scale=16,
                                     color=col, lw=2.0))
        ax.text(4.98, y + 1.42, verdict, ha="center", fontsize=9.5,
                weight="bold", color=col)
        ax.text(4.98, y + 0.95, f"{r['shared_files']} of {r['agent_files']} files",
                ha="center", fontsize=8.5, color=col)
        if r["attribution_of_shared"]:
            ax.text(4.98, y + 0.62, r["attribution_of_shared"][:46],
                    ha="center", fontsize=8, color="#444")

    fig.suptitle("Agent edits per turn band vs. commits linked via checkpoint_pk",
                 fontsize=13, y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.965])
    plt.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()