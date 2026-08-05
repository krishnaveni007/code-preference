"""
CodePref-Bench: corpus-wide code-diff coverage census.

Scans ALL of SWE-chat (not just your 5 selected sessions) to answer: per
session, how many files actually have a usable agent-vs-committed diff to
score? This is the census that would have told you upfront that
timelabsad-dot's session only had 3 usable files out of 133 rows, instead
of discovering it session-by-session.

For every commit attributable to exactly one session (checkpoints shared
across multiple sessions - like the 22-session shared ancestor checkpoint
found earlier - are excluded, since a commit under one of those can't be
cleanly attributed to a single session), each file in that commit's
file_attribution is bucketed as:

  - no_agent_snapshot : agent_version is null (majority case - see prior
                         finding that only ~1/3 of commits typically have one)
  - agent_only_identical : agent_version present but byte-identical to
                            committed_version (zero diff, nothing to score)
  - mixed_trivial     : real diff, but small (< MIN_DIFF_LINES changed lines)
  - mixed_usable      : real diff, at or above the size threshold - this is
                         the number that actually matters for scoring

Runs in STREAMING mode deliberately - the full commits table embeds full
file text per file per commit, which is large enough across ~14,459
commits that materializing it all into a pandas DataFrame at once risks
running out of memory. Streaming processes one commit at a time and only
keeps lightweight per-session counters, discarding file text immediately
after measuring it.

Expect this to take a while (multiple minutes at least) given full-file
diffing on every mixed file across the whole corpus - this is a one-time
census, not something to re-run casually.

Output: session_diff_coverage.csv - one row per session with counts in
each bucket, joined with user_id/repo_id/turn_count from the sessions table.

Requires: pip install datasets pandas
"""

import json
import difflib

import pandas as pd
from datasets import load_dataset

MIN_DIFF_LINES = 3  # changed (+ or -) lines required to count as "usable"


def diff_line_count(agent_text: str, committed_text: str) -> int:
    agent_lines = agent_text.splitlines(keepends=True)
    committed_lines = committed_text.splitlines(keepends=True)
    diff = difflib.unified_diff(agent_lines, committed_lines, n=0)
    return sum(1 for line in diff if line.startswith(("+", "-")) and not line.startswith(("+++", "---")))


def main():
    print("Loading sessions table (for user_id/repo_id/turn_count join)...")
    sessions = load_dataset("SALT-NLP/SWE-chat", "sessions", split="train").to_pandas()
    session_meta = sessions.set_index("session_id")[["user_id", "repo_id", "turn_count"]].to_dict("index")

    print("Loading checkpoints table (for commit -> session mapping)...")
    checkpoints = load_dataset("SALT-NLP/SWE-chat", "checkpoints", split="train").to_pandas()

    # Build checkpoint_pk -> session_id, but ONLY for checkpoints that map to
    # exactly one session. Checkpoints shared across many sessions (like the
    # 22-session shared ancestor found earlier) are ambiguous and excluded -
    # a commit under one of those can't be cleanly credited to one session.
    checkpoint_to_session = {}
    ambiguous_checkpoints = 0
    for _, row in checkpoints.iterrows():
        try:
            pks = json.loads(row["session_pks"]) if isinstance(row["session_pks"], str) else row["session_pks"]
        except (TypeError, json.JSONDecodeError):
            continue
        if pks and len(pks) == 1:
            checkpoint_to_session[row["checkpoint_pk"]] = pks[0]
        elif pks and len(pks) > 1:
            ambiguous_checkpoints += 1
    print(f"  {len(checkpoint_to_session)} checkpoints map to exactly one session "
          f"({ambiguous_checkpoints} excluded as shared/ambiguous)")

    print("Streaming commits table (this is the slow part)...")
    commits_stream = load_dataset("SALT-NLP/SWE-chat", "commits", split="train", streaming=True)

    counters = {}  # session_id -> dict of bucket counts
    n_commits_seen = 0
    n_commits_attributed = 0

    for row in commits_stream:
        n_commits_seen += 1
        if n_commits_seen % 2000 == 0:
            print(f"  ...{n_commits_seen} commits processed")

        if not row.get("commit_sha"):
            continue
        cp_pk = row.get("checkpoint_pk")
        sid = checkpoint_to_session.get(cp_pk)
        if sid is None:
            continue
        n_commits_attributed += 1

        fa_raw = row.get("file_attribution")
        if not fa_raw:
            continue
        try:
            fa = json.loads(fa_raw)
        except (TypeError, json.JSONDecodeError):
            continue

        c = counters.setdefault(sid, {
            "no_agent_snapshot": 0, "agent_only_identical": 0,
            "mixed_trivial": 0, "mixed_usable": 0, "n_commits": set(),
        })
        c["n_commits"].add(row["commit_sha"])

        for fpath, d in fa.items():
            if fpath == "__aggregate__":
                continue
            agent_v = d.get("agent_version")
            committed_v = d.get("committed_version")
            if agent_v is None or committed_v is None:
                c["no_agent_snapshot"] += 1
                continue
            if agent_v == committed_v:
                c["agent_only_identical"] += 1
                continue
            n_lines = diff_line_count(agent_v, committed_v)
            if n_lines >= MIN_DIFF_LINES:
                c["mixed_usable"] += 1
            else:
                c["mixed_trivial"] += 1

    print(f"\nDone. {n_commits_seen} commits streamed, {n_commits_attributed} attributable to a single session.")

    rows = []
    for sid, c in counters.items():
        meta = session_meta.get(sid, {})
        rows.append({
            "session_id": sid,
            "user_id": meta.get("user_id"),
            "repo_id": meta.get("repo_id"),
            "turn_count": meta.get("turn_count"),
            "n_commits": len(c["n_commits"]),
            "no_agent_snapshot": c["no_agent_snapshot"],
            "agent_only_identical": c["agent_only_identical"],
            "mixed_trivial": c["mixed_trivial"],
            "mixed_usable": c["mixed_usable"],
        })

    out = pd.DataFrame(rows).sort_values("mixed_usable", ascending=False)
    out.to_csv("session_diff_coverage.csv", index=False)
    print(f"Wrote session_diff_coverage.csv ({len(out)} sessions)")
    print("\nTop 20 sessions by usable diff count:")
    print(out.head(20).to_string(index=False))


if __name__ == "__main__":
    main()