"""
CodePref-Bench vectorization pipeline: data extraction stage.

Purpose
-------
This script does NOT compute preference vectors itself (that's your rubric
LLM call, run separately). It prepares the two raw input streams your
rubric needs to be pointed at:

  1. turns.csv        -> one row per developer turn, with enough context to
                          run your rubric prompt and get a vector at that
                          point in the conversation. Segmented by which
                          "checkpoint window" (the stretch between commits)
                          each turn falls into.

  2. code_units.csv    -> one row per (file, checkpoint) where BOTH an
                          agent_version and committed_version snapshot
                          exist, ready to run your rubric prompt on code
                          instead of conversation. Rows where agent_version
                          is missing are dropped here (see note below) —
                          you don't want to score those, they're not a real
                          code-vs-committed comparison.

Also written:
  3. checkpoints_commits.csv -> the checkpoint/commit backbone for each
     session, so you can see the window boundaries you're segmenting on.

Requires: pip install datasets pandas
Run locally — this needs Hugging Face Hub access.
"""

import json
import os

import pandas as pd
from datasets import load_dataset

OUTPUT_DIR = "codepref_pipeline_export"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# The 5 sessions finalized after richness + code-coverage + pushback-turn
# screening (see prior analysis in data/ scripts).
SESSIONS = [
    {"user_id": "timelabsad-dot", "session_id": "2a84a5a3-0636-4047-aa95-342b1ebe7d83"},
    {"user_id": "ckeith26",       "session_id": "443d9387-9119-4760-a36b-33768f03ff49"},
    {"user_id": "maoxiaoke",      "session_id": "ba75c8c5-be57-4405-93d8-1619f0f1efbb"},
    {"user_id": "itsmaleen",      "session_id": "4810d459-5b4b-4b78-9518-a0eb00118590"},
    {"user_id": "ujuc",           "session_id": "2b3756eb-8c69-4980-ac0a-8d2c534931d8"},
]
SESSION_IDS = {s["session_id"] for s in SESSIONS}


def main():
    # ---- 1. Load only what we need, filtered to our 5 sessions ------------
    print("Loading sessions table...")
    sessions = load_dataset("SALT-NLP/SWE-chat", "sessions", split="train").to_pandas()
    sessions = sessions[sessions.session_id.isin(SESSION_IDS)]

    print("Loading conversations (filtered)...")
    conv_ds = load_dataset("SALT-NLP/SWE-chat", "conversations", split="train")
    conv_ds = conv_ds.filter(lambda x: x["session_id"] in SESSION_IDS)
    conv = conv_ds.to_pandas()
    conv = conv[conv.is_conversational].sort_values(["session_id", "turn_number"])

    print("Loading checkpoints...")
    checkpoints = load_dataset("SALT-NLP/SWE-chat", "checkpoints", split="train").to_pandas()

    print("Loading commits...")
    commits = load_dataset("SALT-NLP/SWE-chat", "commits", split="train").to_pandas()
    commits = commits.dropna(subset=["commit_sha"])
    commits["author_date"] = pd.to_datetime(commits["author_date"])

    # ---- 2. Per-session extraction -----------------------------------------
    all_turn_rows = []
    all_code_rows = []
    all_checkpoint_rows = []

    for sess in SESSIONS:
        sid = sess["session_id"]
        uid = sess["user_id"]

        # Checkpoints that reference this session (session_pks contains sid)
        sess_checkpoints = checkpoints[
            checkpoints["session_pks"].astype(str).str.contains(sid, na=False)
        ]
        cp_pks = sess_checkpoints["checkpoint_pk"].tolist()

        # Commits tied to those checkpoints, ordered chronologically —
        # this ordering is the window backbone. See caveat in the docstring
        # of the earlier single-session script: author_date is the most
        # reliable ordering signal available, not a guaranteed ground truth.
        sess_commits = commits[commits["checkpoint_pk"].isin(cp_pks)].sort_values("author_date")

        for _, c in sess_commits.iterrows():
            all_checkpoint_rows.append({
                "user_id": uid,
                "session_id": sid,
                "checkpoint_pk": c["checkpoint_pk"],
                "commit_sha": c["commit_sha"],
                "author_date": c["author_date"],
                "commit_message": c["commit_message"].split("\n")[0],
            })

            # Code vectorization units: only keep files where BOTH snapshots exist
            try:
                fa = json.loads(c["file_attribution"]) if c["file_attribution"] else {}
            except (TypeError, json.JSONDecodeError):
                fa = {}
            for fpath, d in fa.items():
                if fpath == "__aggregate__":
                    continue
                agent_v = d.get("agent_version")
                committed_v = d.get("committed_version")
                if agent_v is None or committed_v is None:
                    continue  # no real agent-vs-committed comparison possible
                all_code_rows.append({
                    "user_id": uid,
                    "session_id": sid,
                    "checkpoint_pk": c["checkpoint_pk"],
                    "commit_sha": c["commit_sha"],
                    "author_date": c["author_date"],
                    "file_path": fpath,
                    "attribution_label": d.get("attribution"),
                    "agent_version": agent_v,
                    "committed_version": committed_v,
                })

        # Turn-level rows, tagged with which checkpoint window they fall in
        # (window i = turns after commit i-1's author_date, up to commit i's)
        sess_turns = conv[conv.session_id == sid].sort_values("turn_number")
        commit_boundaries = sess_commits["author_date"].tolist()

        for _, t in sess_turns.iterrows():
            ts = pd.to_datetime(t["timestamp"])
            window = sum(1 for b in commit_boundaries if ts > b)  # 0-indexed window
            all_turn_rows.append({
                "user_id": uid,
                "session_id": sid,
                "turn_number": t["turn_number"],
                "conversation_turn_number": t["conversation_turn_number"],
                "role": t["role"],
                "content": t["content"],
                "timestamp": ts,
                "checkpoint_window": window,
                "prompt_intent": t.get("prompt_intent"),
                "prompt_pushback": t.get("prompt_pushback"),
            })

    turns_df = pd.DataFrame(all_turn_rows).sort_values(["session_id", "turn_number"])
    code_df = pd.DataFrame(all_code_rows)
    checkpoints_df = pd.DataFrame(all_checkpoint_rows)

    # ---- 3. Build windowed context per turn --------------------------------
    # Full-history cumulative context gets noisy fast (early turns + every
    # intervening assistant response dilute the signal). Use a sliding
    # window of the last CONTEXT_WINDOW_TURNS turns instead - still includes
    # assistant turns (a user correction needs the preceding agent response
    # to make sense), just bounded rather than growing unboundedly.
    CONTEXT_WINDOW_TURNS = 4

    def build_windowed(group):
        pieces = []
        windowed_texts = []
        for _, row in group.iterrows():
            pieces.append(f"{row['role'].upper()}: {row['content']}")
            window = pieces[-CONTEXT_WINDOW_TURNS:]
            windowed_texts.append("\n\n".join(window))
        group = group.copy()
        group["cumulative_context"] = windowed_texts
        return group

    turns_df = turns_df.groupby("session_id", group_keys=False)[turns_df.columns].apply(build_windowed)

    # ---- 4. Write outputs ---------------------------------------------------
    turns_path = f"{OUTPUT_DIR}/turns.csv"
    code_path = f"{OUTPUT_DIR}/code_units.csv"
    checkpoints_path = f"{OUTPUT_DIR}/checkpoints_commits.csv"

    turns_df.to_csv(turns_path, index=False)
    code_df.to_csv(code_path, index=False)
    checkpoints_df.to_csv(checkpoints_path, index=False)

    print("\nDone.")
    print(f"  turns.csv              -> {len(turns_df)} rows (developer+agent turns, 5 sessions)")
    print(f"  code_units.csv         -> {len(code_df)} rows (file snapshots with BOTH agent+committed version)")
    print(f"  checkpoints_commits.csv -> {len(checkpoints_df)} rows (checkpoint/commit backbone)")
    print("\nCoverage check — code_units rows per session:")
    if len(code_df):
        print(code_df.groupby("session_id").size())
    else:
        print("  (none of the 5 sessions had a usable agent_version/committed_version pair —")
        print("   check checkpoints_commits.csv to see what's there before scoring code vectors)")


if __name__ == "__main__":
    main()