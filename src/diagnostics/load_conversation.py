"""
Print every turn of one SWE-chat session to a text file.

Usage:
    python swe_chat_pull_turns.py --data-dir ./swechat_data --user-id ckeith26 --session-id abc123
    khaong,2026-01-23-79e9becd-0b83-42f1-8cb1-33375d080476

Requires: pip install pandas pyarrow
python3 load_conversation.py --data-dir ./swechat_data --user-id pc035860 --session-id 09edeff1-00e4-46d8-b85d-8a17ec7c18df
Run locally against your downloaded conversations.parquet.
"""

import argparse
import pandas as pd


def tag_for(row):
    """Short header tag like [User], [Agent], [Tool: Bash]."""
    tt = row["turn_type"]
    if tt == "user_prompt":
        return "User"
    if tt == "assistant_response":
        return "Agent"
    if tt == "assistant_thinking":
        return "Agent Thinking"
    if tt in ("tool_use", "tool_result"):
        name = row.get("tool_name") or "unknown"
        return f"Tool: {name} ({tt})"
    return tt  # system_injected, summary, system_event, file_snapshot, progress, queue_operation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="Folder containing conversations.parquet")
    ap.add_argument("--user-id", required=True)
    ap.add_argument("--session-id", required=True)
    ap.add_argument("--out", default=None, help="Output text file (default: turns_<session_id>.txt)")
    args = ap.parse_args()

    df = pd.read_parquet(f"{args.data_dir}/conversations.parquet")
    turns = df[(df["session_id"] == args.session_id) & (df["user_id"] == args.user_id)]
    turns = turns[~turns["turn_type"].isin(["progress", "file_snapshot"])]
    turns = turns.sort_values("turn_number")

    if turns.empty:
        print(f"No turns found for user_id={args.user_id}, session_id={args.session_id}")
        return

    out_path = args.out or f"turns_{args.session_id}.txt"
    with open(out_path, "w") as f:
        for i, (_, row) in enumerate(turns.iterrows(), start=1):
            f.write(f'turn {i} [{tag_for(row)}] "{row["content"]}"\n')

    print(f"Wrote {len(turns)} turns to {out_path}")


if __name__ == "__main__":
    main()