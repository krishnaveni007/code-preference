"""
Print the turns immediately surrounding each pushback turn in a SWE-chat session.

A "pushback turn" is a user_prompt row where prompt_pushback is one of:
correction, rejection, failure_report, pacing_complaint, takeover,
requirement_change (i.e. not null and not "non_pushback").

For each one found, writes a window of +/- N turns (default 10) around it,
with the pushback turn marked with ">>>".

Usage:
    python swe_chat_pushback_window.py --data-dir ./swechat_data --user-id ckeith26 --session-id abc123
    python swe_chat_pushback_window.py --data-dir ./swechat_data --user-id ckeith26 --session-id abc123 --window 5

Requires: pip install pandas pyarrow
"""

import argparse
import pandas as pd

PUSHBACK_CLASSES = {
    "correction", "rejection", "failure_report",
    "pacing_complaint", "takeover", "requirement_change",
}


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
    return tt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="Folder containing conversations.parquet")
    ap.add_argument("--user-id", required=True)
    ap.add_argument("--session-id", required=True)
    ap.add_argument("--window", type=int, default=10, help="Turns to include on each side of a pushback turn")
    ap.add_argument("--out", default=None, help="Output text file (default: pushback_windows_<session_id>.txt)")
    args = ap.parse_args()

    df = pd.read_parquet(f"{args.data_dir}/conversations.parquet")
    turns = df[(df["session_id"] == args.session_id) & (df["user_id"] == args.user_id)]
    turns = turns[~turns["turn_type"].isin(["progress", "file_snapshot"])]
    turns = turns.sort_values("turn_number").reset_index(drop=True)

    if turns.empty:
        print(f"No turns found for user_id={args.user_id}, session_id={args.session_id}")
        return

    is_pushback = (turns["turn_type"] == "user_prompt") & turns["prompt_pushback"].isin(PUSHBACK_CLASSES)
    pushback_positions = turns.index[is_pushback].tolist()

    if not pushback_positions:
        print("No pushback turns found in this session.")
        return

    out_path = args.out or f"pushback_windows_{args.session_id}.txt"
    with open(out_path, "w") as f:
        for pb_pos in pushback_positions:
            pb_row = turns.loc[pb_pos]
            lo = max(0, pb_pos - args.window)
            hi = min(len(turns) - 1, pb_pos + args.window)
            f.write(f'=== Pushback: {pb_row["prompt_pushback"]} at turn {pb_pos + 1} ===\n')
            for pos in range(lo, hi + 1):
                row = turns.loc[pos]
                marker = ">>> " if pos == pb_pos else "    "
                f.write(f'{marker}turn {pos + 1} [{tag_for(row)}] "{row["content"]}"\n')
            f.write("\n")

    print(f"Wrote {len(pushback_positions)} pushback window(s) to {out_path}")


if __name__ == "__main__":
    main()

