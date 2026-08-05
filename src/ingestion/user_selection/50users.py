"""
CodePref-Bench: check ALL sessions for ALL 50 stage1/stage2 users.

Scope: your full annotated user pool (the 50 users who appear in
swechat_stage1_screening.csv), not just the 5 you'd already narrowed to.
For each of these users, we pull every session they have in SWE-chat
(most users have more than one — only some of their sessions were flagged
during your pushback screening) and report, for ALL of them:

  - n_commits / n_code_units: code-vector coverage (commits with BOTH
    agent_version and committed_version present)
  - n_pushback_turns: count of turns where conversations.prompt_pushback
    is a real pushback class (not "non_pushback", not null). This is
    dataset-wide LLM annotation, so it works as a richness proxy even for
    sessions your stage1/stage2 screening never looked at.
  - is_screened_session: whether this exact session_id appears in your
    stage1 file (i.e. has real rubric labels from stage2, not just the
    turn-count proxy)

This lets you find the best (user, session) pairs across your whole
annotated pool, not just within the 5 already picked — including sessions
by these same 50 users that were never screened but look promising on
code coverage + pushback-turn proxy.

Requires: pip install datasets pandas
Run locally — needs Hugging Face Hub access. Assumes
swechat_stage1_screening.csv is in the working directory (adjust
SCREENING_CSV path if not).
"""

import json

import pandas as pd
from datasets import load_dataset

SCREENING_CSV = "swechat_stage1_screening.csv"


def main():
    print("Loading screening file to get the 50-user pool...")
    screening = pd.read_csv(SCREENING_CSV)
    USER_IDS = set(screening["user_id"].unique())
    screened_session_ids = set(screening["session_id"].unique())
    print(f"{len(USER_IDS)} users, {len(screened_session_ids)} screened sessions among them")

    print("Loading sessions table...")
    sessions = load_dataset("SALT-NLP/SWE-chat", "sessions", split="train").to_pandas()
    user_sessions = sessions[sessions.user_id.isin(USER_IDS)]
    print(f"Found {len(user_sessions)} total sessions across these {len(USER_IDS)} users")

    print("Loading checkpoints...")
    checkpoints = load_dataset("SALT-NLP/SWE-chat", "checkpoints", split="train").to_pandas()

    print("Loading commits...")
    commits = load_dataset("SALT-NLP/SWE-chat", "commits", split="train").to_pandas()
    commits = commits.dropna(subset=["commit_sha"])

    all_session_ids = set(user_sessions["session_id"])
    print(f"Loading conversations (filtered to {len(all_session_ids)} sessions) for pushback counts...")
    conv_ds = load_dataset("SALT-NLP/SWE-chat", "conversations", split="train")
    conv_ds = conv_ds.filter(
        lambda x: x["session_id"] in all_session_ids and x["turn_type"] == "user_prompt"
    )
    conv = conv_ds.to_pandas()
    pushback_counts = (
        conv[conv["prompt_pushback"].notna() & (conv["prompt_pushback"] != "non_pushback")]
        .groupby("session_id")
        .size()
        .to_dict()
    )

    rows = []
    for _, srow in user_sessions.iterrows():
        sid = srow["session_id"]
        uid = srow["user_id"]

        sess_checkpoints = checkpoints[
            checkpoints["session_pks"].astype(str).str.contains(sid, na=False)
        ]
        cp_pks = sess_checkpoints["checkpoint_pk"].tolist()
        sess_commits = commits[commits["checkpoint_pk"].isin(cp_pks)]

        n_commits = len(sess_commits)
        n_code_units = 0
        for _, c in sess_commits.iterrows():
            try:
                fa = json.loads(c["file_attribution"]) if c["file_attribution"] else {}
            except (TypeError, json.JSONDecodeError):
                fa = {}
            for fpath, d in fa.items():
                if fpath == "__aggregate__":
                    continue
                if d.get("agent_version") is not None and d.get("committed_version") is not None:
                    n_code_units += 1

        n_pushback = pushback_counts.get(sid, 0)

        rows.append({
            "user_id": uid,
            "session_id": sid,
            "repo_id": srow["repo_id"],
            "turn_count": srow["turn_count"],
            "is_screened_session": sid in screened_session_ids,
            "n_commits": n_commits,
            "n_code_units": n_code_units,
            "n_pushback_turns": n_pushback,
        })

    result = pd.DataFrame(rows)
    result["combined_score"] = result["n_code_units"] * 10 + result["n_pushback_turns"] * 2
    result = result.sort_values("combined_score", ascending=False)
    result.to_csv("all_50users_all_sessions_coverage.csv", index=False)

    print(f"\nSaved {len(result)} rows to all_50users_all_sessions_coverage.csv")

    print("\n--- Top 20 sessions overall by combined_score (any of the 50 users, any session) ---")
    print(result.head(20).to_string(index=False))

    print("\n--- Best session per user (top 1 each, sorted by that best score) ---")
    best_per_user = result.sort_values("combined_score", ascending=False).groupby("user_id").head(1)
    best_per_user = best_per_user.sort_values("combined_score", ascending=False)
    print(best_per_user.to_string(index=False))

    print("\nCAVEAT: is_screened_session=False means no stage1/stage2 rubric labels exist for "
          "that session yet. n_pushback_turns is a decent richness proxy but isn't the same as "
          "having actual rubric_id/direction/justification labels. If a top-ranked unscreened "
          "session looks compelling, you'd want to run it through your stage1/stage2 pipeline "
          "before treating it as equivalent to your already-screened picks.")


if __name__ == "__main__":
    main()