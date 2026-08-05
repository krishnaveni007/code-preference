"""
CodePref-Bench: check ALL sessions for the 5 selected users.

The stage1/stage2 screening only covers sessions that had at least one
flagged pushback turn. These developers may have OTHER sessions in
SWE-chat that never got flagged but still have strong code-vector
coverage (commits with both agent_version and committed_version).

This script pulls every session by each of the 5 selected users and
reports, for ALL of them (not just screened ones):
  - n_commits / n_code_units: code-vector coverage (as before)
  - n_pushback_turns: count of turns where conversations.prompt_pushback
    is a real pushback class (not "non_pushback" and not null). This is
    dataset-wide LLM annotation, not from your stage1/stage2 files, so it
    works as a cheap richness proxy even for unscreened sessions.

Use both numbers together to decide whether to:
  (a) swap in a different session for a user with better combined coverage,
  (b) add a second session per user to get more data, or
  (c) stick with the originally screened session (it has your own
      stage1/stage2 rubric labels attached, which nothing found here has
      — see caveat at the bottom).

Requires: pip install datasets pandas
Run locally — needs Hugging Face Hub access.
"""

import json

import pandas as pd
from datasets import load_dataset

# The 5 selected users, with the session already confirmed via stage1/stage2 screening
SELECTED = {
    "timelabsad-dot": "2a84a5a3-0636-4047-aa95-342b1ebe7d83",
    "ckeith26":        "443d9387-9119-4760-a36b-33768f03ff49",
    "maoxiaoke":       "ba75c8c5-be57-4405-93d8-1619f0f1efbb",
    "itsmaleen":       "4810d459-5b4b-4b78-9518-a0eb00118590",
    "ujuc":            "2b3756eb-8c69-4980-ac0a-8d2c534931d8",
}
USER_IDS = set(SELECTED.keys())


def main():
    print("Loading sessions table...")
    sessions = load_dataset("SALT-NLP/SWE-chat", "sessions", split="train").to_pandas()
    user_sessions = sessions[sessions.user_id.isin(USER_IDS)]
    print(f"Found {len(user_sessions)} total sessions across these {len(USER_IDS)} users "
          f"(vs. 1 screened session each)")

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
    # Real pushback = annotated, and not the "non_pushback" catch-all class
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
            "is_screened_session": sid == SELECTED[uid],
            "n_commits": n_commits,
            "n_code_units": n_code_units,
            "n_pushback_turns": n_pushback,
        })

    result = pd.DataFrame(rows)
    # Combined coverage score: code_units weighted higher since it's the
    # scarcer resource, pushback turns as the conversational richness proxy.
    result["combined_score"] = result["n_code_units"] * 10 + result["n_pushback_turns"] * 2
    result = result.sort_values(
        ["user_id", "combined_score"], ascending=[True, False]
    )
    result.to_csv("user_all_sessions_coverage.csv", index=False)

    print("\nAll sessions per selected user, sorted by combined_score within each user:")
    print(result.to_string(index=False))
    print("\nSaved to user_all_sessions_coverage.csv")

    print("\n--- Per-user best non-screened session (if better than the screened one) ---")
    for uid, screened_sid in SELECTED.items():
        user_rows = result[result.user_id == uid]
        screened_row = user_rows[user_rows.session_id == screened_sid]
        screened_score = screened_row.combined_score.iloc[0] if len(screened_row) else 0
        best_other = user_rows[user_rows.session_id != screened_sid].head(1)
        if len(best_other) and best_other.combined_score.iloc[0] > screened_score:
            b = best_other.iloc[0]
            print(f"{uid}: screened session has combined_score {screened_score} "
                  f"({screened_row.n_code_units.iloc[0]} code_units, "
                  f"{screened_row.n_pushback_turns.iloc[0]} pushback turns), but session "
                  f"{b.session_id} has {b.combined_score} ({b.n_code_units} code_units, "
                  f"{b.n_pushback_turns} pushback turns) — NOT preference-screened, see caveat")
        else:
            print(f"{uid}: screened session (combined_score {screened_score}) is already the "
                  f"best or tied — no swap needed")

    print("\nCAVEAT: any session found here that ISN'T the originally screened one has no "
          "stage1/stage2 pushback/rubric labels attached. It might be great for the CODE half "
          "of your pipeline but you'd have no confirmed developer-preference signal on the "
          "conversation side for it — you'd need to either run fresh screening on it or use it "
          "only for code-vector methodology work, not as one of your 5 paired case studies.")


if __name__ == "__main__":
    main()