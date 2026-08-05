"""
CodePref-Bench: candidate coverage check.

We already ranked sessions by conversational preference-signal richness
(turn count, rubric diversity, label volume). But that ranking says nothing
about whether the session actually has usable code-vector data — i.e.
commits with BOTH agent_version and committed_version present in
file_attribution.

This script checks that second dimension across a WIDER candidate pool
(the top 25 by richness, not just the 5 we originally picked), so the final
5 can be chosen on both axes instead of getting surprised after the fact
(as happened with SnowingFox and gagan114662, which turned out to have zero
usable code_units rows).

Requires: pip install datasets pandas
Run locally — needs Hugging Face Hub access.
"""

import json

import pandas as pd
from datasets import load_dataset

# Load the broader candidate pool (top 25 by richness score from stage1/stage2)
candidates = pd.read_csv("candidate_pool_top25.csv")  # user_id, session_id, repo_id, score
CANDIDATE_IDS = set(candidates["session_id"])


def main():
    print("Loading checkpoints...")
    checkpoints = load_dataset("SALT-NLP/SWE-chat", "checkpoints", split="train").to_pandas()

    print("Loading commits...")
    commits = load_dataset("SALT-NLP/SWE-chat", "commits", split="train").to_pandas()
    commits = commits.dropna(subset=["commit_sha"])

    rows = []
    for _, row in candidates.iterrows():
        sid = row["session_id"]

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

        rows.append({
            "user_id": row["user_id"],
            "session_id": sid,
            "repo_id": row["repo_id"],
            "richness_score": row["score"],
            "n_commits": n_commits,
            "n_code_units": n_code_units,
        })

    result = pd.DataFrame(rows)
    # Combined score: keep richness as the primary signal, but zero out (or
    # heavily penalize) anything with no code units, since that half of the
    # pipeline needs at least a few points to plot anything.
    result["combined_score"] = result["richness_score"] + result["n_code_units"] * 10
    result = result.sort_values("combined_score", ascending=False)

    result.to_csv("candidate_pool_with_coverage.csv", index=False)
    print("\nFull candidate pool with code coverage (sorted by combined score):")
    print(result.to_string(index=False))
    print("\nSaved to candidate_pool_with_coverage.csv")
    print("\nSessions with ZERO code_units (would need to be dropped if code-vector coverage matters):")
    print(result[result.n_code_units == 0][["user_id", "session_id", "richness_score"]].to_string(index=False))


if __name__ == "__main__":
    main()