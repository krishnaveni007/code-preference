# diagnostics

Single-session deep dives used while developing and sanity-checking the vectorization pipeline.
These are exploratory/debugging tools, not part of the main scoring pipeline — most operate on one
session at a time.

| Script | Purpose | Reads | Writes |
|---|---|---|---|
| `load_conversation.py` | Dumps one session's full conversation to a readable `.txt` transcript | `data/swechat_data/conversations.parquet` | `debug_transcripts/turns_<session_id>.txt` |
| `pushback_context.py` | Dumps ±N-turn windows around each "pushback" turn (user rejecting agent output) | `data/swechat_data/conversations.parquet` | `debug_transcripts/pushback_windows_<session_id>.txt` |
| `checkpoint_sharing.py` | Checks whether a session's checkpoints/commits are shared with other sessions | `data/swechat_data/` | stdout only |
| `commit_report.py` | Checkpoint-wise vs time-wise commit attribution + deep-dive, for one session | `data/swechat_data/` | `outputs/commit_report/` |
| `original_pipeline.py` | Experimental 3-stage turn→commit reconstruction (nearest-commit / line-survival / code-replay) | `data/swechat_data/` | `outputs/pipeline_out/` |
| `band_vs_commit.py` | Compares what the agent wrote in a turn "band" vs what actually landed in linked commits | `data/swechat_data/` | `outputs/band_analysis/` |
| `code_chat_link.py` | Turn-to-code alignment and churn chart for one session | `data/swechat_data/` | `outputs/session4_code/` |
| `turns_vs_commits.py` | One-off plot of turns vs commits for a specific session, using manually transcribed timestamps (not a general reusable script) | hardcoded values | stdout/plot only |

Run from the repository root, e.g.:

```bash
python src/diagnostics/load_conversation.py --session-id <id> --out-dir debug_transcripts
```
