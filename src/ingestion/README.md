# ingestion

Scripts for pulling raw data from the SWE-chat dataset and selecting which users/sessions to score.
Run these first, before vectorization.

| Script | Purpose | Reads | Writes |
|---|---|---|---|
| `extract_5_sessions.py` | Pulls raw turns + code-diff units for a small pilot set, for the rubric LLM to score | HF Hub `SALT-NLP/SWE-chat` (network) | `outputs/codepref_pipeline_export/` |
| `sample_users.py` | Selects the 80+20 users/sessions used for chat scoring (high-signal vs random-control arms) | `data/swechat_data/` | `outputs/source1_sample/` |
| `select_case_study_user.py` | Picks the single case-study user closest to the target session/turn count | `outputs/source1_sample/selected_sessions.csv` | stdout only |
| `clean_session_scan.py` | Corpus-wide scan for sessions with clean session→commit attribution | `data/swechat_data/` | `outputs/scan_out/` |
| `clean_cohort_users.py` | Sessions-per-user distribution over the clean session set | `outputs/scan_out/clean_sessions.csv`, `data/swechat_data/` | `outputs/scan_out/clean_sessions_per_user.csv` |
| `chat_coverage.py` | Corpus-wide census of usable agent-vs-committed diffs per session | HF `SALT-NLP/SWE-chat` (streaming) | `outputs/chat_vectors/session_diff_coverage.csv` |
| `user_selection/pick_users.py`, `50users.py`, `user_sess.py` | Candidate-session coverage checks used during manual user selection | HF Hub + `outputs/user_selection/*.csv` | `outputs/user_selection/*.csv` |

Run from the repository root, e.g.:

```bash
python src/ingestion/sample_users.py --out-dir outputs/source1_sample
```
