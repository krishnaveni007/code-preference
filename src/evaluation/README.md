# evaluation

## Phase-0 code-preference audit

`phase0_candidate_audit.py` builds a small chronological human-review set for
the code-preference pairwise-prediction pilot. It selects users with at least
three clean past sessions, searches held-out future sessions for correction or
rejection turns followed by exact/strong committed edits, and writes candidate
events with conversation and edit evidence. It does **not** assign preference
labels; reviewers complete the empty `audit_*` columns.

```bash
python3 src/evaluation/phase0_candidate_audit.py \
  --out-dir outputs/phase0_candidate_audit
```

Metrics that assess how discriminable and reliable the preference vectors are, once vectorization
has produced `outputs/chat_vectors/chat_session_vectors_full100.csv` / `chat_turn_vectors_full100.csv`.

| Script | Purpose | Reads | Writes |
|---|---|---|---|
| `evaluate_full_run.py` | Main discriminability (ICC), coverage, and method-agreement evaluation at full scale | `outputs/chat_vectors/chat_session_vectors_full100.csv` | `outputs/eval_out/` |
| `conditional_discriminability.py` | ICC restricted to nonzero-firing sessions, with bootstrap confidence intervals | session-vectors CSVs | `outputs/eval_out/discriminability_conditional_<method>.csv` |
| `variance_ratio.py` | Naive (unbounded) between/within variance-ratio discriminability, session-level | session-vectors CSVs | `outputs/eval_out/discriminability_varratio_<...>.csv` |
| `variance_turn_level.py` | Same variance-ratio metric but on raw turn-level scores | `outputs/chat_vectors/chat_turn_vectors_full100.csv` | `outputs/eval_out/discriminability_varratio_turnlevel_*.csv` |
| `plot_histogram.py` | Bar chart of the % of preference-active turns per session | `outputs/chat_vectors/chat_turn_vectors_full100.csv` | `outputs/eval_out/` |
| `rubric_item_correlation.py` | Pairwise rubric-trigger correlation with user-cluster bootstrap, FDR correction, arm diagnostics, report, and heatmap | turn-vectors CSVs | `outputs/eval_out/rubric_correlations/` |
| `convergence_analysis.py` | Per-user cumulative-vector convergence over sessions | `data/swechat_data/`, session-vectors CSVs | `outputs/convergence_out/` |

Run from the repository root, e.g.:

```bash
python src/evaluation/evaluate_full_run.py --out-dir outputs/eval_out
```

Evaluate whether rubric items fire on the same turns:

```bash
python3 src/evaluation/rubric_item_correlation.py \
  --turn-vectors outputs/chat_vectors/chat_turn_vectors_full100.csv \
  --out-dir outputs/eval_out/rubric_correlations
```
