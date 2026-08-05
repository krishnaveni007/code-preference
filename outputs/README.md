# outputs

Generated results from running the pipeline scripts in `src/`. **Excluded from git** (this folder's
contents are gitignored except this file) — regenerate by running the corresponding scripts.

Each subfolder is produced by one or more scripts; see the README in the matching `src/` subfolder
for exactly which script writes where. Notable ones:

- `chat_vectors/` — the core scored output (`chat_turn_vectors_full100.csv`,
  `chat_session_vectors_full100.csv`, and the raw LLM call log). Most evaluation and clustering
  scripts consume the `_full100` files directly.
- `cluster_*/` — different clustering runs (varying `--k`, `--center`, `--row-zscore`, and
  high-density-filtered input). See `src/clustering/README.md`.
- `eval_out/`, `eval_out_highdensity/` — discriminability/coverage metrics.
- `source1_sample/`, `source1_pilot_v2/`, `source1_remaining97/`, `source1_case_study/` —
  intermediate vectorization runs (batches later merged into `chat_vectors/*_full100.csv`, or the
  case-study user's data).
- `swe-chat-stats/` — outputs of the separate `src/swe-chat-stats/` sub-pipeline.

To reproduce any of these, run the producing script with `--out-dir outputs/<name>` from the
repository root.
