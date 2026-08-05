# data

Raw source data. **Excluded from git** (this folder's contents are gitignored except this file) —
too large and not reproducible-by-us to commit.

Expected contents:

- `swechat_data/` — the SWE-chat dataset (`sessions`, `conversations`, `commits`, `repositories`,
  `session_logs`, `checkpoints` parquet files). Sourced from the HF Hub dataset `SALT-NLP/SWE-chat`;
  see `src/ingestion/` for scripts that pull from it directly.
- `scored_pairs.jsonl` — the external DECODE-style scored code-completion dataset, used by
  `src/clustering/cluster_decode.py` for cross-dataset comparison. This is not produced by any
  script in this repo — it's an external input.

To reproduce this folder, place the above files here before running any pipeline scripts.
