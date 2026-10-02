# code-preference

Research pipeline for inferring developer **code-style preferences** from two datasets:

- **SWE-chat** — chat-based coding-agent sessions (`data/swechat_data/`), scored turn-by-turn via an LLM rubric to produce preference vectors.
- **DECODE-style scored pairs** (`data/scored_pairs.jsonl`) — an external code-completion preference dataset used for cross-dataset comparison.

The pipeline scores users on a 14-axis preference profile (e.g. verbosity, naming style, error-handling style, ...), then clusters users into personas and evaluates how discriminable/stable those personas are.

## Repository layout

```
src/            pipeline code, grouped by stage (see src/README.md)
data/           raw source data (gitignored — see data/README.md)
outputs/        generated results from running the pipeline (gitignored — see outputs/README.md)
debug_transcripts/  human-readable session dumps used for manual inspection (gitignored)
```

`data/` and `outputs/` are excluded from git because they contain multi-GB source data and multi-MB
generated artifacts. This repo tracks the *code* that produces them, not the data itself.

## Pipeline stages

1. **Ingestion** (`src/ingestion/`) — sample/select users and sessions from SWE-chat, scan for clean
   session→commit attribution.
2. **Vectorization** (`src/vectorization/`) — the core LLM-scoring pipeline: turn a chat session into a
   per-turn / per-session preference vector.
3. **Diagnostics** (`src/diagnostics/`) — single-session deep dives used while developing the pipeline
   (turn-to-commit alignment, checkpoint/commit reconciliation).
4. **Evaluation** (`src/evaluation/`) — discriminability (ICC), variance-ratio, and coverage metrics on
   the scored vectors.
5. **Clustering** (`src/clustering/`) — k-means persona discovery, stability and prediction-strength
   tests, cross-dataset comparison against DECODE.
6. **Case study** (`src/case_study/`) — a full turn-by-turn report and variance breakdown for one
   representative user.
7. **swe-chat-stats** (`src/swe-chat-stats/`) — a self-contained sub-pipeline that computes dataset-wide
   descriptive statistics for the writeup (separate from the preference-scoring pipeline above).
8. **Turn replay** (`src/turn_replay/`) — paired one-turn Codex replays over reconstructed real SWE Chat
   instances, with strict separation between agent-visible inputs and future-turn evaluation data.

See the README in each `src/` subfolder for what each script does and how to run it.

## Running scripts

All scripts assume they are run from the **repository root** so their default relative paths
(`data/...`, `outputs/...`) resolve correctly, e.g.:

```bash
python src/vectorization/chat_pref_vectorise_v2.py --out-dir outputs/chat_vectors
python src/clustering/cluster_user_profiles.py --out-dir outputs/cluster_out
```

Most scripts also accept `--out-dir` / equivalent flags to control where results are written —
pass a path under `outputs/` to keep generated files out of git.
