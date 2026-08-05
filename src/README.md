# src

Pipeline code, grouped by stage. Run in roughly this order:

1. [`ingestion/`](ingestion/README.md) — sample/select users and sessions, scan for clean commit attribution
2. [`vectorization/`](vectorization/README.md) — LLM-score chat sessions into preference vectors
3. [`evaluation/`](evaluation/README.md) — discriminability and reliability metrics on those vectors
4. [`clustering/`](clustering/README.md) — persona discovery and cluster validation
5. [`case_study/`](case_study/README.md) — deep dive on one representative user
6. [`diagnostics/`](diagnostics/README.md) — exploratory single-session tools used during development (not part of the main run)
7. [`swe-chat-stats/`](swe-chat-stats/README.md) — separate sub-pipeline for dataset-wide descriptive statistics

All scripts are run from the repository root, e.g. `python src/vectorization/chat_pref_vectorise_v2.py ...`.
