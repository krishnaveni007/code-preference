#!/usr/bin/env bash
# Full pipeline for the scope_manner line. Run from this directory. Needs LITELLM_API_KEY in <repo>/.env.
# Steps 1-4 call the LLM (about $40 with gemini-3-flash-preview; ~2 h at 32 workers on the CMU gateway).
# Steps 5-6 are local. Every LLM step is resumable: re-running skips ids already done.
set -euo pipefail
cd "$(dirname "$0")"
MODEL=${SCOPE_CHAT_MODEL:-openai/wine-gemini-3-flash-preview}
python restate.py --model "$MODEL" --workers 32          # 1. structured restatement -> data/restated.jsonl
python repair_records.py --model "$MODEL"                # 1b. fix ids whose JSON was malformed
python cluster_manner.py --thr 0.3                       # 2. embed + cluster manner statements
python stance.py --thr 0.3 --model "$MODEL" --workers 16 # 3. stances per cluster
python stance.py --thr 0.3 --model "$MODEL" --retry      # 3b. retry failed clusters leniently
python merge_matters.py --thr 0.3 --merge-thr 0.25 --model "$MODEL"   # 4. merge same-named matters -> matters_final.jsonl
python consistency.py                                    # 5. per-matter within/between-user rates -> consistency.json
python make_figures.py                                   # 6. figures/
