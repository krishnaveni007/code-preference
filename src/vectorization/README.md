# vectorization

The core LLM-scoring pipeline: turns a chat session into a preference vector across 14 style axes.

| Script | Purpose | Reads | Writes |
|---|---|---|---|
| `chat_pref_vectorise.py` | v1: LLM-scores user turns into per-turn and per-**subsession** ternary preference vectors | `data/swechat_data/`, calls OpenAI | `outputs/chat_vectors/{chat_turn_vectors,chat_subsession_vectors}.csv`, appends `outputs/chat_vectors/chat_llm_raw_log.jsonl` |
| `chat_pref_vectorise_v2.py` | v2: same idea, aggregates at the **session** level with two aggregation methods (`recent`/`maxtie`) | `data/swechat_data/`, `outputs/source1_sample/selected_sessions.csv`, calls OpenAI | `--out-dir/{chat_turn_vectors,chat_session_vectors}.csv`, appends `chat_llm_raw_log.jsonl` |
| `filter_high_density_sessions.py` | Filters to sessions where ≥N% of turns are preference-active | `outputs/chat_vectors/chat_session_vectors_full100.csv`, `chat_turn_vectors_full100.csv` | `outputs/high_density/` |

`chat_pref_vectorise_v2.py` is the version used for the full run: it was run in batches
(`outputs/source1_pilot_v2/`, `outputs/source1_remaining97/`) and merged into the top-level
`outputs/chat_vectors/chat_turn_vectors_full100.csv` / `chat_session_vectors_full100.csv`, which
most downstream evaluation and clustering scripts consume.

Requires an OpenAI API key (see script `--help` for the expected environment variable).

Run from the repository root, e.g.:

```bash
python src/vectorization/chat_pref_vectorise_v2.py --out-dir outputs/chat_vectors
```
