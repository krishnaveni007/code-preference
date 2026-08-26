# preference context extraction

`preference_context.py` builds a deterministic LLM-judge input for any
SWE-Chat session and conversational user turn. Its user boundaries exactly
match the turn-to-commit mapper: `role == "user"` and
`is_conversational == True`.

The returned packet contains the previous and next user messages, the
judge-relevant rows in the two intervening trajectories, bounded content
projections, normalized code deltas, and per-action/turn commit-survival
evidence. Selection is deterministic and performs no semantic summarization.

System events, queue bookkeeping, content-free file-history snapshots,
timestamps, repeated checkpoint identifiers, raw tool-input duplicates, and
repeated per-edit commit edges are omitted from the judge-facing packet.

Tool uses and their results are merged by `tool_call_id`; the identifier is
then omitted. The redundant `role: tool` marker is also omitted because the
`tool` key identifies the entry. File-history snapshots are retained only when
they contain literal file contents. Snapshot rows containing backup references
alone are omitted.

Python:

```python
from src.extraction.preference_context import load_preference_context

context = load_preference_context(
    "data/swechat_data",
    "651abcf9-fbfa-4b51-9e92-be97fa1b1884",
    78,
)
```

CLI:

```bash
python3 -m src.extraction.preference_context \
  --data-dir data/swechat_data \
  --session-id 651abcf9-fbfa-4b51-9e92-be97fa1b1884 \
  --turn-number 78 \
  --out outputs/preference_context_651abcf9_t78.json
```
