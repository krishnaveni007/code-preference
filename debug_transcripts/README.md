# debug_transcripts

Human-readable session dumps used for manual inspection during pipeline development. **Excluded
from git** (this folder's contents are gitignored except this file) — they're large, session-specific,
and not consumed by any other script.

- `turns_<session_id>.txt` — produced by `src/diagnostics/load_conversation.py`, a full conversation
  transcript for one session.
- `pushback_windows_<session_id>.txt` — produced by `src/diagnostics/pushback_context.py`, turn
  windows around each "pushback" (user rejecting agent output) in a session.
