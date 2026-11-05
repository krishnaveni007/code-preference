# Preference-memory turn replay

This package tests whether preference memory changes what a coding agent (Claude Code) does at one real turn of a
recorded SWE-chat session.

It takes a turn where a user restated a preference they had already given in an earlier session. It restores the
repo and the conversation to just before that turn, then replays the turn with and without memory. A judge model
decides whether each replay would still draw the user's pushback.

This package is separate from [`../turn_replay/`](../turn_replay/README.md), the paired one-turn Codex replays.

There are two kinds of turn, both taken from the repeated-preference trajectories (see [Inputs](#inputs)):

| Kind | The recorded turn | What is replayed | Judge question |
|---|---|---|---|
| **C**: correction | The user corrected the agent, restating the preference | The agent's response to the prompt **before** the correction | Would this user still need to make that correction? (`correction_warranted`: yes / partly / no) |
| **U**: unprompted | The user stated the preference unprompted | The same prompt, in full and **ablated** (the preference removed) | Does the turn follow the stated preference? (`follows_preference`: yes / partly / no) |

**Pushback likely** means C = yes or partly, and U = no or partly.

The recorded turn is judged as well, as a calibration check. A C case whose recorded turn is not judged
`correction_warranted = yes`, or a U case whose recorded turn is not `follows_preference = yes`, can't separate the
arms. Report those cases apart.

## Arms

| Arm (run dir) | Kind | Prompt | Memory |
|---|---|---|---|
| `c0_r1` | C | original | none |
| `mem_r1` | C | original | real: the user's recovered Claude Code memory store as of the turn |
| `c2p_r1` | C | original | C2 preference notes (c2p pointer format) |
| `full_c0_r1` | U | full | none (upper reference) |
| `ablated_c0_r1` | U | ablated | none |
| `ablated_mem_r1` | U | ablated | real |
| `ablated_c2p_r1` | U | ablated | C2 |

**Real memory.** These arms run only when the recovered stores contain an observation of `MEMORY.md` at or before
the turn, taken from the target session's own Claude Code project folder (its cwd slug).
- Several users can share a repo, so only that one folder is used.
- Nothing is interpolated.

**C2 memory** (`render_c2p.py`).
- Built from the coded preference statements in `explorer_data.json`, using only sessions that started **before**
  the target session.
- Quotes are checked verbatim against the user's prompts.
- Layout, stances, counts and examples are deterministic. Only the "How to apply" and conflict lines are written by
  a model.
- The format spec is [`spec/c2_remember_pointer.md`](spec/c2_remember_pointer.md), and `c2p_format.py` parses and
  lints it.

**How memory reaches the agent.** Memory is mounted as the project's auto-memory. Claude Code loads `MEMORY.md`
silently, and a note counts as used only if the agent Reads it.

## How a turn is restored

- **Conversation:** the SWE-chat transcript is cut at the target prompt, and the original cwd is rewritten to
  `/workspace/<name>`. Claude Code resumes it with `claude -p --resume <sid> --fork-session`, so the agent sees the
  real session history.
- **Repo:** a sealed clone at the **anchor** commit. Only one branch is kept; there is no remote, no reflog, and no
  later commit is reachable.
- **Anchor:** the `git commit` output (`[branch sha]`) after the last file change before the turn. The first
  `git log` line is used instead only when nothing had changed files earlier in the session. Turns where files
  changed after the anchor are rejected.
- **State check:** `prep/state_check.json` compares the transcript's last Read/Write/Edit of each file with the
  restored repo. A mismatch tagged `(after anchor)` means the commit is not the state the agent saw, so drop the
  case.
- **Sandbox:** the agent runs in Docker (`Dockerfile`: Python 3.12, Node 22, git, ripgrep, pnpm, uv, Claude Code
  2.1.273) with `claude-sonnet-5-5[1m]`. Without `[1m]`, contexts over about 200K tokens are silently auto-compacted.
- **Judge:** a tool-less `claude -p` call to `claude-opus-5-5` with a JSON schema. This needs Claude Code ≥ 2.1.280
  on the host.

## Inputs

None of these are in git: they quote real participants' messages. Put them under `data/` (gitignored) or point the
environment variables at them.

| Input | Used by | Variable (default) |
|---|---|---|
| SALT-NLP/SWE-chat snapshot: `transcripts*/`, `sessions.parquet`, `conversations.parquet` | all steps | `SWECHAT_SNAPSHOT` (`data/swechat_hf`) |
| Repeated-preference trajectories: `trajectories.html`, or its extracted JSON list | `find_candidates.py` | passed as `--traj` |
| Coded preference statements: `explorer_data.json` | C2 arm | `PREF_EXPLORER_DATA` (`data/explorer_data.json`) |
| Recovered memory stores: `observations.jsonl`, `stores/` | real-memory arm | `SWECHAT_MEMORY_STORES` (`data/swechat_memory_stores`) |

Other settings (see `common.py`):

| Variable | Default |
|---|---|
| `TURN_REPLAY_DATA` (all outputs) | `data/pref_memory_replay` |
| `TURN_REPLAY_CASES` | `$TURN_REPLAY_DATA/cases` |
| `TURN_REPLAY_CLONES` | `$TURN_REPLAY_DATA/clones` |
| `TURN_REPLAY_ENV_FILE` (token fallback) | `.env` |
| `TURN_REPLAY_IMAGE` | `turn-replay:cc2.1.273` |
| `TURN_REPLAY_JUDGE_CC` | the `--cc-bin` default |

## Setup

You need Docker, Python ≥ 3.10 with `pandas` and `pyarrow`, Node (for the judge CLI), and a Claude Code OAuth token.

```bash
docker build -t turn-replay:cc2.1.273 src/pref_memory_replay/
npm install --prefix ~/.cache/cc-judge @anthropic-ai/claude-code@2.1.318     # judge CLI (Opus 5.5 needs >= 2.1.280)
export TURN_REPLAY_JUDGE_CC=~/.cache/cc-judge/node_modules/.bin/claude
export CLAUDE_CODE_OAUTH_TOKEN=...                                            # or CLAUDE_CODE_OAUTH_TOKEN=... in .env
```

Run every command below from the repository root.

## Adding examples

### 1. Find replayable turns

```bash
python src/pref_memory_replay/find_candidates.py --traj data/trajectories.html \
    --out data/pref_memory_replay/candidates.json
```

This only considers **repeat** statements: the same (user, mode, dimension, side) group was stated in an earlier
session, so memory could carry it. For each one it locates the prompt in the transcript and finds a state anchor.
Each row has either `reject` (the reason) or the cut line, sha, branch and context tokens.

### 2. Select turns

Copy the rows you want into a selection file and give each one `"tier": "pick"`:

```bash
python - <<'EOF'
import json
rows = [r for r in json.load(open("data/pref_memory_replay/candidates.json")) if not r.get("reject")]
pick = [r for r in rows if r["user"] in {"someuser"}]          # your choice
json.dump([r | {"tier": "pick"} for r in pick], open("data/pref_memory_replay/selection.json", "w"), indent=1)
EOF
```

Prefer turns whose context fits in the 1M window (`ctx_tokens`) and whose repo is still on GitHub.

### 3. Write case files

```bash
python src/pref_memory_replay/make_cases.py --selection data/pref_memory_replay/selection.json --tier pick
```

This writes `$TURN_REPLAY_CASES/<user>_<C|U>_<session8>.json`.
- For **U** cases, one model call writes the ablated prompt (`prompt_override`, plus `ablation_removed`).
- **Review every ablation by hand.** It must remove the preference and nothing else, and it must still ask for the
  task. If you fix one, set `"ablation_hand_edited": true`, and reruns will keep your edit.
- Optional: add a `"setup"` shell string (for example a dependency install) that runs in the repo before the agent.

### 4. Prepare inputs

```bash
python src/pref_memory_replay/prepare_cases.py --cases <id>,<id>
```

This does four things per case:
- Clones the repo once into `$TURN_REPLAY_CLONES`, fetching the sha from GitHub if it isn't on a branch.
- Builds `prep/` (sealed repo, truncated transcript, prompts, state check).
- Snapshots `mem_real/` when a store exists.
- Renders `mem_c2p/`.

It prints `STATE MISMATCH after anchor: [...]` when a case should be dropped. Steps whose output already exists are
skipped; add `--force` to redo them, or `--no-c2p` to skip the C2 render.

### 5. Run and judge

```bash
python src/pref_memory_replay/run_all.py --cases <id>,<id> --jobs 4
```

This runs every arm in Docker, judges it into `<arm>/judge.json`, and judges the recorded turn into
`judge_recorded.json`. It can be resumed:
- An arm with `run.json` is not run again, and an arm with `judge.json` is not judged again.
- An interrupted arm has no `run.json`, so it reruns from scratch.
- `--rejudge` re-runs only the judge, for example after changing the rubric in `judge_turn.py`.

Over the first 38 arms, each took up to about 11 minutes and cost $0.12 to $4.50 (median $0.80), depending on
context size.

To run a single arm by hand:

```bash
D=data/pref_memory_replay; C=$D/cases/<id>.json
python src/pref_memory_replay/run_turn.py $C --prep $D/<id>/prep --out $D/<id>/c2p_r1 --memory-dir $D/<id>/mem_c2p
python src/pref_memory_replay/judge_turn.py $C --run $D/<id>/c2p_r1 --out $D/<id>/c2p_r1/judge.json
```

### 6. View

```bash
python src/pref_memory_replay/build_replay_viewer.py      # writes data/pref_memory_replay/viewer/turn_replay.html
python -m http.server 8732 --bind 127.0.0.1 --directory data/pref_memory_replay/viewer
```

The viewer has one tab for C and one for U, each with:
- An overview grid of cases × arms, showing pushback likely and the number of notes opened.
- A per-case table with User turn (n−1), a summary of the agent response, and a summary of the judge's reasoning.
- Side-by-side arm detail: memory index, notes opened, full turn and diff.

Summaries are one Opus call per arm, cached in `<case>/summaries.json`. A case appears once at least one replay
has been judged.

## Files

| File | Role |
|---|---|
| `common.py` | Paths, models, image, token lookup |
| `inspect_transcript.py` | List a transcript's real prompts with agent activity between them (`--around N`) |
| `find_candidates.py` | Locate repeat C/U turns in transcripts; anchor and replayability check |
| `make_cases.py` | Selection → case JSON; model-written U ablation |
| `prepare_turn.py` | One case → `prep/` (sealed clone, truncated transcript, prompts, state check) |
| `snapshot_store.py` | Real memory store as of a timestamp (one project folder), plus a staleness report |
| `render_c2p.py`, `c2p_format.py`, `spec/` | C2 preference notes: render, format spec and lint |
| `prepare_cases.py` | Runs clone + `prepare_turn` + `snapshot_store` + `render_c2p` for each case |
| `run_turn.py` | One arm in Docker → `stream.jsonl`, `diff.patch`, `run.json` |
| `judge_turn.py` | Judge a replayed or recorded turn (C or U rubric) |
| `run_all.py` | All arms of all cases, N at a time, resumable |
| `build_replay_viewer.py`, `replay_viewer_template.html` | Viewer |
| `Dockerfile` | Agent sandbox image |

## Data handling

Case files, prompts, run outputs and the viewer all quote real participants' messages and paths. Keep them under
`data/`, which is gitignored, and don't paste them into external services.
