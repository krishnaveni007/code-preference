# Turn-level SWE Chat replay

This package replays one real user turn exactly once under paired `baseline`
and `preference` conditions. It is intentionally separate from the existing
SWE-Together simulated-user evaluation.

## Input schema

```text
eval_instances/<instance_id>/
  repo/                 # reconstructed state immediately before this turn
  conversation.json     # history + current_user_message (agent-visible)
  metadata.json         # copied verbatim into results
  evaluation.json       # privileged; this package never opens it
  preference.md         # optional
```

`conversation.json` must be an object containing a `history` list of
`{"role": "user"|"assistant", "content": "..."}` objects and a non-empty
`current_user_message`. `metadata.json` must be an object; optional metadata is
preserved without imposing a snapshot-confidence taxonomy. If `instance_id` is
absent, the instance directory name is used.

An explicit `--preference-file` takes precedence over
`<instance>/preference.md`. A preference condition without either fails before
Codex is invoked. A baseline-only run does not select or read a preference
file.

## Context and isolation

The harness serializes history losslessly as role-labelled JSON followed by the
current message. That exact rendered prompt is stored in
`supplied_context.json`. Codex receives it on stdin in one `codex exec` call.
There is no simulator and no second user turn.

Only `repo/`, `conversation.json`, and `metadata.json` are loaded into the
`ReplayInstance` type. There is no evaluation-data field or evaluation loader
in this package. `evaluation.json` is neither copied into the working tree nor
opened by the runner.

Both requested working copies are made directly from the immutable `repo/` and
hash-verified before either condition executes. `metadata.environment.setup_command`,
when present, runs independently inside each copied working directory before
the agent. Use `--skip-setup` to suppress it.

The preference condition appends this block to the root `AGENTS.md`, preserving
the original bytes as the prefix:

```text
<!-- BEGIN LEARNED USER PREFERENCES -->

<preference.md>

<!-- END LEARNED USER PREFERENCES -->
```

Codex runs ephemerally with user config and exec-policy rules ignored, a
workspace-write sandbox, JSONL event output, and the repository's own
`AGENTS.md` instructions still active. This avoids accidentally introducing
machine-local preferences into the baseline.

## Commands

```bash
python -m src.turn_replay run --instance eval_instances/user123_session08_turn14 --dry-run
python -m src.turn_replay run --instance eval_instances/user123_session08_turn14
python -m src.turn_replay run --instance eval_instances/user123_session08_turn14 --condition baseline
python -m src.turn_replay run --instance eval_instances/user123_session08_turn14 --condition preference --preference-file preferences/user123.md
```

Batch execution continues after malformed instances and failed Codex runs by
default. `--fail-fast` changes that behavior.

```bash
python -m src.turn_replay batch --instances-dir eval_instances
python -m src.turn_replay batch --instances-dir eval_instances --user-id user123 --session-id session08
python -m src.turn_replay batch --instances-dir eval_instances --snapshot-confidence validated --agent-type claude-code
```

Useful reproducibility flags are `--model`, `--reasoning-effort`, `--timeout`,
`--codex-binary`, `--work-dir`, `--results-dir`, and `--overwrite`.

## Artifacts

```text
results/<instance_id>/
  run_metadata.json
  baseline/
    result.json
    supplied_context.json
    final_response.txt
    transcript.jsonl
    stdout.log
    stderr.log
    patch.diff
    git_status.txt
    original_AGENTS.md       # when present
  preference/
    ...same files...
    injected_AGENTS.md
```

Setup stdout/stderr are also saved when a setup command exists. `result.json`
preserves all input metadata and records hashes, commit, CLI version, exact
command/configuration, timestamps, status, context, AGENTS.md contents,
response, changed files, preference provenance, and errors.
