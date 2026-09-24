# OpenCode × GPT-5.4 preference evaluation

This study uses one replicate: five conditions for Soph and khaong, and four
for Nagi-ovo, for 41 cells total. `04_direction_plus_user_messages` is
intentionally excluded. Nagi-ovo also excludes `02_high_confidence_only`
because that profile contains no high-confidence preference and duplicates
the baseline intervention.

## Fixed action-agent settings

- Harness: OpenCode `1.15.13`, installed inside each Harbor sandbox
- Provider/model: `openai/gpt-5.4-2026-03-05`
- Reasoning effort: `high`
- Sandbox: E2B
- OpenCode resumes the same session ID across simulated user turns
- Each rendered `CLAUDE.md` profile is prepended to the task repository's
  `AGENTS.md` before the clean `harbor-base` checkpoint is recorded
- OpenCode's OpenAI provider has `setCacheKey=true`; actual cache reads are
  audited from `trajectory.json`, not inferred from configuration

The OpenCode executable does not need to be installed on the host. Harbor
installs the pinned CLI inside each E2B sandbox.

## Credentials

The runner loads `.env` and `third_party/SWE-Together/.env`. Required for the
current smoke path:

```text
OPENAI_API_KEY
E2B_API_KEY
GHCR_USER
GHCR_TOKEN
```

Do not place credentials in profile files, plans, logs, or shell history.

## Plan

```bash
python3 scripts/run_opencode_profile_eval.py plan --smoke
python3 scripts/run_opencode_profile_eval.py plan
```

The smoke plan contains the shortest task, `cli-task-c425e4`, under:

1. `01_baseline`
2. `06_descriptive_no_polarity`

The full plan orders tasks from shortest to longest and deterministically
shuffles the five conditions within each task.

## Run the smoke test

Dry-run first:

```bash
python3 scripts/run_opencode_profile_eval.py run --smoke
```

Execute the smoke test with:

```bash
python3 scripts/run_opencode_profile_eval.py run --smoke --execute \
  --approved-budget-usd <OPTIONAL_ESTIMATE>
```

## Verify prompt caching

```bash
python3 scripts/run_opencode_profile_eval.py audit-cache --smoke \
  --require-cache-hit
```

The gate applies when an OpenCode trial made more than one model turn. It fails
if a resumed trial reports zero cached input tokens. The audit records prompt
tokens, cached input tokens, cache-hit rate, model-turn count, and trajectory
path in `outputs/longitudinal_pilot/opencode_cache_audit.json`.

## Evaluate the smoke test

```bash
python3 scripts/run_opencode_profile_eval.py evaluate --smoke --execute \
  --approved-budget-usd <OPTIONAL_ESTIMATE>
```

The evaluator retains verifier reward, pass status, user turns, corrections,
nudges, user-correction score, intent coverage, input, uncached-input,
cached-input, cache-write, generated-output and reasoning tokens, estimated
cost, wall time, patch size, and infrastructure validity. The run plan retains
the exact model/harness settings and profile/task provenance. The headline
comparison is:

| Condition | Verifier reward | User turns | Corrections | Nudges | Intent coverage |
|---|---:|---:|---:|---:|---:|

Only after the smoke trial passes profile-injection, session-resume, cache,
validity, and metric-completeness checks should the full run be started:

```bash
python3 scripts/run_opencode_profile_eval.py run --execute \
  --parallel 3
```

For the detached, resumable full run (three concurrent Harbor/E2B cells):

```bash
./scripts/start_remaining_opencode_profiles.sh
```

The launcher uses `caffeinate`, writes a PID and log under
`outputs/longitudinal_pilot/opencode_full_run/`, and relies on
`--skip-existing` to omit completed cells. It survives terminal closure but
not a laptop shutdown; after a restart, launch the same script again.
