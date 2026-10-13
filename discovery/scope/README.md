# Scope of preferences: at what level do users' preferences recur?

Branch `discovery`. Code in `discovery/scope/`, outputs in `outputs/discovery/scope/`. Builds on method1
(gate -> statements -> embeddings), re-run on SWE-chat v2. Questions: are a user's preferences consistent
within a session, a repository, across repositories, across agents, or only at population level?

## Decisions (2026-10-08)

- **Data: SWE-chat v2, all users.** `build_conversations.py` turns every human session (tf_eval split v2,
  subagent threads excluded) into method1's `conversations.parquet` format with the native Claude Code /
  Codex / OpenCode parsers: 11,407 sessions, 92,246 user turns, 589 users. Codex's injected context before
  "## My request for Codex:" is cut from `content` (kept in `raw_content`).
- **User identity = SWE-chat `user_id` = the commit author** (resolved via GitHub). Sessions without a
  linked single-author commit have no user (31% of v2 sessions; already removed by the tf_eval cleaning).
  Not backfilled from `owner_id` (possible robustness check: 3,367 such sessions sit in a repo owned by a
  known user). Sessions whose author is a bot or agent identity (`[bot]`, `claude`, `Copilot`,
  `claude@...`, `agent@...`; ~48 sessions) stay in discovery but are left out of same-user statistics.
- **Unit: method1's rewritten statement**, not the raw message (`compare_raw_vs_statement.py`, old data):
  statement similarity separates strictly-same pairs far better (AUC 0.93 vs 0.67), and raw similarity is
  inflated within a user at every level (project vocabulary, language, writing style), 2-4x the
  statement-level lift.
- **Co-occurrence = cosine similarity of two statements >= tau**, counted over all pairs at a level
  (`pair_cooccurrence.py`). No dimension discovery is needed for the measure itself; clusters only describe
  which preferences recur where. Calibration (`strict_probe.py`, 1,447 pairs judged by sonnet-4-6):
  P(strictly same | cos) rises smoothly (0.80-0.85: 6%, 0.88-0.90: 30%, 0.90-0.92: 47%, >0.92: 74%);
  conflicts only occur at 0.75-0.88 and none above 0.88. Main tau 0.88; 0.85 / 0.90 / 0.92 as sensitivity.
  Heuristics to keep in mind: tau, the judge prompt, and the embedding model.
- Why not the earlier "same rubric in two sessions" (method1 `rubric_repetition.py`): a random sample of
  30 same-rubric cross-session pairs had only ~3 strictly-same and ~7 transferable pairs; the measure is
  topic recurrence, and no rubric beat its within-user shuffle baseline.

## Results so far (old release, 149 users; v2 pending)

tau = 0.88, lift over random cross-user pairs (population rate 0.016%):

| level | pairs | users | pooled lift | per-user lift |
|---|---|---|---|---|
| same session | 29,261 | 97 | 28 | 59 |
| same repo, other session | 403,142 | 95 | 13 | 42 |
| other repo, same agent | 8,508 | 14 | 13 | 35 |
| other agent | 14,064 | 8 | 1.4 | 1.5 |

Ordering holds at tau 0.85-0.92. Most users' rate is 0 (median); ~45% of users ever repeat a preference.
The other-repo and other-agent rows rest on 14 and 8 users (top-5 users hold 85% / 99% of the pairs): not
conclusive until v2 (104 multi-repo, 80 multi-agent users).

## v2 run (in progress)

`gate.py --all` on v2 -> `prepare.py positives` -> `summarize.py --all` -> `prepare.py flatten` ->
`cluster_messages.py embed` for statements (`emb_statements/`) and raw messages (`emb_raw/`), chained by
`outputs/discovery/scope/chain.sh`. Then `pair_cooccurrence.py --data v2`.
Early gate positive rate on v2 is ~42% (Claude Code 38%, Codex 62%) vs 23% in method1; to be checked on
the full run (early labels are ordered by session, so few users dominate).

### Which preferences recur (old release, `explore_old.py` + rubric-specific baselines)

Both statements in the rubric; within = same user, same repo, different session; baseline = the same
rubric's cross-user rate (a common preference phrased the same way by everyone is not personal):

| rubric | within | cross-user | lift |
|---|---|---|---|
| Documentation Workflow | 6.9% | 0.31% | 22 |
| Agent Autonomy | 7.5% | 0.81% | 9 |
| Dependency Preference | 3.0% | 0.37% | 8 |
| Execution Parallelism | 18% | 2.8% | 6 |
| Upfront Planning | 3.6% | 0.75% | 5 |
| Test Execution | 0.56% | 0.16% | 3.6 |
| Git Automation | 7.9% | 2.8% | 2.9 (population-level: everyone asks the same) |
| Delivery Phasing | 1.0% | 0.70% | 1.4 |
| Logging / Version Pinning | 0 | - | - |

With a single global baseline Git Automation looked strongly personal (29x); the rubric-specific
baseline shows it is a population habit. Decay (same repo, global baseline): <1 day 22x, 1-14 days ~10x,
14-60 days 4-6x (60+ days: 2,330 pairs, unreliable). Small samples per rubric; redo on v2.
