# scope_manner: at which level do developer preferences hold?

Branch `discovery`. Code here, outputs in `outputs/discovery/scope_manner/` (data/, figures/, dev/).
Second line of the scope study; the first (pairwise judging of message pairs) is in `discovery/scope/`.

**Question.** When a developer tells a coding agent how to work, is that a one-off instruction, a stable personal
preference, a repository convention, or something everyone wants? For each *matter* (e.g. commit granularity,
draft PRs) we measure how consistent stances are within a user across sessions and across users.

**Inputs** (produced by earlier lines; paths in `common.py`):
- `outputs/discovery/scope/gate/positives_dedup.jsonl`: 19,944 preference-gated SWE-chat v2 user messages, 474 users;
  a message's `id` is its line index.
- `data/swechat_v2/conversations.parquet`: gives the previous agent reply shown to the model as context.

**Running.** `pip install -r requirements.txt`; put `LITELLM_API_KEY` in `<repo>/.env` (CMU gateway; override
`LITELLM_API_BASE`, `SCOPE_CHAT_MODEL`, `SCOPE_EMBED_MODEL` for other providers); then `./run.sh`, or the steps below.
Every LLM step appends to a jsonl and skips ids already done, so interrupted runs resume. LLM outputs drift between
runs even at temperature 0, so counts will differ slightly from the numbers below; the conclusions should not.

## Steps

| # | Script | What it does | Output (data/) |
|---|---|---|---|
| 1 | `restate.py` | One call per message (`prompt_v4.txt`): topic, asked, corrects, condition, trigger, kind (task / manner / both), manner, quote. Never invents a rejected alternative or decides whether something "is a preference". | `restated.jsonl` |
| 1b | `repair_records.py` | Re-runs ids whose JSON was malformed, leniently | appended to `restated.jsonl` |
| 2 | `cluster_manner.py` | Embeds "topic. manner" of every manner/both record, average-linkage cosine clustering at 0.3 | `manner_ids.json`, `E_manner.npy`, `clusters_0.3.json` |
| 3 | `stance.py` | One call per cluster (>=3 members): same matter?, stance groups, mutually exclusive pairs. `--retry` for failed clusters. | `stance_0.3.jsonl` |
| 4 | `merge_matters.py` | Embeds matter names, merges clusters naming the same matter (0.25), re-judges merged groups | `stance_merged_0.25.jsonl`, `matters_final.jsonl` |
| 5 | `consistency.py` | Per matter: share of same-user/different-session pairs and of different-user pairs with the same stance, and with opposite stances; mean over users, bootstrap 95% CI | `consistency.json` |
| 6 | `make_figures.py` | The five figures | `figures/` |

Run of 2026-10-12 (wine-gemini-3-flash-preview, gemini-embedding-001): 7,501 manner statements from 397 users;
2,060 clusters (762 with >=3); 551 matters; 32 matters with >=5 users seen in >=2 sessions.

## Findings (figures/)

- fig1_funnel: 92,246 turns -> 19,944 gated -> 7,501 manner statements -> 551 matters -> 32 well-supported matters.
  Within-user depth is intrinsic (median 4 manner statements per user), not a sample-size issue.
- fig2_within_vs_between: three shapes. Personal (within >> between): documentation maintenance, review
  thoroughness, handling PR review comments, minimal diffs, subagent use, draft PRs, commit organisation. Shared
  (both high): agent-team use, planning scope, explanation length. Situational (both low): push or not, model
  choice, pacing, branching workflow, autonomy.
- fig3_pooled_levels: pooled over all matters the same-stance rate is 0.74 / 0.70 / 0.73 for same session / same
  repo / other repo and 0.40 across users; opposite 0.09 / 0.06 / 0.04 vs 0.14. With stated preferences no repo level
  is observable and session adds no extra consistency: the user is the unit. Users almost never state a scope
  (1.3% of statements mention this repo/project, 4.2% use always/never wording), so a repo rule stated once is
  invisible to a repetition-based measure.
- fig4_population_disagreement: where the population is split (commit attribution, CLAUDE.md use, pushing, draft
  PRs, autonomy) vs where it agrees.
- fig5_rewrite_output: 38% of gated messages carry a manner; 31% of messages are corrections of the agent.

Limits: 5-16 users per matter, wide CIs; one judge for stances, no human validation yet; the name merge left some
duplicates (three "subagent" matters); 68 of 397 users have statements in >=2 repos, too few per matter.

## Thresholds (all heuristic, none tuned on labels)
cluster distance 0.3 (0.4 mixed matters within giant clusters, 0.25 left mostly singletons); cluster size >=3;
merge distance 0.25 (largest merge 7 sub-clusters; 0.3 gave 16); chunk 90 statements per judging call;
fig2 requires >=5 users with >=2 sessions; fig4 requires >=15 users and >=100 cross-user pairs; 300 bootstrap draws.

## dev/
Prompt iterations v1-v3 and the task/manner classifier prompt, with the small test runners. Test sets and model
outputs (30/40/400 messages; haiku, sonnet-4-6, gemini-3.1-pro, flash) and the partial sonnet-4-6 run on ids
0-4647 are in `outputs/discovery/scope_manner/dev/`, together with a rubric-grouping trial that was set aside.
