# Method 1: discovering preference rubrics from SWE-Chat user turns

Branch `discovery`. Code in `discovery/method1/`, outputs in `outputs/discovery/method1/` (final line under `final/`).
All model calls go through the CMU LiteLLM gateway (`LITELLM_API_KEY` in `.env`); models used are
`wine-claude-haiku-4-5` for per-message labelling and rewriting, `wine-claude-sonnet-4-6` for naming and rubric
drafting, `wine-gemini-embedding-001` (3072-d) for embeddings.

## Goal

Find the dimensions on which SWE-Chat users express preferences about how an AI coding agent should work or how
code should be written, bottom-up from the conversations, and write them as rubrics (Name / Description / High /
Low / N-A). The existing 14 rubrics in `src/extraction/preference_judge.py` were written top-down; this line asks
what the data itself supports.

## Data

`data/swechat_data/conversations.parquet`: 62,544 conversational user turns (`turn_type == "user_prompt"`) from
187 users, 5,795 sessions. The dataset's own `prompt_pushback` label was inspected and found too noisy to use
(`takeover` is mostly the word "commit"; `correction` contains pasted review reports and plain questions), so it is
not used anywhere below except as a reference column.

## Pipeline

### Step 1 - which turns express a preference (`gate.py`)

**Deterministic prefilter.** Drops turns the developer did not type or that carry no content: any harness tag
(`<task-notification>`, `<command-message>`, `<system_instruction>`, `<bash-input>`, ... anywhere in the text),
`<usage><total_tokens>` blocks, "Full transcript available at:", "Base directory for this skill:",
`[Request interrupted by user]`, `[Image ...]`, "tool loaded.", context-overflow continuation summaries, bare slash
commands, and turns under 5 words.

| dropped by | turns |
|---|---|
| under 5 words | 8,776 |
| harness tag at start | 7,424 |
| `[Request interrupted]` / `[Image]` / `tool loaded.` | 3,395 |
| harness block (`<usage>`, transcript path, skill header) | 1,281 |
| continuation summary | 1,077 |
| harness tag inside the text | 692 |
| bare slash command | 102 |
| **kept** | **39,797** (39,774 distinct session+turn pairs) |

**Classifier.** One call per turn, input truncated to the first 50 words, output `{"label": 0|1}`. Prompt
(system message), verbatim:

```
You are labeling single messages a developer sent to an AI coding agent. You are labeling whether this message shows any preference beyond code details, e.g., not concrete debugging messages, but indicating a more general preference on both codes and interactions. Output 1 if it's preference related message, output 0 for everything else.

Much of the input is not written by the developer: pasted tool output, review findings, CI logs, terminal dumps, and configuration or skill files injected by the agent harness. Output 0 for all of it, even when it is full of preference language.
```

Evaluation: 57 hand-labelled turns (two random samples, half drawn from turns the dataset labelled as pushback).
Labels follow the user's definition: any preference about the code or about the interaction counts, including
one-off instructions; concrete bug reports, specific fix requests, questions and harness content do not. Against
those labels: 48/57 agree, precision 0.81, recall 0.68 (19 positives). A version of the prompt with two added
paragraphs defining "preference" more narrowly scored higher on an earlier, narrower labelling and worse on the
corrected one (recall 0.26), so the prompt was kept as the user wrote it.

Full run: 39,774 turns, 0 failed calls, **9,215 labelled 1 (23.2%)**, 158 of 182 users have at least one. By the
dataset's own label: 38.5% of `correction`, 13.5% of `non_pushback`, 5.5% of `failure_report` turns are labelled 1.
Exact-duplicate texts (795, mostly harness-injected instruction files that passed the classifier, e.g. one review
skill 239 times) were removed, leaving **8,420 messages**.

Cost: ~$19 (about 400 input tokens per call).

### Step 2 - rewriting each message as preference statements (`summarize.py`)

Each message is rewritten into 0-3 short statements in the form "Wants the agent to ..." / "Asks the agent to
...", with project names removed. Prompt, verbatim:

```
A developer sent this message to an AI coding agent. Rewrite the preference it expresses as a general rule the developer would want followed on any project, not just this one.

Rules for each statement:
- One sentence, at most 20 words, starting with "Wants the agent to ..." or "Asks the agent to ...". State only what the developer asked for. Do not invent an alternative they rejected unless they named it themselves.
- No names of files, functions, variables, libraries, frameworks, products, commands, or people. Use generic words instead: "the config file", "a helper function", "an external library".
- State the disposition, not the task. "Wants the login bug fixed" is a task; "Wants bugs fixed one at a time with confirmation in between" is a disposition.
- Specific enough that two reasonable developers could disagree with it. "Wants good code" is too vague to keep.

Most messages carry one preference; write up to three if there are clearly several. If the message carries no preference that would transfer to another project, return an empty list.
```

Result: **9,150 statements** from 8,420 messages; 2,011 messages (24%) returned an empty list (routine
instructions with nothing transferable). Mean 1.09 statements per message. This is the unit clustered below; each
statement keeps its `session_id`, `turn_number`, `user_id`.

Example (original -> statements):

- "can your run it, it fails and can you try to fix it and retry until it's working, if you need to change code
  outside of tests please ask me first" -> "Asks the agent to run code, fix failures iteratively until working." /
  "Asks the agent to request permission before modifying non-test code."
- Chinese and Japanese originals are rewritten into English statements.

An earlier form, "Prefers X over Y", was rejected because the model invented the alternative Y when the message
did not name one.

Cost: ~$7.

### Step 3 - clustering (`cluster_messages.py embed`, `cluster_hdbscan.py`)

Statements are embedded (3072-d, L2-normalised), reduced to 10 dimensions with UMAP (`n_neighbors=30`,
`min_dist=0`, cosine, seed 0), and clustered with HDBSCAN.

**HDBSCAN settings.** With the default excess-of-mass selection, `min_cluster_size >= 50` collapses to two giant
clusters with 0% noise; `leaf` selection with `min_samples=10` gives evenly sized clusters:

| min_cluster_size | clusters | noise | median size | max size |
|---|---|---|---|---|
| 30 | 60 | 50% | 62 | 261 |
| 50 | 39 | 51% | 95 | 309 |
| **100** | **25** | **43%** | **169** | **791** |
| 200 | 14 | 36% | 349 | 757 |

`min_cluster_size=100` (about 1% of statements) was chosen. Result: **25 clusters covering 57% of statements;
43% unassigned**. A random sample of unassigned statements shows specific one-off requests ("graceful shutdown
should process pending items first"), which is what the noise label is meant to catch.

**Reproducibility.** UMAP is stochastic. The 10-d UMAP + HDBSCAN was repeated with two other seeds; clusters
were matched one-to-one to the seed-0 clusters by centroid cosine similarity (Hungarian assignment) and compared
by member overlap (Jaccard). Mean Jaccard 0.64 / 0.75; **17 of 25 clusters have Jaccard >= 0.5 under both
seeds**. The unstable ones are mostly the planning / autonomy region (plan first, act autonomously) and small
clusters (logging, parallelism, naming).

Why HDBSCAN rather than k-means: k-means was run first (k=20/40, three seeds). Because every point must be
assigned, k-means produced several clusters of unrelated leftover statements (negative silhouette, ten random
members on ten topics) that looked like categories once named; only ~half of its clusters reappeared across seeds.
HDBSCAN's noise label removes that failure mode and its clusters reappear more reliably.

### Step 4 - naming clusters (`name_clusters.py`)

For each cluster the model sees 30 random members and the 30 nearest non-members and must write a name (<= 10
words) and one sentence that fit the inside and exclude the outside. Table with examples:
`final/hdbscan/categories_hdbscan_mcs100.md`. Map: `final/hdbscan/map_mcs100.png` (2-d UMAP for
position only; identity by label, not colour).

### Step 5 - from clusters to rubrics (`draft_rubrics.py`, `attach_originals.py`, `consolidate_rubrics.py`, `finalize_rubrics.py`)

1. **Draft, one per cluster.** From the cluster's name, description and 30 random statements the model writes the
   question, side A (what this cluster's developers want), side B (the opposite), and quotes up to three
   statements from the 30 that actually take side B. Every quoted statement was checked to exist verbatim in its
   cluster (75/75), and each quote is stored with its original user message.
2. **Consolidate.** The model reads all 25 drafts at once and merges those that are two answers to one question or
   the same question twice. It merged one pair (ask-first / proceed-autonomously) and kept the rest.
3. **Finalise by hand** (`finalize_rubrics.py`, edits recorded in the script): merged branch-creation into Git
   Automation, research-first into Upfront Planning, and the two UI clusters into one; dropped Session State
   Management (describes the product being built, not the agent); moved "secrets must not be hardcoded" to a
   uniform requirement (no developer wants the opposite). Five further rubrics were removed by decision:
   Review Gating, Commit Granularity, Naming Descriptiveness, UI Polish, Quality Gate Strictness.

**Final set: 15 rubrics** (`final/rubrics_final.md` with examples and originals, `final/rubrics_final_compact.md`
rubric text only, `final/rubrics_final.json`).

| # | rubric | id | turns | Low side has supporting statements |
|---|---|---|---|---|
| 1 | Documentation Workflow | `documentation_workflow` | 724 | no |
| 2 | Test Execution | `test_execution` | 638 | no |
| 3 | Refactoring Tolerance | `refactoring_tolerance` | 260 | no |
| 4 | Git Automation | `git_automation` | 357 | yes |
| 5 | Upfront Planning | `upfront_planning` | 367 | yes |
| 6 | Agent Autonomy | `agent_autonomy` | 346 | yes |
| 7 | Delivery Phasing | `delivery_phasing` | 163 | yes |
| 8 | Dependency Preference | `dependency_preference` | 167 | yes |
| 9 | Version Pinning | `version_pinning` | 99 | yes |
| 10 | Legacy Removal | `legacy_removal` | 107 | yes |
| 11 | Failure Handling | `failure_handling` | 129 | yes |
| 12 | Config Externalization | `config_externalization` | 134 | no |
| 13 | Logging Verbosity | `logging_verbosity` | 97 | yes |
| 14 | Execution Parallelism | `execution_parallelism` | 96 | yes |
| 15 | Uncertainty Disclosure | `uncertainty_disclosure` | 98 | yes |

High/Low follow the convention of the original 14 (High = more documentation / more tests / more restructuring /
more automation / more autonomy / external dependencies / recovery on failure / more logging / parallel). For four
rubrics (documentation, tests, refactoring, config) no statement in the data takes the Low side; the Low text is
the logical opposite and is marked as such.

Rubric text:

**Documentation Workflow**
Description: How much documentation work the agent should do as part of a change
High: Create, update, and consult documentation as an explicit step of every change
Low: Keep documentation minimal; touch it only when asked
N/A: The task is itself a documentation task, or no documentation exists to maintain

**Test Execution**
Description: How actively the agent should run and extend tests while working
High: Run the test suite continuously, add or extend tests with every change, treat tests as part of the deliverable
Low: Run only the tests directly relevant to the change, or only when asked
N/A: The task is itself a testing task, or the project has no test infrastructure

**Refactoring Tolerance**
Description: How much existing code the agent should restructure while making a change
High: Consolidate duplication, separate concerns, and reorganize modules when it improves the structure
Low: Make localized changes and leave the existing structure alone
N/A: The task explicitly requires or forbids restructuring

**Git Automation**
Description: How much of the git workflow (branching, committing, pushing, PRs, merging) the agent should carry out unprompted
High: Create branches, commit, push, open PRs, and merge on its own once checks pass
Low: Stop at code changes; wait for an explicit instruction before each git step, or work on the current branch
N/A: The user gives step-by-step git instructions, or the environment has no git

**Upfront Planning**
Description: How much investigation and planning the agent should do before writing code
High: Read the codebase, find root causes, write and validate a plan before any implementation
Low: Start implementing with the information at hand and adjust as it goes
N/A: The task is trivial or the user has already supplied the plan

**Agent Autonomy**
Description: How much the agent should decide and proceed without checking with the user
High: Proceed through the task on reasonable assumptions; ask only when truly blocked
Low: Pause to ask clarifying questions, present options, and get confirmation before acting
N/A: No discretionary decision arises

**Delivery Phasing**
Description: Whether work should be delivered in prioritized increments or all at once
High: Fix the critical items first and deliver in ordered phases, deferring the rest
Low: Address everything in one comprehensive pass
N/A: The task is a single indivisible change

**Dependency Preference**
Description: Whether to bring in external functionality
High: Prefer established libraries, services, APIs, or tools when useful
Low: Prefer built-ins, existing project facilities, or local implementation
N/A: The dependency choice is predetermined or no meaningful choice exists

**Version Pinning**
Description: How tightly dependency versions and releases should be controlled
High: Pin exact versions and trigger releases deliberately
Low: Track the latest versions and automate releases
N/A: The project has no dependency or release process

**Legacy Removal**
Description: What to do with code that has become unused or obsolete
High: Delete it outright, without compatibility shims or deprecation paths
Low: Keep it behind deprecation paths or compatibility layers
N/A: No obsolete code is involved

**Failure Handling**
Description: What the system should do when execution does not go as expected
High: Recover: retry, fall back, degrade gracefully, keep the workflow unblocked
Low: Fail fast and loudly with an explicit error
N/A: The expected failure behaviour is predetermined

**Config Externalization**
Description: How much behaviour should be exposed as configuration rather than fixed in code
High: Externalize values into environment variables, flags, and config files
Low: Hardcode sensible defaults and keep the configuration surface small
N/A: The value is a secret (never hardcoded; treated as a uniform requirement) or the configuration mechanism is fixed

**Logging Verbosity**
Description: How much the system should log
High: Detailed logs, traces, and metrics for visibility
Low: Minimal, targeted logging; remove debug output
N/A: Logging is not touched by the task

**Execution Parallelism**
Description: Whether independent work should run concurrently or one step at a time
High: Run independent tasks, subagents, and operations in parallel
Low: Run them sequentially, finishing one before starting the next
N/A: The work has no independent parts

**Uncertainty Disclosure**
Description: How the agent should present claims whose certainty varies
High: Qualify claims, mark what is uncertain, distinguish verified from assumed
Low: State conclusions directly and confidently, with few caveats
N/A: The output contains no claims of fact

### Step 6 - relation to the original 14 rubrics

| original rubric | new rubric(s) | relation | note |
|---|---|---|---|
| Solution Scope | delivery_phasing | partial | Old axis: how many related problems to solve. New axis: whether to deliver in prioritized phases or all at once. Overlap on scope of a single pass. |
| Refactoring Tolerance | refactoring_tolerance | same | Same question. |
| Abstraction Preference | refactoring_tolerance | absorbed | Statements about extracting helpers / reuse (213) mostly landed in the refactoring cluster; no separate cluster formed. |
| Dependency Preference | dependency_preference | same | Same question; most statements in the data take the built-in side. |
| Constraint Explicitness | — | not found | Type/schema/assertion statements (216) landed mainly in the quality-gates and testing clusters; typing appears as a gate to enforce rather than a code-design choice. The quality-gates cluster was not kept as a rubric. |
| Failure Handling | failure_handling | same | Same question. |
| Verification / Testing Style | test_execution | partial | Old axis: breadth of verification. New: whether tests are run continuously.  |
| Optimization | — | not found | 67 statements (0.7%) mention performance; two-thirds fall in noise. Too rare and too scattered to form a cluster. |
| Documentation Preference | documentation_workflow | partial | Old axis: how much explanation lives with the code. New: whether docs are maintained as an explicit workflow step. Depth vs. brevity did not separate in this run (it did in the v1 run). |
| Implementation Explicitness | — | not found | 91 statements (1.0%) about compact vs. explicit code; half in noise. No cluster. |
| Explanation Detail | uncertainty_disclosure | weak | Old axis: how much rationale in responses. New: whether claims are qualified. 122 explanation statements exist but 62% are noise. |
| Agent Autonomy | agent_autonomy; upfront_planning | same + split | Ask-vs-proceed is the same axis. The data additionally separates 'how much preparation before coding', which the old axis folded in. |
| Security | (uniform: secrets not hardcoded) | weak | 209 security statements exist but 64% are noise: permissions, injection, secrets, and dependency risk do not cluster together. Only 'never hardcode secrets' is dense, and it has no opposing side. |
| Specification Granularity | — | not applicable | Describes how the user writes instructions, not what they want the agent to do; cannot appear among 'Wants the agent to…' statements by construction. |

New with no counterpart in the original 14: Git Automation, Version Pinning, Legacy Removal, Logging Verbosity,
Execution Parallelism, Config Externalization.

Reading: the original 14 were code-design tradeoffs; what users say in chat is dominated by how the agent should
work (planning, autonomy, git, tests, documentation). Optimization and compact-vs-explicit code each appear in
under 1% of statements and do not form clusters.

### Step 7 - do users group by which rubrics they express? (`user_profiles.py`, `user_clusters.py`)

Each user is represented by the number of distinct turns falling in each rubric's clusters (both sides of a rubric
counted together; direction is not yet labelled). 3,394 turns from 140 users; median 6 turns per user.

| minimum turns per user | users | share of a user's turns in their top-3 rubrics, real / shuffled | k with silhouette above the shuffled 95th percentile |
|---|---|---|---|
| 10 | 52 | 0.64 / 0.55 | k=5 only (0.156 vs 0.148) |
| 5 | 78 | 0.70 / 0.60 | k=5, k=6 |

Clustering: Ward on row-normalised proportions; the baseline shuffles each user's rubric labels. The groups that
emerge are each defined by one dominant rubric (documentation-heavy users, test-heavy users, autonomy-heavy users),
not by combinations. Conclusion so far: weak evidence of structure beyond "each user has one or two favourite
topics". Limits: few users with enough turns; Documentation Workflow is the largest cluster and appears in every
group; High/Low direction is not distinguished, so two users who both talk about autonomy but want opposite things
look identical. Heatmap: `final/user_profiles/user_rubric_heatmap.png`.

## Known limitations and data issues

- Three model stages (classifier, rewrite, naming/drafting) each filter or paraphrase; rare or very specific
  preferences (performance, code style) are the most likely to be lost.
- Some harness-injected content passed the classifier: consecutive `## ...` sections of review skill files
  (OpenCode splits skill files into user turns), and at least one `<teammate-message>` from another agent appears
  among the rubric examples. Exact-duplicate removal caught the repeated ones only.
- 43% of statements are unassigned; the rubrics describe the dense part of the data.
- UMAP/HDBSCAN parameters (10 dims, leaf selection, min_samples=10, min_cluster_size=100) were chosen by inspection
  of the sweep above, not by an external criterion.
- The rewriting step occasionally adds a temporal clause ("before making changes") that the original did not
  state, which nudges statements toward process preferences.

## File index

| file | content |
|---|---|
| `outputs/discovery/method1/gate/gate_labels.jsonl`, `gate/gate_prefilter.csv` | per-turn 0/1 label; prefilter reason for all 62,544 turns |
| `outputs/discovery/method1/gate/positives.jsonl`, `gate/positives_dedup.jsonl` | the 9,215 / 8,420 positive turns with text |
| `outputs/discovery/method1/final/statements.jsonl`, `statements_flat.jsonl` | rewritten statements (per message / one per line) |
| `outputs/discovery/method1/final/embeddings/embeddings.npy` | 9,150 x 3072 |
| `outputs/discovery/method1/final/hdbscan/` | clusters, names, map, UMAP cache |
| `outputs/discovery/method1/final/rubrics_draft.*`, `rubrics.*`, `rubrics_final.*` | per-cluster drafts, model consolidation, final set |
| `outputs/discovery/method1/final/user_profiles/` | user x rubric matrix, clusters, heatmap |
