Please remember any useful preferences, corrections or workflow patterns from this session for future sessions with this developer.

Keep the memory in your auto-memory directory as follows, using your normal Read/Write/Edit tools. Read what is already there first, then revise it for this session.

Future sessions see only `MEMORY.md` without opening anything. It must say *when* to open a note, never *what* the developer wants: the preference itself lives only in the note.

## Files

- `MEMORY.md`: the index, exactly in the form under "Index format". Nothing else goes in it.
- `observations.md`: a ledger, one line per preference-like statement in this session: `YYYY-MM-DD | DIM | mode | side | ≤ 12-word gist | correction? yes/no`. Every statement goes here first. Not indexed.
- One note per dimension, fixed file names: `autonomy.md`, `planning.md`, `execution.md`, `delegation.md`, `latitude.md`, `communication.md`, `documentation.md`, `testing.md`, `commits.md`, `branching.md`, `architecture.md`, `errors.md`, `tooling.md`, `product.md`.
- `project-context.md`: non-preference facts (repo conventions, environment quirks, how a past bug was solved). Task identifiers, debugging methods and narratives go here or nowhere, never in a dimension note.
- `other.md`: statements that fit no dimension or side below, each with `Proposed dimension/side: …`. Not indexed.

## Phases and dimensions (name, file, sides; use these words, do not invent sides)

### Before acting
- AUT · Autonomy and approval · autonomy.md: ask or wait first | proceed on its own
- PLAN · Planning and research depth · planning.md: plan or research deeply first | keep planning light
### While working
- EXEC · Execution pacing · execution.md: step by step | in parallel | all at once | stop early or cherry-pick | finish everything
- DELEG · Delegation and orchestration · delegation.md: use subagents or a team | do it yourself | a named model
- LAT · Latitude beyond the request · latitude.md: stick to the request | improve what you find
### Talking and writing
- COMM · Communication and reporting · communication.md: tell me more | keep it short | a particular tone or format
- DOC · Documentation and written output · documentation.md: a fixed structure or format | a particular tone or audience | keep it short | go into detail | a particular language
### Verifying and shipping
- TEST · Testing, verification and review rigor · testing.md: test thoroughly | test lightly or by hand | test first (TDD)
- COMMIT · Commit granularity and messages · commits.md: commit in small steps | few big commits | a fixed message format | detailed messages | short messages
- PR · Branching, PR and push workflow · branching.md: use branches and PRs | push or merge freely | wait for review before moving on | do not push | open as draft | not as draft
### In the code
- ARCH · Code architecture and style · architecture.md: keep it simple and small | a naming or layout rule | build for flexibility | reuse and centralize | keep it local, duplication is fine | match existing patterns
- ERR · Error handling and robustness · errors.md: fail fast and loud | handle gracefully | log and monitor
- TOOL · Tooling, dependencies and environment · tooling.md: use a named tool or model | avoid a named tool | secrets and env vars
- UI · UI, copy and product behaviour · product.md: a product-behaviour rule | keep copy short and plain | more detail on screen | match the existing look | a particular language or locale | a particular tone

## Modes and their trigger phrases (copy verbatim)

Modes: discuss, plan, build, verify, ship, configure. The mode is what the developer was doing when they said it (asking or discussing; planning; building; checking or reviewing; committing, pushing or releasing; setting up tools or config).

| dim | discuss | plan | build | verify | ship | configure |
|---|---|---|---|---|---|---|
| AUT | when asked a question or for an opinion | before acting on a proposed plan | before editing files or running changes | before fixing something found while checking | before committing, pushing or opening a PR | before changing settings or the environment |
| PLAN | when asked how or why something works | when planning a change | before starting a multi-file change | before claiming a cause or that something works | before preparing a release or PR | before setting up tools or the environment |
| EXEC | when answering several questions at once | when splitting work into steps | when a change has several parts | when working through a list of findings | when shipping several changes | when changing several configs |
| DELEG | when a question needs broad research | when planning work that could be split up | before spawning subagents or choosing a model | when reviewing or checking large changes | when a release has many parts | when choosing models or agent settings |
| LAT | when noticing something beyond the question | when scoping a plan | when noticing problems outside the request | after a fix, rename or rewrite | before adding extra changes to a commit or PR | before changing settings that were not asked about |
| COMM | when answering a question or explaining | when presenting a plan | when reporting progress | when reporting results or findings | when summarising what was shipped | when reporting setup steps |
| DOC | when writing an explanation to keep | when writing a plan or spec document | when writing docs, comments or READMEs | when writing up review findings | when writing PR descriptions, issues or release notes | when documenting setup or config |
| TEST | when asked whether something is tested | when planning a change | when adding or changing functionality | after a bug fix or before reporting done | before committing or opening a PR | after changing build, CI or config |
| COMMIT | when asked about commit history | when planning how to split work into commits | after finishing a unit of work | after a fix passes | when committing or writing commit messages | when changing git hooks or commit settings |
| PR | when asked about branches or PRs | when planning branches | before starting work on a branch | before moving on from a review | before pushing, merging or opening a PR | when changing CI or repository settings |
| ARCH | when discussing design or structure | when designing a change | when writing or restructuring code | when reviewing code structure | before finalising code for a PR | when organising config or project layout |
| ERR | when discussing failure cases | when planning failure handling | when code can fail or an error is caught | when checking failure paths | before shipping code with new failure paths | when a tool or config can fail |
| TOOL | when recommending tools or libraries | when choosing tools, libraries or models | when adding a dependency or running a tool | when choosing how to run checks | when choosing release tooling | when setting up tools, env vars or secrets |
| UI | when discussing product behaviour or wording | when planning user-facing changes | when writing user-facing output, copy or defaults | when checking user-facing output | before shipping user-facing changes | when changing user-facing defaults |

## Index format (`MEMORY.md`)

```
# Memory

Preferences of this developer, one note per topic. Open a note before acting when its line matches what you are about to do.

## Before acting
- [AUT · Autonomy and approval](autonomy.md) — before editing files or running changes; before committing, pushing or opening a PR

## Verifying and shipping
- [TEST · Testing, verification and review rigor](testing.md) — when adding or changing functionality; after a bug fix or before reporting done

## Project
- [Project context](project-context.md) — before changing code in this repository
```

- Phase headings in the order above; leave out a phase with no note.
- One line per dimension note: `- [KEY · Name](file.md) — ` then the trigger phrases of the modes that note has a section for, copied verbatim from the table, in mode order, joined with `; `. If a note has more than 3 mode sections, use the 3 with the highest n.
- The `## Project` line is exactly as shown, present only if `project-context.md` exists.
- No side, stance, quote or instruction anywhere in `MEMORY.md`.

## Note format (one per dimension)

```
---
name: testing
description: Testing, verification and review rigor
metadata:
  type: feedback
---
Phase: Verifying and shipping
Dimension: TEST · Testing, verification and review rigor

## verify — after a bug fix or before reporting done
Stance: test thoroughly (n=4 in 3 sessions, 2026-01-06 → 2026-01-15)
How to apply: Whenever a bug is fixed, add a test that reproduces the original scenario before reporting done.
Examples:
- 2026-01-13: "can you add an explicit test for this?"
- 2026-01-15: "can we add a test (probably integration?) that tests this works end2end?"

Stance: test first (TDD) (n=2 in 2 sessions, 2026-01-20 → 2026-01-22)
How to apply: …
Examples: …

Conflicts: <omit unless an opposing side was observed: side, n, and the scope that separates them>
```

- `description` is the dimension's name exactly as listed above, never the stance.
- One `## <mode> — <trigger phrase>` section per mode with a kept stance, in mode order; the trigger phrase is copied from the table.
- Inside a mode section, one `Stance:` block (Stance, How to apply, Examples) per kept side. Every kept side gets a block unless it is opposed to another kept side in the same mode; opposed sides go in `Conflicts:` instead.
- `Stance:` one side from the list for that dimension, verbatim, with n (ledger lines), sessions and first → last date.
- `How to apply:` one or two sentences: a behaviour the developer asked for or corrected, phrased so they could have said it. No debugging method, no task narrative, no file or function names.
- `Examples:` at most 3 per stance, verbatim from the recordings, each at most 25 words (cut with …), dated by session.
- At most about 900 tokens per note; to save space, shorten examples and How to apply lines, never drop a kept stance.

## Update protocol, per statement in this session that reads as a preference, a correction of your way of working, or a workflow habit

1. Add a ledger line. Classify to a dimension, mode and side from the lists above; mark whether it was a correction.
2. A (dimension, mode, side) is kept when it has n ≥ 2 ledger lines, or one that was an explicit correction. Every kept one must appear in its note, as a `Stance:` block or in `Conflicts:`. When one becomes kept or is reinforced: create or update its stance block (increment n, update dates, replace an example only if the new one is more representative).
3. Opposite side in the same dimension and mode: do not overwrite. Add a `Conflicts:` line and keep the existing stance until the opposing side has been seen twice or the developer says so explicitly. If the two sides differ by scope, say the scope in each `How to apply:`.
4. Rewrite, do not append: after updating notes, regenerate every `MEMORY.md` line from the notes' mode sections.
5. Do not write methodology (how to debug, what to re-read, which artifacts to inspect) into a dimension note; if worth keeping, put one line in `project-context.md`.
