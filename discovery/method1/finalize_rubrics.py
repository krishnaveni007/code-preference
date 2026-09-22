#!/usr/bin/env python3
"""Final rubric set (v2), hand-edited from the consolidated draft: merges, drops, and the
Name / Description / High / Low / N-A format. Example statements (with the original user
message) are pulled from rubrics_draft.json by cluster rank. Also writes the mapping table
from the repository's original 14 rubrics to the new set."""

from __future__ import annotations

import json
from pathlib import Path

V = Path("outputs/discovery/method1/final")
drafts = {d["rank"]: d for d in json.loads((V / "rubrics_draft.json").read_text())}

# (id, name, description, high, low, na, high_clusters, low_clusters, low_evidence, note)
FINAL = [
 ("documentation_workflow", "Documentation Workflow",
  "How much documentation work the agent should do as part of a change.",
  "Create, update, and consult documentation as an explicit step of every change.",
  "Keep documentation minimal; touch it only when asked.",
  "The task is itself a documentation task, or no documentation exists to maintain.",
  [1], [], "none", "Low side: no supporting statements in this run."),
 ("test_execution", "Test Execution",
  "How actively the agent should run and extend tests while working.",
  "Run the test suite continuously, add or extend tests with every change, treat tests as part of the deliverable.",
  "Run only the tests directly relevant to the change, or only when asked.",
  "The task is itself a testing task, or the project has no test infrastructure.",
  [2], [], "none", "Low side: no supporting statements in this run."),
 ("refactoring_tolerance", "Refactoring Tolerance",
  "How much existing code the agent should restructure while making a change.",
  "Consolidate duplication, separate concerns, and reorganize modules when it improves the structure.",
  "Make localized changes and leave the existing structure alone.",
  "The task explicitly requires or forbids restructuring.",
  [3], [], "none", "Low side: no supporting statements in this run."),
 ("git_automation", "Git Automation",
  "How much of the git workflow (branching, committing, pushing, PRs, merging) the agent should carry out unprompted.",
  "Create branches, commit, push, open PRs, and merge on its own once checks pass.",
  "Stop at code changes; wait for an explicit instruction before each git step, or work on the current branch.",
  "The user gives step-by-step git instructions, or the environment has no git.",
  [4, 17], [], "in_cluster", "Merged clusters 4 (automation level) and 17 (branch creation); both had statements on both sides."),
 ("upfront_planning", "Upfront Planning",
  "How much investigation and planning the agent should do before writing code.",
  "Read the codebase, find root causes, write and validate a plan before any implementation.",
  "Start implementing with the information at hand and adjust as it goes.",
  "The task is trivial or the user has already supplied the plan.",
  [5, 7], [], "in_cluster", "Merged clusters 5 (research first) and 7 (plan first)."),
 ("agent_autonomy", "Agent Autonomy",
  "How much the agent should decide and proceed without checking with the user.",
  "Proceed through the task on reasonable assumptions; ask only when truly blocked.",
  "Pause to ask clarifying questions, present options, and get confirmation before acting.",
  "No discretionary decision arises.",
  [14], [6], "other_cluster", "Clusters 6 (ask first) and 14 (proceed autonomously) are the two sides. Follows the original convention: High = autonomous."),
 ("delivery_phasing", "Delivery Phasing",
  "Whether work should be delivered in prioritized increments or all at once.",
  "Fix the critical items first and deliver in ordered phases, deferring the rest.",
  "Address everything in one comprehensive pass.",
  "The task is a single indivisible change.",
  [12], [], "in_cluster", ""),
 ("dependency_preference", "Dependency Preference",
  "Whether to bring in external functionality.",
  "Prefer established libraries, services, APIs, or tools when useful.",
  "Prefer built-ins, existing project facilities, or local implementation.",
  "The dependency choice is predetermined or no meaningful choice exists.",
  [10], [], "in_cluster", "Same axis as the original rubric. Note: most statements in cluster 10 take the Low side."),
 ("version_pinning", "Version Pinning",
  "How tightly dependency versions and releases should be controlled.",
  "Pin exact versions and trigger releases deliberately.",
  "Track the latest versions and automate releases.",
  "The project has no dependency or release process.",
  [21], [], "in_cluster", ""),
 ("legacy_removal", "Legacy Removal",
  "What to do with code that has become unused or obsolete.",
  "Delete it outright, without compatibility shims or deprecation paths.",
  "Keep it behind deprecation paths or compatibility layers.",
  "No obsolete code is involved.",
  [20], [], "in_cluster", ""),
 ("failure_handling", "Failure Handling",
  "What the system should do when execution does not go as expected.",
  "Recover: retry, fall back, degrade gracefully, keep the workflow unblocked.",
  "Fail fast and loudly with an explicit error.",
  "The expected failure behaviour is predetermined.",
  [16], [], "in_cluster", "Same axis as the original rubric."),
 ("config_externalization", "Config Externalization",
  "How much behaviour should be exposed as configuration rather than fixed in code.",
  "Externalize values into environment variables, flags, and config files.",
  "Hardcode sensible defaults and keep the configuration surface small.",
  "The value is a secret (never hardcoded; treated as a uniform requirement) or the configuration mechanism is fixed.",
  [15], [], "none", "Low side: no supporting statements. Secrets-in-code is excluded: no developer wants it, so it is a uniform requirement, not a preference."),
 ("logging_verbosity", "Logging Verbosity",
  "How much the system should log.",
  "Detailed logs, traces, and metrics for visibility.",
  "Minimal, targeted logging; remove debug output.",
  "Logging is not touched by the task.",
  [23], [], "in_cluster", "Cluster 23 contains both directions; High/Low follow the original convention (High = more)."),
 ("execution_parallelism", "Execution Parallelism",
  "Whether independent work should run concurrently or one step at a time.",
  "Run independent tasks, subagents, and operations in parallel.",
  "Run them sequentially, finishing one before starting the next.",
  "The work has no independent parts.",
  [24], [], "in_cluster", ""),
 ("uncertainty_disclosure", "Uncertainty Disclosure",
  "How the agent should present claims whose certainty varies.",
  "Qualify claims, mark what is uncertain, distinguish verified from assumed.",
  "State conclusions directly and confidently, with few caveats.",
  "The output contains no claims of fact.",
  [22], [], "in_cluster", ""),
]
DROPPED = [
 ("session_state_management", 8, "Describes features of the product being built (session persistence, checkpoints), not how the agent should work. Project-specific."),
 ("review_gating", 13, "Removed by decision (2026-09-21)."),
 ("commit_granularity", 9, "Removed by decision (2026-09-21)."),
 ("naming_descriptiveness", 20, "Removed by decision (2026-09-21)."),
 ("ui_polish", "11, 19", "Removed by decision (2026-09-21)."),
 ("quality_gate_strictness", 16, "Removed by decision (2026-09-21)."),
]
UNIFORM = [("Secrets and credentials must not be hardcoded", 14)]

# fix: quality gates is draft rank 15 in the *rubrics* list but cluster rank 16 in the drafts
for i, f in enumerate(FINAL):
    if f[0] == "quality_gate_strictness":
        FINAL[i] = f[:6] + ([16], [], "none", f[9])
    if f[0] == "config_externalization":
        FINAL[i] = f[:6] + ([15], [], "none", f[9])
    if f[0] == "failure_handling":
        FINAL[i] = f[:6] + ([17], [], "in_cluster", f[9])
    if f[0] == "naming_descriptiveness":
        FINAL[i] = f[:6] + ([20], [], "in_cluster", f[9])
    if f[0] == "legacy_removal":
        FINAL[i] = f[:6] + ([21], [], "in_cluster", f[9])
    if f[0] == "version_pinning":
        FINAL[i] = f[:6] + ([22], [], "in_cluster", f[9])
    if f[0] == "uncertainty_disclosure":
        FINAL[i] = f[:6] + ([23], [], "in_cluster", f[9])
    if f[0] == "logging_verbosity":
        FINAL[i] = f[:6] + ([24], [], "in_cluster", f[9])
    if f[0] == "execution_parallelism":
        FINAL[i] = f[:6] + ([25], [], "in_cluster", f[9])
    if f[0] == "git_automation":
        FINAL[i] = f[:6] + ([4, 18], [], "in_cluster", f[9])
    if f[0] == "ui_polish":
        FINAL[i] = f[:6] + ([11, 19], [], "none", f[9])

def quotes(ranks, side):
    out = []
    for r in ranks:
        d = drafts[r]
        out += d["pole_a_quotes" if side == "a" else "pole_b_quotes"]
    return out

rubrics = []
for (rid, name, desc, high, low, na, hc, lc, low_ev, note) in FINAL:
    turns = sum(drafts[r]["n_turns"] for r in hc + lc)
    # which draft side maps to High? the draft's pole A is what the cluster wanted; decide per rubric
    high_from_a = rid not in ("agent_autonomy", "dependency_preference", "logging_verbosity")
    if rid == "agent_autonomy":
        high_q, low_q = quotes([14], "a"), quotes([6], "a")
    elif rid == "review_gating":
        high_q, low_q = quotes([13], "a"), quotes([14], "a")
    elif rid == "dependency_preference":
        high_q, low_q = quotes([10], "b"), quotes([10], "a")
    elif rid == "logging_verbosity":
        high_q, low_q = quotes([24], "b"), quotes([24], "a")
    else:
        high_q, low_q = quotes(hc, "a"), (quotes(hc, "b") if low_ev != "none" else [])
    rubrics.append({"id": rid, "name": name, "description": desc, "high": high, "low": low, "na": na,
                    "clusters": hc + lc, "turns": turns, "low_evidence": low_ev, "note": note,
                    "examples_high": high_q[:3], "examples_low": low_q[:3]})

MAPPING = [
 ("solution_scope", "Solution Scope", "delivery_phasing", "partial", "Old axis: how many related problems to solve. New axis: whether to deliver in prioritized phases or all at once. Overlap on scope of a single pass."),
 ("refactoring_tolerance", "Refactoring Tolerance", "refactoring_tolerance", "same", "Same question."),
 ("abstraction_preference", "Abstraction Preference", "refactoring_tolerance", "absorbed", "Statements about extracting helpers / reuse (213) mostly landed in the refactoring cluster; no separate cluster formed."),
 ("dependency_preference", "Dependency Preference", "dependency_preference", "same", "Same question; most statements in the data take the built-in side."),
 ("constraint_explicitness", "Constraint Explicitness", "—", "not found", "Type/schema/assertion statements (216) landed mainly in the quality-gates and testing clusters; typing appears as a gate to enforce rather than a code-design choice. The quality-gates cluster was not kept as a rubric."),
 ("failure_handling_preference", "Failure Handling", "failure_handling", "same", "Same question."),
 ("verification_testing_style", "Verification / Testing Style", "test_execution", "partial", "Old axis: breadth of verification. New: whether tests are run continuously. "),
 ("optimization", "Optimization", "—", "not found", "67 statements (0.7%) mention performance; two-thirds fall in noise. Too rare and too scattered to form a cluster."),
 ("documentation_preference", "Documentation Preference", "documentation_workflow", "partial", "Old axis: how much explanation lives with the code. New: whether docs are maintained as an explicit workflow step. Depth vs. brevity did not separate in this run (it did in the v1 run)."),
 ("implementation_explicitness", "Implementation Explicitness", "—", "not found", "91 statements (1.0%) about compact vs. explicit code; half in noise. No cluster."),
 ("explanation_detail", "Explanation Detail", "uncertainty_disclosure", "weak", "Old axis: how much rationale in responses. New: whether claims are qualified. 122 explanation statements exist but 62% are noise."),
 ("agent_autonomy", "Agent Autonomy", "agent_autonomy; upfront_planning", "same + split", "Ask-vs-proceed is the same axis. The data additionally separates 'how much preparation before coding', which the old axis folded in."),
 ("security", "Security", "(uniform: secrets not hardcoded)", "weak", "209 security statements exist but 64% are noise: permissions, injection, secrets, and dependency risk do not cluster together. Only 'never hardcode secrets' is dense, and it has no opposing side."),
 ("specification_granularity", "Specification Granularity", "—", "not applicable", "Describes how the user writes instructions, not what they want the agent to do; cannot appear among 'Wants the agent to…' statements by construction."),
]
NEW_ONLY = ["git_automation", "version_pinning", "legacy_removal", "logging_verbosity", "execution_parallelism", "config_externalization"]

out = {"rubrics": rubrics, "dropped": [{"id": i, "cluster": c, "reason": r} for i, c, r in DROPPED],
       "uniform_requests": [{"name": n, "cluster": c} for n, c in UNIFORM], "mapping_from_original_14": [dict(zip(("old_id","old_name","new_id","relation","note"), m)) for m in MAPPING],
       "new_without_original_counterpart": NEW_ONLY}
(V / "rubrics_final.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))

L = [f"# Preference rubrics discovered from SWE-Chat (method 1, v2) — {len(rubrics)} rubrics", "",
     "Format follows the repository's original rubric set. `turns` = distinct user turns in the source clusters. "
     "Where the Low side has no supporting statements this is stated. Examples show the rewritten statement in bold and the original user message beneath.", ""]
for k, r in enumerate(rubrics, 1):
    L += [f"## {k}. {r['name']}  (`{r['id']}`)", "",
          f"**Description:** {r['description']}  ", f"**High:** {r['high']}  ", f"**Low:** {r['low']}  ", f"**N/A:** {r['na']}  ", "",
          f"- source clusters {r['clusters']}, {r['turns']} turns; Low-side evidence: {r['low_evidence']}" + (f"; {r['note']}" if r['note'] else ""), ""]
    for side in ("high", "low"):
        ex = r[f"examples_{side}"]
        if ex:
            L.append(f"**Examples — {side.capitalize()}:**")
            for q in ex:
                o = " ".join(q["original"].split())
                L += [f"- **{q['statement']}**", f"  - original: {o[:400]}{'…' if len(o) > 400 else ''}"]
            L.append("")
L += ["## Dropped", ""] + [f"- `{i}` (cluster {c}): {r}" for i, c, r in DROPPED] + ["", "## Uniform requirements (no opposing side in the data)", ""] + [f"- {n} (cluster {c})" for n, c in UNIFORM]
L += ["", "## Mapping from the original 14 rubrics", "", "| Original rubric | New rubric(s) | Relation | Note |", "|---|---|---|---|"]
for old_id, old_name, new_id, rel, note in MAPPING:
    L.append(f"| {old_name} (`{old_id}`) | {new_id} | {rel} | {note} |")
L += ["", "**New rubrics with no counterpart in the original 14:** " + ", ".join(f"`{n}`" for n in NEW_ONLY), ""]
(V / "rubrics_final.md").write_text("\n".join(L))
print(f"-> {V/'rubrics_final.md'}: {len(rubrics)} rubrics, {len(DROPPED)} dropped, {len(UNIFORM)} uniform")
for k, r in enumerate(rubrics, 1):
    print(f"{k:2d}. {r['name']:26s} clusters {str(r['clusters']):9s} {r['turns']:>4} turns  low-evidence={r['low_evidence']}  ex H/L={len(r['examples_high'])}/{len(r['examples_low'])}")
