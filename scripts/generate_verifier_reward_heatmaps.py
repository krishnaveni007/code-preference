#!/usr/bin/env python3
"""Create one readable Matplotlib heatmap per verifier-reward experiment family."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "outputs/longitudinal_pilot/eval_metrics_ledger.json"
OUT_DIR = ROOT / "outputs/longitudinal_pilot/presentation_visualizations/verifier_reward_heatmaps"


def label(row: dict) -> str:
    return f"{row['user_id']} · {row['task_id']}"


def original_baselines(rows: list[dict]) -> list[dict]:
    return [
        r for r in rows
        if r["rubric_used"] == "Original extraction rubric"
        and r["agent"] == "OpenCode"
        and r["intervention_type"] == "Baseline"
    ]


def experiments(rows: list[dict]) -> list[dict]:
    baseline = original_baselines(rows)
    original = [r for r in rows if r["rubric_used"] == "Original extraction rubric" and r["agent"] == "OpenCode"]
    updated = [r for r in rows if r["rubric_used"] == "Updated extraction rubric" and r["agent"] == "OpenCode"]
    skill = [r for r in rows if r["rubric_used"] == "Personalized SKILL.md" and r["agent"] == "OpenCode"]
    codex = [r for r in rows if r["rubric_used"] == "Pilot preference representation" and r["agent"] == "Codex"]
    claude = [r for r in rows if r["rubric_used"] == "Pilot preference representation" and r["agent"] == "Claude Code"]
    return [
        {
            "slug": "01-original-rubric-opencode",
            "title": "Original extraction rubric · OpenCode",
            "subtitle": "Verifier reward by task and intervention",
            "rows": original,
            "conditions": ["Baseline", "High confidence only", "All confidence bands", "Direction plus judge reasoning", "Descriptive, no polarity"],
            "names": ["Baseline", "High confidence", "All confidence", "Direction + reasoning", "Descriptive"],
        },
        {
            "slug": "02-updated-rubric-opencode",
            "title": "Updated extraction rubric · OpenCode",
            "subtitle": "Verifier reward · baseline reused from the original-rubric experiment",
            "rows": baseline + updated,
            "conditions": ["Baseline", "High confidence only", "All confidence bands", "Preference contexts by direction", "Descriptive, no polarity"],
            "names": ["Baseline*", "High confidence", "All confidence", "Contexts by direction", "Descriptive"],
        },
        {
            "slug": "03-personalized-skill-opencode",
            "title": "Personalized SKILL.md · OpenCode",
            "subtitle": "Verifier reward · baseline reused from the original-rubric experiment",
            "rows": baseline + skill,
            "conditions": ["Baseline", "Personalized OpenCode skill"],
            "names": ["Baseline*", "Personalized skill"],
        },
        {
            "slug": "04-pilot-codex",
            "title": "Pilot preference representation · Codex",
            "subtitle": "Verifier reward by task and profile representation",
            "rows": codex,
            "conditions": ["Baseline", "Directional profile with high/low context", "Compact directional profile", "Detailed natural-language profile"],
            "names": ["Baseline", "Directional high/low", "Compact directional", "Detailed natural language"],
        },
        {
            "slug": "05-pilot-claude-code",
            "title": "Pilot preference representation · Claude Code",
            "subtitle": "Verifier reward by profile representation",
            "rows": claude,
            "conditions": ["Baseline", "Directional profile with high/low context", "Compact directional profile", "Detailed natural-language profile"],
            "names": ["Baseline", "Directional high/low", "Compact directional", "Detailed natural language"],
        },
        {
            "slug": "06-pilot-codex-claude-combined",
            "title": "",
            "subtitle": "",
            "rows": codex + claude,
            "conditions": ["Baseline", "Directional profile with high/low context", "Compact directional profile", "Detailed natural-language profile"],
            "names": ["Baseline", "Directional high/low", "Compact directional", "Detailed natural language"],
        },
    ]


def matrix(exp: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    task_labels = sorted({label(r) for r in exp["rows"]})
    values = pd.DataFrame(np.nan, index=task_labels, columns=exp["conditions"])
    status = pd.DataFrame("not_run", index=task_labels, columns=exp["conditions"])
    for task in task_labels:
        for condition in exp["conditions"]:
            attempts = [r for r in exp["rows"] if label(r) == task and r["intervention_type"] == condition]
            valid = [r for r in attempts if r["run_status"] == "Valid" and r.get("verifier_reward") is not None]
            if valid:
                values.loc[task, condition] = sum(float(r["verifier_reward"]) for r in valid) / len(valid)
                status.loc[task, condition] = "valid"
            elif attempts:
                status.loc[task, condition] = "failed"
    values.columns = exp["names"]
    status.columns = exp["names"]
    return values, status


def wrap_axis_label(value: str, width: int = 20) -> str:
    return "\n".join(textwrap.wrap(value, width=width, break_long_words=False))


def render(exp: dict) -> Path:
    values, status = matrix(exp)
    n_rows, n_cols = values.shape
    sns.set_theme(style="white", context="talk")
    fig_w = max(8.5, 2.2 * n_cols + 4.8)
    fig_h = max(4.0, 0.72 * n_rows + 2.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)

    cmap = sns.color_palette("Blues", as_cmap=True)
    sns.heatmap(
        values,
        mask=values.isna(),
        cmap=cmap,
        vmin=0,
        vmax=1,
        linewidths=1.5,
        linecolor="white",
        cbar_kws={"label": "Verifier reward", "shrink": 0.82, "ticks": [0, .25, .5, .75, 1]},
        annot=False,
        square=False,
        ax=ax,
    )

    for i, task in enumerate(values.index):
        for j, condition in enumerate(values.columns):
            state = status.loc[task, condition]
            value = values.loc[task, condition]
            if state == "valid":
                color = "white" if value >= 0.62 else "#172033"
                ax.text(j + 0.5, i + 0.5, f"{value:.2f}", ha="center", va="center", fontsize=15, fontweight="bold", color=color)
            else:
                fill = "#f8d7da" if state == "failed" else "#eceff1"
                text = "FAILED" if state == "failed" else "—"
                text_color = "#a61b29" if state == "failed" else "#6b7280"
                ax.add_patch(Rectangle((j, i), 1, 1, facecolor=fill, edgecolor="white", linewidth=1.5, zorder=2))
                ax.text(j + 0.5, i + 0.5, text, ha="center", va="center", fontsize=12, fontweight="bold", color=text_color, zorder=3)

    if exp["title"]:
        ax.set_title(exp["title"], loc="left", fontsize=23, fontweight="bold", pad=34)
    if exp["subtitle"]:
        ax.text(0, 1.035, exp["subtitle"], transform=ax.transAxes, ha="left", va="bottom", fontsize=13, color="#596273")
    ax.set_xlabel("Intervention condition", fontsize=14, fontweight="bold", labelpad=16)
    ax.set_ylabel("User · task", fontsize=14, fontweight="bold", labelpad=16)
    ax.set_xticklabels([wrap_axis_label(x) for x in values.columns], rotation=0, ha="center", fontsize=12)
    ax.set_yticklabels(values.index, rotation=0, fontsize=12)
    ax.tick_params(axis="both", length=0, pad=8)
    legend_handles = []
    observed_states = set(status.to_numpy().ravel())
    if "failed" in observed_states:
        legend_handles.append(Patch(facecolor="#f8d7da", label="Failed simulator run"))
    if "not_run" in observed_states:
        legend_handles.append(Patch(facecolor="#eceff1", label="Not run"))
    if legend_handles:
        ax.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.30),
            ncol=len(legend_handles),
            frameon=False,
            fontsize=11,
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUT_DIR / f"{exp['slug']}.png"
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


def main() -> None:
    rows = json.loads(LEDGER.read_text())["rows"]
    for exp in experiments(rows):
        print(render(exp))


if __name__ == "__main__":
    main()
