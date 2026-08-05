#!/usr/bin/env python3
"""
Section 1.5 "Data Distribution" — figure data.

Produces one CSV per subsubsection, ready to plot, plus a stats.json with
the numbers that go in the prose. Uses checkpoints_count (in-session
created), NOT checkpoint_ids (many-to-many referencing) — see the intro
section's note on why these differ.

Usage:
    python section15_distributions.py --data-dir ./swechat_data --out-dir ./section15_out
"""

import argparse, glob, json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Journal-friendly defaults: no seaborn styling, single-color bars, readable
# at half-page width, 300 dpi so they hold up when embedded in LaTeX.
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
})
BAR_COLOR = "#3B6FA0"
PIE_COLORS = ["#3B6FA0", "#C77B3B", "#5B9279", "#A15C7A", "#8C8C8C",
              "#B8A13B", "#6E5A9E", "#4F8FA6", "#A63B3B", "#5A7A3B"]


def find_parquet(data_dir, name):
    for p in [os.path.join(data_dir, f"{name}.parquet"),
              os.path.join(data_dir, "data", f"{name}.parquet")]:
        if os.path.exists(p):
            return p
    hits = glob.glob(os.path.join(data_dir, "**", f"{name}*.parquet"), recursive=True)
    if not hits:
        raise FileNotFoundError(name)
    return hits[0]


AGENT_NORM = {"claude code": "Claude Code", "claude-code": "Claude Code",
              "opencode": "OpenCode", "codex": "Codex", "gemini cli": "Gemini CLI",
              "cursor": "Cursor", "copilot cli": "Copilot CLI"}


def norm_agent(x):
    if pd.isna(x):
        return "unknown"
    return AGENT_NORM.get(str(x).strip().lower(), str(x).strip())


def med(s):
    s = pd.to_numeric(pd.Series(s), errors="coerce").dropna()
    if not len(s):
        return None
    return {"n": int(len(s)), "mean": round(float(s.mean()), 2),
            "median": round(float(s.median()), 1),
            "p75": round(float(s.quantile(.75)), 1),
            "p90": round(float(s.quantile(.90)), 1),
            "max": round(float(s.max()), 1)}


def plot_hist_binned(bin_edges, counts, xlabel, ylabel, title, path):
    """Bar chart over irregular (Fibonacci-style or integer) bins. A bin
    spanning a single value (width 1, e.g. exactly "2 checkpoints") is
    labeled with that number alone; only genuinely multi-value bins get
    the "lo-hi" range label. Avoids the "0-0", "1-1" clutter that shows up
    when every checkpoint count from 0 up gets its own bin."""
    labels = [f"{bin_edges[i]}-{bin_edges[i+1]-1}" if bin_edges[i+1] - 1 > bin_edges[i]
              else f"{bin_edges[i]}"
              for i in range(len(bin_edges) - 1)]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(len(counts)), counts, color=BAR_COLOR, width=0.85)
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    for i, c in enumerate(counts):
        if c > 0:
            ax.annotate(str(int(c)), (i, c), textcoords="offset points",
                       xytext=(0, 3), ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_pie(series, title, path, min_pct_labeled=3.0):
    """Pie chart. Slices under min_pct_labeled get their label suppressed
    (still shown in the legend) so tiny agents don't produce unreadable
    overlapping text."""
    vals = series.values.astype(float)
    labels = series.index.tolist()

    def autopct(pct):
        return f"{pct:.1f}%" if pct >= min_pct_labeled else ""

    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, _, autotexts = ax.pie(
        vals, autopct=autopct, startangle=90,
        colors=PIE_COLORS[:len(vals)],
        pctdistance=0.75,
        wedgeprops={"edgecolor": "white", "linewidth": 1})
    ax.legend(wedges, [f"{l} ({int(v)})" for l, v in zip(labels, vals)],
              loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9,
              frameon=False)
    ax.set_title(title)
    ax.axis("equal")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_bar_categorical(series, xlabel, ylabel, title, path, color=BAR_COLOR,
                         rotate=20):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(series.index.astype(str), series.values, color=color, width=0.7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=rotate, ha="right")
    for i, v in enumerate(series.values):
        ax.annotate(str(int(v)), (i, v), textcoords="offset points",
                   xytext=(0, 3), ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", default="./section15_out")
    args = ap.parse_args()
    fig_dir = os.path.join(args.out_dir, "figures")
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    R = {}

    sessions = pd.read_parquet(find_parquet(args.data_dir, "sessions"))
    sessions["_agent"] = sessions["agent"].map(norm_agent)
    uid = sessions["user_id"]
    has_uid = uid.notna() & (uid.astype(str).str.strip() != "")

    # ---------------------------------------------------- 1.5.1 Developers
    per_user = sessions[has_uid].groupby("user_id")["session_id"].nunique()
    per_user.rename("n_sessions").to_csv(
        os.path.join(args.out_dir, "hist_sessions_per_developer.csv"))
    # log-scale-friendly bin edges given max=450, median=6
    bins = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 451]
    counts, edges = np.histogram(per_user.values, bins=bins)
    R["developers"] = {
        "n_distinct": int(per_user.shape[0]),
        "summary": med(per_user),
        "histogram_bins": [int(b) for b in edges],
        "histogram_counts": [int(c) for c in counts],
    }
    plot_hist_binned(edges, counts, "Sessions per developer", "Developers",
                     "Sessions per developer (n=189)",
                     os.path.join(fig_dir, "hist_sessions_per_developer.png"))

    # ------------------------------------------------- 1.5.2 Coding agents
    agent_sessions = sessions["_agent"].value_counts()
    agent_sessions.rename("n_sessions").to_csv(
        os.path.join(args.out_dir, "pie_agent_sessions.csv"))
    agent_devs = sessions[has_uid].groupby("_agent")["user_id"].nunique()
    agent_devs.rename("n_developers").to_csv(
        os.path.join(args.out_dir, "pie_agent_developers.csv"))
    R["agents"] = {
        "sessions": {k: int(v) for k, v in agent_sessions.items()},
        "developers": {k: int(v) for k, v in agent_devs.items()},
    }
    plot_pie(agent_sessions, "Sessions by coding agent",
             os.path.join(fig_dir, "pie_agent_sessions.png"))
    plot_pie(agent_devs, "Developers by coding agent (a developer may use >1)",
             os.path.join(fig_dir, "pie_agent_developers.png"))

    # -------------------------------------------------- 1.5.3 Session length
    conv = pd.read_parquet(
        find_parquet(args.data_dir, "conversations"),
        columns=["session_id", "is_conversational"])
    conv_only = conv[conv["is_conversational"].fillna(False)]
    per_sess_turns = conv_only.groupby("session_id").size()
    per_sess_turns.rename("n_conversational_turns").to_csv(
        os.path.join(args.out_dir, "hist_conversational_turns_per_session.csv"))
    bins2 = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 900]
    counts2, edges2 = np.histogram(per_sess_turns.values, bins=bins2)
    R["session_length"] = {
        "summary": med(per_sess_turns),
        "note": "is_conversational == True turns only (user_prompt + "
                "assistant_response); excludes tool calls/results/events.",
        "histogram_bins": [int(b) for b in edges2],
        "histogram_counts": [int(c) for c in counts2],
    }
    plot_hist_binned(edges2, counts2, "Conversational turns per session",
                     "Sessions", "Session length (conversational turns)",
                     os.path.join(fig_dir, "hist_conversational_turns.png"))

    # -------------------------------------------------- 1.5.4 Pushback turns
    conv_pb = pd.read_parquet(
        find_parquet(args.data_dir, "conversations"),
        columns=["session_id", "turn_type", "prompt_pushback"])
    lab = conv_pb[conv_pb["prompt_pushback"].notna()]
    class_counts = lab["prompt_pushback"].value_counts()
    class_counts.rename("n").to_csv(
        os.path.join(args.out_dir, "bar_pushback_classes.csv"))
    real_pb = {"correction", "failure_report", "rejection", "takeover",
               "requirement_change", "pacing_complaint"}
    pb = lab["prompt_pushback"].astype(str).isin(real_pb)
    R["pushback"] = {
        "n_labeled": int(len(lab)),
        "n_pushback": int(pb.sum()),
        "rate": round(float(pb.mean()), 4),
        "class_counts": {str(k): int(v) for k, v in class_counts.items()},
        "documented_but_absent": sorted(
            real_pb - set(class_counts.index.astype(str))),
    }
    plot_bar_categorical(class_counts, "Pushback class", "Labeled prompts",
                         "Pushback class distribution (labeled prompts)",
                         os.path.join(fig_dir, "bar_pushback_classes.png"))

    # -------------------------------------------------- 1.5.5 Checkpoints
    # IMPORTANT: checkpoints_count (in-session created), not len(checkpoint_ids)
    # (many-to-many referencing) -- see intro section note.
    #
    # BUGFIX: per-session checkpoint counts must be computed over ALL
    # sessions (n=5,851), not just developer-attributed ones. Checkpoint
    # creation doesn't depend on whether we could resolve the developer's
    # identity. The earlier version filtered both the per-session and
    # per-developer frames through `sessions[has_uid]`, which silently
    # dropped 1,943 sessions from the per-session figure -- the chart title
    # said n=3,908 while the caption (built from the correctly-unfiltered
    # summary stat below) said n=5,851. Only the per-developer aggregation
    # actually needs the has_uid filter, since checkpoints can't be
    # attributed to an unknown developer.
    cc = pd.to_numeric(sessions["checkpoints_count"], errors="coerce")

    per_sess_ck = sessions[["session_id"]].assign(checkpoints_count=cc).dropna()
    per_sess_ck.to_csv(
        os.path.join(args.out_dir, "hist_checkpoints_per_session.csv"), index=False)
    assert len(per_sess_ck) == int(cc.notna().sum()), \
        "per-session checkpoint CSV row count must match all dated sessions"

    df_ck_dev = sessions[has_uid].assign(_cc=cc)
    per_user_ck = df_ck_dev.groupby("user_id")["_cc"].sum()
    per_user_ck.rename("total_checkpoints").to_csv(
        os.path.join(args.out_dir, "hist_checkpoints_per_developer.csv"))

    R["checkpoints"] = {
        "per_session": med(cc),
        "per_session_n": int(len(per_sess_ck)),
        "per_developer_total": med(per_user_ck),
        "frac_sessions_zero_checkpoints": round(float((cc == 0).mean()), 4),
        "frac_sessions_multi_checkpoint": round(float((cc > 1).mean()), 4),
        "note": "Uses sessions.checkpoints_count (created in-session), "
                "not len(checkpoint_ids) (many-to-many referencing; "
                "inflated by checkpoints shared with other sessions). "
                "per_session is over ALL sessions; per_developer_total is "
                "over developer-attributed sessions only (has_uid).",
    }
    # small integer range (mode 0-1, tail out past ~10) -> plain integer bins
    ck_vals = per_sess_ck["checkpoints_count"].values
    cap = int(np.percentile(ck_vals, 99)) or 10
    bins3 = list(range(0, cap + 2)) + [int(ck_vals.max()) + 1]
    bins3 = sorted(set(bins3))
    counts3, edges3 = np.histogram(ck_vals, bins=bins3)
    plot_hist_binned(edges3, counts3, "Checkpoints created in session",
                     "Sessions", f"Checkpoints per session (n={len(ck_vals)})",
                     os.path.join(fig_dir, "hist_checkpoints_per_session.png"))

    uck_vals = per_user_ck.values
    cap2 = int(np.percentile(uck_vals, 99)) or 10
    bins4 = sorted(set(list(range(0, cap2 + 2)) + [int(uck_vals.max()) + 1]))
    counts4, edges4 = np.histogram(uck_vals, bins=bins4)
    plot_hist_binned(edges4, counts4, "Total checkpoints created",
                     "Developers", "Checkpoints per developer",
                     os.path.join(fig_dir, "hist_checkpoints_per_developer.png"))

    with open(os.path.join(args.out_dir, "section15_stats.json"), "w") as f:
        json.dump(R, f, indent=2, default=str)

    print(json.dumps(R, indent=2, default=str))
    print(f"\nwrote CSVs + stats.json to {args.out_dir}/")


if __name__ == "__main__":
    main()