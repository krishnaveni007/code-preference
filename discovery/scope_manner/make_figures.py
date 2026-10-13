"""Step 6: the five figures in outputs/discovery/scope_manner/figures/ (no API calls).

Usage: python make_figures.py [--inp restated] [--matters matters_final]
"""
import argparse, collections, itertools, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import common as C

ap = argparse.ArgumentParser(); ap.add_argument("--inp", default="restated"); ap.add_argument("--matters", default="matters_final"); a = ap.parse_args()
BLUE, ORANGE, INK, INK2, GRID, SURF = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
SEQ = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#256abf", "#1c5cab"]
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False})
C.FIG.mkdir(parents=True, exist_ok=True)
msgs = C.load_messages(); R = C.load_records(C.DATA / f"{a.inp}.jsonl"); Cons = json.load(open(C.DATA / "consistency.json"))
S = [o for o in (json.loads(l) for l in open(C.DATA / f"{a.matters}.jsonl")) if "error" not in o and o.get("one_matter")]
n_manner = sum(1 for o in R.values() if o["kind"] != "task" and o["manner"].strip())
n_sel = sum(1 for r in Cons if r["users_multi_session"] >= 5)

# fig1 funnel (first two numbers come from the gate line's README)
stages = [("User turns in SWE-chat v2", 92246), ("Passed deterministic prefilter", 60364), ("Labelled as preference (gate)", len(msgs)),
          ('Restated as a "manner" statement', n_manner), ("Matters (clusters judged as one thing)", len(S)), ("Matters with >=5 users seen in >=2 sessions", n_sel)]
fig, ax = plt.subplots(figsize=(8.5, 3.6)); y = np.arange(len(stages))[::-1]
for i, (lab, v) in enumerate(stages):
    ax.barh(y[i], v, color=SEQ[i], height=0.62); ax.text(v + 900, y[i], f"{v:,}", va="center", color=INK)
ax.set_yticks(y); ax.set_yticklabels([s[0] for s in stages], color=INK); ax.set_xlim(0, 105000); ax.set_xticks([]); ax.tick_params(axis="y", length=0)
ax.set_title("From raw turns to analysable preference matters", loc="left", color=INK, fontsize=12, pad=12)
fig.tight_layout(); fig.savefig(C.FIG / "fig1_funnel.png", dpi=200); plt.close(fig)

# fig2 within vs between
rows = sorted([r for r in Cons if r["users_multi_session"] >= 5], key=lambda r: r["within_same"] - r["between_same"])
fig, ax = plt.subplots(figsize=(9.5, 10.5)); y = np.arange(len(rows))
for i, r in enumerate(rows):
    ax.plot([r["within_lo"], r["within_hi"]], [i, i], color=BLUE, alpha=0.25, lw=5, solid_capstyle="butt", zorder=1)
    ax.plot([r["between_same"], r["within_same"]], [i, i], color=GRID, lw=1.5, zorder=2)
    ax.scatter(r["between_same"], i, color=ORANGE, s=46, zorder=3, edgecolor=SURF, linewidth=1.5)
    ax.scatter(r["within_same"], i, color=BLUE, s=46, zorder=4, edgecolor=SURF, linewidth=1.5)
ax.set_yticks(y); ax.set_yticklabels([f"{r['matter'][:58]}  (n={r['users_multi_session']})" for r in rows], color=INK, fontsize=9); ax.tick_params(axis="y", length=0)
ax.set_xlim(0, 1.02); ax.set_ylim(-0.8, len(rows) + 1.6); ax.set_xlabel("Share of statement pairs with the same stance"); ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
ax.scatter([], [], color=BLUE, s=46, label="Same user, different sessions (band = 95% CI over users)"); ax.scatter([], [], color=ORANGE, s=46, label="Different users")
ax.legend(loc="upper left", frameon=False, fontsize=9)
fig.suptitle(f"Which matters are personal?\nSame-stance rate within a user vs across users; {len(rows)} matters with >=5 users seen in >=2 sessions (n = such users)", x=0.01, ha="left", color=INK, fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(C.FIG / "fig2_within_vs_between.png", dpi=200); plt.close(fig)

# fig3 pooled four levels
rng = np.random.default_rng(0); lv = collections.defaultdict(lambda: collections.defaultdict(list))
for o in S:
    st = {}
    for g in o.get("stances", []):
        if g.get("label") == "off-matter": continue
        for i in g.get("members", []):
            try:
                i = int(i)
                if i in R: st[i] = g["label"]
            except (TypeError, ValueError): pass
    opp = {frozenset(p) for p in o.get("opposite", []) if isinstance(p, list) and len(p) == 2}
    for x, yy in itertools.combinations(list(st), 2):
        A, B = msgs[x], msgs[yy]; same = st[x] == st[yy]; io = frozenset([st[x], st[yy]]) in opp
        if A["user_id"] != B["user_id"]: l = 3; k = tuple(sorted([A["user_id"], B["user_id"]]))
        elif A["session_id"] == B["session_id"]: l = 0; k = A["user_id"]
        elif A["repo_id"] == B["repo_id"]: l = 1; k = A["user_id"]
        else: l = 2; k = A["user_id"]
        lv[l][k].append((same, io))
names = ["Same session", "Same repo,\nother session", "Other repo", "Other user"]; stats = []
for l in range(4):
    vs = np.array([np.mean([x[0] for x in v]) for v in lv[l].values()]); vo = np.array([np.mean([x[1] for x in v]) for v in lv[l].values()])
    bs = [vs[rng.integers(0, len(vs), len(vs))].mean() for _ in range(500)]; bo = [vo[rng.integers(0, len(vo), len(vo))].mean() for _ in range(500)]
    stats.append((vs.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), vo.mean(), np.percentile(bo, 2.5), np.percentile(bo, 97.5), len(vs), sum(len(v) for v in lv[l].values())))
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
for ax, (j, title, col) in zip(axes, [(0, "Same stance", BLUE), (3, "Opposite stance", ORANGE)]):
    m = [s[j] for s in stats]; lo = [s[j + 1] for s in stats]; hi = [s[j + 2] for s in stats]
    ax.bar(range(4), m, color=col, width=0.62)
    ax.errorbar(range(4), m, yerr=[np.array(m) - np.array(lo), np.array(hi) - np.array(m)], fmt="none", ecolor=INK2, elinewidth=1.2, capsize=3)
    for i, v in enumerate(m): ax.text(i, hi[i] + 0.015, f"{v:.2f}", ha="center", color=INK, fontsize=9)
    ax.set_xticks(range(4)); ax.set_xticklabels(names, fontsize=9, color=INK); ax.tick_params(axis="x", length=0); ax.set_ylim(0, max(hi) * 1.25); ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
axes[0].set_ylabel("Share of pairs (mean over users, 95% CI)")
fig.suptitle(f"Pooled over all {len(S)} matters: the only step is between a user and other users\n(pairs: " + ", ".join(f"{s[7]:,}" for s in stats) + ")", x=0.01, ha="left", color=INK, fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(C.FIG / "fig3_pooled_levels.png", dpi=200); plt.close(fig)

# fig4 population disagreement
rows = sorted([r for r in Cons if r["users"] >= 15 and r["n_between_pairs"] >= 100], key=lambda r: r["between_opp"])[-22:]
fig, ax = plt.subplots(figsize=(9.5, 6.8)); y = np.arange(len(rows))
ax.barh(y, [r["between_opp"] for r in rows], color=ORANGE, height=0.62)
for i, r in enumerate(rows): ax.text(r["between_opp"] + 0.005, i, f"{r['between_opp']:.2f}", va="center", color=INK, fontsize=9)
ax.set_yticks(y); ax.set_yticklabels([f"{r['matter'][:52]}  ({r['users']} users)" for r in rows], color=INK, fontsize=9); ax.tick_params(axis="y", length=0)
ax.set_xlim(0, max(r["between_opp"] for r in rows) * 1.2); ax.set_xlabel("Share of cross-user pairs with mutually exclusive stances"); ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
fig.suptitle("Where developers disagree with each other\n(matters with >=15 users; the top of the list is where personalisation matters)", x=0.01, ha="left", color=INK, fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(C.FIG / "fig4_population_disagreement.png", dpi=200); plt.close(fig)

# fig5 rewrite output composition
kind = collections.Counter(o["kind"] for o in R.values()); trig = collections.Counter(o["trigger"] for o in R.values())
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
for ax, (cnt, title, order) in zip(axes, [(kind, "Kind of request", ["task", "both", "manner"]), (trig, "What triggered it", ["unprompted", "correction", "answer"])]):
    vals = [cnt[k] for k in order]; ax.barh(range(len(order)), vals, color=BLUE, height=0.6)
    for i, v in enumerate(vals): ax.text(v + 200, i, f"{v:,} ({v / sum(cnt.values()):.0%})", va="center", color=INK, fontsize=9)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order, color=INK); ax.tick_params(axis="y", length=0); ax.set_xticks([]); ax.set_xlim(0, max(vals) * 1.35); ax.invert_yaxis()
    ax.set_title(title, loc="left", color=INK, fontsize=11)
fig.suptitle(f"Structured restatement of {len(R):,} preference-gated messages", x=0.01, ha="left", color=INK, fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(C.FIG / "fig5_rewrite_output.png", dpi=200); plt.close(fig)
print("figures written to", C.FIG)
