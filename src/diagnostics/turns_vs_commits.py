"""Session 3dded5ac — preference-active turns (upper lane) against commits
(lower lane) on one shared time axis.

NOTE: commit times/stats are real (from commit_report.py). Turn timestamps
below are read off the earlier session-4 plot and are APPROXIMATE. Replace
them with real values:

    conv = pd.read_parquet("swechat_data/conversations.parquet")
    s = conv[conv.session_id == "3dded5ac-a667-436b-a093-ad8efbdf0e31"]
    print(s[s.turn_number.isin(TURN_NUMBERS)][["turn_number", "timestamp"]])
"""

from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

T = lambda h, m, s=0: datetime(2026, 2, 25, h, m, s)
WIN = (T(4, 13, 50), T(6, 9, 23))

# turn, time, R01 (-1 low / +1 high / 0 none), other rubric axes firing
TURNS = [
    (11,  T(4, 14), -1, "R09 R12"),
    (28,  T(4, 58), -1, "R12"),
    (43,  T(5,  2),  0, "R12"),
    (49,  T(5,  3),  0, "R12"),
    (55,  T(5,  4),  0, "R12"),
    (88,  T(5,  9),  0, "R05"),
    (99,  T(5, 19), -1, "R09"),
    (115, T(5, 37), +1, "R10 R13"),
    (380, T(5, 51), +1, "R05"),
    (395, T(5, 57), +1, "R04 R05"),
    (608, T(6, 10),  0, "R12"),
]

# sha, time, files, add, del, linkage
COMMITS = [
    ("9290cb27", T(3, 33, 36), 29, 1888, 655, "none"),
    ("89ad18a3", T(5, 36, 40),  1,    1,   2, "conv"),
    ("1f780fac", T(6,  9, 10), 12,  123,  29, "hidden"),
    ("c8b33c75", T(6, 18, 58),  1,  302,   1, "none"),
    ("16bbf49f", T(6, 42,  2),  8,   70,  46, "none"),
    ("7569faea", T(7,  1, 44),  4,   81,  20, "none"),
    ("a46f6c26", T(7, 13, 36), 34, 1419, 488, "none_attr"),
]

C_LOW, C_HIGH, C_NONE = "#b2182b", "#2166ac", "#9e9e9e"
CCOL = {"conv": "#2166ac", "hidden": "#e08214",
        "none": "#cfcfcf", "none_attr": "#2166ac"}
TURN_Y, COMMIT_Y = 1.0, 0.0

fig, ax = plt.subplots(figsize=(15, 8.8))

ax.axvspan(*WIN, color="#2166ac", alpha=0.06, zorder=0)
for x in WIN:
    ax.axvline(x, color="#2166ac", lw=1, alpha=0.3, ls=":")
ax.text(WIN[0], 1.92, "  session window  04:13 – 06:09", fontsize=10,
        color="#2166ac", style="italic")

for y, lab in [(TURN_Y, "PREFERENCE-ACTIVE TURNS"), (COMMIT_Y, "COMMITS")]:
    ax.axhline(y, color="#d5d5d5", lw=1.2, zorder=1)
    ax.text(T(3, 18), y + 0.13, lab, fontsize=10, weight="bold", color="#666")

# ---- turns -----------------------------------------------------------
stagger = {43: 0.20, 49: 0.36, 55: 0.52}          # 43/49/55 are ~1 min apart
for n, t, r01, axes in TURNS:
    col = C_LOW if r01 == -1 else (C_HIGH if r01 == 1 else C_NONE)
    ax.scatter(t, TURN_Y, s=175, color=col, edgecolor="white", lw=1.2, zorder=4)
    dy = 0.20 + stagger.get(n, 0.0)
    ax.plot([t, t], [TURN_Y + 0.05, TURN_Y + dy - 0.03],
            color="#cccccc", lw=0.7, zorder=2)
    ax.text(t, TURN_Y + dy, f"{n}\n{axes}", ha="center", va="bottom",
            fontsize=8.5, linespacing=1.4, color="#333")

# ---- commits ---------------------------------------------------------
size = lambda a, d: 80 + 560 * ((a + d) / 1907) ** 0.5
for i, (sha, t, nf, add, dele, link) in enumerate(COMMITS):
    ax.scatter(t, COMMIT_Y, s=size(add, dele), color=CCOL[link],
               edgecolor="white", lw=1.3, zorder=4)
    dark = link != "none"
    ax.text(t, COMMIT_Y - 0.24, f"{sha}\n{nf} files  +{add:,}/−{dele:,}",
            ha="center", va="top", fontsize=8.8, linespacing=1.4,
            color="#222" if dark else "#8a8a8a")

# ---- the two links that matter --------------------------------------
ax.annotate("", xy=(T(6, 9, 10), COMMIT_Y + 0.13),
            xytext=(T(5, 57), TURN_Y - 0.13),
            arrowprops=dict(arrowstyle="->", color="#e08214", lw=1.8,
                            connectionstyle="arc3,rad=0.18"))
ax.text(T(5, 58), -0.78,
        "turns 115 / 395 → 1f780fac\n"
        "matches item for item, and is linked to this\n"
        "session — but only via a 2nd checkpoint that\n"
        "conversations.checkpoint_pk never reports.",
        fontsize=9.2, color="#b06a10", linespacing=1.7, va="top")

ax.annotate("", xy=(T(7, 13, 36), COMMIT_Y + 0.16),
            xytext=(T(5, 19), TURN_Y - 0.13),
            arrowprops=dict(arrowstyle="->", color="#b2182b", lw=1.8,
                            ls="--", connectionstyle="arc3,rad=-0.22"))
ax.text(T(3, 20), -0.78,
        "turns 11–99 discuss the schema change.\n"
        "Nearest commit after them is 89ad18a3 — a 3-line wording fix.\n"
        "The change itself lands in a46f6c26, 64 min after the last turn,\n"
        "in a squash with 0 agent_changes and 34/34 files human_only.",
        fontsize=9.2, color="#8c1a24", linespacing=1.7, va="top")

ax.set_ylim(-1.75, 2.15)
ax.set_xlim(T(3, 15), T(7, 42))
ax.get_yaxis().set_visible(False)
for sp in ("left", "right", "top"):
    ax.spines[sp].set_visible(False)
ax.spines["bottom"].set_color("#aaaaaa")
ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0, 30]))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax.tick_params(axis="x", labelsize=10.5, colors="#444")
ax.set_xlabel("25 Feb 2026, UTC", fontsize=11, labelpad=8)
ax.set_title("Session 3dded5ac — 11 of 626 turns carry a preference signal;\n"
             "the commits they map to are reached by three different paths",
             fontsize=13.5, pad=16, linespacing=1.5)

ax.legend(handles=[
    Line2D([], [], marker="o", ls="", ms=10, mfc=C_LOW, mec="white",
           label="turn: R01 low"),
    Line2D([], [], marker="o", ls="", ms=10, mfc=C_HIGH, mec="white",
           label="turn: R01 high"),
    Line2D([], [], marker="o", ls="", ms=10, mfc=C_NONE, mec="white",
           label="turn: other rubric axes only"),
    Line2D([], [], marker="o", ls="", ms=11, mfc=CCOL["conv"], mec="white",
           label="commit: via conversations.checkpoint_pk"),
    Line2D([], [], marker="o", ls="", ms=11, mfc=CCOL["hidden"], mec="white",
           label="commit: 2nd checkpoint, not in conversations"),
    Line2D([], [], marker="o", ls="", ms=11, mfc=CCOL["none"], mec="white",
           label="commit: not linked to this session"),
], loc="upper center", bbox_to_anchor=(0.5, -0.13), frameon=False,
    fontsize=9.5, ncol=3, columnspacing=2.0)

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/turns_vs_commits.png", dpi=200,
            bbox_inches="tight", facecolor="white")
print("saved")