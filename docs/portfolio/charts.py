import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

# Validated categorical slots 1-3 (dataviz reference palette, light mode)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8985", "#e4e3df"

plt.rcParams.update({
    "svg.fonttype": "path",            # text -> outlines: renders identically in print
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

# ---- data: replayed from cached Stage 3 answers (src/sweep.py) ----
tau = np.array([.05,.10,.15,.20,.25,.30,.35,.40,.45,.50,.55,.60,.65,.70,.75,.80,.85,.90])
micro = np.array([.914,.914,.914,.914,.914,.914,.914,.914,.914,.914,.913,.911,.910,.909,.907,.905,.902,.895])
macro = np.array([.543,.543,.543,.542,.542,.543,.546,.550,.553,.560,.580,.586,.596,.607,.614,.623,.640,.647])
zs    = np.array([.214,.214,.214,.214,.214,.214,.237,.237,.241,.263,.287,.310,.330,.341,.349,.366,.397,.407])

fig, ax = plt.subplots(figsize=(6.3, 3.05))
ax.grid(True, axis="y", color=GRID, lw=.6, zorder=0)
ax.set_axisbelow(True)
for y, c, lab in ((micro, BLUE, "micro"), (macro, ORANGE, "macro"), (zs, AQUA, "zero-shot macro")):
    ax.plot(tau, y, color=c, lw=2, solid_capstyle="round", zorder=3)
    ax.annotate(lab, (tau[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
                color=c, fontsize=8.5, fontweight="bold", va="center")
ax.axvline(.90, color=MUTED, lw=1, ls=(0, (3, 3)), zorder=1)
ax.annotate("operating point\n$\\tau$ = 0.900", (.90, .17), xytext=(-8, 0),
            textcoords="offset points", ha="right", va="bottom", color=INK2, fontsize=7.5)
ax.set_xlim(.03, 1.06); ax.set_ylim(.15, .98)
ax.set_xlabel("deferral threshold  $\\tau$", color=INK2)
ax.set_ylabel("accuracy on 12,200 held-out items", color=INK2)
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))
ax.set_xticks([.1,.3,.5,.7,.9])
fig.tight_layout(pad=.4)
fig.savefig("fig_frontier.svg", format="svg", bbox_inches="tight")
plt.close(fig)

# ---- latency: 200 served requests, live Anthropic provider (src/serve/bench.py) ----
stages = ["S1\n82% of requests", "S3\n17%", "S3-abstain\n1%"]
p50 = np.array([18.8, 2128.1, 2106.5]); p95 = np.array([78.8, 2410.3, 2165.1])
y = np.arange(len(stages))[::-1]

fig, ax = plt.subplots(figsize=(6.3, 2.35))
ax.grid(True, axis="x", color=GRID, lw=.6, zorder=0); ax.set_axisbelow(True)
for i, yy in enumerate(y):
    ax.plot([p50[i], p95[i]], [yy, yy], color=GRID, lw=3, solid_capstyle="round", zorder=2)
    ax.scatter([p50[i]], [yy], s=46, color=BLUE, zorder=4, edgecolor="white", linewidth=1.6)
    ax.scatter([p95[i]], [yy], s=46, color=ORANGE, zorder=4, edgecolor="white", linewidth=1.6)
    # Labels go OUTSIDE the pair, not above it: on a log axis p50 and p95 for the
    # escalated rows are ~300ms apart and centred labels overprint each other.
    ax.annotate(f"{p50[i]:,.0f}", (p50[i], yy), xytext=(-9, 0), textcoords="offset points",
                ha="right", va="center", color=BLUE, fontsize=7.5, fontweight="bold")
    ax.annotate(f"{p95[i]:,.0f}", (p95[i], yy), xytext=(9, 0), textcoords="offset points",
                ha="left", va="center", color=ORANGE, fontsize=7.5, fontweight="bold")
ax.set_xscale("log"); ax.set_xlim(5.5, 26000)   # headroom for outboard labels
ax.set_yticks(y); ax.set_yticklabels(stages, fontsize=8, color=INK)
ax.set_ylim(-0.65, len(stages) - 0.25)
ax.set_xticks([10, 100, 1000]); ax.set_xticklabels(["10 ms", "100 ms", "1 s"])
ax.set_xlabel("end-to-end latency, log scale", color=INK2)
ax.scatter([], [], s=46, color=BLUE, label="p50"); ax.scatter([], [], s=46, color=ORANGE, label="p95")
ax.legend(loc="lower left", frameon=False, fontsize=8, ncol=2,
          handletextpad=.3, columnspacing=1.2, bbox_to_anchor=(0.015, 0.02))
fig.tight_layout(pad=.4)
fig.savefig("fig_latency.svg", format="svg", bbox_inches="tight")
plt.close(fig)
print("wrote fig_frontier.svg, fig_latency.svg")
