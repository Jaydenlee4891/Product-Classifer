"""One real item through all four tiers. Every value is read from the run artefacts."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image
import numpy as np, os

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, RULE, SURF = "#141412", "#4a4945", "#8a8985", "#e4e3df", "#f8f8f6"
W, H, DPI = 10.0, 5.9, 110
plt.rcParams["font.family"] = "DejaVu Sans"

ITEM_T = "AmazonBasics Modern Euro Toilet Paper Holder"
ITEM_B = "Brand: AmazonBasics   ·   ASIN B07765Z6S4   ·   stratum: tail"
GOLD = "TOILET_PAPER_HOLDER"
BI = ["TOILET_PAPER_HOLDER","PAPER_TOWEL_HOLDER","CLEANING_BRUSH","TOOTHBRUSH_HOLDER",
      "PLUMBING_FIXTURE","JANITORIAL_SUPPLY","TOWEL_HOLDER","WASTE_BAG","PUMP_DISPENSER","TOILET_SEAT"]
CE = ["TOILET_PAPER_HOLDER","CLEANING_BRUSH","PLUMBING_FIXTURE","HOME","FOOD_SERVICE_SUPPLY",
      "PAPER_TOWEL_HOLDER","POT_HOLDER","TOWEL_HOLDER","BEAUTY","AUTO_ACCESSORY"]
FUSED = ["TOILET_PAPER_HOLDER","PAPER_TOWEL_HOLDER","CLEANING_BRUSH","PLUMBING_FIXTURE",
         "TOOTHBRUSH_HOLDER","HOME","FOOD_SERVICE_SUPPLY","JANITORIAL_SUPPLY","TOWEL_HOLDER","POT_HOLDER"]
TIERS = [("S1", "DistilBERT\n69 classes + OTHER"), ("S2a", "bi-encoder\nretrieve 50 of 530"),
         ("S2b", "cross-encoder\nrerank + fuse"), ("S3", "LLM agent\nforced tool call")]

def base(active=None, step=""):
    fig = plt.figure(figsize=(W, H), dpi=DPI, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.add_patch(Rectangle((0, .955), 1, .045, color=INK, zorder=1))
    ax.text(.028, .971, "LONG-TAIL CASCADE CLASSIFIER", color="white", fontsize=9.5,
            fontweight="bold", family="DejaVu Sans")
    ax.text(.972, .971, "one item · four tiers · real trace", color="#b9b8b2", fontsize=9,
            ha="right")
    for i, (tag, desc) in enumerate(TIERS):
        y = .775 - i * .175
        on = active == i; done = active is not None and i < active
        fc = BLUE if on else (SURF if not done else "#eef4fc")
        ec = BLUE if on else (RULE if not done else "#c3dbf5")
        ax.add_patch(FancyBboxPatch((.028, y), .185, .128, boxstyle="round,pad=0.006",
                     fc=fc, ec=ec, lw=1.4, zorder=2))
        ax.text(.045, y + .086, tag, fontsize=12, fontweight="bold", zorder=3,
                color="white" if on else (INK if not done else BLUE))
        ax.text(.045, y + .028, desc, fontsize=7.4, zorder=3, linespacing=1.45,
                color="white" if on else (INK2 if not done else "#6c9bd6"))
        if i < 3:
            ax.annotate("", xy=(.1205, y - .012), xytext=(.1205, y - .001),
                        arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2))
    if step:
        ax.text(.028, .06, step, fontsize=8, color=MUTED)
    return fig, ax

def card(ax, x, y, w, h, fc=SURF, ec=RULE, lw=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006",
                 fc=fc, ec=ec, lw=lw, zorder=2))

def item_card(ax, y=.80):
    card(ax, .255, y, .715, .115, fc="white", ec=INK, lw=1.6)
    ax.text(.272, y + .068, ITEM_T, fontsize=11.5, fontweight="bold", color=INK, zorder=3)
    ax.text(.272, y + .028, ITEM_B, fontsize=8, color=MUTED, zorder=3)

def ranklist(ax, items, x, y0, title, hi=None, marks=None, w=.325, fs=7.8):
    # +.068 not +.045: the first card's top edge is at y0+.046, so a lower baseline
    # puts the header inside the card and the border strikes through it.
    ax.text(x, y0 + .070, title, fontsize=8.2, fontweight="bold", color=INK2, zorder=3)
    for i, name in enumerate(items):
        yy = y0 - i * .0545
        is_gold = name == GOLD
        fc = "#eaf6f0" if is_gold else "white"
        ec = AQUA if is_gold else RULE
        card(ax, x, yy, w, .046, fc=fc, ec=ec, lw=1.5 if is_gold else .9)
        ax.text(x + .012, yy + .015, f"{i+1}", fontsize=7, color=MUTED, zorder=3)
        ax.text(x + .038, yy + .014, name, fontsize=fs, zorder=3,
                color=INK if is_gold else INK2,
                fontweight="bold" if is_gold else "normal")
        if marks and name in marks:
            ax.text(x + w - .012, yy + .014, marks[name], fontsize=7.4, ha="right",
                    zorder=3, color=marks[name + "__c"] if name + "__c" in marks else ORANGE,
                    fontweight="bold")

# Hold time per frame in ms. Repeating identical frames does NOT work: PIL's optimiser
# collapses them back into one and the pacing is silently lost.
frames, holds = [], []
def emit(fig, ms=2500):
    p = f"f{len(frames):03d}.png"; fig.savefig(p, dpi=DPI); plt.close(fig)
    frames.append(p); holds.append(ms)

# 0 — the item
fig, ax = base(step="Every number in this animation is read from the run artefacts on disk.")
item_card(ax, .62)
ax.text(.255, .50, "530 leaf categories.  90 of them have no training data at all.",
        fontsize=11, color=INK)
ax.text(.255, .43, "Which one is this?", fontsize=13, fontweight="bold", color=INK)
emit(fig, 2100)

# 1 — S1 fires
fig, ax = base(0, "Tier 1 · one DistilBERT forward pass · ~21 ms")
item_card(ax)
ax.text(.255, .70, "S1 scores 69 head classes plus an explicit OTHER class",
        fontsize=9.5, color=INK2)
for i, (name, pv) in enumerate([("__OTHER__", .9987), ("HOME", .0003), ("TOOLS", .0002)]):
    yy = .565 - i * .075
    card(ax, .255, yy, .715, .062, fc="white" if i else "#fdf1ec", ec=ORANGE if i == 0 else RULE,
         lw=1.6 if i == 0 else .9)
    ax.text(.272, yy + .021, name, fontsize=10 if i == 0 else 8.8, zorder=3,
            color=INK, fontweight="bold" if i == 0 else "normal")
    ax.text(.955, yy + .021, f"{pv:.4f}", fontsize=10 if i == 0 else 8.8, ha="right",
            zorder=3, color=ORANGE if i == 0 else MUTED,
            fontweight="bold" if i == 0 else "normal")
    ax.add_patch(Rectangle((.60, yy + .014), .30 * pv, .028, color=ORANGE if i == 0 else RULE,
                 zorder=3, alpha=.25))
emit(fig, 2600)

# 2 — the gate: confident AND deferred
fig, ax = base(0, "Tier 1 · the deferral gate")
item_card(ax)
card(ax, .255, .565, .715, .062, fc="#fdf1ec", ec=ORANGE, lw=1.6)
ax.text(.272, .586, "__OTHER__", fontsize=10, fontweight="bold", zorder=3, color=INK)
ax.text(.955, .586, "0.9987", fontsize=10, ha="right", zorder=3, color=ORANGE, fontweight="bold")
card(ax, .255, .30, .715, .21, fc="white", ec=ORANGE, lw=1.6)
ax.text(.275, .445, "defer  if   argmax == OTHER   OR   p_max < 0.900",
        fontsize=11.5, fontweight="bold", color=INK, family="DejaVu Sans Mono", zorder=3)
ax.text(.275, .385, "99.87% confident — and deferred anyway.", fontsize=10.5,
        color=ORANGE, fontweight="bold", zorder=3)
ax.text(.275, .335, "The gate is a disjunction, not a threshold. A class the model cannot\n"
                    "represent does not produce low confidence — it produces a confident OTHER.",
        fontsize=8.6, color=INK2, linespacing=1.6, zorder=3)
emit(fig, 3800)

# 3 — retrieval
fig, ax = base(1, "Tier 2a · frozen Qwen3-Embedding-0.6B against 530 label documents")
ax.text(.255, .855, "Retrieve: cosine over label documents, not label names",
        fontsize=9.5, color=INK2)
ax.text(.255, .805, "530  →  top 50  →  showing top 10", fontsize=8.4, color=MUTED)
ranklist(ax, BI, .255, .700, "BI-ENCODER ORDER", w=.335)
card(ax, .635, .30, .335, .42, fc=SURF, ec=RULE)
ax.text(.655, .655, "Why documents, not names", fontsize=9, fontweight="bold", color=INK, zorder=3)
ax.text(.655, .40, "A leaf named ACCESSORY is\nunfindable. The same leaf with its\n"
        "hierarchy path and three real product\ntitles is findable — including at zero\n"
        "training examples.\n\nmacro recall@50\n"
        "  name only          0.837\n  + hierarchy path   0.945\n  + examples         0.977",
        fontsize=8, color=INK2, linespacing=1.75, zorder=3, family="DejaVu Sans")
emit(fig, 3200)

# 4 — rerank
fig, ax = base(2, "Tier 2b · fine-tuned MiniLM cross-encoder re-scores the 50")
ranklist(ax, BI, .255, .785, "BI-ENCODER", w=.245, fs=7.2)
ranklist(ax, CE, .530, .785, "CROSS-ENCODER", w=.245, fs=7.2,
         marks={"HOME": "new", "FOOD_SERVICE_SUPPLY": "new", "BEAUTY": "new",
                "AUTO_ACCESSORY": "new"})
card(ax, .805, .245, .165, .60, fc=SURF, ec=RULE)
ax.text(.820, .795, "The reranker is\nworse overall", fontsize=8.6, fontweight="bold",
        color=INK, zorder=3, linespacing=1.5)
ax.text(.820, .40, "macro recall@10\n\n  bi-encoder   0.919\n  cross-enc.   0.760\n\n"
        "It buries the 90\nzero-training leaves:\nthey appear in its\ntraining groups only\n"
        "as hard negatives.\n\nzero-shot recall\n  0.124", fontsize=7.4, color=INK2,
        linespacing=1.7, zorder=3)
emit(fig, 3600)

# 5 — fuse
fig, ax = base(2, "Tier 2b · interleave both orderings, dedup, take 10")
ranklist(ax, FUSED, .255, .785, "FUSED SHORTLIST — what the agent will read", w=.40)
card(ax, .690, .245, .280, .60, fc=SURF, ec=RULE)
ax.text(.708, .795, "Keep both, not the\nbetter one", fontsize=8.8, fontweight="bold",
        color=INK, zorder=3, linespacing=1.5)
ax.text(.708, .40, "Fusing costs 0.006 on few-shot\nand 0.016 on zero-shot, and buys\n"
        "0.137 on head and 0.074 on torso.\n\nThe cross-encoder is not a\nshortlist producer. It is a\n"
        "complement to a retriever that\nnever learned which labels\nare rare.\n\n"
        "macro recall@10  0.931", fontsize=7.8, color=INK2, linespacing=1.7, zorder=3)
emit(fig, 3200)

# 6 — the agent
fig, ax = base(3, "Tier 3 · candidates shuffled, forced tool call · ~1,800 ms")
ax.text(.255, .855, "The 10 candidates are SHUFFLED before the model sees them",
        fontsize=9.5, color=INK, fontweight="bold")
ax.text(.255, .805, "Presented in rank order an LLM anchors on position 1 — and you have built\n"
        "an expensive way to agree with the bi-encoder.", fontsize=8.2, color=INK2, linespacing=1.6)
card(ax, .255, .30, .715, .44, fc="#0f1720", ec="#0f1720")
ax.text(.278, .665, "tool_choice = {\"type\": \"tool\", \"name\": \"record_category\"}",
        fontsize=8.6, color="#7fb2f0", family="DejaVu Sans Mono", zorder=3)
ax.text(.278, .545, "record_category(", fontsize=10.5, color="#e8e6df",
        family="DejaVu Sans Mono", zorder=3)
ax.text(.300, .470, "leaf_id     = \"TOILET_PAPER_HOLDER\",", fontsize=10.5, color="#8fe3bd",
        family="DejaVu Sans Mono", zorder=3)
ax.text(.300, .400, "confidence  = \"certain\"", fontsize=10.5, color="#f0b48e",
        family="DejaVu Sans Mono", zorder=3)
ax.text(.278, .330, ")", fontsize=10.5, color="#e8e6df", family="DejaVu Sans Mono", zorder=3)
emit(fig, 3600)

# 7 — result
fig, ax = base(3, "Answered by S3 · one of the 16.5% of items that reach the agent tier")
item_card(ax, .78)
card(ax, .255, .445, .715, .28, fc="#eaf6f0", ec=AQUA, lw=2)
ax.text(.278, .625, "TOILET_PAPER_HOLDER", fontsize=19, fontweight="bold", color=INK, zorder=3)
ax.text(.278, .545, "correct  ·  gold leaf was rank 1 of 10 in the shortlist", fontsize=9.5,
        color=AQUA, fontweight="bold", zorder=3)
ax.text(.278, .490, "tail stratum — 231 of the 530 leaves have fewer than ten training items",
        fontsize=8.4, color=INK2, zorder=3)
for i, (k, v, c) in enumerate([("83.5%", "answered by S1 alone", BLUE),
                               ("16.5%", "escalate, like this one", ORANGE),
                               ("0.647", "macro over 530 leaves", AQUA)]):
    x = .255 + i * .242
    ax.text(x, .30, k, fontsize=17, fontweight="bold", color=c)
    ax.text(x, .245, v, fontsize=7.8, color=MUTED)
emit(fig, 4400)

imgs = [Image.open(f).convert("P", palette=Image.ADAPTIVE, colors=128) for f in frames]
imgs[0].save("cascade_walkthrough.gif", save_all=True, append_images=imgs[1:],
             duration=holds, loop=0, optimize=True, disposal=2)
print("total runtime:", sum(holds) / 1000, "s")
for f in set(frames): os.remove(f)
print("frames:", len(frames), "| size:", round(os.path.getsize("cascade_walkthrough.gif")/1e6, 2), "MB")
