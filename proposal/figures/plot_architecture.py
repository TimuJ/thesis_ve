"""
LR-VCC Architecture Diagram
Generates fig4_lr_vcc_architecture.png — boxes-and-arrows diagram showing
the LR-VCC pipeline: SR video → three parallel sub-metrics → composition → score.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ── colour palette ────────────────────────────────────────────────────────────
C_INPUT      = "#D0E8FF"   # light blue-grey — input box
C_APP        = "#AED6F1"   # blue  — Appearance sub-metric
C_TMP        = "#A9DFBF"   # green — Temporal sub-metric
C_IDN        = "#FAD7A0"   # orange — Identity sub-metric
C_REL        = "#F2F3F4"   # near-white — reliability test boxes
C_COMP       = "#D7BDE2"   # lavender — composition block
C_OUT        = "#FDEDEC"   # light rose — output box
C_EDGE       = "#2C3E50"   # near-black — box edges & arrows
C_EDGE_THIN  = "#7F8C8D"   # grey — thin/secondary arrows
C_EDGE_BOLD  = "#1A252F"   # darker — dominant-path arrows

FONT_MAIN  = 9
FONT_SMALL = 7.5
FONT_TITLE = 10

# ── figure setup ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 7))
ax.set_xlim(0, 14)
ax.set_ylim(0, 7)
ax.axis("off")
ax.set_facecolor("white")
fig.patch.set_facecolor("white")

# ── helper functions ──────────────────────────────────────────────────────────

def fancy_box(ax, x, y, w, h, color, edgecolor=C_EDGE, lw=1.4,
              label="", sublabel="", fontsize=FONT_MAIN, bold=False,
              radius=0.18):
    """Draw a rounded rectangle and optional centred text."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=color, edgecolor=edgecolor, linewidth=lw, zorder=2
    )
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    cx, cy = x + w / 2, y + h / 2
    if sublabel:
        ax.text(cx, cy + 0.13, label, ha="center", va="center",
                fontsize=fontsize, weight=weight, color=C_EDGE, zorder=3)
        ax.text(cx, cy - 0.17, sublabel, ha="center", va="center",
                fontsize=FONT_SMALL, weight="normal", color="#555555", zorder=3,
                style="italic")
    else:
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=fontsize, weight=weight, color=C_EDGE, zorder=3)


def arrow(ax, x1, y1, x2, y2, lw=1.6, color=C_EDGE, style="->",
          arrowsize=12, conn="arc3,rad=0.0"):
    """Draw an arrow from (x1,y1) to (x2,y2)."""
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=f"-|>",
        connectionstyle=conn,
        color=color, linewidth=lw,
        mutation_scale=arrowsize,
        zorder=1
    )
    ax.add_patch(arr)


def rel_box(ax, x, y, w, h, label):
    """Small reliability-test box."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0,rounding_size=0.12",
        facecolor=C_REL, edgecolor=C_EDGE_THIN, linewidth=1.0, zorder=2
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=FONT_SMALL, color="#444444", zorder=3)


# ═══════════════════════════════════════════════════════════════════════════════
# LAYOUT  (all coordinates in data units; figure is 14 × 7)
# Column centres: INPUT=1.2, SUB=4.5, REL=7.1, COMP=10.3, OUT=13.0
# ═══════════════════════════════════════════════════════════════════════════════

# ── column x positions ────────────────────────────────────────────────────────
X_IN   = 0.4
W_IN   = 1.8

X_SUB  = 2.9
W_SUB  = 2.4

X_REL  = 5.6
W_REL  = 1.8

X_COMP = 7.7
W_COMP = 2.8

X_OUT  = 10.8
W_OUT  = 2.6

# ── row y centres for three sub-metrics ──────────────────────────────────────
H_SM   = 0.85    # height of a sub-metric box
H_REL  = 0.58    # height of a reliability box

Y_APP  = 5.0
Y_TMP  = 3.35
Y_IDN  = 1.70

# y centre of composition block
Y_COMP = 3.0
H_COMP = 1.9

# input box
Y_IN   = Y_TMP + (H_SM / 2) - 0.45
H_IN   = 0.9

# output box
Y_OUT  = Y_COMP + 0.35
H_OUT  = 0.65


# ── 1. INPUT box ─────────────────────────────────────────────────────────────
fancy_box(ax, X_IN, Y_IN, W_IN, H_IN, C_INPUT,
          label="SR Video",
          sublabel="input sequence",
          bold=True, fontsize=FONT_TITLE, lw=2.0)

# ── 2. Sub-metric boxes ───────────────────────────────────────────────────────
sm_defs = [
    (Y_APP, C_APP,  "Sub-metric A — Appearance",  "CLIP-IQA mean − λ·std"),
    (Y_TMP, C_TMP,  "Sub-metric T — Temporal",    "log(1+k)-wtd tOF, k∈{1…120}"),
    (Y_IDN, C_IDN,  "Sub-metric I — Identity",    "slow-fast ArcFace adapter"),
]

for (yc, col, lbl, sub) in sm_defs:
    # Sub-metric box (bold border for dominant temporal path)
    lw = 2.2 if "Temporal" in lbl else 1.6
    ec = C_EDGE_BOLD if "Temporal" in lbl else C_EDGE
    fancy_box(ax, X_SUB, yc, W_SUB, H_SM, col,
              label=lbl, sublabel=sub,
              fontsize=FONT_MAIN, lw=lw, edgecolor=ec)

# ── 3. Reliability boxes ──────────────────────────────────────────────────────
rel_defs = [
    (Y_APP + (H_SM - H_REL) / 2, "Reliability A\nstd-floor · saturation"),
    (Y_TMP + (H_SM - H_REL) / 2, "Reliability T\nmask-coverage"),
    (Y_IDN + (H_SM - H_REL) / 2, "Reliability I\nface-rate · close-up bbox"),
]
for (yr, lbl) in rel_defs:
    rel_box(ax, X_REL, yr, W_REL, H_REL, lbl)

# ── 4. Composition block ──────────────────────────────────────────────────────
fancy_box(ax, X_COMP, Y_COMP, W_COMP, H_COMP, C_COMP,
          label="Composition",
          sublabel="softmax(reliabilities/τ=0.2)\n→ reliability-wtd log-mean",
          bold=True, fontsize=FONT_MAIN, lw=2.0)

# ── 5. Output box ─────────────────────────────────────────────────────────────
fancy_box(ax, X_OUT, Y_OUT, W_OUT, H_OUT, C_OUT,
          label="LR-VCC score",
          sublabel="∈ [0, 1]  (per video × method)",
          bold=True, fontsize=FONT_MAIN, lw=2.2, edgecolor=C_EDGE_BOLD)

# ═══════════════════════════════════════════════════════════════════════════════
# ARROWS
# ═══════════════════════════════════════════════════════════════════════════════

# 1. Input → each sub-metric (left edge of sub-metric box)
arrow_lw_map = {"A": 1.5, "T": 2.4, "I": 1.5}
for (yc, col, lbl, _) in sm_defs:
    key = lbl.split("—")[0].strip().split()[-1]   # "A", "T", or "I"
    lw  = arrow_lw_map.get(key, 1.6)
    ec  = C_EDGE_BOLD if key == "T" else C_EDGE
    arrow(ax,
          X_IN + W_IN,       Y_IN + H_IN / 2,
          X_SUB,             yc + H_SM / 2,
          lw=lw, color=ec, arrowsize=10,
          conn=f"arc3,rad={0.0}")

# 2. Sub-metric → reliability (right edge of SM → left edge of REL)
for (yc, _, lbl, _) in sm_defs:
    yr = yc + (H_SM - H_REL) / 2 + H_REL / 2
    arrow(ax,
          X_SUB + W_SUB, yc + H_SM / 2,
          X_REL,         yr,
          lw=1.2, color=C_EDGE_THIN, arrowsize=8)

# 3. Sub-metric score → composition  (right edge of SM → left edge of COMP)
#    and reliability → composition   (right edge of REL → left edge of COMP)
for (yc, _, lbl, _) in sm_defs:
    key = lbl.split("—")[0].strip().split()[-1]
    lw  = 2.4 if key == "T" else 1.5
    ec  = C_EDGE_BOLD if key == "T" else C_EDGE

    # score arrow from SM right edge to comp left edge
    yr_comp = Y_COMP + H_COMP / 2
    arrow(ax,
          X_SUB + W_SUB, yc + H_SM * 0.35,
          X_COMP,        yr_comp,
          lw=lw, color=ec, arrowsize=10,
          conn="arc3,rad=0.0")

    # reliability arrow from REL right edge to comp left edge
    yr_rel = yc + (H_SM - H_REL) / 2 + H_REL / 2
    arrow(ax,
          X_REL + W_REL, yr_rel,
          X_COMP,        yr_comp,
          lw=1.2, color=C_EDGE_THIN, arrowsize=8,
          conn="arc3,rad=0.0")

# 4. Composition → Output
arrow(ax,
      X_COMP + W_COMP, Y_COMP + H_COMP / 2,
      X_OUT,           Y_OUT + H_OUT / 2,
      lw=2.5, color=C_EDGE_BOLD, arrowsize=12)

# ═══════════════════════════════════════════════════════════════════════════════
# LABELS & ANNOTATIONS
# ═══════════════════════════════════════════════════════════════════════════════

# "(score, reliability)" badges on each SM box right side
badge_y_offsets = [0.0, 0.0, 0.0]
for i, (yc, _, _, _) in enumerate(sm_defs):
    ax.text(X_SUB + W_SUB - 0.06, yc + H_SM * 0.67,
            "(score, reliability)",
            ha="right", va="center",
            fontsize=FONT_SMALL - 0.5, color="#2C6E8A",
            style="italic", zorder=4)

# Column headers (above the boxes)
header_y = 6.25
col_headers = [
    (X_IN + W_IN / 2,           "Input"),
    (X_SUB + W_SUB / 2,         "Sub-metrics"),
    (X_REL + W_REL / 2,         "Reliability tests"),
    (X_COMP + W_COMP / 2,       "Composition"),
    (X_OUT + W_OUT / 2,         "Output"),
]
for (cx, lbl) in col_headers:
    ax.text(cx, header_y, lbl, ha="center", va="center",
            fontsize=FONT_SMALL + 0.5, color="#555555",
            style="oblique", weight="semibold")
    # thin horizontal rule under header
    ax.plot([cx - 0.7, cx + 0.7], [header_y - 0.22, header_y - 0.22],
            color="#AAAAAA", lw=0.8, zorder=1)

# Legend note — dominant path
ax.text(0.42, 0.36,
        "Thicker border / heavier arrows = dominant\nlong-range temporal path (Sub-metric T)",
        ha="left", va="bottom", fontsize=FONT_SMALL - 0.5,
        color="#444444", transform=ax.transData,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FDFEFE",
                  edgecolor="#AAAAAA", linewidth=0.8))

# Figure title
ax.set_title(
    "Figure 4 — LR-VCC Architecture: Long-Range Video Consistency Composite",
    fontsize=FONT_TITLE + 1, weight="bold", color=C_EDGE, pad=8
)

# ── save ──────────────────────────────────────────────────────────────────────
out_path = "/Users/ana/Desktop/Timur/thesis_ve/proposal/figures/fig4_lr_vcc_architecture.png"
plt.tight_layout(pad=0.5)
plt.savefig(out_path, dpi=300, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close()
print(f"Saved: {out_path}")
