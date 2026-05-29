"""
LR-VCC Architecture Diagram.

Generates fig4_lr_vcc_architecture.png — boxes-and-arrows diagram showing
the LR-VCC pipeline: SR video → five parallel sub-metrics (A, T, I, D, E),
each with its own reliability gate → reliability-weighted composition →
final score. No embedded figure title (the LaTeX caption provides it).
No dominant-path emphasis: all sub-metrics are treated equally in the
composition arithmetic.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


# ── colour palette ────────────────────────────────────────────────────────────
C_INPUT      = "#D0E8FF"   # light blue-grey — input box
C_A          = "#AED6F1"   # blue   — Appearance
C_T          = "#A9DFBF"   # green  — Temporal
C_I          = "#FAD7A0"   # orange — Identity
C_D          = "#F5CBA7"   # peach  — Colour stability (D)
C_E          = "#D7DBDD"   # grey   — Colour slope (E)
C_REL        = "#F2F3F4"   # near-white — reliability gate boxes
C_COMP       = "#D7BDE2"   # lavender — composition block
C_OUT        = "#FDEDEC"   # light rose — output box
C_EDGE       = "#2C3E50"   # near-black — box edges & main arrows
C_EDGE_THIN  = "#7F8C8D"   # grey — reliability arrows


FONT_MAIN  = 9
FONT_SMALL = 7.5
FONT_TITLE = 10


# ── figure setup (no embedded title; taller for 5 rows) ──────────────────────
fig, ax = plt.subplots(figsize=(14, 9))
ax.set_xlim(0, 14)
ax.set_ylim(0, 9.5)
ax.axis("off")
ax.set_facecolor("white")
fig.patch.set_facecolor("white")


# ── primitives ────────────────────────────────────────────────────────────────

def fancy_box(x, y, w, h, color, edgecolor=C_EDGE, lw=1.4,
              label="", sublabel="", fontsize=FONT_MAIN, bold=False,
              radius=0.18):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=color, edgecolor=edgecolor, linewidth=lw, zorder=2,
    )
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    cx, cy = x + w / 2, y + h / 2
    if sublabel:
        ax.text(cx, cy + 0.13, label, ha="center", va="center",
                fontsize=fontsize, weight=weight, color=C_EDGE, zorder=3)
        ax.text(cx, cy - 0.16, sublabel, ha="center", va="center",
                fontsize=FONT_SMALL, weight="normal", color="#555555",
                zorder=3, style="italic")
    else:
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=fontsize, weight=weight, color=C_EDGE, zorder=3)


def rel_box(x, y, w, h, label):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0,rounding_size=0.10",
        facecolor=C_REL, edgecolor=C_EDGE_THIN, linewidth=1.0, zorder=2,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=FONT_SMALL, color="#444444", zorder=3)


def arrow(x1, y1, x2, y2, lw=1.6, color=C_EDGE, arrowsize=10):
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>",
        connectionstyle="arc3,rad=0.0",
        color=color, linewidth=lw,
        mutation_scale=arrowsize, zorder=1,
    )
    ax.add_patch(arr)


# ── layout: 5 sub-metric rows, evenly spaced ──────────────────────────────────
X_IN, W_IN     = 0.4, 1.7
X_SUB, W_SUB   = 2.7, 3.0
X_REL, W_REL   = 6.4, 2.0
X_COMP, W_COMP = 9.0, 2.4
X_OUT, W_OUT   = 11.8, 1.9

H_SM  = 0.85
H_REL = 0.58

# 5 sub-metric row centres
Y_A, Y_T, Y_I, Y_D, Y_E = 8.0, 6.5, 5.0, 3.5, 2.0

# composition + output centred on the column of sub-metrics
Y_COMP, H_COMP = 4.2, 2.5
Y_OUT, H_OUT   = 4.7, 0.7

# input box centred vertically with the column of sub-metrics
Y_IN, H_IN     = 4.65, 0.9


# ── column headers (top) ─────────────────────────────────────────────────────
HEADER_Y = 9.0
headers = [
    (X_IN + W_IN / 2, "Input"),
    (X_SUB + W_SUB / 2, "Sub-metrics"),
    (X_REL + W_REL / 2, "Reliability gates"),
    (X_COMP + W_COMP / 2, "Composition"),
    (X_OUT + W_OUT / 2, "Output"),
]
for cx, lbl in headers:
    ax.text(cx, HEADER_Y, lbl, ha="center", va="center",
            fontsize=FONT_SMALL + 0.5, color="#555555",
            style="oblique", weight="semibold")
    ax.plot([cx - 0.8, cx + 0.8], [HEADER_Y - 0.22, HEADER_Y - 0.22],
            color="#AAAAAA", lw=0.8, zorder=1)


# ── INPUT box ─────────────────────────────────────────────────────────────────
fancy_box(X_IN, Y_IN, W_IN, H_IN, C_INPUT,
          label="SR Video", sublabel="input sequence",
          bold=True, fontsize=FONT_TITLE, lw=2.0)


# ── 5 sub-metric boxes ────────────────────────────────────────────────────────
sm_defs = [
    (Y_A, C_A, "Sub-metric A — Appearance",
     "CLIP-IQA mean − λ·std"),
    (Y_T, C_T, "Sub-metric T — Temporal",
     "multi-k tOF, k∈{1, 5, 10, 30, 60, 120}"),
    (Y_I, C_I, "Sub-metric I — Identity",
     "slow-fast ArcFace adapter"),
    (Y_D, C_D, "Sub-metric D — Colour Stability",
     "Lab histogram L1 over k∈{60, 120}"),
    (Y_E, C_E, "Sub-metric E — Colour Slope",
     "linear regression on Lab channel means"),
]
for yc, col, lbl, sub in sm_defs:
    fancy_box(X_SUB, yc, W_SUB, H_SM, col,
              label=lbl, sublabel=sub,
              fontsize=FONT_MAIN, lw=1.6)


# ── 5 reliability gate boxes ──────────────────────────────────────────────────
rel_defs = [
    (Y_A, "std-floor · saturation"),
    (Y_T, "mask-coverage"),
    (Y_I, "face-rate · close-up bbox"),
    (Y_D, "histogram entropy"),
    (Y_E, "R² floor"),
]
for yc, lbl in rel_defs:
    yr = yc + (H_SM - H_REL) / 2
    rel_box(X_REL, yr, W_REL, H_REL, lbl)


# ── COMPOSITION block ────────────────────────────────────────────────────────
fancy_box(X_COMP, Y_COMP, W_COMP, H_COMP, C_COMP,
          label="Composition",
          sublabel="softmax(reliabilities / τ=0.2)\n→ log-mean of scores",
          bold=True, fontsize=FONT_MAIN, lw=2.0)


# ── OUTPUT box ────────────────────────────────────────────────────────────────
fancy_box(X_OUT, Y_OUT, W_OUT, H_OUT, C_OUT,
          label="LR-VCC score", sublabel="∈ [0, 1]",
          bold=True, fontsize=FONT_MAIN, lw=2.0,
          edgecolor=C_EDGE)


# ── ARROWS ────────────────────────────────────────────────────────────────────
# Input → each sub-metric
for yc, _, _, _ in sm_defs:
    arrow(X_IN + W_IN, Y_IN + H_IN / 2,
          X_SUB,       yc + H_SM / 2,
          lw=1.6, color=C_EDGE)

# Each sub-metric → its reliability gate (thin grey)
for yc, _, _, _ in sm_defs:
    yr_mid = yc + H_SM / 2
    arrow(X_SUB + W_SUB, yr_mid,
          X_REL,         yr_mid,
          lw=1.0, color=C_EDGE_THIN, arrowsize=8)

# Each reliability gate → composition (thin grey, converges)
y_comp_centre = Y_COMP + H_COMP / 2
for yc, _, _, _ in sm_defs:
    yr_mid = yc + H_SM / 2
    arrow(X_REL + W_REL, yr_mid,
          X_COMP,        y_comp_centre,
          lw=1.0, color=C_EDGE_THIN, arrowsize=8)

# Each sub-metric → composition (score, dark)
for yc, _, _, _ in sm_defs:
    arrow(X_SUB + W_SUB, yc + H_SM * 0.3,
          X_COMP,        y_comp_centre,
          lw=1.4, color=C_EDGE, arrowsize=10)

# Composition → Output
arrow(X_COMP + W_COMP, y_comp_centre,
      X_OUT,           Y_OUT + H_OUT / 2,
      lw=2.0, color=C_EDGE, arrowsize=12)


# ── small legend at the bottom (no dominant path, just dark vs. light arrows) ─
ax.text(0.4, 0.7,
        "Dark arrows: scores (along the composition's geometric mean).   "
        "Light arrows: reliabilities (along the composition's softmax weights).",
        ha="left", va="center", fontsize=FONT_SMALL - 0.5,
        color="#666666", style="italic")


# ── save ──────────────────────────────────────────────────────────────────────
OUT = "/Users/ana/Desktop/Timur/thesis_ve/proposal/figures/fig4_lr_vcc_architecture.png"
plt.tight_layout(pad=0.5)
plt.savefig(OUT, dpi=300, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close()
print(f"Saved: {OUT}")
