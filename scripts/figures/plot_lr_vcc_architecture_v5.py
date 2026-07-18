"""LR-VCC v5 architecture diagram (7 sub-metrics) for the thesis.

Derived from proposal/figures/plot_architecture.py (5 sub-metrics); adds the
anchored sub-metrics D' and D'' introduced in v5 and corrects the reliability
labels to match the production implementation (D: sequence-length floor, not
histogram entropy). Saves into zjuthesis/figure/.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


C_INPUT = "#D0E8FF"
C_A = "#AED6F1"
C_T = "#A9DFBF"
C_I = "#FAD7A0"
C_D = "#F5CBA7"
C_DP = "#F9E79F"    # yellow — anchored colour (D')
C_DPP = "#F5B7B1"   # salmon — semantic trajectory (D'')
C_E = "#D7DBDD"
C_REL = "#F2F3F4"
C_COMP = "#D7BDE2"
C_OUT = "#FDEDEC"
C_EDGE = "#2C3E50"
C_EDGE_THIN = "#7F8C8D"

FONT_MAIN = 9
FONT_SMALL = 7.5
FONT_TITLE = 10

fig, ax = plt.subplots(figsize=(14, 11.5))
ax.set_xlim(0, 14)
ax.set_ylim(0, 12.2)
ax.axis("off")
ax.set_facecolor("white")
fig.patch.set_facecolor("white")


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


X_IN, W_IN = 0.4, 1.7
X_SUB, W_SUB = 2.7, 3.2
X_REL, W_REL = 6.6, 2.1
X_COMP, W_COMP = 9.2, 2.4
X_OUT, W_OUT = 12.0, 1.8

H_SM = 0.85
H_REL = 0.58

ROW_Y = [10.4, 9.0, 7.6, 6.2, 4.8, 3.4, 2.0]

Y_COMP, H_COMP = 5.0, 2.5
Y_OUT, H_OUT = 5.85, 0.7
Y_IN, H_IN = 5.75, 0.9

HEADER_Y = 11.6
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

fancy_box(X_IN, Y_IN, W_IN, H_IN, C_INPUT,
          label="SR Video", sublabel="input sequence",
          bold=True, fontsize=FONT_TITLE, lw=2.0)

sm_defs = [
    (ROW_Y[0], C_A, "A — Appearance", "CLIP-IQA mean − λ·std"),
    (ROW_Y[1], C_T, "T — Temporal", "multi-k tOF, k∈{1, 5, 10, 30, 60, 120}"),
    (ROW_Y[2], C_I, "I — Identity", "slow-fast ArcFace adapter"),
    (ROW_Y[3], C_D, "D — Colour Stability", "Lab histogram L1, pairs k∈{60, 120}"),
    (ROW_Y[4], C_DP, "D′ — Anchored Colour", "Lab-descriptor drift from 60-frame anchor"),
    (ROW_Y[5], C_DPP, "D″ — Semantic Trajectory", "CLIP-embedding drift from anchor"),
    (ROW_Y[6], C_E, "E — Colour Slope", "linear regression on Lab channel means"),
]
for yc, col, lbl, sub in sm_defs:
    fancy_box(X_SUB, yc, W_SUB, H_SM, col,
              label=lbl, sublabel=sub, fontsize=FONT_MAIN, lw=1.6)

rel_defs = [
    (ROW_Y[0], "std-floor · saturation"),
    (ROW_Y[1], "flow-mask coverage"),
    (ROW_Y[2], "face-rate · close-up bbox"),
    (ROW_Y[3], "sequence-length floor"),
    (ROW_Y[4], "anchor coverage"),
    (ROW_Y[5], "anchor coverage"),
    (ROW_Y[6], "R² floor"),
]
for yc, lbl in rel_defs:
    yr = yc + (H_SM - H_REL) / 2
    rel_box(X_REL, yr, W_REL, H_REL, lbl)

fancy_box(X_COMP, Y_COMP, W_COMP, H_COMP, C_COMP,
          label="Composition",
          sublabel="softmax(reliabilities / τ=0.2)\n→ log-mean of scores",
          bold=True, fontsize=FONT_MAIN, lw=2.0)

fancy_box(X_OUT, Y_OUT, W_OUT, H_OUT, C_OUT,
          label="LR-VCC score", sublabel="∈ [0, 1]",
          bold=True, fontsize=FONT_MAIN, lw=2.0, edgecolor=C_EDGE)

for yc, _, _, _ in sm_defs:
    arrow(X_IN + W_IN, Y_IN + H_IN / 2, X_SUB, yc + H_SM / 2,
          lw=1.6, color=C_EDGE)
for yc, _, _, _ in sm_defs:
    yr_mid = yc + H_SM / 2
    arrow(X_SUB + W_SUB, yr_mid, X_REL, yr_mid,
          lw=1.0, color=C_EDGE_THIN, arrowsize=8)
y_comp_centre = Y_COMP + H_COMP / 2
for yc, _, _, _ in sm_defs:
    yr_mid = yc + H_SM / 2
    arrow(X_REL + W_REL, yr_mid, X_COMP, y_comp_centre,
          lw=1.0, color=C_EDGE_THIN, arrowsize=8)
for yc, _, _, _ in sm_defs:
    arrow(X_SUB + W_SUB, yc + H_SM * 0.3, X_COMP, y_comp_centre,
          lw=1.4, color=C_EDGE, arrowsize=10)
arrow(X_COMP + W_COMP, y_comp_centre, X_OUT, Y_OUT + H_OUT / 2,
      lw=2.0, color=C_EDGE, arrowsize=12)

ax.text(0.4, 0.7,
        "Dark arrows: scores (along the composition's geometric mean).   "
        "Light arrows: reliabilities (along the composition's softmax weights).",
        ha="left", va="center", fontsize=FONT_SMALL - 0.5,
        color="#666666", style="italic")

OUT = ("/Users/ana/Desktop/Timur/thesis_ve/zjuthesis/figure/"
       "fig4_lr_vcc_architecture_v5.png")
plt.tight_layout(pad=0.5)
plt.savefig(OUT, dpi=300, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close()
print(f"Saved: {OUT}")
