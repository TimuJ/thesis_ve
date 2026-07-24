"""Figures for the extreme retrieval-distance experiment (report §3.5).
Two PNGs, project-validated palette, dataviz mark specs (thin marks, >=8px
markers, recessive grid, direct labels, no chartjunk). Light surface to match
the existing reports/figures/pi/fig1-4."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator

BLUE, GREEN, AMBER, PURPLE = "#2a78d6", "#1baf7a", "#eda100", "#4a3aa7"
SURFACE = "#fcfcfb"
INK, MUTED, GRID = "#1c1c1c", "#666666", "#e6e6e3"
OUT = os.path.expanduser("~/Desktop/Timur/thesis_ve/reports/figures/sensitivity")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.linewidth": 1.0,
})


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, which="major", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


# ---- Fig A: temporal retrieval-distance dose-response (§3.5) ----
s   = np.array([1, 1.25, 1.5, 2, 3, 5, 10, 20, 50, 125, 250], float)
dist = 8 * s
d   = np.array([0, -0.12, -0.22, -0.25, -0.95, -1.34, -1.03, -0.31, -1.57, -1.33, -1.56])

fig, ax = plt.subplots(figsize=(7.2, 4.3))
style(ax)
# plateau band (mean of s>=5 points)
plateau = d[s >= 5].mean()
ax.axhspan(plateau - 0.18, plateau + 0.18, color=BLUE, alpha=0.06, zorder=0)
ax.axhline(plateau, color=BLUE, lw=1.0, ls=(0, (4, 3)), alpha=0.5, zorder=1)
ax.text(1700, plateau + 0.05, f"bounded plateau  {plateau:+.2f} dB",
        color=BLUE, fontsize=9.5, ha="right", va="bottom")
# trained-window edge (relative distance ~20 latents)
ax.axvline(20, color=MUTED, lw=1.0, ls=(0, (2, 2)), alpha=0.7, zorder=1)
ax.text(20, 0.13, "trained window\n(~20 latents)", color=MUTED, fontsize=8.5,
        ha="center", va="bottom")
ax.plot(dist, d, "-", color=BLUE, lw=2.0, zorder=3)
ax.plot(dist, d, "o", color=BLUE, ms=8, mfc=SURFACE, mew=2.0, zorder=4)
# annotate key points
ax.annotate("“frame 0 from frame 1000”", xy=(1000, -1.33), xytext=(300, -0.72),
            color=INK, fontsize=9.5, ha="center",
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=1.0))
ax.annotate("RoPE periodicity\n(phases re-align)", xy=(160, -0.31),
            xytext=(160, 0.02), color=MUTED, fontsize=8.5, ha="center",
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=1.0))
ax.set_xscale("log")
ax.set_xlim(6, 2600)
ax.set_ylim(-1.9, 0.42)
ax.xaxis.set_major_locator(FixedLocator([8, 20, 40, 80, 160, 400, 1000, 2000]))
ax.xaxis.set_minor_locator(NullLocator())
ax.set_xticklabels(["8", "20", "40", "80", "160", "400", "1000", "2000"])
ax.set_xlabel("effective query–key relative distance  (latent frames, log scale)",
              labelpad=6)
ax.set_ylabel("ΔPSNR vs stock  (dB)")
ax.set_title("Temporal position extrapolation degrades gracefully — no collapse",
             color=INK, fontsize=12.5, pad=10, loc="left")
fig.subplots_adjust(left=0.10, right=0.97, top=0.89, bottom=0.20)
fig.text(0.10, 0.035, "FlashVSR · DOVE-UDM10, 10 clips · content fixed, only the "
         "position label stretched (continuous PI hook)",
         color=MUTED, fontsize=8.5)
fig.savefig(os.path.join(OUT, "extreme_distance_temporal.png"), dpi=150)
plt.close(fig)

# ---- Fig B: time vs space at matched stretch (§3.1) ----
ss = np.array([0.5, 0.75, 1.25, 1.5, 2, 3])
series = {
    "time (t)":   (BLUE,  [-0.66, 0.01, -0.12, -0.22, -0.25, -0.95]),
    "height (h)": (GREEN, [-1.00, 0.02, -0.05, -0.07, -0.15, -2.55]),
    "width (w)":  (AMBER, [-0.83, -0.03, -0.02, -0.03, -0.07, -2.30]),
}
fig, ax = plt.subplots(figsize=(7.2, 4.3))
style(ax)
ax.axvline(1.0, color=MUTED, lw=1.0, ls=(0, (2, 2)), alpha=0.7, zorder=1)
ax.text(1.0, 0.28, "stock", color=MUTED, fontsize=8.5, ha="center", va="bottom")
for name, (c, ys) in series.items():
    ax.plot(ss, ys, "-", color=c, lw=2.0, zorder=3)
    ax.plot(ss, ys, "o", color=c, ms=8, mfc=SURFACE, mew=2.0, zorder=4)
    ax.text(3.06, ys[-1], name, color=c, fontsize=10, va="center", ha="left")
ax.annotate("space craters;\ntime stays gentle", xy=(3, -2.4), xytext=(1.5, -2.05),
            color=MUTED, fontsize=9, ha="center",
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=1.0))
ax.set_xscale("log")
ax.set_xlim(0.45, 5.4)
ax.set_ylim(-2.8, 0.55)
ax.xaxis.set_major_locator(FixedLocator([0.5, 0.75, 1.0, 1.5, 2, 3]))
ax.xaxis.set_minor_locator(NullLocator())
ax.set_xticklabels(["0.5", "0.75", "1.0", "1.5", "2", "3"])
ax.set_xlabel("stretch factor s  (relative-distance geometry, log scale)",
              labelpad=6)
ax.set_ylabel("ΔPSNR vs stock  (dB)")
ax.set_title("Space is far more sensitive than time to position geometry",
             color=INK, fontsize=12.5, pad=10, loc="left")
fig.subplots_adjust(left=0.10, right=0.90, top=0.89, bottom=0.20)
fig.text(0.10, 0.035, "FlashVSR · DOVE-UDM10, 10 clips · per-axis position "
         "stretch, content fixed", color=MUTED, fontsize=8.5)
fig.savefig(os.path.join(OUT, "time_vs_space_stretch.png"), dpi=150)
plt.close(fig)

print("wrote:", os.listdir(OUT))
