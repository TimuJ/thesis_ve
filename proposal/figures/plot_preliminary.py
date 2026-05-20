"""Generate figures for the LR-VCC proposal preliminary-work section.

Reads cached per-frame anatomy traces and tOF JSONs.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # non-interactive backend; safe on headless servers and M1 Macs
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

REPO = Path(__file__).resolve().parents[2]

ANATOMY_DIR = REPO / "results" / "vbench2_anatomy"
TOF_DIR = REPO / "results" / "long_range_temporal"


# ---------------------------------------------------------------------------
# Figure 1 — per-frame abnormal-rate histograms (2×2 grid)
# ---------------------------------------------------------------------------

def fig1_abnormal_histograms():
    """2×2 grid: rows = video (KZ, hhsz), columns = method (MGLD, UAV)."""

    videos = [
        ("KZ8p6b1zJ9U",  "diagnostic_KZ8p6b1zJ9U"),
        ("hhszUXL1Cu8",   "diagnostic_hhszUXL1Cu8"),
    ]
    methods = ["mgld", "uav"]
    method_labels = {"mgld": "MGLD", "uav": "UAV"}
    video_display = {"KZ8p6b1zJ9U": "KZ8p6b1zJ9U", "hhszUXL1Cu8": "hhszUXL1Cu8"}

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    fig.suptitle("Per-frame abnormal-rate distribution", fontsize=13, fontweight="bold")

    for row, (vid_id, diag_dir) in enumerate(videos):
        for col, method in enumerate(methods):
            ax = axes[row][col]

            json_path = ANATOMY_DIR / diag_dir / f"{method}_{vid_id}_per_frame.json"
            with open(json_path) as fh:
                data = json.load(fh)

            frame_results = data["frame_results"]

            rates = []
            for fr in frame_results:
                pc = fr.get("person_count", 0)
                ac = fr.get("abnormal_count", 0)
                if pc > 0:
                    rates.append(ac / pc)

            rates = np.array(rates, dtype=float)

            color = "#2874A6" if method == "mgld" else "#E67E22"
            ax.hist(rates, bins=20, range=(0.0, 1.0), color=color, edgecolor="white",
                    linewidth=0.4, alpha=0.85)

            median_rate = float(np.median(rates)) if len(rates) > 0 else 0.0
            pct_high = 100.0 * float(np.mean(rates >= 0.9)) if len(rates) > 0 else 0.0

            ax.set_title(
                f"{video_display[vid_id]} — {method_labels[method]}\n"
                f"n={len(rates)} frames  |  median={median_rate:.2f}  |  ≥0.9: {pct_high:.1f}%",
                fontsize=9,
            )
            ax.set_xlabel("abnormal rate", fontsize=8)
            ax.set_ylabel("frame count", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.set_xlim(0, 1)
            ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    plt.savefig(
        REPO / "proposal" / "figures" / "fig1_kz_vs_hhsz_abnormal_histograms.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print("fig1 saved.")


# ---------------------------------------------------------------------------
# Figure 2 — hand bbox p50 vs MGLD-minus-UAV anatomy gap
# ---------------------------------------------------------------------------

def fig2_handbbox_vs_gap():
    """Scatter: hand bbox p50 (x) vs MGLD−UAV anatomy gap (y), one point per video."""

    # Values from docs/notes/2026-05-13-kz-regime-shift-trigger.md cross-video table
    videos = ["7WHI", "BrRLK", "KZ", "hhsz", "mJog"]
    hand_p50 = [0.5, 7.0, 18.0, 1.0, 3.0]   # % of frame area
    gap      = [+0.097, +0.085, -0.291, +0.047, +0.036]  # MGLD anatomy − UAV anatomy

    colors = ["#2874A6" if g > 0 else "#C0392B" for g in gap]

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)

    for x, y, label, c in zip(hand_p50, gap, videos, colors):
        ax.scatter(x, y, color=c, s=90, zorder=3)
        # Offset labels to avoid overlap
        x_offset = 0.4 if x < 15 else -0.4
        ha = "left" if x < 15 else "right"
        ax.annotate(label, (x, y), xytext=(x + x_offset, y + 0.010),
                    fontsize=8, ha=ha, color="#333333")

    ax.axhline(0, color="black", linewidth=0.9, linestyle="--", label="y = 0 (tie)")

    ax.set_xlabel("Hand bbox p50 (% of frame area)", fontsize=10)
    ax.set_ylabel("MGLD − UAV anatomy score", fontsize=10)
    ax.set_title(
        "Hand bbox size correlates with MGLD-vs-UAV anatomy gap",
        fontsize=11, fontweight="bold",
    )
    ax.set_xlim(-1, 21)
    ax.set_ylim(-0.38, 0.16)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8, loc="upper right")

    # Annotate quadrant interpretation
    ax.text(16, 0.06, "KZ close-up:\nMGLD penalised", fontsize=7,
            color="#C0392B", ha="center", style="italic")
    ax.text(3.5, -0.30, "UAV wins anatomy", fontsize=7,
            color="#C0392B", ha="center", style="italic")

    plt.savefig(
        REPO / "proposal" / "figures" / "fig2_handbbox_vs_anatomy_gap.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print("fig2 saved.")


# ---------------------------------------------------------------------------
# Figure 3 — tOF per-k curves, MGLD vs UAV, mean across 5 videos
# ---------------------------------------------------------------------------

def fig3_tof_curves():
    """Mean tOF vs k for MGLD and UAV with crossover band."""

    k_vals = [1, 5, 10, 30, 60, 120]

    # Mean across 5 videos from docs/notes/2026-05-14-tof-tlp-long-range-results.md
    mgld_tof = [0.0216, 0.0406, 0.0500, 0.0804, 0.1110, 0.1441]
    uav_tof  = [0.0177, 0.0424, 0.0618, 0.0922, 0.1314, 0.1682]

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    # Crossover band k=5-10
    ax.axvspan(5, 10, alpha=0.12, color="#F39C12", label="crossover region (k=5–10)")

    ax.plot(k_vals, mgld_tof, marker="o", color="#2874A6", linewidth=1.8,
            markersize=6, label="MGLD-VSR")
    ax.plot(k_vals, uav_tof,  marker="s", color="#E67E22", linewidth=1.8,
            markersize=6, label="Upscale-A-Video (UAV)")

    # Annotate who wins at endpoints
    ax.annotate("UAV wins\n(smoother)", xy=(1, uav_tof[0]), xytext=(1.3, 0.008),
                fontsize=7.5, color="#E67E22",
                arrowprops=dict(arrowstyle="-", color="#E67E22", lw=0.7))
    ax.annotate("MGLD wins\n(long-range)", xy=(120, mgld_tof[-1]), xytext=(55, 0.155),
                fontsize=7.5, color="#2874A6",
                arrowprops=dict(arrowstyle="-", color="#2874A6", lw=0.7))

    ax.set_xscale("log")
    ax.set_xticks(k_vals)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel("Frame gap k (log scale)", fontsize=10)
    ax.set_ylabel("Mean tOF  (lower = better)", fontsize=10)
    ax.set_title(
        "Long-range tOF — UAV wins adjacent frames, MGLD wins long-range",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=9, loc="upper left")
    ax.tick_params(labelsize=8)
    ax.set_ylim(0, 0.20)

    plt.savefig(
        REPO / "proposal" / "figures" / "fig3_tof_per_k_curves.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print("fig3 saved.")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    fig1_abnormal_histograms()
    fig2_handbbox_vs_gap()
    fig3_tof_curves()
    print("done")
