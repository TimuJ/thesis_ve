"""Plot severity-response curves for 9 metric families x 2 artefacts.

Output: 3 PNG figures + 1 CSV.

Usage:
    python proposal/figures/plot_severity_response.py
"""
import csv
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

REPO = Path(__file__).resolve().parents[2]
EVAL = REPO / "results/synthetic_artefacts_eval"
LR_VCC_DIR = REPO / "results/lr_vcc/composite_artefacts"
OUT_FIG = REPO / "proposal/figures"
OUT_CSV = REPO / "results/lr_vcc/severity_response_table.csv"

ARTEFACTS = ["color_drift", "chunk_boundary"]
BASES = ["7WHI2L_FDNg", "hhszUXL1Cu8"]
SEVERITIES = [0.02, 0.05, 0.10, 0.20, 0.40]
SEV_STR = ["0p02", "0p05", "0p10", "0p20", "0p40"]

BASE_COLORS = {"7WHI2L_FDNg": "#2196F3", "hhszUXL1Cu8": "#FF9800"}
BASE_LABELS = {"7WHI2L_FDNg": "7WHI (167 s)", "hhszUXL1Cu8": "hhsz (80 s)"}

METRIC_LABELS = {
    "clip_iqa": "CLIP-IQA",
    "tof_k1": "tOF k=1",
    "tof_k120": "tOF k=120",
    "tlp_k1": "tLP k=1",
    "tlp_k120": "tLP k=120",
    "dover": "DOVER",
    "ewarp": "E*warp",
    "identity_fused": "Identity (fused)",
    "lr_vcc": "LR-VCC",
}

# For each metric: True if higher = better (used to set arrow direction expectation)
HIGHER_IS_BETTER = {
    "clip_iqa": True,
    "tof_k1": False,
    "tof_k120": False,
    "tlp_k1": False,
    "tlp_k120": False,
    "dover": True,
    "ewarp": False,
    "identity_fused": True,
    "lr_vcc": True,
}

METRICS = list(METRIC_LABELS.keys())


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_clip_iqa(artefact: str, b: str, s_str: str) -> float:
    p = EVAL / "clip_iqa" / artefact / f"{b}_sev{s_str}_clip_iqa.json"
    with open(p) as f:
        d = json.load(f)
    qs = d["clip_iqa"]
    return float(sum(qs) / len(qs))


def load_tof_tlp(artefact: str, b: str, s_str: str) -> dict:
    p = EVAL / "tof_tlp" / artefact / f"{b}_sev{s_str}_tof_tlp.json"
    with open(p) as f:
        d = json.load(f)
    return {
        "tof_k1": float(d["tof"]["1"]),
        "tof_k120": float(d["tof"]["120"]),
        "tlp_k1": float(d["tlp"]["1"]),
        "tlp_k120": float(d["tlp"]["120"]),
    }


def load_dover(artefact: str) -> dict:
    """Returns {basename: overall_score} for all videos in this artefact."""
    p = EVAL / "dover" / artefact / "results.csv"
    out = {}
    with open(p, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            path_key = next(k for k in r if "path" in k.lower())
            score_key = next(
                k for k in r if "overall" in k.lower() or "final" in k.lower()
            )
            basename = Path(r[path_key].strip()).stem
            out[basename] = float(r[score_key])
    return out


def load_ewarp(artefact: str) -> dict:
    """Returns {basename: warp_error} from metrics_ewarp.json per_sample dict."""
    p = EVAL / "ewarp" / artefact / "metrics_ewarp.json"
    with open(p) as f:
        d = json.load(f)
    return {k: float(v) for k, v in d["per_sample"].items()}


def load_identity(artefact: str) -> dict:
    """Returns {basename: fused_score} from the latest eval_results.json."""
    results_dir = EVAL / "identity" / artefact
    js_files = sorted(results_dir.glob("results_*_eval_results.json"))
    with open(js_files[-1]) as f:
        d = json.load(f)
    return {k: float(v["fused"]) for k, v in d["per_video"].items()}


def load_lr_vcc_val(artefact: str, b: str, s_str: str) -> float:
    p = LR_VCC_DIR / artefact / f"{b}_sev{s_str}.json"
    with open(p) as f:
        d = json.load(f)
    return float(d["lr_vcc"])


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_data() -> list:
    """Collect all metric values. Returns list of dicts (one per video)."""
    rows = []
    for art in ARTEFACTS:
        dover_map = load_dover(art)
        ewarp_map = load_ewarp(art)
        identity_map = load_identity(art)

        for b in BASES:
            for sev, s_str in zip(SEVERITIES, SEV_STR):
                key = f"{b}_sev{s_str}"
                clip_iqa = load_clip_iqa(art, b, s_str)
                tt = load_tof_tlp(art, b, s_str)
                dover_val = dover_map.get(key, float("nan"))
                ewarp_val = ewarp_map.get(key, float("nan"))
                identity_val = identity_map.get(key, float("nan"))
                lr_vcc_val = load_lr_vcc_val(art, b, s_str)

                rows.append(
                    {
                        "artefact": art,
                        "base": b,
                        "severity": sev,
                        "clip_iqa": clip_iqa,
                        "tof_k1": tt["tof_k1"],
                        "tof_k120": tt["tof_k120"],
                        "tlp_k1": tt["tlp_k1"],
                        "tlp_k120": tt["tlp_k120"],
                        "dover": dover_val,
                        "ewarp": ewarp_val,
                        "identity_fused": identity_val,
                        "lr_vcc": lr_vcc_val,
                    }
                )
    return rows


def write_csv(rows: list) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["artefact", "base", "severity"] + METRICS
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote CSV: {OUT_CSV}")


# ---------------------------------------------------------------------------
# Monotonicity check
# ---------------------------------------------------------------------------

def is_monotonic(values: list, higher_is_better: bool) -> bool:
    """Check if values move consistently in the expected direction as severity increases."""
    n = len(values)
    if n < 2:
        return True
    # Compute number of pairs in the expected direction
    direction = -1 if higher_is_better else 1  # as sev increases, quality falls
    pairs = [(values[i], values[i + 1]) for i in range(n - 1)]
    consistent = sum(1 for a, b in pairs if direction * (b - a) >= 0)
    return consistent >= (n - 1)  # allow at most 0 reversals


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def _get_series(rows: list, artefact: str, base: str, metric: str) -> list:
    """Extract metric values for a given artefact/base combo sorted by severity."""
    subset = [r for r in rows if r["artefact"] == artefact and r["base"] == base]
    subset.sort(key=lambda r: r["severity"])
    return [r[metric] for r in subset]


# ---------------------------------------------------------------------------
# Figure 5 & 6: per-artefact 3x3 severity-response grids
# ---------------------------------------------------------------------------

def plot_artefact_grid(rows: list, artefact: str, fig_path: Path) -> None:
    art_label = artefact.replace("_", " ").title()
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    axes_flat = axes.flatten()

    for idx, metric in enumerate(METRICS):
        ax = axes_flat[idx]
        hib = HIGHER_IS_BETTER[metric]
        any_monotonic = False

        for base in BASES:
            vals = _get_series(rows, artefact, base, metric)
            mono = is_monotonic(vals, hib)
            if mono:
                any_monotonic = True
            ls = "-" if mono else "--"
            marker = "o" if mono else "s"
            ax.plot(
                SEVERITIES,
                vals,
                color=BASE_COLORS[base],
                linestyle=ls,
                marker=marker,
                markersize=5,
                linewidth=1.8,
                label=BASE_LABELS[base],
                zorder=3,
            )

        ax.set_xscale("log")
        ax.set_xticks(SEVERITIES)
        ax.get_xaxis().set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        ax.tick_params(axis="x", labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.set_title(METRIC_LABELS[metric], fontsize=9, fontweight="bold")

        # Shade background: green if monotonic, red if not
        bg = "#e8f5e9" if any_monotonic else "#ffebee"
        ax.set_facecolor(bg)
        ax.grid(True, alpha=0.3, linestyle=":")

        direction_str = "higher=better" if hib else "lower=better"
        ax.set_xlabel(f"Severity ({direction_str})", fontsize=7)

    # Shared legend
    handles = [
        plt.Line2D([0], [0], color=BASE_COLORS[b], marker="o", linewidth=1.8,
                   markersize=5, label=BASE_LABELS[b])
        for b in BASES
    ]
    mono_line = plt.Line2D([0], [0], color="gray", linestyle="-", linewidth=1.8, label="Monotonic")
    nonmono_line = plt.Line2D([0], [0], color="gray", linestyle="--", linewidth=1.8, label="Non-monotonic")
    fig.legend(
        handles=handles + [mono_line, nonmono_line],
        loc="lower center",
        ncol=4,
        fontsize=8,
        frameon=True,
        bbox_to_anchor=(0.5, -0.01),
    )

    fig.suptitle(
        f"{art_label}: severity-response across 9 metrics",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_path}")


# ---------------------------------------------------------------------------
# Figure 7: headline summary 3x2 grid
# ---------------------------------------------------------------------------

def plot_summary_grid(rows: list, fig_path: Path) -> None:
    """3 rows x 2 cols: rows=metric groups, cols=artefact types."""
    # Row 0: adjacent-frame temporal (tOF k=1, E*warp) — pick tOF k=1
    # Row 1: long-range temporal (tOF k=120)
    # Row 2: composite (LR-VCC, DOVER) — pick LR-VCC
    SUMMARY_METRICS = ["tof_k1", "tof_k120", "lr_vcc"]
    SUMMARY_LABELS = ["Adjacent-frame temporal (tOF k=1)", "Long-range temporal (tOF k=120)", "Composite metric (LR-VCC)"]
    ART_TITLES = ["Color drift", "Chunk-boundary jumps"]

    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharey="row")

    for row_idx, (metric, row_label) in enumerate(zip(SUMMARY_METRICS, SUMMARY_LABELS)):
        hib = HIGHER_IS_BETTER[metric]
        for col_idx, artefact in enumerate(ARTEFACTS):
            ax = axes[row_idx, col_idx]
            any_monotonic = False

            for base in BASES:
                vals = _get_series(rows, artefact, base, metric)
                mono = is_monotonic(vals, hib)
                if mono:
                    any_monotonic = True
                ls = "-" if mono else "--"
                marker = "o" if mono else "s"
                ax.plot(
                    SEVERITIES,
                    vals,
                    color=BASE_COLORS[base],
                    linestyle=ls,
                    marker=marker,
                    markersize=6,
                    linewidth=2.2,
                    label=BASE_LABELS[base],
                    zorder=3,
                )

            ax.set_xscale("log")
            ax.set_xticks(SEVERITIES)
            ax.get_xaxis().set_major_formatter(ticker.FormatStrFormatter("%.2f"))
            ax.tick_params(axis="x", labelsize=8)
            ax.tick_params(axis="y", labelsize=8)
            ax.grid(True, alpha=0.35, linestyle=":")

            # Background shading
            verdict = "PASS" if any_monotonic else "FAIL"
            bg = "#e8f5e9" if any_monotonic else "#ffebee"
            ax.set_facecolor(bg)

            # Column headers
            if row_idx == 0:
                ax.set_title(ART_TITLES[col_idx], fontsize=11, fontweight="bold", pad=8)

            # Row labels on left column
            if col_idx == 0:
                ax.set_ylabel(row_label + "\n" + METRIC_LABELS[metric], fontsize=8.5, labelpad=6)

            # PASS/FAIL badge
            verdict_color = "#2e7d32" if verdict == "PASS" else "#c62828"
            ax.text(
                0.97, 0.05, verdict,
                transform=ax.transAxes,
                ha="right", va="bottom",
                fontsize=10, fontweight="bold",
                color=verdict_color,
            )

            # x-label only on bottom row
            if row_idx == 2:
                direction_str = "higher=better" if hib else "lower=better"
                ax.set_xlabel(f"Severity ({direction_str})", fontsize=8)

    # Legend
    handles = [
        plt.Line2D([0], [0], color=BASE_COLORS[b], marker="o", linewidth=2.2,
                   markersize=6, label=BASE_LABELS[b])
        for b in BASES
    ]
    mono_line = plt.Line2D([0], [0], color="gray", linestyle="-", linewidth=2.0, label="Monotonic")
    nonmono_line = plt.Line2D([0], [0], color="gray", linestyle="--", linewidth=2.0, label="Non-monotonic")
    fig.legend(
        handles=handles + [mono_line, nonmono_line],
        loc="lower center",
        ncol=4,
        fontsize=9,
        frameon=True,
        bbox_to_anchor=(0.5, -0.01),
    )

    fig.suptitle(
        "Severity-response summary: key metrics vs artefact type\n"
        "(green=monotonic/PASS, red=non-monotonic/FAIL)",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_path}")


# ---------------------------------------------------------------------------
# Verdict summary (for reporting)
# ---------------------------------------------------------------------------

def print_verdicts(rows: list) -> dict:
    """Print PASS/FAIL verdict per metric per artefact. Returns verdict dict."""
    print("\n=== Monotonicity verdicts (mean over 2 base videos) ===")
    verdicts = {}
    for artefact in ARTEFACTS:
        verdicts[artefact] = {}
        for metric in METRICS:
            passes = []
            for base in BASES:
                vals = _get_series(rows, artefact, base, metric)
                passes.append(is_monotonic(vals, HIGHER_IS_BETTER[metric]))
            # PASS if both base videos are monotonic
            v = "PASS" if all(passes) else ("PARTIAL" if any(passes) else "FAIL")
            verdicts[artefact][metric] = v
            print(f"  {artefact}/{metric}: {v}  (7WHI={passes[0]}, hhsz={passes[1]})")
    return verdicts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Collecting data...")
    rows = collect_data()

    print("Writing CSV...")
    write_csv(rows)

    print("Plotting Figure 5 (color drift)...")
    plot_artefact_grid(rows, "color_drift", OUT_FIG / "fig5_color_drift_severity.png")

    print("Plotting Figure 6 (chunk boundary)...")
    plot_artefact_grid(rows, "chunk_boundary", OUT_FIG / "fig6_chunk_boundary_severity.png")

    print("Plotting Figure 7 (summary grid)...")
    plot_summary_grid(rows, OUT_FIG / "fig7_severity_summary_grid.png")

    print_verdicts(rows)
    print("\nDone.")


if __name__ == "__main__":
    main()
