"""Severity-response summary grid: 4 artefact families x 3 key metrics.

Reads:
    results/synthetic_artefacts_eval/tof_tlp/<artefact>/<base>_sev*_tof_tlp.json
    results/lr_vcc/composite_artefacts_v3_slope_b200/<artefact>/<base>_sev*.json
Writes:
    proposal/figures/fig7_severity_summary_grid.png (overwrite)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


REPO = Path(__file__).resolve().parents[2]

ARTEFACTS = [
    ("color_drift",         "Colour drift"),
    ("chunk_boundary",      "Chunk-boundary jumps"),
    ("flicker",             "Periodic flicker"),
    ("identity_degradation","Identity degradation"),
]
BASES = ["7WHI2L_FDNg", "hhszUXL1Cu8"]
SEV_STR = ["0p02", "0p05", "0p10", "0p20", "0p40"]
SEVERITIES = [0.02, 0.05, 0.10, 0.20, 0.40]

BASE_COLORS = {"7WHI2L_FDNg": "#2196F3", "hhszUXL1Cu8": "#FF9800"}
BASE_LABELS = {"7WHI2L_FDNg": "video A (167 s)", "hhszUXL1Cu8": "video B (80 s)"}

# Lower-is-better for tOF; higher-is-better for LR-VCC.
METRICS = [
    ("tof_k1",  "Adjacent-frame temporal\n(tOF, k=1)",            False),
    ("tof_k120","Long-range temporal\n(tOF, k=120)",              False),
    ("lr_vcc",  "Composite metric\n(LR-VCC, v3 + slope)",         True),
]


def _load_tof(artefact, base, sev, k):
    p = REPO / f"results/synthetic_artefacts_eval/tof_tlp/{artefact}/{base}_sev{sev}_tof_tlp.json"
    if not p.is_file():
        return None
    d = json.load(open(p))
    return d.get("tof", {}).get(str(k))


def _load_lr_vcc(artefact, base, sev):
    p = REPO / f"results/lr_vcc/composite_artefacts_v3_slope_b200/{artefact}/{base}_sev{sev}.json"
    if not p.is_file():
        return None
    d = json.load(open(p))
    return d.get("lr_vcc")


def get_series(metric_key, artefact, base):
    vals = []
    for s in SEV_STR:
        if metric_key == "tof_k1":
            vals.append(_load_tof(artefact, base, s, 1))
        elif metric_key == "tof_k120":
            vals.append(_load_tof(artefact, base, s, 120))
        elif metric_key == "lr_vcc":
            vals.append(_load_lr_vcc(artefact, base, s))
    return vals


def is_monotonic_strict(values, higher_is_better,
                        rel_min_change=0.05, rel_step_tol=0.01,
                        max_step_violations=1):
    """A series counts as 'monotonic' under severity if:
      (a) the end-to-end change goes in the correct direction (worsen with severity),
      (b) the relative magnitude of the change is at least rel_min_change of the starting
          value (so genuinely flat series do not trivially pass), and
      (c) the number of intermediate step-reversals beyond a per-step tolerance does not
          exceed max_step_violations (so a single small noise bump in the middle does not
          disqualify an otherwise clean response)."""
    if any(v is None for v in values):
        return False
    start, end = values[0], values[-1]
    rel_change = abs(end - start) / max(abs(start), 1e-9)
    if rel_change < rel_min_change:
        return False
    step_tol = rel_step_tol * abs(start)
    violations = 0
    for i in range(len(values) - 1):
        if higher_is_better:
            if values[i + 1] > values[i] + step_tol:
                violations += 1
        else:
            if values[i + 1] < values[i] - step_tol:
                violations += 1
    if violations > max_step_violations:
        return False
    return (end < start) if higher_is_better else (end > start)


def verdict_for_cell(metric_key, artefact, higher_is_better):
    """Return ('PASS', 'PARTIAL', 'FAIL') based on monotonicity across both bases."""
    pass_count = 0
    for base in BASES:
        vals = get_series(metric_key, artefact, base)
        if any(v is None for v in vals):
            continue
        if is_monotonic_strict(vals, higher_is_better):
            pass_count += 1
    if pass_count == 2:
        return "PASS"
    elif pass_count == 1:
        return "PARTIAL"
    return "FAIL"


def main():
    fig, axes = plt.subplots(
        len(METRICS), len(ARTEFACTS),
        figsize=(15, 8.5), sharey="row",
    )

    VERDICT_BG    = {"PASS": "#e8f5e9", "PARTIAL": "#fff8e1", "FAIL": "#ffebee"}
    VERDICT_COLOR = {"PASS": "#2e7d32", "PARTIAL": "#ef6c00", "FAIL": "#c62828"}

    for col_idx, (art_key, art_title) in enumerate(ARTEFACTS):
        for row_idx, (metric_key, metric_label, hib) in enumerate(METRICS):
            ax = axes[row_idx, col_idx]
            verdict = verdict_for_cell(metric_key, art_key, hib)

            for base in BASES:
                vals = get_series(metric_key, art_key, base)
                if any(v is None for v in vals):
                    continue
                mono = is_monotonic_strict(vals, hib)
                ls = "-" if mono else "--"
                marker = "o" if mono else "s"
                ax.plot(
                    SEVERITIES, vals,
                    color=BASE_COLORS[base],
                    linestyle=ls, marker=marker,
                    markersize=5.5, linewidth=2.0,
                    label=BASE_LABELS[base], zorder=3,
                )

            ax.set_xscale("log")
            ax.set_xticks(SEVERITIES)
            ax.get_xaxis().set_major_formatter(ticker.FormatStrFormatter("%.2f"))
            ax.tick_params(axis="x", labelsize=7.5)
            ax.tick_params(axis="y", labelsize=7.5)
            ax.grid(True, alpha=0.3, linestyle=":")
            ax.set_facecolor(VERDICT_BG[verdict])

            if row_idx == 0:
                ax.set_title(art_title, fontsize=10.5, fontweight="bold", pad=6)

            if col_idx == 0:
                ax.set_ylabel(metric_label, fontsize=8.5, labelpad=4)

            ax.text(
                0.97, 0.05, verdict,
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=9.5, fontweight="bold", color=VERDICT_COLOR[verdict],
            )

            if row_idx == len(METRICS) - 1:
                ax.set_xlabel("Severity (higher = stronger artefact)", fontsize=8)

    handles = [
        plt.Line2D([0], [0], color=BASE_COLORS[b], marker="o", linewidth=2.0,
                   markersize=6, label=BASE_LABELS[b])
        for b in BASES
    ]
    mono_line = plt.Line2D([0], [0], color="gray", linestyle="-",  linewidth=2.0, label="Monotonic")
    nonmono   = plt.Line2D([0], [0], color="gray", linestyle="--", linewidth=2.0, label="Non-monotonic")
    fig.legend(
        handles=handles + [mono_line, nonmono],
        loc="lower center", ncol=4, fontsize=9,
        frameon=True, bbox_to_anchor=(0.5, -0.01),
    )

    fig.suptitle(
        "Severity response across four artefact families and three key metrics\n"
        "(green = monotonic on both videos; amber = monotonic on one; red = flat / non-monotonic on both)",
        fontsize=11.5, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])

    out = REPO / "proposal/figures/fig7_severity_summary_grid.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
