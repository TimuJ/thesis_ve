"""Three-matrix comparison: original D vs D' (anchor-window) vs D'' (CLIP-trajectory).

D / D' / D'' all share the same {score, reliability, details} shape and live in:
- D     : results/lr_vcc/composite_artefacts_{v3_slope_b200,v4}/<art>/<vid>.json
          → sub_metrics.color_stability.score
- D'    : results/lr_vcc/color_hist_anchor/<art>/<vid>_color_hist_anchor.json
          → score recomputed from details.trajectory_mean_per_quarter (anchor L1)
- D''   : results/lr_vcc/clip_trajectory/<art>/<vid>_clip_trajectory.json
          → score recomputed from details.trajectory_mean_per_quarter (CLIP cos-dist)

Rescoring formula for D' and D'': `exp(-beta * |q4 - q1|)` where q1..q4 are mean
distances per video-quarter. `|...|` (not max(0, ...)) so videos whose natural
trajectory decreases over time (e.g. mJog) don't clamp to zero score.

Usage:
    python -m scripts.lr_vcc.compare_d_variants \
        --out reports/figures/d_variants_matrix.md \
        --beta_dprime 0.5 --beta_dprime2 3.0
"""
import argparse
import glob
import json
import os
from math import exp
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
BASES = ["hhszUXL1Cu8", "7WHI2L_FDNg", "KZ8p6b1zJ9U", "BrRLKMbBTYQ", "mJog8DlRk_4"]
ARTS = ["color_drift", "chunk_boundary", "flicker",
        "identity_degradation", "identity_drift", "background_drift",
        "flip_horizontal", "flip_transpose", "flip_periodic", "flip_elastic",
        "flip_channel_shuffle", "flip_invert"]


def verdict(delta):
    if delta is None:
        return "—"
    if delta <= -0.05:
        return "PASS"
    if delta <= -0.02:
        return "WEAK"
    if delta < +0.02:
        return "FLAT"
    return "INV"


def _abs_drift_score(path, beta):
    q = json.load(open(path))["details"]["trajectory_mean_per_quarter"]
    return exp(-beta * abs(q[3] - q[0]))


def _collect(pattern, ext, beta):
    out = {}
    for f in glob.glob(pattern):
        parts = f.split("/")
        art, name = parts[-2], parts[-1][:-len(ext)]
        if "_sev" not in name:
            continue
        base, sev = name.rsplit("_sev", 1)
        out.setdefault((art, base), {})[sev] = _abs_drift_score(f, beta)
    return out


def _collect_d(repo):
    """Original D scores from existing composite JSONs."""
    out = {}
    roots = [
        repo / "results" / "lr_vcc" / "composite_artefacts_v4",
        repo / "results" / "lr_vcc" / "composite_artefacts_v3_slope_b200",
    ]
    for root in roots:
        if not root.is_dir():
            continue
        for art_dir in root.iterdir():
            if not art_dir.is_dir():
                continue
            art_name = art_dir.name.removesuffix("_v2")
            for f in art_dir.glob("*sev*.json"):
                c = json.load(open(f))
                sm = c.get("sub_metrics", {})
                if "color_stability" not in sm:
                    continue
                vid = c["video"]
                if "_sev" not in vid:
                    continue
                base, sev = vid.rsplit("_sev", 1)
                # don't overwrite v4 with v3
                out.setdefault((art_name, base), {}).setdefault(
                    sev, sm["color_stability"]["score"])
    return out


def render_matrix(data, label):
    lines = [f"\n### {label}\n",
             "| artefact | " + " | ".join(b[:6] for b in BASES) + " | clean |",
             "|---|" + "---:|" * len(BASES) + "---:|"]
    total = passes = 0
    for a in ARTS:
        cells = []
        arow = 0
        for b in BASES:
            r = data.get((a, b), {})
            lo, hi = r.get("0p02"), r.get("0p40")
            if lo is None or hi is None:
                cells.append("—")
                continue
            d = hi - lo
            v = verdict(d)
            total += 1
            if v in ("PASS", "WEAK"):
                passes += 1
                arow += 1
            cells.append(f"{d:+.3f} {v}")
        lines.append(f"| {a} | " + " | ".join(cells) + f" | {arow}/5 |")
    lines.append(f"\n**{label}: {passes}/{total} PASS/WEAK**")
    return "\n".join(lines), passes, total


def best_of(*sources):
    keys = set().union(*(set(s.keys()) for s in sources))
    combined = {}
    for k in keys:
        sevs = set().union(*(s.get(k, {}).keys() for s in sources))
        for sev in sevs:
            vals = [s[k][sev] for s in sources if k in s and sev in s[k]]
            if vals:
                combined.setdefault(k, {})[sev] = min(vals)  # most negative composite wins
    return combined


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "reports" / "figures" / "d_variants_matrix.md"))
    ap.add_argument("--beta_dprime", type=float, default=0.5)
    ap.add_argument("--beta_dprime2", type=float, default=3.0)
    args = ap.parse_args()

    d = _collect_d(REPO)
    dp = _collect(str(REPO / "results/lr_vcc/color_hist_anchor/*/*_color_hist_anchor.json"),
                  "_color_hist_anchor.json", args.beta_dprime)
    dpp = _collect(str(REPO / "results/lr_vcc/clip_trajectory/*/*_clip_trajectory.json"),
                   "_clip_trajectory.json", args.beta_dprime2)

    sections = [
        "# Sub-metric D variants — three-matrix comparison\n",
        "Δ = score(sev 0.40) − score(sev 0.02). "
        "PASS ≤ −0.05, WEAK ≤ −0.02, FLAT < +0.02 ≤ INV.\n",
        f"D' rescored with β={args.beta_dprime}, "
        f"D'' rescored with β={args.beta_dprime2}, "
        "both use `exp(-β · |q4-q1|)` over per-quarter trajectory means.\n",
    ]

    for variant, label in [(d, "D (original sub_metric color_stability)"),
                           (dp, "D' anchor-window L1"),
                           (dpp, "D'' CLIP-trajectory"),
                           (best_of(dp, dpp), "Best-of(D', D'')")]:
        text, _, _ = render_matrix(variant, label)
        sections.append(text)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(sections) + "\n")
    print(out)


if __name__ == "__main__":
    main()
