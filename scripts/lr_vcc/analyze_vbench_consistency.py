"""Turn VBench consistency outputs on artefact families into Δ-verdict rows.

Reads eval_long outputs laid out as
  <root>/<dimension>/<family>/results_*_eval_results.json
(each JSON: {dim: [overall, [per-clip records]]}; clip source video parsed
from 'split_clip/<video_id>/' in video_path, where video_id is
'<base>_sev0pXX'), computes per-artefact-clip means, then the same
Δ(sev 0.02 -> 0.40) verdict protocol as the LR-VCC matrix.

Usage:
  python -m scripts.lr_vcc.analyze_vbench_consistency \
      --results_root results/vbench_consistency_artefacts \
      --out reports/figures/vbench_sota_verdicts.md
"""
import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

from .build_verdict_matrix import verdict

_SPLIT_RE = re.compile(r"split_clip/(?P<vid>[^/]+)/")
_SEV_RE = re.compile(r"^(?P<base>.+)_sev(?P<sev>\dp\d+)$")


def per_video_scores(eval_json_path):
    payload = json.load(open(eval_json_path))
    candidates = [v for v in payload.values() if isinstance(v, list)]
    _overall, records = candidates[0]
    by_vid = defaultdict(list)
    for rec in records:
        m = _SPLIT_RE.search(rec.get("video_path", ""))
        if m:
            by_vid[m["vid"]].append(float(rec["video_results"]))
    return {vid: mean(scores) for vid, scores in by_vid.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root = Path(args.results_root)

    lines = ["# SOTA consistency dimensions — severity-response verdicts",
             "",
             "Same Δ(sev 0.02→0.40) protocol and thresholds as the LR-VCC "
             "verdict matrix. Positive Δ = the dimension REWARDS the "
             "corruption.", ""]
    for dim_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        dim = dim_dir.name
        for fam_dir in sorted(p for p in dim_dir.iterdir() if p.is_dir()):
            fam = fam_dir.name
            files = sorted(glob.glob(str(fam_dir / "**" /
                                         "results_*_eval_results.json"),
                                     recursive=True))
            if not files:
                lines.append(f"## {dim} / {fam}: no results yet")
                lines.append("")
                continue
            scores = per_video_scores(files[-1])
            by_base = defaultdict(dict)
            for vid, s in scores.items():
                m = _SEV_RE.match(vid)
                if m:
                    by_base[m["base"]][m["sev"]] = s
            lines.append(f"## {dim} / {fam}")
            lines.append("")
            lines.append("| base | sev 0.02 | sev 0.10 | sev 0.40 | Δ | verdict |")
            lines.append("|---|---|---|---|---|---|")
            n_inv = n_clean = 0
            for base in sorted(by_base):
                sv = by_base[base]
                if "0p02" not in sv or "0p40" not in sv:
                    lines.append(f"| {base} | — incomplete — | | | | |")
                    continue
                d = sv["0p40"] - sv["0p02"]
                v = verdict(d)
                n_inv += v == "INVERTED"
                n_clean += v in ("PASS", "WEAK")
                mid = f"{sv.get('0p10', float('nan')):.4f}"
                lines.append(f"| {base} | {sv['0p02']:.4f} | {mid} "
                             f"| {sv['0p40']:.4f} | {d:+.4f} | {v} |")
            lines.append("")
            lines.append(f"clean {n_clean}/5 · inverted {n_inv}/5")
            lines.append("")
    Path(args.out).write_text("\n".join(lines) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
