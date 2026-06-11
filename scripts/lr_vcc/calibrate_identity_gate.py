"""Print per-video clip-score dispersion across all existing identity JSONs.

Goal: pick _CLIP_DISPERSION_THRESHOLD separating well-tracked multi-face videos
(hhsz: should stay reliable) from flappy single-face ones (7WHI: should abstain).

Usage: python scripts/lr_vcc/calibrate_identity_gate.py
"""
import glob
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.lr_vcc.identity import clip_score_dispersion

EVAL_DIR = REPO / "results" / "synthetic_artefacts_eval" / "identity"


def newest_json(artefact_dir):
    files = sorted(glob.glob(str(artefact_dir / "*.json")), key=os.path.getmtime)
    return files[-1] if files else None


def main():
    rows = []
    for artefact_dir in sorted(EVAL_DIR.iterdir()):
        path = newest_json(artefact_dir)
        if path is None:
            continue
        per_video = json.load(open(path))["per_video"]
        for vid, pv in per_video.items():
            disp = clip_score_dispersion(pv)
            if disp is None:
                continue
            base = "hhsz" if vid.startswith("hhsz") else ("7WHI" if vid.startswith("7WHI") else vid.split("_")[0])
            rows.append((base, artefact_dir.name, vid, disp))

    rows.sort(key=lambda r: r[3])
    print("| base | artefact | video | dispersion |")
    print("|---|---|---|---:|")
    for base, art, vid, disp in rows:
        print(f"| {base} | {art} | {vid} | {disp:.3f} |")

    by_base = {}
    for base, _, _, disp in rows:
        by_base.setdefault(base, []).append(disp)
    print()
    for base, ds in sorted(by_base.items()):
        ds.sort()
        print(f"{base}: n={len(ds)} min={ds[0]:.3f} median={ds[len(ds)//2]:.3f} max={ds[-1]:.3f}")
    if "hhsz" in by_base and "7WHI" in by_base:
        hi_ok = sorted(by_base["hhsz"])[int(0.9 * (len(by_base["hhsz"]) - 1))]
        lo_bad = sorted(by_base["7WHI"])[int(0.1 * (len(by_base["7WHI"]) - 1))]
        print(f"\nsuggested threshold (midpoint hhsz-p90={hi_ok:.3f}, 7WHI-p10={lo_bad:.3f}): {(hi_ok + lo_bad) / 2:.3f}")


if __name__ == "__main__":
    main()
