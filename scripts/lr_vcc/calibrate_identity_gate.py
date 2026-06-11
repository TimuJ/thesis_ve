"""Print per-video clip-score dispersion across all existing identity JSONs.

Goal: pick _CLIP_DISPERSION_THRESHOLD separating well-tracked multi-face videos
(default good base: hhsz) from flappy single-face ones (default bad base: 7WHI).

Usage:
    python scripts/lr_vcc/calibrate_identity_gate.py
    python scripts/lr_vcc/calibrate_identity_gate.py --good_bases hhsz baseA --bad_bases 7WHI baseB

Flags:
    --good_bases   one or more base IDs known to be reliable  (default: hhsz)
    --bad_bases    one or more base IDs known to be pathological (default: 7WHI)

Any base present in the data but absent from both sets triggers a WARNING so it
can never be silently ignored when running on new data.
"""
import argparse
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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--good_bases", nargs="+", default=["hhsz"],
                        help="Base IDs considered reliable (default: hhsz)")
    parser.add_argument("--bad_bases", nargs="+", default=["7WHI"],
                        help="Base IDs considered pathological (default: 7WHI)")
    args = parser.parse_args()

    good_set = set(args.good_bases)
    bad_set = set(args.bad_bases)

    if not EVAL_DIR.exists():
        sys.exit(f"ERROR: evaluation directory not found: {EVAL_DIR}")

    rows = []
    for artefact_dir in sorted(EVAL_DIR.iterdir()):
        path = newest_json(artefact_dir)
        if path is None:
            continue
        try:
            data = json.load(open(path))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"WARNING: could not load {path}: {exc}", file=sys.stderr)
            continue
        if "per_video" not in data:
            print(f"WARNING: skipping {path} — no 'per_video' key", file=sys.stderr)
            continue
        per_video = data["per_video"]
        for vid, pv in per_video.items():
            disp = clip_score_dispersion(pv)
            if disp is None:
                continue
            # vid.split("_")[0] assumes <baseid>_<suffix> naming convention
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

    # Pool dispersions across all good bases and all bad bases for threshold suggestion
    good_present = [b for b in good_set if b in by_base]
    bad_present = [b for b in bad_set if b in by_base]
    if good_present and bad_present:
        good_pool = sorted(d for b in good_present for d in by_base[b])
        bad_pool = sorted(d for b in bad_present for d in by_base[b])
        hi_ok = good_pool[int(0.9 * (len(good_pool) - 1))]
        lo_bad = bad_pool[int(0.1 * (len(bad_pool) - 1))]
        good_label = "+".join(sorted(good_present))
        bad_label = "+".join(sorted(bad_present))
        print(f"\nsuggested threshold (midpoint {good_label}-p90={hi_ok:.3f}, {bad_label}-p10={lo_bad:.3f}): {(hi_ok + lo_bad) / 2:.3f}")

    # Warn about bases in the data that fall outside both partition sets
    all_data_bases = set(by_base.keys())
    excluded = all_data_bases - good_set - bad_set
    if excluded:
        print(f"WARNING: bases excluded from suggestion: {', '.join(sorted(excluded))}")


if __name__ == "__main__":
    main()
