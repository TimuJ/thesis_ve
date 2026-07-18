"""Generate a subset of the synthetic artefact battery (CLI wrapper around
generate_all.process_one) with per-clip multiprocessing.

Usage (repo root, any env with cv2+numpy):
    python -m scripts.synthetic_artefacts.gen_subset \
        --families background_drift color_drift --jobs 8
"""
import argparse
from multiprocessing import Pool
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.synthetic_artefacts.generate_all import (  # noqa: E402
    BASE_VIDEOS, SEVERITIES, SRC_DIR, OUT_DIR, process_one)


def _run(task):
    family, base, sev = task
    src = SRC_DIR / (base + ".mp4")
    out_name = base + "_sev" + format(sev, ".2f").replace(".", "p") + ".mp4"
    out = OUT_DIR / family / out_name
    if out.is_file() and out.stat().st_size > 0:
        return f"SKIP {family}/{out_name}"
    process_one(src, out, family, sev, base=base)
    return f"DONE {family}/{out_name}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="+", required=True)
    ap.add_argument("--bases", nargs="*", default=None)
    ap.add_argument("--sevs", nargs="*", type=float, default=None)
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()

    bases = args.bases or list(BASE_VIDEOS)
    sevs = args.sevs or list(SEVERITIES)
    tasks = [(f, b, s) for f in args.families for b in bases for s in sevs
             if (SRC_DIR / (b + ".mp4")).is_file()]
    print(f"{len(tasks)} clips over {args.jobs} workers")
    with Pool(args.jobs) as pool:
        for msg in pool.imap_unordered(_run, tasks):
            print(msg, flush=True)
    print("ALL_DONE")


if __name__ == "__main__":
    main()
