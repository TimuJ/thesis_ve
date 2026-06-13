"""Sub-metric D' — anchor-window colour-histogram divergence.

Diagnosis from the v4 verdict matrix: sub-metric D (pairwise temporal stability
of Lab histograms) systematically *rewards* convergence-type corruptions
because progressively static videos become genuinely more stable over time.
The supervisor independently confirmed: "degradation makes videos more stable."

This prototype replaces "are consecutive-frame histograms similar?" with
"are *current* histograms similar to *the first N frames*?" Drift away from
the baseline produces growing distance; convergence-to-static produces low
distance (which is the correct signal — but ONLY for content that was static
at the start, which the anchor captures).

Score: exp(-alpha * mean per-frame L1 distance to the anchor descriptor).
A clean video has frames similar to its own opening → high score.
background_drift cumulatively walks away from the opening → low score.

Recomputable purely from already-cached video files; no server time required.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.lr_vcc.color_histogram import _frame_to_lab_descriptor, _descriptor_distance


_DEFAULT_ANCHOR_LEN = 60   # ≈2 s at 30 fps
_DEFAULT_ALPHA = 5.0
_DEFAULT_BINS = 32


def anchor_window_descriptors(video_path: str, anchor_len: int = _DEFAULT_ANCHOR_LEN,
                              n_bins: int = _DEFAULT_BINS, stride: int = 1):
    """Yield (idx, descriptor) for each frame; cap idx-stride to keep cost bounded.

    Anchor descriptor = mean of the first `anchor_len` frame descriptors.
    Per-frame descriptor = `_frame_to_lab_descriptor` (histogram + per-channel
    normalized mean) so we get sub-bin sensitivity on small drifts.
    """
    cap = cv2.VideoCapture(video_path)
    descriptors = []
    idx = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            descriptors.append(_frame_to_lab_descriptor(fr, n_bins=n_bins))
        idx += 1
    cap.release()
    if len(descriptors) < anchor_len // stride + 1:
        return None, descriptors  # too short
    arr = np.stack(descriptors)
    anchor_count = max(1, anchor_len // stride)
    anchor = arr[:anchor_count].mean(axis=0)
    return anchor, arr


def anchor_score(video_path: str, anchor_len: int = _DEFAULT_ANCHOR_LEN,
                 alpha: float = _DEFAULT_ALPHA, n_bins: int = _DEFAULT_BINS,
                 stride: int = 1) -> dict:
    """Return {"score", "reliability", "details": {...}} for one video."""
    anchor, descs = anchor_window_descriptors(video_path, anchor_len, n_bins, stride)
    if anchor is None:
        return {"score": 0.0, "reliability": 0.0,
                "details": {"error": "video too short", "n_frames": len(descs)}}
    anchor_count = max(1, anchor_len // stride)
    post_anchor = descs[anchor_count:]
    if len(post_anchor) == 0:
        return {"score": 1.0, "reliability": 0.0,
                "details": {"error": "no frames after anchor"}}
    dists = np.array([_descriptor_distance(d, anchor, n_bins=n_bins) for d in post_anchor])
    mean_dist = float(dists.mean())
    score = float(np.exp(-alpha * mean_dist))
    score = max(0.0, min(1.0, score))
    return {
        "score": score,
        "reliability": 1.0,
        "details": {
            "anchor_len_frames": anchor_len,
            "anchor_count_sampled": anchor_count,
            "post_anchor_count": int(len(post_anchor)),
            "stride": stride,
            "n_bins": n_bins,
            "alpha": alpha,
            "mean_l1_dist_to_anchor": mean_dist,
            "max_l1_dist_to_anchor": float(dists.max()),
            "trajectory_mean_per_quarter": [
                float(np.mean(c)) for c in np.array_split(dists, 4)
            ],
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos_dir", required=True,
                    help="dir of source mp4s (uses already-generated artefact clips)")
    ap.add_argument("--output_path", required=True,
                    help="dir to write <basename>_color_hist_anchor.json files")
    ap.add_argument("--anchor_len", type=int, default=_DEFAULT_ANCHOR_LEN)
    ap.add_argument("--alpha", type=float, default=_DEFAULT_ALPHA)
    ap.add_argument("--stride", type=int, default=1,
                    help="frame stride; >1 to subsample for speed")
    ap.add_argument("--n_bins", type=int, default=_DEFAULT_BINS)
    args = ap.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    videos = sorted(f for f in os.listdir(args.videos_dir) if f.endswith(".mp4"))
    for vname in videos:
        base = vname[:-4]
        out_file = os.path.join(args.output_path, base + "_color_hist_anchor.json")
        if os.path.isfile(out_file):
            print(f"[skip] {out_file}")
            continue
        vpath = os.path.join(args.videos_dir, vname)
        print(f"=== {vname} ===")
        out = anchor_score(vpath, anchor_len=args.anchor_len,
                           alpha=args.alpha, n_bins=args.n_bins, stride=args.stride)
        payload = {"video_path": vpath, **out}
        with open(out_file, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"  score={out['score']:.4f} mean_dist={out['details'].get('mean_l1_dist_to_anchor', 0):.4f}")


if __name__ == "__main__":
    main()
