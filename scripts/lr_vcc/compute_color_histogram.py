"""Per-video color-histogram sub-metric runner. CPU-only.

Reads videos from --videos_path, computes Lab-histogram temporal-stability
score per video, dumps one JSON per video to --output_path.
"""
import argparse
import json
import os
import sys

import cv2

from scripts.lr_vcc.color_histogram import color_histogram_score_from_frames


def _load_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos_path", required=True)
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--k_values", default="60,120")
    ap.add_argument("--max_pairs", type=int, default=200)
    ap.add_argument("--n_bins", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=5.0)
    args = ap.parse_args()

    k_values = [int(k) for k in args.k_values.split(",")]
    os.makedirs(args.output_path, exist_ok=True)

    video_files = sorted(
        f for f in os.listdir(args.videos_path)
        if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
    )
    if not video_files:
        sys.exit("No videos in " + args.videos_path)

    for vname in video_files:
        vpath = os.path.join(args.videos_path, vname)
        base = os.path.splitext(vname)[0]
        out_file = os.path.join(args.output_path, base + "_color_hist.json")
        if os.path.isfile(out_file):
            print("[skip] " + out_file)
            continue
        print("=== " + vname + " ===")
        frames = _load_frames(vpath)
        if not frames:
            print("  no frames; skipping")
            continue
        out = color_histogram_score_from_frames(
            frames, k_values=k_values, max_pairs=args.max_pairs,
            n_bins=args.n_bins, alpha=args.alpha,
        )
        payload = {"video_path": vpath, "n_frames": len(frames),
                   "k_values": k_values,
                   "mean_hist_dist": out["details"].get("mean_l1_dist"),
                   **out}
        with open(out_file, "w") as f:
            json.dump(payload, f, indent=2)
        print("  score=" + format(out["score"], ".4f")
              + " reliability=" + format(out["reliability"], ".4f")
              + " mean_dist=" + format(out["details"]["mean_l1_dist"], ".4f"))


if __name__ == "__main__":
    main()
