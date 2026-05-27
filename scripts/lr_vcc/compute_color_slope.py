"""Per-video color-slope sub-metric runner. CPU-only.

Reads videos from --videos_path, computes per-channel L*a*b* mean linear
regression score per video, dumps one JSON per video to --output_path.
"""
import argparse
import json
import os
import sys

import cv2

from scripts.lr_vcc.color_slope import color_slope_score_from_frames


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
    ap.add_argument("--beta", type=float, default=50.0)
    ap.add_argument("--r2_floor", type=float, default=0.15)
    args = ap.parse_args()

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
        out_file = os.path.join(args.output_path, base + "_color_slope.json")
        if os.path.isfile(out_file):
            print("[skip] " + out_file)
            continue
        print("=== " + vname + " ===")
        frames = _load_frames(vpath)
        if not frames:
            print("  no frames; skipping")
            continue
        out = color_slope_score_from_frames(frames, beta=args.beta, r2_floor=args.r2_floor)
        payload = {"video_path": vpath, "n_frames": len(frames), **out}
        with open(out_file, "w") as f:
            json.dump(payload, f, indent=2)
        d = out["details"]
        print("  score=" + format(out["score"], ".4f")
              + " reliability=" + format(out["reliability"], ".4f")
              + " max_abs_slope=" + format(d.get("max_abs_slope", 0.0), ".5f")
              + " max_r2=" + format(d.get("max_r2", 0.0), ".4f"))


if __name__ == "__main__":
    main()
