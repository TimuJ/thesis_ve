"""Per-frame CLIP-IQA dump for one or more videos in a directory.

Server-side; needs GPU and pyiqa. Run from the vsr conda env.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/lr_vcc/compute_clip_iqa.py \
        --videos_path /data/disk2/timur/results/mgld_synthetic_mp4 \
        --output_path /data/disk2/timur/results/lr_vcc/clip_iqa/mgld
"""
import argparse
import json
import os
import sys

import cv2
import torch
import pyiqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos_path", required=True)
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--frame_stride", type=int, default=1,
                    help="evaluate every Nth frame (default 1 = every frame)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_path, exist_ok=True)

    print("Loading CLIP-IQA...")
    model = pyiqa.create_metric("clipiqa", device=device)

    video_files = sorted(
        f for f in os.listdir(args.videos_path)
        if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
    )
    if not video_files:
        sys.exit("No videos in " + args.videos_path)

    for vname in video_files:
        vpath = os.path.join(args.videos_path, vname)
        base = os.path.splitext(vname)[0]
        out_file = os.path.join(args.output_path, base + "_clip_iqa.json")
        if os.path.isfile(out_file):
            print("[skip] " + out_file)
            continue

        print("\n=== " + vname + " ===")
        cap = cv2.VideoCapture(vpath)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
        qualities = []
        frame_idx = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if frame_idx % args.frame_stride == 0:
                rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                t = torch.from_numpy(rgb).to(device).permute(2, 0, 1).float() / 255.0
                t = t.unsqueeze(0)
                with torch.no_grad():
                    q = float(model(t).item())
                qualities.append(q)
            frame_idx += 1
            if frame_idx % 500 == 0:
                print("  frame " + str(frame_idx) + "/" + str(n_frames))
        cap.release()
        payload = {
            "video_path": vpath,
            "n_frames": n_frames,
            "fps": fps,
            "frame_stride": args.frame_stride,
            "clip_iqa": qualities,
        }
        with open(out_file, "w") as f:
            json.dump(payload, f)
        print("  wrote " + out_file + " (" + str(len(qualities)) + " quality values)")


if __name__ == "__main__":
    main()
