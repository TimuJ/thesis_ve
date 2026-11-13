#!/usr/bin/env python3
"""Generate DOVE-style LQ video from HQ, using the vendored DOVE degradation.

Replicates DOVE's RealSRDataset.degrade (finetune/datasets/real_sr_dataset.py):
  first_stage  = [blur -> resize -> noise -> jpeg -> video_compression]
  second_stage = [blur -> resize -> noise -> jpeg -> shuffle{ video_compression
                   | [resize->(H/scale,W/scale), sinc_blur] }]
scale x4 (the final resize target is set to input H//scale, W//scale, so net
downscale is exactly x4 regardless of the intermediate random resizes). Params from
dove_degrade/degradation.yaml.

Operates on an in-memory frame list — a long clip is heavy (a 5601-frame 720p clip
~15 GB as uint8). Use --max_frames / --start for a segment; chunked long-video mode
is a separate driver (TBD). Run from repo root with the vsr-degrade env:
  conda run -n vsr-degrade python -m src.data.make_lq --input data/starter/bus-20.mp4 \
      --out_dir /tmp/lq_bus20 --max_frames 200 --seed 42
"""
import argparse
import os
import random
import sys
import time

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data.dove_degrade.degradation import (  # noqa: E402
    RandomBlur, RandomResize, RandomNoise, RandomJPEGCompression,
    RandomVideoCompression, DegradationsWithShuffle,
)


def build_stages(opt):
    d1, d2 = opt["degradation_1"], opt["degradation_2"]
    def op(cls, node):
        return cls(node["params"], node["keys"])
    first = [
        op(RandomBlur, d1["random_blur"]),
        op(RandomResize, d1["random_resize"]),
        op(RandomNoise, d1["random_noise"]),
        op(RandomJPEGCompression, d1["random_jpeg"]),
        op(RandomVideoCompression, d1["random_mpeg"]),
    ]
    dws = DegradationsWithShuffle(
        d2["degradation_with_shuffle"]["degradations"],
        d2["degradation_with_shuffle"]["keys"],
    )
    second = [
        op(RandomBlur, d2["random_blur"]),
        op(RandomResize, d2["random_resize"]),
        op(RandomNoise, d2["random_noise"]),
        op(RandomJPEGCompression, d2["random_jpeg"]),
        dws,
    ]
    return first, second, dws


def set_final_target(dws, H, W, scale):
    """Set the final-resize target to (H//scale, W//scale) on the RandomResize
    object inside the shuffle block (robust to the in-place shuffle)."""
    th, tw = H // scale, W // scale
    def walk(items):
        for o in items:
            if isinstance(o, (list, tuple)):
                walk(o)
            elif isinstance(o, RandomResize):
                o.params["target_size"] = (th, tw)
    walk(dws.degradations)
    return th, tw


def read_frames_rgb(mp4, start=0, max_frames=0):
    cap = cv2.VideoCapture(mp4)
    frames, i = [], 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if i >= start:
            frames.append(bgr[:, :, ::-1].astype(np.uint8))  # BGR -> RGB
        i += 1
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    return frames


def degrade_frames(frames, first, second):
    data = {"lqs": [f.copy() for f in frames]}
    for o in first:
        data = o(data)
    for o in second:
        data = o(data)
    return [np.clip(f, 0, 255).astype(np.uint8) for f in data["lqs"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="HQ mp4")
    ap.add_argument("--out_dir", required=True, help="LQ frame output dir (PNG)")
    ap.add_argument("--config", default="src/data/dove_degrade/degradation.yaml")
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--max_frames", type=int, default=0, help="0 = all (memory!)")
    ap.add_argument("--save_hq0", action="store_true", help="also dump first HQ frame")
    a = ap.parse_args()

    opt = yaml.safe_load(open(a.config))
    random.seed(a.seed)
    np.random.seed(a.seed)

    frames = read_frames_rgb(a.input, a.start, a.max_frames)
    assert frames, f"no frames read from {a.input}"
    H, W = frames[0].shape[:2]
    first, second, dws = build_stages(opt)
    th, tw = set_final_target(dws, H, W, a.scale)

    t0 = time.time()
    lq = degrade_frames(frames, first, second)
    dt = time.time() - t0

    os.makedirs(a.out_dir, exist_ok=True)
    for i, f in enumerate(lq):
        cv2.imwrite(os.path.join(a.out_dir, f"{i:08d}.png"), f[:, :, ::-1])  # RGB->BGR
    if a.save_hq0:
        cv2.imwrite(os.path.join(a.out_dir, "_hq0.png"), frames[0][:, :, ::-1])

    sizes = sorted({f.shape for f in lq})
    print(f"in {len(frames)} frames @ {W}x{H}  ->  LQ {len(lq)} frames @ {tw}x{th} "
          f"(target x{a.scale})  in {dt:.1f}s ({dt/len(frames)*1000:.0f} ms/frame)")
    print("LQ frame sizes:", sizes, "| expected one size:", (th, tw, 3))


if __name__ == "__main__":
    main()
