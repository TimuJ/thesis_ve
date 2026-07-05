"""D''-causal-check driver: one long video, one position condition, SINGLE PASS.

Arms of the experiment (vs the segmented benchmark outputs, arm A):
  B  --mode single    : stock-identical positions, but the run is one
                        continuous pass — positions grow past the 1024-row
                        table and are served by the verified extended table
                        (override = modulo 10^9: identity values, non-noop
                        path so extension engages).
  C  --mode mod336    : positions cycle p % 336 — magnitude stays bounded
                        near the table's low region, no content/cache seam,
                        rare position wraps.

Outputs mirror the benchmark convention: 1280x720 full-content PNGs
(reflect-pad 180->192 in, crop 24/24 out) + an mp4 (quality=6) for the D''
stage, plus a stats JSON.

Run from ~/repos/FlashVSR/examples/WanVSR in the `flashvsr` env:
  PYTHONPATH=/home/timur/thesis_ve CUDA_VISIBLE_DEVICES=<g> python -m \
    scripts.rope_probe.run_long_condition \
    --input ~/synthetic_data/synthetic/7WHI2L_FDNg.mp4 \
    --mode single --out ~/results/rope_probe/dpp_causal/single
"""
import argparse
import gc
import importlib.util
import json
import os
import sys
import time

sys.path.insert(0, os.getcwd())

import cv2
import numpy as np
import torch
from PIL import Image

from scripts.rope_probe.flashvsr_hook import default_table_builder, install_position_hook
from scripts.rope_probe.position_override import PositionOverride

PAD_LR = 6
CROP_HR = 24
TH, TW = 768, 1280

MODES = {
    "single": PositionOverride(modulo=10**9),   # identity positions, extended table
    "mod336": PositionOverride(modulo=336),
}


def load_long_infer():
    spec = importlib.util.spec_from_file_location(
        "infer_long", "infer_flashvsr_v1.1_tiny_long_video.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["infer_long"] = m
    spec.loader.exec_module(m)
    return m


def read_video_rgb(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames, fps


def prepare_cpu(lr_frames):
    """Full video -> (1,C,F,H,W) bf16 CPU tensor (tiny_long stages on CPU)."""
    n_real = len(lr_frames)
    k = (5 - n_real) % 8
    frames = lr_frames + [lr_frames[-1]] * (k + 4)
    ts = []
    for fr in frames:
        p = cv2.copyMakeBorder(fr, PAD_LR, PAD_LR, 0, 0, cv2.BORDER_REFLECT_101)
        up = cv2.resize(p, (TW, TH), interpolation=cv2.INTER_CUBIC)
        t = torch.from_numpy(up).to(torch.float32).permute(2, 0, 1) / 255.0 * 2.0 - 1.0
        ts.append(t.to(torch.bfloat16))
    vid = torch.stack(ts, 0).permute(1, 0, 2, 3).unsqueeze(0)
    return vid, vid.shape[2], n_real


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--mode", required=True, choices=sorted(MODES))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    vid_id = os.path.splitext(os.path.basename(args.input))[0]
    png_dir = os.path.join(args.out, vid_id)
    mp4_path = os.path.join(args.out, vid_id + ".mp4")
    if os.path.isfile(mp4_path):
        print(f"[{vid_id}] already done, skipping", flush=True)
        return
    os.makedirs(png_dir, exist_ok=True)

    infer = load_long_infer()
    pipe = infer.init_pipeline()
    dit = pipe.denoising_model()
    t_dim = dit.freqs[0].shape[1] * 2

    lr, fps = read_video_rgb(os.path.expanduser(args.input))
    LQ, F, n_real = prepare_cpu(lr)
    del lr
    gc.collect()
    print(f"[{vid_id}] {n_real} frames, F={F}, latents~{(F-1)//4}, mode={args.mode}",
          flush=True)

    restore = install_position_hook(dit, MODES[args.mode],
                                    default_table_builder(t_dim))
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    try:
        video = pipe(
            prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=1,
            seed=0, LQ_video=LQ, num_frames=F, height=TH, width=TW,
            is_full_block=False, if_buffer=True,
            topk_ratio=2.0 * 768 * 1280 / (TH * TW), kv_ratio=3.0,
            local_range=11, color_fix=True,
        )
    finally:
        restore()
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"[{vid_id}] inference {dt:.0f}s, peak {peak:.1f} GiB", flush=True)

    frames_out = infer.tensor2video(video)
    del video, LQ
    gc.collect(); torch.cuda.empty_cache()

    import imageio
    w = imageio.get_writer(mp4_path, fps=fps, quality=6)
    for i in range(n_real):
        fr = np.asarray(frames_out[i])[CROP_HR:-CROP_HR]
        Image.fromarray(fr).save(os.path.join(png_dir, f"{i:05d}.png"))
        w.append_data(fr)
    w.close()
    with open(os.path.join(args.out, vid_id + "_stats.json"), "w") as f:
        json.dump({"video": vid_id, "mode": args.mode, "frames": n_real,
                   "sec": round(dt, 1), "peak_vram_gib": round(peak, 2)}, f)
    print(f"[{vid_id}] DONE mode={args.mode}: {n_real} PNGs + {mp4_path}", flush=True)


if __name__ == "__main__":
    main()
