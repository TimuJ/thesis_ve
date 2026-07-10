"""Direct resolution-extrapolation ladder on YouHQ40 (square 270x270 LQ).

Rungs grow the model's spatial grid with pixel statistics held fixed where
possible (crop-based), plus one upsampled rung beyond native:

  rung 180 : LQ center-crop 180x180 -> out 720x720   (latent 45x45, in-regime)
  rung 270 : full LQ 270x270        -> out 1152x1152 (latent 72x72)
  rung 360 : LQ bicubic-up 360x360  -> out 1440x1440 (latent 90x90;
             input-blur confound — comparisons are within-rung)

Per rung, TWO conditions:
  stock : stock spatial positions (grid runs past the distillation extent)
  pi    : spatial PI — h and w positions compressed continuously by
          (TRAINED_EXTENT / grid) on BOTH axes (composed hooks), keeping
          spatial position geometry inside the trained extent.
If PI recovers the stock rung's quality loss, resolution-extrapolation
damage is attributable to RoPE spatial positions; if not, to capacity.

Also writes per-rung GT reference dirs (center-cropped to the rung's output
field of view) so score_conditions can run per rung unchanged; the 360 rung
scores against full GT with --resize_to_ref.

Run from ~/repos/FlashVSR/examples/WanVSR (flashvsr env):
  PYTHONPATH=/home/timur/thesis_ve CUDA_VISIBLE_DEVICES=0 python -m \
    scripts.rope_probe.run_resolution_ladder \
    --lq_dir ~/data/YouHQ40/LQ/000 --gt_dir ~/data/YouHQ40/GT/000 \
    --out ~/results/rope_probe/res_ladder/000
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.getcwd())

import cv2
import torch

from scripts.rope_probe.consistency_metrics import write_condition_json
from scripts.rope_probe.flashvsr_hook import (
    default_row_builder, default_table_builder, install_position_hook)
from scripts.rope_probe.flashvsr_runner import (
    load_infer_module, pad_amounts, run_once, to_uint8_frames)
from scripts.rope_probe.position_override import PositionOverride

TRAINED_EXTENT = 48   # FlashVSR distillation res 768x1408 -> latent H extent 48


def _read_frames(d, n=None):
    ps = sorted(glob.glob(os.path.join(d, "*.png")))
    if n:
        ps = ps[:n]
    return [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in ps]


def _center_crop(img, size):
    H, W = img.shape[:2]
    t, l = (H - size) // 2, (W - size) // 2
    return img[t:t + size, l:l + size]


def _prep_rung_input(lq_frames, rung):
    native = lq_frames[0].shape[0]
    if rung <= native:
        return [_center_crop(f, rung) for f in lq_frames]
    return [cv2.resize(f, (rung, rung), interpolation=cv2.INTER_CUBIC)
            for f in lq_frames]


def _to_lq_tensor(frames):
    (pt, pb, pl, pr), (TH, TW) = pad_amounts(*frames[0].shape[:2])
    n_real = len(frames)
    k = (5 - n_real) % 8
    frames = frames + [frames[-1]] * (k + 4)
    ts = []
    for fr in frames:
        p = cv2.copyMakeBorder(fr, pt, pb, pl, pr, cv2.BORDER_REFLECT_101)
        up = cv2.resize(p, (TW, TH), interpolation=cv2.INTER_CUBIC)
        t = torch.from_numpy(up).to(torch.float32).permute(2, 0, 1) / 255.0 * 2.0 - 1.0
        ts.append(t.to(torch.bfloat16))
    vid = torch.stack(ts, 0).permute(1, 0, 2, 3).unsqueeze(0)
    return vid.to("cuda"), vid.shape[2], (TH, TW)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lq_dir", required=True)
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rungs", default="180,270,360")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    lq = _read_frames(os.path.expanduser(args.lq_dir))
    gt = _read_frames(os.path.expanduser(args.gt_dir))
    n = len(lq)

    infer = load_infer_module()
    pipe = infer.init_pipeline()
    dit = pipe.denoising_model()

    for rung in [int(r) for r in args.rungs.split(",")]:
        rung_in = _prep_rung_input(lq, rung)
        LQ, F, (th, tw) = _to_lq_tensor(rung_in)
        grid = th // 16
        # per-rung GT reference (crop rungs only; upsampled rung scores vs
        # full GT with --resize_to_ref in the scorer)
        ref_dir = os.path.join(args.out, f"rung{rung}_GT")
        os.makedirs(ref_dir, exist_ok=True)
        gt_size = min(rung * 4, gt[0].shape[0])
        for i, g in enumerate(gt):
            cv2.imwrite(os.path.join(ref_dir, f"{i:04d}.png"),
                        cv2.cvtColor(_center_crop(g, gt_size), cv2.COLOR_RGB2BGR))

        for cond in ("stock", "pi"):
            cid = f"rung{rung}_{cond}"
            fdir = os.path.join(args.out, cid)
            if os.path.isdir(fdir) and len(glob.glob(fdir + "/*.png")) == n:
                print(f"skip {cid}", flush=True)
                continue
            os.makedirs(fdir, exist_ok=True)
            restores = []
            factor = TRAINED_EXTENT / grid
            if cond == "pi" and factor < 1.0:
                ov = PositionOverride(stretch=factor, continuous=True)
                for axis in (1, 2):
                    ax_dim = dit.freqs[axis].shape[1] * 2
                    restores.append(install_position_hook(
                        dit, ov, default_table_builder(ax_dim),
                        default_row_builder(ax_dim), axis=axis))
            try:
                video = run_once(pipe, LQ, F, th, tw)
            finally:
                for r in reversed(restores):
                    r()
            frames = to_uint8_frames(video, n)
            del video
            torch.cuda.empty_cache()
            for i, fr in enumerate(frames):
                cv2.imwrite(os.path.join(fdir, f"{i:04d}.png"),
                            cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
            peak = torch.cuda.max_memory_allocated() / 1024**3
            write_condition_json(
                os.path.join(args.out, cid + ".json"),
                {"rung": rung, "cond": cond, "grid": grid,
                 "pi_factor": factor if cond == "pi" else None,
                 "frames": n, "peak_vram_gib": round(peak, 2)},
                None, None)
            print(f"done {cid}: grid {grid}x{grid}, peak {peak:.1f} GiB", flush=True)
        del LQ
        torch.cuda.empty_cache()

    print("LADDER_OK", flush=True)


if __name__ == "__main__":
    main()
