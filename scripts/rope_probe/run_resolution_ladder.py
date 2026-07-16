"""Resolution-extrapolation ladder + collapse-decomposition arms (YouHQ40).

Rung tokens (input side, native LQ is 270x270):
  "180"   center-crop 180        -> output content 720   (grid 48, trained extent)
  "270"   native full frame      -> output content 1080  (grid 72)
  "270u"  crop 202, upscale x1.33-> output content 1080  (grid 72; blur arm)
  "320"   full frame upscaled    -> output content 1280  (grid 80)
  "352"   full frame upscaled    -> output content 1408  (grid 88)
  "360"   full frame upscaled    -> output content 1440  (grid 96, collapse rung)

Conditions: stock | pi (spatial positions compressed to the trained extent)
| pinned (stock positions, attention-sparsity topk pinned via --pin_topk —
the adaptive-sparsity decomposition arm).

Outputs are center-cropped to the content area (reflect-pad bands removed)
before saving, and each condition JSON records `score_mode` + a per-rung GT
reference dir so score_ladder.py compares like with like.

Run from ~/repos/FlashVSR/examples/WanVSR (flashvsr env):
  PYTHONPATH=/home/timur/thesis_ve CUDA_VISIBLE_DEVICES=0 python -m \
    scripts.rope_probe.run_resolution_ladder \
    --lq_dir ~/data/YouHQ40/LQ/000 --gt_dir ~/data/YouHQ40/GT/000 \
    --rungs 270u,320,352,360 --conds stock --out .../000
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
UP_FACTOR = 4.0 / 3.0


def parse_rung(token):
    """-> (input_size, crop_size). crop_size = true-content FOV in LR px."""
    if token.endswith("u"):
        size = int(token[:-1])
        return size, int(round(size / UP_FACTOR))
    return int(token), int(token)


def _read_frames(d, n=None):
    ps = sorted(glob.glob(os.path.join(d, "*.png")))
    if n:
        ps = ps[:n]
    return [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in ps]


def _center_crop(img, size):
    H, W = img.shape[:2]
    t, l = (H - size) // 2, (W - size) // 2
    return img[t:t + size, l:l + size]


def _prep_input(lq_frames, in_size, crop_size):
    native = lq_frames[0].shape[0]
    out = []
    for f in lq_frames:
        g = _center_crop(f, min(crop_size, native))
        if g.shape[0] != in_size:
            g = cv2.resize(g, (in_size, in_size), interpolation=cv2.INTER_CUBIC)
        out.append(g)
    return out


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
    ap.add_argument("--conds", default="stock,pi")
    ap.add_argument("--pin_topk", type=float, default=None,
                    help="topk_ratio for the 'pinned' condition")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    lq = _read_frames(os.path.expanduser(args.lq_dir))
    gt = _read_frames(os.path.expanduser(args.gt_dir))
    n = len(lq)
    conds = args.conds.split(",")

    infer = load_infer_module()
    pipe = infer.init_pipeline()
    dit = pipe.denoising_model()

    for token in args.rungs.split(","):
        in_size, crop_size = parse_rung(token)
        rung_in = _prep_input(lq, in_size, crop_size)
        LQ, F, (th, tw) = _to_lq_tensor(rung_in)
        grid = th // 16
        content_px = in_size * 4                      # output content w/o pad bands
        ref_px = min(crop_size * 4, gt[0].shape[0])   # matching GT field of view
        score_mode = "resize" if content_px != ref_px else "equal"

        ref_dir = os.path.join(args.out, f"rung{token}_GT")
        os.makedirs(ref_dir, exist_ok=True)
        for i, g in enumerate(gt):
            cv2.imwrite(os.path.join(ref_dir, f"{i:04d}.png"),
                        cv2.cvtColor(_center_crop(g, ref_px), cv2.COLOR_RGB2BGR))

        for cond in conds:
            cid = f"rung{token}_{cond}"
            fdir = os.path.join(args.out, cid)
            if os.path.isdir(fdir) and len(glob.glob(fdir + "/*.png")) == n:
                print(f"skip {cid}", flush=True)
                continue
            os.makedirs(fdir, exist_ok=True)
            restores, topk = [], None
            factor = TRAINED_EXTENT / grid
            if cond == "pi" and factor < 1.0:
                ov = PositionOverride(stretch=factor, continuous=True)
                for axis in (1, 2):
                    ax_dim = dit.freqs[axis].shape[1] * 2
                    restores.append(install_position_hook(
                        dit, ov, default_table_builder(ax_dim),
                        default_row_builder(ax_dim), axis=axis))
            elif cond == "pinned":
                assert args.pin_topk is not None, "--pin_topk required for 'pinned'"
                topk = args.pin_topk
            try:
                video = run_once(pipe, LQ, F, th, tw, topk_ratio=topk)
            finally:
                for r in reversed(restores):
                    r()
            frames = to_uint8_frames(video, n)
            del video
            torch.cuda.empty_cache()
            for i, fr in enumerate(frames):
                cv2.imwrite(os.path.join(fdir, f"{i:04d}.png"),
                            cv2.cvtColor(_center_crop(fr, content_px),
                                         cv2.COLOR_RGB2BGR))
            peak = torch.cuda.max_memory_allocated() / 1024**3
            write_condition_json(
                os.path.join(args.out, cid + ".json"),
                {"rung": token, "cond": cond, "grid": grid,
                 "in_size": in_size, "crop_size": crop_size,
                 "content_px": content_px, "ref_px": ref_px,
                 "score_mode": score_mode,
                 "pi_factor": factor if cond == "pi" else None,
                 "topk_pinned": topk, "frames": n,
                 "peak_vram_gib": round(peak, 2)},
                None, None)
            print(f"done {cid}: grid {grid}x{grid}, topk="
                  f"{topk if topk else 'formula'}, peak {peak:.1f} GiB", flush=True)
        del LQ
        torch.cuda.empty_cache()

    print("LADDER_OK", flush=True)


if __name__ == "__main__":
    main()
