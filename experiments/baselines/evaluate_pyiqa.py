#!/usr/bin/env python3
"""Evaluate VSR baselines using IQA-PyTorch (pyiqa) for standardized metrics.

Computes metrics matching VSR paper conventions:
- PSNR on Y channel (YCbCr) with border crop
- SSIM on Y channel with border crop
- LPIPS on RGB

Usage:
    python experiments/baselines/evaluate_pyiqa.py \
        --results experiments/baselines/results/upscale_a_video/UDM10 \
        --gt experiments/baselines/data/UDM10/GT \
        --output experiments/baselines/results/upscale_a_video/upscale_a_video_UDM10_metrics.json \
        --crop_border 0
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def load_frames_as_tensors(clip_dir):
    """Load all image frames from a directory as (N, C, H, W) float tensors in [0, 1]."""
    clip_path = Path(clip_dir)
    files = sorted(f for f in clip_path.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS)
    if not files:
        raise FileNotFoundError(f"No image files found in {clip_dir}")
    frames = []
    for f in files:
        img = np.array(Image.open(f).convert("RGB")).astype(np.float32) / 255.0
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        frames.append(tensor)
    return frames


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate VSR baselines with pyiqa")
    parser.add_argument("--results", required=True, help="Path to model output frames")
    parser.add_argument("--gt", required=True, help="Path to ground truth frames")
    parser.add_argument("--output", required=True, help="Path to output JSON")
    parser.add_argument("--crop_border", type=int, default=0,
                        help="Border pixels to crop before evaluation (default: 0)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    import pyiqa

    gt_path = Path(args.gt)
    results_path = Path(args.results)
    device = torch.device(args.device)
    crop = args.crop_border

    # Create metrics
    psnr_metric = pyiqa.create_metric('psnr', device=device, test_y_channel=True)
    ssim_metric = pyiqa.create_metric('ssim', device=device, test_y_channel=True)
    lpips_metric = pyiqa.create_metric('lpips', device=device)

    clip_dirs = sorted(d for d in gt_path.iterdir() if d.is_dir())
    if not clip_dirs:
        print(f"Error: No clip subdirectories found in {gt_path}")
        sys.exit(1)

    per_clip = {}
    all_psnr, all_ssim, all_lpips = [], [], []

    for clip_dir in clip_dirs:
        clip_name = clip_dir.name
        results_clip = results_path / clip_name

        if not results_clip.exists():
            print(f"Warning: clip '{clip_name}' missing from results, skipping")
            continue

        print(f"Evaluating clip: {clip_name}")
        gt_frames = load_frames_as_tensors(str(clip_dir))
        pred_frames = load_frames_as_tensors(str(results_clip))

        # Truncate to shorter length
        n = min(len(gt_frames), len(pred_frames))
        if len(gt_frames) != len(pred_frames):
            print(f"  Warning: frame count mismatch ({len(pred_frames)} vs {len(gt_frames)} GT), using first {n}")
        gt_frames = gt_frames[:n]
        pred_frames = pred_frames[:n]

        clip_psnr, clip_ssim, clip_lpips = [], [], []

        for i, (pred_t, gt_t) in enumerate(zip(pred_frames, gt_frames)):
            # Crop pred to GT size if needed
            gh, gw = gt_t.shape[2], gt_t.shape[3]
            ph, pw = pred_t.shape[2], pred_t.shape[3]
            if (ph, pw) != (gh, gw):
                pred_t = pred_t[:, :, :gh, :gw]

            # Border crop
            if crop > 0:
                pred_t = pred_t[:, :, crop:-crop, crop:-crop]
                gt_t = gt_t[:, :, crop:-crop, crop:-crop]

            pred_t = pred_t.to(device)
            gt_t = gt_t.to(device)

            clip_psnr.append(psnr_metric(pred_t, gt_t).item())
            clip_ssim.append(ssim_metric(pred_t, gt_t).item())
            clip_lpips.append(lpips_metric(pred_t, gt_t).item())

        clip_metrics = {
            "PSNR_mean": float(np.mean(clip_psnr)),
            "SSIM_mean": float(np.mean(clip_ssim)),
            "LPIPS_mean": float(np.mean(clip_lpips)),
        }
        per_clip[clip_name] = clip_metrics
        all_psnr.append(clip_metrics["PSNR_mean"])
        all_ssim.append(clip_metrics["SSIM_mean"])
        all_lpips.append(clip_metrics["LPIPS_mean"])
        print(f"  PSNR: {clip_metrics['PSNR_mean']:.2f}, SSIM: {clip_metrics['SSIM_mean']:.4f}, LPIPS: {clip_metrics['LPIPS_mean']:.4f}")

    overall = {
        "PSNR_mean": float(np.mean(all_psnr)),
        "SSIM_mean": float(np.mean(all_ssim)),
        "LPIPS_mean": float(np.mean(all_lpips)),
    }

    output = {
        "evaluation": {
            "tool": "pyiqa",
            "psnr": "Y channel (YCbCr)",
            "ssim": "Y channel (YCbCr)",
            "lpips": "RGB (AlexNet)",
            "crop_border": crop,
        },
        "overall": overall,
        "per_clip": per_clip,
    }

    os.makedirs(Path(args.output).parent, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {args.output}")
    print(f"Overall — PSNR: {overall['PSNR_mean']:.2f}, SSIM: {overall['SSIM_mean']:.4f}, LPIPS: {overall['LPIPS_mean']:.4f}")


if __name__ == "__main__":
    main()
