#!/usr/bin/env python3
"""Shared evaluation script for baseline VSR models.

Computes PSNR, SSIM, and optionally LPIPS between model output frames
and ground truth frames. Outputs JSON with per-clip and overall metrics.

Usage:
    python experiments/baselines/evaluate.py \
        --results experiments/baselines/results/upscale_a_video/UDM10 \
        --gt experiments/baselines/data/UDM10/GT \
        --output experiments/baselines/results/upscale_a_video/UDM10_metrics.json
"""
import argparse
import json
import sys
import os
from pathlib import Path

import numpy as np
from PIL import Image

# Add repo root to path so we can import src/
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.metrics import evaluate_sequence


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def load_frames(clip_dir: str) -> list[np.ndarray]:
    """Load all image frames from a directory, sorted by filename."""
    clip_path = Path(clip_dir)
    files = sorted(
        f for f in clip_path.iterdir()
        if f.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not files:
        raise FileNotFoundError(f"No image files found in {clip_dir}")
    return [np.array(Image.open(f).convert("RGB")) for f in files]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate VSR baseline results")
    parser.add_argument("--results", required=True, help="Path to model output frames")
    parser.add_argument("--gt", required=True, help="Path to ground truth frames")
    parser.add_argument("--output", required=True, help="Path to output JSON")
    parser.add_argument("--no-lpips", action="store_true", help="Skip LPIPS computation")
    args = parser.parse_args(argv)

    gt_path = Path(args.gt)
    results_path = Path(args.results)
    compute_lpips = not args.no_lpips

    # Find clips — subdirectories in GT
    clip_dirs = sorted(d for d in gt_path.iterdir() if d.is_dir())
    if not clip_dirs:
        print(f"Error: No clip subdirectories found in {gt_path}")
        sys.exit(1)

    per_clip = {}
    psnr_means, ssim_means, lpips_means = [], [], []

    for clip_dir in clip_dirs:
        clip_name = clip_dir.name
        results_clip = results_path / clip_name

        if not results_clip.exists():
            print(f"Warning: clip '{clip_name}' missing from results, skipping")
            continue

        print(f"Evaluating clip: {clip_name}")
        gt_frames = load_frames(str(clip_dir))
        pred_frames = load_frames(str(results_clip))

        # Truncate to shorter length if mismatch
        n = min(len(gt_frames), len(pred_frames))
        if len(gt_frames) != len(pred_frames):
            print(f"  Warning: frame count mismatch ({len(pred_frames)} vs {len(gt_frames)} GT), using first {n}")
        gt_frames = gt_frames[:n]
        pred_frames = pred_frames[:n]

        result = evaluate_sequence(pred_frames, gt_frames)
        clip_metrics = {
            "PSNR_mean": result["PSNR_mean"],
            "SSIM_mean": result["SSIM_mean"],
        }

        if compute_lpips:
            try:
                import lpips
                import torch
                loss_fn = lpips.LPIPS(net="alex")
                lpips_scores = []
                for pred, gt in zip(pred_frames, gt_frames):
                    pred_t = torch.from_numpy(pred).permute(2, 0, 1).float().unsqueeze(0) / 127.5 - 1.0
                    gt_t = torch.from_numpy(gt).permute(2, 0, 1).float().unsqueeze(0) / 127.5 - 1.0
                    lpips_scores.append(loss_fn(pred_t, gt_t).item())
                lpips_mean = float(np.mean(lpips_scores))
                clip_metrics["LPIPS_mean"] = lpips_mean
                lpips_means.append(lpips_mean)
            except ImportError:
                print("  Warning: lpips package not available, skipping LPIPS")
                compute_lpips = False

        per_clip[clip_name] = clip_metrics
        psnr_means.append(result["PSNR_mean"])
        ssim_means.append(result["SSIM_mean"])

    overall = {
        "PSNR_mean": float(np.mean(psnr_means)) if psnr_means else 0.0,
        "SSIM_mean": float(np.mean(ssim_means)) if ssim_means else 0.0,
    }
    if lpips_means:
        overall["LPIPS_mean"] = float(np.mean(lpips_means))

    output = {
        "overall": overall,
        "per_clip": per_clip,
    }

    # Write JSON
    os.makedirs(Path(args.output).parent, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {args.output}")
    print(f"Overall — PSNR: {overall['PSNR_mean']:.2f}, SSIM: {overall['SSIM_mean']:.4f}", end="")
    if "LPIPS_mean" in overall:
        print(f", LPIPS: {overall['LPIPS_mean']:.4f}")
    else:
        print()


if __name__ == "__main__":
    main()
