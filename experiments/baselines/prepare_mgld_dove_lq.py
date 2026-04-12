"""Resize DOVE LQ to 128x128 for MGLD-VSR, then resize output back to GT resolution.

MGLD-VSR expects 128x128 LQ input (Resize+CenterCrop to 512 -> 4x upscale -> 512x512).
But DOVE LQ is 318x180 (non-square). To get full-frame SR:
  1. Resize full LQ to 128x128 (squash aspect ratio)
  2. MGLD produces 512x512 output
  3. Resize 512x512 output back to GT resolution (1272x720)

Usage:
  # Step 1: Prepare LQ
  python prepare_mgld_dove_lq.py prepare --input data/UDM10/LQ --output data/UDM10_LQ_128

  # Step 2: Run MGLD inference (default --input_size 512)
  # ... produces 512x512 output in results/mgld_vsr/UDM10_dove_512/

  # Step 3: Resize output to GT resolution
  python prepare_mgld_dove_lq.py postprocess \
      --input results/mgld_vsr/UDM10_dove_512 \
      --gt data/UDM10/GT \
      --output results/mgld_vsr/UDM10_dove_fullres
"""
import os
import sys
import argparse
from pathlib import Path
from PIL import Image


def resize_lq(input_dir, output_dir, size=128):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    for clip_dir in sorted(input_path.iterdir()):
        if not clip_dir.is_dir():
            continue
        out_clip = output_path / clip_dir.name
        out_clip.mkdir(parents=True, exist_ok=True)
        for img_file in sorted(clip_dir.iterdir()):
            if img_file.suffix.lower() in (".png", ".jpg", ".jpeg"):
                img = Image.open(img_file).convert("RGB")
                img_resized = img.resize((size, size), Image.LANCZOS)
                img_resized.save(out_clip / img_file.name)
        n = len(list(out_clip.glob("*.png")))
        print(f"Resized {clip_dir.name}: {n} frames to {size}x{size}")


def resize_output(pred_dir, gt_dir, output_dir):
    pred_path = Path(pred_dir)
    gt_path = Path(gt_dir)
    output_path = Path(output_dir)
    for clip_dir in sorted(pred_path.iterdir()):
        if not clip_dir.is_dir():
            continue
        gt_clip = gt_path / clip_dir.name
        if not gt_clip.exists():
            continue
        gt_frame = next(gt_clip.glob("*.png"), None)
        if gt_frame is None:
            continue
        gt_img = Image.open(gt_frame)
        target_w, target_h = gt_img.size

        out_clip = output_path / clip_dir.name
        out_clip.mkdir(parents=True, exist_ok=True)
        for img_file in sorted(clip_dir.iterdir()):
            if img_file.suffix.lower() in (".png", ".jpg", ".jpeg"):
                img = Image.open(img_file).convert("RGB")
                img_resized = img.resize((target_w, target_h), Image.LANCZOS)
                img_resized.save(out_clip / img_file.name)
        print(f"Resized {clip_dir.name}: 512x512 -> {target_w}x{target_h}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "postprocess"])
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gt", default="")
    parser.add_argument("--size", type=int, default=128)
    args = parser.parse_args()

    if args.mode == "prepare":
        resize_lq(args.input, args.output, args.size)
    elif args.mode == "postprocess":
        if not args.gt:
            print("Error: --gt required for postprocess mode")
            sys.exit(1)
        resize_output(args.input, args.gt, args.output)
