#!/usr/bin/env python3
"""Evaluate VSR baselines using no-reference IQA metrics (pyiqa).

For datasets without ground truth (e.g., VideoLQ).
Computes: NIQE, BRISQUE, MUSIQ, CLIPIQA.

Usage:
    python experiments/baselines/evaluate_pyiqa_nr.py \
        --results experiments/baselines/results/mgld_vsr/VideoLQ \
        --output experiments/baselines/results/mgld_vsr/mgld_vsr_VideoLQ_nr.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def main(argv=None):
    parser = argparse.ArgumentParser(description="No-reference IQA evaluation")
    parser.add_argument("--results", required=True, help="Path to model output frames")
    parser.add_argument("--output", required=True, help="Path to output JSON")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    import pyiqa

    results_path = Path(args.results)
    device = torch.device(args.device)

    # Create no-reference metrics
    metrics = {}
    metric_names = ["niqe", "brisque", "musiq", "clipiqa"]
    for name in metric_names:
        try:
            metrics[name] = pyiqa.create_metric(name, device=device)
            print(f"Loaded metric: {name}")
        except Exception as e:
            print(f"Warning: could not load {name}: {e}")

    if not metrics:
        print("Error: no metrics loaded")
        sys.exit(1)

    clip_dirs = sorted(d for d in results_path.iterdir() if d.is_dir())
    if not clip_dirs:
        print(f"Error: No clip subdirectories found in {results_path}")
        sys.exit(1)

    per_clip = {}
    all_scores = {name: [] for name in metrics}

    for clip_dir in clip_dirs:
        clip_name = clip_dir.name
        frames = sorted(f for f in clip_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS)
        if not frames:
            print(f"Warning: no frames in {clip_name}, skipping")
            continue

        print(f"Evaluating clip: {clip_name} ({len(frames)} frames)")
        clip_scores = {name: [] for name in metrics}

        for frame_path in frames:
            for name, metric_fn in metrics.items():
                try:
                    score = metric_fn(str(frame_path)).item()
                    clip_scores[name].append(score)
                except Exception as e:
                    pass  # skip frames that fail

        clip_metrics = {}
        for name in metrics:
            if clip_scores[name]:
                mean_val = float(np.mean(clip_scores[name]))
                clip_metrics[f"{name.upper()}_mean"] = mean_val
                all_scores[name].append(mean_val)

        per_clip[clip_name] = clip_metrics
        scores_str = ", ".join(f"{k}: {v:.4f}" for k, v in clip_metrics.items())
        print(f"  {scores_str}")

    overall = {}
    for name in metrics:
        if all_scores[name]:
            overall[f"{name.upper()}_mean"] = float(np.mean(all_scores[name]))

    output = {
        "evaluation": {
            "tool": "pyiqa",
            "type": "no-reference",
            "metrics": list(metrics.keys()),
        },
        "overall": overall,
        "per_clip": per_clip,
    }

    os.makedirs(Path(args.output).parent, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {args.output}")
    print("Overall —", ", ".join(f"{k}: {v:.4f}" for k, v in overall.items()))


if __name__ == "__main__":
    main()
