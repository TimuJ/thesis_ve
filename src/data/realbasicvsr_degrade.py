#!/usr/bin/env python3
"""Apply RealBasicVSR degradation pipeline to generate LQ frames from GT.

This is the standard degradation used by both Upscale-A-Video and MGLD-VSR papers.
Pipeline: blur → resize → noise → jpeg → mpeg → (repeat) → final_resize → final_blur

Usage:
    python src/data/realbasicvsr_degrade.py \
        --gt_dir experiments/baselines/data/YouHQ40-Test \
        --output_dir experiments/baselines/data/YouHQ40-Test-RealBasicVSR-LQ \
        --scale 4 \
        --seed 42

Requires: the mgldvsr conda env (has basicsr with degradation transforms).
"""
import argparse
import os
import sys
import random
from pathlib import Path
from copy import deepcopy

import cv2
import numpy as np
import torch


def setup_degradation_pipeline(scale, gt_size=None):
    """Create the RealBasicVSR two-stage degradation pipeline.

    Uses the same parameters as mgldvsr_512_realbasicvsr_deg.yaml config.
    """
    from basicsr.data.mmcv_transforms import (
        RandomBlur, RandomResize, RandomNoise,
        RandomJPEGCompression, RandomVideoCompression,
        Clip, UnsharpMasking, RescaleToZeroOne
    )

    # First degradation stage
    blur_1 = RandomBlur(
        params={
            'kernel_size': [7, 9, 11, 13, 15, 17, 19, 21],
            'kernel_list': ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso', 'sinc'],
            'kernel_prob': [0.405, 0.225, 0.108, 0.027, 0.108, 0.027, 0.1],
            'sigma_x': [0.2, 3], 'sigma_y': [0.2, 3],
            'rotate_angle': [-3.1416, 3.1416],
            'beta_gaussian': [0.5, 4], 'beta_plateau': [1, 2],
            'sigma_x_step': 0.02, 'sigma_y_step': 0.02,
            'rotate_angle_step': 0.31416,
            'beta_gaussian_step': 0.05, 'beta_plateau_step': 0.1,
            'omega_step': 0.0628,
        }, keys=['lqs']
    )
    resize_1 = RandomResize(
        params={
            'resize_mode_prob': [0.2, 0.7, 0.1],
            'resize_scale': [0.15, 1.5],
            'resize_opt': ['bilinear', 'area', 'bicubic'],
            'resize_prob': [0.3333, 0.3333, 0.3334],
            'resize_step': 0.015,
            'is_size_even': True,
        }, keys=['lqs']
    )
    noise_1 = RandomNoise(
        params={
            'noise_type': ['gaussian', 'poisson'],
            'noise_prob': [0.5, 0.5],
            'gaussian_sigma': [1, 30],
            'gaussian_gray_noise_prob': 0.4,
            'poisson_scale': [0.05, 3],
            'poisson_gray_noise_prob': 0.4,
            'gaussian_sigma_step': 0.1,
            'poisson_scale_step': 0.005,
        }, keys=['lqs']
    )
    jpeg_1 = RandomJPEGCompression(
        params={'quality': [30, 95], 'quality_step': 3},
        keys=['lqs']
    )
    mpeg_1 = RandomVideoCompression(
        params={
            'codec': ['libx264', 'h264', 'mpeg4'],
            'codec_prob': [0.3333, 0.3333, 0.3334],
            'bitrate': [1e4, 1e5],
        }, keys=['lqs']
    )

    # Second degradation stage
    blur_2 = RandomBlur(
        params={
            'prob': 0.8,
            'kernel_size': [7, 9, 11, 13, 15, 17, 19, 21],
            'kernel_list': ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso', 'sinc'],
            'kernel_prob': [0.405, 0.225, 0.108, 0.027, 0.108, 0.027, 0.1],
            'sigma_x': [0.2, 1.5], 'sigma_y': [0.2, 1.5],
            'rotate_angle': [-3.1416, 3.1416],
            'beta_gaussian': [0.5, 4], 'beta_plateau': [1, 2],
            'sigma_x_step': 0.02, 'sigma_y_step': 0.02,
            'rotate_angle_step': 0.31416,
            'beta_gaussian_step': 0.05, 'beta_plateau_step': 0.1,
            'omega_step': 0.0628,
        }, keys=['lqs']
    )
    resize_2 = RandomResize(
        params={
            'resize_mode_prob': [0.3, 0.4, 0.3],
            'resize_scale': [0.3, 1.2],
            'resize_opt': ['bilinear', 'area', 'bicubic'],
            'resize_prob': [0.3333, 0.3333, 0.3334],
            'resize_step': 0.03,
            'is_size_even': True,
        }, keys=['lqs']
    )
    noise_2 = RandomNoise(
        params={
            'noise_type': ['gaussian', 'poisson'],
            'noise_prob': [0.5, 0.5],
            'gaussian_sigma': [1, 25],
            'gaussian_gray_noise_prob': 0.4,
            'poisson_scale': [0.05, 2.5],
            'poisson_gray_noise_prob': 0.4,
            'gaussian_sigma_step': 0.1,
            'poisson_scale_step': 0.005,
        }, keys=['lqs']
    )
    jpeg_2 = RandomJPEGCompression(
        params={'quality': [30, 95], 'quality_step': 3},
        keys=['lqs']
    )
    mpeg_2 = RandomVideoCompression(
        params={
            'codec': ['libx264', 'h264', 'mpeg4'],
            'codec_prob': [0.3333, 0.3333, 0.3334],
            'bitrate': [1e4, 1e5],
        }, keys=['lqs']
    )

    # Final resize to target LQ size
    resize_final = RandomResize(
        params={
            'target_size': None,  # set per-clip based on GT size / scale
            'resize_opt': ['bilinear', 'area', 'bicubic'],
            'resize_prob': [0.3333, 0.3333, 0.3334],
        }, keys=['lqs']
    )
    blur_final = RandomBlur(
        params={
            'prob': 0.8,
            'kernel_size': [7, 9, 11, 13, 15, 17, 19, 21],
            'kernel_list': ['sinc'],
            'kernel_prob': [1],
            'omega': [1.0472, 3.1416],
            'omega_step': 0.0628,
        }, keys=['lqs']
    )

    # Transforms
    usm = UnsharpMasking(kernel_size=51, sigma=0, weight=0.5, threshold=10, keys=['gts'])
    clip_t = Clip(keys=['lqs'])
    rescale = RescaleToZeroOne(keys=['lqs', 'gts'])

    return {
        'blur_1': blur_1, 'resize_1': resize_1, 'noise_1': noise_1,
        'jpeg_1': jpeg_1, 'mpeg_1': mpeg_1,
        'blur_2': blur_2, 'resize_2': resize_2, 'noise_2': noise_2,
        'jpeg_2': jpeg_2, 'mpeg_2': mpeg_2,
        'resize_final': resize_final, 'blur_final': blur_final,
        'usm': usm, 'clip': clip_t, 'rescale': rescale,
    }


def degrade_frames(frames, pipeline, scale):
    """Apply degradation pipeline to a list of HWC uint8 frames."""
    img_gts = [f.copy() for f in frames]
    img_lqs = deepcopy(img_gts)

    out_dict = {'lqs': img_lqs, 'gts': img_gts}

    # Unsharp mask on GT
    out_dict = pipeline['usm'].transform(out_dict)

    # First degradation
    out_dict = pipeline['blur_1'](out_dict)
    out_dict = pipeline['resize_1'](out_dict)
    out_dict = pipeline['noise_1'](out_dict)
    out_dict = pipeline['jpeg_1'](out_dict)
    out_dict = pipeline['mpeg_1'](out_dict)

    # Second degradation
    out_dict = pipeline['blur_2'](out_dict)
    out_dict = pipeline['resize_2'](out_dict)
    out_dict = pipeline['noise_2'](out_dict)
    out_dict = pipeline['jpeg_2'](out_dict)
    out_dict = pipeline['mpeg_2'](out_dict)

    # Final resize to target LQ size
    h, w = frames[0].shape[:2]
    target_h, target_w = h // scale, w // scale
    pipeline['resize_final'].params['target_size'] = [target_h, target_w]
    out_dict = pipeline['resize_final'](out_dict)
    out_dict = pipeline['blur_final'](out_dict)

    # Post-process
    out_dict = pipeline['clip'](out_dict)

    return out_dict['lqs']


IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def main():
    parser = argparse.ArgumentParser(description='Apply RealBasicVSR degradation')
    parser.add_argument('--gt_dir', required=True, help='GT frames root (with clip subdirs)')
    parser.add_argument('--output_dir', required=True, help='Output LQ frames root')
    parser.add_argument('--scale', type=int, default=4, help='Downscale factor')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_frames', type=int, default=5, help='Frames per degradation batch')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    gt_root = Path(args.gt_dir)
    out_root = Path(args.output_dir)

    pipeline = setup_degradation_pipeline(args.scale)

    clip_dirs = sorted(d for d in gt_root.iterdir() if d.is_dir())
    print(f"Processing {len(clip_dirs)} clips, scale={args.scale}x, seed={args.seed}")

    for clip_dir in clip_dirs:
        clip_name = clip_dir.name
        out_clip = out_root / clip_name
        out_clip.mkdir(parents=True, exist_ok=True)

        frame_paths = sorted(f for f in clip_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS)
        if not frame_paths:
            continue

        print(f"  {clip_name}: {len(frame_paths)} frames", end="")

        # Load all frames
        frames = [cv2.imread(str(p)) for p in frame_paths]  # BGR HWC uint8
        h, w = frames[0].shape[:2]
        print(f" ({w}x{h} -> {w//args.scale}x{h//args.scale})")

        # Process in batches of num_frames
        all_lq = []
        for i in range(0, len(frames), args.num_frames):
            batch = frames[i:i + args.num_frames]
            # Pad last batch if needed
            while len(batch) < args.num_frames:
                batch.append(batch[-1])
            lq_batch = degrade_frames(batch, pipeline, args.scale)
            all_lq.extend(lq_batch[:min(args.num_frames, len(frames) - i)])

        # Save LQ frames
        for frame_path, lq in zip(frame_paths, all_lq):
            if isinstance(lq, np.ndarray):
                # Already uint8 HWC
                if lq.max() <= 1.0:
                    lq = (lq * 255).clip(0, 255).astype(np.uint8)
                cv2.imwrite(str(out_clip / frame_path.name), lq)
            else:
                print(f"    Warning: unexpected type {type(lq)} for {frame_path.name}")

    print(f"\nDone. LQ frames saved to: {out_root}")


if __name__ == '__main__':
    main()
