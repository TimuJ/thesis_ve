"""Build a long-video LR/GT set: curated HR PNG frames -> bicubic x4 LR + GT.

Run on the Mac; bridge the resulting LR+GT dirs to the server via the
GitHub-branch method. PNG only (no MP4 re-encode -> avoids ~7 dB PSNR loss).
"""
import glob
import os
import cv2


def downsample_x4(hr):
    h, w = hr.shape[:2]
    return cv2.resize(hr, (w // 4, h // 4), interpolation=cv2.INTER_CUBIC)


def build_pair(hr_frames_dir, out_lr_dir, out_gt_dir):
    os.makedirs(out_lr_dir, exist_ok=True)
    os.makedirs(out_gt_dir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(hr_frames_dir, "*.png")))
    for i, p in enumerate(paths):
        hr = cv2.imread(p)
        h, w = hr.shape[:2]
        h4, w4 = (h // 4) * 4, (w // 4) * 4
        hr = hr[:h4, :w4]
        cv2.imwrite(os.path.join(out_gt_dir, f"{i:04d}.png"), hr)
        cv2.imwrite(os.path.join(out_lr_dir, f"{i:04d}.png"), downsample_x4(hr))
    return len(paths)
