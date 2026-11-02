"""GT-anchored reduced-reference consistency + dual (full-frame & crop) fidelity.

The honest target in real footage is not "identical" — the object genuinely
changed across the gap — so we score the SR pair's similarity against the GT
pair's own similarity. This penalises both drift (c_sr < c_gt) and fabricated
over-consistency (c_sr > c_gt, e.g. a blur matching both sides more than the real
footage does). Fidelity is a separate axis, reported full-frame and on the crop.
"""
from __future__ import annotations

import numpy as np

from src.evaluation.metrics import HAS_LPIPS, psnr, ssim

from .crops import bbox_crop


def _psnr(a, b) -> float:
    """PSNR, with perfect reconstruction (identical frames) reported as +inf
    directly rather than through skimage's divide-by-zero on a zero MSE."""
    if np.array_equal(a, b):
        return float("inf")
    return float(psnr(a, b))


def rr_consistency(c_gt: float, c_sr: float) -> float:
    return max(0.0, 1.0 - abs(c_sr - c_gt))


def consistency_scores(c_gt: dict, c_sr: dict) -> dict:
    """Per-metric reduced-reference consistency over the metrics present in both."""
    return {k: rr_consistency(c_gt[k], c_sr[k]) for k in c_gt if k in c_sr}


def _fid_pair(sr_a, gt_a, sr_b, gt_b, compute_lpips: bool) -> dict:
    d = {
        "psnr": float((_psnr(sr_a, gt_a) + _psnr(sr_b, gt_b)) / 2.0),
        "ssim": float((ssim(sr_a, gt_a) + ssim(sr_b, gt_b)) / 2.0),
    }
    if compute_lpips and HAS_LPIPS:
        from src.evaluation.metrics import lpips_score
        d["lpips"] = float((lpips_score(sr_a, gt_a) + lpips_score(sr_b, gt_b)) / 2.0)
    return d


def fidelity(sr_a, gt_a, sr_b, gt_b, box_a=None, box_b=None, compute_lpips: bool = True) -> dict:
    """Full-reference PSNR/SSIM(/LPIPS) averaged over the two paired frames, both
    full-frame and (when boxes are given and crops are valid) on the object crop."""
    out = {}
    full = _fid_pair(sr_a, gt_a, sr_b, gt_b, compute_lpips)
    out["psnr_full"], out["ssim_full"] = full["psnr"], full["ssim"]
    if "lpips" in full:
        out["lpips_full"] = full["lpips"]
    if box_a is not None and box_b is not None:
        crops = (bbox_crop(sr_a, box_a), bbox_crop(gt_a, box_a),
                 bbox_crop(sr_b, box_b), bbox_crop(gt_b, box_b))
        if all(c is not None for c in crops):
            cr = _fid_pair(*crops, compute_lpips)
            out["psnr_crop"], out["ssim_crop"] = cr["psnr"], cr["ssim"]
            if "lpips" in cr:
                out["lpips_crop"] = cr["lpips"]
    return out
