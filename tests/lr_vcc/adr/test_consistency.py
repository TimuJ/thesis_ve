"""Task 6: GT-anchored reduced-reference consistency and dual fidelity."""
import math
import warnings

import numpy as np

from scripts.lr_vcc.adr.consistency import rr_consistency, consistency_scores, fidelity


def test_rr_consistency_equal_is_one():
    assert rr_consistency(0.8, 0.8) == 1.0


def test_rr_consistency_penalizes_over_consistency():   # blur reductio
    assert rr_consistency(0.7, 0.95) < 1.0
    assert rr_consistency(0.7, 0.95) == rr_consistency(0.7, 0.45)


def test_rr_consistency_clamps_at_zero():
    assert rr_consistency(0.1, 0.95) >= 0.0


def test_consistency_scores_per_metric():
    out = consistency_scores({"sift": 0.8, "dino": 0.6}, {"sift": 0.8, "dino": 0.4})
    assert out["sift"] == 1.0 and abs(out["dino"] - 0.8) < 1e-9


def test_fidelity_near_identical_is_high_and_reports_crop_and_full():
    a = np.random.default_rng(0).integers(0, 255, (64, 64, 3)).astype(np.uint8)
    sr = a.copy()
    sr[10, 10, 0] = (int(a[10, 10, 0]) + 2) % 256  # one pixel off, inside the crop → finite PSNR
    d = fidelity(sr, a, sr, a, box_a=(8, 8, 16, 16), box_b=(8, 8, 16, 16), compute_lpips=False)
    assert d["psnr_full"] > 40 and "psnr_crop" in d


def test_fidelity_identical_frames_is_inf_without_warning():
    a = np.random.default_rng(0).integers(0, 255, (64, 64, 3)).astype(np.uint8)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # perfect SR must not route through skimage's divide-by-zero
        d = fidelity(a, a, a, a, box_a=(8, 8, 16, 16), box_b=(8, 8, 16, 16), compute_lpips=False)
    assert math.isinf(d["psnr_full"]) and math.isinf(d["psnr_crop"])
