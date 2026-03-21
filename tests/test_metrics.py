"""Tests for VSR evaluation metrics."""
import numpy as np
import pytest
from src.evaluation.metrics import psnr, ssim, temporal_consistency, evaluate_sequence


def test_psnr_identical():
    """Identical images should give very high PSNR."""
    img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    assert psnr(img, img) == float('inf') or psnr(img, img) > 50


def test_psnr_different():
    """Different images should give finite PSNR."""
    img1 = np.zeros((64, 64, 3), dtype=np.uint8)
    img2 = np.full((64, 64, 3), 128, dtype=np.uint8)
    result = psnr(img1, img2)
    assert 0 < result < 50


def test_ssim_identical():
    """Identical images should give SSIM = 1.0."""
    img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    assert ssim(img, img) == pytest.approx(1.0)


def test_ssim_range():
    """SSIM should be between -1 and 1."""
    img1 = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    img2 = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    result = ssim(img1, img2)
    assert -1 <= result <= 1


def test_temporal_consistency_static():
    """Static sequence (identical frames) should have 0 temporal difference."""
    frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    frames = [frame] * 5
    assert temporal_consistency(frames) == 0.0


def test_temporal_consistency_single_frame():
    """Single frame should return 0."""
    frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    assert temporal_consistency([frame]) == 0.0


def test_evaluate_sequence_keys():
    """evaluate_sequence should return expected keys."""
    frames = [np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8) for _ in range(3)]
    gts = [np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8) for _ in range(3)]
    result = evaluate_sequence(frames, gts)
    assert "PSNR_mean" in result
    assert "SSIM_mean" in result
    assert "temporal_consistency" in result
    assert len(result["PSNR_per_frame"]) == 3
