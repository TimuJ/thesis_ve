"""Tests for LPIPS metric."""
import numpy as np
import pytest

from src.evaluation.metrics import lpips_score, HAS_LPIPS


@pytest.mark.skipif(not HAS_LPIPS, reason="LPIPS requires torch and lpips packages")
def test_lpips_identical():
    """Identical images should give LPIPS close to 0."""
    img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    result = lpips_score(img, img)
    assert result < 0.01


@pytest.mark.skipif(not HAS_LPIPS, reason="LPIPS requires torch and lpips packages")
def test_lpips_different():
    """Different images should give positive LPIPS."""
    img1 = np.zeros((64, 64, 3), dtype=np.uint8)
    img2 = np.full((64, 64, 3), 255, dtype=np.uint8)
    result = lpips_score(img1, img2)
    assert result > 0.0


def test_lpips_disabled_excludes_key():
    """When compute_lpips=False, evaluate_sequence should not include LPIPS."""
    from src.evaluation.metrics import evaluate_sequence
    frames = [np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8) for _ in range(3)]
    gts = [np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8) for _ in range(3)]
    result = evaluate_sequence(frames, gts, compute_lpips=False)
    assert "LPIPS_mean" not in result
