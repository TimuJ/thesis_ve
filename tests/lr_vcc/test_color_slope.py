"""Tests for sub-metric E (color_slope) — linear-regression drift detector.

Contract:
- Clean video (low slope, low R^2) → high score, LOW reliability
  (so composite down-weights it — slope was just noise).
- Linear drift video (high slope, high R^2) → very low score, high reliability.
- Sinusoidal flicker (zero slope, zero R^2) → score near 1, low reliability
  (correct abstention: this sub-metric DOESN'T fire on flicker).
- Too-few-frames edge case → reliability == 0.0.
"""
import numpy as np

from scripts.lr_vcc.color_slope import color_slope_score_from_means


def test_clean_video_high_score():
    # Small noise around a constant L*a*b* mean. With sigma=0.1 over 120 frames,
    # |slope| stays < 1e-3 → exp(-50 * 1e-3) ≈ 0.95+, and R^2 is near zero for
    # pure noise (E[R^2] ≈ 1/(n-2) ≈ 0.008 per channel, max across 3 channels
    # stays around 0.04). The reliability gate (R^2 floor 0.15) puts the
    # reliability in the 0.18-0.28 band — well below the 0.5 mid-point —
    # so the composite will down-weight this sub-metric, which is the
    # whole point: a clean video gives this metric nothing to do.
    rng = np.random.default_rng(seed=0)
    means = 128.0 + rng.standard_normal((120, 3)) * 0.1
    out = color_slope_score_from_means(means)
    assert out["score"] > 0.95, out
    assert out["reliability"] < 0.3, out


def test_linear_drift_low_score():
    means = np.full((120, 3), 100.0, dtype=np.float64)
    means[:, 0] = 100.0 + np.arange(120) * 0.2  # slope = 0.2 per frame on L*
    out = color_slope_score_from_means(means)
    # beta * slope = 50 * 0.2 = 10 → exp(-10) ≈ 4.5e-5
    assert out["score"] < 0.05, out
    assert out["reliability"] > 0.95, out


def test_sinusoidal_flicker_low_reliability():
    means = np.full((120, 3), 128.0, dtype=np.float64)
    means[:, 0] = 128.0 + 10.0 * np.sin(2.0 * np.pi * np.arange(120) / 15.0)
    out = color_slope_score_from_means(means)
    # Linear fit slope ≈ 0; R^2 ≈ 0 → reliability collapses; metric abstains.
    assert out["reliability"] < 0.3, out


def test_too_few_frames_returns_zero_reliability():
    means = np.full((10, 3), 128.0, dtype=np.float64)
    out = color_slope_score_from_means(means)
    assert out["reliability"] == 0.0, out
