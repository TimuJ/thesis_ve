"""Sub-metric E — Color-slope drift detection (L*a*b*, NR).

A direct detector for the monotonic linear color drift that sub-metric D
under-responds to. We fit a linear regression of each L*a*b* channel's
per-frame mean over the frame index, and use the steepest slope magnitude
as the artefact signal:

    score       = exp(-beta * max(|slope_L|, |slope_a|, |slope_b|))
    reliability = max(R^2_L, R^2_a, R^2_b)  gated by an R^2 floor

A clean video has near-zero slope AND near-zero R^2 (so the slope it does
find is just noise) -- the R^2 gate means the score doesn't falsely flag
clean videos AS clean for the wrong reason, but instead reports low
reliability so the composite down-weights this sub-metric.

A drifting video has a large slope AND R^2 near 1.0 -- the metric fires.

A flicker / sinusoidal artefact has a near-zero best-fit slope and
near-zero R^2 (since the residual is enormous) -- reliability collapses,
and the metric correctly abstains rather than false-positives.
"""
import math
from typing import List

import numpy as np
import cv2

from .reliability import below_threshold_penalty


_DEFAULT_BETA = 50.0  # tuneable; reflects "score = 0.5 when max|slope| = 0.0139 per frame"
_DEFAULT_R2_FLOOR = 0.15  # reliability floor — below this, signal is noise, not drift


def per_frame_channel_means(frames: List[np.ndarray]) -> np.ndarray:
    """Return (N, 3) array of L*, a*, b* means per frame, each in [0, 255]."""
    out = np.zeros((len(frames), 3), dtype=np.float64)
    for i, fr in enumerate(frames):
        lab = cv2.cvtColor(fr, cv2.COLOR_BGR2LAB)
        for c in range(3):
            out[i, c] = float(lab[..., c].mean())
    return out


def _linregress(y: np.ndarray):
    """Return (slope, r_squared) of linear fit of y vs np.arange(len(y))."""
    n = len(y)
    x = np.arange(n, dtype=np.float64)
    x_mean = x.mean()
    y_mean = y.mean()
    sxy = float(((x - x_mean) * (y - y_mean)).sum())
    sxx = float(((x - x_mean) ** 2).sum())
    if sxx == 0:
        return 0.0, 0.0
    slope = sxy / sxx
    intercept = y_mean - slope * x_mean
    y_pred = slope * x + intercept
    ss_tot = float(((y - y_mean) ** 2).sum())
    if ss_tot == 0:
        return slope, 0.0
    ss_res = float(((y - y_pred) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot
    return slope, r2


def color_slope_score_from_means(channel_means: np.ndarray,
                                  beta: float = _DEFAULT_BETA,
                                  r2_floor: float = _DEFAULT_R2_FLOOR) -> dict:
    """Compute slope sub-metric.

    Score = exp(-beta * max(|slope_L|, |slope_a|, |slope_b|))
    Reliability = max(R^2_L, R^2_a, R^2_b), gated by r2_floor sigmoid.
    """
    if len(channel_means) < 30:
        return {"score": 1.0, "reliability": 0.0,
                "details": {"reason": "too_few_frames", "n_frames": int(len(channel_means))}}
    slopes = []
    r2s = []
    for c in range(3):
        s, r2 = _linregress(channel_means[:, c])
        slopes.append(s)
        r2s.append(r2)
    abs_slopes = [abs(s) for s in slopes]
    max_abs = max(abs_slopes)
    max_r2 = max(r2s)
    score = math.exp(-beta * max_abs)
    reliability = 1.0 - below_threshold_penalty(max_r2, r2_floor, sharpness=10.0)
    return {
        "score": max(0.0, min(1.0, float(score))),
        "reliability": max(0.0, min(1.0, float(reliability))),
        "details": {
            "slope_L": slopes[0], "slope_a": slopes[1], "slope_b": slopes[2],
            "r2_L": r2s[0], "r2_a": r2s[1], "r2_b": r2s[2],
            "max_abs_slope": max_abs,
            "max_r2": max_r2,
            "beta": beta,
            "n_frames": int(len(channel_means)),
        },
    }


def color_slope_score_from_frames(frames: List[np.ndarray],
                                   beta: float = _DEFAULT_BETA,
                                   r2_floor: float = _DEFAULT_R2_FLOOR) -> dict:
    means = per_frame_channel_means(frames)
    return color_slope_score_from_means(means, beta=beta, r2_floor=r2_floor)
