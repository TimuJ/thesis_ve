"""GT-based per-axis reliability.

Each PairScorer may expose an optional ``reliability(crop_a, crop_b) -> float`` in
[0, 1] reporting how much *signal* it has on a crop pair (e.g. SIFT texture / keypoint
count). We compute it on the **GT** crop pair, so the weight an axis gets in the
composite reflects whether the measurement is meaningful on this pair — not whether the
SR happens to be smooth. Keying it on SR would let an over-smoothed SR (few keypoints /
matches) register as "no signal" and down-weight — and thereby hide — its own
over-consistency failure. Scorers without a ``reliability`` method default to 1.0.
"""
from __future__ import annotations

import numpy as np


def axis_reliabilities(scorers, gt_crop_a: np.ndarray, gt_crop_b: np.ndarray) -> dict:
    """``{scorer.name: scorer.reliability(gt_crop_a, gt_crop_b) or 1.0}`` — the GT-side
    per-axis signal used to weight each axis in the composite."""
    out = {}
    for M in scorers:
        rel = getattr(M, "reliability", None)
        out[M.name] = float(rel(gt_crop_a, gt_crop_b)) if callable(rel) else 1.0
    return out
