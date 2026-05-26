"""Sub-metric D — Color-histogram temporal stability (L*a*b*, NR).

Closes the color-drift blind spot identified in Section 5.3 of the proposal:
existing tOF/tLP/CLIP-IQA all fail to detect long-range color drift because
optical-flow warping absorbs uniform color shifts and learned-representation
metrics are partially color-invariant. Histogram-distance over long-k pairs
is a direct measure of distribution drift that the other metrics can't see.

Implementation note on sub-bin sensitivity
------------------------------------------
A pure 32-bin histogram L1 distance has a bin-width of ~8 LAB units and will
return exactly 0.0 for drifts that don't cross a bin boundary — a dead-zone
that breaks monotonicity on low-severity inputs. We augment each channel's
descriptor with its normalized channel mean (1 extra float), which captures
continuous sub-bin shifts and ensures the distance function is monotonic even
for small colour drifts. The L1 term still dominates for large drifts where
histogram mass visibly shifts across bins.
"""
import math
from typing import List, Sequence

import numpy as np
import cv2

from .reliability import below_threshold_penalty


_DEFAULT_ALPHA = 5.0
_DEFAULT_ENTROPY_FLOOR = 1.0

# Scale factor applied to the normalized mean-shift term so it is comparable
# in magnitude to the histogram L1 term for moderate drifts.
# hist_l1 range: 0–6 (3 ch × max 2.0)
# mean_shift range: 0–3 (3 ch × max 1.0 when normalized to [0,1])
# → multiply by 2 to match the hist_l1 scale.
_MEAN_SHIFT_SCALE = 2.0


def frame_to_lab_histogram(frame_bgr: np.ndarray, n_bins: int = 32) -> np.ndarray:
    """Return concatenated L*+a*+b* histograms, each normalized to sum=1.

    Output shape: (n_bins * 3,).  Used externally and in tests.
    """
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    parts = []
    for c in range(3):
        h, _ = np.histogram(lab[..., c], bins=n_bins, range=(0, 256))
        h = h.astype(np.float64)
        s = h.sum()
        if s > 0:
            h /= s
        parts.append(h)
    return np.concatenate(parts)


def _frame_to_lab_descriptor(frame_bgr: np.ndarray, n_bins: int = 32) -> np.ndarray:
    """Internal: histogram + per-channel normalized mean.

    Shape: (n_bins * 3 + 3,).  The extra 3 floats are L*, a*, b* means
    divided by 255 so they live in [0, 1], enabling sub-bin sensitivity.
    """
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    parts = []
    for c in range(3):
        h, _ = np.histogram(lab[..., c], bins=n_bins, range=(0, 256))
        h = h.astype(np.float64)
        s = h.sum()
        if s > 0:
            h /= s
        parts.append(h)
        parts.append(np.array([float(lab[..., c].mean()) / 255.0]))
    return np.concatenate(parts)


def histogram_l1_distance(h1: np.ndarray, h2: np.ndarray) -> float:
    """L1 distance between two normalized histogram vectors.

    Accepts vectors of shape (n_bins * 3,) as returned by
    ``frame_to_lab_histogram``.  Max value is 6.0.
    """
    return float(np.abs(h1 - h2).sum())


def _descriptor_distance(d1: np.ndarray, d2: np.ndarray, n_bins: int = 32) -> float:
    """Combined histogram-L1 + scaled channel-mean distance.

    Range: 0 to 12.0  (hist_l1 ≤ 6 + mean_shift_scaled ≤ 6).
    """
    block = n_bins + 1
    hist_l1 = 0.0
    mean_dist = 0.0
    for c in range(3):
        s = c * block
        hist_l1 += float(np.abs(d1[s:s + n_bins] - d2[s:s + n_bins]).sum())
        mean_dist += abs(float(d1[s + n_bins]) - float(d2[s + n_bins]))
    return hist_l1 + _MEAN_SHIFT_SCALE * mean_dist


def histogram_entropy(h: np.ndarray) -> float:
    """Shannon entropy of a single-channel histogram (in nats)."""
    p = h[h > 0]
    return float(-(p * np.log(p)).sum()) if p.size > 0 else 0.0


def color_histogram_score_from_frames(frames: List[np.ndarray],
                                       k_values: Sequence[int] = (60, 120),
                                       max_pairs: int = 200,
                                       n_bins: int = 32,
                                       alpha: float = _DEFAULT_ALPHA,
                                       entropy_floor: float = _DEFAULT_ENTROPY_FLOOR) -> dict:
    """Compute color-histogram temporal-stability score over a list of frames.

    Returns {"score", "reliability", "details": {...}}.
    """
    if len(frames) < min(k_values) + 1:
        return {"score": 0.0, "reliability": 0.0, "details": {}}

    # Precompute per-frame descriptors (histogram + channel means) once.
    descriptors = [_frame_to_lab_descriptor(fr, n_bins=n_bins) for fr in frames]

    # Also precompute pure histograms for entropy computation.
    histograms = [frame_to_lab_histogram(fr, n_bins=n_bins) for fr in frames]

    # Per-frame mean entropy (averaged over the L, a, b channels).
    entropies = []
    for h in histograms:
        total = sum(histogram_entropy(h[c * n_bins:(c + 1) * n_bins]) for c in range(3))
        entropies.append(total / 3.0)
    mean_entropy = float(np.mean(entropies))

    # Distance over all sampled pairs at each k.
    dists = []
    per_k_mean = {}
    for k in k_values:
        n_possible = len(descriptors) - k
        if n_possible <= 0:
            per_k_mean[k] = None
            continue
        stride = max(1, n_possible // max_pairs)
        t_indices = list(range(0, n_possible, stride))[:max_pairs]
        ds = [_descriptor_distance(descriptors[t], descriptors[t + k], n_bins=n_bins)
              for t in t_indices]
        per_k_mean[k] = float(np.mean(ds))
        dists.extend(ds)

    if not dists:
        return {"score": 0.0, "reliability": 0.0, "details": {}}

    mean_dist = float(np.mean(dists))
    score = math.exp(-alpha * mean_dist)
    reliability = 1.0 - below_threshold_penalty(mean_entropy, entropy_floor, sharpness=2.0)

    return {
        "score": max(0.0, min(1.0, score)),
        "reliability": max(0.0, min(1.0, reliability)),
        "details": {
            "mean_l1_dist": mean_dist,
            "per_k_mean_dist": per_k_mean,
            "mean_entropy": mean_entropy,
            "alpha": alpha,
            "n_frames": len(frames),
            "n_pairs_total": len(dists),
        },
    }
