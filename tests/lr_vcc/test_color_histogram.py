import math
import numpy as np
import cv2
from scripts.lr_vcc.color_histogram import (
    frame_to_lab_histogram,
    histogram_l1_distance,
    color_histogram_score_from_frames,
)
from scripts.lr_vcc.color_stability import color_stability_score


def _make_constant_frames(n=100, color=(128, 128, 128)):
    """Return n identical frames of the given BGR color."""
    fr = np.full((100, 100, 3), color, dtype=np.uint8)
    return [fr.copy() for _ in range(n)]


def _make_drifting_frames(n=100, severity=0.40):
    """Linear color drift, same as our synthetic test set."""
    out = []
    for i in range(n):
        drift = severity * (i / (n - 1))
        fr = np.full((100, 100, 3), 100, dtype=np.float32)
        fr[..., 2] *= 1.0 + drift          # R up
        fr[..., 1] *= 1.0 - drift / 2.0
        fr[..., 0] *= 1.0 - drift / 2.0
        out.append(np.clip(fr, 0, 255).astype(np.uint8))
    return out


def test_histogram_l1_zero_for_identical():
    fr = np.full((50, 50, 3), 128, dtype=np.uint8)
    h1 = frame_to_lab_histogram(fr, n_bins=32)
    h2 = frame_to_lab_histogram(fr, n_bins=32)
    assert histogram_l1_distance(h1, h2) == 0.0


def test_histogram_l1_positive_for_different():
    fr1 = np.full((50, 50, 3), 50, dtype=np.uint8)
    fr2 = np.full((50, 50, 3), 200, dtype=np.uint8)
    h1 = frame_to_lab_histogram(fr1, n_bins=32)
    h2 = frame_to_lab_histogram(fr2, n_bins=32)
    assert histogram_l1_distance(h1, h2) > 0.5  # very different colors


def test_static_video_high_score():
    frames = _make_constant_frames(n=200)
    out = color_histogram_score_from_frames(frames, k_values=[60, 120], max_pairs=100)
    assert out["score"] > 0.95  # static video is super-stable


def test_drifting_video_lower_score_than_static():
    static = _make_constant_frames(n=200)
    drifting = _make_drifting_frames(n=200, severity=0.40)
    s_static = color_histogram_score_from_frames(static, k_values=[60, 120], max_pairs=100)
    s_drift = color_histogram_score_from_frames(drifting, k_values=[60, 120], max_pairs=100)
    assert s_drift["score"] < s_static["score"]
    # Concretely expect a clear gap
    assert s_static["score"] - s_drift["score"] > 0.1


def test_score_monotonic_in_severity():
    scores = []
    for sev in [0.02, 0.10, 0.40]:
        frames = _make_drifting_frames(n=150, severity=sev)
        out = color_histogram_score_from_frames(frames, k_values=[60, 120], max_pairs=60)
        scores.append(out["score"])
    # higher severity -> lower score
    assert scores[0] > scores[1] > scores[2]


def test_reliability_drops_on_low_entropy():
    # constant grey - entropy is very low
    frames = _make_constant_frames(n=100, color=(128, 128, 128))
    out = color_histogram_score_from_frames(frames, k_values=[60], max_pairs=50)
    assert out["reliability"] < 0.5


# ---------------------------------------------------------------------------
# Sub-metric D — color_stability_score (JSON-wrapper around mean_hist_dist)
# ---------------------------------------------------------------------------

def test_zero_distance_perfect_score():
    pv = {"n_frames": 5000, "mean_hist_dist": 0.0}
    out = color_stability_score(pv)
    assert abs(out["score"] - 1.0) < 1e-6
    assert out["reliability"] > 0.95


def test_large_distance_low_score():
    pv = {"n_frames": 5000, "mean_hist_dist": 1.0}
    out = color_stability_score(pv)
    assert out["score"] < 0.2  # exp(-2 * 1.0) = 0.135
    assert out["reliability"] > 0.95


def test_short_video_low_reliability():
    pv = {"n_frames": 100, "mean_hist_dist": 0.1}  # well below 240 floor
    out = color_stability_score(pv)
    assert out["reliability"] < 0.5


def test_missing_dist_zero():
    pv = {"n_frames": 5000, "mean_hist_dist": None}
    out = color_stability_score(pv)
    assert out["score"] == 0.0
    assert out["reliability"] == 0.0


def test_alpha_controls_decay():
    pv = {"n_frames": 5000, "mean_hist_dist": 0.5}
    out_low = color_stability_score(pv, alpha=1.0)
    out_high = color_stability_score(pv, alpha=5.0)
    assert out_low["score"] > out_high["score"]
