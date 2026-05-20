"""Tests for sub-metric T (temporal — tOF + mask coverage reliability)."""
from scripts.lr_vcc.temporal import temporal_score


def _fixture_tof_payload(tof_values: dict, mask_coverage: dict) -> dict:
    """Same shape as scripts/long_range_temporal/eval_tof_tlp.py output."""
    return {
        "video_path": "/fake/video.mp4",
        "n_frames": 5000,
        "fps": 30.0,
        "k_values": [1, 5, 10, 30, 60, 120],
        "tof": {str(k): v for k, v in tof_values.items()},
        "tlp": {str(k): 0.0 for k in tof_values},
        "n_pairs_used": {str(k): 200 for k in tof_values},
        "mean_mask_coverage": {str(k): v for k, v in mask_coverage.items()},
    }


def test_low_tof_high_coverage_yields_high_score():
    tof = {1: 0.01, 5: 0.02, 10: 0.03, 30: 0.04, 60: 0.05, 120: 0.06}
    cov = {1: 0.9, 5: 0.7, 10: 0.5, 30: 0.4, 60: 0.3, 120: 0.2}
    out = temporal_score(_fixture_tof_payload(tof, cov))
    assert out["score"] > 0.9
    assert out["reliability"] > 0.9


def test_high_tof_yields_low_score():
    tof = {1: 0.3, 5: 0.4, 10: 0.5, 30: 0.6, 60: 0.7, 120: 0.8}
    cov = {1: 0.9, 5: 0.7, 10: 0.5, 30: 0.4, 60: 0.3, 120: 0.2}
    out = temporal_score(_fixture_tof_payload(tof, cov))
    assert out["score"] < 0.5
    assert out["reliability"] > 0.9


def test_low_mask_coverage_drops_reliability():
    tof = {1: 0.01, 5: 0.02, 10: 0.03, 30: 0.04, 60: 0.05, 120: 0.06}
    cov = {1: 0.05, 5: 0.04, 10: 0.03, 30: 0.02, 60: 0.01, 120: 0.005}  # all below 0.10 floor
    out = temporal_score(_fixture_tof_payload(tof, cov))
    assert out["reliability"] < 0.4
