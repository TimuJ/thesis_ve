"""Tests for the clip-score-dispersion reliability gate on sub-metric I."""
import pytest
from scripts.lr_vcc.identity import identity_score, clip_score_dispersion


def _pv(clip_scores, fused=0.6):
    n = len(clip_scores)
    return {
        "slow": fused, "fast": fused, "fused": fused,
        "n_clips": n, "n_clips_with_faces": n,
        "clip_detail": [
            {"clip_index": i, "clip_path": f"c{i}.mp4", "score": s}
            for i, s in enumerate(clip_scores)
        ],
    }


def test_dispersion_zero_for_constant_scores():
    assert clip_score_dispersion(_pv([0.7] * 6)) == pytest.approx(0.0, abs=1e-12)


def test_dispersion_none_without_clip_detail():
    assert clip_score_dispersion({"fused": 0.6, "n_clips": 6, "n_clips_with_faces": 6}) is None


def test_dispersion_none_with_single_valid_clip():
    assert clip_score_dispersion(_pv([0.7])) is None


def test_invalid_scores_excluded_from_dispersion():
    pv = _pv([0.7, 0.7, 0.7])
    pv["clip_detail"].append({"clip_index": 3, "clip_path": "c3.mp4", "score": -1.0})
    assert clip_score_dispersion(pv) == pytest.approx(0.0, abs=1e-12)


def test_low_dispersion_keeps_reliability():
    # disp=0.0 gives a small sigmoid tail penalty (~0.076), so reliability is
    # multiplied by ~0.924 vs the no-clip-detail case (exact 1.0).
    # We assert the gated reliability is at least 90% of the ungated value.
    gated = identity_score(_pv([0.7] * 6))
    ungated = identity_score({"fused": 0.6, "n_clips": 6, "n_clips_with_faces": 6})
    assert gated["reliability"] >= 0.9 * ungated["reliability"]


def test_high_dispersion_cuts_reliability():
    flappy = identity_score(_pv([0.05, 0.85, 0.77, 0.05, 0.75, 0.05]))
    stable = identity_score(_pv([0.7] * 6))
    assert flappy["reliability"] < 0.5 * stable["reliability"]
    assert flappy["details"]["clip_score_dispersion"] > 0.3
    assert flappy["details"]["dispersion_penalty"] > 0.5
