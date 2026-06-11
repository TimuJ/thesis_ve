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


def test_dispersion_none_when_all_scores_invalid():
    assert clip_score_dispersion(_pv([-1.0, -1.0, -1.0, -1.0])) is None


def test_invalid_scores_excluded_from_dispersion():
    pv = _pv([0.7, 0.7, 0.7])
    pv["clip_detail"].append({"clip_index": 3, "clip_path": "c3.mp4", "score": -1.0})
    assert clip_score_dispersion(pv) == pytest.approx(0.0, abs=1e-12)


def test_low_dispersion_keeps_reliability():
    # disp=0.0 gives a small sigmoid tail penalty (~0.031 at threshold 0.25),
    # so reliability is multiplied by ~0.969 vs the no-clip-detail case (1.0).
    # We assert the gated reliability is at least 90% of the ungated value.
    gated = identity_score(_pv([0.7] * 6), dispersion_threshold=0.25)
    ungated = identity_score({"fused": 0.6, "n_clips": 6, "n_clips_with_faces": 6},
                             dispersion_threshold=0.25)
    assert gated["reliability"] >= 0.9 * ungated["reliability"]


def test_high_dispersion_cuts_reliability():
    flappy = identity_score(_pv([0.05, 0.85, 0.77, 0.05, 0.75, 0.05]),
                            dispersion_threshold=0.25)
    stable = identity_score(_pv([0.7] * 6), dispersion_threshold=0.25)
    assert flappy["reliability"] < 0.5 * stable["reliability"]
    assert flappy["details"]["clip_score_dispersion"] > 0.3
    assert flappy["details"]["dispersion_penalty"] > 0.5


def test_gate_off_by_default():
    # Without dispersion_threshold the gate is OFF: reliability equals the two-factor
    # formula (no dispersion penalty), even for a highly flappy per_video.
    flappy_scores = [0.05, 0.85, 0.77, 0.05, 0.75, 0.05]
    result = identity_score(_pv(flappy_scores))
    # dispersion_penalty must be 0.0 (gate off)
    assert result["details"]["dispersion_penalty"] == pytest.approx(0.0, abs=1e-9)
    # clip_score_dispersion is still reported (observability)
    assert result["details"]["clip_score_dispersion"] is not None
    assert result["details"]["clip_score_dispersion"] > 0.3
    # reliability equals the same per_video evaluated with clip_detail absent (gate truly off)
    pv_no_detail = {k: v for k, v in _pv(flappy_scores).items() if k != "clip_detail"}
    ref = identity_score(pv_no_detail)
    assert result["reliability"] == pytest.approx(ref["reliability"], abs=1e-9)
