"""Tests for the probability-weighted 1-5 scorer (G-Eval / Q-Align trick)."""
import math

import pytest

from scripts.lr_vcc.mllm_judge.scorer import axis_scores, weighted_score


def test_uniform_logits_score_three():
    # Equal logits over '1'..'5' -> symmetric mass -> expected value exactly 3.
    assert weighted_score([0.0, 0.0, 0.0, 0.0, 0.0]) == pytest.approx(3.0)


def test_mass_on_five_scores_near_five():
    assert weighted_score([0.0, 0.0, 0.0, 0.0, 50.0]) == pytest.approx(5.0, abs=1e-6)


def test_mass_on_one_scores_near_one():
    assert weighted_score([50.0, 0.0, 0.0, 0.0, 0.0]) == pytest.approx(1.0, abs=1e-6)


def test_symmetric_distribution_scores_three():
    # Mass mirrored around the centre bin stays at 3.
    assert weighted_score([1.0, 2.0, 3.0, 2.0, 1.0]) == pytest.approx(3.0)


def test_lean_high_is_between_three_and_five():
    s = weighted_score([0.0, 0.0, 1.0, 2.0, 1.0])
    assert 3.0 < s < 5.0


def test_numerically_stable_with_large_equal_logits():
    # Must not overflow; equal logits still average to 3 regardless of scale.
    assert weighted_score([1000.0, 1000.0, 1000.0, 1000.0, 1000.0]) == pytest.approx(3.0)


def test_wrong_length_raises():
    with pytest.raises(ValueError):
        weighted_score([0.0, 0.0, 0.0])


def test_non_finite_raises():
    with pytest.raises(ValueError):
        weighted_score([0.0, 0.0, float("inf"), 0.0, 0.0])


def test_axis_scores_maps_each_axis():
    out = axis_scores({
        "texture": [50.0, 0.0, 0.0, 0.0, 0.0],
        "geometry": [0.0, 0.0, 0.0, 0.0, 50.0],
        "semantic": [0.0, 0.0, 0.0, 0.0, 0.0],
    })
    assert out["texture"] == pytest.approx(1.0, abs=1e-6)
    assert out["geometry"] == pytest.approx(5.0, abs=1e-6)
    assert out["semantic"] == pytest.approx(3.0)
    assert set(out) == {"texture", "geometry", "semantic"}
