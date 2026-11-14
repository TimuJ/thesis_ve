"""Pure numeric core of the RoMa geometry axis: turning a dense match-certainty map
into a [0, 1] consistency / pose-match score (mean certainty) and a [0, 1] GT-side
availability signal (fraction of pixels RoMa can match at all). No torch -- this runs
on the M1; the heavy romatch adapter is contract-tested on the server.
"""
import numpy as np
from pytest import approx

from scripts.lr_vcc.adr.scorers_gpu import roma_match_coverage, roma_mean_certainty


def test_mean_certainty_is_the_clamped_mean():
    c = np.array([[0.0, 0.5], [1.0, 0.5]])
    assert roma_mean_certainty(c) == approx(0.5)


def test_mean_certainty_empty_is_zero():
    assert roma_mean_certainty(np.empty((0,))) == 0.0


def test_mean_certainty_clamped_to_unit_interval():
    # romatch certainty is probability-like, but guard against >1 / <0 leakage so the
    # consistency formula s_M = max(0, 1 - |c_sr - c_gt|) stays well-defined.
    assert roma_mean_certainty(np.array([2.0, 2.0])) == approx(1.0)
    assert roma_mean_certainty(np.array([-1.0, -1.0])) == 0.0


def test_match_coverage_is_fraction_above_floor():
    c = np.array([0.1, 0.3, 0.9, 0.95])          # 3 of 4 are >= 0.2
    assert roma_match_coverage(c, floor=0.2) == approx(0.75)


def test_match_coverage_flattens_2d():
    c = np.array([[0.1, 0.9], [0.9, 0.9]])        # 3 of 4 are >= 0.2
    assert roma_match_coverage(c, floor=0.2) == approx(0.75)


def test_match_coverage_empty_is_zero():
    assert roma_match_coverage(np.empty((0,)), floor=0.2) == 0.0


def test_match_coverage_none_above_floor_is_zero():
    # a GT pair RoMa cannot match anywhere carries no geometry signal -> reliability 0,
    # so the axis is down-weighted instead of free-passing on c_gt ~ 0.
    assert roma_match_coverage(np.array([0.0, 0.1, 0.05]), floor=0.2) == 0.0
