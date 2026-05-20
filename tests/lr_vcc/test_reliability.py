import math
from scripts.lr_vcc.reliability import below_threshold_penalty, above_threshold_penalty


def test_below_threshold_penalty_at_threshold_is_half():
    # exactly at threshold -> sigmoid(0) = 0.5
    assert abs(below_threshold_penalty(value=0.10, threshold=0.10, sharpness=10) - 0.5) < 1e-6


def test_below_threshold_penalty_well_above_is_near_zero():
    # far above threshold -> sigmoid(-large) -> ~0 penalty
    assert below_threshold_penalty(value=0.50, threshold=0.10, sharpness=10) < 0.05


def test_below_threshold_penalty_well_below_is_near_one():
    # far below threshold -> sigmoid(+large) -> ~1 penalty
    assert below_threshold_penalty(value=0.01, threshold=0.10, sharpness=10) > 0.6


def test_above_threshold_penalty_symmetric():
    # mirror semantics: high value -> high penalty (used for saturation, close-up flags)
    assert abs(above_threshold_penalty(value=0.10, threshold=0.10, sharpness=10) - 0.5) < 1e-6
    assert above_threshold_penalty(value=0.50, threshold=0.10, sharpness=10) > 0.6
    assert above_threshold_penalty(value=0.01, threshold=0.10, sharpness=10) < 0.3
