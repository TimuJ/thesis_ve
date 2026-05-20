"""Smooth (sigmoid) reliability penalties around documented thresholds.

A penalty in [0, 1] is converted to a reliability via `reliability = 1 - penalty`
in the calling sub-metric. We keep penalties separate so callers can combine
multiple penalties (max, sum, weighted) before forming the reliability.
"""
import math


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ez = math.exp(x)
    return ez / (1.0 + ez)


def below_threshold_penalty(value: float, threshold: float, sharpness: float = 10.0) -> float:
    """Returns ~1 when value << threshold, ~0 when value >> threshold.

    Used when LOW values indicate a bad regime (e.g. mask coverage too low).
    """
    return _sigmoid(sharpness * (threshold - value))


def above_threshold_penalty(value: float, threshold: float, sharpness: float = 10.0) -> float:
    """Returns ~1 when value >> threshold, ~0 when value << threshold.

    Used when HIGH values indicate a bad regime (e.g. saturation, close-up ratio).
    """
    return _sigmoid(sharpness * (value - threshold))
