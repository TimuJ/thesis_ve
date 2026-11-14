"""Task 3: GT-based per-axis reliability feeding the composite."""
import numpy as np

from scripts.lr_vcc.adr.reliability_axes import axis_reliabilities
from scripts.lr_vcc.composite import compose_score


class _S:
    def __init__(self, name, rel):
        self.name = name
        self._rel = rel

    def __call__(self, a, b):
        return 0.9

    def reliability(self, a, b):
        return self._rel


class _NoRel:
    name = "plain"

    def __call__(self, a, b):
        return 0.9


def test_axis_reliabilities_reads_each_scorers_signal():
    crop = np.zeros((8, 8, 3), np.uint8)
    rel = axis_reliabilities([_S("sift", 0.2), _S("dino", 1.0)], crop, crop)
    assert rel == {"sift": 0.2, "dino": 1.0}


def test_scorer_without_reliability_defaults_one():
    crop = np.zeros((8, 8, 3), np.uint8)
    rel = axis_reliabilities([_NoRel()], crop, crop)
    assert rel == {"plain": 1.0}


def test_zero_signal_axis_gets_low_weight_in_composite():
    crop = np.zeros((8, 8, 3), np.uint8)
    scorers = [_S("sift", 1.0), _S("dino", 1.0), _S("roma", 0.0)]
    rel = axis_reliabilities(scorers, crop, crop)
    names = ["sift", "dino", "roma"]
    per_axis = [1.0 * rel[n] for n in names]
    comp = compose_score([0.9, 0.9, 0.1], per_axis)
    w = comp["weights"]
    assert w[2] < w[0] / 10 and w[2] < w[1] / 10       # no-signal axis gets ~0 weight
