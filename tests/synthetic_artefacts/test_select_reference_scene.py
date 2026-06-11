import numpy as np
import pytest

from scripts.synthetic_artefacts.select_reference_scene import cosine_distance, pick_most_distant


def test_cosine_distance_orthogonal_is_one():
    assert abs(cosine_distance(np.array([1.0, 0.0]), np.array([0.0, 1.0])) - 1.0) < 1e-6


def test_cosine_distance_identical_is_zero():
    v = np.array([0.3, 0.7, 0.1])
    assert cosine_distance(v, v) < 1e-6


def test_pick_most_distant_returns_farthest():
    base = np.array([1.0, 0.0])
    cands = {"near": np.array([0.9, 0.1]), "far": np.array([-1.0, 0.2])}
    name, dist = pick_most_distant(cands, base, tau=0.5)
    assert name == "far" and dist > 1.0


def test_pick_most_distant_raises_below_tau():
    base = np.array([1.0, 0.0])
    cands = {"near": np.array([0.99, 0.01])}
    with pytest.raises(ValueError):
        pick_most_distant(cands, base, tau=0.5)
