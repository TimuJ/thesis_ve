"""Task 3: window bounding, selection cost, pair choice."""
import numpy as np

from scripts.lr_vcc.adr.annotations import Gap
from scripts.lr_vcc.adr.selection import bound_windows, selection_cost, choose_pair


def test_cost_zero_for_identical():
    e = np.array([1.0, 0.0])
    assert selection_cost(e, e, 0.2, 0.2, 1.0, 1.0, 1.0, 1.0) == 0.0


def test_cost_rises_with_area_and_aspect_mismatch():
    e = np.array([1.0, 0.0])
    base = selection_cost(e, e, 0.2, 0.2, 1.0, 1.0, 1.0, 1.0)
    assert selection_cost(e, e, 0.4, 0.1, 1.0, 1.0, 1.0, 1.0) > base
    assert selection_cost(e, e, 0.2, 0.2, 2.0, 0.5, 1.0, 1.0) > base


def test_choose_pair_picks_min_cost_admissible():
    emb = np.array([[1, 0], [0, 1], [1, 0]], float)
    areas = np.array([.3, .3, .3]); asp = np.ones(3)
    c = choose_pair([0, 1], [2], emb, areas, asp, a_min=0.1, lambda_area=1.0, lambda_aspect=1.0)
    assert (c.i, c.j) == (0, 2)              # 0 matches 2 exactly; 1 is orthogonal


def test_choose_pair_none_when_all_below_a_min_one_side():
    emb = np.zeros((3, 2)); areas = np.array([.3, .3, .005]); asp = np.ones(3)
    c = choose_pair([0, 1], [2], emb, areas, asp, a_min=0.01, lambda_area=1.0, lambda_aspect=1.0)
    assert c.i is None and c.j is None and c.n_admissible == 0


def test_bound_windows_truncates_to_offset():
    present = np.array([1, 1, 1, 1, 0, 0, 0, 1, 1, 1], bool)  # gap 4..6
    pre, post = bound_windows(present, Gap(4, 6, 3.0, (0, 3), (7, 9)), max_offset=2)
    assert pre == [2, 3] and post == [7, 8]
