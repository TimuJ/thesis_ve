"""Task 3: window bounding, selection cost, pair choice."""
import numpy as np

from scripts.lr_vcc.adr.annotations import Gap
from scripts.lr_vcc.adr.selection import (
    bound_windows,
    choose_pair,
    global_big_windows,
    selection_cost,
)


def test_cost_zero_for_identical():
    e = np.array([1.0, 0.0])
    assert selection_cost(e, e, 0.2, 0.2, 1.0, 1.0, 1.0, 1.0) == 0.0


def test_size_term_off_by_default_and_rewards_larger_when_on():
    e = np.array([1.0, 0.0])
    assert selection_cost(e, e, 0.2, 0.2, 1.0, 1.0, 1.0, 1.0) == 0.0  # lambda_size default 0
    big = selection_cost(e, e, 0.40, 0.40, 1.0, 1.0, 1.0, 1.0, lambda_size=0.5)
    small = selection_cost(e, e, 0.05, 0.05, 1.0, 1.0, 1.0, 1.0, lambda_size=0.5)
    assert small > big  # a smaller (equally-matched) pair costs more when size-weighted


def test_choose_pair_prefers_larger_equally_matched_pair():
    emb = np.array([[1, 0], [1, 0], [1, 0]], float)   # all identical appearance
    areas = np.array([0.3, 0.05, 0.3]); asp = np.ones(3)
    c = choose_pair([0], [1, 2], emb, areas, asp, a_min=0.01,
                    lambda_area=0.0, lambda_aspect=0.0, lambda_size=0.5)
    assert c.j == 2   # idx2 (0.3) is larger than idx1 (0.05), equally matched to idx0


def test_global_big_windows_picks_biggest_in_each_half():
    present = np.ones(10, bool)
    areas = np.array([0.1, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.6, 0.1, 0.1])  # big at 1 and 7
    early, late = global_big_windows(present, areas, topk=2)
    assert 1 in early and 7 in late
    assert all(i < 5 for i in early) and all(j >= 5 for j in late)


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


def test_choose_pair_respects_min_sep():
    # All frames look identical, so cost ties and the FIRST pair in iteration order
    # wins -> the temporally adjacent pair. min_sep must exclude the too-close pair
    # and pick a separated one instead (fixes a long-range object biggest near mid).
    emb = np.tile([1.0, 0.0], (21, 1)); areas = np.full(21, 0.3); asp = np.ones(21)
    near = choose_pair([9, 0], [10, 20], emb, areas, asp, a_min=0.01,
                       lambda_area=0.0, lambda_aspect=0.0, lambda_size=0.0)
    assert abs(near.i - near.j) == 1                      # adjacent pair wins without min_sep
    far = choose_pair([9, 0], [10, 20], emb, areas, asp, a_min=0.01,
                      lambda_area=0.0, lambda_aspect=0.0, lambda_size=0.0, min_sep=3)
    assert far.i is not None and abs(far.i - far.j) >= 3  # min_sep forces separation


def test_bound_windows_truncates_to_offset():
    present = np.array([1, 1, 1, 1, 0, 0, 0, 1, 1, 1], bool)  # gap 4..6
    pre, post = bound_windows(present, Gap(4, 6, 3.0, (0, 3), (7, 9)), max_offset=2)
    assert pre == [2, 3] and post == [7, 8]


def test_appearance_weight_amplifies_pose_distance():
    # lambda_appear scales the embedding cos-distance term, so a pose-mismatched
    # pair costs more as the weight rises (fixes "different angles" selections).
    a = np.array([1.0, 0.0]); b = np.array([0.0, 1.0])  # orthogonal -> cos-dist 1
    assert selection_cost(a, b, 0.2, 0.2, 1.0, 1.0, 0.0, 0.0, lambda_appear=3.0) == 3.0
    assert selection_cost(a, b, 0.2, 0.2, 1.0, 1.0, 0.0, 0.0, lambda_appear=1.0) == 1.0


def test_choose_pair_appearance_gate_rejects_dissimilar_even_when_bigger():
    # idx2 is BIGGER (size reward picks it ungated) but points in a slightly
    # different direction (cos-dist 0.1, a different pose); a gate below that
    # distance must reject it and fall back to the same-pose idx1.
    emb = np.array([[1.0, 0.0], [1.0, 0.0], [0.9, 0.4359]])  # idx2: cos-dist ~0.1 from idx0
    areas = np.array([0.30, 0.05, 0.30]); asp = np.ones(3)
    ungated = choose_pair([0], [1, 2], emb, areas, asp, a_min=0.01,
                          lambda_area=0.0, lambda_aspect=0.0, lambda_size=0.5)
    assert ungated.j == 2                     # bigger idx2 wins without the gate
    gated = choose_pair([0], [1, 2], emb, areas, asp, a_min=0.01,
                        lambda_area=0.0, lambda_aspect=0.0, lambda_size=0.5,
                        max_appearance_dist=0.05)
    assert gated.j == 1                       # dissimilar-but-bigger idx2 rejected


def test_choose_pair_gate_returns_none_when_all_pairs_dissimilar():
    emb = np.array([[1, 0], [0, 1]], float)
    areas = np.array([0.3, 0.3]); asp = np.ones(2)
    c = choose_pair([0], [1], emb, areas, asp, a_min=0.01, max_appearance_dist=0.5)
    assert c.i is None and c.n_admissible == 0


def test_choose_pair_blur_penalty_avoids_softer_frame():
    # idx2 is slightly bigger (size reward favors it) but blurred; the blur penalty
    # must outweigh the size reward and pick the sharp idx1.
    emb = np.array([[1, 0], [1, 0], [1, 0]], float)   # identical appearance
    areas = np.array([0.30, 0.05, 0.06]); asp = np.ones(3)  # idx0 big so the post side drives min-area
    blur = np.array([0.0, 0.0, 0.9])                  # idx2 is soft
    sharp_pick = choose_pair([0], [1, 2], emb, areas, asp, a_min=0.01,
                             lambda_area=0.0, lambda_aspect=0.0, lambda_size=0.5,
                             blur_pen=blur, lambda_blur=1.0)
    assert sharp_pick.j == 1
    no_blur = choose_pair([0], [1, 2], emb, areas, asp, a_min=0.01,
                          lambda_area=0.0, lambda_aspect=0.0, lambda_size=0.5,
                          blur_pen=blur, lambda_blur=0.0)
    assert no_blur.j == 2                             # without the penalty, size wins idx2
