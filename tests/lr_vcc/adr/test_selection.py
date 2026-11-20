"""Task 3: window bounding, selection cost, pair choice."""
import numpy as np

from scripts.lr_vcc.adr.annotations import Gap
from scripts.lr_vcc.adr.selection import (
    _pairs_too_close,
    bound_windows,
    choose_pair,
    choose_pairs,
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


# --- Task 1: choose_pairs (top-K diverse selection) ---------------------------

def test_pairs_too_close_requires_both_endpoints_within_min_sep():
    # too-close iff BOTH endpoints differ by < min_sep.
    assert _pairs_too_close(4, 14, 5, 15, min_sep=10) is True     # 1 and 1, both < 10
    assert _pairs_too_close(0, 50, 10, 50, min_sep=10) is False   # pre differs by 10 (not < 10)
    assert _pairs_too_close(0, 50, 0, 60, min_sep=10) is False    # post differs by 10
    assert _pairs_too_close(0, 50, 9, 59, min_sep=10) is True     # 9 and 9, both < 10


def test_choose_pairs_returns_topk_sorted():
    emb = np.tile([1.0, 0.0], (7, 1))                     # identical appearance
    areas = np.array([0.3, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05])  # idx0 pre; 1..6 posts
    asp = np.ones(7)
    res = choose_pairs([0], [1, 2, 3, 4, 5, 6], emb, areas, asp,
                       top_k=3, min_pair_sep=0, a_min=0.01,
                       lambda_area=1.0, lambda_aspect=0.0, lambda_appear=0.0)
    assert len(res) == 3
    assert [p.j for p in res] == [1, 2, 3]               # 3 lowest area-mismatch posts
    costs = [p.cost for p in res]
    assert costs == sorted(costs)                        # non-decreasing


def test_choose_pairs_diversity_collapses_and_keeps_threshold_apart():
    # Collapse: four near-duplicate pairs (all within < min_sep on both ends) -> 1 kept.
    emb = np.tile([1.0, 0.0], (23, 1)); areas = np.full(23, 0.3); asp = np.ones(23)
    collapsed = choose_pairs([0, 2], [20, 22], emb, areas, asp,
                             top_k=5, min_pair_sep=10, a_min=0.01,
                             lambda_area=0.0, lambda_aspect=0.0, lambda_appear=0.0)
    assert len(collapsed) == 1

    # Survive: pairs exactly min_sep apart on both ends are NOT too close -> all kept.
    emb2 = np.tile([1.0, 0.0], (31, 1)); areas2 = np.full(31, 0.3); asp2 = np.ones(31)
    kept = choose_pairs([0, 10], [20, 30], emb2, areas2, asp2,
                        top_k=5, min_pair_sep=10, a_min=0.01,
                        lambda_area=0.0, lambda_aspect=0.0, lambda_appear=0.0)
    ij = {(p.i, p.j) for p in kept}
    assert len(kept) == 4
    assert (0, 20) in ij and (10, 30) in ij             # diagonal, 10 apart on both


def test_choose_pairs_pose_gate_filters():
    emb = np.tile([1.0, 0.0], (3, 1)); areas = np.full(3, 0.3); asp = np.ones(3)
    gate = lambda i, j: not (i == 0 and j == 2)          # reject exactly (0, 2)
    res = choose_pairs([0], [1, 2], emb, areas, asp, top_k=5, min_pair_sep=0,
                       a_min=0.01, pose_gate_fn=gate)
    ij = {(p.i, p.j) for p in res}
    assert (0, 2) not in ij and (0, 1) in ij


def test_choose_pairs_empty_when_none_admissible():
    emb = np.tile([1.0, 0.0], (2, 1)); areas = np.array([0.3, 0.005]); asp = np.ones(2)
    res = choose_pairs([0], [1], emb, areas, asp, top_k=5, min_pair_sep=0, a_min=0.01)
    assert res == []


# --- pose-gate budget (cap the expensive RoMa verification) -------------------

def test_choose_pairs_pose_budget_caps_gate_calls():
    # Many admissible candidates, gate rejects every one. Without a budget the gate
    # (RoMa, one forward pass each) runs on all of them -- the grind we observed.
    # pose_budget must cap how many candidates are geometry-verified.
    n = 30
    emb = np.tile([1.0, 0.0], (n, 1)); areas = np.full(n, 0.3); asp = np.ones(n)
    posts = list(range(1, n))                       # 29 admissible candidate posts
    calls = []
    gate = lambda i, j: calls.append((i, j)) or False   # reject all, count the calls
    res = choose_pairs([0], posts, emb, areas, asp, top_k=5, min_pair_sep=0,
                       a_min=0.01, pose_gate_fn=gate, pose_budget=10)
    assert len(calls) == 10        # verified at most the budget, not all 29
    assert res == []               # nothing passed the gate


def test_choose_pairs_pose_budget_stops_early_at_topk():
    # When the gate passes, the top_k early-stop still wins and the budget is not
    # spent down -- we verify only as many candidates as we need to fill top_k.
    n = 30
    emb = np.tile([1.0, 0.0], (n, 1)); areas = np.full(n, 0.3); asp = np.ones(n)
    posts = list(range(1, n))
    calls = []
    gate = lambda i, j: (calls.append((i, j)) or True)  # accept all, count the calls
    res = choose_pairs([0], posts, emb, areas, asp, top_k=5, min_pair_sep=0,
                       a_min=0.01, pose_gate_fn=gate, pose_budget=20)
    assert len(res) == 5           # top_k reached
    assert len(calls) == 5         # stopped at top_k, budget barely touched


# --- backfill: always hand back top_k candidates for manual review ------------

def test_choose_pairs_backfills_to_topk_when_pose_gate_rejects_all():
    # Manual-filter workflow: even if nothing passes the (strict) pose gate, we
    # still want top_k best-by-cost diverse candidates to review, not an empty set.
    n = 20
    emb = np.tile([1.0, 0.0], (n, 1)); areas = np.full(n, 0.3); asp = np.ones(n)
    posts = list(range(1, n))
    gate = lambda i, j: False                    # pose gate rejects everything
    res = choose_pairs([0], posts, emb, areas, asp, top_k=5, min_pair_sep=0,
                       a_min=0.01, pose_gate_fn=gate, pose_budget=50, backfill=True)
    assert len(res) == 5


def test_choose_pairs_no_backfill_still_returns_empty_when_all_rejected():
    # Default (backfill=False) keeps the strict hard-gate behaviour.
    n = 20
    emb = np.tile([1.0, 0.0], (n, 1)); areas = np.full(n, 0.3); asp = np.ones(n)
    posts = list(range(1, n))
    gate = lambda i, j: False
    res = choose_pairs([0], posts, emb, areas, asp, top_k=5, min_pair_sep=0,
                       a_min=0.01, pose_gate_fn=gate, backfill=False)
    assert res == []


def test_choose_pairs_backfill_ranks_pose_passers_first():
    # A pair that passes the gate must rank ahead of backfilled (unverified) ones.
    n = 20
    emb = np.tile([1.0, 0.0], (n, 1)); areas = np.full(n, 0.3); asp = np.ones(n)
    posts = list(range(1, n))
    gate = lambda i, j: (i == 0 and j == 7)      # only (0, 7) passes
    res = choose_pairs([0], posts, emb, areas, asp, top_k=5, min_pair_sep=0,
                       a_min=0.01, pose_gate_fn=gate, pose_budget=50, backfill=True)
    assert (res[0].i, res[0].j) == (0, 7)        # the passer is first
    assert len(res) == 5                          # the rest are backfilled
