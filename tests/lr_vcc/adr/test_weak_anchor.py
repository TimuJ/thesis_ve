"""Task 4: weak-anchor fallback and anchor reliability."""
from scripts.lr_vcc.adr.annotations import Gap
from scripts.lr_vcc.adr.selection import PairChoice
from scripts.lr_vcc.adr.weak_anchor import anchor_reliability, select_with_fallback

PAIR = PairChoice(0, 2, 0.1, 1)


def _gap(dur):
    return Gap(0, 1, dur, (0, 0), (2, 2))


def test_first_strong_gap_accepted_full_reliability():
    gaps = [_gap(10.0), _gap(4.0)]
    r = select_with_fallback(gaps, choose_fn=lambda g: PAIR, c_gt_fn=lambda g, p: 0.9, floor=0.6)
    assert r.status == "ok" and r.reliability == 1.0 and r.gap is gaps[0]


def test_all_weak_keeps_best_with_reduced_reliability():
    gaps = [_gap(10.0), _gap(4.0)]
    cgt = {10.0: 0.3, 4.0: 0.5}
    r = select_with_fallback(gaps, lambda g: PAIR, lambda g, p: cgt[g.duration_s], floor=0.6)
    assert r.status == "weak" and r.gap is gaps[1] and 0.0 < r.reliability < 1.0


def test_no_admissible_pair_anywhere():
    r = select_with_fallback([_gap(10.0)], lambda g: PairChoice(None, None, float("inf"), 0),
                             lambda g, p: 0.0, floor=0.6)
    assert r.status == "no_pair" and r.pair is None


def test_anchor_reliability_monotonic_in_c_gt():
    assert anchor_reliability(0.9, 0.6) > anchor_reliability(0.5, 0.6) > anchor_reliability(0.1, 0.6)
