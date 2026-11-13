"""Task 4: weak-anchor fallback and anchor reliability."""
from scripts.lr_vcc.adr.annotations import Gap
from scripts.lr_vcc.adr.selection import PairChoice
from scripts.lr_vcc.adr.weak_anchor import (
    anchor_reliability,
    select_all_gaps,
    select_with_fallback,
    size_reliability,
)

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


def test_select_all_gaps_one_entry_per_admissible_gap():
    gaps = [_gap(10.0), _gap(4.0)]
    res = select_all_gaps(gaps, lambda g: PAIR, lambda g, p: 0.9, floor=0.6)
    assert [r.gap for r in res] == gaps
    assert [r.status for r in res] == ["ok", "ok"]
    assert all(r.reliability == 1.0 for r in res)


def test_select_all_gaps_skips_gaps_without_a_pair_and_marks_weak():
    gaps = [_gap(10.0), _gap(6.0), _gap(4.0)]
    # gap 10: no admissible pair -> skipped; gap 6: weak (below floor); gap 4: ok
    def choose(g):
        return PairChoice(None, None, float("inf"), 0) if g.duration_s == 10.0 else PAIR
    cgt = {6.0: 0.3, 4.0: 0.8}
    res = select_all_gaps(gaps, choose, lambda g, p: cgt[g.duration_s], floor=0.6)
    assert [r.gap.duration_s for r in res] == [6.0, 4.0]
    assert [r.status for r in res] == ["weak", "ok"]
    assert 0.0 < res[0].reliability < 1.0 and res[1].reliability == 1.0


def test_size_reliability_saturates_and_is_monotonic():
    assert size_reliability(0.10, 0.05) == 1.0        # big object -> full
    assert size_reliability(0.05, 0.05) == 1.0        # at the threshold -> full
    assert 0.0 < size_reliability(0.01, 0.05) < size_reliability(0.03, 0.05) < 1.0
    assert size_reliability(0.0, 0.05) == 0.0


def test_select_all_gaps_empty_when_no_pairs():
    res = select_all_gaps([_gap(10.0)], lambda g: PairChoice(None, None, float("inf"), 0),
                          lambda g, p: 0.0, floor=0.6)
    assert res == []
