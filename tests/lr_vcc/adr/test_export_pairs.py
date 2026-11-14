"""Pure helpers for the review gallery: the Keep/Exclude reason from a pair's
reliability factors, and the neighbour frames shown in the pose-stability strip."""
from pytest import approx

from scripts.lr_vcc.adr.export_pairs import (
    consistency_band,
    exclusion_reasons,
    pct_gain,
    strip_indices,
    video_summary,
)


def test_video_summary_surfaces_worst_pair():
    # 5 pairs, one collapsed: the summary must surface the worst/min pair and the
    # failure count, not let a mean hide the single failure.
    pairs = [
        {"i": 1, "j": 10, "consistency": 0.97},
        {"i": 2, "j": 20, "consistency": 0.30},   # the one failure
        {"i": 3, "j": 30, "consistency": 0.95},
        {"i": 4, "j": 40, "consistency": 0.98},
        {"i": 5, "j": 50, "consistency": 0.96},
    ]
    s = video_summary(pairs)
    assert s["n_pairs"] == 5
    assert s["min_consistency"] == approx(0.30)
    assert s["worst_pair"] == (2, 20)
    assert s["n_failures"] == 1                    # only the 0.30 pair < 0.90
    assert s["mean_consistency"] > 0.30            # mean alone would hide the failure


def test_pct_gain_negative_when_sr_drifts_below_gt():
    # SR less self-similar across the gap than GT -> drift -> negative %gain.
    assert pct_gain(0.80, 0.60) == approx(-25.0)


def test_pct_gain_positive_when_sr_over_consistent():
    # SR MORE self-similar than GT: positive number, but this is fabricated
    # over-consistency, not an improvement -- hence %gain is reference-only.
    assert pct_gain(0.50, 0.75) == approx(50.0)


def test_pct_gain_na_when_gt_in_the_noise():
    # SIFT on a low-texture object (leopard/fox): c_gt ~ 0 -> ratio meaningless -> n/a.
    assert pct_gain(0.003, 0.006) is None
    assert pct_gain(0.04, 0.20) is None          # just under the 0.05 floor
    assert pct_gain(0.06, 0.06) == approx(0.0)   # just over the floor: defined (no change)


def test_pct_gain_na_on_missing_axis():
    assert pct_gain(None, 0.5) is None
    assert pct_gain(0.5, None) is None


def test_consistency_band_splits_failures_from_preserved():
    assert consistency_band(0.80, 0.90) == "diverges"
    assert consistency_band(0.95, 0.90) == "preserves"
    assert consistency_band(0.90, 0.90) == "preserves"   # boundary counts as preserved


def test_exclusion_reasons_low_sharpness_alone_does_not_exclude():
    # Laplacian sharpness proved an unreliable proxy for the reviewer's "motion blur"
    # judgment (the flagged-blurry crops had the HIGHEST variance), so a low sharp
    # factor never excludes on its own -- motion blur stays an eye-call.
    assert exclusion_reasons({"size": 1.0, "sharp": 0.2, "edge": 1.0, "anchor": 1.0}) == []


def test_exclusion_reasons_flags_frame_edge():
    r = exclusion_reasons({"size": 1.0, "sharp": 1.0, "edge": 0.2, "anchor": 1.0})
    assert any("frame edge" in x for x in r)


def test_exclusion_reasons_flags_weak_anchor():
    r = exclusion_reasons({"size": 1.0, "sharp": 1.0, "edge": 1.0, "anchor": 0.1})
    assert any("anchor" in x for x in r)


def test_exclusion_reasons_very_small_object_excluded_but_moderate_kept():
    # a truly tiny object (area well under the floor) can't be reviewed -> excluded;
    # a small-but-reviewable object (the gecko, the readable plate) is kept.
    assert any("too small" in x for x in exclusion_reasons(
        {"size": 0.2, "sharp": 1.0, "edge": 1.0, "anchor": 1.0}))
    assert exclusion_reasons({"size": 0.5, "sharp": 1.0, "edge": 1.0, "anchor": 1.0}) == []


def test_exclusion_reasons_empty_when_clean():
    assert exclusion_reasons({"size": 1.0, "sharp": 1.0, "edge": 1.0, "anchor": 1.0}) == []


def test_strip_indices_clamped_to_bounds():
    assert strip_indices(0, lo=0, hi=9, radius=2) == [0, 1, 2]        # clamped at the start
    assert strip_indices(9, lo=0, hi=9, radius=2) == [7, 8, 9]        # clamped at the end
    assert strip_indices(5, lo=0, hi=9, radius=1) == [4, 5, 6]        # centred run
