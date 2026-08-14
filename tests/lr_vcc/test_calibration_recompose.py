import pytest

from scripts.lr_vcc.calibration import recompose as R
from scripts.lr_vcc.calibration import response_table as RT
from scripts.lr_vcc.calibration import expectations as E
from scripts.lr_vcc.color_stability import color_stability_score
from scripts.lr_vcc.sweep_sensitivity import (
    PROD, artefact_units, compose_unit, realmodel_units,
)


@pytest.fixture(scope="module")
def table():
    return RT.build_table()


def test_bit_exact_vs_evaluate_one_video_on_artefacts(table):
    by_unit = {}
    for r in table["artefacts"]:
        by_unit.setdefault(r["unit"], []).append(r)
    checked = 0
    for unit in artefact_units():
        ref = compose_unit(unit, PROD, full=True)
        for row in by_unit[unit[0]]:
            got = R.composite(row, R.PROD_PARAMS)["lr_vcc"]
            assert abs(got - ref[row["clip"]]["lr_vcc"]) < 1e-12, row["clip"]
            checked += 1
    assert checked == 300


def test_bit_exact_vs_evaluate_one_video_on_realmodels(table):
    units = realmodel_units(identity_variant="corrected", closeup=True,
                            methods=("mgld", "uav", "flashvsr"))
    by_unit = {}
    for r in table["realmodels"]:
        by_unit.setdefault(r["unit"], []).append(r)
    checked = 0
    for unit in units:
        ref = compose_unit(unit, PROD, full=True)
        for row in by_unit[unit[0]]:
            got = R.composite(row, R.PROD_PARAMS)["lr_vcc"]
            assert abs(got - ref[row["clip"]]["lr_vcc"]) < 1e-12, row["clip"]
            checked += 1
    assert checked == 15


def test_submetric_order_is_canonical(table):
    names = [n for n, _, _ in R.sub_metric_values(table["artefacts"][0],
                                                  R.PROD_PARAMS)]
    assert tuple(names) == E.SUB_METRICS


def test_beta_t_is_monotone_decreasing_in_tof(table):
    row = dict(table["artefacts"][0])
    p = dict(R.PROD_PARAMS, beta_t=10.0)
    prev = None
    for scale in (1.0, 1.5, 2.0, 3.0):
        r = dict(row, tof={k: (None if v is None else v * scale)
                           for k, v in row["tof"].items()})
        t = [s for n, s, _ in R.sub_metric_values(r, p) if n == "temporal"][0]
        if prev is not None:
            assert t < prev
        prev = t


def test_beta_t_one_approximates_the_linear_v5_form(table):
    """exp(-x) ~= 1 - x over the observed tOF range [0.0224, 0.1594].

    Sampled a few rows per artefact family (not just the first 40 rows,
    which — since the table is laid out as contiguous 25-row blocks, one per
    family — would span only 2 of the 12 families).
    """
    by_unit = {}
    for row in table["artefacts"]:
        by_unit.setdefault(row["unit"], []).append(row)
    sample = [row for rows in by_unit.values() for row in rows[:4]]
    assert len({row["unit"] for row in sample}) == 12
    for row in sample:
        lin = [s for n, s, _ in
               R.sub_metric_values(row, dict(R.PROD_PARAMS, beta_t=None))
               if n == "temporal"][0]
        exp1 = [s for n, s, _ in
                R.sub_metric_values(row, dict(R.PROD_PARAMS, beta_t=1.0))
                if n == "temporal"][0]
        assert abs(lin - exp1) < 0.02


def test_low_confidence_flag(table):
    row = table["artefacts"][0]
    # Assert the wiring, not the identity: composite's low_confidence must
    # actually track low_confidence_floor, not just echo whatever expression
    # was used to compute it (every row in the corpus has a reliability of
    # exactly 1.0 somewhere, so a fixed floor of 0.2 is always False on both
    # sides and would pass even if low_confidence were hardcoded False).
    assert R.composite(row, dict(R.PROD_PARAMS,
                                 low_confidence_floor=1.01))["low_confidence"] is True
    assert R.composite(row, dict(R.PROD_PARAMS,
                                 low_confidence_floor=0.0))["low_confidence"] is False
    out = R.composite(row, R.PROD_PARAMS)
    assert out["low_confidence"] == all(r < 0.2 for r in out["reliabilities"])


def test_color_stability_reliability_matches_canonical_at_low_frame_count(table):
    """Pins D's reliability against color_stability_score at a frame count
    (120, below the 240 floor) where sharpness=0.02 and the default
    sharpness=10.0 diverge — the bit-exactness gate can't catch a regression
    to the wrong sharpness because every real row has hist_n_frames >> 240,
    where both sharpness values round to a reliability of 1.0.
    """
    row = dict(table["artefacts"][0], hist_n_frames=120)
    d_score, d_rel = [(s, r) for n, s, r in
                      R.sub_metric_values(row, R.PROD_PARAMS)
                      if n == "color_stability"][0]
    ref = color_stability_score(
        {"n_frames": 120, "mean_hist_dist": row["hist_dist"]},
        alpha=R.PROD_PARAMS["alpha"])
    assert abs(d_score - ref["score"]) < 1e-12
    assert abs(d_rel - ref["reliability"]) < 1e-12
    # Confirm this row actually exercises the divergent regime (reliability
    # meaningfully below 1.0, unlike every real row in the corpus).
    assert d_rel < 0.5
