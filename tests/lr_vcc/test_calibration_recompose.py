import math

import pytest

from scripts.lr_vcc.calibration import recompose as R
from scripts.lr_vcc.calibration import response_table as RT
from scripts.lr_vcc.calibration import expectations as E
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
    """exp(-x) ~= 1 - x over the observed tOF range [0.04, 0.17]."""
    for row in table["artefacts"][:40]:
        lin = [s for n, s, _ in
               R.sub_metric_values(row, dict(R.PROD_PARAMS, beta_t=None))
               if n == "temporal"][0]
        exp1 = [s for n, s, _ in
                R.sub_metric_values(row, dict(R.PROD_PARAMS, beta_t=1.0))
                if n == "temporal"][0]
        assert abs(lin - exp1) < 0.02


def test_low_confidence_flag(table):
    row = table["artefacts"][0]
    out = R.composite(row, R.PROD_PARAMS)
    assert out["low_confidence"] == all(r < 0.2 for r in out["reliabilities"])
