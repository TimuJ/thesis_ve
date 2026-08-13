import math

import pytest

from scripts.lr_vcc.calibration import objective as O
from scripts.lr_vcc.calibration import recompose as R
from scripts.lr_vcc.calibration import response_table as RT

FALLING = {"0p02": 0.90, "0p05": 0.85, "0p10": 0.80, "0p20": 0.75, "0p40": 0.70}
FLAT = {s: 0.80 for s in FALLING}
BUMPY = {"0p02": 0.90, "0p05": 0.95, "0p10": 0.80, "0p20": 0.85, "0p40": 0.70}


def test_response_is_negative_delta():
    assert O.response(FALLING) == pytest.approx(0.20)
    assert O.response(FLAT) == pytest.approx(0.0)


def test_monotonicity_violation_counts_upward_steps_only():
    assert O.monotonicity_violation(FALLING) == pytest.approx(0.0)
    assert O.monotonicity_violation(BUMPY) == pytest.approx(0.10)


def test_respond_cell_is_penalised_for_under_responding():
    cfg = O.LOSS_CFG
    assert O.cell_loss("flicker", FLAT, cfg) > 0
    assert O.cell_loss("flicker", FALLING, cfg) == pytest.approx(0.0)


def test_silent_cell_is_penalised_for_responding():
    cfg = O.LOSS_CFG
    assert O.cell_loss("flip_horizontal", FLAT, cfg) == pytest.approx(0.0)
    assert O.cell_loss("flip_horizontal", FALLING, cfg) > 0


def test_silence_penalty_is_asymmetric():
    """w_silence actually scales the SILENT-cell penalty, by exactly its ratio.

    (The previous version of this test compared a RESPOND cell against a
    SILENT cell using the same w_silence=3.0 on both sides — the RESPOND
    branch never reads w_silence, so the comparison held for any weight,
    including one that made the "asymmetric" penalty a no-op.)
    """
    over = {"0p02": 0.90, "0p05": 0.88, "0p10": 0.86, "0p20": 0.84, "0p40": 0.78}
    heavy = O.cell_loss("flip_horizontal", over, dict(O.LOSS_CFG, w_silence=3.0))
    light = O.cell_loss("flip_horizontal", over, dict(O.LOSS_CFG, w_silence=1.0))
    assert light > 0
    assert heavy > light
    assert heavy / light == pytest.approx(3.0)


def test_silent_cell_penalised_for_interior_oscillation_returning_to_baseline():
    """A control that swings mid-ladder and returns to baseline must still
    be penalised — an endpoint-only read would score this ladder 0.0 despite
    a 0.35 interior excursion, which is exactly the unwanted sensitivity the
    silence penalty exists to catch.
    """
    oscillating = {"0p02": 0.80, "0p05": 0.95, "0p10": 0.60,
                  "0p20": 0.95, "0p40": 0.79}
    assert O.response(oscillating) == pytest.approx(0.01)
    assert O.cell_loss("flip_horizontal", oscillating, O.LOSS_CFG) > 0


def test_unconstrained_family_contributes_nothing():
    assert O.cell_loss("flip_transpose", FALLING, O.LOSS_CFG) == 0.0
    assert O.cell_loss("flip_transpose", FLAT, O.LOSS_CFG) == 0.0


def test_matrix_scores_covers_sixty_cells():
    rows = RT.build_table()["artefacts"]
    scored = O.matrix_scores(rows, R.PROD_PARAMS)
    assert len(scored) == 60
    cell = scored[("flicker", "7WHI2L_FDNg")]
    assert cell["verdict"] == "FLAT"
    assert cell["delta"] == pytest.approx(-0.001, abs=5e-3)


def test_matrix_loss_respects_base_subset():
    rows = RT.build_table()["artefacts"]
    four = [b for b in ("7WHI2L_FDNg", "BrRLKMbBTYQ", "KZ8p6b1zJ9U",
                        "hhszUXL1Cu8")]
    assert len(O.matrix_scores(rows, R.PROD_PARAMS, bases=four)) == 48

    full_loss = O.matrix_loss(rows, R.PROD_PARAMS)
    assert math.isfinite(full_loss)
    assert full_loss > 0

    subset_loss = O.matrix_loss(rows, R.PROD_PARAMS, bases=four)
    assert math.isfinite(subset_loss)
    assert subset_loss > 0
    assert subset_loss != pytest.approx(full_loss)


def test_production_parameters_pass_the_guards():
    table = RT.build_table()
    assert O.guards_ok(table["realmodels"], R.PROD_PARAMS) is True


def test_guards_ok_fails_closed_on_asymmetric_low_confidence(monkeypatch):
    """A swept parameter vector can make one method low-confidence on a base
    the other method isn't. That must reject the vector, not KeyError out of
    the fitting loop.
    """
    rows = [
        {"unit": "mgld", "base": "v1"}, {"unit": "mgld", "base": "v2"},
        {"unit": "flashvsr", "base": "v1"}, {"unit": "flashvsr", "base": "v2"},
        {"unit": "uav", "base": "v1"}, {"unit": "uav", "base": "v2"},
    ]

    def fake_composite(row, params):
        if row["unit"] == "uav" and row["base"] == "v2":
            return {"lr_vcc": 0.5, "low_confidence": True}
        score = {"mgld": 0.8, "flashvsr": 0.6, "uav": 0.4}[row["unit"]]
        return {"lr_vcc": score, "low_confidence": False}

    monkeypatch.setattr(O, "composite", fake_composite)
    assert O.guards_ok(rows, {}) is False
