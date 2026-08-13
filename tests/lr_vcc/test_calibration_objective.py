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
    """Equal-magnitude misses cost more on a control than on a target."""
    cfg = O.LOSS_CFG
    over = {"0p02": 0.90, "0p05": 0.88, "0p10": 0.86, "0p20": 0.84, "0p40": 0.78}
    assert O.cell_loss("flip_horizontal", over, cfg) > O.cell_loss("flicker", over, cfg)


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


def test_production_parameters_pass_the_guards():
    table = RT.build_table()
    assert O.guards_ok(table["realmodels"], R.PROD_PARAMS) is True
