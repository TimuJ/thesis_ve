import pytest

from scripts.lr_vcc.calibration import failure_analysis as FA
from scripts.lr_vcc.calibration import recompose as R
from scripts.lr_vcc.calibration import response_table as RT


@pytest.fixture(scope="module")
def rows():
    return RT.build_table()["artefacts"]


def test_stage_vocabulary_is_closed():
    assert FA.STAGES == ("measurement", "reward_direction", "normalisation",
                         "gate", "composition", "ok")
    assert set(FA.ADDRESSABLE) == {"normalisation", "gate", "composition"}


def test_identity_degradation_on_7WHI_is_reward_direction(rows):
    """I rises 0.375 -> 0.489 as identity degrades; the cell is INVERTED."""
    result = FA.analyse(rows, R.PROD_PARAMS)
    cell = result[("identity_degradation", "7WHI2L_FDNg")]
    stages = {d["sub_metric"]: d["stage"] for d in cell["sub_metrics"]}
    assert stages["identity"] == "reward_direction"
    assert cell["conforms"] is False


def test_flicker_on_7WHI_is_a_composition_failure(rows):
    """A and T both respond; D, E and D' outweigh them in the wrong direction."""
    result = FA.analyse(rows, R.PROD_PARAMS)
    cell = result[("flicker", "7WHI2L_FDNg")]
    stages = {d["sub_metric"]: d["stage"] for d in cell["sub_metrics"]}
    assert stages["temporal"] == "composition"
    assert cell["conforms"] is False


def test_conforming_cells_are_marked_ok(rows):
    result = FA.analyse(rows, R.PROD_PARAMS)
    cell = result[("flip_invert", "KZ8p6b1zJ9U")]
    assert cell["conforms"] is True
    assert all(d["stage"] == "ok" for d in cell["sub_metrics"])


def test_unconstrained_cells_report_none(rows):
    result = FA.analyse(rows, R.PROD_PARAMS)
    assert result[("flip_transpose", "KZ8p6b1zJ9U")]["conforms"] is None


def test_every_cell_is_analysed(rows):
    assert len(FA.analyse(rows, R.PROD_PARAMS)) == 60


def test_weight_drift_flag_fires_on_background_drift_brrlk(rows):
    """I's weight moves 0.017 -> 0.176 across this ladder."""
    result = FA.analyse(rows, R.PROD_PARAMS)
    cell = result[("background_drift", "BrRLKMbBTYQ")]
    drift = {d["sub_metric"]: d["weight_drift"] for d in cell["sub_metrics"]}
    assert drift["identity"] is True


def test_assigned_stages_stay_inside_the_vocabulary(rows):
    """Coverage check: no cell gets a stage outside STAGES, and the two
    mechanisms the probes demonstrated both occur somewhere in the matrix."""
    result = FA.analyse(rows, R.PROD_PARAMS)
    seen = {d["stage"] for cell in result.values() for d in cell["sub_metrics"]}
    assert seen <= set(FA.STAGES)
    assert "reward_direction" in seen
    assert "composition" in seen
