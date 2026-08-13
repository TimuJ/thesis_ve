import pytest

from scripts.lr_vcc.calibration import fit as F
from scripts.lr_vcc.calibration import objective as O
from scripts.lr_vcc.calibration import recompose as R
from scripts.lr_vcc.calibration import response_table as RT
from scripts.lr_vcc.calibration import expectations as E


@pytest.fixture(scope="module")
def table():
    return RT.build_table()


@pytest.fixture(scope="module")
def lobo_result(table):
    return F.lobo(table, O.LOSS_CFG)


def test_logspace_endpoints_and_length():
    xs = F.logspace(0.1, 10.0, 5)
    assert len(xs) == 5
    assert xs[0] == pytest.approx(0.1)
    assert xs[-1] == pytest.approx(10.0)


def test_v5_is_reachable_from_every_grid():
    """The fit must be able to decline each new lever."""
    assert None in F.GRIDS["beta_t"]
    for key in ("alpha", "beta_e", "beta_dp", "beta_dpp", "tau", "lambda_a"):
        prod = R.PROD_PARAMS[key]
        assert min(F.GRIDS[key]) <= prod <= max(F.GRIDS[key]), key


def test_coordinate_search_does_not_increase_loss(table):
    bases = E.BASES[:4]
    start_loss = O.matrix_loss(table["artefacts"], R.PROD_PARAMS,
                               O.LOSS_CFG, bases=bases)
    params, loss = F.coordinate_search(table["artefacts"], table["realmodels"],
                                       bases, O.LOSS_CFG, R.PROD_PARAMS,
                                       passes=1)
    assert loss <= start_loss + 1e-12
    assert O.guards_ok(table["realmodels"], params, bases=bases) is True


def test_coordinate_search_is_deterministic(table):
    bases = E.BASES[:4]
    a, la = F.coordinate_search(table["artefacts"], table["realmodels"],
                                bases, O.LOSS_CFG, R.PROD_PARAMS, passes=1)
    b, lb = F.coordinate_search(table["artefacts"], table["realmodels"],
                                bases, O.LOSS_CFG, R.PROD_PARAMS, passes=1)
    assert a == b and la == lb


def test_lobo_folds_are_disjoint(lobo_result):
    """The central methodological claim: a fold never trains on its own base."""
    assert len(lobo_result["folds"]) == 5
    for fold in lobo_result["folds"]:
        assert fold["held_out"] not in fold["train_bases"]
        assert len(fold["train_bases"]) == 4
        assert set(fold["train_bases"]) | {fold["held_out"]} == set(E.BASES)


def test_lobo_heldout_matrix_has_one_column_per_fold(lobo_result):
    cells = lobo_result["heldout_matrix"]
    assert len(cells) == 60
    for (_family, base), cell in cells.items():
        assert cell["fitted_without"] == base
