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
    # Load-bearing: the returned loss must actually be the loss of the
    # returned params, not just some value <= start_loss. A stub returning
    # (PROD_PARAMS, 0.0) would satisfy the assertion above but fail this one.
    assert loss == pytest.approx(O.matrix_loss(table["artefacts"], params,
                                               O.LOSS_CFG, bases=bases))
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


def test_loss_surface_is_deterministic_and_covers_the_declared_grids(table):
    """The sensitivity deliverable (F3): one point per grid value, for
    every searched parameter, and re-running it changes nothing.
    """
    surf1 = F.loss_surface(table["artefacts"], R.PROD_PARAMS, O.LOSS_CFG)
    surf2 = F.loss_surface(table["artefacts"], R.PROD_PARAMS, O.LOSS_CFG)
    assert surf1 == surf2
    assert set(surf1) == set(F.SEARCH_ORDER) | set(F.GATE_ORDER)
    for key, points in surf1.items():
        grid = F.GRIDS[key] if key in F.GRIDS else F.GATE_GRIDS[key]
        assert [p["value"] for p in points] == grid
        assert all(p["loss"] >= 0 for p in points)


def test_lobo_result_includes_loss_surfaces_at_the_final_params(lobo_result):
    """lobo() stores the scan around final_params, not around PROD_PARAMS
    or some other point — the point a reader will actually be looking at
    in the 'Final parameters' table.
    """
    surfaces = lobo_result["loss_surfaces"]
    final_params = lobo_result["final_params"]
    for key, points in surfaces.items():
        grid = F.GRIDS[key] if key in F.GRIDS else F.GATE_GRIDS[key]
        chosen_index = grid.index(final_params[key])
        # The loss recorded at the chosen value's own grid point must
        # equal final_loss: everything else is held at final_params, and
        # this point *is* final_params.
        assert points[chosen_index]["loss"] == pytest.approx(
            lobo_result["final_loss"])


def test_lobo_heldout_matrix_has_one_column_per_fold(lobo_result):
    cells = lobo_result["heldout_matrix"]
    assert len(cells) == 60
    for (_family, base), cell in cells.items():
        assert cell["fitted_without"] == base


def test_fitting_never_receives_its_own_held_out_base(monkeypatch, table):
    """Give the disjointness claim teeth.

    test_lobo_folds_are_disjoint only reads the labels fit.py itself writes
    (fold["held_out"], fold["train_bases"]) — both come from the same local
    `train` variable the implementation controls. An implementation that
    secretly fit every fold on all five bases while still recording
    "train_bases": list(train) would pass every assertion there. This test
    instead intercepts every guards_ok / matrix_loss call made while a fold
    is actually being fit and checks the REAL `bases` argument each one
    received, not a label.

    Patches fit._search rather than the public coordinate_search: lobo()
    calls _search directly (to also recover the convergence flag), so
    _search is the actual fitting entry point whose calls need bracketing.
    """
    calls = []
    real_search = F._search
    real_guards_ok = F.guards_ok
    real_matrix_loss = F.matrix_loss
    active = []

    def spy_search(art_rows, real_rows, bases, cfg, start, passes):
        active.append(tuple(bases))
        try:
            return real_search(art_rows, real_rows, bases, cfg, start, passes)
        finally:
            active.pop()

    def spy_guards_ok(real_rows, params, bases=None):
        if active:
            calls.append(("guards_ok", active[-1], bases))
        return real_guards_ok(real_rows, params, bases=bases)

    def spy_matrix_loss(rows, params, cfg=O.LOSS_CFG, bases=None):
        if active:
            calls.append(("matrix_loss", active[-1], bases))
        return real_matrix_loss(rows, params, cfg, bases=bases)

    monkeypatch.setattr(F, "_search", spy_search)
    monkeypatch.setattr(F, "guards_ok", spy_guards_ok)
    monkeypatch.setattr(F, "matrix_loss", spy_matrix_loss)

    F.lobo(table, O.LOSS_CFG, passes=1)

    # Fold-time calls are the ones made while a 4-base (not the full 5-base
    # final refit) search is active.
    fold_time_calls = [(fn, expected, actual) for fn, expected, actual in calls
                       if len(expected) == 4]
    assert fold_time_calls, "expected fitting-time calls to be recorded"
    for fn, expected_bases, actual_bases in fold_time_calls:
        held = (set(E.BASES) - set(expected_bases)).pop()
        assert actual_bases == expected_bases, (fn, expected_bases, actual_bases)
        assert held not in actual_bases, (fn, held, actual_bases)
