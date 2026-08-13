"""Deterministic coordinate search inside leave-one-base-out folds.

Determinism matters more than optimality at this scale: the same table and the
same grids must always produce the same parameters, or the reported numbers
are not reproducible. No randomness, no early stopping on wall-clock.

Usage (repo root):
  python -m scripts.lr_vcc.calibration.fit
"""
import json
from pathlib import Path

from . import expectations as E
from .objective import LOSS_CFG, guards_ok, matrix_loss, matrix_scores
from .recompose import PROD_PARAMS
from .response_table import TABLE_PATH, build_table

FIT_DIR = TABLE_PATH.parent


def logspace(lo, hi, n):
    return [lo * (hi / lo) ** (i / (n - 1)) for i in range(n)]


# Response parameters, searched first. v5's value lies inside every grid, and
# beta_t=None keeps v5's linear T reachable, so the fit can decline any lever.
GRIDS = {
    "lambda_a": [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0],
    "beta_t": [None] + logspace(1.0, 50.0, 11),
    "alpha": logspace(0.05, 3.0, 11),
    "beta_e": logspace(20.0, 2000.0, 11),
    "beta_dp": logspace(0.1, 5.0, 11),
    "beta_dpp": logspace(0.5, 30.0, 11),
    "tau": logspace(0.05, 5.0, 11),
}

# Gate thresholds, searched in a second pass with response parameters fixed.
GATE_GRIDS = {
    "a_drift_floor": [0.0, 0.01, 0.02, 0.05, 0.10],
    "a_sat_ceiling": [0.90, 0.95, 0.98, 1.0],
    "mask_cov_floor": [0.0, 0.05, 0.10, 0.20],
    "face_rate_floor": [0.0, 0.10, 0.20, 0.40],
    "closeup_threshold": [0.02, 0.05, 0.10, 1.0],
}

SEARCH_ORDER = ("tau", "beta_t", "lambda_a", "alpha", "beta_e", "beta_dp",
                "beta_dpp")
GATE_ORDER = ("mask_cov_floor", "a_drift_floor", "a_sat_ceiling",
              "face_rate_floor", "closeup_threshold")


def _evaluate(art_rows, real_rows, bases, cfg, params):
    """Loss, or None when the leaderboard guards reject this vector."""
    if not guards_ok(real_rows, params, bases=bases):
        return None
    return matrix_loss(art_rows, params, cfg, bases=bases)


def coordinate_search(art_rows, real_rows, bases, cfg=LOSS_CFG,
                      start=PROD_PARAMS, passes=3):
    """Minimise the loss one parameter at a time. Returns (params, loss)."""
    params = dict(start)
    best = _evaluate(art_rows, real_rows, bases, cfg, params)
    if best is None:
        raise ValueError("starting parameters violate the leaderboard guards")
    for _ in range(passes):
        improved = False
        for key in SEARCH_ORDER + GATE_ORDER:
            grid = GRIDS.get(key) or GATE_GRIDS[key]
            for value in grid:
                if value == params[key]:
                    continue
                trial = dict(params, **{key: value})
                loss = _evaluate(art_rows, real_rows, bases, cfg, trial)
                if loss is not None and loss < best - 1e-12:
                    params, best, improved = trial, loss, True
        if not improved:
            break
    return params, best


def lobo(table, cfg=LOSS_CFG, passes=3):
    """Five folds. Each fold's parameters never saw its held-out base."""
    art, real = table["artefacts"], table["realmodels"]
    folds, heldout = [], {}
    for held in E.BASES:
        train = tuple(b for b in E.BASES if b != held)
        params, train_loss = coordinate_search(art, real, train, cfg,
                                               PROD_PARAMS, passes)
        test_loss = matrix_loss(art, params, cfg, bases=(held,))
        for key, cell in matrix_scores(art, params, bases=(held,)).items():
            heldout[key] = dict(cell, fitted_without=held)
        folds.append({"held_out": held, "train_bases": list(train),
                      "params": params, "train_loss": train_loss,
                      "test_loss": test_loss})
    final_params, final_loss = coordinate_search(art, real, E.BASES, cfg,
                                                 PROD_PARAMS, passes)
    insample = {k: dict(v) for k, v in
                matrix_scores(art, final_params).items()}
    return {"folds": folds, "heldout_matrix": heldout,
            "insample_matrix": insample, "final_params": final_params,
            "final_loss": final_loss,
            "v5_loss": matrix_loss(art, PROD_PARAMS, cfg)}


def _jsonable(matrix):
    return {"{}|{}".format(f, b): v for (f, b), v in matrix.items()}


if __name__ == "__main__":
    table = build_table()
    result = lobo(table)
    FIT_DIR.mkdir(parents=True, exist_ok=True)
    out = dict(result,
               heldout_matrix=_jsonable(result["heldout_matrix"]),
               insample_matrix=_jsonable(result["insample_matrix"]))
    json.dump(out, open(FIT_DIR / "lobo_result.json", "w"), indent=2)
    print("v5 loss           {:.6f}".format(result["v5_loss"]))
    print("v6 in-sample loss {:.6f}".format(result["final_loss"]))
    for f in result["folds"]:
        print("fold {:14s} train {:.6f}  held-out {:.6f}".format(
            f["held_out"], f["train_loss"], f["test_loss"]))
