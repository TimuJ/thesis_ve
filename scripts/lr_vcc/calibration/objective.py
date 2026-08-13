"""Severity-response loss over the artefact matrix.

Sign convention: build_verdict_matrix works in delta = y(0.40) - y(0.02), so
PASS is delta <= -0.05. This module works in R = -delta so that larger is
better. The two never mix inside one function.

The loss reads all five ladder points. The v5 verdict protocol reads only the
two endpoints; severities 0.05 / 0.10 / 0.20 are cached and were previously
unused.
"""
from statistics import mean

from ..build_verdict_matrix import verdict
from . import expectations as E
from .recompose import composite

LOSS_CFG = {
    "r_target": 0.10,   # wanted response for a RESPOND cell; PASS is 0.05
    "r_silent": 0.02,   # the FLAT band; a SILENT cell must stay inside it
    "w_mono": 1.0,
    "w_silence": 3.0,   # asymmetric: over-calibration is the guarded failure
}

# Real-model guards. v6 may not buy matrix cells with the leaderboard.
#
# This is the gated-canonical order (reports/figures/realmodel_v5_gated.md,
# which supersedes the earlier table for thesis use), not the production
# variant's order reported by sweep_sensitivity.real_summary(): MGLD 0.622 >
# FlashVSR 0.610 > UAV 0.589. build_table()["realmodels"] deliberately
# contains gated-canonical rows, so the guard must match that protocol.
GUARD_ORDER = ("mgld", "flashvsr", "uav")


def response(ladder):
    """R = y(0.02) - y(0.40). Positive means the corruption lowered the score."""
    return ladder["0p02"] - ladder["0p40"]


def monotonicity_violation(ladder):
    """Total upward movement along the ladder — zero for a clean response."""
    seq = [ladder[s] for s in E.SEVERITIES]
    return sum(max(0.0, b - a) for a, b in zip(seq, seq[1:]))


def cell_loss(family, ladder, cfg=LOSS_CFG):
    exp = E.EXPECTATION[family]
    if exp == E.UNCONSTRAINED:
        return 0.0
    r = response(ladder)
    if exp == E.RESPOND:
        shortfall = max(0.0, cfg["r_target"] - r)
        return shortfall ** 2 + cfg["w_mono"] * monotonicity_violation(ladder)
    excess = max(0.0, abs(r) - cfg["r_silent"])
    return cfg["w_silence"] * excess ** 2


def matrix_scores(rows, params, bases=None):
    """{(family, base): {"ladder", "delta", "verdict", "response"}}."""
    ladders = {}
    for row in rows:
        if bases is not None and row["base"] not in bases:
            continue
        key = (row["unit"], row["base"])
        ladders.setdefault(key, {})[row["severity"]] = \
            composite(row, params)["lr_vcc"]
    out = {}
    for key, ladder in ladders.items():
        if set(ladder) != set(E.SEVERITIES):
            raise ValueError("incomplete ladder for {}".format(key))
        r = response(ladder)
        out[key] = {"ladder": ladder, "response": r, "delta": -r,
                    "verdict": verdict(-r)}
    return out


def matrix_loss(rows, params, cfg=LOSS_CFG, bases=None):
    scored = matrix_scores(rows, params, bases=bases)
    respond, silent = [], []
    for (family, _base), cell in scored.items():
        exp = E.EXPECTATION[family]
        if exp == E.UNCONSTRAINED:
            continue
        loss = cell_loss(family, cell["ladder"], cfg)
        (respond if exp == E.RESPOND else silent).append(loss)
    return (mean(respond) if respond else 0.0) + (mean(silent) if silent else 0.0)


def guards_ok(realmodel_rows, params, bases=None):
    """Canonical order mgld > flashvsr > uav, and MGLD > UAV on every video.

    During LOBO fitting, pass the fold's TRAINING bases so held-out videos do
    not leak into the fit through the guard.
    """
    per_method = {}
    for row in realmodel_rows:
        if bases is not None and row["base"] not in bases:
            continue
        out = composite(row, params)
        if out["low_confidence"]:
            continue
        per_method.setdefault(row["unit"], {})[row["base"]] = out["lr_vcc"]
    if set(per_method) != set(GUARD_ORDER):
        return False
    means = {m: mean(v.values()) for m, v in per_method.items()}
    if sorted(means, key=means.get, reverse=True) != list(GUARD_ORDER):
        return False
    videos = sorted(per_method["mgld"])
    return all(per_method["mgld"][v] > per_method["uav"][v] for v in videos)
