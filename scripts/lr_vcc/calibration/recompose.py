"""(row, parameter vector) -> LR-VCC composite, as pure arithmetic.

No disk access, no JSON, no video. This is what makes a five-fold search over
a real parameter grid affordable: a full 315-row matrix recomposes in a few
milliseconds, against 2.7 s for the JSON-reading path in sweep_sensitivity.

The bit-exactness test against run_lr_vcc.evaluate_one_video is what keeps
this module honest. Any change here that breaks it is a bug here, not there.

This module assumes each row is well-formed. The canonical sub-metric
modules guard degenerate inputs (e.g. appearance_score returns (0.0, 0.0) on
an empty clip_iqa list; color_stability_score returns (0.0, 0.0) when dist is
None) — this module does not re-implement those guards and would diverge or
raise on such a row. That's fine today because response_table.py raises on
such inputs first, upstream of here; the next reader should know the guard
lives there, not in this module.
"""
import math

from ..reliability import above_threshold_penalty, below_threshold_penalty
from .expectations import SUB_METRICS

# v5 production settings, expressed in this module's parameter vocabulary.
# beta_t=None selects T's original linear form and is what makes bit-exact
# reproduction of v5 possible; a float switches T to the exponential response.
PROD_PARAMS = {
    "lambda_a": 0.5,
    "beta_t": None,
    "alpha": 0.394,
    "beta_e": 200.0,
    "beta_dp": 0.5,
    "beta_dpp": 3.0,
    "tau": 0.2,
    "a_drift_floor": 0.02,
    "a_sat_ceiling": 0.98,
    "mask_cov_floor": 0.10,
    "face_rate_floor": 0.20,
    "closeup_threshold": 0.05,
    "temporal_weight": "uniform",
    "hist_min_frames": 240,
    "low_confidence_floor": 0.2,
    "eps": 1e-6,
}

_WEIGHT_FNS = {
    "log": lambda k: math.log(1 + k),
    "uniform": lambda k: 1.0,
    "sqrt": lambda k: math.sqrt(k),
}


def _clamp01(x):
    return max(0.0, min(1.0, x))


def _appearance(row, p):
    score = _clamp01(row["a_mean"] - p["lambda_a"] * row["a_std"])
    drift_pen = below_threshold_penalty(row["a_std"], p["a_drift_floor"])
    sat_pen = above_threshold_penalty(row["a_mean"], p["a_sat_ceiling"])
    return score, max(0.0, 1.0 - max(drift_pen, sat_pen))


def _temporal(row, p):
    weight_func = _WEIGHT_FNS.get(p["temporal_weight"], _WEIGHT_FNS["log"])
    tofs, covs = row["tof"], row["cov"]
    weighted_sum = weight_total = 0.0
    for k_str in tofs:
        if tofs[k_str] is None:
            continue
        if float(covs.get(k_str, 0.0)) < p["mask_cov_floor"]:
            continue
        w = weight_func(int(k_str))
        weighted_sum += w * float(tofs[k_str])
        weight_total += w
    if weight_total == 0:
        score = 0.0
    else:
        wm = weighted_sum / weight_total
        score = _clamp01(1.0 - wm if p["beta_t"] is None
                         else math.exp(-p["beta_t"] * wm))
    rel_terms = [1.0 - below_threshold_penalty(float(covs.get(k, 0.0)),
                                               p["mask_cov_floor"])
                 for k in tofs]
    reliability = sum(rel_terms) / len(rel_terms) if rel_terms else 0.0
    return score, reliability


def _identity(row, p):
    n_clips = row["n_clips"]
    face_rate = row["n_clips_with_faces"] / n_clips if n_clips > 0 else 0.0
    face_pen = below_threshold_penalty(face_rate, p["face_rate_floor"])
    if row["closeup_p50"] is None:
        closeup_pen = 0.0
    else:
        closeup_pen = above_threshold_penalty(float(row["closeup_p50"]),
                                              p["closeup_threshold"])
    # The dispersion gate stays parked: its penalty is always 0.0.
    return _clamp01(row["identity_fused"]), (1.0 - face_pen) * (1.0 - closeup_pen)


def _exp_sub(raw, beta, reliability):
    return _clamp01(math.exp(-beta * raw)), reliability


def sub_metric_values(row, p):
    """[(name, score, reliability)] in canonical SUB_METRICS order."""
    a_s, a_r = _appearance(row, p)
    t_s, t_r = _temporal(row, p)
    i_s, i_r = _identity(row, p)
    d_s, d_r = _exp_sub(
        row["hist_dist"], p["alpha"],
        _clamp01(1.0 - below_threshold_penalty(row["hist_n_frames"],
                                               p["hist_min_frames"],
                                               sharpness=0.02)))
    e_s, e_r = _exp_sub(row["slope_abs"], p["beta_e"], row["slope_rel"])
    dp_s, dp_r = _exp_sub(row["anchor_q14"], p["beta_dp"], row["anchor_rel"])
    dpp_s, dpp_r = _exp_sub(row["clip_q14"], p["beta_dpp"], row["clip_rel"])
    values = [a_s, t_s, i_s, d_s, e_s, dp_s, dpp_s]
    rels = [a_r, t_r, i_r, d_r, e_r, dp_r, dpp_r]
    return list(zip(SUB_METRICS, values, rels))


def composite(row, p):
    triples = sub_metric_values(row, p)
    scores = [s for _, s, _ in triples]
    rels = [r for _, _, r in triples]
    z = [r / p["tau"] for r in rels]
    z_max = max(z)
    exps = [math.exp(zi - z_max) for zi in z]
    total = sum(exps)
    weights = [e / total for e in exps]
    log_sum = sum(w * math.log(s + p["eps"]) for w, s in zip(weights, scores))
    return {
        "lr_vcc": math.exp(log_sum),
        "scores": scores,
        "reliabilities": rels,
        "weights": weights,
        "low_confidence": all(r < p["low_confidence_floor"] for r in rels),
    }
