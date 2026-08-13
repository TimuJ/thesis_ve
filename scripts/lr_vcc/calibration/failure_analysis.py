"""Attribute every non-conforming matrix cell to the stage that lost the signal.

Five stages, matched in order. Only three of them are reachable by
re-parameterisation; `measurement` and `reward_direction` need a different
measurement, not a different constant, so the count of cells in those two
classes is the honest ceiling on what a fitted v6 can deliver.
"""
import math

from . import expectations as E
from .objective import matrix_scores
from .recompose import composite, sub_metric_values

STAGES = ("measurement", "reward_direction", "normalisation", "gate",
          "composition", "ok")
ADDRESSABLE = ("normalisation", "gate", "composition")

_RAW_FIELD = {
    "appearance": "a_mean",
    "identity": "identity_fused",
    "color_stability": "hist_dist",
    "color_slope": "slope_abs",
    "color_hist_anchor": "anchor_q14",
    "clip_trajectory": "clip_q14",
}

_EPS = 1e-6
_RAW_STATIC = 0.05
_RAW_STRONG = 0.20
_SCORE_DEAD = 0.02
_WRONG_WAY = 0.01
_WEIGHT_DEAD = 0.05
_WEIGHT_DRIFT = 0.05


def _raw_value(row, sub_metric):
    if sub_metric == "temporal":
        vals = [v for v in row["tof"].values() if v is not None]
        return sum(vals) / len(vals) if vals else 0.0
    return float(row[_RAW_FIELD[sub_metric]])


def _traces(rows_by_severity, params):
    """{sub_metric: {"raw": [...], "score": [...], "weight": [...]}} over the ladder."""
    out = {name: {"raw": [], "score": [], "weight": []} for name in E.SUB_METRICS}
    for sev in E.SEVERITIES:
        row = rows_by_severity[sev]
        comp = composite(row, params)
        for idx, (name, score, _rel) in enumerate(sub_metric_values(row, params)):
            out[name]["raw"].append(_raw_value(row, name))
            out[name]["score"].append(score)
            out[name]["weight"].append(comp["weights"][idx])
    return out


def attribute(rows_by_severity, family, params, conforming):
    traces = _traces(rows_by_severity, params)
    designed = E.DESIGNED_FOR.get(family, ())
    contributions = {
        name: (sum(t["weight"]) / len(t["weight"])) *
              (math.log(t["score"][-1] + _EPS) - math.log(t["score"][0] + _EPS))
        for name, t in traces.items()
    }
    findings = []
    for name in designed:
        t = traces[name]
        raw0, raw1 = t["raw"][0], t["raw"][-1]
        rel_raw = abs(raw1 - raw0) / (abs(raw0) + _EPS)
        delta_score = t["score"][-1] - t["score"][0]
        mean_w = sum(t["weight"]) / len(t["weight"])
        drift = (max(t["weight"]) - min(t["weight"])) > _WEIGHT_DRIFT

        if conforming:
            stage = "ok"
        elif rel_raw < _RAW_STATIC:
            stage = "measurement"
        elif delta_score > _WRONG_WAY:
            stage = "reward_direction"
        elif rel_raw >= _RAW_STRONG and abs(delta_score) < _SCORE_DEAD:
            stage = "normalisation"
        elif delta_score <= -_SCORE_DEAD and mean_w < _WEIGHT_DEAD:
            stage = "gate"
        else:
            opposing = sum(c for n, c in contributions.items()
                           if n != name and c > 0)
            stage = ("composition" if opposing >= abs(contributions[name])
                     else "normalisation")

        findings.append({
            "sub_metric": name, "stage": stage,
            "rel_raw": rel_raw, "delta_score": delta_score,
            "mean_weight": mean_w, "contribution": contributions[name],
            "weight_drift": drift,
        })
    return findings


def analyse(rows, params):
    """{(family, base): {"verdict", "delta", "conforms", "sub_metrics"}}."""
    by_cell = {}
    for row in rows:
        by_cell.setdefault((row["unit"], row["base"]), {})[row["severity"]] = row
    scored = matrix_scores(rows, params)
    out = {}
    for key, ladder_rows in by_cell.items():
        family, _base = key
        cell = scored[key]
        conforming = E.conforms(family, cell["verdict"])
        out[key] = {
            "verdict": cell["verdict"], "delta": cell["delta"],
            "conforms": conforming,
            "sub_metrics": attribute(ladder_rows, family, params,
                                     conforming is True),
        }
    return out
