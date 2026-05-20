"""Sub-metric T — long-k-weighted tOF + mask-coverage reliability.

Reads the per-video JSON produced by scripts/long_range_temporal/eval_tof_tlp.py.
"""
import math
from typing import Iterable

from .reliability import below_threshold_penalty


_DEFAULT_MASK_COV_FLOOR = 0.10


def _weight_fn(k: int) -> float:
    """log(1+k) — weights long-k more than adjacent k."""
    return math.log(1 + k)


def temporal_score(tof_payload: dict,
                   mask_cov_floor: float = _DEFAULT_MASK_COV_FLOOR) -> dict:
    """Returns {"score", "reliability", "details": {...}}.

    score = 1 - weighted_mean(tof_k) over k with mask_coverage[k] >= floor.
    reliability = mean over k of (1 - below_threshold_penalty(coverage[k], floor)).
    """
    tofs = tof_payload["tof"]
    covs = tof_payload["mean_mask_coverage"]
    k_strs = list(tofs.keys())

    weighted_sum = 0.0
    weight_total = 0.0
    used_ks = []
    for k_str in k_strs:
        if tofs[k_str] is None:
            continue
        cov = float(covs.get(k_str, 0.0))
        if cov < mask_cov_floor:
            continue
        k = int(k_str)
        w = _weight_fn(k)
        weighted_sum += w * float(tofs[k_str])
        weight_total += w
        used_ks.append(k_str)

    if weight_total == 0:
        score = 0.0
    else:
        weighted_mean = weighted_sum / weight_total
        score = max(0.0, min(1.0, 1.0 - weighted_mean))

    rel_terms = []
    for k_str in k_strs:
        cov = float(covs.get(k_str, 0.0))
        rel_terms.append(1.0 - below_threshold_penalty(cov, mask_cov_floor))
    reliability = sum(rel_terms) / len(rel_terms) if rel_terms else 0.0

    return {
        "score": score,
        "reliability": reliability,
        "details": {
            "used_ks": used_ks,
            "weighted_mean_tof": (weighted_sum / weight_total) if weight_total else None,
            "mean_mask_coverage_over_all_k": sum(float(covs[k]) for k in k_strs) / len(k_strs),
        },
    }
