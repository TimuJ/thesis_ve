"""Sub-metric A — appearance stability (CLIP-IQA mean - lambda * std).

Reads a per-video JSON produced by compute_clip_iqa.py (server-side).
"""
import statistics

from .reliability import below_threshold_penalty, above_threshold_penalty


_DRIFT_FLOOR = 0.02       # if std(quality) < this -> sub-metric undiscriminating
_SATURATION_CEILING = 0.98  # if mean(quality) > this -> ceiling regime
_DEFAULT_LAMBDA = 0.5


def appearance_score(per_video_clip_iqa: dict, lam: float = _DEFAULT_LAMBDA) -> dict:
    qs = per_video_clip_iqa["clip_iqa"]
    if not qs:
        return {"score": 0.0, "reliability": 0.0, "details": {}}
    mean_q = statistics.mean(qs)
    std_q = statistics.pstdev(qs)  # population std for stability with small n
    score = max(0.0, min(1.0, mean_q - lam * std_q))

    drift_pen = below_threshold_penalty(std_q, _DRIFT_FLOOR)
    sat_pen = above_threshold_penalty(mean_q, _SATURATION_CEILING)
    reliability = max(0.0, 1.0 - max(drift_pen, sat_pen))

    return {
        "score": score,
        "reliability": reliability,
        "details": {
            "mean_quality": mean_q,
            "std_quality": std_q,
            "drift_penalty": drift_pen,
            "saturation_penalty": sat_pen,
        },
    }
