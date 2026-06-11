"""Sub-metric I — slow-fast Human_Identity + face-rate + close-up reliability.

Wraps the per_video[v] output of scripts/vbench2_long/human_identity_long.py.
"""
import statistics
from typing import Optional

from .reliability import below_threshold_penalty, above_threshold_penalty


_FACE_RATE_FLOOR = 0.20
_CLOSEUP_BBOX_THRESHOLD = 0.05  # face / hand bbox p50 as fraction of frame area
# calibrated 2026-06-11 on 2-base data (gap 0.003 — too thin); parked until 8-base recalibration
_CLIP_DISPERSION_THRESHOLD = 0.346


def clip_score_dispersion(per_video: dict) -> Optional[float]:
    """Population std-dev of valid per-clip slow scores; None when < 2 valid
    clips or clip_detail absent (older JSONs without --detail)."""
    detail = per_video.get("clip_detail") or []
    valid = [float(c["score"]) for c in detail if float(c.get("score", -1.0)) >= 0.0]
    if len(valid) < 2:
        return None
    return statistics.pstdev(valid)


def identity_score(per_video: dict, closeup_bbox_p50: Optional[float] = None,
                   dispersion_threshold: Optional[float] = None) -> dict:
    """Returns {"score", "reliability", "details": {...}}.

    score = per_video["fused"] (output of slow-fast Identity adapter).
    reliability = (1 - face_rate_penalty) * (1 - closeup_penalty) * (1 - dispersion_penalty).

    dispersion_threshold: when None (default), the dispersion gate is OFF and
    dispersion_penalty is always 0.0 (two-factor formula).  Pass
    dispersion_threshold=_CLIP_DISPERSION_THRESHOLD explicitly to enable the gate.
    clip_score_dispersion is always computed and reported in details regardless.
    """
    score = float(per_video.get("fused", 0.0))
    n_clips = int(per_video.get("n_clips", 0))
    n_faces = int(per_video.get("n_clips_with_faces", 0))
    face_rate = n_faces / n_clips if n_clips > 0 else 0.0

    face_pen = below_threshold_penalty(face_rate, _FACE_RATE_FLOOR)
    if closeup_bbox_p50 is None:
        closeup_pen = 0.0
    else:
        closeup_pen = above_threshold_penalty(float(closeup_bbox_p50), _CLOSEUP_BBOX_THRESHOLD)

    disp = clip_score_dispersion(per_video)
    disp_pen = (
        0.0
        if (disp is None or dispersion_threshold is None)
        else above_threshold_penalty(disp, dispersion_threshold)
    )

    reliability = (1.0 - face_pen) * (1.0 - closeup_pen) * (1.0 - disp_pen)
    return {
        "score": max(0.0, min(1.0, score)),
        "reliability": reliability,
        "details": {
            "face_rate": face_rate,
            "face_penalty": face_pen,
            "closeup_bbox_p50": closeup_bbox_p50,
            "closeup_penalty": closeup_pen,
            "clip_score_dispersion": disp,
            "dispersion_penalty": disp_pen,
        },
    }
