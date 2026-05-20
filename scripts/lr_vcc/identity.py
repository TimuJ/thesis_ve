"""Sub-metric I — slow-fast Human_Identity + face-rate + close-up reliability.

Wraps the per_video[v] output of scripts/vbench2_long/human_identity_long.py.
"""
from typing import Optional

from .reliability import below_threshold_penalty, above_threshold_penalty


_FACE_RATE_FLOOR = 0.20
_CLOSEUP_BBOX_THRESHOLD = 0.05  # face / hand bbox p50 as fraction of frame area


def identity_score(per_video: dict, closeup_bbox_p50: Optional[float] = None) -> dict:
    """Returns {"score", "reliability", "details": {...}}.

    score = per_video["fused"] (output of slow-fast Identity adapter).
    reliability = (1 - face_rate_penalty) * (1 - closeup_penalty).
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

    reliability = (1.0 - face_pen) * (1.0 - closeup_pen)
    return {
        "score": max(0.0, min(1.0, score)),
        "reliability": reliability,
        "details": {
            "face_rate": face_rate,
            "face_penalty": face_pen,
            "closeup_bbox_p50": closeup_bbox_p50,
            "closeup_penalty": closeup_pen,
        },
    }
