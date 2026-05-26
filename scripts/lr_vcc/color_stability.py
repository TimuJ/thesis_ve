"""Sub-metric D — color stability. Wraps the per-video JSON produced by
compute_color_histogram.py.

Accepts two JSON formats:
  1. Simple: {"n_frames": int, "mean_hist_dist": float}
  2. Full (produced by compute_color_histogram.py):
     {"n_frames": int, "score": ..., "reliability": ...,
      "details": {"mean_l1_dist": float, ...}}

In format 2 the score/reliability are already computed by color_histogram_score_from_frames;
this wrapper re-derives them from mean_hist_dist for a consistent interface.
"""
import math

from .reliability import below_threshold_penalty


_MIN_FRAMES_FLOOR = 240  # need at least 240 frames to evaluate at k=120 reliably
_DEFAULT_ALPHA = 2.0


def color_stability_score(per_video_hist: dict, alpha: float = _DEFAULT_ALPHA) -> dict:
    """Returns {"score", "reliability", "details"}.

    score = exp(-alpha * mean_hist_dist), in [0, 1]. Lower distance -> higher score.
    reliability = 1 - below_threshold_penalty(n_frames, 240, sharpness=0.02).

    The sharpness=0.02 gives a smooth ramp: n_frames=240 -> rel~0.50,
    n_frames=100 -> rel~0.03, n_frames=5000 -> rel~0.97.
    """
    n_frames = int(per_video_hist.get("n_frames", 0))

    # Accept mean_hist_dist as a top-level key (simple format) or from details
    # (full compute_color_histogram.py output which uses mean_l1_dist internally).
    dist = per_video_hist.get("mean_hist_dist")
    if dist is None:
        # Fall back to details.mean_l1_dist from the full format
        details = per_video_hist.get("details") or {}
        dist = details.get("mean_l1_dist")

    if dist is None:
        return {"score": 0.0, "reliability": 0.0,
                "details": {"mean_hist_dist": None, "n_frames": n_frames}}

    score = math.exp(-alpha * float(dist))
    score = max(0.0, min(1.0, score))
    reliability = 1.0 - below_threshold_penalty(n_frames, _MIN_FRAMES_FLOOR, sharpness=0.02)
    reliability = max(0.0, min(1.0, reliability))

    return {
        "score": score,
        "reliability": reliability,
        "details": {
            "mean_hist_dist": float(dist),
            "alpha": alpha,
            "n_frames": n_frames,
        },
    }
