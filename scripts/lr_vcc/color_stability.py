"""Sub-metric D — color stability. Wraps the per-video JSON produced by
compute_color_histogram.py.

Accepts two JSON formats:
  1. Simple: {"n_frames": int, "mean_hist_dist": float}
  2. Full (produced by compute_color_histogram.py):
     {"n_frames": int, "score": ..., "reliability": ...,
      "details": {"mean_l1_dist": float, ...}}

Score-loading policy (backwards-compat with v2):
  - If ``alpha`` is None (default): derive score = exp(-_DEFAULT_ALPHA * mean_hist_dist)
    using the module-level _DEFAULT_ALPHA. This preserves the existing v2 behaviour
    where any JSON-stored "score" was ignored in favour of re-derivation.
  - If ``alpha`` is a float (e.g. supplied by --color_hist_alpha):
    score = exp(-alpha * mean_hist_dist). This is the recalibration path.
"""
import math

from .reliability import below_threshold_penalty


_MIN_FRAMES_FLOOR = 240  # need at least 240 frames to evaluate at k=120 reliably
_DEFAULT_ALPHA = 2.0


def color_stability_score(per_video_hist: dict, alpha: float = None) -> dict:
    """Returns {"score", "reliability", "details"}.

    Parameters
    ----------
    per_video_hist : dict
        Loaded JSON payload. See module docstring for accepted formats.
    alpha : float or None
        If None (default), use _DEFAULT_ALPHA — preserves v2 behaviour where the
        score is always re-derived from mean_hist_dist (JSON's "score" field is
        ignored). If a float, use that alpha instead. This is the CLI override
        path used by --color_hist_alpha for sub-metric D recalibration.
    """
    used_alpha = float(_DEFAULT_ALPHA if alpha is None else alpha)
    override = alpha is not None

    n_frames = int(per_video_hist.get("n_frames", 0))

    # Accept mean_hist_dist as a top-level key (simple format) or from details
    # (full compute_color_histogram.py output which uses mean_l1_dist internally).
    dist = per_video_hist.get("mean_hist_dist")
    if dist is None:
        details = per_video_hist.get("details") or {}
        dist = details.get("mean_l1_dist")

    reliability = 1.0 - below_threshold_penalty(n_frames, _MIN_FRAMES_FLOOR, sharpness=0.02)
    reliability = max(0.0, min(1.0, reliability))

    if dist is None:
        return {"score": 0.0, "reliability": 0.0,
                "details": {"mean_hist_dist": None, "n_frames": n_frames,
                            "alpha": used_alpha, "override": override}}

    score = math.exp(-used_alpha * float(dist))
    score = max(0.0, min(1.0, score))

    return {
        "score": score,
        "reliability": reliability,
        "details": {
            "mean_hist_dist": float(dist),
            "alpha": used_alpha,
            "override": override,
            "n_frames": n_frames,
        },
    }
