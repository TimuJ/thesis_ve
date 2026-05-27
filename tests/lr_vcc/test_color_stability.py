"""Tests for sub-metric D wrapper (color_stability_score).

Focuses on the alpha-override path added for --color_hist_alpha CLI flag.
The default-alpha behaviour is covered indirectly in test_color_histogram.py;
here we lock in the contract: explicit alpha => recomputed score, None => default.
"""
import math

from scripts.lr_vcc.color_stability import color_stability_score, _DEFAULT_ALPHA


def _payload(mean_hist_dist, n_frames=2400, stored_score=None):
    p = {"n_frames": n_frames, "mean_hist_dist": mean_hist_dist}
    if stored_score is not None:
        p["score"] = stored_score
    return p


def test_default_alpha_uses_json_score():
    """With alpha=None, score is derived from mean_hist_dist using _DEFAULT_ALPHA.

    Backwards-compat guarantee: the wrapper has historically ignored the JSON's
    "score" field and re-derived with the module default. Locking that in.
    """
    dist = 1.5
    # JSON carries a stored score from a different alpha (e.g. compute-time alpha=5).
    p = _payload(dist, stored_score=math.exp(-5.0 * dist))

    out = color_stability_score(p)  # alpha=None -> default path

    expected = math.exp(-_DEFAULT_ALPHA * dist)
    assert abs(out["score"] - expected) < 1e-9, \
        f"default-alpha score {out['score']} != exp(-{_DEFAULT_ALPHA}*{dist})={expected}"
    assert out["details"]["alpha"] == _DEFAULT_ALPHA
    assert out["details"]["override"] is False


def test_explicit_alpha_overrides_score():
    """With alpha=0.394 (the recalibrated value), score = exp(-0.394 * dist),
    regardless of any stored score in the JSON."""
    dist = 1.7591  # MGLD median from the calibration run
    p = _payload(dist, stored_score=0.123)  # bogus stored score, must be ignored

    chosen_alpha = 0.394
    out = color_stability_score(p, alpha=chosen_alpha)

    expected = math.exp(-chosen_alpha * dist)
    assert abs(out["score"] - expected) < 1e-9
    assert abs(out["score"] - 0.5) < 0.01, \
        f"alpha=0.394 with MGLD-median dist should land near 0.5, got {out['score']}"
    assert out["details"]["alpha"] == chosen_alpha
    assert out["details"]["override"] is True
