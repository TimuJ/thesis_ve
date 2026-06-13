"""Tests for D' (color_hist_anchor) and D'' (clip_trajectory) sub-metrics in run_lr_vcc.

Contract:
1. Default (no D'/D'' paths) → sub_metrics has exactly
   {appearance, temporal, identity, color_stability, color_slope}.
2. D' added when anchor path provided → correct score and beta_override.
3. D'' added when clip_trajectory path provided → correct score and beta_override.
4. Missing trajectory_mean_per_quarter in JSON → D' not added at all.
"""
import json
import math
from pathlib import Path

from scripts.lr_vcc.run_lr_vcc import evaluate_one_video


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------

def _write_clip_iqa(tmp_path: Path) -> Path:
    p = tmp_path / "clip_iqa.json"
    clip_iqa = [0.7 + 0.05 * (i % 3) for i in range(100)]
    json.dump({
        "video_path": "/fake.mp4", "n_frames": 100, "fps": 30.0,
        "frame_stride": 1, "clip_iqa": clip_iqa,
    }, open(p, "w"))
    return p


def _write_tof(tmp_path: Path) -> Path:
    p = tmp_path / "tof.json"
    tof = {1: 0.02, 5: 0.04, 10: 0.05, 30: 0.07, 60: 0.10, 120: 0.13}
    cov = {1: 0.9, 5: 0.7, 10: 0.5, 30: 0.4, 60: 0.3, 120: 0.2}
    json.dump({
        "video_path": "/fake.mp4", "n_frames": 100, "fps": 30.0,
        "k_values": list(tof.keys()),
        "tof": {str(k): v for k, v in tof.items()},
        "tlp": {str(k): 0.0 for k in tof},
        "n_pairs_used": {str(k): 200 for k in tof},
        "mean_mask_coverage": {str(k): v for k, v in cov.items()},
    }, open(p, "w"))
    return p


def _write_id(tmp_path: Path) -> Path:
    p = tmp_path / "id.json"
    id_pv = {"slow": 0.8, "fast": 0.7, "fused": 0.75,
              "n_clips": 50, "n_clips_with_faces": 40}
    json.dump({"per_video": {"fake": id_pv}}, open(p, "w"))
    return p


def _write_color_hist(tmp_path: Path) -> Path:
    p = tmp_path / "fake_color_hist.json"
    json.dump({"score": 0.85, "reliability": 0.9, "details": {}}, open(p, "w"))
    return p


def _write_color_slope(tmp_path: Path) -> Path:
    p = tmp_path / "fake_color_slope.json"
    json.dump({
        "score": 0.9, "reliability": 0.8,
        "details": {"max_abs_slope": 0.001},
    }, open(p, "w"))
    return p


def _write_color_hist_anchor(tmp_path: Path, trajectory: list) -> Path:
    p = tmp_path / "fake_color_hist_anchor.json"
    json.dump({
        "score": 0.7, "reliability": 0.95,
        "details": {"trajectory_mean_per_quarter": trajectory},
    }, open(p, "w"))
    return p


def _write_clip_trajectory(tmp_path: Path, trajectory: list) -> Path:
    p = tmp_path / "fake_clip_trajectory.json"
    json.dump({
        "score": 0.6, "reliability": 0.88,
        "details": {"trajectory_mean_per_quarter": trajectory},
    }, open(p, "w"))
    return p


def _call_evaluate(tmp_path: Path, **extra_kwargs):
    """Build all 5 base sub-metric fixtures and call evaluate_one_video."""
    fa = _write_clip_iqa(tmp_path)
    ft = _write_tof(tmp_path)
    fi = _write_id(tmp_path)
    fc = _write_color_hist(tmp_path)
    fe = _write_color_slope(tmp_path)
    return evaluate_one_video(
        video_id="fake",
        clip_iqa_path=fa,
        tof_path=ft,
        identity_results_path=fi,
        color_hist_path=fc,
        color_slope_path=fe,
        **extra_kwargs,
    )


# ---------------------------------------------------------------------------
# Test 1: default behaviour — exactly 5 sub-metrics, no D'/D''
# ---------------------------------------------------------------------------

def test_default_no_d_variants_byte_identical_to_pre_change_output(tmp_path):
    """No color_hist_anchor_path / clip_trajectory_path → sub_metrics has only
    the original 5 keys: appearance, temporal, identity, color_stability,
    color_slope.  D' / D'' must NOT appear."""
    out = _call_evaluate(tmp_path)
    assert set(out["sub_metrics"].keys()) == {
        "appearance", "temporal", "identity", "color_stability", "color_slope"
    }, out["sub_metrics"].keys()
    # diagnostics must NOT expose the new keys when not used
    assert "color_hist_anchor_used" in out["diagnostics"]
    assert "clip_trajectory_used" in out["diagnostics"]
    assert out["diagnostics"]["color_hist_anchor_used"] is False
    assert out["diagnostics"]["clip_trajectory_used"] is False


# ---------------------------------------------------------------------------
# Test 2: D' added and score correct
# ---------------------------------------------------------------------------

def test_dprime_added_when_path_provided(tmp_path):
    """color_hist_anchor path provided with known trajectory → score matches
    exp(-beta * |q3 - q0|) to 9 decimal places; beta_override stored."""
    trajectory = [0.10, 0.15, 0.20, 0.30]
    anchor_path = _write_color_hist_anchor(tmp_path, trajectory)

    out = _call_evaluate(tmp_path,
                         color_hist_anchor_path=anchor_path,
                         dprime_beta=0.5)

    assert "color_hist_anchor" in out["sub_metrics"], (
        "D' key missing; got: " + str(list(out["sub_metrics"].keys()))
    )
    dp = out["sub_metrics"]["color_hist_anchor"]

    expected_score = math.exp(-0.5 * abs(trajectory[3] - trajectory[0]))  # exp(-0.5*0.20)
    assert abs(dp["score"] - expected_score) < 1e-9, (
        f"score mismatch: {dp['score']} vs expected {expected_score}"
    )
    assert dp["details"]["beta_override"] == 0.5, dp["details"]

    # Also assert overall composite still runs fine
    assert 0.0 <= out["lr_vcc"] <= 1.0


# ---------------------------------------------------------------------------
# Test 3: D'' added and score correct
# ---------------------------------------------------------------------------

def test_dprime2_added_when_path_provided(tmp_path):
    """clip_trajectory path provided with known trajectory → score matches
    exp(-beta * |q3 - q0|); beta_override stored."""
    trajectory = [0.30, 0.28, 0.27, 0.25]
    clip_path = _write_clip_trajectory(tmp_path, trajectory)

    out = _call_evaluate(tmp_path,
                         clip_trajectory_path=clip_path,
                         dprime2_beta=3.0)

    assert "clip_trajectory" in out["sub_metrics"], (
        "D'' key missing; got: " + str(list(out["sub_metrics"].keys()))
    )
    dpp = out["sub_metrics"]["clip_trajectory"]

    expected_score = math.exp(-3.0 * abs(trajectory[3] - trajectory[0]))  # exp(-3.0*0.05)
    assert abs(dpp["score"] - expected_score) < 1e-9, (
        f"score mismatch: {dpp['score']} vs expected {expected_score}"
    )
    assert dpp["details"]["beta_override"] == 3.0, dpp["details"]

    assert 0.0 <= out["lr_vcc"] <= 1.0


# ---------------------------------------------------------------------------
# Test 4: missing trajectory_mean_per_quarter silently skips D'
# ---------------------------------------------------------------------------

def test_missing_trajectory_skips_d_variant(tmp_path):
    """anchor JSON exists but details.trajectory_mean_per_quarter absent →
    D' not added (no key in sub_metrics)."""
    p = tmp_path / "fake_color_hist_anchor.json"
    json.dump({
        "score": 0.7, "reliability": 0.9,
        "details": {},   # deliberately absent
    }, open(p, "w"))

    out = _call_evaluate(tmp_path,
                         color_hist_anchor_path=p,
                         dprime_beta=0.5)

    assert "color_hist_anchor" not in out["sub_metrics"], (
        "D' should be absent when trajectory missing; got: "
        + str(list(out["sub_metrics"].keys()))
    )
    # The other 5 must still be present
    assert set(out["sub_metrics"].keys()) == {
        "appearance", "temporal", "identity", "color_stability", "color_slope"
    }
