"""Tests for sub-metric I (identity — slow-fast + face-rate + closeup reliability)."""
from scripts.lr_vcc.identity import identity_score


def _fixture_id_per_video(slow, fast, fused, n_clips, n_faces):
    """Matches the per_video[video] shape from human_identity_long.py."""
    return {"slow": slow, "fast": fast, "fused": fused,
            "n_clips": n_clips, "n_clips_with_faces": n_faces}


def test_high_id_high_face_rate_high_score_high_rel():
    pv = _fixture_id_per_video(0.7, 0.6, 0.65, n_clips=80, n_faces=60)
    out = identity_score(pv, closeup_bbox_p50=None)
    assert out["score"] == 0.65
    assert out["reliability"] > 0.85


def test_low_face_rate_drops_reliability():
    pv = _fixture_id_per_video(0.6, 0.5, 0.55, n_clips=80, n_faces=8)  # 10% rate < 20% floor
    out = identity_score(pv, closeup_bbox_p50=None)
    assert out["score"] == 0.55
    assert out["reliability"] < 0.5


def test_closeup_partial_downweight():
    pv = _fixture_id_per_video(0.7, 0.6, 0.65, n_clips=80, n_faces=60)
    out_no_closeup = identity_score(pv, closeup_bbox_p50=0.01)
    out_closeup = identity_score(pv, closeup_bbox_p50=0.18)  # well above 0.05 threshold
    assert out_closeup["reliability"] < out_no_closeup["reliability"]
