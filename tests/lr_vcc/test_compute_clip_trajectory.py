"""Tests for the pure helpers in compute_clip_trajectory.

CLIP itself isn't tested (no GPU and no model download in the local env);
we test cosine_distance and trajectory_score's math on synthetic embeddings
by monkey-patching _embed_frames.
"""
import numpy as np

from scripts.lr_vcc import compute_clip_trajectory as ct


def test_cosine_distance_identical_is_zero():
    v = np.array([0.3, 0.7, 0.1])
    assert ct.cosine_distance(v, v) < 1e-6


def test_cosine_distance_orthogonal_is_one():
    assert abs(ct.cosine_distance(np.array([1.0, 0.0]),
                                  np.array([0.0, 1.0])) - 1.0) < 1e-6


def test_trajectory_score_clean_video_scores_high(monkeypatch):
    monkeypatch.setattr(ct, "_load_sampled_frames",
                        lambda p, s: ["dummy"] * 100)
    monkeypatch.setattr(ct, "_embed_frames",
                        lambda fr, m, p, d, batch_size=32: np.tile(np.array([1.0, 0.0]), (len(fr), 1)))
    out = ct.trajectory_score("ignored", None, None, "cpu",
                              anchor_len=20, beta=5.0, stride=1)
    assert out["score"] > 0.99


def test_trajectory_score_drifting_video_scores_low(monkeypatch):
    monkeypatch.setattr(ct, "_load_sampled_frames",
                        lambda p, s: ["dummy"] * 100)
    embs = np.array([[1.0, 0.0]] * 20 + [[0.0, 1.0]] * 80)
    monkeypatch.setattr(ct, "_embed_frames",
                        lambda fr, m, p, d, batch_size=32: embs.astype(np.float32))
    out = ct.trajectory_score("ignored", None, None, "cpu",
                              anchor_len=20, beta=5.0, stride=1)
    assert out["score"] < 0.02
    assert out["details"]["mean_cos_dist_to_anchor"] > 0.95


def test_trajectory_score_too_short_returns_zero(monkeypatch):
    monkeypatch.setattr(ct, "_load_sampled_frames", lambda p, s: ["dummy"] * 2)
    out = ct.trajectory_score("ignored", None, None, "cpu",
                              anchor_len=60, beta=5.0, stride=1)
    assert out["score"] == 0.0
    assert out["reliability"] == 0.0
