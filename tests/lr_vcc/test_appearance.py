from scripts.lr_vcc.appearance import appearance_score


def _fixture(qualities):
    return {"video_path": "/fake.mp4", "n_frames": len(qualities),
            "clip_iqa": qualities}


def test_high_mean_low_drift_high_score():
    out = appearance_score(_fixture([0.7] * 100))  # constant high quality
    assert out["score"] > 0.6
    assert out["reliability"] < 0.5  # drift too small => sub-metric undiscriminating


def test_high_mean_some_drift_high_score_high_rel():
    qs = [0.7 + 0.1 * (i % 2) for i in range(100)]  # std ~0.05
    out = appearance_score(_fixture(qs))
    assert out["score"] > 0.5
    assert out["reliability"] > 0.5


def test_low_mean_low_score():
    out = appearance_score(_fixture([0.2] * 100))
    assert out["score"] < 0.3


def test_lambda_penalizes_drift():
    high_drift = [0.5 + 0.4 * ((-1) ** i) for i in range(100)]  # std ~0.4
    out_lo_lambda = appearance_score(_fixture(high_drift), lam=0.0)
    out_hi_lambda = appearance_score(_fixture(high_drift), lam=2.0)
    assert out_lo_lambda["score"] > out_hi_lambda["score"]
