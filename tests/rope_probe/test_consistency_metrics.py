# tests/rope_probe/test_consistency_metrics.py
import json
import pytest
import numpy as np
from scripts.rope_probe.consistency_metrics import (
    score_condition, write_condition_json,
)


def _frames(n, val):
    return [np.full((16, 16, 3), val, dtype=np.uint8) for _ in range(n)]


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_identical_frames_score_infinite_psnr():
    # Identical frames give zero MSE, causing skimage's peak_signal_noise_ratio
    # to return inf and emit a divide-by-zero RuntimeWarning, which we ignore
    # via the filterwarnings mark. Assert > 90.0 holds for inf.
    a = _frames(3, 100)
    out = score_condition(a, a, compute_lpips=False)
    assert out["PSNR_mean"] > 90.0
    assert out["SSIM_mean"] == 1.0


def test_frame_count_mismatch_raises():
    with pytest.raises(AssertionError):
        score_condition(_frames(3, 100), _frames(2, 100), compute_lpips=False)


def test_write_condition_json_shape(tmp_path):
    p = tmp_path / "cond.json"
    write_condition_json(
        str(p),
        condition={"shift": 100, "stretch": 1.0},
        scores_vs_baseline={"PSNR_mean": 42.0},
        scores_vs_gt=None,
    )
    payload = json.loads(p.read_text())
    assert payload["condition"]["shift"] == 100
    assert payload["vs_baseline"]["PSNR_mean"] == 42.0
    assert payload["vs_gt"] is None
