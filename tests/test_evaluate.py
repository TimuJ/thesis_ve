"""Tests for baselines evaluate.py frame loading and metric aggregation."""
import json
import os
import tempfile
import numpy as np
from PIL import Image
import pytest


def _create_test_dataset(root, clip_names, num_frames=3, size=(64, 64)):
    """Create a fake dataset with GT and results frame dirs."""
    gt_dir = os.path.join(root, "gt")
    res_dir = os.path.join(root, "results")
    for clip in clip_names:
        os.makedirs(os.path.join(gt_dir, clip), exist_ok=True)
        os.makedirs(os.path.join(res_dir, clip), exist_ok=True)
        for i in range(num_frames):
            img = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
            Image.fromarray(img).save(os.path.join(gt_dir, clip, f"{i:08d}.png"))
            Image.fromarray(img).save(os.path.join(res_dir, clip, f"{i:08d}.png"))
    return gt_dir, res_dir


def test_evaluate_produces_json():
    """evaluate.py should produce a valid JSON with expected keys."""
    with tempfile.TemporaryDirectory() as tmpdir:
        gt_dir, res_dir = _create_test_dataset(tmpdir, ["clip_a", "clip_b"])
        out_json = os.path.join(tmpdir, "metrics.json")

        import importlib.util
        from pathlib import Path
        eval_path = str(Path(__file__).resolve().parents[1] / "experiments" / "baselines" / "evaluate.py")
        spec = importlib.util.spec_from_file_location(
            "evaluate", eval_path
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        mod.main(["--results", res_dir, "--gt", gt_dir, "--output", out_json, "--no-lpips"])

        assert os.path.exists(out_json)
        with open(out_json) as f:
            data = json.load(f)
        assert "overall" in data
        assert "per_clip" in data
        assert "PSNR_mean" in data["overall"]
        assert "SSIM_mean" in data["overall"]
        assert len(data["per_clip"]) == 2


def test_evaluate_identical_frames_high_psnr():
    """Identical GT and results should produce high PSNR."""
    with tempfile.TemporaryDirectory() as tmpdir:
        gt_dir, res_dir = _create_test_dataset(tmpdir, ["clip_a"])
        out_json = os.path.join(tmpdir, "metrics.json")

        import importlib.util
        from pathlib import Path
        eval_path = str(Path(__file__).resolve().parents[1] / "experiments" / "baselines" / "evaluate.py")
        spec = importlib.util.spec_from_file_location(
            "evaluate", eval_path
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        mod.main(["--results", res_dir, "--gt", gt_dir, "--output", out_json, "--no-lpips"])

        with open(out_json) as f:
            data = json.load(f)
        assert data["overall"]["PSNR_mean"] > 50 or data["overall"]["PSNR_mean"] == float("inf")
