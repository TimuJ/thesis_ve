import json
from pathlib import Path
from scripts.lr_vcc.run_lr_vcc import evaluate_one_video


def _make_fixture(tmp_path: Path, clip_iqa, tof, mask_cov, id_pv):
    clip_iqa_file = tmp_path / "clip_iqa.json"
    tof_file = tmp_path / "tof.json"
    id_file = tmp_path / "id.json"
    json.dump({"video_path": "/fake.mp4", "n_frames": 100, "fps": 30.0,
               "frame_stride": 1, "clip_iqa": clip_iqa}, open(clip_iqa_file, "w"))
    json.dump({"video_path": "/fake.mp4", "n_frames": 100, "fps": 30.0,
               "k_values": list(tof.keys()), "tof": {str(k): v for k, v in tof.items()},
               "tlp": {str(k): 0.0 for k in tof},
               "n_pairs_used": {str(k): 200 for k in tof},
               "mean_mask_coverage": {str(k): v for k, v in mask_cov.items()}},
              open(tof_file, "w"))
    json.dump({"per_video": {"fake": id_pv}}, open(id_file, "w"))
    return clip_iqa_file, tof_file, id_file


def test_good_video_high_lr_vcc(tmp_path):
    clip_iqa = [0.7 + 0.05 * (i % 3) for i in range(100)]
    tof = {1: 0.02, 5: 0.04, 10: 0.05, 30: 0.07, 60: 0.10, 120: 0.13}
    cov = {1: 0.9, 5: 0.7, 10: 0.5, 30: 0.4, 60: 0.3, 120: 0.2}
    id_pv = {"slow": 0.8, "fast": 0.7, "fused": 0.75,
             "n_clips": 50, "n_clips_with_faces": 40}
    fa, ft, fi = _make_fixture(tmp_path, clip_iqa, tof, cov, id_pv)
    out = evaluate_one_video(video_id="fake", clip_iqa_path=fa, tof_path=ft,
                             identity_results_path=fi, closeup_bbox_p50=0.03)
    assert out["lr_vcc"] > 0.5
    assert not out["low_confidence"]


def test_low_face_rate_downweights_identity(tmp_path):
    clip_iqa = [0.7] * 100
    tof = {1: 0.02, 5: 0.04, 10: 0.05, 30: 0.07, 60: 0.10, 120: 0.13}
    cov = {1: 0.9, 5: 0.7, 10: 0.5, 30: 0.4, 60: 0.3, 120: 0.2}
    id_pv = {"slow": 0.3, "fast": 0.2, "fused": 0.25,
             "n_clips": 50, "n_clips_with_faces": 5}
    fa, ft, fi = _make_fixture(tmp_path, clip_iqa, tof, cov, id_pv)
    out = evaluate_one_video(video_id="fake", clip_iqa_path=fa, tof_path=ft,
                             identity_results_path=fi, closeup_bbox_p50=0.03)
    assert out["sub_metrics"]["identity"]["reliability"] < 0.3
    assert out["lr_vcc"] > 0.4
