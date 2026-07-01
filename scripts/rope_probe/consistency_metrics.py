"""Score a perturbation condition against a reference (baseline output or GT)."""
import json
from src.evaluation.metrics import evaluate_sequence


def score_condition(pred_frames, ref_frames, compute_lpips=True):
    """Return PSNR/SSIM/(LPIPS)/temporal_consistency of pred vs ref frames."""
    return evaluate_sequence(pred_frames, ref_frames, compute_lpips=compute_lpips)


def write_condition_json(out_path, condition, scores_vs_baseline, scores_vs_gt):
    payload = {
        "condition": condition,
        "vs_baseline": scores_vs_baseline,
        "vs_gt": scores_vs_gt,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
