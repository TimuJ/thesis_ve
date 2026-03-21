"""
Evaluation metrics for video super-resolution.
Computes: PSNR, SSIM, temporal consistency (tOF).

All metrics operate on images as (H, W, C) uint8 or float arrays.
"""
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio between predicted and ground-truth frames."""
    return float(peak_signal_noise_ratio(gt, pred))


def ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    """Structural Similarity Index between predicted and ground-truth frames."""
    channel_axis = 2 if pred.ndim == 3 else None
    return float(structural_similarity(gt, pred, channel_axis=channel_axis))


def temporal_consistency(frames: list[np.ndarray]) -> float:
    """
    Temporal consistency via mean absolute difference between consecutive frames.
    Lower = more consistent. Returns mean over sequence.
    """
    if len(frames) < 2:
        return 0.0
    diffs = []
    for i in range(len(frames) - 1):
        diff = np.abs(frames[i].astype(float) - frames[i + 1].astype(float)).mean()
        diffs.append(diff)
    return float(np.mean(diffs))


def evaluate_sequence(
    preds: list[np.ndarray],
    gts: list[np.ndarray],
) -> dict:
    """
    Evaluate a single video sequence.

    Args:
        preds: List of predicted HR frames (H, W, C)
        gts: List of ground-truth HR frames (H, W, C)

    Returns:
        Dict with PSNR_mean, SSIM_mean, per-frame scores, temporal_consistency
    """
    assert len(preds) == len(gts), f"Mismatch: {len(preds)} preds vs {len(gts)} gts"

    psnr_scores = [psnr(p, g) for p, g in zip(preds, gts)]
    ssim_scores = [ssim(p, g) for p, g in zip(preds, gts)]
    t_consist = temporal_consistency(preds)

    return {
        "PSNR_mean": float(np.mean(psnr_scores)),
        "SSIM_mean": float(np.mean(ssim_scores)),
        "PSNR_per_frame": psnr_scores,
        "SSIM_per_frame": ssim_scores,
        "temporal_consistency": t_consist,
    }


def evaluate_dataset(
    all_preds: dict[str, list[np.ndarray]],
    all_gts: dict[str, list[np.ndarray]],
) -> dict:
    """
    Evaluate across all sequences in a dataset.

    Args:
        all_preds: {sequence_name: [pred_frames]}
        all_gts: {sequence_name: [gt_frames]}

    Returns:
        Dict with per-sequence and overall metrics
    """
    per_sequence = {}
    psnr_means, ssim_means = [], []

    for seq_name in all_gts:
        if seq_name not in all_preds:
            print(f"Warning: sequence '{seq_name}' missing from predictions, skipping")
            continue
        result = evaluate_sequence(all_preds[seq_name], all_gts[seq_name])
        per_sequence[seq_name] = result
        psnr_means.append(result["PSNR_mean"])
        ssim_means.append(result["SSIM_mean"])

    overall_psnr = float(np.mean(psnr_means)) if psnr_means else 0.0
    overall_ssim = float(np.mean(ssim_means)) if ssim_means else 0.0

    return {
        "overall": {
            "PSNR_mean": overall_psnr,
            "SSIM_mean": overall_ssim,
            "num_sequences": len(per_sequence),
        },
        "per_sequence": per_sequence,
    }
