"""
Evaluation metrics for video super-resolution.
Computes: PSNR, SSIM, temporal consistency (tOF).

All metrics operate on images as (H, W, C) uint8 or float arrays.
"""
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

try:
    import torch
    import lpips as lpips_lib
    _lpips_net = None

    def _get_lpips_net():
        global _lpips_net
        if _lpips_net is None:
            _lpips_net = lpips_lib.LPIPS(net="alex")
            if torch.cuda.is_available():
                _lpips_net = _lpips_net.cuda()
        return _lpips_net

    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False


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


def lpips_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Learned Perceptual Image Patch Similarity. Lower = more similar.
    Requires torch and lpips packages. Images should be (H, W, C) uint8."""
    if not HAS_LPIPS:
        raise RuntimeError("lpips not available — install torch and lpips")
    net = _get_lpips_net()
    device = next(net.parameters()).device

    def _to_tensor(img):
        t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        t = t * 2.0 - 1.0  # normalize to [-1, 1]
        return t.unsqueeze(0).to(device)

    with torch.no_grad():
        score = net(_to_tensor(pred), _to_tensor(gt))
    return float(score.item())


def evaluate_sequence(
    preds: list[np.ndarray],
    gts: list[np.ndarray],
    compute_lpips: bool = True,
) -> dict:
    """
    Evaluate a single video sequence.

    Args:
        preds: List of predicted HR frames (H, W, C)
        gts: List of ground-truth HR frames (H, W, C)
        compute_lpips: Whether to compute LPIPS (requires torch + lpips packages)

    Returns:
        Dict with PSNR_mean, SSIM_mean, per-frame scores, temporal_consistency,
        and optionally LPIPS_mean if compute_lpips=True and lpips is available.
    """
    assert len(preds) == len(gts), f"Mismatch: {len(preds)} preds vs {len(gts)} gts"

    psnr_scores = [psnr(p, g) for p, g in zip(preds, gts)]
    ssim_scores = [ssim(p, g) for p, g in zip(preds, gts)]
    t_consist = temporal_consistency(preds)

    result = {
        "PSNR_mean": float(np.mean(psnr_scores)),
        "SSIM_mean": float(np.mean(ssim_scores)),
        "PSNR_per_frame": psnr_scores,
        "SSIM_per_frame": ssim_scores,
        "temporal_consistency": t_consist,
    }

    if compute_lpips and HAS_LPIPS:
        lpips_scores = [lpips_score(p, g) for p, g in zip(preds, gts)]
        result["LPIPS_mean"] = float(np.mean(lpips_scores))
        result["LPIPS_per_frame"] = lpips_scores

    return result


def evaluate_dataset(
    all_preds: dict[str, list[np.ndarray]],
    all_gts: dict[str, list[np.ndarray]],
    compute_lpips: bool = True,
) -> dict:
    """
    Evaluate across all sequences in a dataset.

    Args:
        all_preds: {sequence_name: [pred_frames]}
        all_gts: {sequence_name: [gt_frames]}
        compute_lpips: Whether to compute LPIPS

    Returns:
        Dict with per-sequence and overall metrics
    """
    per_sequence = {}
    psnr_means, ssim_means, lpips_means = [], [], []

    for seq_name in all_gts:
        if seq_name not in all_preds:
            print(f"Warning: sequence '{seq_name}' missing from predictions, skipping")
            continue
        result = evaluate_sequence(all_preds[seq_name], all_gts[seq_name], compute_lpips=compute_lpips)
        per_sequence[seq_name] = result
        psnr_means.append(result["PSNR_mean"])
        ssim_means.append(result["SSIM_mean"])
        if "LPIPS_mean" in result:
            lpips_means.append(result["LPIPS_mean"])

    overall = {
        "PSNR_mean": float(np.mean(psnr_means)) if psnr_means else 0.0,
        "SSIM_mean": float(np.mean(ssim_means)) if ssim_means else 0.0,
        "num_sequences": len(per_sequence),
    }
    if lpips_means:
        overall["LPIPS_mean"] = float(np.mean(lpips_means))

    return {
        "overall": overall,
        "per_sequence": per_sequence,
    }
