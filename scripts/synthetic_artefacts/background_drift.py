"""Synthetic artefact: progressive background drift toward a reference scene.

Tests the second half of the "long-term consistency" framing: the subject
(detected human silhouette) stays unchanged, but everything else in the
frame slowly shifts to a different scene over the duration of the video.

Mechanism:
    For each frame i of T:
        blend = severity * (i / (T - 1))      # 0 at frame 0, severity at frame T-1
    The entire frame is alpha-blended toward the reference background image;
    then the original pixels are restored at every location flagged as
    "human" by the per-frame mask (precomputed via Detectron2 Mask R-CNN —
    see `precompute_human_masks.py`).

severity = 0 leaves the video unchanged.
severity = 1.0 means the non-human region is fully replaced by the reference
scene at the final frame.

The per-frame masks are supplied by the caller (typically the
`generate_all.py` driver, which loads a cached `.npz` produced by
`precompute_human_masks.py`). When no mask is available (mask is None or
empty for that frame), the artefact treats the entire frame as background
and blends everything — appropriate for frames where no human is present.
"""
from typing import Optional

import cv2
import numpy as np


def apply_background_drift(frame_bgr: np.ndarray,
                            idx: int,
                            n_frames: int,
                            reference_bg_bgr: np.ndarray,
                            human_mask: Optional[np.ndarray],
                            severity: float) -> np.ndarray:
    """Blend the background of the frame toward `reference_bg_bgr`, preserving
    pixels flagged as human by `human_mask`.

    Parameters
    ----------
    frame_bgr : np.ndarray
        (H, W, 3) BGR uint8 input frame.
    idx : int
        Zero-based frame index.
    n_frames : int
        Total number of frames in the video.
    reference_bg_bgr : np.ndarray
        (H_ref, W_ref, 3) BGR uint8 reference scene image. Resized to the
        frame dimensions at composition time.
    human_mask : np.ndarray or None
        (H, W) bool array. True at pixels to preserve (human silhouette);
        False at pixels to blend. If None, the entire frame is blended.
    severity : float
        In [0, 1]. severity = 0 returns the frame unchanged.
        severity = 1.0 means the background is fully replaced at the final frame.

    Returns
    -------
    np.ndarray
        (H, W, 3) BGR uint8 array of the same shape as input.
    """
    if severity <= 0.0 or n_frames <= 1:
        return frame_bgr.copy()

    blend = float(severity) * float(idx) / float(n_frames - 1)
    blend = max(0.0, min(1.0, blend))
    if blend <= 0.0:
        return frame_bgr.copy()

    h, w = frame_bgr.shape[:2]
    ref_resized = cv2.resize(
        reference_bg_bgr, (w, h),
        interpolation=cv2.INTER_AREA,
    )
    full_blend = cv2.addWeighted(frame_bgr, 1.0 - blend, ref_resized, blend, 0.0)

    if human_mask is None or not human_mask.any():
        # No human detected this frame; treat the entire frame as background.
        return full_blend

    out = full_blend
    out[human_mask] = frame_bgr[human_mask]
    return out


def load_packed_masks(npz_path: str) -> np.ndarray:
    """Load and unpack the per-frame human masks written by
    `precompute_human_masks.py`. Returns a (n_frames, H, W) bool array."""
    data = np.load(npz_path)
    packed = data["masks_packed"]
    shape = tuple(data["shape"])
    flat_len = shape[1] * shape[2]
    flat = np.unpackbits(packed, axis=-1)[..., :flat_len]
    return flat.reshape(shape).astype(bool)
