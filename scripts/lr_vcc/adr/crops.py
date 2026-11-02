"""Object crop normalization.

bbox crop + resize + box rescale are pure/CPU. The SAM mask-on-white upgrade is a
GPU adapter and lives with the other model adapters (Task 7 / scorers_gpu), so this
module imports without torch.
"""
from __future__ import annotations

import cv2
import numpy as np


def bbox_crop(frame: np.ndarray, box):
    """Crop frame to an (x, y, w, h) box, clamped to frame bounds. Returns None for
    a zero-area or fully out-of-bounds box."""
    H, W = frame.shape[:2]
    x, y, w, h = (int(round(v)) for v in box)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return None
    return frame[y0:y1, x0:x1]


def resize_crop(crop: np.ndarray, short_side: int) -> np.ndarray:
    """Aspect-preserving resize so the shorter side equals short_side."""
    h, w = crop.shape[:2]
    s = short_side / min(h, w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    return cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)


def rescale_box(box, from_hw, to_hw):
    """Scale an (x, y, w, h) box from one frame grid (H, W) to another."""
    fh, fw = from_hw
    th, tw = to_hw
    sx, sy = tw / fw, th / fh
    x, y, w, h = box
    return (x * sx, y * sy, w * sx, h * sy)
