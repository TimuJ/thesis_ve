"""Self-modifying midpoint flip artefacts for LR-VCC validation.

Six transforms with carefully chosen statistical-preservation properties:
- horizontal:      HorizontalFlip after T/2 (preserves histogram exactly)
- transpose:       rotate90 then resize-back after T/2 (preserves histogram)
- periodic:        HorizontalFlip every PERIODIC_BLOCK_FRAMES (preserves histogram)
- elastic:         deterministic Gaussian-filtered displacement remap after T/2 (~preserves histogram)
- channel_shuffle: BGR -> RGB (reverse last axis) after T/2 (preserves per-channel multisets)
- invert:          255 - x after T/2 (control: disrupts histogram)

Severity -> alpha = min(1.0, severity / 0.40). severity 0.40 = full transform,
severity 0.02 = 5% ghost overlay. Pre-midpoint frames always returned unchanged
(except for `periodic`, where alternating PERIODIC_BLOCK_FRAMES-frame blocks are
flipped instead of a single midpoint cut).

Pure cv2 + numpy. No reference assets needed. Works on any content (no Haar /
Detectron2 dependency) which is the entire point of this artefact family.
"""
from typing import Tuple

import cv2
import numpy as np


_PERIODIC_BLOCK_FRAMES = 30
_ELASTIC_ALPHA = 80.0
_ELASTIC_SIGMA = 8.0
_ELASTIC_SEED = 42
_ELASTIC_CACHE = {}  # (h, w) -> (map_x, map_y)


def _alpha_from_severity(severity: float) -> float:
    return max(0.0, min(1.0, severity / 0.40))


def _elastic_maps(h: int, w: int) -> Tuple[np.ndarray, np.ndarray]:
    key = (h, w)
    if key not in _ELASTIC_CACHE:
        rng = np.random.RandomState(_ELASTIC_SEED)
        dx = cv2.GaussianBlur(rng.uniform(-1, 1, (h, w)).astype(np.float32),
                              (0, 0), _ELASTIC_SIGMA) * _ELASTIC_ALPHA
        dy = cv2.GaussianBlur(rng.uniform(-1, 1, (h, w)).astype(np.float32),
                              (0, 0), _ELASTIC_SIGMA) * _ELASTIC_ALPHA
        gx, gy = np.meshgrid(np.arange(w, dtype=np.float32),
                             np.arange(h, dtype=np.float32))
        _ELASTIC_CACHE[key] = ((gx + dx).astype(np.float32),
                               (gy + dy).astype(np.float32))
    return _ELASTIC_CACHE[key]


def _pure_transform(frame_bgr: np.ndarray, transform: str) -> np.ndarray:
    """Return the fully-transformed frame (alpha = 1.0). Shape is preserved."""
    if transform == "horizontal":
        return cv2.flip(frame_bgr, 1)
    if transform == "transpose":
        h, w = frame_bgr.shape[:2]
        rotated = cv2.rotate(frame_bgr, cv2.ROTATE_90_CLOCKWISE)
        # rotate90 swaps dimensions; resize back to original (h, w) to keep shape stable
        return cv2.resize(rotated, (w, h), interpolation=cv2.INTER_LINEAR)
    if transform == "elastic":
        h, w = frame_bgr.shape[:2]
        map_x, map_y = _elastic_maps(h, w)
        return cv2.remap(frame_bgr, map_x, map_y,
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    if transform == "channel_shuffle":
        return frame_bgr[:, :, ::-1].copy()
    if transform == "invert":
        return (255 - frame_bgr.astype(np.int16)).astype(np.uint8)
    raise ValueError("unknown transform: " + transform)


def _blend(transformed: np.ndarray, original: np.ndarray, alpha: float) -> np.ndarray:
    if alpha >= 1.0:
        return transformed
    if alpha <= 0.0:
        return original
    return cv2.addWeighted(transformed, alpha, original, 1.0 - alpha, 0)


def apply_flip(frame_bgr: np.ndarray, idx: int, n_frames: int,
               transform: str, severity: float) -> np.ndarray:
    """Apply a self-modifying flip artefact.

    For transform != "periodic":
        idx < n_frames // 2: returns original.
        idx >= n_frames // 2: returns alpha * transformed + (1 - alpha) * original.

    For transform == "periodic":
        Block index = idx // PERIODIC_BLOCK_FRAMES.
        Even blocks: returns original. Odd blocks: alpha-blended HorizontalFlip.

    Raises ValueError on unknown transform name (validated even when severity == 0
    by computing alpha first then deferring to _pure_transform only when needed —
    but we still validate the transform name eagerly for safer pipeline behaviour).
    """
    # Validate transform name eagerly so misconfigured pipelines fail fast.
    valid = {"horizontal", "transpose", "periodic", "elastic",
             "channel_shuffle", "invert"}
    if transform not in valid:
        raise ValueError("unknown transform: " + transform)

    if severity <= 0 or n_frames <= 1:
        return frame_bgr

    alpha = _alpha_from_severity(severity)

    if transform == "periodic":
        block = idx // _PERIODIC_BLOCK_FRAMES
        if block % 2 == 0:
            return frame_bgr
        flipped = _pure_transform(frame_bgr, "horizontal")
        return _blend(flipped, frame_bgr, alpha)

    if idx < n_frames // 2:
        return frame_bgr

    transformed = _pure_transform(frame_bgr, transform)
    return _blend(transformed, frame_bgr, alpha)
