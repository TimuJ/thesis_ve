"""Pair scorers: the interface, and the CPU SIFT texture scorer.

A PairScorer compares two object crops and returns a similarity in [0, 1]. The
selection embedding (Task 7) is deliberately a different model from these scorers.
GPU scorers (DINOv3 semantic, LoFTR geometry) live in scorers_gpu (Task 7); SIFT
is pure CPU OpenCV and lives here.
"""
from __future__ import annotations

from typing import Protocol

import cv2
import numpy as np

from .config import SIFT_RELIABILITY_KP


class PairScorer(Protocol):
    name: str

    def __call__(self, crop_a: np.ndarray, crop_b: np.ndarray) -> float: ...

    # Optional: GT-side signal in [0, 1] — how much this axis can measure on a crop
    # pair (e.g. texture / keypoint count). Computed on GT crops to weight the axis in
    # the composite (see reliability_axes.axis_reliabilities). Absent => treated as 1.0.
    # def reliability(self, crop_a: np.ndarray, crop_b: np.ndarray) -> float: ...


def _gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img


class SiftTextureScorer:
    """Texture consistency = Lowe-ratio good-match count / min(#keypoints)."""

    name = "sift"

    def __init__(self, ratio: float = 0.75):
        self._sift = cv2.SIFT_create()
        self._bf = cv2.BFMatcher()
        self._ratio = ratio

    def __call__(self, crop_a: np.ndarray, crop_b: np.ndarray) -> float:
        ka, da = self._sift.detectAndCompute(_gray(crop_a), None)
        kb, db = self._sift.detectAndCompute(_gray(crop_b), None)
        if da is None or db is None or len(ka) < 2 or len(kb) < 2:
            return 0.0
        good = 0
        for pair in self._bf.knnMatch(da, db, k=2):
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self._ratio * n.distance:
                good += 1
        # good counts query (da) descriptors with a confident match; several may map
        # to the same kb keypoint, so good can exceed min(#kp) for repetitive texture.
        # Clamp to [0, 1] -- values above 1 all mean "saturated texture match", and an
        # unbounded score breaks the consistency formula s_M = max(0, 1 - |c_sr - c_gt|).
        denom = min(len(ka), len(kb))
        return float(min(1.0, good / denom)) if denom else 0.0

    def reliability(self, crop_a: np.ndarray, crop_b: np.ndarray) -> float:
        """GT-side texture signal = min keypoint count / SIFT_RELIABILITY_KP, clamped to
        [0, 1]. A low-texture crop (few keypoints) carries little SIFT signal, so the
        SIFT axis is down-weighted there instead of free-passing on c_gt ~ 0."""
        ka = self._sift.detect(_gray(crop_a), None)
        kb = self._sift.detect(_gray(crop_b), None)
        return float(min(1.0, min(len(ka), len(kb)) / SIFT_RELIABILITY_KP))
