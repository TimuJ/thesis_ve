"""Synthetic artefact: progressive whole-frame Gaussian blur.

Companion to identity_degradation (which blurs only detected faces). This one
blurs the ENTIRE frame, to test the blur-inversion hypothesis on global frame
metrics: destroying high-frequency detail lowers adjacent-frame differences and
makes interpolation easier, so a self-referential temporal/smoothness metric is
expected to RISE (rate the more-degraded video as better) as severity grows.

Mechanism (matched to identity_degradation for comparability):
    sigma  = severity * 10.0
    kernel = odd integer ~= 6*sigma + 1 (clamped >= 3)
    GaussianBlur the whole frame. severity=0 returns the frame unchanged;
    severity=0.40 -> sigma=4.0, a clearly visible global blur.
"""
import cv2
import numpy as np


def apply_global_blur(frame_bgr: np.ndarray, idx: int, severity: float) -> np.ndarray:
    """Gaussian-blur the whole frame by sigma = severity * 10.0.

    idx is unused (kept for API parity with the other artefact appliers).
    """
    if severity <= 0.0:
        return frame_bgr.copy()
    sigma = float(severity) * 10.0
    k = max(3, int(round(sigma * 6)) | 1)  # odd kernel
    return cv2.GaussianBlur(frame_bgr, (k, k), sigmaX=sigma, sigmaY=sigma)
