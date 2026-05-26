"""Periodic sinusoidal brightness flicker - mimics diffusion attention-reset artefacts."""
import math
import numpy as np


def apply_periodic_flicker(frame: np.ndarray, frame_idx: int,
                            period_frames: int, severity: float) -> np.ndarray:
    """Apply multiplicative brightness modulation with sinusoidal periodicity.

    Parameters
    ----------
    frame : np.ndarray
        (H, W, 3) BGR uint8 array (OpenCV format).
    frame_idx : int
        Zero-based index of this frame in the video.
    period_frames : int
        Oscillation period in frames (default 15 ≈ 0.5 sec at 30 fps).
    severity : float
        Peak relative modulation magnitude, in [0, 1].
        0.02 produces ±5 grey levels modulation.
        0.40 produces ±102 grey levels modulation.

    Returns
    -------
    np.ndarray
        (H, W, 3) BGR uint8 array with periodic flicker applied.

    Notes
    -----
    Brightness modulation: out = clip(frame * (1 + severity * sin(2π * frame_idx / period_frames)), 0, 255).
    Mimics periodic resets in long-video diffusion model attention states.
    """
    if severity == 0.0 or period_frames <= 0:
        return frame.copy()
    mod = 1.0 + severity * math.sin(2.0 * math.pi * frame_idx / period_frames)
    f = frame.astype(np.float32) * mod
    return np.clip(f, 0, 255).astype(np.uint8)
