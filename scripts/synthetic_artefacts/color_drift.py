"""Gradual color cast accumulating over the full video duration."""
import numpy as np


def apply_color_drift(frame: np.ndarray, frame_idx: int, total_frames: int,
                      severity: float) -> np.ndarray:
    """Apply a gradual red-warming color drift to a single frame.

    Parameters
    ----------
    frame : np.ndarray
        (H, W, 3) BGR uint8 array (OpenCV format).
    frame_idx : int
        Zero-based index of this frame in the video.
    total_frames : int
        Total number of frames in the video.
    severity : float
        Maximum drift magnitude at the last frame, in [0, 1].
        0.40 produces an obvious warming tint by the end.

    Returns
    -------
    np.ndarray
        (H, W, 3) BGR uint8 array with drift applied.
    """
    if total_frames <= 1 or severity == 0.0:
        return frame.copy()
    drift = severity * (frame_idx / (total_frames - 1))
    f = frame.astype(np.float32)
    # OpenCV BGR: idx 0=B, 1=G, 2=R
    f[..., 2] *= 1.0 + drift          # R up
    f[..., 1] *= 1.0 - drift / 2.0    # G down
    f[..., 0] *= 1.0 - drift / 2.0    # B down
    return np.clip(f, 0, 255).astype(np.uint8)
