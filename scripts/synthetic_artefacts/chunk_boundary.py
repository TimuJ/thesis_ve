"""Per-chunk additive offset — creates brightness jumps at every chunk boundary."""
import random
import numpy as np


def _chunk_offset(chunk_idx: int, severity: float) -> float:
    """Return a deterministic offset for a given chunk index and severity.

    The offset is in [-severity, +severity].  The same (chunk_idx, severity)
    pair always produces the same value (reproducible across runs).
    """
    seed = chunk_idx * 12345 + 67
    rng = random.Random(seed)
    return severity * rng.uniform(-1.0, 1.0)


def apply_chunk_boundary_jumps(frame: np.ndarray, frame_idx: int,
                                chunk_size_frames: int, severity: float) -> np.ndarray:
    """Add a per-chunk uniform brightness offset to every pixel in the frame.

    Parameters
    ----------
    frame : np.ndarray
        (H, W, 3) BGR uint8 array (OpenCV format).
    frame_idx : int
        Zero-based index of this frame in the video.
    chunk_size_frames : int
        Number of frames per chunk (default 60 = 2 s at 30 fps).
    severity : float
        Maximum offset magnitude as a fraction of 255.
        0.40 means the per-chunk offset is in [-102, +102] grey levels.

    Returns
    -------
    np.ndarray
        (H, W, 3) BGR uint8 array with the chunk offset applied and clipped.
    """
    if severity == 0.0:
        return frame.copy()
    chunk_idx = frame_idx // max(1, chunk_size_frames)
    offset = _chunk_offset(chunk_idx, severity) * 255.0
    f = frame.astype(np.float32) + offset
    return np.clip(f, 0, 255).astype(np.uint8)
