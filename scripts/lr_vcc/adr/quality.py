"""Per-frame quality signals used to avoid bad review pairs (reviewer feedback).

Two CPU signals on the GT object crop/box, both frozen on GT like the rest of
selection:

* **sharpness** — variance of the Laplacian; a motion-blurred frame has a low
  value. Turned into a *relative* blur penalty against the clip's own median, so
  it is scale-free (no absolute threshold) and only fires when a frame is softer
  than its neighbours — a uniformly soft clip is not penalised, there being
  nothing sharper to prefer.
* **edge proximity** — how close the box sits to a frame border, *scaled by how
  small the object is*, so a small object on the edge is penalised while a big
  object that legitimately spans the edge (a bus at the clip start) is not.
"""
from __future__ import annotations

import cv2
import numpy as np

from .config import BIG_AREA_FRAC, EDGE_MARGIN_FRAC
from .weak_anchor import size_reliability


def laplacian_var(crop: np.ndarray) -> float:
    """Variance of the Laplacian of a crop (grayscale); lower = blurrier."""
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def blur_penalties(lap_var, present_idx) -> np.ndarray:
    """Per-frame blur penalty in [0, 1], 0 for frames at/above the clip's median
    sharpness and rising toward 1 as a frame falls below it. Absent frames (and all
    frames when the clip cannot discriminate, e.g. a flat median) get 0."""
    lap_var = np.asarray(lap_var, float)
    pen = np.zeros(len(lap_var))
    vals = np.array([lap_var[i] for i in present_idx], float)
    med = float(np.median(vals)) if len(vals) else 0.0
    if med <= 0:
        return pen
    for i in present_idx:
        pen[i] = max(0.0, min(1.0, 1.0 - lap_var[i] / med))
    return pen


def edge_penalty(box, frame_w: int, frame_h: int, area: float,
                 margin_frac: float = EDGE_MARGIN_FRAC, big_area: float = BIG_AREA_FRAC) -> float:
    """Edge-proximity penalty in [0, 1]: high when a *small* box sits within
    ``margin_frac`` of a frame border, zero for a centred box or a big object."""
    x, y, w, h = box
    margins = (x / frame_w, (frame_w - (x + w)) / frame_w,
               y / frame_h, (frame_h - (y + h)) / frame_h)
    min_margin = min(margins)
    closeness = max(0.0, min(1.0, (margin_frac - min_margin) / margin_frac)) if margin_frac > 0 else 0.0
    smallness = 1.0 - size_reliability(area, big_area)
    return float(closeness * smallness)


def _gray(crop: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop


# --- P4: detail / contrast (reviewer feedback distinguished "small & clear" ----
# objects we should keep (gecko, readable plate) from "small & dark" ones we should
# cut (mouse). Laplacian sharpness could not: it tracks clutter, not legibility. ---

def rms_contrast(crop: np.ndarray) -> float:
    """RMS contrast in [0, 1]: standard deviation of grayscale intensities / 255.
    ~0 for a flat or uniformly dark crop, high for one with strong light/dark
    structure. Absolute (not clip-relative): legibility is meaningful per crop."""
    return float(np.std(_gray(crop).astype(np.float64)) / 255.0)


def shannon_entropy(crop: np.ndarray) -> float:
    """Shannon entropy of the 256-bin grayscale histogram, normalised to [0, 1]
    (bits / 8). ~0 for a flat/low-detail crop, high for a richly textured one."""
    g = _gray(crop).astype(np.uint8).ravel()
    hist = np.bincount(g, minlength=256).astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 0.0
    p = hist / total
    nz = p[p > 0]
    return float(-(nz * np.log2(nz)).sum() / 8.0)


def detail_signal(crop: np.ndarray) -> float:
    """Reviewability-detail signal in [0, 1]: geometric mean of RMS contrast and
    normalised entropy, so a crop must have *both* tonal range and structure to
    score high. High = legible enough to review (gecko, readable plate); low = a
    flat/dark crop (the 'small & dark' mouse). Higher is better (unlike a penalty)."""
    return float(np.sqrt(max(0.0, rms_contrast(crop)) * max(0.0, shannon_entropy(crop))))


# --- P3: optical-flow motion (what the Laplacian missed for motion blur -- a fast ---
# object is blurred regardless of its texture-variance, so measure its motion). ---

def flow_magnitude(prev_crop: np.ndarray, next_crop: np.ndarray) -> float:
    """Mean dense optical-flow magnitude (pixels) between two crops (Farneback, CPU).
    ~0 for a static object, large for one that moves or deforms between the frames.
    The two crops are brought to a common size if they differ."""
    g0, g1 = _gray(prev_crop).astype(np.uint8), _gray(next_crop).astype(np.uint8)
    if g0.shape != g1.shape:
        g1 = cv2.resize(g1, (g0.shape[1], g0.shape[0]))
    flow = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    return float(mag.mean())


def motion_signal(prev_crop: np.ndarray, next_crop: np.ndarray, box_diag: float) -> float:
    """Scale-free motion signal in [0, 1]: mean flow magnitude as a fraction of the
    object's extent (box diagonal). The same pixel motion matters more for a small
    object, so dividing by size makes a tiny jitter on a large object cheap and a
    large shift on a small object expensive. High = motion-blur-prone / pose-unstable.
    A degenerate box (``box_diag <= 0``) returns 0 rather than dividing."""
    if box_diag <= 0:
        return 0.0
    return float(max(0.0, min(1.0, flow_magnitude(prev_crop, next_crop) / box_diag)))


def reblur_blur(crop: np.ndarray) -> float:
    """No-reference blur in [0, 1] (Crete, Dolmiere, Ladret & Nicolas, 2007): low-pass
    the crop again and measure how much its neighbour-to-neighbour variation drops. A
    sharp crop loses a lot of variation when re-blurred (score near 0); an already
    blurry one barely changes (score near 1). Unlike the Laplacian this is a direct
    blur estimate, not a texture/clutter proxy. Flat crops (no variation) return 0."""
    g = _gray(crop).astype(np.float64)
    if g.shape[0] < 2 or g.shape[1] < 2:
        return 0.0
    k = (np.ones(9) / 9.0)
    b_h_img = cv2.filter2D(g, -1, k.reshape(1, 9))
    b_v_img = cv2.filter2D(g, -1, k.reshape(9, 1))
    d_f_h, d_b_h = np.abs(np.diff(g, axis=1)), np.abs(np.diff(b_h_img, axis=1))
    d_f_v, d_b_v = np.abs(np.diff(g, axis=0)), np.abs(np.diff(b_v_img, axis=0))
    s_f_h, s_f_v = d_f_h.sum(), d_f_v.sum()
    b_h = (s_f_h - np.maximum(0.0, d_f_h - d_b_h).sum()) / s_f_h if s_f_h > 0 else 0.0
    b_v = (s_f_v - np.maximum(0.0, d_f_v - d_b_v).sum()) / s_f_v if s_f_v > 0 else 0.0
    return float(max(b_h, b_v))
