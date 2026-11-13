"""Frame-quality signals from reviewer feedback: sharpness (motion blur) and
size-scaled edge proximity. Pure CPU (cv2/numpy)."""
import cv2
import numpy as np

from scripts.lr_vcc.adr.quality import (
    blur_penalties,
    detail_signal,
    edge_penalty,
    flow_magnitude,
    laplacian_var,
    motion_signal,
    reblur_blur,
    rms_contrast,
    shannon_entropy,
)


def _texture(seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)


def _smooth(seed=0, size=96):
    """A trackable textured frame (blurred noise keeps brightness gradients that
    Farneback can follow, unlike pure salt-and-pepper noise)."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
    return cv2.GaussianBlur(base, (0, 0), 1.5)


def _shift(img, dx, dy):
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REPLICATE)


def test_laplacian_var_drops_when_blurred():
    sharp = _texture()
    blurred = cv2.GaussianBlur(sharp, (9, 9), 4)
    assert laplacian_var(sharp) > laplacian_var(blurred)


def test_blur_penalties_flag_the_soft_frame_only():
    # three sharp frames and one blurred one; the blurred frame must carry a
    # penalty near 1 while the sharp frames (at/above the median) carry ~0.
    sharp = [laplacian_var(_texture(s)) for s in range(3)]
    soft = laplacian_var(cv2.GaussianBlur(_texture(0), (9, 9), 4))
    lap = np.array(sharp + [soft])
    pen = blur_penalties(lap, present_idx=[0, 1, 2, 3])
    assert pen[3] > 0.5
    assert all(pen[k] < 0.1 for k in (0, 1, 2))


def test_blur_penalties_zero_when_no_present_frames_or_flat():
    # a degenerate clip (all-zero laplacian variance) cannot discriminate -> no penalty
    pen = blur_penalties(np.zeros(4), present_idx=[0, 1, 2, 3])
    assert np.all(pen == 0.0)


def test_edge_penalty_high_for_small_object_on_edge():
    # a small box flush against the left edge of a 100x100 frame
    pen = edge_penalty((0.0, 40.0, 8.0, 8.0), 100, 100, area=8 * 8 / 1e4)
    assert pen > 0.5


def test_edge_penalty_zero_for_big_object_touching_edge():
    # a big object legitimately touches the edge (a bus at frame start) -> size
    # cancels the border penalty, so it is NOT penalised
    pen = edge_penalty((0.0, 0.0, 90.0, 90.0), 100, 100, area=90 * 90 / 1e4)
    assert pen < 0.05


def test_edge_penalty_zero_for_small_object_centered():
    pen = edge_penalty((45.0, 45.0, 8.0, 8.0), 100, 100, area=8 * 8 / 1e4)
    assert pen == 0.0


# --- P4: detail / contrast (separates "small & clear" from "small & dark") ---

def test_rms_contrast_zero_for_flat_crop():
    flat = np.full((32, 32, 3), 128, dtype=np.uint8)
    assert rms_contrast(flat) < 1e-6


def test_rms_contrast_high_for_black_white_split():
    crop = np.zeros((32, 32, 3), dtype=np.uint8)
    crop[:, 16:] = 255                     # half black, half white -> std = 127.5/255
    assert rms_contrast(crop) > 0.45


def test_shannon_entropy_zero_for_flat_high_for_textured():
    flat = np.full((64, 64, 3), 80, dtype=np.uint8)
    assert shannon_entropy(flat) < 1e-6
    assert shannon_entropy(_texture(1)) > 0.7      # near-uniform histogram -> high entropy


def test_detail_signal_textured_beats_dark_flat():
    # the gecko / readable plate (textured) must out-score the "small & dark" mouse
    detailed = _texture(2)
    dark_flat = np.full((64, 64, 3), 12, dtype=np.uint8)   # dark, no structure
    assert detail_signal(detailed) > 0.3
    assert detail_signal(dark_flat) < 0.05
    assert detail_signal(detailed) > 6 * detail_signal(dark_flat)   # clear separation


# --- P3: optical-flow motion (catches the blur the Laplacian missed) ---

def test_flow_magnitude_near_zero_for_static_object():
    f = _smooth(0)
    assert flow_magnitude(f, f) < 0.5


def test_flow_magnitude_rises_with_motion():
    f = _smooth(0)
    moved = _shift(f, 6, 0)
    assert flow_magnitude(f, moved) > flow_magnitude(f, f)
    assert flow_magnitude(f, moved) > 1.0


def test_motion_signal_is_scale_free():
    # the SAME pixel motion is a bigger deal for a small object than a large one
    f = _smooth(0)
    moved = _shift(f, 6, 0)
    small = motion_signal(f, moved, box_diag=12.0)
    large = motion_signal(f, moved, box_diag=240.0)
    assert small > large
    assert motion_signal(f, f, box_diag=12.0) < 0.1       # static object -> ~0
    assert motion_signal(f, moved, box_diag=0.0) == 0.0   # degenerate box -> 0, no divide


# --- P3 (take 2): no-reference blur (Crete 2007 re-blur), what flow/Laplacian missed ---

def test_reblur_blur_higher_for_blurred():
    sharp = _texture(0)
    blurred = cv2.GaussianBlur(sharp, (9, 9), 4)
    assert reblur_blur(blurred) > reblur_blur(sharp)


def test_reblur_blur_in_unit_range_and_safe_on_flat():
    for img in (_texture(0), cv2.GaussianBlur(_texture(0), (9, 9), 4)):
        assert 0.0 <= reblur_blur(img) <= 1.0
    assert reblur_blur(np.full((32, 32, 3), 100, dtype=np.uint8)) == 0.0   # flat -> no divide
