import numpy as np
import pytest

from scripts.synthetic_artefacts.flip import apply_flip


def _frame(h=64, w=80, c=3, val=128):
    return (np.ones((h, w, c), dtype=np.uint8) * val) + np.arange(w, dtype=np.uint8)[None, :, None]


@pytest.mark.parametrize("transform", ["horizontal", "transpose", "periodic",
                                       "elastic", "channel_shuffle", "invert"])
def test_severity_zero_is_pass_through(transform):
    fr = _frame()
    out = apply_flip(fr, idx=50, n_frames=100, transform=transform, severity=0.0)
    assert np.array_equal(out, fr)


@pytest.mark.parametrize("transform", ["horizontal", "transpose", "periodic",
                                       "elastic", "channel_shuffle", "invert"])
def test_degenerate_n_frames_returns_input(transform):
    fr = _frame()
    out = apply_flip(fr, idx=0, n_frames=1, transform=transform, severity=0.40)
    assert np.array_equal(out, fr)


def test_horizontal_pre_midpoint_unchanged():
    fr = _frame()
    out = apply_flip(fr, idx=10, n_frames=100, transform="horizontal", severity=0.40)
    assert np.array_equal(out, fr)


def test_horizontal_post_midpoint_full_alpha_is_cv2_flip():
    import cv2
    fr = _frame()
    out = apply_flip(fr, idx=80, n_frames=100, transform="horizontal", severity=0.40)
    assert np.array_equal(out, cv2.flip(fr, 1))


def test_invert_post_midpoint_full_alpha_is_255_minus():
    fr = _frame()
    out = apply_flip(fr, idx=80, n_frames=100, transform="invert", severity=0.40)
    assert np.array_equal(out, 255 - fr)


def test_channel_shuffle_post_midpoint_full_alpha_swaps_bgr():
    fr = _frame()
    out = apply_flip(fr, idx=80, n_frames=100, transform="channel_shuffle", severity=0.40)
    assert np.array_equal(out, fr[:, :, ::-1])


def test_transpose_preserves_shape():
    fr = _frame()
    out = apply_flip(fr, idx=80, n_frames=100, transform="transpose", severity=0.40)
    assert out.shape == fr.shape


def test_elastic_changes_pixels_but_preserves_shape_and_dtype():
    fr = _frame()
    out = apply_flip(fr, idx=80, n_frames=100, transform="elastic", severity=0.40)
    assert out.shape == fr.shape
    assert out.dtype == fr.dtype
    assert not np.array_equal(out, fr)


def test_elastic_deterministic_across_calls_same_size():
    # Same (h, w) at same idx -> same output (cached deterministic displacement field).
    fr = _frame()
    a = apply_flip(fr, idx=80, n_frames=100, transform="elastic", severity=0.40)
    b = apply_flip(fr, idx=80, n_frames=100, transform="elastic", severity=0.40)
    assert np.array_equal(a, b)


def test_periodic_block_0_unchanged_block_1_blended():
    fr = _frame()
    # idx 10 -> block 0 (idx // 30 = 0) -> even -> unchanged
    out_even = apply_flip(fr, idx=10, n_frames=200, transform="periodic", severity=0.40)
    assert np.array_equal(out_even, fr)
    # idx 35 -> block 1 -> odd -> alpha-blended horizontal flip with alpha = 1.0
    import cv2
    out_odd = apply_flip(fr, idx=35, n_frames=200, transform="periodic", severity=0.40)
    assert np.array_equal(out_odd, cv2.flip(fr, 1))


def test_partial_severity_alpha_blend():
    fr = _frame()
    # severity 0.20 -> alpha = 0.5; post-midpoint horizontal expects 0.5*flip + 0.5*orig
    import cv2
    out = apply_flip(fr, idx=80, n_frames=100, transform="horizontal", severity=0.20)
    expected = cv2.addWeighted(cv2.flip(fr, 1), 0.5, fr, 0.5, 0)
    assert np.array_equal(out, expected)


def test_unknown_transform_raises():
    fr = _frame()
    with pytest.raises(ValueError):
        apply_flip(fr, idx=80, n_frames=100, transform="not_a_real_transform", severity=0.40)
