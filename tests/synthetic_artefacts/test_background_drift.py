"""Unit tests for background_drift artefact generator."""
import numpy as np

from scripts.synthetic_artefacts.background_drift import apply_background_drift


def _solid(h: int, w: int, color: tuple) -> np.ndarray:
    return np.full((h, w, 3), color, dtype=np.uint8)


def test_zero_severity_returns_unchanged_frame():
    frame = _solid(200, 200, (100, 150, 200))
    ref = _solid(200, 200, (50, 50, 50))
    mask = np.zeros((200, 200), dtype=bool)
    out = apply_background_drift(frame, idx=50, n_frames=100,
                                  reference_bg_bgr=ref,
                                  human_mask=mask, severity=0.0)
    assert np.array_equal(out, frame)


def test_first_frame_returns_unchanged_regardless_of_severity():
    frame = _solid(200, 200, (100, 150, 200))
    ref = _solid(200, 200, (50, 50, 50))
    mask = np.zeros((200, 200), dtype=bool)
    for severity in (0.05, 0.2, 1.0):
        out = apply_background_drift(frame, idx=0, n_frames=100,
                                      reference_bg_bgr=ref,
                                      human_mask=mask, severity=severity)
        assert np.array_equal(out, frame), f"expected pass-through at idx=0, severity={severity}"


def test_none_mask_blends_entire_frame_at_max_blend():
    # Solid red input, all-blue reference, severity=1.0 at the final frame.
    frame = _solid(200, 200, (0, 0, 255))
    ref = _solid(200, 200, (255, 0, 0))
    out = apply_background_drift(frame, idx=99, n_frames=100,
                                  reference_bg_bgr=ref,
                                  human_mask=None, severity=1.0)
    assert np.allclose(out, ref, atol=1)


def test_full_mask_preserves_entire_frame():
    # When the mask flags every pixel as human, no blending should occur
    # regardless of severity.
    frame = _solid(200, 200, (0, 0, 255))
    ref = _solid(200, 200, (255, 0, 0))
    mask = np.ones((200, 200), dtype=bool)
    out = apply_background_drift(frame, idx=99, n_frames=100,
                                  reference_bg_bgr=ref,
                                  human_mask=mask, severity=1.0)
    assert np.array_equal(out, frame)


def test_partial_mask_preserves_only_inside():
    # Half-image mask: top half preserved, bottom half blended.
    frame = _solid(200, 200, (0, 0, 255))    # solid red
    ref = _solid(200, 200, (255, 0, 0))      # solid blue
    mask = np.zeros((200, 200), dtype=bool)
    mask[:100, :] = True
    out = apply_background_drift(frame, idx=99, n_frames=100,
                                  reference_bg_bgr=ref,
                                  human_mask=mask, severity=1.0)
    # Top half: original red preserved
    assert np.array_equal(out[:100], frame[:100])
    # Bottom half: blended toward blue
    assert np.allclose(out[100:], ref[100:], atol=1)


def test_intermediate_blend_factor():
    # severity=0.5, idx=50, n_frames=100 -> blend = 0.5 * 50/99 ~= 0.2525
    frame = _solid(200, 200, (0, 0, 255))
    ref = _solid(200, 200, (255, 0, 0))
    mask = np.zeros((200, 200), dtype=bool)
    out = apply_background_drift(frame, idx=50, n_frames=100,
                                  reference_bg_bgr=ref,
                                  human_mask=mask, severity=0.5)
    expected_b = round(0.2525 * 255)
    expected_r = round((1 - 0.2525) * 255)
    assert abs(int(out[100, 100, 0]) - expected_b) < 5
    assert int(out[100, 100, 1]) == 0
    assert abs(int(out[100, 100, 2]) - expected_r) < 5


def test_short_video_passes_through():
    frame = _solid(200, 200, (100, 150, 200))
    ref = _solid(200, 200, (50, 50, 50))
    mask = np.zeros((200, 200), dtype=bool)
    out = apply_background_drift(frame, idx=0, n_frames=1,
                                  reference_bg_bgr=ref,
                                  human_mask=mask, severity=1.0)
    assert np.array_equal(out, frame)


def test_uint8_output_shape_preserved():
    frame = _solid(200, 200, (100, 150, 200))
    ref = _solid(200, 200, (50, 50, 50))
    mask = np.zeros((200, 200), dtype=bool)
    out = apply_background_drift(frame, idx=10, n_frames=100,
                                  reference_bg_bgr=ref,
                                  human_mask=mask, severity=0.5)
    assert out.shape == frame.shape
    assert out.dtype == np.uint8
