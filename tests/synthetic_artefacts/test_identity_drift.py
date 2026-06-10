"""Unit tests for identity_drift artefact generator."""
import numpy as np
import pytest

from scripts.synthetic_artefacts.identity_drift import apply_identity_drift


def _solid(h: int, w: int, color: tuple) -> np.ndarray:
    return np.full((h, w, 3), color, dtype=np.uint8)


def _reference_face(h: int = 80, w: int = 80) -> np.ndarray:
    """A distinctive synthetic 'face' (gradient pattern) for tests where the
    actual face content does not need to be a real face, only distinguishable
    from the input frame."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = np.linspace(0, 255, w, dtype=np.uint8)[None, :].repeat(h, axis=0)
    img[:, :, 1] = np.linspace(0, 255, h, dtype=np.uint8)[:, None].repeat(w, axis=1)
    img[:, :, 2] = 128
    return img


def test_zero_severity_returns_unchanged_frame():
    frame = _solid(200, 200, (100, 150, 200))
    ref = _reference_face()
    out = apply_identity_drift(frame, idx=50, n_frames=100,
                                reference_face_bgr=ref, severity=0.0)
    assert np.array_equal(out, frame)


def test_first_frame_returns_unchanged_regardless_of_severity():
    # At idx=0 the blend factor is 0 by construction, so the frame should be
    # returned unchanged whatever severity is selected.
    frame = _solid(200, 200, (100, 150, 200))
    ref = _reference_face()
    for severity in (0.05, 0.2, 1.0):
        out = apply_identity_drift(frame, idx=0, n_frames=100,
                                    reference_face_bgr=ref, severity=severity)
        assert np.array_equal(out, frame), f"expected pass-through at idx=0, severity={severity}"


def test_no_face_detected_returns_unchanged_frame():
    # A solid-colour frame has no detectable face, so the generator should
    # return the frame unchanged at any blend factor.
    frame = _solid(200, 200, (128, 128, 128))
    ref = _reference_face()
    out = apply_identity_drift(frame, idx=99, n_frames=100,
                                reference_face_bgr=ref, severity=1.0)
    assert np.array_equal(out, frame)


def test_short_video_passes_through():
    # n_frames <= 1 is degenerate: returning unchanged is the only sensible
    # behaviour because the blend ramp is undefined.
    frame = _solid(200, 200, (100, 150, 200))
    ref = _reference_face()
    out = apply_identity_drift(frame, idx=0, n_frames=1,
                                reference_face_bgr=ref, severity=1.0)
    assert np.array_equal(out, frame)


def test_uint8_output_shape_preserved():
    frame = _solid(200, 200, (100, 150, 200))
    ref = _reference_face()
    out = apply_identity_drift(frame, idx=10, n_frames=100,
                                reference_face_bgr=ref, severity=0.5)
    assert out.shape == frame.shape
    assert out.dtype == np.uint8
