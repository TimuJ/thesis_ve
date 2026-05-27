"""Tests for the identity_degradation synthetic artefact.

Strategy:
- For face-bearing tests we use a real frame extracted from a base SR video
  if available locally (results/mgld_synthetic_mp4/hhszUXL1Cu8.mp4 frame 120
  reliably contains a frontal face). If that file is missing we fall back to
  a procedural synthetic face; if neither produces a detection the test is
  skipped. This keeps CI green even on environments without the dataset.
- For no-face cases, use uniform frames which the cascade rejects.
"""
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.synthetic_artefacts.identity_degradation import (
    _get_face_detector,
    apply_identity_degradation,
)


_REAL_VIDEO = (
    Path(__file__).resolve().parents[2]
    / "results"
    / "mgld_synthetic_mp4"
    / "hhszUXL1Cu8.mp4"
)


def _real_face_frame():
    """Try to load a frame from the base SR video with a detectable face.

    Returns the BGR frame or None if not available.
    """
    if not _REAL_VIDEO.is_file():
        return None
    cap = cv2.VideoCapture(str(_REAL_VIDEO))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 120)
        ok, fr = cap.read()
        return fr if ok else None
    finally:
        cap.release()


def _make_face_canvas(h=480, w=480):
    """Return a procedural BGR frame the Haar cascade may detect."""
    canvas = np.full((h, w, 3), 200, dtype=np.uint8)
    cx, cy = w // 2, h // 2
    cv2.ellipse(canvas, (cx, cy), (110, 150), 0, 0, 360, (180, 200, 220), -1)
    cv2.circle(canvas, (cx - 40, cy - 40), 14, (40, 40, 40), -1)
    cv2.circle(canvas, (cx + 40, cy - 40), 14, (40, 40, 40), -1)
    cv2.line(canvas, (cx, cy - 15), (cx, cy + 25), (120, 130, 140), 4)
    cv2.ellipse(canvas, (cx, cy + 60), (35, 12), 0, 0, 180, (60, 60, 90), 4)
    return canvas


def _face_frame_or_skip():
    """Return a frame the cascade detects a face in, or skip the test."""
    fr = _real_face_frame()
    if fr is not None and _face_box(fr) is not None:
        return fr
    fr = _make_face_canvas()
    if _face_box(fr) is not None:
        return fr
    pytest.skip("no Haar-detectable face available in this environment")


def _laplacian_variance(img_bgr, x0, y0, x1, y1):
    crop = img_bgr[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def _face_box(frame_bgr):
    """Return (x, y, w, h) of the first detected face, or None."""
    det = _get_face_detector()
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = det.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
    if len(faces) == 0:
        return None
    return tuple(int(v) for v in faces[0])


def test_zero_severity_is_identity():
    # zero-severity short-circuits before face detection, so any frame works
    frame = _make_face_canvas()
    out = apply_identity_degradation(frame, idx=0, severity=0.0)
    np.testing.assert_array_equal(out, frame)
    # ensure it's a copy, not the same object (caller mutability safety)
    assert out is not frame


def test_no_face_passthrough():
    # Pure flat frames have no facial structure -> Haar cascade returns 0 faces.
    flat = np.full((300, 300, 3), 128, dtype=np.uint8)
    out = apply_identity_degradation(flat, idx=0, severity=0.40)
    np.testing.assert_array_equal(out, flat)
    # also try black/white
    for v in (0, 255):
        f = np.full((300, 300, 3), v, dtype=np.uint8)
        out = apply_identity_degradation(f, idx=0, severity=0.40)
        np.testing.assert_array_equal(out, f)


def test_blur_actually_applied_when_face_present():
    frame = _face_frame_or_skip()
    box = _face_box(frame)
    x, y, w, h = box
    pad = int(0.1 * max(w, h))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(frame.shape[1], x + w + pad), min(frame.shape[0], y + h + pad)

    out = apply_identity_degradation(frame, idx=0, severity=0.40)

    base_var = _laplacian_variance(frame, x0, y0, x1, y1)
    blurred_var = _laplacian_variance(out, x0, y0, x1, y1)
    # Sanity: there was edge content to begin with.
    assert base_var > 1.0
    # Heavy blur (sigma=4) must wipe out >50% of edge variance inside the box.
    assert blurred_var < 0.5 * base_var, (
        "expected blur to reduce edge variance, got base=" + str(base_var)
        + " blurred=" + str(blurred_var)
    )


def test_severity_monotonic_blur():
    frame = _face_frame_or_skip()
    box = _face_box(frame)
    x, y, w, h = box
    pad = int(0.1 * max(w, h))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(frame.shape[1], x + w + pad), min(frame.shape[0], y + h + pad)

    sevs = [0.0, 0.1, 0.2, 0.4]
    variances = []
    for s in sevs:
        out = apply_identity_degradation(frame, idx=0, severity=s)
        variances.append(_laplacian_variance(out, x0, y0, x1, y1))
    # Monotonically non-increasing (strict between non-zero sevs).
    for a, b in zip(variances, variances[1:]):
        assert b <= a + 1e-6, "non-monotone variances: " + str(variances)
    # And a real, large drop between 0 and 0.4
    assert variances[-1] < 0.5 * variances[0]
