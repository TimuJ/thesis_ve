import math
import numpy as np
from scripts.synthetic_artefacts.flicker import apply_periodic_flicker


def test_zero_severity_no_change():
    frame = np.full((10, 10, 3), 128, dtype=np.uint8)
    out = apply_periodic_flicker(frame, frame_idx=5, period_frames=15, severity=0.0)
    np.testing.assert_array_equal(out, frame)


def test_frame_0_no_modulation():
    # At frame 0, sin(0) = 0 -> mod = 1.0, no change.
    frame = np.full((10, 10, 3), 128, dtype=np.uint8)
    out = apply_periodic_flicker(frame, frame_idx=0, period_frames=15, severity=0.40)
    np.testing.assert_array_equal(out, frame)


def test_period_repeats():
    # frame N (period) and frame 0 should both give sin(2π) = 0, so same output.
    frame = np.full((10, 10, 3), 100, dtype=np.uint8)
    o0 = apply_periodic_flicker(frame, frame_idx=0, period_frames=15, severity=0.40)
    oN = apply_periodic_flicker(frame, frame_idx=15, period_frames=15, severity=0.40)
    np.testing.assert_array_equal(o0, oN)


def test_quarter_period_peak():
    # At frame = period/4, sin(π/2) = 1, so brightness multiplied by 1+severity.
    # period=15, frame_idx=3: 2π*3/15 = 1.2566 rad, sin(1.2566) ≈ 0.951
    # mod ≈ 1 + 0.40 * 0.951 = 1.3804
    frame = np.full((10, 10, 3), 100, dtype=np.uint8)
    out = apply_periodic_flicker(frame, frame_idx=3, period_frames=15, severity=0.40)
    angle = 2.0 * math.pi * 3 / 15
    mod = 1.0 + 0.40 * math.sin(angle)
    expected = int(100 * mod)
    # Allow ±3 due to float->int rounding
    assert abs(int(out[0, 0, 0]) - expected) <= 3


def test_clipped_to_uint8():
    frame = np.full((10, 10, 3), 250, dtype=np.uint8)
    # peak boost will cause overflow -> must clip to uint8
    out = apply_periodic_flicker(frame, frame_idx=3, period_frames=15, severity=0.40)
    assert out.dtype == np.uint8
    assert out.max() <= 255
    assert out.min() >= 0
