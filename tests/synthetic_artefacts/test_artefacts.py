import numpy as np
from scripts.synthetic_artefacts.color_drift import apply_color_drift
from scripts.synthetic_artefacts.chunk_boundary import apply_chunk_boundary_jumps


def test_color_drift_zero_severity_no_change():
    frame = np.full((100, 100, 3), 128, dtype=np.uint8)
    out = apply_color_drift(frame, frame_idx=50, total_frames=100, severity=0.0)
    np.testing.assert_array_equal(out, frame)


def test_color_drift_increases_red_at_end():
    frame = np.full((10, 10, 3), 100, dtype=np.uint8)  # mid-gray
    out0 = apply_color_drift(frame, frame_idx=0, total_frames=100, severity=0.40)
    out_last = apply_color_drift(frame, frame_idx=99, total_frames=100, severity=0.40)
    # at frame 0, drift = 0 -> no change
    np.testing.assert_array_equal(out0, frame)
    # at last frame, drift ~= 0.40, R should be ~140
    assert out_last[0, 0, 2] > frame[0, 0, 2]  # OpenCV BGR: index 2 is R
    assert out_last[0, 0, 0] < frame[0, 0, 0]  # B reduced
    assert out_last[0, 0, 1] < frame[0, 0, 1]  # G reduced


def test_color_drift_clipped_to_uint8():
    frame = np.full((10, 10, 3), 250, dtype=np.uint8)
    out = apply_color_drift(frame, frame_idx=99, total_frames=100, severity=0.40)
    assert out.dtype == np.uint8
    assert out.max() <= 255
    assert out.min() >= 0


def test_chunk_boundary_zero_severity_no_change():
    frame = np.full((100, 100, 3), 128, dtype=np.uint8)
    out = apply_chunk_boundary_jumps(frame, frame_idx=70, chunk_size_frames=60, severity=0.0)
    np.testing.assert_array_equal(out, frame)


def test_chunk_boundary_same_chunk_same_offset():
    frame = np.full((10, 10, 3), 128, dtype=np.uint8)
    # frames in the same chunk should get the same offset (so identical pixel changes)
    o1 = apply_chunk_boundary_jumps(frame, frame_idx=60, chunk_size_frames=60, severity=0.10)
    o2 = apply_chunk_boundary_jumps(frame, frame_idx=119, chunk_size_frames=60, severity=0.10)
    np.testing.assert_array_equal(o1, o2)  # same chunk -> same output


def test_chunk_boundary_different_chunks_different_offsets():
    frame = np.full((10, 10, 3), 128, dtype=np.uint8)
    # frames in different chunks should generally produce different outputs
    # (could rarely coincide if RNG hits same uniform value, but at severity 0.4 it's vanishingly rare)
    outputs = [apply_chunk_boundary_jumps(frame, frame_idx=c * 60, chunk_size_frames=60, severity=0.40)
               for c in range(10)]
    # not all the same (at least some pair differs)
    all_same = all(np.array_equal(outputs[0], o) for o in outputs[1:])
    assert not all_same


def test_chunk_boundary_clipped():
    frame = np.full((10, 10, 3), 250, dtype=np.uint8)
    out = apply_chunk_boundary_jumps(frame, frame_idx=0, chunk_size_frames=60, severity=0.40)
    assert out.dtype == np.uint8
    assert out.max() <= 255
    assert out.min() >= 0
