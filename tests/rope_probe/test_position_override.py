from scripts.rope_probe.position_override import (
    PositionOverride, temporal_indices, is_noop,
)


def test_default_is_identity():
    ov = PositionOverride()
    assert temporal_indices(5, ov) == [0, 1, 2, 3, 4]
    assert is_noop(ov) is True


def test_shift_adds_constant():
    ov = PositionOverride(shift=100)
    assert temporal_indices(4, ov) == [100, 101, 102, 103]
    assert is_noop(ov) is False


def test_stretch_scales_positions():
    ov = PositionOverride(stretch=3.0)
    assert temporal_indices(4, ov) == [0, 3, 6, 9]


def test_stretch_then_shift_compose():
    ov = PositionOverride(shift=10, stretch=2.0)
    assert temporal_indices(3, ov) == [10, 12, 14]


def test_stretch_rounds_to_nearest_int():
    # stretch=1.5, base_len=4: i*1.5 = 0, 1.5, 3.0, 4.5
    # Python banker's rounding: round(1.5)=2, round(3.0)=3, round(4.5)=4
    ov = PositionOverride(stretch=1.5)
    assert temporal_indices(4, ov) == [0, 2, 3, 4]


def test_explicit_indices_win():
    ov = PositionOverride(indices=[0, 7, 42])
    assert temporal_indices(3, ov) == [0, 7, 42]
    assert is_noop(ov) is False


def test_explicit_indices_length_must_match():
    ov = PositionOverride(indices=[0, 1])
    import pytest
    with pytest.raises(ValueError):
        temporal_indices(3, ov)
