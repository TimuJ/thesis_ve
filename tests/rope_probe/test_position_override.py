from scripts.rope_probe.position_override import (
    transform_indices,
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


def test_transform_indices_noop_is_identity():
    ov = PositionOverride()
    assert transform_indices([4, 5], ov) == [4, 5]
    assert transform_indices([0, 1, 2, 3, 4, 5], ov) == [0, 1, 2, 3, 4, 5]


def test_transform_indices_shift_on_chunk_base():
    ov = PositionOverride(shift=100)
    assert transform_indices([4, 5], ov) == [104, 105]


def test_transform_indices_stretch_scales_chunk_base():
    ov = PositionOverride(stretch=2.0)
    assert transform_indices([4, 5], ov) == [8, 10]


def test_transform_indices_explicit_indices_win():
    ov = PositionOverride(indices=[7, 42])
    assert transform_indices([4, 5], ov) == [7, 42]


def test_transform_indices_explicit_length_mismatch_raises():
    import pytest
    ov = PositionOverride(indices=[7])
    with pytest.raises(ValueError):
        transform_indices([4, 5], ov)


def test_temporal_indices_consistent_with_transform():
    ov = PositionOverride(shift=3, stretch=1.5)
    assert temporal_indices(4, ov) == transform_indices([0, 1, 2, 3], ov)


def test_continuous_transform_preserves_fractions():
    ov = PositionOverride(stretch=0.5, continuous=True)
    assert transform_indices([4, 5], ov) == [2.0, 2.5]


def test_continuous_identity_returns_float_positions():
    ov = PositionOverride(continuous=True)
    assert transform_indices([4, 5], ov) == [4.0, 5.0]


def test_continuous_false_still_rounds():
    ov = PositionOverride(stretch=0.5)
    assert transform_indices([4, 5], ov) == [2, 2]


def test_continuous_override_is_not_noop():
    # continuous identity recomputes rows via a float path — not bit-exact,
    # so it must NOT be treated as a no-op
    assert is_noop(PositionOverride(continuous=True)) is False


def test_modulo_cycles_positions():
    ov = PositionOverride(modulo=16)
    assert transform_indices([4, 5], ov) == [4, 5]
    assert transform_indices([18, 19], ov) == [2, 3]
    assert transform_indices([336, 337], ov) == [0, 1]


def test_modulo_applies_after_shift_stretch():
    ov = PositionOverride(shift=2, modulo=10)
    assert transform_indices([9, 10], ov) == [1, 2]


def test_modulo_not_noop():
    assert is_noop(PositionOverride(modulo=16)) is False
