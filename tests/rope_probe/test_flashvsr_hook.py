"""Unit tests for the temporal-freq-table injection hook (no GPU, fake table)."""
import pytest
import torch

from scripts.rope_probe.flashvsr_hook import TemporalFreqTable, install_position_hook
from scripts.rope_probe.position_override import PositionOverride


def _fake_table(end, dim=4):
    """Deterministic stand-in for precompute_freqs_cis: row p encodes p."""
    pos = torch.arange(end, dtype=torch.float64).unsqueeze(1)
    k = torch.arange(dim, dtype=torch.float64).unsqueeze(0)
    return torch.polar(torch.ones(end, dim, dtype=torch.float64), pos * (k + 1) * 0.01)


class _StubDit:
    def __init__(self, table):
        self.freqs = (table, "h_table", "w_table")


def test_noop_slice_is_bitwise_identical():
    table = _fake_table(16)
    wrapped = TemporalFreqTable(table, PositionOverride())
    assert torch.equal(wrapped[2:6], table[2:6])
    assert torch.equal(wrapped[:6], table[:6])


def test_shift_translates_slice():
    table = _fake_table(16)
    wrapped = TemporalFreqTable(table, PositionOverride(shift=3))
    assert torch.equal(wrapped[2:6], table[5:9])


def test_stretch_scales_positions():
    table = _fake_table(16)
    wrapped = TemporalFreqTable(table, PositionOverride(stretch=2.0))
    assert torch.equal(wrapped[2:4], table[torch.tensor([4, 6])])


def test_out_of_table_uses_extended_builder():
    table = _fake_table(16)
    wrapped = TemporalFreqTable(table, PositionOverride(shift=20),
                                table_builder=_fake_table)
    got = wrapped[2:4]  # positions 22, 23 — beyond the 16-row table
    big = _fake_table(32)
    assert torch.equal(got, big[torch.tensor([22, 23])])


def test_out_of_table_without_builder_raises():
    table = _fake_table(16)
    wrapped = TemporalFreqTable(table, PositionOverride(shift=20))
    with pytest.raises(RuntimeError):
        wrapped[2:4]


def test_negative_position_raises():
    table = _fake_table(16)
    wrapped = TemporalFreqTable(table, PositionOverride(shift=-5))
    with pytest.raises(ValueError):
        wrapped[2:4]


def test_install_and_restore_roundtrip():
    table = _fake_table(16)
    dit = _StubDit(table)
    restore = install_position_hook(dit, PositionOverride(shift=1))
    assert isinstance(dit.freqs[0], TemporalFreqTable)
    assert dit.freqs[1] == "h_table" and dit.freqs[2] == "w_table"
    assert torch.equal(dit.freqs[0][0:2], table[1:3])
    restore()
    assert dit.freqs[0] is table


def _fake_base_freqs(dim=4):
    return (torch.arange(dim, dtype=torch.float64) + 1) * 0.01


def _fake_row_builder(positions):
    pos = torch.as_tensor(positions, dtype=torch.float64).unsqueeze(1)
    k = _fake_base_freqs().unsqueeze(0)
    return torch.polar(torch.ones(len(positions), 4, dtype=torch.float64), pos * k)


def test_continuous_rows_match_table_at_integer_positions():
    table = _fake_table(16)
    wrapped = TemporalFreqTable(table, PositionOverride(continuous=True),
                                row_builder=_fake_row_builder)
    got = wrapped[2:6]  # continuous identity -> positions 2.0..5.0
    assert torch.allclose(got, table[2:6])


def test_continuous_compression_uses_fractional_positions():
    table = _fake_table(16)
    wrapped = TemporalFreqTable(table, PositionOverride(stretch=0.5, continuous=True),
                                row_builder=_fake_row_builder)
    got = wrapped[4:6]  # positions 2.0, 2.5
    expect = _fake_row_builder([2.0, 2.5])
    assert torch.equal(got, expect)
    # and crucially: NOT equal to integer-rounded rows (2, 2)
    assert not torch.allclose(got[1], table[2])


def test_continuous_without_builder_raises():
    table = _fake_table(16)
    wrapped = TemporalFreqTable(table, PositionOverride(stretch=0.5, continuous=True))
    with pytest.raises(RuntimeError):
        wrapped[4:6]


def test_continuous_subsumes_extension_beyond_table():
    table = _fake_table(16)
    wrapped = TemporalFreqTable(table, PositionOverride(shift=20, continuous=True),
                                row_builder=_fake_row_builder)
    got = wrapped[2:4]  # positions 22.0, 23.0 — beyond 16 rows, no ext table needed
    assert torch.equal(got, _fake_row_builder([22.0, 23.0]))


# --- runner padding math (UDM10 318x180 support) ---

def test_pad_amounts_synthetic_320x180():
    from scripts.rope_probe.flashvsr_runner import pad_amounts
    (top, bot, left, right), (th, tw) = pad_amounts(180, 320)
    assert (top, bot, left, right) == (6, 6, 0, 0)
    assert (th, tw) == (768, 1280)


def test_pad_amounts_udm10_318x180():
    from scripts.rope_probe.flashvsr_runner import pad_amounts
    (top, bot, left, right), (th, tw) = pad_amounts(180, 318)
    assert (top, bot) == (6, 6)
    assert (left, right) == (1, 1)
    assert (th, tw) == (768, 1280)


def test_center_crop_to_ref_shape():
    import numpy as np
    from scripts.rope_probe.score_conditions import center_crop
    pred = np.zeros((768, 1280, 3), dtype=np.uint8)
    out = center_crop(pred, (720, 1272))
    assert out.shape == (720, 1272, 3)
    same = center_crop(pred, (768, 1280))
    assert same.shape == (768, 1280, 3)


def test_prepare_reads_frames_dir_and_pads(tmp_path):
    # guards against the silent-refactor breakage that killed the first
    # UDM10 sweep: prepare must return (tensor, F, (TH, TW)) and handle
    # UDM10-sized (318x180) PNG dirs
    import cv2
    import numpy as np
    from scripts.rope_probe.flashvsr_runner import prepare
    d = tmp_path / "lq"
    d.mkdir()
    for i in range(13):  # 13 % 8 == 5 -> F = 17, no trim
        img = np.random.randint(0, 256, (180, 318, 3), dtype=np.uint8)
        cv2.imwrite(str(d / f"{i:04d}.png"), img)
    vid, F, (th, tw) = prepare(str(d), 13, device="cpu")
    assert F == 17
    assert (th, tw) == (768, 1280)
    assert tuple(vid.shape) == (1, 3, 17, 768, 1280)


def test_slice_beyond_table_not_clamped():
    # single-pass long videos slice past the table end; the hook must compute
    # baseline positions from the RAW slice bounds, not clamp to table length
    table = _fake_table(16)
    wrapped = TemporalFreqTable(table, PositionOverride(modulo=10**9),
                                table_builder=_fake_table)
    got = wrapped[18:20]   # beyond the 16-row table; modulo huge = identity
    big = _fake_table(32)
    assert got.shape[0] == 2
    assert torch.equal(got, big[torch.tensor([18, 19])])


def test_modulo_cycles_via_table():
    table = _fake_table(16)
    wrapped = TemporalFreqTable(table, PositionOverride(modulo=8))
    assert torch.equal(wrapped[18:20], table[torch.tensor([2, 3])])


def test_axis_selects_spatial_table():
    t0, t1, t2 = _fake_table(16), _fake_table(16) * 2, _fake_table(16) * 3
    dit = _StubDit(t0)
    dit.freqs = (t0, t1, t2)
    restore = install_position_hook(dit, PositionOverride(shift=1), axis=1)
    assert dit.freqs[0] is t0 and dit.freqs[2] is t2       # untouched axes
    assert torch.equal(dit.freqs[1][0:2], t1[1:3])          # H axis shifted
    restore()
    assert dit.freqs[1] is t1


def test_axes_compose_and_restore_in_reverse():
    t0, t1, t2 = _fake_table(16), _fake_table(16) * 2, _fake_table(16) * 3
    dit = _StubDit(t0)
    dit.freqs = (t0, t1, t2)
    r_t = install_position_hook(dit, PositionOverride(shift=1), axis=0)
    r_h = install_position_hook(dit, PositionOverride(shift=2), axis=1)
    assert torch.equal(dit.freqs[0][0:2], t0[1:3])
    assert torch.equal(dit.freqs[1][0:2], t1[2:4])
    assert dit.freqs[2] is t2
    r_h(); r_t()
    assert dit.freqs == (t0, t1, t2)
