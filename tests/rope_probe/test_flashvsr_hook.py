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
