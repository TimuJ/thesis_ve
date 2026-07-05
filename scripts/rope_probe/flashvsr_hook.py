"""Temporal-RoPE position injection for FlashVSR (Wan2.1 DiT).

Every FlashVSR pipeline reads temporal RoPE positions exclusively by slicing
the precomputed table `dit.freqs[0]` (see
docs/notes/2026-07-02-flashvsr-rope-site.md). This hook swaps that table for a
duck-typed wrapper whose `__getitem__` translates the requested baseline slice
through a `PositionOverride` — one injection point, pipeline-agnostic
(tiny / tiny_long / full and `WanModel.forward` all go through it), and zero
modification of the FlashVSR repo (runtime attribute swap only; the repo stays
at its `pristine-2026-07-02` tag).

No-op faithfulness: with a no-op override the wrapper returns `table[key]`
directly — the identical tensor view the stock code would read — so the
forward pass is bit-exact by construction.

Positions beyond the stock 1024-row table (extrapolation experiments) are
served from an extended table built on demand by `table_builder(end)`; the
builder must be the model's own `precompute_freqs_cis` partial so overlapping
rows are bitwise identical (asserted at build time). Use
`default_table_builder(...)` on the server.
"""
from __future__ import annotations

from typing import Callable, Optional

import torch

from scripts.rope_probe.position_override import PositionOverride, is_noop, transform_indices


class TemporalFreqTable:
    """Duck-typed stand-in for `dit.freqs[0]`: only `__getitem__` is served.

    Integer overrides index the (possibly extended) precomputed table;
    continuous overrides bypass tables entirely and compute rows at the exact
    fractional positions via `row_builder` (true position interpolation).
    """

    def __init__(self, table: torch.Tensor, override: PositionOverride,
                 table_builder: Optional[Callable[[int], torch.Tensor]] = None,
                 row_builder: Optional[Callable[[list], torch.Tensor]] = None):
        self._table = table
        self._ov = override
        self._builder = table_builder
        self._row_builder = row_builder
        self._ext: Optional[torch.Tensor] = None

    def _extended(self, needed: int) -> torch.Tensor:
        if self._ext is None or self._ext.shape[0] <= needed:
            if self._builder is None:
                raise RuntimeError(
                    f"position {needed} beyond table end {self._table.shape[0]} "
                    "and no table_builder given")
            end = self._table.shape[0]
            while end <= needed:
                end *= 2
            ext = self._builder(end)
            if not torch.equal(ext[: self._table.shape[0]], self._table):
                raise RuntimeError(
                    "extended table prefix mismatch — builder is not the "
                    "model's own precompute_freqs_cis")
            self._ext = ext
        return self._ext

    def __getitem__(self, key):
        if not isinstance(key, slice):
            raise TypeError(f"only slice access expected from pipelines, got {key!r}")
        if is_noop(self._ov):
            return self._table[key]
        base = list(range(*key.indices(self._table.shape[0])))
        idx = transform_indices(base, self._ov)
        if any(i < 0 for i in idx):
            raise ValueError(f"override maps to negative position(s): {idx[:4]}...")
        if self._ov.continuous:
            if self._row_builder is None:
                raise RuntimeError(
                    "continuous override needs a row_builder "
                    "(use default_row_builder(t_dim) on the server)")
            return self._row_builder(idx)
        table = self._table
        top = max(idx)
        if top >= table.shape[0]:
            table = self._extended(top)
        return table[torch.tensor(idx, dtype=torch.long)]


def install_position_hook(dit, override: PositionOverride,
                          table_builder: Optional[Callable[[int], torch.Tensor]] = None,
                          row_builder: Optional[Callable[[list], torch.Tensor]] = None):
    """Swap dit.freqs[0] for the wrapped table; returns a restore() callable."""
    orig = dit.freqs
    dit.freqs = (TemporalFreqTable(orig[0], override, table_builder, row_builder),
                 orig[1], orig[2])

    def restore():
        dit.freqs = orig

    return restore


def default_table_builder(t_dim: int, theta: float = 10000.0):
    """Builder using the model's own precompute_freqs_cis (server-only import).

    t_dim is the temporal axis dim: `head_dim - 2 * (head_dim // 3)`, or
    equivalently `dit.freqs[0].shape[1] * 2`.
    """
    from diffsynth.models.wan_video_dit import precompute_freqs_cis

    return lambda end: precompute_freqs_cis(t_dim, end, theta)


def default_row_builder(t_dim: int, theta: float = 10000.0):
    """Continuous-position row builder: `polar(1, p·freqs)` at arbitrary real
    positions — same formula as precompute_freqs_cis, evaluated at fractional
    p instead of arange rows. Mirrors its dtype path (float32) so integer
    positions reproduce the table's values."""
    base = 1.0 / (theta ** (torch.arange(0, t_dim, 2)[: t_dim // 2].float() / t_dim))

    def build(positions):
        pos = torch.tensor(positions, dtype=torch.float32)
        ang = torch.outer(pos, base)
        return torch.polar(torch.ones_like(ang), ang)

    return build
