"""Temporal position-index overrides for the RoPE extrapolation probe.

Pure logic: no torch, no GPU. Given a baseline temporal length, produce the
list of temporal position indices to feed the model's RoPE. Holding pixel
content fixed while varying these indices is the whole experiment.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class PositionOverride:
    shift: int = 0                        # constant added to every temporal index
    stretch: float = 1.0                  # multiply index i by this before shifting
    indices: Optional[list] = None        # explicit index list; overrides shift/stretch
    length: Optional[int] = None          # for the chunked/extended path (informational)


def temporal_indices(base_len: int, ov: PositionOverride) -> list:
    if ov.indices is not None:
        if len(ov.indices) != base_len:
            raise ValueError(
                f"explicit indices len {len(ov.indices)} != base_len {base_len}")
        return list(ov.indices)
    return [int(round(i * ov.stretch)) + ov.shift for i in range(base_len)]


def is_noop(ov: PositionOverride) -> bool:
    return (ov.shift == 0 and ov.stretch == 1.0
            and ov.indices is None and ov.length is None)
