"""Weak-anchor fallback and anchor reliability.

When the best straddling pair for the longest gap is poorly matched (GT cross-gap
similarity below a floor) or no admissible pair exists, fall back to the next
longest qualifying gap; if none clears the floor, keep the clip at reduced
consistency reliability. Never silently drop the clip or widen the span.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from scripts.lr_vcc.reliability import below_threshold_penalty

from .config import C_GT_FLOOR


@dataclass
class FallbackResult:
    gap: object
    pair: object
    c_gt: float
    reliability: float
    status: str  # "ok" | "weak" | "no_pair"


def anchor_reliability(c_gt: float, floor: float = C_GT_FLOOR, sharpness: float = 10.0) -> float:
    """1 when the GT anchor is well above the floor, →0 as it falls below."""
    return 1.0 - below_threshold_penalty(c_gt, floor, sharpness)


def select_with_fallback(gaps_desc, choose_fn: Callable, c_gt_fn: Callable,
                         floor: float = C_GT_FLOOR) -> FallbackResult:
    """gaps_desc is duration-descending. Accept the first gap whose chosen pair has
    c_gt >= floor at full reliability; else keep the highest-c_gt admissible pair at
    reduced reliability; else no_pair."""
    best = None  # (gap, pair, c_gt)
    for gap in gaps_desc:
        pair = choose_fn(gap)
        if pair is None or pair.i is None:
            continue
        c_gt = c_gt_fn(gap, pair)
        if c_gt >= floor:
            return FallbackResult(gap, pair, c_gt, 1.0, "ok")
        if best is None or c_gt > best[2]:
            best = (gap, pair, c_gt)
    if best is None:
        return FallbackResult(None, None, 0.0, 0.0, "no_pair")
    return FallbackResult(best[0], best[1], best[2], anchor_reliability(best[2], floor), "weak")
