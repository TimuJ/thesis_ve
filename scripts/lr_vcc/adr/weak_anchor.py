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

from .config import BIG_AREA_FRAC, C_GT_FLOOR
from .selection import _pairs_too_close


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


def size_reliability(min_area: float, big_area: float = BIG_AREA_FRAC) -> float:
    """1 when the smaller crop is at least ``big_area`` of the frame, scaling linearly
    to 0 for a vanishing object. Folds object size into reliability so tiny crops (a
    distant bus, a 1%-area mouse) are kept as low-reliability examples, not trusted."""
    return float(max(0.0, min(1.0, min_area / big_area)))


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


def select_all_gaps(gaps_desc, choose_fn: Callable, c_gt_fn: Callable,
                    floor: float = C_GT_FLOOR):
    """One FallbackResult per gap that yields an admissible pair, in gaps_desc
    order (used for multi-pair review: "distinct pairs per video"). A gap with
    c_gt >= floor is "ok" at full reliability; a lower one is "weak" at reduced
    reliability; a gap with no admissible pair is skipped. Returns [] when no gap
    yields a pair."""
    results = []
    for gap in gaps_desc:
        pair = choose_fn(gap)
        if pair is None or pair.i is None:
            continue
        c_gt = c_gt_fn(gap, pair)
        rel = 1.0 if c_gt >= floor else anchor_reliability(c_gt, floor)
        results.append(FallbackResult(gap, pair, c_gt, rel, "ok" if c_gt >= floor else "weak"))
    return results


def select_all_pairs(gaps_desc, choose_pairs_fn: Callable, c_gt_fn: Callable,
                     *, floor: float = C_GT_FLOOR, top_k: int, min_pair_sep: int):
    """Up to ``top_k`` distinct pairs per video for the richer review set. Each gap's
    ``choose_pairs_fn`` returns its cost-sorted, diversity-filtered candidate pairs;
    we fill **round-robin across gaps** (best pair from each gap first, then the next
    from each, ...) so a multi-gap video gets coverage rather than one gap monopolising
    every slot — multiple pairs from one gap are still allowed. A candidate within
    ``min_pair_sep`` of an already-kept pair on BOTH endpoints is skipped (video-level
    dedup across gaps). Returns a list of ``FallbackResult`` (status "ok"/"weak" by the
    c_gt floor), ``[]`` when no gap yields a pair."""
    per_gap = []
    for gap in gaps_desc:
        pairs = list(choose_pairs_fn(gap))
        if pairs:
            per_gap.append((gap, pairs))
    kept = []
    round_idx = 0
    while len(kept) < top_k and any(round_idx < len(pairs) for _, pairs in per_gap):
        for gap, pairs in per_gap:
            if round_idx >= len(pairs):
                continue
            pair = pairs[round_idx]
            if any(_pairs_too_close(pair.i, pair.j, k.pair.i, k.pair.j, min_pair_sep)
                   for k in kept):
                continue
            c_gt = c_gt_fn(gap, pair)
            rel = 1.0 if c_gt >= floor else anchor_reliability(c_gt, floor)
            kept.append(FallbackResult(gap, pair, c_gt, rel, "ok" if c_gt >= floor else "weak"))
            if len(kept) >= top_k:
                break
        round_idx += 1
    return kept
