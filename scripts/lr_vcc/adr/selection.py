"""Configuration-matched cross-gap pair selection.

Selection uses an embedding (a model separate from the scorers) plus geometric
terms on object area and aspect, so the chosen pre/post pair is genuinely alike
("same configuration"), not merely the same object at a different scale. The pair
is chosen on GT and reused for every model (frozen on GT).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import A_MIN, GLOBAL_TOPK, LAMBDA_AREA, LAMBDA_ASPECT, LAMBDA_SIZE, MAX_WINDOW_OFFSET


@dataclass
class PairChoice:
    i: "int | None"
    j: "int | None"
    cost: float
    n_admissible: int


def _cos_dist(a, b) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 1.0
    return 1.0 - float(np.dot(a, b) / (na * nb))


def selection_cost(emb_a, emb_b, area_a, area_b, aspect_a, aspect_b,
                   lambda_area: float = LAMBDA_AREA, lambda_aspect: float = LAMBDA_ASPECT,
                   lambda_size: float = 0.0, lambda_appear: float = 1.0) -> float:
    cost = lambda_appear * _cos_dist(emb_a, emb_b)
    cost += lambda_area * abs(math.log(area_a / area_b))
    cost += lambda_aspect * abs(math.log(aspect_a / aspect_b))
    # reward larger objects: -log(min-area) shrinks as the smaller side grows, so a
    # bigger well-matched pair wins over a tiny one. Off (0.0) by default.
    if lambda_size:
        cost += lambda_size * (-math.log(max(min(area_a, area_b), 1e-9)))
    return float(cost)


def bound_windows(present, gap, max_offset: int = MAX_WINDOW_OFFSET):
    """The present frames adjacent to each gap edge, truncated to the max_offset
    frames nearest the edge. Uses the gap's recorded adjacent present runs, so it
    never jumps across an intervening gap."""
    pre_lo, pre_hi = gap.pre_run
    post_lo, post_hi = gap.post_run
    pre = list(range(pre_lo, pre_hi + 1))[-max_offset:]   # nearest the gap edge
    post = list(range(post_lo, post_hi + 1))[:max_offset]
    return pre, post


def choose_pair(pre_idx, post_idx, embeddings, areas, aspects,
                a_min: float = A_MIN, lambda_area: float = LAMBDA_AREA,
                lambda_aspect: float = LAMBDA_ASPECT,
                lambda_size: float = LAMBDA_SIZE, min_sep: int = 0,
                blur_pen=None, edge_pen=None, lambda_blur: float = 0.0,
                lambda_border: float = 0.0, lambda_appear: float = 1.0,
                max_appearance_dist: "float | None" = None) -> PairChoice:
    """Argmin selection_cost over straddling pairs with both sides' area >= a_min.
    With lambda_size > 0 the cost rewards larger objects, so a bigger well-matched
    pair wins over a tiny one. ``min_sep`` rejects pairs closer than that many frames
    (used by the long-range path, whose candidates share no gap to guarantee
    separation).

    Reviewer-feedback gates: ``max_appearance_dist`` rejects a pair whose two frames'
    embeddings are farther apart than that (a different pose/angle, which can't be
    consistency-checked); ``lambda_appear`` amplifies that same embedding distance in
    the cost; ``blur_pen``/``edge_pen`` are per-frame penalties (see quality.py) added
    at weights ``lambda_blur``/``lambda_border`` so a soft or small-on-the-edge frame
    loses to a sharper, better-placed one. Returns i=j=None when no admissible pair
    exists (including when the gate rejects every pair)."""
    embeddings = np.asarray(embeddings, float)
    areas = np.asarray(areas, float)
    aspects = np.asarray(aspects, float)
    pre_adm = [a for a in pre_idx if areas[a] >= a_min]
    post_adm = [b for b in post_idx if areas[b] >= a_min]
    best = None
    n_admissible = 0
    for a in pre_adm:
        for b in post_adm:
            if min_sep and abs(b - a) < min_sep:
                continue
            if max_appearance_dist is not None and _cos_dist(embeddings[a], embeddings[b]) > max_appearance_dist:
                continue
            n_admissible += 1
            c = selection_cost(embeddings[a], embeddings[b], areas[a], areas[b],
                               aspects[a], aspects[b], lambda_area, lambda_aspect,
                               lambda_size, lambda_appear)
            if blur_pen is not None:
                c += lambda_blur * (float(blur_pen[a]) + float(blur_pen[b]))
            if edge_pen is not None:
                c += lambda_border * (float(edge_pen[a]) + float(edge_pen[b]))
            if best is None or c < best[2]:
                best = (a, b, c)
    if best is None:
        return PairChoice(None, None, float("inf"), 0)
    return PairChoice(best[0], best[1], best[2], n_admissible)


def _pairs_too_close(ai: int, aj: int, bi: int, bj: int, min_sep: int) -> bool:
    """Two pairs are "too close" (near-duplicates) iff BOTH endpoints differ by less
    than ``min_sep`` frames. So pairs one window apart on either side stay distinct."""
    return abs(ai - bi) < min_sep and abs(aj - bj) < min_sep


def choose_pairs(pre_idx, post_idx, embeddings, areas, aspects, *,
                 top_k: int, min_pair_sep: int, pose_gate_fn=None,
                 a_min: float = A_MIN, lambda_area: float = LAMBDA_AREA,
                 lambda_aspect: float = LAMBDA_ASPECT, lambda_size: float = LAMBDA_SIZE,
                 blur_pen=None, edge_pen=None, lambda_blur: float = 0.0,
                 lambda_border: float = 0.0, lambda_appear: float = 1.0,
                 max_appearance_dist: "float | None" = None) -> "list[PairChoice]":
    """Up to ``top_k`` admissible straddling pairs, lowest ``selection_cost`` first,
    kept so no two are :func:`_pairs_too_close` (diverse frame offsets) and each passes
    ``pose_gate_fn(i, j)`` when given. Same admissibility (area >= a_min, appearance
    gate) and cost terms as :func:`choose_pair`; this is its top-K, diversity-aware,
    pose-gated generalization. ``n_admissible`` on each returned pair is the total
    admissible count. Empty list when nothing qualifies."""
    embeddings = np.asarray(embeddings, float)
    areas = np.asarray(areas, float)
    aspects = np.asarray(aspects, float)
    pre_adm = [a for a in pre_idx if areas[a] >= a_min]
    post_adm = [b for b in post_idx if areas[b] >= a_min]
    cands = []
    n_admissible = 0
    for a in pre_adm:
        for b in post_adm:
            if max_appearance_dist is not None and _cos_dist(embeddings[a], embeddings[b]) > max_appearance_dist:
                continue
            n_admissible += 1
            c = selection_cost(embeddings[a], embeddings[b], areas[a], areas[b],
                               aspects[a], aspects[b], lambda_area, lambda_aspect,
                               lambda_size, lambda_appear)
            if blur_pen is not None:
                c += lambda_blur * (float(blur_pen[a]) + float(blur_pen[b]))
            if edge_pen is not None:
                c += lambda_border * (float(edge_pen[a]) + float(edge_pen[b]))
            cands.append((c, a, b))
    cands.sort(key=lambda t: t[0])
    kept: "list[PairChoice]" = []
    for c, a, b in cands:
        if pose_gate_fn is not None and not pose_gate_fn(a, b):
            continue
        if any(_pairs_too_close(a, b, k.i, k.j, min_pair_sep) for k in kept):
            continue
        kept.append(PairChoice(a, b, c, n_admissible))
        if len(kept) >= top_k:
            break
    return kept


def global_big_windows(present, areas, topk: int = GLOBAL_TOPK):
    """Candidate index sets for a long-range (non-gap) pair: the ``topk`` largest
    present objects in each half of the clip. Pairing one from the early half with
    one from the late half guarantees temporal separation; choose_pair with a size
    reward then takes the biggest well-matched pair (Option B — a big object that is
    big only at the clip's start and end still yields a reviewable pair)."""
    present = np.asarray(present, bool)
    areas = np.asarray(areas, float)
    n = len(present)
    mid = n // 2
    early = [i for i in range(mid) if present[i] and areas[i] > 0]
    late = [i for i in range(mid, n) if present[i] and areas[i] > 0]
    early.sort(key=lambda i: -areas[i])
    late.sort(key=lambda i: -areas[i])
    return sorted(early[:topk]), sorted(late[:topk])
