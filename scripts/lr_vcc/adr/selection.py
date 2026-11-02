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

from .config import A_MIN, LAMBDA_AREA, LAMBDA_ASPECT, MAX_WINDOW_OFFSET


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
                   lambda_area: float = LAMBDA_AREA, lambda_aspect: float = LAMBDA_ASPECT) -> float:
    cost = _cos_dist(emb_a, emb_b)
    cost += lambda_area * abs(math.log(area_a / area_b))
    cost += lambda_aspect * abs(math.log(aspect_a / aspect_b))
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
                lambda_aspect: float = LAMBDA_ASPECT) -> PairChoice:
    """Argmin selection_cost over straddling pairs with both sides' area >= a_min.
    Returns i=j=None when no admissible pair exists."""
    embeddings = np.asarray(embeddings, float)
    areas = np.asarray(areas, float)
    aspects = np.asarray(aspects, float)
    pre_adm = [a for a in pre_idx if areas[a] >= a_min]
    post_adm = [b for b in post_idx if areas[b] >= a_min]
    n_admissible = len(pre_adm) * len(post_adm)
    best = None
    for a in pre_adm:
        for b in post_adm:
            c = selection_cost(embeddings[a], embeddings[b], areas[a], areas[b],
                               aspects[a], aspects[b], lambda_area, lambda_aspect)
            if best is None or c < best[2]:
                best = (a, b, c)
    if best is None:
        return PairChoice(None, None, float("inf"), 0)
    return PairChoice(best[0], best[1], best[2], n_admissible)
