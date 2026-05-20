"""Softmax-weighted log-mean composition of sub-metric (score, reliability) pairs."""
import math
from typing import Sequence


def _softmax(xs: Sequence[float], temperature: float = 0.2) -> list[float]:
    z = [x / temperature for x in xs]
    z_max = max(z)
    exps = [math.exp(zi - z_max) for zi in z]
    s = sum(exps)
    return [e / s for e in exps]


def compose_score(scores: Sequence[float], reliabilities: Sequence[float],
                  temperature: float = 0.2, eps: float = 1e-6,
                  low_confidence_floor: float = 0.2) -> dict:
    """Softmax-weight reliabilities, then exp(sum w_i log(score_i + eps)).

    Returns: {"score": float, "weights": list[float], "low_confidence": bool}.
    """
    assert len(scores) == len(reliabilities)
    weights = _softmax(reliabilities, temperature=temperature)
    log_sum = sum(w * math.log(s + eps) for w, s in zip(weights, scores))
    score = math.exp(log_sum)
    low_conf = all(r < low_confidence_floor for r in reliabilities)
    return {"score": score, "weights": weights, "low_confidence": low_conf}
