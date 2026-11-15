"""Probability-weighted 1-5 scoring for the MLLM consistency judge.

The judge is asked for an integer 1-5 per axis. Instead of taking the argmax
token (which clusters on 3-4 and is unstable run-to-run), we read the model's
logits for the five digit tokens '1'..'5' at the answer position, softmax over
just those five, and return the expected value E[score] = sum_k p_k * k in
[1, 5] -- the G-Eval / Q-Align scoring trick. It yields a continuous, tie-broken
score that is more reproducible than a single argmax token.

This module is pure: it takes the five logits as plain floats. Extracting them
from a model's output (tokenizer ids for '1'..'5', the answer position) lives in
judge.py, which needs the model and a GPU.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

SCALE = (1, 2, 3, 4, 5)


def weighted_score(digit_logits: Sequence[float]) -> float:
    """Expected 1-5 score from the logits of the tokens '1','2','3','4','5', in
    that order. Softmax over just those five, weighted by [1..5]. Numerically
    stable (subtracts the max logit before exp). Raises ValueError unless given
    exactly five finite logits."""
    logits = list(digit_logits)
    if len(logits) != len(SCALE):
        raise ValueError(f"expected {len(SCALE)} logits (for '1'..'5'), got {len(logits)}")
    if any(not math.isfinite(float(x)) for x in logits):
        raise ValueError("logits must be finite")
    hi = max(logits)
    exps = [math.exp(float(x) - hi) for x in logits]
    z = sum(exps)
    return sum((e / z) * k for e, k in zip(exps, SCALE))


def axis_scores(logits_by_axis: Mapping[str, Sequence[float]]) -> dict[str, float]:
    """Apply :func:`weighted_score` to each axis's five logits, preserving keys
    (e.g. {"texture": [...], "geometry": [...], "semantic": [...]})."""
    return {axis: weighted_score(logits) for axis, logits in logits_by_axis.items()}
