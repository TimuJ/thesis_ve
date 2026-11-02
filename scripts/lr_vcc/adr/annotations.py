"""LaSOT annotation parsing, present-frame mask, and gap detection.

LaSOT ships per-sequence: `groundtruth.txt` (one `x,y,w,h` box per line),
`out_of_view.txt` and `full_occlusion.txt` (a single comma-separated row of 0/1
per frame). We have boxes, not masks.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import MIN_GAP_S


@dataclass
class Annotations:
    boxes: np.ndarray        # (N, 4) float, x, y, w, h
    out_of_view: np.ndarray  # (N,) bool
    full_occlusion: np.ndarray  # (N,) bool
    n: int


@dataclass
class Gap:
    start: int               # first absent frame index (inclusive)
    end: int                 # last absent frame index (inclusive)
    duration_s: float
    pre_run: tuple           # (lo, hi) inclusive present run adjacent before the gap
    post_run: tuple          # (lo, hi) inclusive present run adjacent after the gap


def _read_flag_row(path: Path) -> np.ndarray:
    """Parse a LaSOT flag file: a single comma-separated row (or one value per
    line) of 0/1 into a bool array."""
    txt = path.read_text().strip()
    vals = [v for v in txt.replace("\n", ",").split(",") if v != ""]
    return np.array([int(float(v)) for v in vals], dtype=bool)


def load_annotations(seq_dir) -> Annotations:
    seq_dir = Path(seq_dir)
    boxes = np.loadtxt(seq_dir / "groundtruth.txt", delimiter=",", ndmin=2).astype(float)
    return Annotations(
        boxes=boxes,
        out_of_view=_read_flag_row(seq_dir / "out_of_view.txt"),
        full_occlusion=_read_flag_row(seq_dir / "full_occlusion.txt"),
        n=boxes.shape[0],
    )


def present_mask(ann: Annotations) -> np.ndarray:
    """A frame is present when the object is neither out-of-view nor fully
    occluded and its box is non-degenerate."""
    w, h = ann.boxes[:, 2], ann.boxes[:, 3]
    valid = (w > 0) & (h > 0)
    return valid & ~ann.out_of_view & ~ann.full_occlusion


def box_area_frac(box, frame_w: int, frame_h: int) -> float:
    _, _, w, h = box
    return float(w * h) / float(frame_w * frame_h)


def box_aspect(box) -> float:
    _, _, w, h = box
    return float(w) / float(h) if h else 0.0


def find_gaps(present: np.ndarray, fps: float, min_gap_s: float = MIN_GAP_S) -> list:
    """Maximal absent runs of duration >= min_gap_s that have a present frame on
    both sides, each with its adjacent present runs, sorted by duration desc."""
    present = np.asarray(present, dtype=bool)
    n = len(present)
    gaps: list[Gap] = []
    i = 0
    while i < n:
        if present[i]:
            i += 1
            continue
        j = i
        while j < n and not present[j]:
            j += 1
        start, end = i, j - 1
        dur = (end - start + 1) / fps
        has_pre = start - 1 >= 0 and present[start - 1]
        has_post = end + 1 < n and present[end + 1]
        if dur >= min_gap_s and has_pre and has_post:
            a = start - 1
            while a - 1 >= 0 and present[a - 1]:
                a -= 1
            b = end + 1
            while b + 1 < n and present[b + 1]:
                b += 1
            gaps.append(Gap(start, end, dur, (a, start - 1), (end + 1, b)))
        i = j
    gaps.sort(key=lambda g: g.duration_s, reverse=True)
    return gaps
