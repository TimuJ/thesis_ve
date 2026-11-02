"""Cross-gap consistency orchestration: evaluate one clip, and a thin CLI.

``evaluate_clip`` ties the ADR pipeline together on in-memory frames, with the
heavy models injected as a FrameEmbedder (selection) and a list of PairScorers
(scoring) so the logic is pure and CPU-testable with fakes. Selection is frozen
on GT; the SR output is only ever scored at the pair GT chose. The CLI (``main``)
wires the real GPU adapters lazily and is exercised at the integration run, not
in unit tests.
"""
from __future__ import annotations

from typing import Protocol, Sequence

import cv2
import numpy as np

from scripts.lr_vcc.composite import compose_score

from .alignment import check_alignment
from .annotations import box_area_frac, box_aspect, find_gaps, present_mask
from .config import C_GT_FLOOR, MAX_WINDOW_OFFSET, MIN_GAP_S
from .consistency import consistency_scores, fidelity
from .crops import bbox_crop
from .selection import bound_windows, choose_pair
from .weak_anchor import select_with_fallback


class FrameEmbedder(Protocol):
    """Selection embedder: maps a list of object crops to a (N, D) array.

    Deliberately a different model from the PairScorers used for scoring.
    Concrete implementations (Task 7) conform structurally; nothing imports this.
    """

    def embed(self, crops: Sequence[np.ndarray]) -> np.ndarray: ...


def _norm_sr_to_gt_grid(gt_frames, sr_frames):
    """Bring SR onto the GT pixel grid, or report an aspect mismatch.

    Returns (sr_frames_on_gt_grid, None) when usable, else (None, reason).
    """
    gh, gw = gt_frames[0].shape[:2]
    sh, sw = sr_frames[0].shape[:2]
    if (sh, sw) == (gh, gw):
        return sr_frames, None
    if abs((sw / sh) - (gw / gh)) > 1e-3:
        return None, f"SR aspect {sw}x{sh} != GT aspect {gw}x{gh}"
    return [cv2.resize(f, (gw, gh)) for f in sr_frames], None


def evaluate_clip(
    gt_frames,
    sr_frames,
    ann,
    fps: float,
    manifest_frames: int,
    embedder: FrameEmbedder,
    scorers,
    *,
    c_gt_floor: float = C_GT_FLOOR,
    min_gap_s: float = MIN_GAP_S,
    max_offset: int = MAX_WINDOW_OFFSET,
    compute_lpips: bool = True,
    r_frame_rate: str = "30/1",
    avg_frame_rate: str = "30/1",
) -> dict:
    """Score one clip's SR against GT at the GT-frozen matched cross-gap pair.

    Statuses: "misaligned" (quarantined), "no_pair" (no qualifying gap / anchor),
    "ok"/"weak" (scored, the latter on a weak-anchor fallback).
    """
    # --- alignment gate -----------------------------------------------------
    counts = {
        "decoded": len(gt_frames),
        "groundtruth": ann.n,
        "out_of_view": len(ann.out_of_view),
        "full_occlusion": len(ann.full_occlusion),
        "manifest": manifest_frames,
    }
    align = check_alignment(counts, r_frame_rate, avg_frame_rate)
    reasons = list(align.reasons)
    if len(sr_frames) != len(gt_frames):
        reasons.append(f"sr {len(sr_frames)} != gt {len(gt_frames)} frames")
    if reasons:
        return {"status": "misaligned", "reasons": reasons}

    sr_grid, reason = _norm_sr_to_gt_grid(gt_frames, sr_frames)
    if sr_grid is None:
        return {"status": "misaligned", "reasons": [reason]}

    h, w = gt_frames[0].shape[:2]

    # --- gaps ---------------------------------------------------------------
    present = present_mask(ann)
    gaps = find_gaps(present, fps, min_gap_s=min_gap_s)
    if not gaps:
        return {"status": "no_pair", "reasons": ["no qualifying gap"]}

    # --- per-present-frame selection features (GT only) ---------------------
    boxes = [tuple(float(x) for x in ann.boxes[i]) for i in range(ann.n)]
    gt_crops = [bbox_crop(gt_frames[i], boxes[i]) if present[i] else None for i in range(ann.n)]
    present_idx = [i for i in range(ann.n) if present[i] and gt_crops[i] is not None]
    emb = embedder.embed([gt_crops[i] for i in present_idx])
    emb = np.asarray(emb, dtype=float)
    embeddings = np.zeros((ann.n, emb.shape[1]))
    for k, i in enumerate(present_idx):
        embeddings[i] = emb[k]
    areas = np.array([box_area_frac(boxes[i], w, h) for i in range(ann.n)])
    aspects = np.array([box_aspect(boxes[i]) for i in range(ann.n)])

    def choose_fn(gap):
        pre_idx, post_idx = bound_windows(present, gap, max_offset=max_offset)
        return choose_pair(pre_idx, post_idx, embeddings, areas, aspects)

    def c_gt_fn(gap, pair):
        return float(scorers[0](gt_crops[pair.i], gt_crops[pair.j]))

    fb = select_with_fallback(gaps, choose_fn, c_gt_fn, floor=c_gt_floor)
    if fb.status == "no_pair" or fb.pair is None:
        return {"status": "no_pair", "reasons": ["no admissible anchor"]}

    i, j = fb.pair.i, fb.pair.j
    box_i, box_j = boxes[i], boxes[j]
    gci, gcj = gt_crops[i], gt_crops[j]
    sci, scj = bbox_crop(sr_grid[i], box_i), bbox_crop(sr_grid[j], box_j)

    c_gt, c_sr = {}, {}
    for M in scorers:
        c_gt[M.name] = float(M(gci, gcj))
        c_sr[M.name] = float(M(sci, scj))
    s_M = consistency_scores(c_gt, c_sr)
    rel = fb.reliability
    composite = compose_score(list(s_M.values()), [rel] * len(s_M))

    fid = fidelity(sr_grid[i], gt_frames[i], sr_grid[j], gt_frames[j],
                   box_a=box_i, box_b=box_j, compute_lpips=compute_lpips)

    return {
        "status": fb.status,
        "i": i,
        "j": j,
        "gap_s": fb.gap.duration_s,
        "n_admissible": fb.pair.n_admissible,
        "c_gt": c_gt,
        "c_sr": c_sr,
        "s_M": s_M,
        "consistency": composite["score"],
        "reliability": rel,
        "fidelity": fid,
    }


def main(argv=None):  # pragma: no cover - thin IO glue, validated at integration
    """CLI: evaluate every clip in a manifest and write per-clip JSON records.

    The GPU adapters (CLIP embedder, DINOv3/LoFTR/SIFT scorers, SAM crops) are
    imported lazily here so the module stays torch-free at top level.
    """
    import argparse
    import csv
    import json

    from .annotations import load_annotations

    p = argparse.ArgumentParser(description="ADR cross-gap consistency evaluation")
    p.add_argument("--manifest", required=True)
    p.add_argument("--gt-dir", required=True)
    p.add_argument("--sr-dir", required=True)
    p.add_argument("--ann-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--no-lpips", action="store_true")
    args = p.parse_args(argv)

    from .embed_clip import ClipEmbedder
    from .scorers import SiftTextureScorer
    from .scorers_gpu import DinoSemanticScorer, LoftrGeometryScorer
    from .video_io import decode_frames  # provided at integration

    embedder = ClipEmbedder()
    scorers = [SiftTextureScorer(), DinoSemanticScorer(), LoftrGeometryScorer()]

    records = []
    with open(args.manifest) as f:
        for row in csv.DictReader(f):
            name = row["video"]
            stem = name.rsplit(".", 1)[0]
            gt = decode_frames(f"{args.gt_dir}/{name}")
            sr = decode_frames(f"{args.sr_dir}/{name}")
            ann = load_annotations(f"{args.ann_dir}/{stem}")
            rec = evaluate_clip(gt, sr, ann, float(row["fps"]), int(row["frames"]),
                                embedder, scorers, compute_lpips=not args.no_lpips)
            rec["video"] = name
            records.append(rec)

    with open(args.out, "w") as f:
        json.dump(records, f, indent=2)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
