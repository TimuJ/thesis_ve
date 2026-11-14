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
from .config import (
    C_GT_FLOOR,
    GLOBAL_MIN_SEP_S,
    LAMBDA_APPEAR,
    LAMBDA_BLUR,
    LAMBDA_BORDER,
    LONG_RANGE_RELIABILITY,
    MAX_APPEARANCE_DIST,
    MAX_WINDOW_OFFSET,
    MIN_GAP_S,
    MIN_PAIR_SEP_S,
    ROMA_POSE_GATE_THRESH,
    TOP_K_PER_VIDEO,
)
from .consistency import consistency_scores, fidelity
from .crops import bbox_crop
from .quality import blur_penalties, edge_penalty, laplacian_var
from .reliability_axes import axis_reliabilities
from .selection import bound_windows, choose_pair, choose_pairs, global_big_windows
from .weak_anchor import (
    anchor_reliability,
    select_all_pairs,
    select_with_fallback,
    size_reliability,
)


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


def _prepare_clip(gt_frames, sr_frames, ann, fps, manifest_frames, embedder, scorers,
                  *, anchor_metric, min_gap_s, max_offset, r_frame_rate, avg_frame_rate,
                  allow_no_gap=False, pose_gate=None, top_k=TOP_K_PER_VIDEO):
    """Shared setup for the single-pair and per-gap evaluations.

    Returns ``(error_dict, None)`` when the clip is quarantined or has no
    qualifying gap, else ``(None, ctx)`` where ctx carries the SR grid, the gaps,
    the GT crops and boxes, and the frozen-on-GT ``choose_fn`` / ``c_gt_fn``.

    ``allow_no_gap`` keeps going when there is no straddled gap, so the per-gap
    path can still emit an Option-B long-range pair (a big object that never
    disappears, e.g. present in every frame); the single-pair path leaves it False
    and returns no_pair as before.
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
        return {"status": "misaligned", "reasons": reasons}, None

    sr_grid, reason = _norm_sr_to_gt_grid(gt_frames, sr_frames)
    if sr_grid is None:
        return {"status": "misaligned", "reasons": [reason]}, None
    res_mismatch = sr_grid is not sr_frames  # SR was resampled onto the GT grid

    h, w = gt_frames[0].shape[:2]

    # --- gaps ---------------------------------------------------------------
    present = present_mask(ann)
    gaps = find_gaps(present, fps, min_gap_s=min_gap_s)
    if not gaps and not allow_no_gap:
        return {"status": "no_pair", "reasons": ["no qualifying gap"]}, None

    # --- per-present-frame selection features (GT only) ---------------------
    boxes = [tuple(float(x) for x in ann.boxes[i]) for i in range(ann.n)]
    gt_crops = [bbox_crop(gt_frames[i], boxes[i]) if present[i] else None for i in range(ann.n)]
    present_idx = [i for i in range(ann.n) if present[i] and gt_crops[i] is not None]
    if not present_idx:
        return {"status": "no_pair", "reasons": ["no croppable present frame"]}, None
    emb = np.asarray(embedder.embed([gt_crops[i] for i in present_idx]), dtype=float)
    embeddings = np.zeros((ann.n, emb.shape[1]))
    for k, i in enumerate(present_idx):
        embeddings[i] = emb[k]
    # area 0 for present-but-uncroppable frames (off-frame box) so they are never
    # selected into a pair and never reach a scorer as a None crop
    areas = np.array([
        box_area_frac(boxes[i], w, h) if gt_crops[i] is not None else 0.0
        for i in range(ann.n)
    ])
    aspects = np.array([box_aspect(boxes[i]) for i in range(ann.n)])
    # reviewer-feedback quality signals on the GT crop/box (frozen on GT, like selection):
    # blur (relative sharpness) and size-scaled edge proximity. They gate + re-rank
    # selection and fold into reliability so soft / small-on-the-edge pairs sort down.
    lap = np.zeros(ann.n)
    for i in present_idx:
        lap[i] = laplacian_var(gt_crops[i])
    blur_pen = blur_penalties(lap, present_idx)
    edge_pen = np.array([
        edge_penalty(boxes[i], w, h, areas[i]) if gt_crops[i] is not None else 0.0
        for i in range(ann.n)
    ])

    def choose_fn(gap):
        pre_idx, post_idx = bound_windows(present, gap, max_offset=max_offset)
        return choose_pair(pre_idx, post_idx, embeddings, areas, aspects,
                           blur_pen=blur_pen, edge_pen=edge_pen,
                           lambda_blur=LAMBDA_BLUR, lambda_border=LAMBDA_BORDER,
                           lambda_appear=LAMBDA_APPEAR, max_appearance_dist=MAX_APPEARANCE_DIST)

    # the admissibility anchor is a semantic metric (DINO) chosen by name, not by
    # list position, so the c_gt floor is compared against a comparable quantity
    anchor = next((M for M in scorers if M.name == anchor_metric), scorers[0])

    def c_gt_fn(gap, pair):
        return float(anchor(gt_crops[pair.i], gt_crops[pair.j]))

    # multi-pair selection: RoMa pose gate (decision (a) — gate only, CLIP still ranks)
    min_pair_sep = max(1, round(MIN_PAIR_SEP_S * fps))
    pose_gate_fn = None
    if pose_gate is not None:
        def pose_gate_fn(i, j):
            return float(pose_gate(gt_crops[i], gt_crops[j])) >= ROMA_POSE_GATE_THRESH

    def choose_pairs_fn(gap):
        pre_idx, post_idx = bound_windows(present, gap, max_offset=max_offset)
        return choose_pairs(pre_idx, post_idx, embeddings, areas, aspects,
                            top_k=top_k, min_pair_sep=min_pair_sep, pose_gate_fn=pose_gate_fn,
                            blur_pen=blur_pen, edge_pen=edge_pen,
                            lambda_blur=LAMBDA_BLUR, lambda_border=LAMBDA_BORDER,
                            lambda_appear=LAMBDA_APPEAR, max_appearance_dist=MAX_APPEARANCE_DIST)

    ctx = {
        "sr_grid": sr_grid, "res_mismatch": res_mismatch, "gaps": gaps,
        "gt_crops": gt_crops, "boxes": boxes,
        "choose_fn": choose_fn, "c_gt_fn": c_gt_fn, "choose_pairs_fn": choose_pairs_fn,
        "min_pair_sep": min_pair_sep,
        "present": present, "embeddings": embeddings, "areas": areas,
        "aspects": aspects, "anchor": anchor, "fps": fps,
        "blur_pen": blur_pen, "edge_pen": edge_pen,
    }
    return None, ctx


def _pair_reliability(ctx, i, j, base_rel, type_factor: float = 1.0):
    """Fold object size, sharpness and edge-proximity (and pair type) into
    reliability: a tiny, soft, or small-on-the-edge crop, or a non-gap
    (no-disappearance) pair, is kept as a low-reliability example, not trusted.
    Returns ``(reliability, factors)`` where factors breaks out each multiplier so
    the gallery can name WHY a pair is weak."""
    amin = min(ctx["areas"][i], ctx["areas"][j])
    size_f = size_reliability(amin)
    sharp_f = min(1.0 - float(ctx["blur_pen"][i]), 1.0 - float(ctx["blur_pen"][j]))
    edge_f = min(1.0 - float(ctx["edge_pen"][i]), 1.0 - float(ctx["edge_pen"][j]))
    rel = float(base_rel * size_f * sharp_f * edge_f * type_factor)
    factors = {"anchor": float(base_rel), "size": float(size_f),
               "sharp": float(sharp_f), "edge": float(edge_f), "type": float(type_factor)}
    return rel, factors


def _score_pair(scorers, gt_frames, ctx, i, j, *, gap_s, n_admissible, status,
                reliability, reliability_factors, compute_lpips, pair_type: str = "adr") -> dict:
    """Score one GT-frozen pair (i, j) against SR and build its result record."""
    sr_grid, gt_crops, boxes = ctx["sr_grid"], ctx["gt_crops"], ctx["boxes"]
    box_i, box_j = boxes[i], boxes[j]
    gci, gcj = gt_crops[i], gt_crops[j]
    sci, scj = bbox_crop(sr_grid[i], box_i), bbox_crop(sr_grid[j], box_j)

    c_gt, c_sr = {}, {}
    for M in scorers:
        c_gt[M.name] = float(M(gci, gcj))
        c_sr[M.name] = float(M(sci, scj))
    s_M = consistency_scores(c_gt, c_sr)
    # per-axis reliability: the pair's scalar reliability times each axis's own GT-side
    # signal, so a no-signal axis (c_gt in the noise) no longer free-passes the composite
    axis_rel = axis_reliabilities(scorers, gci, gcj)
    names = list(s_M.keys())
    composite = compose_score([s_M[n] for n in names],
                              [reliability * axis_rel.get(n, 1.0) for n in names])
    fid = fidelity(sr_grid[i], gt_frames[i], sr_grid[j], gt_frames[j],
                   box_a=box_i, box_b=box_j, compute_lpips=compute_lpips)
    return {
        "status": status,
        "pair_type": pair_type,
        "i": i,
        "j": j,
        "gap_s": gap_s,
        "n_admissible": n_admissible,
        "c_gt": c_gt,
        "c_sr": c_sr,
        "s_M": s_M,
        "consistency": composite["score"],
        "reliability": reliability,
        "reliability_factors": reliability_factors,
        "axis_reliability": axis_rel,
        "resolution_mismatch": ctx["res_mismatch"],
        "fidelity": fid,
    }


def _long_range_pair(ctx, scorers, gt_frames, best_gap_area, c_gt_floor, compute_lpips):
    """Option B: the biggest well-matched temporally-separated pair across the whole
    clip (no disappearance). Returned only when it is meaningfully bigger than any
    gap pair (or there is none) and spans at least GLOBAL_MIN_SEP_S — so a large
    object that is big only at the clip's start and end still yields a reviewable
    pair, flagged low-reliability because it does not straddle a disappearance."""
    import math

    early, late = global_big_windows(ctx["present"], ctx["areas"])
    min_sep = int(math.ceil(GLOBAL_MIN_SEP_S * ctx["fps"]))
    # the biggest well-matched pair that is at least min_sep apart: without the
    # separation constraint choose_pair would grab the biggest frames even when they
    # cluster near the clip's middle, and the pair would then be discarded as too close
    pair = choose_pair(early, late, ctx["embeddings"], ctx["areas"], ctx["aspects"],
                       min_sep=min_sep, blur_pen=ctx["blur_pen"], edge_pen=ctx["edge_pen"],
                       lambda_blur=LAMBDA_BLUR, lambda_border=LAMBDA_BORDER,
                       lambda_appear=LAMBDA_APPEAR, max_appearance_dist=MAX_APPEARANCE_DIST)
    if pair.i is None:
        return None
    i, j = pair.i, pair.j
    lr_area = min(ctx["areas"][i], ctx["areas"][j])
    if abs(j - i) < min_sep:                         # not temporally separated enough
        return None
    if lr_area <= best_gap_area * 1.2:               # gap already offers an equal/bigger object
        return None
    c_gt = float(ctx["anchor"](ctx["gt_crops"][i], ctx["gt_crops"][j]))
    base = 1.0 if c_gt >= c_gt_floor else anchor_reliability(c_gt, c_gt_floor)
    rel, factors = _pair_reliability(ctx, i, j, base, type_factor=LONG_RANGE_RELIABILITY)
    return _score_pair(scorers, gt_frames, ctx, i, j, gap_s=abs(j - i) / ctx["fps"],
                       n_admissible=pair.n_admissible,
                       status="ok" if c_gt >= c_gt_floor else "weak",
                       reliability=rel, reliability_factors=factors,
                       compute_lpips=compute_lpips, pair_type="long_range")


def evaluate_clip(
    gt_frames,
    sr_frames,
    ann,
    fps: float,
    manifest_frames: int,
    embedder: FrameEmbedder,
    scorers,
    *,
    anchor_metric: str = "dino",
    c_gt_floor: float = C_GT_FLOOR,
    min_gap_s: float = MIN_GAP_S,
    max_offset: int = MAX_WINDOW_OFFSET,
    compute_lpips: bool = True,
    r_frame_rate: str = "30/1",
    avg_frame_rate: str = "30/1",
) -> dict:
    """Score one clip's SR against GT at the single GT-frozen matched cross-gap
    pair (the best-anchored qualifying gap).

    Statuses: "misaligned" (quarantined), "no_pair" (no qualifying gap / anchor),
    "ok"/"weak" (scored, the latter on a weak-anchor fallback).
    """
    err, ctx = _prepare_clip(
        gt_frames, sr_frames, ann, fps, manifest_frames, embedder, scorers,
        anchor_metric=anchor_metric, min_gap_s=min_gap_s, max_offset=max_offset,
        r_frame_rate=r_frame_rate, avg_frame_rate=avg_frame_rate)
    if err is not None:
        return err
    fb = select_with_fallback(ctx["gaps"], ctx["choose_fn"], ctx["c_gt_fn"], floor=c_gt_floor)
    if fb.status == "no_pair" or fb.pair is None:
        return {"status": "no_pair", "reasons": ["no admissible anchor"]}
    rel, factors = _pair_reliability(ctx, fb.pair.i, fb.pair.j, fb.reliability)
    return _score_pair(scorers, gt_frames, ctx, fb.pair.i, fb.pair.j,
                       gap_s=fb.gap.duration_s, n_admissible=fb.pair.n_admissible,
                       status=fb.status, reliability=rel, reliability_factors=factors,
                       compute_lpips=compute_lpips)


def evaluate_clip_pairs(
    gt_frames,
    sr_frames,
    ann,
    fps: float,
    manifest_frames: int,
    embedder: FrameEmbedder,
    scorers,
    *,
    anchor_metric: str = "dino",
    c_gt_floor: float = C_GT_FLOOR,
    min_gap_s: float = MIN_GAP_S,
    max_offset: int = MAX_WINDOW_OFFSET,
    compute_lpips: bool = True,
    r_frame_rate: str = "30/1",
    avg_frame_rate: str = "30/1",
    pose_gate=None,
    top_k: int = TOP_K_PER_VIDEO,
) -> dict:
    """Score up to ``top_k`` GT-frozen matched pairs per video ("distinct pairs per
    video" — multiple per gap allowed), selected round-robin across gaps so every
    qualifying gap is covered. ``pose_gate`` (optional, GT crops → pose-match score)
    gates reviewability; CLIP still ranks. Returns ``{"status", "pairs": [record, ...]}``
    where each record has the same shape as :func:`evaluate_clip`'s. Quarantine / no-gap
    return the same error dict with an empty ``pairs`` list; the per-clip headline is
    the best-anchored pair (``pairs[0]``).
    """
    err, ctx = _prepare_clip(
        gt_frames, sr_frames, ann, fps, manifest_frames, embedder, scorers,
        anchor_metric=anchor_metric, min_gap_s=min_gap_s, max_offset=max_offset,
        r_frame_rate=r_frame_rate, avg_frame_rate=avg_frame_rate, allow_no_gap=True,
        pose_gate=pose_gate, top_k=top_k)
    if err is not None:
        return {**err, "pairs": []}
    fbs = select_all_pairs(ctx["gaps"], ctx["choose_pairs_fn"], ctx["c_gt_fn"],
                           floor=c_gt_floor, top_k=top_k, min_pair_sep=ctx["min_pair_sep"])
    pairs, best_gap_area = [], 0.0
    for fb in fbs:
        i, j = fb.pair.i, fb.pair.j
        rel, factors = _pair_reliability(ctx, i, j, fb.reliability)
        pairs.append(_score_pair(scorers, gt_frames, ctx, i, j, gap_s=fb.gap.duration_s,
                                 n_admissible=fb.pair.n_admissible, status=fb.status,
                                 reliability=rel, reliability_factors=factors,
                                 compute_lpips=compute_lpips, pair_type="adr"))
        best_gap_area = max(best_gap_area, min(ctx["areas"][i], ctx["areas"][j]))
    lr = _long_range_pair(ctx, scorers, gt_frames, best_gap_area, c_gt_floor, compute_lpips)
    if lr is not None:
        pairs.append(lr)
    if pairs:
        return {"status": "ok", "pairs": pairs}
    # diagnostic: record why neither a gap pair nor a long-range pair was produced
    import math

    early, late = global_big_windows(ctx["present"], ctx["areas"])
    min_sep = int(math.ceil(GLOBAL_MIN_SEP_S * ctx["fps"]))
    diag = {"n_gaps": len(ctx["gaps"]), "n_fbs": len(fbs),
            "lr_early_ct": len(early), "lr_late_ct": len(late),
            "sep_threshold_fr": min_sep, "best_gap_area": best_gap_area}
    if early and late:
        pr = choose_pair(early, late, ctx["embeddings"], ctx["areas"], ctx["aspects"],
                         min_sep=min_sep, blur_pen=ctx["blur_pen"], edge_pen=ctx["edge_pen"],
                         lambda_blur=LAMBDA_BLUR, lambda_border=LAMBDA_BORDER,
                         lambda_appear=LAMBDA_APPEAR, max_appearance_dist=MAX_APPEARANCE_DIST)
        if pr.i is None:
            diag["lr_pair"] = "choose_pair_none"
        else:
            diag.update(lr_i=int(pr.i), lr_j=int(pr.j), lr_sep_fr=int(abs(pr.j - pr.i)),
                        lr_area=float(min(ctx["areas"][pr.i], ctx["areas"][pr.j])))
    return {"status": "no_pair", "pairs": [], **diag}


def _clip_paths(video, gt_dir, sr_dir, ann_dir):
    """Resolve a manifest ``video`` field to (gt_mp4, sr_mp4, ann_seq_dir).

    The manifest stores bare clip names ("leopard-9"); GT and SR are flat
    ``<dir>/<clip>.mp4`` files and annotations a ``<dir>/<clip>`` directory.
    A name that already carries an extension resolves identically.
    """
    stem = video.rsplit(".", 1)[0]
    return (f"{gt_dir}/{stem}.mp4", f"{sr_dir}/{stem}.mp4", f"{ann_dir}/{stem}")


def _decode_frames(path):  # pragma: no cover - IO glue, validated at integration
    """Decode a video to a list of RGB uint8 frames (PyAV, imported lazily)."""
    import av

    frames = []
    with av.open(path) as container:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
    return frames


def _build_scorers():  # pragma: no cover - IO glue, validated at integration
    """SIFT (texture) + DINO (semantic), plus LoFTR (geometry) when kornia is
    installed. Mirrors Ring Forcing's trio, degrading to the available axes."""
    import importlib.util

    from .scorers import SiftTextureScorer
    from .scorers_gpu import DinoSemanticScorer

    scorers = [SiftTextureScorer(), DinoSemanticScorer()]
    if importlib.util.find_spec("kornia") is not None:
        from .scorers_gpu import LoftrGeometryScorer
        scorers.append(LoftrGeometryScorer())
    return scorers


def main(argv=None):  # pragma: no cover - thin IO glue, validated at integration
    """CLI: evaluate every clip in a manifest and write per-clip JSON records.

    The GPU adapters (CLIP embedder, DINO/LoFTR/SIFT scorers) are imported
    lazily here so the module stays torch-free at top level.
    """
    import argparse
    import csv
    import json

    from .annotations import load_annotations
    from .embed_clip import ClipEmbedder

    p = argparse.ArgumentParser(description="ADR cross-gap consistency evaluation")
    p.add_argument("--manifest", required=True)
    p.add_argument("--gt-dir", required=True)
    p.add_argument("--sr-dir", required=True)
    p.add_argument("--ann-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--no-lpips", action="store_true")
    p.add_argument("--per-gap", action="store_true",
                   help="emit one matched pair per qualifying gap (more review examples)")
    p.add_argument("--clip-batch", type=int, default=256,
                   help="CLIP embedding batch size (lower to fit a contended shared GPU)")
    args = p.parse_args(argv)

    embedder = ClipEmbedder(batch_size=args.clip_batch)
    scorers = _build_scorers()

    records = []
    with open(args.manifest) as f:
        for row in csv.DictReader(f):
            name = row["video"]
            gt_path, sr_path, ann_path = _clip_paths(
                name, args.gt_dir, args.sr_dir, args.ann_dir)
            gt = _decode_frames(gt_path)
            sr = _decode_frames(sr_path)
            ann = load_annotations(ann_path)
            fps, frames = float(row["fps"]), int(row["frames"])
            if args.per_gap:
                res = evaluate_clip_pairs(gt, sr, ann, fps, frames, embedder, scorers,
                                          compute_lpips=not args.no_lpips)
                pairs = res.get("pairs", [])
                for k, pr in enumerate(pairs):
                    pr["video"], pr["pair_id"] = name, k
                    records.append(pr)
                if not pairs:  # keep quarantined / no-gap clips visible in the JSON
                    records.append({"video": name, **{k: v for k, v in res.items() if k != "pairs"}})
                print(f"{name}: {res['status']} ({len(pairs)} pairs)", flush=True)
            else:
                rec = evaluate_clip(gt, sr, ann, fps, frames, embedder, scorers,
                                    compute_lpips=not args.no_lpips)
                rec["video"] = name
                print(f"{name}: {rec['status']}", flush=True)
                records.append(rec)

    with open(args.out, "w") as f:
        json.dump(records, f, indent=2)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
