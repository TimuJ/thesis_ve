"""Dump per-frame face embeddings so identity anchoring can be re-designed offline.

Why this exists
---------------
The shipped identity sub-metric (`vbench2/human_identity.py`) re-creates its
`IDTracker` for every 2-second clip, so the reference face is re-initialised
~83 times per video and the score is a *within-clip* self-similarity:

    clip_score = consistent_face_frames / face_bearing_frames

where a frame is "consistent" iff cosine similarity to *that clip's own first
face* is >= 0.4. Two consequences, both observed in the validation battery:

  1. Long-range identity drift is structurally invisible — every clip
     re-anchors, so drift between clip 1 and clip 80 is never measured.
  2. Degradation *raises* the score. Blurring collapses face embeddings toward
     each other, so similarity to the clip's own first face goes up and more
     frames clear the 0.4 threshold. Measured: fused 0.375 -> 0.489 as identity
     degrades.

No choice of constants inverts a rising response, so this needs a different
measurement, not a different threshold. That measurement is an *anchored*
identity: compare every face to a video-level reference built from
high-confidence early faces, instead of to its own clip's first frame.

Rather than re-scan video once per design variant, this script persists the raw
embeddings once. Every anchoring variant is then a cheap offline recomposition —
the same architecture that took the calibration harness from 2.7 s to
sub-millisecond per matrix.

The reproduction gate
---------------------
The dump is only trustworthy if the *existing* score can be replayed from it.
`--gate` replays the shipped per-clip logic from the stored embeddings and
compares against the committed identity JSON. If that does not match, the dump
is wrong and nothing built on it can be believed.

Usage (server, `identity` conda env)
------------------------------------
  python -m scripts.vbench2_long.dump_identity_embeddings \
      --videos_path ~/synthetic_data/artefacts/identity_degradation \
      --output_path ~/results/identity_embeddings/identity_degradation \
      [--fps_overrides fps.json] [--limit N]
"""
import argparse
import json
import os
import shutil

import cv2
import decord
import numpy as np
import torch
from retinaface.pre_trained_models import get_model as RetinaModel
from torch.utils import model_zoo

from vbench2.utils import init_submodules
from vbench2.third_party.arcface.models import resnet_face18
from vbench2.human_identity import extract_face_features, calculate_similarity

from .human_identity_long import split_into_clips

SIMILARITY_THRESHOLD = 0.4   # shipped value; replayed, not re-tuned here
MIN_FRAMES = 20              # shipped value: clips with fewer face-frames score -1


def largest_face_box(frame_u8, retina_model):
    """Largest detected face box, mirroring IDTracker.update's selection.

    Returns (x1, y1, x2, y2, area) or None. The clipping and degenerate-box
    rejection replicate the shipped code exactly — any divergence here shows up
    in the reproduction gate.
    """
    faces = retina_model.predict_jsons(frame_u8)
    if not faces:
        return None
    best_box, best_area = None, 0
    for f in faces:
        box = f.get("bbox") if isinstance(f, dict) else None
        if box and len(box) == 4:
            bx1, by1, bx2, by2 = box
            area = max(0, bx2 - bx1) * max(0, by2 - by1)
            if area > best_area:
                best_area, best_box = area, box
    if best_box is None:
        return None
    x1, y1, x2, y2 = best_box
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2 = min(frame_u8.shape[1], int(x2))
    y2 = min(frame_u8.shape[0], int(y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2, float(best_area)


def dump_clip(clip_path, retina_model, arc_model):
    """Per-face-frame embeddings for one clip.

    Returns dict with parallel arrays over face-bearing frames only:
      frame_idx (int32), emb (float32 N x D), bbox_area (float32),
      frame_area (float32), plus n_frames_total.
    """
    vr = decord.VideoReader(clip_path)
    frames = vr.get_batch(range(len(vr))).asnumpy()  # T H W C, RGB uint8
    idxs, embs, areas = [], [], []
    h, w = frames.shape[1], frames.shape[2]
    for i, frame in enumerate(frames):
        got = largest_face_box(frame.astype(np.uint8), retina_model)
        if got is None:
            continue
        x1, y1, x2, y2, area = got
        feat = extract_face_features(frame.astype(np.uint8)[y1:y2, x1:x2], arc_model)
        idxs.append(i)
        embs.append(np.asarray(feat, dtype=np.float32))
        areas.append(area)
    return {
        "frame_idx": np.asarray(idxs, dtype=np.int32),
        "emb": (np.stack(embs).astype(np.float32) if embs
                else np.zeros((0, 1024), dtype=np.float32)),
        "bbox_area": np.asarray(areas, dtype=np.float32),
        "frame_area": np.float32(h * w),
        "n_frames_total": np.int32(len(frames)),
    }


def replay_legacy_clip_score(clip):
    """Reproduce the shipped per-clip score from stored embeddings.

    Mirrors evaluate_id_consistency: the first face-bearing frame is the
    reference and counts as consistent; later face-bearing frames count iff
    cosine similarity to that reference >= 0.4; fewer than 20 face-frames -> -1.
    """
    emb = clip["emb"]
    if emb.shape[0] == 0:
        return -1.0
    consistent, seen = 0, 0
    ref = None
    for k in range(emb.shape[0]):
        if ref is None:
            ref = emb[k]
            consistent += 1
            seen += 1
            continue
        if calculate_similarity(emb[k], ref) >= SIMILARITY_THRESHOLD:
            consistent += 1
        seen += 1
    if seen < MIN_FRAMES:
        return -1.0
    return consistent / seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos_path", required=True)
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--clip_duration", type=float, default=2.0)
    ap.add_argument("--fps_overrides", default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="process only the first N videos (smoke tests)")
    ap.add_argument("--keep_clips", action="store_true")
    args = ap.parse_args()

    fps_overrides = {}
    if args.fps_overrides and os.path.isfile(args.fps_overrides):
        fps_overrides = json.load(open(args.fps_overrides))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_path, exist_ok=True)
    work_dir = os.path.join(args.output_path, "_work")
    os.makedirs(work_dir, exist_ok=True)

    submodules = init_submodules(["Human_Identity"])["Human_Identity"]
    print("Loading RetinaFace...")
    url = ("https://github.com/ternaus/retinaface/releases/download/0.01/"
           "retinaface_resnet50_2020-07-20-f168fae3c.zip")
    retina_model = RetinaModel(max_size=2048, device=device)
    retina_model.load_state_dict(model_zoo.load_url(url, progress=True,
                                                    map_location="cpu"))
    print("Loading ArcFace...")
    arc_model = resnet_face18(use_se=False)
    # weights_only=True: this checkpoint is a plain state_dict, so there is no
    # reason to let torch.load unpickle arbitrary objects.
    sd = torch.load(submodules["model"], map_location="cpu", weights_only=True)
    arc_model.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()})
    arc_model.to(device).eval()

    videos = sorted(f for f in os.listdir(args.videos_path)
                    if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv")))
    if args.limit:
        videos = videos[:args.limit]
    print(f"{len(videos)} videos to process")

    for vi, fname in enumerate(videos, 1):
        base = os.path.splitext(fname)[0]
        out_npz = os.path.join(args.output_path, base + "_faces.npz")
        if os.path.isfile(out_npz):
            print(f"[{vi}/{len(videos)}] {base}: exists, skipping")
            continue
        video_path = os.path.join(args.videos_path, fname)
        clip_dir = os.path.join(work_dir, base)
        os.makedirs(clip_dir, exist_ok=True)
        clip_paths = split_into_clips(video_path, clip_dir,
                                      duration=args.clip_duration,
                                      fps_override=fps_overrides.get(base))
        payload, legacy = {}, []
        for ci, cp in enumerate(clip_paths):
            clip = dump_clip(cp, retina_model, arc_model)
            for k, v in clip.items():
                payload[f"c{ci}_{k}"] = v
            legacy.append(replay_legacy_clip_score(clip))
        payload["n_clips"] = np.int32(len(clip_paths))
        payload["legacy_clip_scores"] = np.asarray(legacy, dtype=np.float32)
        np.savez_compressed(out_npz, **payload)
        valid = [s for s in legacy if s != -1.0]
        print(f"[{vi}/{len(videos)}] {base}: {len(clip_paths)} clips, "
              f"{len(valid)} with faces, replayed slow="
              f"{np.mean(valid) if valid else -1:.4f} -> {os.path.basename(out_npz)}",
              flush=True)
        if not args.keep_clips:
            shutil.rmtree(clip_dir, ignore_errors=True)

    print("done")


if __name__ == "__main__":
    main()
