"""
Long-video slow-fast adapter for VBench-2.0 Human_Identity.

Slow branch: run VBench-2.0 human_identity on each ~2-second clip, average per video.
Fast branch: concatenate first frame of each clip into a "fast" video,
             run identity tracking on it -> measures cross-clip identity drift.
Fuse: weighted average (default 50/50).

Usage:
    python human_identity_long.py \
        --videos_path /path/to/long_videos_dir \
        --output_path /path/to/out \
        [--w_slow 0.5] [--w_fast 0.5] [--clip_duration 2]
"""
import os
import sys
import json
import argparse
import shutil
from datetime import datetime

import cv2
import torch
import numpy as np
from tqdm import tqdm

# VBench-2.0 path
VBENCH2_PATH = "/data/disk2/timur/repos/VBench/VBench-2.0"
VBENCH_LONG_PATH = "/data/disk2/timur/repos/VBench/vbench2_beta_long"
sys.path.insert(0, VBENCH2_PATH)

import vbench2.hack_registry  # noqa: F401
from vbench2.utils import init_submodules
from vbench2.human_identity import evaluate_id_consistency, IDTracker  # patched version
from retinaface.predict_single import Model as RetinaModel
from torch.utils import model_zoo

# arcface model
from vbench2.third_party.arcface.models import resnet_face18  # noqa: F401


def split_into_clips(video_path, output_dir, duration=2.0):
    """Split a long video into N clips of ~duration seconds. Returns list of clip paths."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    seg_frames = max(1, int(round(fps * duration)))
    n_clips = total // seg_frames
    if n_clips == 0:
        cap.release()
        return []

    os.makedirs(output_dir, exist_ok=True)
    clip_paths = []
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    base = os.path.splitext(os.path.basename(video_path))[0]

    for i in range(n_clips):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * seg_frames)
        out_path = os.path.join(output_dir, f"{base}_{i:04d}.mp4")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        for _ in range(seg_frames):
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
        writer.release()
        clip_paths.append(out_path)

    cap.release()
    return clip_paths


def cat_first_frames(clip_paths, out_path, fps=2.0):
    """Build a 'fast' video by concatenating the first frame of each clip."""
    if not clip_paths:
        return None
    first_frame = cv2.VideoCapture(clip_paths[0])
    ok, f0 = first_frame.read()
    first_frame.release()
    if not ok:
        return None
    h, w = f0.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    for cp in clip_paths:
        cap = cv2.VideoCapture(cp)
        ok, frame = cap.read()
        cap.release()
        if ok:
            writer.write(frame)
    writer.release()
    return out_path


def evaluate_videos_identity(video_paths, retina_model, model):
    """Run patched identity evaluation on a list of videos. Return per-video and overall."""
    prompt_dict = [{
        "prompt_en": os.path.basename(v),
        "dimension": ["human_identity"],
        "video_list": [v],
    } for v in video_paths]
    overall, processed = evaluate_id_consistency(prompt_dict, retina_model, model)
    return overall, processed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos_path", required=True, help="dir containing long videos (mp4)")
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--clip_duration", type=float, default=2.0)
    ap.add_argument("--w_slow", type=float, default=0.5)
    ap.add_argument("--w_fast", type=float, default=0.5)
    ap.add_argument("--keep_clips", action="store_true", help="don't delete temp clips after eval")
    ap.add_argument(
        "--save_clip_detail",
        action="store_true",
        help="persist per-clip and per-fast-frame raw scores under per_video[<v>].clip_detail / .fast_detail",
    )
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_path, exist_ok=True)
    work_dir = os.path.join(args.output_path, "_work")
    os.makedirs(work_dir, exist_ok=True)

    # Init submodules (RetinaFace + ArcFace) -- reuse VBench-2.0 init
    submodules = init_submodules(["Human_Identity"])["Human_Identity"]

    # Load models
    print("Loading RetinaFace...")
    url = "https://github.com/ternaus/retinaface/releases/download/0.01/retinaface_resnet50_2020-07-20-f168fae3c.zip"
    retina_state_dict = model_zoo.load_url(url, progress=True, map_location="cpu")
    retina_model = RetinaModel(max_size=2048, device=device)
    retina_model.load_state_dict(retina_state_dict)

    print("Loading ArcFace...")
    model = resnet_face18(use_se=False)
    state_dict = torch.load(submodules["model"], map_location="cpu")
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.to(device).eval()

    # Find videos
    video_files = sorted(
        f for f in os.listdir(args.videos_path)
        if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
    )

    results = {"per_video": {}, "weights": {"slow": args.w_slow, "fast": args.w_fast}}
    fused_scores = []

    for vname in video_files:
        vpath = os.path.join(args.videos_path, vname)
        base = os.path.splitext(vname)[0]
        print(f"\n=== {vname} ===")

        # Split into clips
        clip_dir = os.path.join(work_dir, "clips", base)
        if not os.path.isdir(clip_dir) or len(os.listdir(clip_dir)) == 0:
            print(f"Splitting into {args.clip_duration}s clips...")
            split_into_clips(vpath, clip_dir, duration=args.clip_duration)
        clip_paths = sorted(
            os.path.join(clip_dir, f) for f in os.listdir(clip_dir) if f.endswith(".mp4")
        )
        print(f"  {len(clip_paths)} clips")
        if not clip_paths:
            results["per_video"][base] = {"slow": -1.0, "fast": -1.0, "fused": -1.0, "n_clips": 0}
            continue

        # Slow branch: average score over clips
        print("Slow branch (per-clip identity)...")
        clip_overall, clip_detail = evaluate_videos_identity(clip_paths, retina_model, model)
        valid = [d["video_results"] for d in clip_detail if d["video_results"] != -1]
        slow_score = float(np.mean(valid)) if valid else -1.0
        print(f"  slow = {slow_score:.4f} ({len(valid)}/{len(clip_paths)} clips with faces)")

        # Fast branch: identity across clip first-frames
        print("Fast branch (cross-clip identity)...")
        fast_video = os.path.join(work_dir, f"{base}_firstframes.mp4")
        cat_first_frames(clip_paths, fast_video, fps=2.0)
        fast_overall, fast_detail = evaluate_videos_identity([fast_video], retina_model, model)
        fast_score = (
            float(fast_detail[0]["video_results"])
            if fast_detail and fast_detail[0]["video_results"] != -1
            else -1.0
        )
        print(f"  fast = {fast_score:.4f}")

        # Fuse
        if slow_score == -1.0 and fast_score == -1.0:
            fused = -1.0
        elif slow_score == -1.0:
            fused = fast_score
        elif fast_score == -1.0:
            fused = slow_score
        else:
            fused = args.w_slow * slow_score + args.w_fast * fast_score
        print(f"  fused = {fused:.4f}")

        results["per_video"][base] = {
            "slow": slow_score,
            "fast": fast_score,
            "fused": fused,
            "n_clips": len(clip_paths),
            "n_clips_with_faces": len(valid),
        }
        if args.save_clip_detail:
            # per-clip slow scores in clip-index order; -1 = no faces in that clip
            results["per_video"][base]["clip_detail"] = [
                {
                    "clip_index": i,
                    "clip_path": os.path.basename(d.get("video_path", clip_paths[i])),
                    "score": float(d["video_results"]),
                }
                for i, d in enumerate(clip_detail)
            ]
            # fast branch: identity over the concat-of-clip-first-frames synthetic video
            results["per_video"][base]["fast_detail"] = (
                {"score": float(fast_detail[0]["video_results"])} if fast_detail else None
            )
        if fused != -1.0:
            fused_scores.append(fused)

    results["overall_fused"] = float(np.mean(fused_scores)) if fused_scores else -1.0
    results["overall_slow"] = float(
        np.mean([v["slow"] for v in results["per_video"].values() if v["slow"] != -1.0]) or -1
    ) if any(v["slow"] != -1.0 for v in results["per_video"].values()) else -1.0
    results["overall_fast"] = float(
        np.mean([v["fast"] for v in results["per_video"].values() if v["fast"] != -1.0]) or -1
    ) if any(v["fast"] != -1.0 for v in results["per_video"].values()) else -1.0

    if not args.keep_clips:
        shutil.rmtree(work_dir, ignore_errors=True)

    timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    out_file = os.path.join(args.output_path, f"results_{timestamp}_eval_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_file}")
    print(f"Overall fused: {results['overall_fused']:.4f}")
    print(f"Overall slow:  {results['overall_slow']:.4f}")
    print(f"Overall fast:  {results['overall_fast']:.4f}")


if __name__ == "__main__":
    main()
