"""
Long-video slow-fast wrapper for VBench-2.0 Human_Anatomy.

Single-command end-to-end pipeline: for each video in --videos_path, run the
ViTDetector ensemble per frame, then aggregate into slow-fast at the LQ source's
fps. Output shape mirrors human_identity_long.py — one JSON with per_video and
overall {slow, fast, fused}.

Combines diagnose_anatomy_per_frame.py (per-frame trace) and
aggregate_slow_fast_anatomy.py (slow-fast aggregation) into one CLI so Anatomy
has the same default experience as Identity.

Usage:
    cd /data/disk2/timur/repos/VBench/VBench-2.0
    export VBENCH2_CACHE_DIR=/data/disk2/timur/cache/vbench2
    export PYTHONPATH="$PWD:/data/disk2/timur/repos/YOLO-World:${PYTHONPATH:-}"
    conda activate vbench
    CUDA_VISIBLE_DEVICES=0 python human_anatomy_long.py \
        --videos_path /path/to/mp4_dir \
        --output_path /path/to/out \
        [--clip_duration 2.0] [--w_slow 0.5] [--w_fast 0.5] \
        [--fps_overrides fps_map.json] [--save_per_frame] [--save_clip_detail]
"""
import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any

import numpy as np
import torch  # noqa: F401

import vbench2.hack_registry  # noqa: F401
from vbench2.utils import init_submodules
from vbench2.third_party.ViTDetector.detect import Detector, Analyzer

# slow-fast aggregator (same package)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from aggregate_slow_fast_anatomy import aggregate as aggregate_slow_fast  # noqa: E402


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--videos_path", required=True, help="dir containing long videos (mp4)")
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--clip_duration", type=float, default=2.0)
    ap.add_argument("--w_slow", type=float, default=0.5)
    ap.add_argument("--w_fast", type=float, default=0.5)
    ap.add_argument(
        "--fps_overrides",
        type=str,
        default=None,
        help="JSON {basename: fps} to override cv2.CAP_PROP_FPS (use LQ-source fps "
             "to fix SR-pipeline fps-tag mismatches)",
    )
    ap.add_argument(
        "--save_per_frame",
        action="store_true",
        help="also persist the per-frame trace under <output>/<basename>_per_frame.json",
    )
    ap.add_argument(
        "--save_clip_detail",
        action="store_true",
        help="persist per-clip aggregator output under per_video[v].clip_detail",
    )
    ap.add_argument(
        "--continuous",
        action="store_true",
        help="our tweak: use 1 - mean(p_abnormal) instead of upstream's "
             "1 - fraction_above_threshold. Default is upstream's threshold formula.",
    )
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    fps_overrides = {}
    if args.fps_overrides:
        with open(args.fps_overrides) as f:
            fps_overrides = json.load(f)
        print(f"Loaded fps overrides for {len(fps_overrides)} videos: {fps_overrides}")

    os.makedirs(args.output_path, exist_ok=True)

    video_files = sorted(
        f for f in os.listdir(args.videos_path)
        if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
    )
    if not video_files:
        sys.exit(f"No videos found in {args.videos_path}")

    print("Initializing ViTDetector ensemble...")
    submodules = init_submodules(["Human_Anatomy"], local=True)["Human_Anatomy"]
    detector = Detector(
        config_file=submodules["detector_config"],
        weight_file=submodules["detector_weights"],
        device=args.device,
    )
    analyzer = Analyzer(
        model_configs=submodules["analyzer_configs"],
        device=args.device,
        batch_size=submodules["batch_size"],
        class_thresholds={k: v["threshold"] for k, v in submodules["analyzer_configs"].items()},
    )

    results = {
        "per_video": {},
        "weights": {"slow": args.w_slow, "fast": args.w_fast},
        "clip_duration": args.clip_duration,
    }

    for vname in video_files:
        vpath = os.path.join(args.videos_path, vname)
        base = os.path.splitext(vname)[0]
        print(f"\n=== {vname} ===")

        print("Detector pass...")
        detections = detector.detect_video(vpath)
        print(f"  detections: {len(detections)}")

        print("Analyzer pass...")
        result = analyzer.analyze(video_path=vpath, detection_results=detections)
        per_frame = {
            "video_path": vpath,
            "video_results": float(result["video_results"]),
            "frame_results": _to_jsonable(result["frame_results"]),
        }
        if args.save_per_frame:
            pf_path = os.path.join(args.output_path, f"{base}_per_frame.json")
            with open(pf_path, "w") as f:
                json.dump(per_frame, f)
            print(f"  wrote {pf_path}")

        fps = fps_overrides.get(base)
        if fps is None:
            # fall back to source mp4 fps
            import cv2
            cap = cv2.VideoCapture(vpath)
            fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
            cap.release()
            print(f"  fps (from source): {fps}")
        else:
            print(f"  fps (override): {fps}")

        sf = aggregate_slow_fast(per_frame, fps, args.clip_duration, args.w_slow, args.w_fast,
                                 continuous=args.continuous)
        per_video = {
            "slow": sf["slow"],
            "fast": sf["fast"],
            "fused": sf["fused"],
            "whole_video": sf["whole_video"],
            "n_clips": sf["n_clips"],
            "n_clips_with_people": sf["n_clips_with_people"],
            "fps_used": sf["fps_used"],
        }
        if args.save_clip_detail:
            per_video["clip_detail"] = sf["clip_detail"]
        results["per_video"][base] = per_video
        print(f"  slow={sf['slow']:.4f}  fast={sf['fast']:.4f}  "
              f"fused={sf['fused']:.4f}  whole={sf['whole_video']:.4f}")

    # overall means (skip None values)
    def _mean(key: str) -> float:
        vals = [v[key] for v in results["per_video"].values() if v.get(key) is not None]
        return float(np.mean(vals)) if vals else -1.0
    results["overall_slow"] = _mean("slow")
    results["overall_fast"] = _mean("fast")
    results["overall_fused"] = _mean("fused")
    results["overall_whole_video"] = _mean("whole_video")

    timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    out_file = os.path.join(args.output_path, f"results_{timestamp}_eval_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_file}")
    print(f"Overall slow-fast fused: {results['overall_fused']:.4f}")
    print(f"Overall whole-video    : {results['overall_whole_video']:.4f}")


if __name__ == "__main__":
    main()
