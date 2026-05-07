"""
Per-frame Human_Anatomy diagnostic for one video.

Wraps the upstream `compute_abnormality` pipeline so the per-frame `frame_results`
list (which the upstream code computes then drops) is persisted to JSON.

Output schema per video:
    {
        "video_path": <str>,
        "video_results": <float>,             # whole-video score (matches eval_results.json)
        "frame_results": [
            {
                "frame": <int>,
                "person_count": <int>,
                "abnormal_count": <int>,
                "persons": [                  # one entry per detected person
                    {
                        "person_id": <int>,
                        "abnormal": <bool>,
                        "scores": {           # per anomaly detector
                            "human": [[ [p_normal, p_abnormal], [x1,y1,x2,y2] ], ...],
                            "face":  [...],
                            "hand":  [...],
                        }
                    },
                    ...
                ]
            },
            ...
        ]
    }

Usage (on the lab GPU server):
    cd /data/disk2/timur/repos/VBench/VBench-2.0
    export VBENCH2_CACHE_DIR=/data/disk2/timur/cache/vbench2
    export PYTHONPATH="$PWD:/data/disk2/timur/repos/YOLO-World:${PYTHONPATH:-}"
    conda activate vbench
    CUDA_VISIBLE_DEVICES=0 python diagnose_anatomy_per_frame.py \
        --video /data/disk2/timur/results/mgld_synthetic_mp4/KZ8p6b1zJ9U.mp4 \
        --output /data/disk2/timur/results/vbench2_human_test/diagnostic/mgld_KZ8p6b1zJ9U_per_frame.json
"""
import argparse
import json
import os
import sys
from typing import Any

import numpy as np
import torch

import vbench2.hack_registry  # noqa: F401  — registers mmdet/mmyolo modules
from vbench2.utils import init_submodules
from vbench2.third_party.ViTDetector.detect import Detector, Analyzer


def _to_jsonable(obj: Any) -> Any:
    """Convert numpy/torch scalars and arrays to plain Python."""
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="Path to a single .mp4 video")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if not os.path.isfile(args.video):
        sys.exit(f"Video not found: {args.video}")

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

    print(f"[diagnose] running detector on {args.video}", flush=True)
    detections = detector.detect_video(args.video)
    print(f"[diagnose] detections: {len(detections)}", flush=True)

    print(f"[diagnose] running analyzer", flush=True)
    result = analyzer.analyze(video_path=args.video, detection_results=detections)

    payload = {
        "video_path": args.video,
        "video_results": float(result["video_results"]),
        "frame_results": _to_jsonable(result["frame_results"]),
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(payload, f)

    n_frames = len(payload["frame_results"])
    n_with_people = sum(1 for fr in payload["frame_results"] if fr["person_count"] > 0)
    total_people = sum(fr["person_count"] for fr in payload["frame_results"])
    total_abnormal = sum(fr["abnormal_count"] for fr in payload["frame_results"])
    print(
        f"[diagnose] frames={n_frames} with_people={n_with_people} "
        f"total_people={total_people} total_abnormal={total_abnormal} "
        f"score={payload['video_results']:.4f}",
        flush=True,
    )
    print(f"[diagnose] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
