"""
Slow-fast aggregation over per-frame Human_Anatomy traces.

Reads a per-frame JSON produced by diagnose_anatomy_per_frame.py and computes the
same slow-fast structure used by the Identity adapter:

    slow  = mean over clips with people of (1 - clip.abnormal / clip.people)
    fast  = on the synthetic "clip-first-frames" video,
            (1 - sum(abnormal) / sum(people)) over the first-frame stream
    fused = w_slow * slow + w_fast * fast   (default 50/50)

Both branches use the per-clip frame split at `2 sec * fps` where `fps` is passed
as an argument (use the LQ-source fps to avoid SR-pipeline fps-tag mismatches).

The whole-video Anatomy score that vbench2 originally produced is preserved as
`whole_video` for comparison — that score is fps-invariant since vbench2's
custom_input mode does not use fps anywhere.

Usage:
    python aggregate_slow_fast_anatomy.py \
        --per-frame results/vbench2_anatomy/diagnostic_KZ8p6b1zJ9U/mgld_KZ8p6b1zJ9U_per_frame.json \
        --fps 29.97 \
        --output results/vbench2_anatomy/anatomy_slow_fast/mgld_KZ8p6b1zJ9U.json
"""
import argparse
import json
import os
import sys
from typing import Any


def aggregate(per_frame: dict, fps: float, clip_duration: float = 2.0,
              w_slow: float = 0.5, w_fast: float = 0.5) -> dict:
    frames = per_frame["frame_results"]
    seg_frames = max(1, int(round(fps * clip_duration)))
    n_clips = len(frames) // seg_frames
    if n_clips == 0:
        return {"n_frames": len(frames), "n_clips": 0, "fps_used": fps,
                "slow": None, "fast": None, "fused": None,
                "whole_video": float(per_frame.get("video_results", -1.0)),
                "clip_detail": []}

    clip_scores = []
    clip_detail = []
    for ci in range(n_clips):
        clip = frames[ci * seg_frames:(ci + 1) * seg_frames]
        people = sum(f["person_count"] for f in clip)
        abnormal = sum(f["abnormal_count"] for f in clip)
        score = (1 - abnormal / people) if people > 0 else None
        clip_detail.append({"clip_index": ci, "people": people,
                            "abnormal": abnormal, "score": score})
        if score is not None:
            clip_scores.append(score)
    slow = sum(clip_scores) / len(clip_scores) if clip_scores else None

    # fast branch: take the first frame of each clip and treat as one stream
    first_frames = [frames[ci * seg_frames] for ci in range(n_clips)]
    fast_people = sum(f["person_count"] for f in first_frames)
    fast_abnormal = sum(f["abnormal_count"] for f in first_frames)
    fast = (1 - fast_abnormal / fast_people) if fast_people > 0 else None

    if slow is not None and fast is not None:
        fused = w_slow * slow + w_fast * fast
    elif slow is not None:
        fused = slow
    elif fast is not None:
        fused = fast
    else:
        fused = None

    return {
        "n_frames": len(frames),
        "n_clips": n_clips,
        "n_clips_with_people": len(clip_scores),
        "seg_frames": seg_frames,
        "fps_used": fps,
        "weights": {"slow": w_slow, "fast": w_fast},
        "slow": slow,
        "fast": fast,
        "fused": fused,
        "whole_video": float(per_frame.get("video_results", -1.0)),
        "clip_detail": clip_detail,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-frame", required=True, help="per-frame anatomy JSON")
    ap.add_argument("--fps", type=float, required=True, help="fps for clip splitting (use LQ source's fps)")
    ap.add_argument("--clip-duration", type=float, default=2.0)
    ap.add_argument("--w-slow", type=float, default=0.5)
    ap.add_argument("--w-fast", type=float, default=0.5)
    ap.add_argument("--output", help="output JSON; if omitted, prints summary to stdout only")
    args = ap.parse_args()

    if not os.path.isfile(args.per_frame):
        sys.exit(f"per-frame file not found: {args.per_frame}")

    with open(args.per_frame) as f:
        per_frame = json.load(f)

    out = aggregate(per_frame, args.fps, args.clip_duration, args.w_slow, args.w_fast)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out, f)

    def fmt(x: Any) -> str:
        return f"{x:.4f}" if isinstance(x, float) else str(x)
    print(f"video             : {per_frame.get('video_path', '?')}")
    print(f"n_frames / n_clips: {out['n_frames']} / {out['n_clips']} (with people: {out['n_clips_with_people']})")
    print(f"fps_used          : {out['fps_used']}  seg_frames={out['seg_frames']}")
    print(f"slow              : {fmt(out['slow'])}")
    print(f"fast              : {fmt(out['fast'])}")
    print(f"fused (50/50)     : {fmt(out['fused'])}")
    print(f"whole_video (orig): {fmt(out['whole_video'])}")


if __name__ == "__main__":
    main()
