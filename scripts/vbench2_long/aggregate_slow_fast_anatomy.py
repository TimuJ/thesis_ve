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


_DETECTOR_THRESHOLDS = {"human": 0.4545, "face": 0.3030, "hand": 0.3232}


def _frame_continuous_p_abnormal(fr: dict) -> dict:
    """Per-frame mean p_abnormal per detector category, plus total people.

    Continuous score for a frame is `1 - mean(p_abnormal)` averaged across
    detector categories that fired. Returns the raw aggregates so the
    clip-level aggregation can pool over (sum of probabilities) /
    (count of detections) instead of (mean of frame means).
    """
    sum_pabn = {"human": 0.0, "face": 0.0, "hand": 0.0}
    n_det = {"human": 0, "face": 0, "hand": 0}
    for p in fr.get("persons", []):
        for cat, entries in p.get("scores", {}).items():
            if cat not in _DETECTOR_THRESHOLDS:
                continue
            for entry in entries:
                sc = entry[0] if isinstance(entry, list) else entry
                if isinstance(sc, list) and len(sc) >= 2:
                    sum_pabn[cat] += float(sc[0])
                    n_det[cat] += 1
    return {"sum_pabn": sum_pabn, "n_det": n_det,
            "person_count": fr.get("person_count", 0),
            "abnormal_count": fr.get("abnormal_count", 0)}


def aggregate(per_frame: dict, fps: float, clip_duration: float = 2.0,
              w_slow: float = 0.5, w_fast: float = 0.5,
              continuous: bool = False) -> dict:
    """Slow-fast aggregation over a per-frame Anatomy trace.

    continuous=False (default) : upstream VBench-2.0 threshold-based formula —
                                 per-frame score = 1 - abnormal_count / person_count.
                                 Matches the paper for reproducibility.
    continuous=True            : our tweak — per-frame score = 1 - mean(p_abnormal)
                                 averaged across detector categories that fired.
                                 More robust to threshold-near-boundary discontinuities
                                 (halves the KZ8p6b1zJ9U flip gap, see
                                 docs/notes/2026-05-13-kz-regime-shift-trigger.md).
    """
    frames = per_frame["frame_results"]
    seg_frames = max(1, int(round(fps * clip_duration)))
    n_clips = len(frames) // seg_frames
    if n_clips == 0:
        return {"n_frames": len(frames), "n_clips": 0, "fps_used": fps,
                "scoring": "continuous" if continuous else "threshold",
                "slow": None, "fast": None, "fused": None,
                "whole_video": float(per_frame.get("video_results", -1.0)),
                "clip_detail": []}

    clip_scores = []
    clip_detail = []
    for ci in range(n_clips):
        clip = frames[ci * seg_frames:(ci + 1) * seg_frames]
        if continuous:
            sum_pabn = {"human": 0.0, "face": 0.0, "hand": 0.0}
            n_det = {"human": 0, "face": 0, "hand": 0}
            for fr in clip:
                ag = _frame_continuous_p_abnormal(fr)
                for c in sum_pabn:
                    sum_pabn[c] += ag["sum_pabn"][c]
                    n_det[c] += ag["n_det"][c]
            per_cat = [sum_pabn[c] / n_det[c] for c in sum_pabn if n_det[c] > 0]
            mean_pabn = sum(per_cat) / len(per_cat) if per_cat else None
            score = (1.0 - mean_pabn) if mean_pabn is not None else None
            clip_detail.append({"clip_index": ci,
                                "n_det": sum(n_det.values()),
                                "mean_pabn": mean_pabn,
                                "score": score})
        else:
            people = sum(f["person_count"] for f in clip)
            abnormal = sum(f["abnormal_count"] for f in clip)
            score = (1 - abnormal / people) if people > 0 else None
            clip_detail.append({"clip_index": ci, "people": people,
                                "abnormal": abnormal, "score": score})
        if score is not None:
            clip_scores.append(score)
    slow = sum(clip_scores) / len(clip_scores) if clip_scores else None

    # fast branch: first frame of each clip, treated as one stream
    first_frames = [frames[ci * seg_frames] for ci in range(n_clips)]
    if continuous:
        sum_pabn = {"human": 0.0, "face": 0.0, "hand": 0.0}
        n_det = {"human": 0, "face": 0, "hand": 0}
        for fr in first_frames:
            ag = _frame_continuous_p_abnormal(fr)
            for c in sum_pabn:
                sum_pabn[c] += ag["sum_pabn"][c]
                n_det[c] += ag["n_det"][c]
        per_cat = [sum_pabn[c] / n_det[c] for c in sum_pabn if n_det[c] > 0]
        fast = (1.0 - sum(per_cat) / len(per_cat)) if per_cat else None
    else:
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
        "scoring": "continuous" if continuous else "threshold",
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
    ap.add_argument("--continuous", action="store_true",
                    help="our tweak: use 1 - mean(p_abnormal) instead of upstream's "
                         "1 - fraction_above_threshold. More robust to threshold-near-"
                         "boundary content. Default is upstream's threshold formula.")
    ap.add_argument("--output", help="output JSON; if omitted, prints summary to stdout only")
    args = ap.parse_args()

    if not os.path.isfile(args.per_frame):
        sys.exit(f"per-frame file not found: {args.per_frame}")

    with open(args.per_frame) as f:
        per_frame = json.load(f)

    out = aggregate(per_frame, args.fps, args.clip_duration, args.w_slow, args.w_fast,
                    continuous=args.continuous)

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
