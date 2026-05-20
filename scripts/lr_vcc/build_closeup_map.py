"""Compute face/hand bbox p50 per video from cached per-frame anatomy traces.

Used to feed sub-metric I's close-up reliability test.
"""
import argparse
import json
import os
import sys
from pathlib import Path


def _bbox_area(bbox):
    if not bbox or len(bbox) < 4:
        return None
    x1, y1, x2, y2 = bbox[:4]
    return max(0, x2 - x1) * max(0, y2 - y1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_frame_dir", required=True,
                    help="dir with <method>_<video>_per_frame.json")
    ap.add_argument("--method", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--frame_area", type=int, default=1280 * 720)
    args = ap.parse_args()

    out = {}
    pattern = args.method + "_*_per_frame.json"
    for p in sorted(Path(args.per_frame_dir).glob(pattern)):
        # extract video id between method_ and _per_frame.json
        name = p.name
        video_id = name[len(args.method) + 1: -len("_per_frame.json")]
        d = json.load(open(p))
        areas = []
        for fr in d["frame_results"]:
            for person in fr.get("persons", []):
                for cat in ("face", "hand"):
                    for entry in person["scores"].get(cat, []):
                        bb = entry[1] if isinstance(entry, list) and len(entry) > 1 else None
                        a = _bbox_area(bb)
                        if a is not None and a > 0:
                            areas.append(a)
        if not areas:
            out[video_id] = 0.0
            continue
        areas.sort()
        p50 = areas[len(areas) // 2]
        out[video_id] = p50 / args.frame_area
        print(video_id + ": p50 = " + format(out[video_id] * 100, ".1f") + "% of frame")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote " + args.output)


if __name__ == "__main__":
    main()
