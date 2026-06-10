"""Extract a reference background image from a source video, for use by
background_drift.

Usage:
    python scripts/synthetic_artefacts/extract_reference_background.py \
        --video results/mgld_synthetic_mp4/mJog8DlRk_4.mp4 \
        --output reference_bg_mJog.png \
        --frame 500

Saves the full frame at the requested index. The face area (if any) is
included; at composition time, the background_drift artefact will preserve
the source video's face region and discard the reference's, so the reference
frame's face is not a concern.
"""
import argparse
from pathlib import Path

import cv2


def extract_background(video_path: str, frame_index: int, output_path: str) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Failed to read frame {frame_index} from {video_path}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, frame)
    h, w = frame.shape[:2]
    print(f"Saved {output_path}: {w}x{h} from frame {frame_index} of {video_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="source video path")
    ap.add_argument("--output", required=True, help="output PNG path")
    ap.add_argument("--frame", type=int, default=100, help="frame index to sample (default 100)")
    args = ap.parse_args()
    extract_background(args.video, args.frame, args.output)


if __name__ == "__main__":
    main()
