"""Extract a reference face crop from a source video, for use by identity_drift.

Usage:
    python scripts/synthetic_artefacts/extract_reference_face.py \
        --video results/mgld_synthetic_mp4/mJog8DlRk_4.mp4 \
        --output reference_face_mJog.png \
        --frame 100

Picks the largest detected face in the requested frame and saves it as a
PNG. The output is intended to be passed to `apply_identity_drift` as the
morph target.
"""
import argparse
from pathlib import Path

import cv2


def extract_face(video_path: str, frame_index: int, output_path: str) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Failed to read frame {frame_index} from {video_path}")

    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60)
    )
    if len(faces) == 0:
        raise RuntimeError(
            f"No face detected in frame {frame_index} of {video_path}. "
            "Try a different frame index."
        )
    # Pick the largest face by bbox area.
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    crop = frame[y:y + h, x:x + w]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, crop)
    print(f"Saved {output_path}: {w}x{h} face from frame {frame_index} of {video_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="source video path")
    ap.add_argument("--output", required=True, help="output PNG path for the face crop")
    ap.add_argument("--frame", type=int, default=100, help="frame index to sample (default 100)")
    args = ap.parse_args()
    extract_face(args.video, args.frame, args.output)


if __name__ == "__main__":
    main()
