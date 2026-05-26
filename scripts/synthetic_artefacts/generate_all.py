"""Generate all synthetic test videos: 2 base videos x 2 artefacts x 5 severities = 20 videos.

Reads source videos from results/mgld_synthetic_mp4/, writes to
results/synthetic_artefacts/<artefact>/<base>_sev<S>.mp4.

Usage:
    python scripts/synthetic_artefacts/generate_all.py
"""
import os
import sys
from pathlib import Path

import cv2

# add repo root to path so we can import the artefact modules
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.synthetic_artefacts.color_drift import apply_color_drift
from scripts.synthetic_artefacts.chunk_boundary import apply_chunk_boundary_jumps
from scripts.synthetic_artefacts.flicker import apply_periodic_flicker


BASE_VIDEOS = ["hhszUXL1Cu8", "7WHI2L_FDNg"]
SEVERITIES = [0.02, 0.05, 0.10, 0.20, 0.40]
SRC_DIR = REPO / "results" / "mgld_synthetic_mp4"
OUT_DIR = REPO / "results" / "synthetic_artefacts"
CHUNK_FRAMES = 60  # 2 sec at 30 fps
FLICKER_PERIOD = 15  # 0.5 sec at 30 fps


def process_one(src_path: Path, out_path: Path, artefact: str, severity: float):
    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        print("  FAILED to open " + str(src_path))
        return
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    idx = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if artefact == "color_drift":
            out = apply_color_drift(fr, idx, n_frames, severity)
        elif artefact == "chunk_boundary":
            out = apply_chunk_boundary_jumps(fr, idx, CHUNK_FRAMES, severity)
        elif artefact == "flicker":
            out = apply_periodic_flicker(fr, idx, FLICKER_PERIOD, severity)
        else:
            raise ValueError("unknown artefact: " + artefact)
        writer.write(out)
        idx += 1
    cap.release()
    writer.release()
    print("  wrote " + str(out_path) + " (" + str(idx) + " frames)")


def main():
    for artefact in ["color_drift", "chunk_boundary", "flicker"]:
        for base in BASE_VIDEOS:
            src = SRC_DIR / (base + ".mp4")
            if not src.is_file():
                print("MISSING source: " + str(src))
                continue
            for sev in SEVERITIES:
                out_name = base + "_sev" + format(sev, ".2f").replace(".", "p") + ".mp4"
                out = OUT_DIR / artefact / out_name
                print(artefact + " sev=" + str(sev) + " on " + base + ":")
                process_one(src, out, artefact, sev)


if __name__ == "__main__":
    main()
