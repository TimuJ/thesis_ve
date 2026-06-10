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
from scripts.synthetic_artefacts.identity_degradation import apply_identity_degradation
from scripts.synthetic_artefacts.identity_drift import apply_identity_drift
from scripts.synthetic_artefacts.background_drift import apply_background_drift, load_packed_masks


BASE_VIDEOS = ["hhszUXL1Cu8", "7WHI2L_FDNg"]
SEVERITIES = [0.02, 0.05, 0.10, 0.20, 0.40]
SRC_DIR = REPO / "results" / "mgld_synthetic_mp4"
OUT_DIR = REPO / "results" / "synthetic_artefacts"
CHUNK_FRAMES = 60  # 2 sec at 30 fps
FLICKER_PERIOD = 15  # 0.5 sec at 30 fps

# Reference face images for identity_drift. Keyed by base-video id. Each base
# video morphs toward a face extracted from a different base video, so the
# target identity is clearly distinct from the source.
REFERENCE_FACES = {
    "hhszUXL1Cu8":  REPO / "results" / "synthetic_artefacts" / "_references" / "ref_face_for_hhsz.png",
    "7WHI2L_FDNg":  REPO / "results" / "synthetic_artefacts" / "_references" / "ref_face_for_7WHI.png",
}
_REF_CACHE = {}

# Reference background images for background_drift. Same per-base pattern.
REFERENCE_BGS = {
    "hhszUXL1Cu8":  REPO / "results" / "synthetic_artefacts" / "_references" / "ref_bg_for_hhsz.png",
    "7WHI2L_FDNg":  REPO / "results" / "synthetic_artefacts" / "_references" / "ref_bg_for_7WHI.png",
}
_BG_CACHE = {}

# Per-base human silhouette masks (.npz) precomputed by
# `precompute_human_masks.py`. Shared across all severity levels of the same
# base video — masks only depend on the source frames.
HUMAN_MASKS = {
    "hhszUXL1Cu8":  REPO / "results" / "synthetic_artefacts" / "_human_masks" / "hhszUXL1Cu8.npz",
    "7WHI2L_FDNg":  REPO / "results" / "synthetic_artefacts" / "_human_masks" / "7WHI2L_FDNg.npz",
}
_MASK_CACHE = {}


def _load_reference_face(base: str):
    if base in _REF_CACHE:
        return _REF_CACHE[base]
    path = REFERENCE_FACES.get(base)
    if path is None or not path.is_file():
        raise FileNotFoundError(
            f"Missing reference face for base={base}. Expected at {path}. "
            f"Generate it first with scripts/synthetic_artefacts/extract_reference_face.py."
        )
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"Failed to read reference face image at {path}")
    _REF_CACHE[base] = img
    return img


def _load_reference_bg(base: str):
    if base in _BG_CACHE:
        return _BG_CACHE[base]
    path = REFERENCE_BGS.get(base)
    if path is None or not path.is_file():
        raise FileNotFoundError(
            f"Missing reference background for base={base}. Expected at {path}. "
            f"Generate it first with scripts/synthetic_artefacts/extract_reference_background.py."
        )
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"Failed to read reference background image at {path}")
    _BG_CACHE[base] = img
    return img


def _load_human_masks(base: str):
    """Return a (n_frames, H, W) bool array of per-frame human silhouette masks
    for the named base video. Cached across artefact / severity combinations."""
    if base in _MASK_CACHE:
        return _MASK_CACHE[base]
    path = HUMAN_MASKS.get(base)
    if path is None or not path.is_file():
        raise FileNotFoundError(
            f"Missing human masks for base={base}. Expected at {path}. "
            f"Generate them with scripts/synthetic_artefacts/precompute_human_masks.py."
        )
    masks = load_packed_masks(str(path))
    _MASK_CACHE[base] = masks
    return masks


def process_one(src_path: Path, out_path: Path, artefact: str, severity: float, base: str = ""):
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
    ref_face = _load_reference_face(base) if artefact == "identity_drift" else None
    ref_bg = _load_reference_bg(base) if artefact == "background_drift" else None
    human_masks = _load_human_masks(base) if artefact == "background_drift" else None
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
        elif artefact == "identity_degradation":
            out = apply_identity_degradation(fr, idx, severity)
        elif artefact == "identity_drift":
            out = apply_identity_drift(fr, idx, n_frames, ref_face, severity)
        elif artefact == "background_drift":
            mask = human_masks[idx] if (human_masks is not None and idx < len(human_masks)) else None
            out = apply_background_drift(fr, idx, n_frames, ref_bg, mask, severity)
        else:
            raise ValueError("unknown artefact: " + artefact)
        writer.write(out)
        idx += 1
    cap.release()
    writer.release()
    print("  wrote " + str(out_path) + " (" + str(idx) + " frames)")


def main():
    for artefact in ["color_drift", "chunk_boundary", "flicker",
                     "identity_degradation", "identity_drift", "background_drift"]:
        for base in BASE_VIDEOS:
            src = SRC_DIR / (base + ".mp4")
            if not src.is_file():
                print("MISSING source: " + str(src))
                continue
            for sev in SEVERITIES:
                out_name = base + "_sev" + format(sev, ".2f").replace(".", "p") + ".mp4"
                out = OUT_DIR / artefact / out_name
                if out.is_file() and out.stat().st_size > 0:
                    print(artefact + " sev=" + str(sev) + " on " + base + ": SKIP (exists)")
                    continue
                print(artefact + " sev=" + str(sev) + " on " + base + ":")
                process_one(src, out, artefact, sev, base=base)


if __name__ == "__main__":
    main()
