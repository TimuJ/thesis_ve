"""Pre-compute per-frame human silhouette masks for a source video.

Uses a Detectron2 Mask R-CNN (COCO-pretrained) and keeps only person-class
instance masks. Per-frame masks are saved as a packed boolean array in an
.npz file for efficient downstream use by `background_drift`.

Usage:
    python scripts/synthetic_artefacts/precompute_human_masks.py \
        --video results/mgld_synthetic_mp4/hhszUXL1Cu8.mp4 \
        --output results/synthetic_artefacts/_human_masks/hhszUXL1Cu8.npz

This script needs a Python environment with detectron2 + torch + opencv. On
the lab server the `vbench` conda env satisfies this.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def _build_predictor(device: str = "cuda"):
    """Lazy-build a Detectron2 Mask R-CNN predictor."""
    from detectron2 import model_zoo
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor

    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    )
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
    cfg.MODEL.DEVICE = device
    return DefaultPredictor(cfg)


def _frame_to_human_mask(predictor, frame_bgr: np.ndarray) -> np.ndarray:
    """Return (H, W) bool array; True where the union of detected persons is."""
    outputs = predictor(frame_bgr)
    instances = outputs["instances"]
    is_person = instances.pred_classes == 0  # COCO person class id is 0
    if is_person.sum() == 0:
        return np.zeros(frame_bgr.shape[:2], dtype=bool)
    person_masks = instances.pred_masks[is_person].cpu().numpy()
    return person_masks.any(axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="source video path")
    ap.add_argument("--output", required=True, help="output .npz path")
    ap.add_argument("--device", default="cuda", help="cuda or cpu")
    ap.add_argument("--max_frames", type=int, default=-1,
                    help="optional cap on number of frames (default: all)")
    args = ap.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"Failed to open {args.video}")
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.max_frames > 0:
        n_frames = min(n_frames, args.max_frames)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(f"Video {args.video}: {n_frames} frames, {w}x{h}")

    predictor = _build_predictor(device=args.device)

    masks = np.zeros((n_frames, h, w), dtype=bool)
    for idx in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            n_frames = idx
            masks = masks[:n_frames]
            break
        masks[idx] = _frame_to_human_mask(predictor, frame)
        if (idx + 1) % 100 == 0 or idx == n_frames - 1:
            pct = (idx + 1) / n_frames * 100
            print(f"  {idx + 1}/{n_frames} frames ({pct:.0f}%) — last mask covered "
                  f"{masks[idx].mean() * 100:.1f}% of frame area")
    cap.release()

    # Pack-bits to keep the file size manageable: a 5000-frame 1280x720 boolean
    # array is 4.6 GB raw; packbits gets it to ~575 MB. np.packbits stores
    # along the last axis by default, so reshape to (n_frames, h*w) first.
    flat = masks.reshape(n_frames, -1)
    packed = np.packbits(flat, axis=-1)

    np.savez_compressed(args.output, masks_packed=packed, shape=masks.shape)
    coverage = masks.mean() * 100
    print(f"Wrote {args.output} — packed shape {packed.shape}, "
          f"original {masks.shape}, mean human coverage {coverage:.1f}%")


if __name__ == "__main__":
    main()
