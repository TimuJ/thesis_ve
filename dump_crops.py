"""Dump the exact (ref, gt, recon) crops judge_pairs.py builds, as PNGs, so the
Claude-API judge can score the identical stimuli the open Qwen judge sees.

ref = GT crop at i (pre-gap), gt = GT crop at j, recon = SR crop at j. Mirrors
judge_pairs.py crop assembly exactly.

    python dump_crops.py <all_adr.json> <gt_dir> <sr_dir> <ann_dir> <out_dir>
"""
import json
import os
import sys
from collections import defaultdict

import cv2
from PIL import Image

from scripts.lr_vcc.adr.annotations import load_annotations
from scripts.lr_vcc.adr.crops import bbox_crop


def _clip_paths(video, gt_dir, sr_dir, ann_dir):
    """Inlined from run_adr to avoid its consistency->skimage import chain."""
    stem = video.rsplit(".", 1)[0]
    return (f"{gt_dir}/{stem}.mp4", f"{sr_dir}/{stem}.mp4", f"{ann_dir}/{stem}")


def valid_box(boxes, f):
    n = len(boxes)
    for d in range(21):
        for g in ({f} if d == 0 else {f - d, f + d}):
            if 0 <= g < n and boxes[g][2] > 0 and boxes[g][3] > 0:
                return boxes[g]
    return None


def decode_select(path, needed):
    import av

    want, hi, out = set(needed), max(needed), {}
    with av.open(path) as c:
        for k, fr in enumerate(c.decode(video=0)):
            if k in want:
                out[k] = fr.to_ndarray(format="rgb24")
            if k >= hi:
                break
    return out


def sr_grid(sr_frame, gt_hw):
    gh, gw = gt_hw
    if sr_frame.shape[:2] == (gh, gw):
        return sr_frame
    return cv2.resize(sr_frame, (gw, gh), interpolation=cv2.INTER_AREA)


def main():
    all_adr, gt_dir, sr_dir, ann_dir, out_dir = sys.argv[1:6]
    os.makedirs(out_dir, exist_ok=True)
    recs = json.load(open(all_adr))

    by_video = defaultdict(list)
    for idx, r in enumerate(recs):
        by_video[r["video"]].append(idx)

    manifest = []
    for video, idxs in by_video.items():
        gt_path, sr_path, ann_path = _clip_paths(video, gt_dir, sr_dir, ann_dir)
        ann = load_annotations(ann_path)
        needed = sorted({recs[k]["i"] for k in idxs} | {recs[k]["j"] for k in idxs})
        gt = decode_select(gt_path, needed)
        sr = decode_select(sr_path, needed)
        gt_hw = next(iter(gt.values())).shape[:2]
        for idx in idxs:
            r = recs[idx]
            i, j = r["i"], r["j"]
            stem = f"{video.rsplit('.', 1)[0]}__{i}__{j}"
            rec = {"video": video, "i": i, "j": j, "stem": stem}
            try:
                bi, bj = valid_box(ann.boxes, i), valid_box(ann.boxes, j)
                Image.fromarray(bbox_crop(gt[i], bi)).save(os.path.join(out_dir, stem + "_ref.png"))
                Image.fromarray(bbox_crop(gt[j], bj)).save(os.path.join(out_dir, stem + "_gt.png"))
                Image.fromarray(bbox_crop(sr_grid(sr[j], gt_hw), bj)).save(os.path.join(out_dir, stem + "_recon.png"))
            except Exception as e:
                rec["error"] = repr(e)
            manifest.append(rec)
        ok = sum(1 for m in manifest if "error" not in m)
        print(f"{video}: {len(idxs)} pairs ({ok}/{len(manifest)} ok so far)", flush=True)

    json.dump(manifest, open(os.path.join(out_dir, "crops_manifest.json"), "w"), indent=2)
    ok = sum(1 for m in manifest if "error" not in m)
    print(f"DUMPED_OK={ok}/{len(manifest)}")


if __name__ == "__main__":
    main()
