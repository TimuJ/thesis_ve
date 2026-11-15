"""Run the Qwen3-VL consistency judge on every final pair and add its scores.

For pair (i, j): ref = GT crop at i (clean, pre-gap), gt = GT crop at j (ground
truth), recon = SR crop at j (FlashVSR reconstruction to judge). The judge returns
probability-weighted 1-5 Texture/Geometry/Semantic, added as record["mllm"].
Decodes only the frames each pair needs.

    python judge_pairs.py <all_adr.json> <gt_dir> <sr_dir> <ann_dir> <out.json> [model_path]
"""
import json
import sys
from collections import defaultdict

import cv2
from PIL import Image

from scripts.lr_vcc.adr.annotations import load_annotations
from scripts.lr_vcc.adr.crops import bbox_crop
from scripts.lr_vcc.adr.run_adr import _clip_paths
from scripts.lr_vcc.mllm_judge.judge import DEFAULT_MODEL, Qwen3VLJudge


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
    all_adr, gt_dir, sr_dir, ann_dir, out_path = sys.argv[1:6]
    model_path = sys.argv[6] if len(sys.argv) > 6 else DEFAULT_MODEL
    recs = json.load(open(all_adr))
    judge = Qwen3VLJudge(model_path=model_path)
    print(f"loaded judge: {model_path}", flush=True)

    by_video = defaultdict(list)
    for idx, r in enumerate(recs):
        by_video[r["video"]].append(idx)

    done = 0
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
            bi, bj = valid_box(ann.boxes, i), valid_box(ann.boxes, j)
            try:
                ref = Image.fromarray(bbox_crop(gt[i], bi))
                gtc = Image.fromarray(bbox_crop(gt[j], bj))
                recon = Image.fromarray(bbox_crop(sr_grid(sr[j], gt_hw), bj))
                r["mllm"] = judge.score(ref, gtc, recon).as_dict()
            except Exception as e:
                r["mllm"] = {"error": repr(e)}
            done += 1
        print(f"{video}: judged {len(idxs)} pairs ({done}/{len(recs)})", flush=True)

    json.dump(recs, open(out_path, "w"), indent=2)
    print(f"JUDGED={done}")


if __name__ == "__main__":
    main()
