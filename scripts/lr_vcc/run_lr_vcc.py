"""CLI runner — LR-VCC composite over a method's videos.

Inputs per video:
  - CLIP-IQA JSON (from compute_clip_iqa.py)
  - tOF JSON (from scripts/long_range_temporal/eval_tof_tlp.py)
  - Identity JSON (from scripts/vbench2_long/human_identity_long.py) — one file
    per method holding per_video[<v>].
Optional input: a closeup-bbox-p50 map per video (from anatomy per-frame trace).

Output: one per-video JSON per method + one aggregate JSON per method.
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path
from statistics import mean

from .appearance import appearance_score
from .temporal import temporal_score
from .identity import identity_score
from .composite import compose_score


def evaluate_one_video(video_id, clip_iqa_path, tof_path, identity_results_path,
                       closeup_bbox_p50=None,
                       temperature=0.2, low_confidence_floor=0.2):
    clip_iqa = json.load(open(clip_iqa_path))
    tof_payload = json.load(open(tof_path))
    id_full = json.load(open(identity_results_path))
    id_pv = id_full["per_video"].get(video_id)
    if id_pv is None:
        raise ValueError("video_id '" + video_id + "' not in identity results")

    a = appearance_score(clip_iqa)
    t = temporal_score(tof_payload)
    i = identity_score(id_pv, closeup_bbox_p50=closeup_bbox_p50)

    comp = compose_score([a["score"], t["score"], i["score"]],
                         [a["reliability"], t["reliability"], i["reliability"]],
                         temperature=temperature,
                         low_confidence_floor=low_confidence_floor)
    return {
        "video": video_id,
        "lr_vcc": comp["score"],
        "weights_used": comp["weights"],
        "low_confidence": comp["low_confidence"],
        "sub_metrics": {
            "appearance": a,
            "temporal": t,
            "identity": i,
        },
        "diagnostics": {
            "closeup_bbox_p50": closeup_bbox_p50,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, help="method name, e.g. mgld or uav")
    ap.add_argument("--clip_iqa_dir", required=True,
                    help="dir of <basename>_clip_iqa.json files")
    ap.add_argument("--tof_dir", required=True,
                    help="dir of <basename>_tof_tlp.json files")
    ap.add_argument("--identity_results", required=True,
                    help="single JSON from human_identity_long.py with per_video[<v>]")
    ap.add_argument("--closeup_p50_map", default=None,
                    help="optional JSON {video_id: face_or_hand_bbox_p50}")
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--low_confidence_floor", type=float, default=0.2)
    args = ap.parse_args()

    closeup_map = {}
    if args.closeup_p50_map and os.path.isfile(args.closeup_p50_map):
        closeup_map = json.load(open(args.closeup_p50_map))

    os.makedirs(args.output_path, exist_ok=True)

    clip_iqa_files = sorted(glob.glob(os.path.join(args.clip_iqa_dir, "*_clip_iqa.json")))
    if not clip_iqa_files:
        sys.exit("no clip_iqa JSONs in " + args.clip_iqa_dir)

    per_video_results = []
    for fa in clip_iqa_files:
        base = os.path.basename(fa).replace("_clip_iqa.json", "")
        ft = os.path.join(args.tof_dir, base + "_tof_tlp.json")
        if not os.path.isfile(ft):
            print("[skip] no tof for " + base)
            continue
        try:
            out = evaluate_one_video(
                video_id=base,
                clip_iqa_path=fa,
                tof_path=ft,
                identity_results_path=args.identity_results,
                closeup_bbox_p50=closeup_map.get(base),
                temperature=args.temperature,
                low_confidence_floor=args.low_confidence_floor,
            )
        except Exception as e:
            print("[error] " + base + ": " + str(e))
            continue
        per_video_results.append(out)
        out_file = os.path.join(args.output_path, base + ".json")
        with open(out_file, "w") as f:
            json.dump(out, f, indent=2)
        print(base + ": lr_vcc=" + format(out["lr_vcc"], ".4f")
              + (" (LOW_CONF)" if out["low_confidence"] else ""))

    high_conf = [r for r in per_video_results if not r["low_confidence"]]
    aggregate = {
        "method": args.method,
        "n_videos": len(per_video_results),
        "n_high_confidence": len(high_conf),
        "mean_lr_vcc": mean([r["lr_vcc"] for r in high_conf]) if high_conf else None,
        "per_video": per_video_results,
    }
    with open(os.path.join(args.output_path, "_aggregate.json"), "w") as f:
        json.dump(aggregate, f, indent=2)
    print("Aggregate mean LR-VCC: " + format(aggregate["mean_lr_vcc"] or -1, ".4f"))


if __name__ == "__main__":
    main()
