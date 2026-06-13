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
from .color_stability import color_stability_score


def evaluate_one_video(video_id, clip_iqa_path, tof_path, identity_results_path,
                       closeup_bbox_p50=None,
                       color_hist_path=None,
                       color_hist_alpha=None,
                       color_slope_path=None,
                       color_slope_beta=None,
                       color_hist_anchor_path=None,
                       dprime_beta=None,
                       clip_trajectory_path=None,
                       dprime2_beta=None,
                       temperature=0.2, low_confidence_floor=0.2,
                       temporal_weight="log"):
    clip_iqa = json.load(open(clip_iqa_path))
    tof_payload = json.load(open(tof_path))
    id_full = json.load(open(identity_results_path))
    id_pv = id_full["per_video"].get(video_id)
    if id_pv is None:
        raise ValueError("video_id '" + video_id + "' not in identity results")

    a = appearance_score(clip_iqa)
    t = temporal_score(tof_payload, weight_fn=temporal_weight)
    i = identity_score(id_pv, closeup_bbox_p50=closeup_bbox_p50)

    # Sub-metric D: color-histogram temporal stability (optional).
    c = None
    if color_hist_path is not None and os.path.isfile(str(color_hist_path)):
        raw = json.load(open(color_hist_path))
        c = color_stability_score(raw, alpha=color_hist_alpha)

    # Sub-metric E: color-slope drift detector (optional).
    e = None
    if color_slope_path is not None and os.path.isfile(str(color_slope_path)):
        raw_e = json.load(open(color_slope_path))
        # The compute_color_slope.py payload already has {score, reliability, details}.
        # If color_slope_beta is supplied, re-derive score = exp(-beta * max_abs_slope)
        # using the stored raw slope — avoids re-scanning videos when retuning beta.
        details_e = raw_e.get("details", {}) or {}
        if color_slope_beta is not None and "max_abs_slope" in details_e:
            import math as _math
            new_score = _math.exp(-float(color_slope_beta) * float(details_e["max_abs_slope"]))
            new_score = max(0.0, min(1.0, new_score))
            new_details = dict(details_e)
            new_details["beta"] = float(color_slope_beta)
            new_details["beta_override"] = True
            e = {
                "score": new_score,
                "reliability": float(raw_e.get("reliability", 0.0)),
                "details": new_details,
            }
        else:
            e = {
                "score": float(raw_e.get("score", 0.0)),
                "reliability": float(raw_e.get("reliability", 0.0)),
                "details": details_e,
            }

    # Sub-metric D' — anchor-window Lab histogram drift (optional).
    dp_score = None
    if color_hist_anchor_path is not None and os.path.isfile(str(color_hist_anchor_path)):
        raw = json.load(open(color_hist_anchor_path))
        q = raw.get("details", {}).get("trajectory_mean_per_quarter")
        if q is not None and len(q) >= 4:
            import math as _math
            beta = float(dprime_beta) if dprime_beta is not None else 0.5
            new_score = _math.exp(-beta * abs(float(q[3]) - float(q[0])))
            new_score = max(0.0, min(1.0, new_score))
            dp_score = {
                "score": new_score,
                "reliability": float(raw.get("reliability", 1.0)),
                "details": {**(raw.get("details") or {}), "beta_override": beta},
            }

    # Sub-metric D'' — CLIP-trajectory drift (optional).
    dpp_score = None
    if clip_trajectory_path is not None and os.path.isfile(str(clip_trajectory_path)):
        raw = json.load(open(clip_trajectory_path))
        q = raw.get("details", {}).get("trajectory_mean_per_quarter")
        if q is not None and len(q) >= 4:
            import math as _math
            beta = float(dprime2_beta) if dprime2_beta is not None else 3.0
            new_score = _math.exp(-beta * abs(float(q[3]) - float(q[0])))
            new_score = max(0.0, min(1.0, new_score))
            dpp_score = {
                "score": new_score,
                "reliability": float(raw.get("reliability", 1.0)),
                "details": {**(raw.get("details") or {}), "beta_override": beta},
            }

    scores = [a["score"], t["score"], i["score"]]
    rels = [a["reliability"], t["reliability"], i["reliability"]]
    sub_metrics = {"appearance": a, "temporal": t, "identity": i}

    if c is not None:
        scores.append(c["score"])
        rels.append(c["reliability"])
        sub_metrics["color_stability"] = c

    if e is not None:
        scores.append(e["score"])
        rels.append(e["reliability"])
        sub_metrics["color_slope"] = e

    if dp_score is not None:
        scores.append(dp_score["score"])
        rels.append(dp_score["reliability"])
        sub_metrics["color_hist_anchor"] = dp_score

    if dpp_score is not None:
        scores.append(dpp_score["score"])
        rels.append(dpp_score["reliability"])
        sub_metrics["clip_trajectory"] = dpp_score

    comp = compose_score(scores, rels,
                         temperature=temperature,
                         low_confidence_floor=low_confidence_floor)

    return {
        "video": video_id,
        "lr_vcc": comp["score"],
        "weights_used": comp["weights"],
        "low_confidence": comp["low_confidence"],
        "sub_metrics": sub_metrics,
        "diagnostics": {
            "closeup_bbox_p50": closeup_bbox_p50,
            "color_hist_used": color_hist_path is not None,
            "color_slope_used": color_slope_path is not None,
            "color_hist_anchor_used": color_hist_anchor_path is not None,
            "clip_trajectory_used": clip_trajectory_path is not None,
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
    ap.add_argument("--color_hist_dir", default=None,
                    help="optional dir of <basename>_color_hist.json files (sub-metric D)")
    ap.add_argument("--color_hist_alpha", type=float, default=None,
                    help="optional override of sub-metric D's alpha for "
                         "score = exp(-alpha * mean_hist_dist). Default None "
                         "uses the JSON's stored alpha (backwards compatible).")
    ap.add_argument("--color_slope_dir", default=None,
                    help="optional dir of <basename>_color_slope.json files "
                         "(sub-metric E — linear-regression color drift detector)")
    ap.add_argument("--color_slope_beta", type=float, default=None,
                    help="optional override of sub-metric E's beta for "
                         "score = exp(-beta * max_abs_slope). Default None uses "
                         "the JSON's stored score (re-derive without re-scanning).")
    ap.add_argument("--color_hist_anchor_dir", default=None,
                    help="optional dir of <basename>_color_hist_anchor.json files "
                         "(sub-metric D' — anchor-window Lab histogram drift)")
    ap.add_argument("--dprime_beta", type=float, default=0.5,
                    help="beta for D' score = exp(-beta * |q4-q1|). Default 0.5.")
    ap.add_argument("--clip_trajectory_dir", default=None,
                    help="optional dir of <basename>_clip_trajectory.json files "
                         "(sub-metric D'' — CLIP-trajectory drift)")
    ap.add_argument("--dprime2_beta", type=float, default=3.0,
                    help="beta for D'' score = exp(-beta * |q4-q1|). Default 3.0.")
    ap.add_argument("--temporal_weight", choices=["log", "uniform", "sqrt"], default="log",
                    help="tOF weighting scheme: log (default), uniform, or sqrt")
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
        color_hist_path = None
        if args.color_hist_dir:
            color_hist_path = os.path.join(args.color_hist_dir, base + "_color_hist.json")
        color_slope_path = None
        if args.color_slope_dir:
            color_slope_path = os.path.join(args.color_slope_dir, base + "_color_slope.json")
        color_hist_anchor_path = None
        if args.color_hist_anchor_dir:
            color_hist_anchor_path = os.path.join(
                args.color_hist_anchor_dir, base + "_color_hist_anchor.json")
        clip_trajectory_path = None
        if args.clip_trajectory_dir:
            clip_trajectory_path = os.path.join(
                args.clip_trajectory_dir, base + "_clip_trajectory.json")
        try:
            out = evaluate_one_video(
                video_id=base,
                clip_iqa_path=fa,
                tof_path=ft,
                identity_results_path=args.identity_results,
                closeup_bbox_p50=closeup_map.get(base),
                color_hist_path=color_hist_path,
                color_hist_alpha=args.color_hist_alpha,
                color_slope_path=color_slope_path,
                color_slope_beta=args.color_slope_beta,
                color_hist_anchor_path=color_hist_anchor_path,
                dprime_beta=args.dprime_beta,
                clip_trajectory_path=clip_trajectory_path,
                dprime2_beta=args.dprime2_beta,
                temperature=args.temperature,
                low_confidence_floor=args.low_confidence_floor,
                temporal_weight=args.temporal_weight,
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
