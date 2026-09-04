"""Anchored identity (I') probe — offline, over dumped face embeddings.

The shipped identity sub-metric scores each clip against its own first face
(re-anchored per 2-second clip), so it cannot distinguish "consistently the
right person" from "consistently a blur" and rises under degradation. This
probe scores every clip against a VIDEO-LEVEL reference identity built from
the opening clips, using the same embeddings the legacy score is replayed
from — same detector, same era, so no cross-environment confound.

Inputs: the *_faces.npz files written by dump_identity_embeddings.py
(per-clip: c{i}_emb (N,1024), c{i}_bbox_area, c{i}_frame_area; plus
legacy_clip_scores replaying the shipped metric).

Variants probed: anchor window W (clips), reference = mean vs medoid,
bbox-area gating on/off. For each family x variant: mean score per severity,
response R = score(0.02) - score(0.40) (positive = correct direction),
monotonicity across the 5-point ladder.

Controls: background_drift npz (identity constant -> I' should stay ~flat).

Usage (server):
  python scripts/lr_vcc/anchored_identity_probe.py \
      --dirs ~/results/identity_embeddings/identity_degradation \
             ~/results/identity_embeddings/identity_drift \
             ~/results/identity_embeddings/_gate_background_drift
"""
import argparse
import glob
import os
import re
from collections import defaultdict

import numpy as np

SEVS = ["0p02", "0p05", "0p10", "0p20", "0p40"]
_RE = re.compile(r"^(?P<base>.+)_sev(?P<sev>\dp\d+)_faces\.npz$")


def load_video(path):
    z = np.load(path)
    n = int(z["n_clips"])
    clips = []
    for i in range(n):
        emb = z[f"c{i}_emb"]
        clips.append({
            "emb": emb.astype(np.float32),
            "area": z[f"c{i}_bbox_area"].astype(np.float32),
            "frame_area": float(z[f"c{i}_frame_area"]),
        })
    return clips, z["legacy_clip_scores"].astype(np.float32)


def _norm(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


def build_reference(clips, window, gate_area):
    """Reference embedding set from the first `window` clips with any faces.

    gate_area: keep only faces with bbox_area >= median area of the collected
    window (drops small/marginal detections). Expands past `window` if the
    opening clips carry no faces at all (the spec's no-face fallback).
    """
    got, used = [], 0
    for c in clips:
        if used >= window and got:
            break
        if c["emb"].shape[0]:
            got.append(c)
            used += 1
        elif used < window:
            used += 1  # empty clip inside the window still consumes it
    if not got:                      # fallback: first clips with faces anywhere
        got = [c for c in clips if c["emb"].shape[0]][:window]
    if not got:
        return None
    embs = np.concatenate([c["emb"] for c in got], axis=0)
    areas = np.concatenate([c["area"] for c in got], axis=0)
    if gate_area and embs.shape[0] >= 4:
        keep = areas >= np.median(areas)
        if keep.sum() >= 2:
            embs = embs[keep]
    return _norm(embs)


def video_score(clips, ref, ref_kind):
    if ref is None:
        return None
    center = _norm(ref.mean(axis=0)) if ref_kind == "mean" else None
    if ref_kind == "medoid":
        sims = ref @ ref.T
        center = ref[np.argmax(sims.sum(axis=0))]
    per_clip = []
    for c in clips:
        if not c["emb"].shape[0]:
            continue
        e = _norm(c["emb"])
        per_clip.append(float(np.mean(e @ center)))
    return float(np.mean(per_clip)) if per_clip else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    args = ap.parse_args()

    variants = [(w, rk, g) for w in (3, 5, 10) for rk in ("mean", "medoid")
                for g in (True, False)]

    for d in args.dirs:
        fam = os.path.basename(d.rstrip("/")).replace("_gate_", "")
        files = sorted(glob.glob(os.path.join(d, "*_faces.npz")))
        by = defaultdict(dict)      # base -> sev -> (clips, legacy)
        for f in files:
            m = _RE.match(os.path.basename(f))
            if m:
                by[m["base"]][m["sev"]] = load_video(f)
        bases = sorted(b for b in by if all(s in by[b] for s in SEVS))
        print(f"\n######## {fam} — {len(bases)} bases with full ladders ########")

        # Legacy replay row (the same-era baseline)
        print(f"{'variant':24s} " + " ".join(f"{s:>7s}" for s in SEVS)
              + f" {'resp':>8s} {'mono':>5s}")
        rows = []
        leg = []
        for s in SEVS:
            vals = []
            for b in bases:
                _, lc = by[b][s]
                ok = lc[lc >= 0]
                if ok.size:
                    vals.append(float(ok.mean()))
            leg.append(np.mean(vals) if vals else float("nan"))
        resp = leg[0] - leg[-1]
        mono = all(leg[i+1] <= leg[i] + 1e-9 for i in range(4))
        print(f"{'LEGACY (replayed)':24s} " + " ".join(f"{v:7.4f}" for v in leg)
              + f" {resp:+8.4f} {'yes' if mono else 'NO':>5s}")

        for w, rk, g in variants:
            curve = []
            for s in SEVS:
                vals = []
                for b in bases:
                    clips, _ = by[b][s]
                    ref = build_reference(clips, w, g)
                    sc = video_score(clips, ref, rk)
                    if sc is not None:
                        vals.append(sc)
                curve.append(np.mean(vals) if vals else float("nan"))
            resp = curve[0] - curve[-1]
            mono = all(curve[i+1] <= curve[i] + 1e-9 for i in range(4))
            tag = f"I' W={w:<2d} {rk:6s} gate={'Y' if g else 'N'}"
            print(f"{tag:24s} " + " ".join(f"{v:7.4f}" for v in curve)
                  + f" {resp:+8.4f} {'yes' if mono else 'NO':>5s}")


if __name__ == "__main__":
    main()
