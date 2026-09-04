"""Probe v3 — reference-similarity face selection over candidate dumps.

Selection rule under test: per frame, score the candidate whose embedding is
closest to the video-level reference (vs inherited largest-bbox rule).
Reference: mean of largest-face embeddings from the first 10 face-bearing
clips (floor 0.003 of frame area). Aggregation: median per clip, mean over
clips. Compares BOTH selection rules from the SAME npz files.
"""
import glob, os, re
from collections import defaultdict
import numpy as np

SEVS = ["0p02", "0p05", "0p10", "0p20", "0p40"]
_RE = re.compile(r"^(?P<base>.+)_sev(?P<sev>\dp\d+)_faces\.npz$")
FLOOR = 0.003

def load(path):
    z = np.load(path)
    n = int(z["n_clips"])
    out = []
    for i in range(n):
        out.append({"emb": z[f"c{i}_emb"].astype(np.float32),
                    "area": z[f"c{i}_bbox_area"].astype(np.float32),
                    "cand_emb": z[f"c{i}_cand_emb"].astype(np.float32),
                    "cand_frame": z[f"c{i}_cand_frame"].astype(np.int64),
                    "cand_area": z[f"c{i}_cand_area"].astype(np.float32),
                    "fa": float(z[f"c{i}_frame_area"])})
    return out

def norm(v): return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)

def ref_of(clips, W=10):
    got, used = [], 0
    for c in clips:
        if used >= W and got: break
        keep = (c["area"] / c["fa"]) >= FLOOR
        e = c["emb"][keep]
        if e.shape[0]: got.append(e); used += 1
        elif used < W: used += 1
    if not got: return None
    return norm(np.concatenate(got, 0).mean(0))

def score(clips, ref, select):
    if ref is None: return None
    per = []
    for c in clips:
        if select == "largest":
            keep = (c["area"] / c["fa"]) >= FLOOR
            e = c["emb"][keep]
            if not e.shape[0]: continue
            sims = norm(e) @ ref
        else:  # bestmatch: per frame, candidate with max similarity to ref
            if not c["cand_emb"].shape[0]: continue
            keep = (c["cand_area"] / c["fa"]) >= FLOOR
            ce, cf = c["cand_emb"][keep], c["cand_frame"][keep]
            if not ce.shape[0]: continue
            s = norm(ce) @ ref
            best = defaultdict(lambda: -2.0)
            for fr, sv in zip(cf, s):
                if sv > best[int(fr)]: best[int(fr)] = float(sv)
            sims = np.array(list(best.values()), dtype=np.float32)
        per.append(float(np.median(sims)))
    return float(np.mean(per)) if per else None

H = os.path.expanduser("~")
d = f"{H}/results/identity_embeddings/_cand_background_drift"
by = defaultdict(dict)
for f in sorted(glob.glob(d + "/*_faces.npz")):
    m = _RE.match(os.path.basename(f))
    if m: by[m["base"]][m["sev"]] = load(f)
bases = sorted(b for b in by if all(s in by[b] for s in SEVS))
print(f"### background_drift CONTROL, candidate dump ({len(bases)} bases) ###")
print("target: bestmatch response ~= 0 (flat); largest reproduces the failure")
print(f"{'rule':12s} {'base':14s} " + " ".join(f"{s:>7s}" for s in SEVS) + f" {'resp':>8s}")
for rule in ("largest", "bestmatch"):
    agg = []
    for b in bases:
        curve = [score(by[b][s], ref_of(by[b][s]), rule) for s in SEVS]
        if None in curve: continue
        print(f"{rule:12s} {b[:13]:14s} " + " ".join(f"{v:7.4f}" for v in curve)
              + f" {curve[0]-curve[-1]:+8.4f}")
        agg.append(curve)
    if agg:
        m = np.mean(agg, 0)
        print(f"{rule:12s} {'MEAN':14s} " + " ".join(f"{v:7.4f}" for v in m)
              + f" {m[0]-m[-1]:+8.4f}")
