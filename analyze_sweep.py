"""Correlate the human ratings against every judge in the sweep, on the same
human-rated pairs, with bootstrap 95% CIs on overall Pearson.

    python analyze_sweep.py <human.json> [opts] LABEL=FILE.json ...

opts (optional, order-free):
    --qwen-local <full18_judged.json>   local Qwen-8B (mllm{Texture..}), joined by (video,i,j)
    --claude-inchat <claude_ratings.json>   in-chat Claude (id-keyed, non-blind)

Each LABEL=FILE is an OpenRouter judge output (records: video,i,j,texture,geometry,semantic).
"""
import json
import sys

import numpy as np

AX = ("texture", "geometry", "semantic")
MX = {"texture": "Texture", "geometry": "Geometry", "semantic": "Semantic"}


def stem(v):
    return v.rsplit(".", 1)[0]


args = sys.argv[1:]
human = json.load(open(args.pop(0)))
qwen_local = claude_inchat = None
judges = []  # (label, key->scores dict keyed by (stem,i,j))
i = 0
while i < len(args):
    a = args[i]
    if a == "--qwen-local":
        qwen_local = json.load(open(args[i + 1])); i += 2
    elif a == "--claude-inchat":
        claude_inchat = json.load(open(args[i + 1])); i += 2
    else:
        label, path = a.split("=", 1)
        recs = json.load(open(path))
        judges.append((label, {(stem(r["video"]), r["i"], r["j"]): r for r in recs}))
        i += 1

# id -> (stem,i,j) from human, to place the id-keyed in-chat Claude on the same join
id2key = {h["id"]: (stem(h["video"]), h["i"], h["j"]) for h in human if "id" in h}

if qwen_local is not None:
    km = {(stem(r["video"]), r["i"], r["j"]): {ax: (r.get("mllm") or {}).get(MX[ax]) for ax in AX}
          for r in qwen_local if "Texture" in (r.get("mllm") or {})}
    judges.insert(0, ("qwen8B-local", km))
if claude_inchat is not None:
    km = {}
    for r in claude_inchat:
        key = id2key.get(r["id"])
        if key:
            km[key] = {ax: r.get(ax) for ax in AX}
    judges.insert(0, ("claude-inchat*", km))

# human rows that are fully rated
H = [h for h in human if all(h.get(a) is not None for a in AX)]
print(f"fully-rated human pairs: {len(H)}\n")


def _c(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    return a[m], b[m]


def pearson(a, b):
    a, b = _c(a, b)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a, b):
    a, b = _c(a, b)
    if len(a) < 3:
        return float("nan")
    return pearson(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))


def ci(a, b, B=10000, seed=0):
    a, b = _c(a, b)
    n = len(a)
    if n < 4:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    out = [pearson(a[idx], b[idx]) for idx in (rng.integers(0, n, n) for _ in range(B))]
    out = [v for v in out if np.isfinite(v)]
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))) if out else (float("nan"),) * 2


def collect(km, axis):
    """Paired (human, judge) values over H where the judge has the pair + axis."""
    hs, js, n = [], [], 0
    for h in H:
        key = (stem(h["video"]), h["i"], h["j"])
        j = km.get(key)
        if not j:
            continue
        if axis == "OVERALL":
            if any(j.get(a) is None for a in AX):
                continue
            hv = sum(float(h[a]) for a in AX) / 3
            jv = sum(float(j[a]) for a in AX) / 3
        else:
            if j.get(axis) is None:
                continue
            hv, jv = float(h[axis]), float(j[axis])
        hs.append(hv); js.append(jv); n += 1
    return hs, js, n


print(f"{'judge':22s} {'n':>3s} | {'texture':>8s} {'geom':>8s} {'seman':>8s} | "
      f"{'OVERALL pearson [95% CI]':>26s} {'srcc':>6s}")
print("-" * 92)
for label, km in judges:
    cells = []
    for axis in AX:
        hs, js, _ = collect(km, axis)
        cells.append(f"{spearman(hs, js):>+8.2f}")
    ho, jo, n = collect(km, "OVERALL")
    p, s = pearson(ho, jo), spearman(ho, jo)
    lo, hi_ = ci(ho, jo)
    print(f"{label:22s} {n:>3d} | {cells[0]} {cells[1]} {cells[2]} | "
          f"{p:>+6.2f} [{lo:+.2f},{hi_:+.2f}]{'':>7s} {s:>+6.2f}")
print("\n(SRCC shown per axis; overall shows PLCC [bootstrap CI] and SRCC.  "
      "* = non-blind, in-chat, for reference only.)")
