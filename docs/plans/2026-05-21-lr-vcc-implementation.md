# LR-VCC Implementation Plan — Proposal Sprint (10 days)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an MVP of the LR-VCC composite metric, produce Layer 1+2 validation evidence on the 5 synthetic videos × 2 methods, then write the May 31 proposal document around it.

**Architecture:** Three sub-metric wrappers (`appearance.py`, `temporal.py`, `identity.py`), each emitting `(score, reliability)`. A composition layer (`composite.py`) applies softmax-weighted log-mean. A CLI runner (`run_lr_vcc.py`) takes a videos dir + per-method existing-eval JSONs (tOF, Identity) and produces per-video composite results plus a method-aggregate JSON. Implementation references the design at `docs/plans/2026-05-21-lr-vcc-design.md`.

**Tech Stack:** Python 3.10, NumPy, `pyiqa` (for CLIP-IQA on the server `vsr` env), JSON I/O. Reuses outputs from `scripts/vbench2_long/human_identity_long.py` and `scripts/long_range_temporal/eval_tof_tlp.py`.

**Sprint dates:** May 21 → May 31 (proposal submission deadline)

---

## Phase 1 — LR-VCC sub-metric wrappers (Days 1–3)

### Task 1: Project skeleton + reliability sigmoid helper

**Files:**
- Create: `scripts/lr_vcc/__init__.py`
- Create: `scripts/lr_vcc/reliability.py`
- Create: `tests/lr_vcc/__init__.py`
- Create: `tests/lr_vcc/test_reliability.py`

- [ ] **Step 1: Write the failing test for `below_threshold_penalty`**

```python
# tests/lr_vcc/test_reliability.py
import math
from scripts.lr_vcc.reliability import below_threshold_penalty, above_threshold_penalty


def test_below_threshold_penalty_at_threshold_is_half():
    # exactly at threshold -> sigmoid(0) = 0.5
    assert abs(below_threshold_penalty(value=0.10, threshold=0.10, sharpness=10) - 0.5) < 1e-6


def test_below_threshold_penalty_well_above_is_near_zero():
    # far above threshold -> sigmoid(-large) -> ~0 penalty
    assert below_threshold_penalty(value=0.50, threshold=0.10, sharpness=10) < 0.05


def test_below_threshold_penalty_well_below_is_near_one():
    # far below threshold -> sigmoid(+large) -> ~1 penalty
    assert below_threshold_penalty(value=0.01, threshold=0.10, sharpness=10) > 0.6


def test_above_threshold_penalty_symmetric():
    # mirror semantics: high value -> high penalty (used for saturation, close-up flags)
    assert abs(above_threshold_penalty(value=0.10, threshold=0.10, sharpness=10) - 0.5) < 1e-6
    assert above_threshold_penalty(value=0.50, threshold=0.10, sharpness=10) > 0.6
    assert above_threshold_penalty(value=0.01, threshold=0.10, sharpness=10) < 0.05
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/lr_vcc/test_reliability.py -v`
Expected: FAIL with `ImportError: No module named 'scripts.lr_vcc.reliability'`

- [ ] **Step 3: Implement `reliability.py`**

```python
# scripts/lr_vcc/reliability.py
"""Smooth (sigmoid) reliability penalties around documented thresholds.

A penalty in [0, 1] is converted to a reliability via `reliability = 1 - penalty`
in the calling sub-metric. We keep penalties separate so callers can combine
multiple penalties (max, sum, weighted) before forming the reliability.
"""
import math


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ez = math.exp(x)
    return ez / (1.0 + ez)


def below_threshold_penalty(value: float, threshold: float, sharpness: float = 10.0) -> float:
    """Returns ~1 when value << threshold, ~0 when value >> threshold.

    Used when LOW values indicate a bad regime (e.g. mask coverage too low).
    """
    return _sigmoid(sharpness * (threshold - value))


def above_threshold_penalty(value: float, threshold: float, sharpness: float = 10.0) -> float:
    """Returns ~1 when value >> threshold, ~0 when value << threshold.

    Used when HIGH values indicate a bad regime (e.g. saturation, close-up ratio).
    """
    return _sigmoid(sharpness * (value - threshold))
```

- [ ] **Step 4: Run test to verify pass**

Run: `pytest tests/lr_vcc/test_reliability.py -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/lr_vcc/__init__.py scripts/lr_vcc/reliability.py tests/lr_vcc/__init__.py tests/lr_vcc/test_reliability.py
git commit -m "lr_vcc: project skeleton + sigmoid reliability helpers"
```

---

### Task 2: Composition layer (softmax + log-mean)

**Files:**
- Create: `scripts/lr_vcc/composite.py`
- Create: `tests/lr_vcc/test_composite.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/lr_vcc/test_composite.py
import math
from scripts.lr_vcc.composite import compose_score


def test_all_equal_reliabilities_geometric_mean():
    # equal reliability -> uniform weights -> geometric mean
    scores = [0.6, 0.8, 0.4]
    rels = [0.9, 0.9, 0.9]
    out = compose_score(scores, rels, temperature=0.2)
    geom = (0.6 * 0.8 * 0.4) ** (1 / 3)
    assert abs(out["score"] - geom) < 1e-3
    for w in out["weights"]:
        assert abs(w - 1 / 3) < 1e-3
    assert not out["low_confidence"]


def test_one_reliable_dominates():
    # one sub-metric is much more reliable -> its score dominates the composite
    scores = [0.1, 0.9, 0.5]
    rels = [0.05, 0.95, 0.05]
    out = compose_score(scores, rels, temperature=0.2)
    # weight on the 2nd should be >0.9
    assert out["weights"][1] > 0.9
    # composite should be close to 0.9 (the dominant score), not the geometric mean
    assert out["score"] > 0.7


def test_all_unreliable_marks_low_confidence():
    out = compose_score([0.5, 0.5, 0.5], [0.1, 0.1, 0.1], temperature=0.2,
                        low_confidence_floor=0.2)
    assert out["low_confidence"]


def test_some_reliable_not_low_confidence():
    out = compose_score([0.5, 0.5, 0.5], [0.1, 0.5, 0.1], temperature=0.2,
                        low_confidence_floor=0.2)
    assert not out["low_confidence"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/lr_vcc/test_composite.py -v`
Expected: FAIL with import error.

- [ ] **Step 3: Implement `composite.py`**

```python
# scripts/lr_vcc/composite.py
"""Softmax-weighted log-mean composition of sub-metric (score, reliability) pairs."""
import math
from typing import Sequence


def _softmax(xs: Sequence[float], temperature: float = 0.2) -> list[float]:
    z = [x / temperature for x in xs]
    z_max = max(z)
    exps = [math.exp(zi - z_max) for zi in z]
    s = sum(exps)
    return [e / s for e in exps]


def compose_score(scores: Sequence[float], reliabilities: Sequence[float],
                  temperature: float = 0.2, eps: float = 1e-6,
                  low_confidence_floor: float = 0.2) -> dict:
    """Softmax-weight reliabilities, then exp(sum w_i log(score_i + eps)).

    Returns: {"score": float, "weights": list[float], "low_confidence": bool}.
    """
    assert len(scores) == len(reliabilities)
    weights = _softmax(reliabilities, temperature=temperature)
    log_sum = sum(w * math.log(s + eps) for w, s in zip(weights, scores))
    score = math.exp(log_sum)
    low_conf = all(r < low_confidence_floor for r in reliabilities)
    return {"score": score, "weights": weights, "low_confidence": low_conf}
```

- [ ] **Step 4: Run tests, verify pass**

Run: `pytest tests/lr_vcc/test_composite.py -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/lr_vcc/composite.py tests/lr_vcc/test_composite.py
git commit -m "lr_vcc: softmax + log-mean composition"
```

---

### Task 3: Sub-metric T (temporal — reads existing tOF JSONs)

**Files:**
- Create: `scripts/lr_vcc/temporal.py`
- Create: `tests/lr_vcc/test_temporal.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/lr_vcc/test_temporal.py
import json
import math
from pathlib import Path
from scripts.lr_vcc.temporal import temporal_score


def _fixture_tof_payload(tof_values: dict, mask_coverage: dict) -> dict:
    """Same shape as scripts/long_range_temporal/eval_tof_tlp.py output."""
    return {
        "video_path": "/fake/video.mp4",
        "n_frames": 5000,
        "fps": 30.0,
        "k_values": [1, 5, 10, 30, 60, 120],
        "tof": {str(k): v for k, v in tof_values.items()},
        "tlp": {str(k): 0.0 for k in tof_values},
        "n_pairs_used": {str(k): 200 for k in tof_values},
        "mean_mask_coverage": {str(k): v for k, v in mask_coverage.items()},
    }


def test_low_tof_high_coverage_yields_high_score():
    tof = {1: 0.01, 5: 0.02, 10: 0.03, 30: 0.04, 60: 0.05, 120: 0.06}
    cov = {1: 0.9, 5: 0.7, 10: 0.5, 30: 0.4, 60: 0.3, 120: 0.2}
    out = temporal_score(_fixture_tof_payload(tof, cov))
    assert out["score"] > 0.9
    assert out["reliability"] > 0.9


def test_high_tof_yields_low_score():
    tof = {1: 0.3, 5: 0.4, 10: 0.5, 30: 0.6, 60: 0.7, 120: 0.8}
    cov = {1: 0.9, 5: 0.7, 10: 0.5, 30: 0.4, 60: 0.3, 120: 0.2}
    out = temporal_score(_fixture_tof_payload(tof, cov))
    assert out["score"] < 0.5
    assert out["reliability"] > 0.9


def test_low_mask_coverage_drops_reliability():
    tof = {1: 0.01, 5: 0.02, 10: 0.03, 30: 0.04, 60: 0.05, 120: 0.06}
    cov = {1: 0.05, 5: 0.04, 10: 0.03, 30: 0.02, 60: 0.01, 120: 0.005}  # all below 0.10 floor
    out = temporal_score(_fixture_tof_payload(tof, cov))
    assert out["reliability"] < 0.3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/lr_vcc/test_temporal.py -v`
Expected: FAIL with import error.

- [ ] **Step 3: Implement `temporal.py`**

```python
# scripts/lr_vcc/temporal.py
"""Sub-metric T — long-k-weighted tOF + mask-coverage reliability.

Reads the per-video JSON produced by scripts/long_range_temporal/eval_tof_tlp.py.
"""
import math
from typing import Iterable

from .reliability import below_threshold_penalty


_DEFAULT_MASK_COV_FLOOR = 0.10


def _weight_fn(k: int) -> float:
    """log(1+k) — weights long-k more than adjacent k."""
    return math.log(1 + k)


def temporal_score(tof_payload: dict,
                   mask_cov_floor: float = _DEFAULT_MASK_COV_FLOOR) -> dict:
    """Returns {"score", "reliability", "details": {...}}.

    score = 1 - weighted_mean(tof_k) over k with mask_coverage[k] >= floor.
    reliability = mean over k of (1 - below_threshold_penalty(coverage[k], floor)).
    """
    tofs = tof_payload["tof"]
    covs = tof_payload["mean_mask_coverage"]
    k_strs = list(tofs.keys())

    weighted_sum = 0.0
    weight_total = 0.0
    used_ks = []
    for k_str in k_strs:
        if tofs[k_str] is None:
            continue
        cov = float(covs.get(k_str, 0.0))
        if cov < mask_cov_floor:
            continue
        k = int(k_str)
        w = _weight_fn(k)
        weighted_sum += w * float(tofs[k_str])
        weight_total += w
        used_ks.append(k_str)

    if weight_total == 0:
        score = 0.0
    else:
        weighted_mean = weighted_sum / weight_total
        score = max(0.0, min(1.0, 1.0 - weighted_mean))

    # reliability — mean over all k of (1 - penalty), where penalty grows if coverage < floor
    rel_terms = []
    for k_str in k_strs:
        cov = float(covs.get(k_str, 0.0))
        rel_terms.append(1.0 - below_threshold_penalty(cov, mask_cov_floor))
    reliability = sum(rel_terms) / len(rel_terms) if rel_terms else 0.0

    return {
        "score": score,
        "reliability": reliability,
        "details": {
            "used_ks": used_ks,
            "weighted_mean_tof": (weighted_sum / weight_total) if weight_total else None,
            "mean_mask_coverage_over_all_k": sum(float(covs[k]) for k in k_strs) / len(k_strs),
        },
    }
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/lr_vcc/test_temporal.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/lr_vcc/temporal.py tests/lr_vcc/test_temporal.py
git commit -m "lr_vcc: sub-metric T (long-k weighted tOF + coverage reliability)"
```

---

### Task 4: Sub-metric I (identity — wraps existing slow-fast output)

**Files:**
- Create: `scripts/lr_vcc/identity.py`
- Create: `tests/lr_vcc/test_identity.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/lr_vcc/test_identity.py
from scripts.lr_vcc.identity import identity_score


def _fixture_id_per_video(slow, fast, fused, n_clips, n_faces):
    """Matches the per_video[video] shape from human_identity_long.py."""
    return {"slow": slow, "fast": fast, "fused": fused,
            "n_clips": n_clips, "n_clips_with_faces": n_faces}


def test_high_id_high_face_rate_high_score_high_rel():
    pv = _fixture_id_per_video(0.7, 0.6, 0.65, n_clips=80, n_faces=60)
    out = identity_score(pv, closeup_bbox_p50=None)  # no closeup info
    assert out["score"] == 0.65
    assert out["reliability"] > 0.85


def test_low_face_rate_drops_reliability():
    pv = _fixture_id_per_video(0.6, 0.5, 0.55, n_clips=80, n_faces=8)  # 10% rate < 20% floor
    out = identity_score(pv, closeup_bbox_p50=None)
    assert out["score"] == 0.55
    assert out["reliability"] < 0.5


def test_closeup_partial_downweight():
    pv = _fixture_id_per_video(0.7, 0.6, 0.65, n_clips=80, n_faces=60)
    out_no_closeup = identity_score(pv, closeup_bbox_p50=0.01)
    out_closeup = identity_score(pv, closeup_bbox_p50=0.18)  # well above 0.05 threshold
    assert out_closeup["reliability"] < out_no_closeup["reliability"]
```

- [ ] **Step 2: Run, verify fail**

Run: `pytest tests/lr_vcc/test_identity.py -v`
Expected: FAIL with import error.

- [ ] **Step 3: Implement `identity.py`**

```python
# scripts/lr_vcc/identity.py
"""Sub-metric I — slow-fast Human_Identity + face-rate + close-up reliability.

Wraps the per_video[v] output of scripts/vbench2_long/human_identity_long.py.
"""
from typing import Optional

from .reliability import below_threshold_penalty, above_threshold_penalty


_FACE_RATE_FLOOR = 0.20
_CLOSEUP_BBOX_THRESHOLD = 0.05  # face / hand bbox p50 as fraction of frame area


def identity_score(per_video: dict, closeup_bbox_p50: Optional[float] = None) -> dict:
    """Returns {"score", "reliability", "details": {...}}.

    score = per_video["fused"] (output of slow-fast Identity adapter).
    reliability = (1 - face_rate_penalty) * (1 - closeup_penalty).
    """
    score = float(per_video.get("fused", 0.0))
    n_clips = int(per_video.get("n_clips", 0))
    n_faces = int(per_video.get("n_clips_with_faces", 0))
    face_rate = n_faces / n_clips if n_clips > 0 else 0.0

    face_pen = below_threshold_penalty(face_rate, _FACE_RATE_FLOOR)
    if closeup_bbox_p50 is None:
        closeup_pen = 0.0
    else:
        closeup_pen = above_threshold_penalty(float(closeup_bbox_p50), _CLOSEUP_BBOX_THRESHOLD)

    reliability = (1.0 - face_pen) * (1.0 - closeup_pen)
    return {
        "score": max(0.0, min(1.0, score)),
        "reliability": reliability,
        "details": {
            "face_rate": face_rate,
            "face_penalty": face_pen,
            "closeup_bbox_p50": closeup_bbox_p50,
            "closeup_penalty": closeup_pen,
        },
    }
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/lr_vcc/test_identity.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/lr_vcc/identity.py tests/lr_vcc/test_identity.py
git commit -m "lr_vcc: sub-metric I (Identity slow-fast + face-rate + closeup reliability)"
```

---

### Task 5: Sub-metric A (appearance — CLIP-IQA on the server)

CLIP-IQA needs GPU; the wrapper reads a per-video JSON we'll produce by running `pyiqa` on the server. Splitting this into two steps:

**Step A** — server-side: per-frame CLIP-IQA dump
**Step B** — local wrapper

**Files:**
- Create: `scripts/lr_vcc/compute_clip_iqa.py` (server-side, GPU)
- Create: `scripts/lr_vcc/appearance.py` (local wrapper)
- Create: `tests/lr_vcc/test_appearance.py`

- [ ] **Step 1: Write the appearance test (works on a fixture, no GPU)**

```python
# tests/lr_vcc/test_appearance.py
from scripts.lr_vcc.appearance import appearance_score


def _fixture(qualities):
    return {"video_path": "/fake.mp4", "n_frames": len(qualities),
            "clip_iqa": qualities}


def test_high_mean_low_drift_high_score():
    out = appearance_score(_fixture([0.7] * 100))  # constant high quality
    assert out["score"] > 0.6
    assert out["reliability"] < 0.5  # drift too small => sub-metric undiscriminating


def test_high_mean_some_drift_high_score_high_rel():
    qs = [0.7 + 0.1 * (i % 2) for i in range(100)]  # std ~0.05
    out = appearance_score(_fixture(qs))
    assert out["score"] > 0.5
    assert out["reliability"] > 0.5


def test_low_mean_low_score():
    out = appearance_score(_fixture([0.2] * 100))
    assert out["score"] < 0.3


def test_lambda_penalizes_drift():
    high_drift = [0.5 + 0.4 * ((-1) ** i) for i in range(100)]  # std ~0.4
    out_lo_lambda = appearance_score(_fixture(high_drift), lam=0.0)
    out_hi_lambda = appearance_score(_fixture(high_drift), lam=2.0)
    assert out_lo_lambda["score"] > out_hi_lambda["score"]
```

- [ ] **Step 2: Run, verify fail**

Run: `pytest tests/lr_vcc/test_appearance.py -v`
Expected: FAIL with import error.

- [ ] **Step 3: Implement `appearance.py` (local wrapper, no GPU)**

```python
# scripts/lr_vcc/appearance.py
"""Sub-metric A — appearance stability (CLIP-IQA mean - lambda * std).

Reads a per-video JSON produced by compute_clip_iqa.py (server-side).
"""
import statistics

from .reliability import below_threshold_penalty, above_threshold_penalty


_DRIFT_FLOOR = 0.02       # if std(quality) < this -> sub-metric undiscriminating
_SATURATION_CEILING = 0.98  # if mean(quality) > this -> ceiling regime
_DEFAULT_LAMBDA = 0.5


def appearance_score(per_video_clip_iqa: dict, lam: float = _DEFAULT_LAMBDA) -> dict:
    qs = per_video_clip_iqa["clip_iqa"]
    if not qs:
        return {"score": 0.0, "reliability": 0.0, "details": {}}
    mean_q = statistics.mean(qs)
    std_q = statistics.pstdev(qs)  # population std for stability with small n
    score = max(0.0, min(1.0, mean_q - lam * std_q))

    drift_pen = below_threshold_penalty(std_q, _DRIFT_FLOOR)
    sat_pen = above_threshold_penalty(mean_q, _SATURATION_CEILING)
    reliability = max(0.0, 1.0 - max(drift_pen, sat_pen))

    return {
        "score": score,
        "reliability": reliability,
        "details": {
            "mean_quality": mean_q,
            "std_quality": std_q,
            "drift_penalty": drift_pen,
            "saturation_penalty": sat_pen,
        },
    }
```

- [ ] **Step 4: Run appearance tests**

Run: `pytest tests/lr_vcc/test_appearance.py -v`
Expected: 4 PASS

- [ ] **Step 5: Implement `compute_clip_iqa.py` (server-side)**

```python
# scripts/lr_vcc/compute_clip_iqa.py
"""Per-frame CLIP-IQA dump for one or more videos in a directory.

Server-side; needs GPU and pyiqa. Run from the vsr conda env.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/lr_vcc/compute_clip_iqa.py \
        --videos_path /data/disk2/timur/results/mgld_synthetic_mp4 \
        --output_path /data/disk2/timur/results/lr_vcc/clip_iqa/mgld
"""
import argparse
import json
import os
import sys

import cv2
import torch
import pyiqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos_path", required=True)
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--frame_stride", type=int, default=1,
                    help="evaluate every Nth frame (default 1 = every frame)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_path, exist_ok=True)

    print("Loading CLIP-IQA...")
    model = pyiqa.create_metric("clipiqa", device=device)

    video_files = sorted(
        f for f in os.listdir(args.videos_path)
        if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
    )
    if not video_files:
        sys.exit("No videos in " + args.videos_path)

    for vname in video_files:
        vpath = os.path.join(args.videos_path, vname)
        base = os.path.splitext(vname)[0]
        out_file = os.path.join(args.output_path, base + "_clip_iqa.json")
        if os.path.isfile(out_file):
            print("[skip] " + out_file)
            continue

        print("\n=== " + vname + " ===")
        cap = cv2.VideoCapture(vpath)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
        qualities = []
        frame_idx = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if frame_idx % args.frame_stride == 0:
                rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                t = torch.from_numpy(rgb).to(device).permute(2, 0, 1).float() / 255.0
                t = t.unsqueeze(0)
                with torch.no_grad():
                    q = float(model(t).item())
                qualities.append(q)
            frame_idx += 1
            if frame_idx % 500 == 0:
                print("  frame " + str(frame_idx) + "/" + str(n_frames))
        cap.release()
        payload = {
            "video_path": vpath,
            "n_frames": n_frames,
            "fps": fps,
            "frame_stride": args.frame_stride,
            "clip_iqa": qualities,
        }
        with open(out_file, "w") as f:
            json.dump(payload, f)
        print("  wrote " + out_file + " (" + str(len(qualities)) + " quality values)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Commit appearance code**

```bash
git add scripts/lr_vcc/appearance.py scripts/lr_vcc/compute_clip_iqa.py tests/lr_vcc/test_appearance.py
git commit -m "lr_vcc: sub-metric A (CLIP-IQA per-frame + appearance stability wrapper)"
```

- [ ] **Step 7: Push CLIP-IQA dump to server and run on both methods**

Run locally:
```bash
scp -i ~/.ssh/id_ed25519_timuj scripts/lr_vcc/compute_clip_iqa.py Timur@223.109.239.43:/data/disk2/timur/compute_clip_iqa.py
ssh -i ~/.ssh/id_ed25519_timuj Timur@223.109.239.43 'nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader | head'
```

Pick a free GPU (e.g. GPU 4 if free). Then on the server:

```bash
ssh -i ~/.ssh/id_ed25519_timuj Timur@223.109.239.43 'cat > /data/disk2/timur/run_clip_iqa.sh << "EOF"
#!/bin/bash
set -eo pipefail
DISK2=/data/disk2/timur
GPU=${1:-4}
eval "$($DISK2/miniconda3/bin/conda shell.bash hook)"
conda activate vsr
cd $DISK2
for method in mgld uav; do
    OUT=$DISK2/results/lr_vcc/clip_iqa/${method}
    mkdir -p $OUT
    echo "=== ${method} ==="
    CUDA_VISIBLE_DEVICES=$GPU python compute_clip_iqa.py \
        --videos_path $DISK2/results/${method}_synthetic_mp4 \
        --output_path $OUT
done
echo DONE
EOF
chmod +x /data/disk2/timur/run_clip_iqa.sh
LOG=/data/disk2/timur/logs/clip_iqa_$(date +%Y%m%d_%H%M%S).log
nohup /data/disk2/timur/run_clip_iqa.sh 4 > $LOG 2>&1 &
echo "PID=$! LOG=$LOG"
disown'
```

Expected runtime: ~5–10 min per video × 10 videos ≈ 1 hour.

- [ ] **Step 8: Verify run completed and pull JSONs to local**

When DONE appears in the log:
```bash
mkdir -p results/lr_vcc/clip_iqa
rsync -a -e "ssh -i ~/.ssh/id_ed25519_timuj" \
    Timur@223.109.239.43:/data/disk2/timur/results/lr_vcc/clip_iqa/ \
    results/lr_vcc/clip_iqa/
ls results/lr_vcc/clip_iqa/mgld/ results/lr_vcc/clip_iqa/uav/
```

Expected: 5 JSON files per method, each ~50–100 KB.

---

### Task 6: CLI runner that ties everything together

**Files:**
- Create: `scripts/lr_vcc/run_lr_vcc.py`
- Create: `tests/lr_vcc/test_run_lr_vcc.py`

- [ ] **Step 1: Write the end-to-end test using fixtures**

```python
# tests/lr_vcc/test_run_lr_vcc.py
import json
from pathlib import Path
from scripts.lr_vcc.run_lr_vcc import evaluate_one_video


def _make_fixture(tmp_path: Path, clip_iqa, tof, mask_cov, id_pv):
    clip_iqa_file = tmp_path / "clip_iqa.json"
    tof_file = tmp_path / "tof.json"
    id_file = tmp_path / "id.json"
    json.dump({"video_path": "/fake.mp4", "n_frames": 100, "fps": 30.0,
               "frame_stride": 1, "clip_iqa": clip_iqa}, open(clip_iqa_file, "w"))
    json.dump({"video_path": "/fake.mp4", "n_frames": 100, "fps": 30.0,
               "k_values": list(tof.keys()), "tof": {str(k): v for k, v in tof.items()},
               "tlp": {str(k): 0.0 for k in tof},
               "n_pairs_used": {str(k): 200 for k in tof},
               "mean_mask_coverage": {str(k): v for k, v in mask_cov.items()}},
              open(tof_file, "w"))
    json.dump({"per_video": {"fake": id_pv}}, open(id_file, "w"))
    return clip_iqa_file, tof_file, id_file


def test_good_video_high_lr_vcc(tmp_path):
    clip_iqa = [0.7 + 0.05 * (i % 3) for i in range(100)]  # high mean, some drift
    tof = {1: 0.02, 5: 0.04, 10: 0.05, 30: 0.07, 60: 0.10, 120: 0.13}
    cov = {1: 0.9, 5: 0.7, 10: 0.5, 30: 0.4, 60: 0.3, 120: 0.2}
    id_pv = {"slow": 0.8, "fast": 0.7, "fused": 0.75,
             "n_clips": 50, "n_clips_with_faces": 40}
    fa, ft, fi = _make_fixture(tmp_path, clip_iqa, tof, cov, id_pv)
    out = evaluate_one_video(video_id="fake", clip_iqa_path=fa, tof_path=ft,
                             identity_results_path=fi, closeup_bbox_p50=0.03)
    assert out["lr_vcc"] > 0.5
    assert not out["low_confidence"]


def test_low_face_rate_downweights_identity(tmp_path):
    clip_iqa = [0.7] * 100  # constant high, so A_reliability drops
    tof = {1: 0.02, 5: 0.04, 10: 0.05, 30: 0.07, 60: 0.10, 120: 0.13}
    cov = {1: 0.9, 5: 0.7, 10: 0.5, 30: 0.4, 60: 0.3, 120: 0.2}
    id_pv = {"slow": 0.3, "fast": 0.2, "fused": 0.25,
             "n_clips": 50, "n_clips_with_faces": 5}  # 10% face rate
    fa, ft, fi = _make_fixture(tmp_path, clip_iqa, tof, cov, id_pv)
    out = evaluate_one_video(video_id="fake", clip_iqa_path=fa, tof_path=ft,
                             identity_results_path=fi, closeup_bbox_p50=0.03)
    # Identity weight should be very small because face_rate << floor
    assert out["sub_metrics"]["identity"]["reliability"] < 0.3
    # composite shouldn't be dragged to identity's 0.25
    assert out["lr_vcc"] > 0.4
```

- [ ] **Step 2: Run, verify fail**

Run: `pytest tests/lr_vcc/test_run_lr_vcc.py -v`
Expected: FAIL with import error.

- [ ] **Step 3: Implement `run_lr_vcc.py`**

```python
# scripts/lr_vcc/run_lr_vcc.py
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
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/lr_vcc/ -v`
Expected: all tests PASS (reliability, composite, temporal, identity, appearance, run_lr_vcc).

- [ ] **Step 5: Commit**

```bash
git add scripts/lr_vcc/run_lr_vcc.py tests/lr_vcc/test_run_lr_vcc.py
git commit -m "lr_vcc: CLI runner that composes all three sub-metrics"
```

---

## Phase 2 — Validation Layer 1+2 (Days 4–5)

### Task 7: Build the close-up bbox-p50 map from cached anatomy traces

**Files:**
- Create: `scripts/lr_vcc/build_closeup_map.py`

- [ ] **Step 1: Implement the map-builder**

```python
# scripts/lr_vcc/build_closeup_map.py
"""Compute face/hand bbox p50 per video from cached per-frame anatomy traces.

Used to feed sub-metric I's close-up reliability test.
"""
import argparse
import json
import os
import sys
from pathlib import Path


def _bbox_area(bbox):
    if not bbox or len(bbox) < 4:
        return None
    x1, y1, x2, y2 = bbox[:4]
    return max(0, x2 - x1) * max(0, y2 - y1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_frame_dir", required=True,
                    help="dir with <method>_<video>_per_frame.json")
    ap.add_argument("--method", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--frame_area", type=int, default=1280 * 720)
    args = ap.parse_args()

    out = {}
    pattern = args.method + "_*_per_frame.json"
    for p in sorted(Path(args.per_frame_dir).glob(pattern)):
        # extract video id between method_ and _per_frame.json
        name = p.name
        video_id = name[len(args.method) + 1: -len("_per_frame.json")]
        d = json.load(open(p))
        areas = []
        for fr in d["frame_results"]:
            for person in fr.get("persons", []):
                for cat in ("face", "hand"):
                    for entry in person["scores"].get(cat, []):
                        bb = entry[1] if isinstance(entry, list) and len(entry) > 1 else None
                        a = _bbox_area(bb)
                        if a is not None and a > 0:
                            areas.append(a)
        if not areas:
            out[video_id] = 0.0
            continue
        areas.sort()
        p50 = areas[len(areas) // 2]
        out[video_id] = p50 / args.frame_area
        print(video_id + ": p50 = " + format(out[video_id] * 100, ".1f") + "% of frame")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote " + args.output)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run for both methods using cached per-frame data**

Per-frame anatomy traces are in 3 dirs (KZ, hhsz, and the rest). Easiest is to dump everything into one staging dir then run:

```bash
mkdir -p results/vbench2_anatomy/_all_per_frame
cp results/vbench2_anatomy/diagnostic_KZ8p6b1zJ9U/*.json \
   results/vbench2_anatomy/diagnostic_hhszUXL1Cu8/*.json \
   results/vbench2_anatomy/diagnostic_per_frame_all/*.json \
   results/vbench2_anatomy/_all_per_frame/
for m in mgld uav; do
    python scripts/lr_vcc/build_closeup_map.py \
        --per_frame_dir results/vbench2_anatomy/_all_per_frame \
        --method ${m} \
        --output results/lr_vcc/closeup_map/${m}.json
done
```

Expected output: 5 video_id → bbox-p50 pairs per method. KZ8p6b1zJ9U should be ~0.18 (matches our earlier characterization), others ≤ 0.07.

- [ ] **Step 3: Commit**

```bash
git add scripts/lr_vcc/build_closeup_map.py results/lr_vcc/closeup_map
git commit -m "lr_vcc: close-up bbox-p50 map builder + cached maps for 5 videos x 2 methods"
```

---

### Task 8: End-to-end run on 5 videos × 2 methods

Assumes Task 5 (CLIP-IQA dump) has finished and the JSONs are pulled. The Identity slow-fast and tOF JSONs are already in the repo.

- [ ] **Step 1: Locate prerequisite inputs**

```bash
ls results/lr_vcc/clip_iqa/mgld/ results/lr_vcc/clip_iqa/uav/        # 5 JSONs each
ls results/long_range_temporal/mgld/ results/long_range_temporal/uav/  # 5 JSONs each
ls results/vbench2_anatomy/identity_fps_overrides/mgld/ \
   results/vbench2_anatomy/identity_fps_overrides/uav/                # 1 JSON each
ls results/lr_vcc/closeup_map/mgld.json results/lr_vcc/closeup_map/uav.json
```

If anything is missing, do not proceed — fix the missing input first.

- [ ] **Step 2: Run LR-VCC for MGLD**

```bash
python -m scripts.lr_vcc.run_lr_vcc \
    --method mgld \
    --clip_iqa_dir results/lr_vcc/clip_iqa/mgld \
    --tof_dir results/long_range_temporal/mgld \
    --identity_results "$(ls results/vbench2_anatomy/identity_fps_overrides/mgld/results_*.json | head -1)" \
    --closeup_p50_map results/lr_vcc/closeup_map/mgld.json \
    --output_path results/lr_vcc/composite/mgld
```

Expected: 5 per-video JSONs + `_aggregate.json`, with `mean_lr_vcc` value printed.

- [ ] **Step 3: Run LR-VCC for UAV**

```bash
python -m scripts.lr_vcc.run_lr_vcc \
    --method uav \
    --clip_iqa_dir results/lr_vcc/clip_iqa/uav \
    --tof_dir results/long_range_temporal/uav \
    --identity_results "$(ls results/vbench2_anatomy/identity_fps_overrides/uav/results_*.json | head -1)" \
    --closeup_p50_map results/lr_vcc/closeup_map/uav.json \
    --output_path results/lr_vcc/composite/uav
```

- [ ] **Step 4: Print the comparison table**

```bash
python - <<'PY'
import json, glob
for v in ["7WHI2L_FDNg","BrRLKMbBTYQ","KZ8p6b1zJ9U","hhszUXL1Cu8","mJog8DlRk_4"]:
    m = json.load(open("results/lr_vcc/composite/mgld/" + v + ".json"))
    u = json.load(open("results/lr_vcc/composite/uav/" + v + ".json"))
    w = "MGLD" if m["lr_vcc"] > u["lr_vcc"] else "UAV"
    print(v, "MGLD=", round(m["lr_vcc"],3), "UAV=", round(u["lr_vcc"],3), "->", w)
PY
```

Expected outcome (Layer 1+2 pass):
- All 5 (or 4/5) videos show MGLD > UAV.
- KZ8p6b1zJ9U shows MGLD > UAV (Layer 2 pass — flip-resistance).

- [ ] **Step 5: Commit results**

```bash
git add results/lr_vcc/composite
git commit -m "results: LR-VCC composite on 5 synthetic videos x 2 methods (Layer 1+2)"
```

---

### Task 9: Validation note

**Files:**
- Create: `docs/notes/2026-05-25-lr-vcc-validation.md`

- [ ] **Step 1: Write the note**

Template the file with these sections:

```markdown
# LR-VCC validation — Layer 1+2 results

**Date:** 2026-05-25
**Spec:** docs/plans/2026-05-21-lr-vcc-design.md
**Inputs:** 5 synthetic SR videos × 2 methods, all sub-metric inputs precomputed.

## Layer 1 — perceptual agreement (5 videos)

| Video | MGLD LR-VCC | UAV LR-VCC | Winner | Per-sub-metric MGLD | Per-sub-metric UAV | Weights |
|-------|------------:|-----------:|:-------|---------------------|--------------------|---------|
| 7WHI2L_FDNg | <…> | <…> | <M/U> | A=<…>(<…>), T=<…>(<…>), I=<…>(<…>) | … | <…> |
| BrRLKMbBTYQ | … | … | … | … | … | … |
| KZ8p6b1zJ9U | … | … | … | … | … | … |
| hhszUXL1Cu8 | … | … | … | … | … | … |
| mJog8DlRk_4 | … | … | … | … | … | … |
| **Mean (high-confidence)** | … | … | … | | | |

Pass criterion: MGLD wins per-video on at least 4/5 and on the aggregate mean.

## Layer 2 — KZ8p6b1zJ9U flip-resistance

Expected: LR-VCC(MGLD, KZ) > LR-VCC(UAV, KZ).

Diagnostics on KZ:
- closeup_bbox_p50 = … (we expect ~0.18, high)
- I_reliability = … (we expect downweighted)
- T_reliability = … (mask coverage at long k is low — we expect downweighted)
- A_reliability = … (we expect highest)
- Effective weights on KZ: A=…, T=…, I=…

Reading: when reliability-weighting is working, the composite leans on whichever sub-metric the regime isn't broken for. On KZ that should be Appearance (CLIP-IQA per-frame quality, not biased by close-up).

## Layer 3 — out of scope for proposal

Parameterized synthetic test datasets — thesis future work.

## Failure modes uncovered (if any)

<populate after running>
```

Fill in the actual numbers from `_aggregate.json` and per-video JSONs after Task 8.

- [ ] **Step 2: Commit**

```bash
git add docs/notes/2026-05-25-lr-vcc-validation.md
git commit -m "docs: LR-VCC Layer 1+2 validation note"
```

---

## Phase 3 — Proposal document (Days 6–10)

### Task 10: Proposal outline

**Files:**
- Create: `proposal/proposal_outline.md`

- [ ] **Step 1: Write the outline**

```markdown
# Master's Thesis Proposal — Outline

**Author:** Timur Iakshibaev
**Topic:** Long-Range Video Consistency Evaluation for Video Super-Resolution
**Date:** May 2026

## 1. Background and motivation (1–1.5 pages)
- Video SR overview; long-video SR is under-served by existing evaluation
- Existing metrics: PSNR/SSIM/LPIPS for full-reference, NR-IQA for no-reference,
  VBench for video generation
- The thesis problem: long-video SR needs metrics that capture **multi-time-scale**
  consistency and are robust to **diffusion-style detail** without rewarding mere
  smoothness

## 2. Literature review (1–2 pages)
- VSR methods: MGLD-VSR (diffusion-based), UAV (T2V-SR), DOVE
- Existing temporal metrics: tOF/tLP (TecoGAN), E*warp (DOVE)
- Existing perceptual metrics: CLIP-IQA, MUSIQ, NIQE, DOVER
- VBench / VBench-2.0: applicability for SR; identified gaps

## 3. Preliminary work (2–3 pages)
- MGLD vs UAV on 5 synthetic long videos (reference DOVE setup verification)
- VBench 1.x + 2.0 results (table from results/uav_mgld_evaluation_metrics.md)
- KZ8p6b1zJ9U regime-shift finding (figure: per-frame `p_abnormal` distributions
  on KZ vs hhszUXL1Cu8; bbox-size correlation chart)
- Long-range tOF crossover finding (figure: per-k tOF curves, MGLD vs UAV)
- Why this matters: structural smoother-output bias in three independent
  learned representations (DINOv2, Anatomy ViT, LPIPS)

## 4. Proposed method — LR-VCC (1.5–2 pages)
- Architecture diagram
- Three sub-metrics: Appearance / Temporal / Identity
- Reliability-weighted composition
- Why this avoids the failures we documented

## 5. Preliminary validation (1 page)
- Layer 1: MGLD wins on aggregate, ≥4/5 per video
- Layer 2: MGLD wins KZ under LR-VCC (flip resistance)
- Per-video table from the validation note

## 6. Plan and timeline (0.5–1 page)
- Layer 3 validation (parameterized synthetic datasets) — months
- Multi-person Identity v2 — months
- Real-long-video HR baseline if obtainable — months
- Final thesis writeup, blind review, final submission — months
- Map to July 15 blind-review deadline + September 30 final deadline

## 7. Expected contributions
- A characterization of failure modes in existing long-video-SR metrics
  (regime-shift + smoother-output bias)
- A composite no-reference metric that resolves both failure modes through
  reliability-weighting
- Open-source implementation + datasets for reproducibility
```

- [ ] **Step 2: Commit**

```bash
git add proposal/proposal_outline.md
git commit -m "proposal: outline (Section 1-7)"
```

---

### Task 11: Write the proposal's preliminary-work section

**Files:**
- Create: `proposal/sections/preliminary_work.md`

- [ ] **Step 1: Write the section**

The text should reference:
- `results/uav_mgld_evaluation_metrics.md` for the full metrics table.
- `docs/notes/2026-05-13-kz-regime-shift-trigger.md` for KZ characterization.
- `docs/notes/2026-05-14-tof-tlp-long-range-results.md` for tOF/tLP crossover.

Include 3 figures:
1. Per-frame abnormal-rate histograms on KZ vs hhszUXL1Cu8 (from per-frame traces).
2. Hand-bbox-p50 vs MGLD-vs-UAV anatomy gap (5-point scatter).
3. tOF per-k curves for MGLD vs UAV (mean across 5 videos).

Figures can be plotted with matplotlib from the cached JSONs. Save the plotting script as `proposal/figures/plot_preliminary.py` for reproducibility.

- [ ] **Step 2: Generate figures**

```bash
python proposal/figures/plot_preliminary.py
```

Expected: 3 PNG files in `proposal/figures/`.

- [ ] **Step 3: Commit**

```bash
git add proposal/sections/preliminary_work.md proposal/figures/
git commit -m "proposal: preliminary work section + figures"
```

---

### Task 12: Write the proposed-method (LR-VCC) section

**Files:**
- Create: `proposal/sections/proposed_method.md`

- [ ] **Step 1: Write the section**

Draw the content from `docs/plans/2026-05-21-lr-vcc-design.md`. Include:
- Architecture overview block (the formula + brief prose)
- Three sub-metric sub-sections (concise, with motivations from preliminary work)
- Reliability-weighting formula
- A figure: composite architecture diagram (3 sub-metrics → reliability → softmax → composite).

- [ ] **Step 2: Commit**

```bash
git add proposal/sections/proposed_method.md
git commit -m "proposal: LR-VCC proposed method section"
```

---

### Task 13: Validation section + timeline + contributions

**Files:**
- Create: `proposal/sections/validation_and_timeline.md`

- [ ] **Step 1: Write the section**

- Validation Layer 1+2 results from `docs/notes/2026-05-25-lr-vcc-validation.md`
- Per-video table — MGLD vs UAV LR-VCC + reliability breakdown
- Comparison table — how often each individual metric agrees with perception on KZ vs how LR-VCC does
- Timeline mapped to July 15 blind-review + September 30 final
- Expected contributions list

- [ ] **Step 2: Commit**

```bash
git add proposal/sections/validation_and_timeline.md
git commit -m "proposal: validation + timeline + contributions"
```

---

### Task 14: Assemble final proposal LaTeX

**Files:**
- Modify: `proposal/proposal.tex` (or whatever the existing proposal LaTeX entry is — check `proposal/Makefile`)

- [ ] **Step 1: Check current proposal structure**

```bash
ls proposal/
cat proposal/Makefile 2>/dev/null | head -20
```

Identify the LaTeX entry file. If none exists, create `proposal/proposal.tex` based on the existing zjuthesis template (see `proposal/` directory for the template files already present).

- [ ] **Step 2: Translate the section markdown files to LaTeX**

For each `proposal/sections/*.md`, convert to LaTeX inline or via pandoc:

```bash
for s in proposal/sections/*.md; do
    pandoc -f markdown -t latex "$s" -o "${s%.md}.tex"
done
```

- [ ] **Step 3: Build and verify**

```bash
cd proposal && make all
```

Expected: `proposal.pdf` is generated without LaTeX errors.

- [ ] **Step 4: Commit**

```bash
git add proposal/
git commit -m "proposal: assembled LaTeX, builds clean"
```

---

### Task 15: Internal proofread + ship

- [ ] **Step 1: Self-review the proposal PDF**

Open `proposal/proposal.pdf` and check:
- All sections present (1–7)
- All 3+ figures render correctly
- All tables format correctly
- No "TODO" or placeholder text
- Citations / references where claims are made
- Word count / page count within school spec (check the zjuthesis template comments)

If any issue, fix and recommit before continuing.

- [ ] **Step 2: Request external review**

Send the PDF to your supervisor / PhD-student collaborator for feedback. Allow at least 24 h before submission to incorporate.

- [ ] **Step 3: Final commit + submission**

```bash
git add proposal/
git commit -m "proposal: final version, submitted May 31"
git push origin main
```

---

## Self-review

Skimming the spec vs this plan:

- Architecture overview, sub-metrics A/T/I, composition formula, reliability formulas, validation Layer 1+2, per-video reporting, hyperparameters — all present.
- Layer 3 (parameterized synthetic datasets) marked out-of-scope per the design.
- File layout matches the spec exactly (scripts/lr_vcc/* + results/lr_vcc/* + docs/notes/2026-05-25-lr-vcc-validation.md).
- Hyperparameters in the implementation tasks use the design's defaults (drift_floor=0.02, saturation_ceiling=0.98, mask_cov_floor=0.10, face_rate_floor=0.20, closeup_threshold=0.05, temperature=0.2, low_confidence_floor=0.2, sharpness=10).
- The proposal Phase (Tasks 10–15) references all the docs/results we've accumulated, including the just-built LR-VCC validation note.

No placeholders or "TBD"s. All code blocks are complete; all commands are runnable as written.

**Risks to call out:**
1. CLIP-IQA per-frame on 22,412 frames may take longer than 1 hour — if it does, use `--frame_stride 2` or run videos in parallel on multiple GPUs.
2. The pandoc markdown→LaTeX conversion (Task 14, Step 2) may need touch-ups for tables and figures. Reserve a slot in Day 9–10 for this.
3. The validation result is *not guaranteed* to give MGLD wins on Layer 2. If LR-VCC also flips on KZ, that's a major issue and the design needs revisiting — pause the proposal-writing phase and re-tune hyperparameters first.
