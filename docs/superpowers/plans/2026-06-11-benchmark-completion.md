# LR-VCC Benchmark Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the two reviewer-critical gaps in the LR-VCC benchmark — n=2 base videos and no real-model discrimination — plus ablations, before the July 1 experiment freeze.

**Architecture:** Three tracks. (1) Local metric-code changes (variance gate, reference curation, analysis scripts) developed TDD on the M1 and pushed; (2) server orchestration (generation + metric battery in tmux on GPU 0/7) following the standard pattern in `docs/onboarding.md` §7; (3) recompute-only analysis (composites, verdict matrix, ablations, sweeps) run locally from pulled JSONs.

**Tech Stack:** Python 3.9 local / conda `vsr`+`vbench` on server, OpenCV, Detectron2 (masks), open_clip (reference curation), pytest. Server connection details: `docs/private/server-setup.md` (gitignored — never copy IP/paths into committed files; `$SSH` below means the ssh command documented there).

**Spec:** `docs/superpowers/specs/2026-06-11-benchmark-completion-design.md`

**Out of scope for this plan:** the thesis writing track (Period=paper switch, chapter rewrites) — gets its own plan ~June 19.

**Runtime inputs:** `<NEW_ID_1..3>` denote the 3 new base-video IDs chosen in Task 4; every later occurrence is resolved by that task's output. Classmate model names resolve when their videos arrive (Task 8).

---

## Week 1 — scale + fixes

### Task 1: Variance gate for sub-metric I (clip-score dispersion)

The 7WHI pathology: slow-fast pooling on single-face content emits flat/inverted scores. Cheap reliable signal available in existing JSONs: the per-clip slow scores under `per_video[v]["clip_detail"]` flap wildly when face evidence is weak (e.g., 0.04, 0.87, 0.77, 0.05 on 7WHI). High dispersion ⇒ I abstains.

**Files:**
- Modify: `scripts/lr_vcc/identity.py`
- Test: `tests/lr_vcc/test_identity_gate.py` (new)

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for the clip-score-dispersion reliability gate on sub-metric I."""
from scripts.lr_vcc.identity import identity_score, clip_score_dispersion


def _pv(clip_scores, fused=0.6):
    n = len(clip_scores)
    return {
        "slow": fused, "fast": fused, "fused": fused,
        "n_clips": n, "n_clips_with_faces": n,
        "clip_detail": [
            {"clip_index": i, "clip_path": f"c{i}.mp4", "score": s}
            for i, s in enumerate(clip_scores)
        ],
    }


def test_dispersion_zero_for_constant_scores():
    assert clip_score_dispersion(_pv([0.7] * 6)) == 0.0


def test_dispersion_none_without_clip_detail():
    assert clip_score_dispersion({"fused": 0.6, "n_clips": 6, "n_clips_with_faces": 6}) is None


def test_dispersion_none_with_single_valid_clip():
    assert clip_score_dispersion(_pv([0.7])) is None


def test_invalid_scores_excluded_from_dispersion():
    pv = _pv([0.7, 0.7, 0.7])
    pv["clip_detail"].append({"clip_index": 3, "clip_path": "c3.mp4", "score": -1.0})
    assert clip_score_dispersion(pv) == 0.0


def test_low_dispersion_keeps_reliability():
    gated = identity_score(_pv([0.7] * 6))
    ungated = identity_score({"fused": 0.6, "n_clips": 6, "n_clips_with_faces": 6})
    assert abs(gated["reliability"] - ungated["reliability"]) < 1e-6


def test_high_dispersion_cuts_reliability():
    flappy = identity_score(_pv([0.05, 0.85, 0.77, 0.05, 0.75, 0.05]))
    stable = identity_score(_pv([0.7] * 6))
    assert flappy["reliability"] < 0.5 * stable["reliability"]
    assert flappy["details"]["clip_score_dispersion"] > 0.3
    assert flappy["details"]["dispersion_penalty"] > 0.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/lr_vcc/test_identity_gate.py -v`
Expected: FAIL — `ImportError: cannot import name 'clip_score_dispersion'`

- [ ] **Step 3: Implement the gate in `scripts/lr_vcc/identity.py`**

Add after the existing constants:

```python
_CLIP_DISPERSION_THRESHOLD = 0.25  # recalibrated by calibrate_identity_gate.py (Task 2)


def clip_score_dispersion(per_video: dict) -> Optional[float]:
    """Std-dev of valid per-clip slow scores; None when < 2 valid clips
    or clip_detail absent (older JSONs without --detail)."""
    detail = per_video.get("clip_detail") or []
    valid = [float(c["score"]) for c in detail if float(c.get("score", -1.0)) >= 0.0]
    if len(valid) < 2:
        return None
    m = sum(valid) / len(valid)
    return (sum((s - m) ** 2 for s in valid) / len(valid)) ** 0.5
```

In `identity_score`, replace the reliability line and extend details:

```python
    disp = clip_score_dispersion(per_video)
    disp_pen = 0.0 if disp is None else above_threshold_penalty(disp, _CLIP_DISPERSION_THRESHOLD)

    reliability = (1.0 - face_pen) * (1.0 - closeup_pen) * (1.0 - disp_pen)
```

and inside `"details"`:

```python
            "clip_score_dispersion": disp,
            "dispersion_penalty": disp_pen,
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/lr_vcc/ tests/synthetic_artefacts/ -v`
Expected: all PASS (new 6 + all prior — `identity_score` callers in `run_lr_vcc.py` are unaffected; old JSONs without `clip_detail` get `disp=None` → no penalty)

- [ ] **Step 5: Commit**

```bash
git add scripts/lr_vcc/identity.py tests/lr_vcc/test_identity_gate.py
git commit -m "lr_vcc: variance gate — sub-metric I abstains on flappy per-clip scores"
```

### Task 2: Calibrate the dispersion threshold on existing data

**Files:**
- Create: `scripts/lr_vcc/calibrate_identity_gate.py`
- Modify: `scripts/lr_vcc/identity.py:_CLIP_DISPERSION_THRESHOLD` (value only, if calibration disagrees with 0.25)

- [ ] **Step 1: Write the calibration script**

```python
"""Print per-video clip-score dispersion across all existing identity JSONs.

Goal: pick _CLIP_DISPERSION_THRESHOLD separating well-tracked multi-face videos
(hhsz: should stay reliable) from flappy single-face ones (7WHI: should abstain).

Usage: python scripts/lr_vcc/calibrate_identity_gate.py
"""
import glob
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.lr_vcc.identity import clip_score_dispersion

EVAL_DIR = REPO / "results" / "synthetic_artefacts_eval" / "identity"


def newest_json(artefact_dir):
    files = sorted(glob.glob(str(artefact_dir / "*.json")), key=os.path.getmtime)
    return files[-1] if files else None


def main():
    rows = []
    for artefact_dir in sorted(EVAL_DIR.iterdir()):
        path = newest_json(artefact_dir)
        if path is None:
            continue
        per_video = json.load(open(path))["per_video"]
        for vid, pv in per_video.items():
            disp = clip_score_dispersion(pv)
            if disp is None:
                continue
            base = "hhsz" if vid.startswith("hhsz") else ("7WHI" if vid.startswith("7WHI") else vid.split("_")[0])
            rows.append((base, artefact_dir.name, vid, disp))

    rows.sort(key=lambda r: r[3])
    print("| base | artefact | video | dispersion |")
    print("|---|---|---|---:|")
    for base, art, vid, disp in rows:
        print(f"| {base} | {art} | {vid} | {disp:.3f} |")

    by_base = {}
    for base, _, _, disp in rows:
        by_base.setdefault(base, []).append(disp)
    print()
    for base, ds in sorted(by_base.items()):
        ds.sort()
        print(f"{base}: n={len(ds)} min={ds[0]:.3f} median={ds[len(ds)//2]:.3f} max={ds[-1]:.3f}")
    if "hhsz" in by_base and "7WHI" in by_base:
        hi_ok = sorted(by_base["hhsz"])[int(0.9 * (len(by_base["hhsz"]) - 1))]
        lo_bad = sorted(by_base["7WHI"])[int(0.1 * (len(by_base["7WHI"]) - 1))]
        print(f"\nsuggested threshold (midpoint hhsz-p90={hi_ok:.3f}, 7WHI-p10={lo_bad:.3f}): {(hi_ok + lo_bad) / 2:.3f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it and read the suggestion**

Run: `python scripts/lr_vcc/calibrate_identity_gate.py`
Expected: a table + a `suggested threshold:` line. Caveat: if hhsz-p90 > 7WHI-p10 the populations overlap — then keep 0.25 and note the overlap in the validation note rather than forcing a separation.

- [ ] **Step 3: Update `_CLIP_DISPERSION_THRESHOLD` if suggestion differs from 0.25 by > 0.05; re-run `pytest tests/lr_vcc/ -v` (adjust the flappy-fixture margin only if the threshold moved a lot)**

- [ ] **Step 4: Recompute existing composites with the gate and check direction**

Run for both bases over `identity_drift` and `identity_degradation` (the two inverted/flat families) using the production CLI from `docs/onboarding.md` §7:

```bash
python -m scripts.lr_vcc.run_lr_vcc --temporal_weight uniform \
  --color_hist_alpha 0.394 --color_slope_beta 200 \
  --artefact identity_drift --out_tag gate_check
```

Expected: 7WHI identity reliability drops (weights shift away from I); hhsz Δ unchanged or stronger. If 7WHI *composite* Δ flips sign or hhsz degrades, stop and re-examine before proceeding.

- [ ] **Step 5: Commit**

```bash
git add scripts/lr_vcc/calibrate_identity_gate.py scripts/lr_vcc/identity.py
git commit -m "lr_vcc: calibrate dispersion gate threshold on existing artefact JSONs"
```

### Task 3: Curated reference-scene selector (CLIP distance)

**Files:**
- Create: `scripts/synthetic_artefacts/select_reference_scene.py`
- Test: `tests/synthetic_artefacts/test_select_reference_scene.py`

- [ ] **Step 1: Write the failing tests (pure functions only — no CLIP locally)**

```python
import numpy as np
import pytest

from scripts.synthetic_artefacts.select_reference_scene import cosine_distance, pick_most_distant


def test_cosine_distance_orthogonal_is_one():
    assert abs(cosine_distance(np.array([1.0, 0.0]), np.array([0.0, 1.0])) - 1.0) < 1e-6


def test_cosine_distance_identical_is_zero():
    v = np.array([0.3, 0.7, 0.1])
    assert cosine_distance(v, v) < 1e-6


def test_pick_most_distant_returns_farthest():
    base = np.array([1.0, 0.0])
    cands = {"near": np.array([0.9, 0.1]), "far": np.array([-1.0, 0.2])}
    name, dist = pick_most_distant(cands, base, tau=0.5)
    assert name == "far" and dist > 1.0


def test_pick_most_distant_raises_below_tau():
    base = np.array([1.0, 0.0])
    cands = {"near": np.array([0.99, 0.01])}
    with pytest.raises(ValueError):
        pick_most_distant(cands, base, tau=0.5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/synthetic_artefacts/test_select_reference_scene.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement the module**

```python
"""Pick a reference scene frame for background_drift that is CLIP-distant
from the base video — prevents the 7WHI inversion (reference too colour-similar).

Server usage (conda env vbench, needs open_clip):
    python scripts/synthetic_artefacts/select_reference_scene.py \
        --base_video results/mgld_synthetic_mp4/7WHI2L_FDNg.mp4 \
        --candidates results/mgld_synthetic_mp4/BrRLKMbBTYQ.mp4:500 \
                     results/mgld_synthetic_mp4/KZ8p6b1zJ9U.mp4:200 \
        --tau 0.25 --out results/synthetic_artefacts/_references/ref_bg_for_7WHI.png

A candidate is "video_path:frame_index". The base video must never be its own
candidate. Prints all distances, saves the most distant frame if it clears tau.
"""
import argparse

import cv2
import numpy as np


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(1.0 - float(np.dot(a, b)))


def pick_most_distant(candidates: dict, base_embedding: np.ndarray, tau: float):
    """candidates: {name: embedding}. Returns (name, distance) of the farthest
    candidate; raises ValueError when even the farthest is below tau."""
    best_name, best_d = None, -1.0
    for name, emb in candidates.items():
        d = cosine_distance(emb, base_embedding)
        if d > best_d:
            best_name, best_d = name, d
    if best_name is None or best_d < tau:
        raise ValueError(f"no candidate clears tau={tau}; best={best_name} d={best_d:.3f}")
    return best_name, best_d


def read_frame(video_path: str, frame_index: int) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read frame {frame_index} of {video_path}")
    return frame


def sample_frames(video_path: str, n: int = 8):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return [read_frame(video_path, i) for i in np.linspace(0, total - 1, n, dtype=int)]


def _embed_bgr_frames(frames, model, preprocess, device):
    import torch
    from PIL import Image
    with torch.no_grad():
        batch = torch.stack([
            preprocess(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))) for f in frames
        ]).to(device)
        emb = model.encode_image(batch).float().cpu().numpy()
    return emb.mean(axis=0)


def main():
    import open_clip
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--base_video", required=True)
    ap.add_argument("--candidates", nargs="+", required=True, help="video_path:frame_index")
    ap.add_argument("--tau", type=float, default=0.25)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k", device=device)

    base_emb = _embed_bgr_frames(sample_frames(args.base_video), model, preprocess, device)

    cand_embs, cand_frames = {}, {}
    for spec in args.candidates:
        path, idx = spec.rsplit(":", 1)
        frame = read_frame(path, int(idx))
        cand_frames[spec] = frame
        cand_embs[spec] = _embed_bgr_frames([frame], model, preprocess, device)

    for name, emb in sorted(cand_embs.items(), key=lambda kv: -cosine_distance(kv[1], base_emb)):
        print(f"{cosine_distance(emb, base_emb):.3f}  {name}")

    name, dist = pick_most_distant(cand_embs, base_emb, args.tau)
    cv2.imwrite(args.out, cand_frames[name])
    print(f"selected {name} (d={dist:.3f}) -> {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/synthetic_artefacts/test_select_reference_scene.py -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/synthetic_artefacts/select_reference_scene.py tests/synthetic_artefacts/test_select_reference_scene.py
git commit -m "synthetic_artefacts: CLIP-distance reference-scene selector"
```

### Task 4: Choose 3 new base videos (decision gate — needs Timur)

No code. Output: 3 video IDs + their roles, recorded in this plan file under "Runtime inputs resolved".

- [ ] **Step 1: List the candidate pool on the server**

```bash
$SSH "ls /data/disk2/timur/results/mgld_synthetic_mp4/; ls /data/disk2/timur/datasets/ | head -30"
```

plus the VBench long-video set location from `docs/private/server-setup.md`.

- [ ] **Step 2: Apply selection criteria** — duration > 1 min; target mix after adding 3: ≥3 single-face total (7WHI + 2 new), ≥3 multi-face total (hhsz + KZ/BrRLK/mJog as applicable), 1–2 scene-dominant/no-face. Check face count by eyeballing 3 sampled frames per candidate (`ffmpeg -ss ... -vframes 1`).

- [ ] **Step 3: Present the candidate table to Timur, get explicit confirmation of the 3 picks.** Promote KZ8p6b1zJ9U, BrRLKMbBTYQ, mJog8DlRk_4 from reference-source to base where they fit the mix — note: a video must never use itself as reference source (cross-assign in Task 5).

- [ ] **Step 4: Record the picks in this plan and commit**

```bash
git add docs/superpowers/plans/2026-06-11-benchmark-completion.md
git commit -m "plan: record 3 new base-video picks"
```

### Task 5: Wire new bases into the generator + prep server assets

**Files:**
- Modify: `scripts/synthetic_artefacts/generate_all.py:27` (`BASE_VIDEOS`) and the `REFERENCE_FACES` / `REFERENCE_BGS` / `HUMAN_MASKS` dicts (lines 37–56)
- Server: reference images + human masks for 6 new bases

- [ ] **Step 1: Extend the three dicts + BASE_VIDEOS.** Pattern per new base (cross-assignment — never self-reference):

```python
BASE_VIDEOS = ["hhszUXL1Cu8", "7WHI2L_FDNg", "KZ8p6b1zJ9U", "BrRLKMbBTYQ",
               "mJog8DlRk_4", "<NEW_ID_1>", "<NEW_ID_2>", "<NEW_ID_3>"]

REFERENCE_FACES["KZ8p6b1zJ9U"] = REPO / "results" / "synthetic_artefacts" / "_references" / "ref_face_for_KZ.png"
# ... one line per new base in each of the three dicts
```

- [ ] **Step 2: Run the existing generator tests** — `pytest tests/synthetic_artefacts/ -v` — expected: all PASS (dicts are data, loaders already raise clean FileNotFoundError for missing refs).

- [ ] **Step 3: Commit + push, pull on server**

```bash
git add scripts/synthetic_artefacts/generate_all.py
git commit -m "synthetic_artefacts: extend to 8 base videos"
git push
$SSH "cd /data/disk2/timur/thesis_ve && git pull"
```

- [ ] **Step 4: On server (env vbench, GPU 0): extract reference faces + curated reference backgrounds for the 6 new bases.** Faces: `extract_reference_face.py` per base from a *different* video. Backgrounds: the new `select_reference_scene.py` with `--tau 0.25`, candidates = frames 200/500/800 of every *other* base. Re-extract `ref_bg_for_7WHI.png` the same way (Fix 2 applied to the documented inversion).

- [ ] **Step 5: Precompute human masks for the 6 new bases** (env vbench — remember `pip install 'setuptools<81'` is already applied; verify with `python -c "import pkg_resources"`):

```bash
tmux new-session -d -s masks "bash -lc '
conda activate vbench
export CUDA_VISIBLE_DEVICES=0
cd /data/disk2/timur/thesis_ve
for v in KZ8p6b1zJ9U BrRLKMbBTYQ mJog8DlRk_4 <NEW_ID_1> <NEW_ID_2> <NEW_ID_3>; do
  python scripts/synthetic_artefacts/precompute_human_masks.py --video results/mgld_synthetic_mp4/\$v.mp4
done 2>&1 | tee /tmp/masks.log
touch /tmp/masks.done
'"
```

(Note the `bash -lc` wrapper — the bare-string form has the documented tee/touch quoting bug, `docs/onboarding.md` §7.)

- [ ] **Step 6: Check disk before generation** — `df -h /data/disk2`. If < 40 GB free, prune the raw-frame dirs (`results/mgld_synthetic` 23 GB, `results/uav_synthetic` 21 GB) **after confirming with Timur** — mp4 versions exist.

### Task 6: Generate 180 clips + run the metric battery (server)

- [ ] **Step 1: Generation (CPU-bound, tmux):**

```bash
tmux new-session -d -s gen8 "bash -lc '
conda activate vsr
cd /data/disk2/timur/thesis_ve
python scripts/synthetic_artefacts/generate_all.py 2>&1 | tee /tmp/gen8.log
touch /tmp/gen8.done
'"
```

Existing outputs are skipped (`generate_all.py` skip-if-exists), so this only produces the 6 new bases × 30.

- [ ] **Step 2: Spot-check 6 outputs (one per artefact) by pulling max-severity clips locally** into `inspection_videos/` and eyeballing — same procedure as the June 10 inspection round.

- [ ] **Step 3: Launch the 7-metric battery, split across GPUs** — model the runner on the server's `run_background_drift_eval.sh` (locations in `docs/private/server-setup.md`): GPU 0 takes 3 new bases, GPU 7 takes the other 3. Identity stage estimate: 90 clips/GPU ≈ 15 h → overnight. Use `bash -lc` tmux blocks with `.done` flags; poll next morning.

- [ ] **Step 4: Pull all result JSONs locally** (rsync per `docs/private/server-setup.md`) into the existing `results/synthetic_artefacts_eval/<stage>/<artefact>/` layout.

- [ ] **Step 5: Recompute composites for all 8 bases** with the production CLI (uniform tOF, α=0.394, β=200, gate active), `--out_tag v4_gate`.

### Task 7: Verdict matrix builder

**Files:**
- Create: `scripts/lr_vcc/build_verdict_matrix.py`
- Test: `tests/lr_vcc/test_build_verdict_matrix.py`

- [ ] **Step 1: Write the failing tests**

```python
import json

from scripts.lr_vcc.build_verdict_matrix import collect_deltas, verdict


def test_verdict_thresholds():
    assert verdict(-0.10) == "PASS"
    assert verdict(-0.03) == "WEAK"
    assert verdict(-0.01) == "FLAT"
    assert verdict(+0.01) == "FLAT"
    assert verdict(+0.05) == "INVERTED"


def test_collect_deltas(tmp_path):
    art = tmp_path / "background_drift"
    art.mkdir()
    for sev, score in [("0p02", 0.532), ("0p40", 0.256)]:
        (art / f"hhszUXL1Cu8_sev{sev}.json").write_text(
            json.dumps({"video": f"hhszUXL1Cu8_sev{sev}", "lr_vcc": score}))
    deltas = collect_deltas(tmp_path)
    assert abs(deltas[("background_drift", "hhszUXL1Cu8")] - (-0.276)) < 1e-9
```

- [ ] **Step 2: Run tests — expect FAIL (module missing)**

- [ ] **Step 3: Implement**

```python
"""Build the artefact × base verdict matrix from per-video composite JSONs.

Usage:
    python scripts/lr_vcc/build_verdict_matrix.py \
        --composites_dir results/lr_vcc/composite_artefacts_v4_gate \
        --out reports/figures/verdict_matrix.md
"""
import argparse
import json
import re
from pathlib import Path

_SEV_RE = re.compile(r"^(?P<base>.+)_sev(?P<sev>\d+p\d+)$")
_LO, _HI = "0p02", "0p40"


def verdict(delta: float) -> str:
    if delta <= -0.05:
        return "PASS"
    if delta <= -0.02:
        return "WEAK"
    if delta < +0.02:
        return "FLAT"
    return "INVERTED"


def collect_deltas(composites_dir) -> dict:
    """{(artefact, base): lr_vcc(sev 0.40) - lr_vcc(sev 0.02)}"""
    scores = {}
    for artefact_dir in Path(composites_dir).iterdir():
        if not artefact_dir.is_dir():
            continue
        for f in artefact_dir.glob("*.json"):
            m = _SEV_RE.match(f.stem)
            if not m:
                continue
            d = json.load(open(f))
            scores[(artefact_dir.name, m["base"], m["sev"])] = float(d["lr_vcc"])
    deltas = {}
    for (art, base, sev), v in scores.items():
        if sev == _HI and (art, base, _LO) in scores:
            deltas[(art, base)] = v - scores[(art, base, _LO)]
    return deltas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--composites_dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    deltas = collect_deltas(args.composites_dir)
    artefacts = sorted({a for a, _ in deltas})
    bases = sorted({b for _, b in deltas})

    lines = ["| artefact | " + " | ".join(bases) + " |",
             "|---|" + "---|" * len(bases)]
    for art in artefacts:
        cells = []
        for base in bases:
            d = deltas.get((art, base))
            cells.append("—" if d is None else f"{d:+.3f} {verdict(d)}")
        lines.append(f"| {art} | " + " | ".join(cells) + " |")

    n_pass = sum(1 for d in deltas.values() if verdict(d) in ("PASS", "WEAK"))
    lines.append(f"\nclean (PASS+WEAK): {n_pass}/{len(deltas)} conditions")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests — expect PASS; then run on the real v4_gate composites and read the matrix**

- [ ] **Step 5: Commit (script + tests + generated matrix); update the bi-weekly report's verdict table with the 6×8 result**

```bash
git add scripts/lr_vcc/build_verdict_matrix.py tests/lr_vcc/test_build_verdict_matrix.py reports/figures/verdict_matrix.md
git commit -m "lr_vcc: verdict-matrix builder + 6x8 matrix"
```

---

## Week 2 — real-model discrimination

### Task 8: LR-input verification + classmate package (do FIRST — longest external lead time)

- [ ] **Step 1: Verify LR inputs for the 5-video set exist on the server** (paths in `docs/private/server-setup.md`; they were the inputs to the MGLD/UAV runs). If absent: regenerate from HR sources with the same degradation pipeline used for the MGLD runs (documented in `docs/private/mgld-vsr-patches.md`) and verify one video round-trips to the same PSNR as the recorded MGLD baseline.

- [ ] **Step 2: Package LR inputs** — zip the 5 LR mp4s + a README.txt with the submission spec (outputs on these inputs, full duration, native fps, resolution ≥ 4× LR, CRF ≤ 18, filename `<model>_<base_id>.mp4`). Upload where classmates can reach (lab NAS / cloud drive — Timur sends the link himself).

- [ ] **Step 3: Draft the ask message for Timur to send** (plain text, short: what we need, the spec, soft deadline June 23 so Week-2 eval can include them).

### Task 9: RealESRGAN frame-wise anchor

- [ ] **Step 1: On server, check whether Real-ESRGAN is already cloned** (`ls /data/disk2/timur/repos/`). If not: clone + install into env `vsr` per its README (pin to a release tag).

- [ ] **Step 2: Run per-frame ×4 SR over the 5 LR videos** in tmux on GPU 7 (frame-wise → embarrassingly parallel, a few hours):

```bash
tmux new-session -d -s resr "bash -lc '
conda activate vsr
export CUDA_VISIBLE_DEVICES=7
cd /data/disk2/timur/repos/Real-ESRGAN
for v in hhszUXL1Cu8 7WHI2L_FDNg KZ8p6b1zJ9U BrRLKMbBTYQ mJog8DlRk_4; do
  python inference_realesrgan_video.py -n RealESRGAN_x4plus \
    -i /data/disk2/timur/<LR_DIR>/\$v.mp4 \
    -o /data/disk2/timur/results/realesrgan_mp4/
done 2>&1 | tee /tmp/resr.log
touch /tmp/resr.done
'"
```

(`<LR_DIR>` resolved by Task 8 Step 1.)

- [ ] **Step 3: Spot-check one output visually** (pull 1 clip locally).

### Task 10: Full-reference standard-metrics runner

**Files:**
- Create: `scripts/evaluation/run_full_reference.py`
- Test: `tests/evaluation/test_run_full_reference.py`

- [ ] **Step 1: Write the failing test**

```python
import json

import cv2
import numpy as np

from scripts.evaluation.run_full_reference import evaluate_pair


def _write_video(path, frames):
    h, w = frames[0].shape[:2]
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
    for f in frames:
        wr.write(f)
    wr.release()


def test_identical_videos_max_psnr(tmp_path):
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(8)]
    a, b = tmp_path / "a.mp4", tmp_path / "b.mp4"
    _write_video(a, frames)
    _write_video(b, frames)
    res = evaluate_pair(str(a), str(b))
    assert res["psnr"] >= 45.0          # codec-limited, not inf
    assert res["ssim"] > 0.97
    assert res["n_frames"] == 8


def test_different_videos_lower_psnr(tmp_path):
    rng = np.random.default_rng(0)
    fa = [rng.integers(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(8)]
    fb = [rng.integers(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(8)]
    a, b = tmp_path / "a.mp4", tmp_path / "b.mp4"
    _write_video(a, fa)
    _write_video(b, fb)
    res = evaluate_pair(str(a), str(b))
    assert res["psnr"] < 15.0
```

- [ ] **Step 2: Run — expect FAIL (module missing)**

- [ ] **Step 3: Implement**

```python
"""PSNR/SSIM (and on GPU: LPIPS) between SR output videos and GT HR videos.

Usage:
    python scripts/evaluation/run_full_reference.py \
        --pred_dir results/realesrgan_mp4 --gt_dir <HR_DIR> \
        --out results/evaluation/full_reference_realesrgan.json [--lpips]

Videos are paired by basename. Frames are read in lockstep; the shorter
stream ends the comparison. PSNR per frame is capped at 100 dB before
averaging (identical-frame inf guard).
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import sys
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.evaluation.metrics import psnr, ssim

_PSNR_CAP = 100.0


def evaluate_pair(pred_path: str, gt_path: str, use_lpips: bool = False) -> dict:
    cap_p, cap_g = cv2.VideoCapture(pred_path), cv2.VideoCapture(gt_path)
    psnrs, ssims, lpips_vals = [], [], []
    lpips_fn = None
    if use_lpips:
        import lpips as lpips_pkg
        import torch
        lpips_fn = lpips_pkg.LPIPS(net="alex").cuda()
    while True:
        ok_p, fp = cap_p.read()
        ok_g, fg = cap_g.read()
        if not (ok_p and ok_g):
            break
        if fp.shape != fg.shape:
            fp = cv2.resize(fp, (fg.shape[1], fg.shape[0]))
        psnrs.append(min(psnr(fp, fg), _PSNR_CAP))
        ssims.append(ssim(fp, fg))
        if lpips_fn is not None:
            import torch
            t = lambda x: torch.from_numpy(x[:, :, ::-1].copy()).permute(2, 0, 1)[None].float().cuda() / 127.5 - 1.0
            with torch.no_grad():
                lpips_vals.append(float(lpips_fn(t(fp), t(fg))))
    cap_p.release()
    cap_g.release()
    out = {
        "psnr": float(np.mean(psnrs)) if psnrs else -1.0,
        "ssim": float(np.mean(ssims)) if ssims else -1.0,
        "n_frames": len(psnrs),
    }
    if lpips_vals:
        out["lpips"] = float(np.mean(lpips_vals))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lpips", action="store_true")
    args = ap.parse_args()

    results = {}
    for pred in sorted(Path(args.pred_dir).glob("*.mp4")):
        gt = Path(args.gt_dir) / pred.name
        if not gt.is_file():
            print(f"SKIP {pred.name}: no GT")
            continue
        print(f"evaluating {pred.name} ...")
        results[pred.stem] = evaluate_pair(str(pred), str(gt), use_lpips=args.lpips)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests — expect 2 PASS**

- [ ] **Step 5: Commit**

```bash
git add scripts/evaluation/run_full_reference.py tests/evaluation/test_run_full_reference.py
git commit -m "evaluation: full-reference PSNR/SSIM/LPIPS video runner"
```

### Task 11: Metric battery + ranking table for all models

- [ ] **Step 1: On server, run the 7-metric battery + LR-VCC over `realesrgan_mp4/` (and classmate dirs as they arrive)** — same runner pattern as Task 6 Step 3, GPU 0, tmux, `.done` flags. ~100 min Identity for 5 videos per model.

- [ ] **Step 2: Run `run_full_reference.py --lpips` on server for every model dir vs the HR GT dir.**

- [ ] **Step 3: Pull JSONs; build the ranking table.** Create `scripts/evaluation/build_model_ranking.py`:

```python
"""Model ranking table: per-frame metrics vs LR-VCC, with rank-disagreement.

Usage:
    python scripts/evaluation/build_model_ranking.py \
        --full_ref results/evaluation/full_reference_<model>.json ... \
        --lr_vcc results/lr_vcc/models_<model>/aggregate.json ... \
        --out reports/figures/model_ranking.md

Each --full_ref / --lr_vcc argument is tagged with the model name taken from
the filename suffix. Emits one row per model (means over the 5 videos) and a
Spearman rank correlation between each per-frame metric column and LR-VCC.
"""
import argparse
import json
import re
from pathlib import Path

from scipy.stats import spearmanr


def model_tag(path, prefix):
    m = re.search(prefix + r"(.+?)\.json$", Path(path).name)
    return m.group(1) if m else Path(path).stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full_ref", nargs="+", required=True)
    ap.add_argument("--lr_vcc", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = {}
    for p in args.full_ref:
        d = json.load(open(p))
        vals = list(d.values())
        rows.setdefault(model_tag(p, "full_reference_"), {}).update({
            "psnr": sum(v["psnr"] for v in vals) / len(vals),
            "ssim": sum(v["ssim"] for v in vals) / len(vals),
            "lpips": sum(v.get("lpips", -1) for v in vals) / len(vals),
        })
    for p in args.lr_vcc:
        d = json.load(open(p))
        rows.setdefault(model_tag(p, "models_"), {})["lr_vcc"] = float(d["mean_lr_vcc"])

    models = sorted(rows)
    cols = ["psnr", "ssim", "lpips", "lr_vcc"]
    lines = ["| model | PSNR↑ | SSIM↑ | LPIPS↓ | LR-VCC↑ |", "|---|---:|---:|---:|---:|"]
    for m in models:
        r = rows[m]
        lines.append(f"| {m} | " + " | ".join(f"{r.get(c, float('nan')):.3f}" for c in cols) + " |")

    lr = [rows[m].get("lr_vcc") for m in models]
    for c in ("psnr", "ssim", "lpips"):
        col = [rows[m].get(c) for m in models]
        if None not in col and None not in lr and len(models) >= 3:
            rho, _ = spearmanr(col, lr)
            lines.append(f"\nSpearman({c}, lr_vcc) = {rho:+.2f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
```

(Check the actual aggregate-JSON key — `mean_lr_vcc` vs whatever `run_lr_vcc.py` writes — before running; adjust one line if needed.)

- [ ] **Step 4: Sanity-read the table.** Expected shape: RealESRGAN wins/ties PSNR-SSIM territory but lands last on LR-VCC; MGLD/UAV separate on LR-VCC. If LR-VCC ranks RealESRGAN *above* temporal methods, stop — investigate per-sub-metric before writing anything.

- [ ] **Step 5: Commit script + table; note the headline numbers in the bi-weekly report.**

### Task 12: Human anchor study (optional — only if classmates engaged)

- [ ] **Step 1: Build the pairing sheet** — 15–20 pairs sampled across (model A vs model B on same video) + (artefact sev 0.05 vs 0.40 on same base), shuffled, blind labels, as a CSV + folder of clip pairs.
- [ ] **Step 2: Collect preferences (Timur + 2–3 classmates), majority vote per pair.**
- [ ] **Step 3: Spearman between human preference rate and LR-VCC difference per pair** (`scipy.stats.spearmanr`, 10-line script next to the CSV). Record ρ in the report — even ρ≈0.6 with n=20 is a thesis-worthy sentence; report honestly whatever it is.

---

## Week 3 — ablations + freeze (no GPU; absorbs overruns)

### Task 13: Leave-one-out sub-metric ablation

**Files:**
- Create: `scripts/lr_vcc/ablation_loo.py`
- Test: `tests/lr_vcc/test_ablation_loo.py`

- [ ] **Step 1: Write the failing test**

```python
import json

from scripts.lr_vcc.ablation_loo import recompose_without


def test_dropping_low_reliability_submetric_barely_moves_score():
    subs = {
        "appearance": {"score": 0.8, "reliability": 0.9},
        "temporal": {"score": 0.9, "reliability": 0.9},
        "identity": {"score": 0.1, "reliability": 0.01},
    }
    full = recompose_without(subs, drop=None, temperature=0.2)
    no_id = recompose_without(subs, drop="identity", temperature=0.2)
    assert abs(full - no_id) < 0.05


def test_dropping_high_reliability_driver_moves_score():
    subs = {
        "appearance": {"score": 0.8, "reliability": 0.3},
        "temporal": {"score": 0.9, "reliability": 0.3},
        "color_slope": {"score": 0.05, "reliability": 0.95},
    }
    full = recompose_without(subs, drop=None, temperature=0.2)
    no_e = recompose_without(subs, drop="color_slope", temperature=0.2)
    assert no_e - full > 0.2
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement**

```python
"""Leave-one-out sub-metric ablation, recomputed from saved composite JSONs.

Usage:
    python scripts/lr_vcc/ablation_loo.py \
        --composites_dir results/lr_vcc/composite_artefacts_v4_gate \
        --out reports/figures/ablation_loo.md
"""
import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.lr_vcc.composite import compose_score

_SUBS = ["appearance", "temporal", "identity", "color_stability", "color_slope"]


def recompose_without(sub_metrics: dict, drop, temperature=0.2) -> float:
    keys = [k for k in sub_metrics if k != drop]
    scores = [float(sub_metrics[k]["score"]) for k in keys]
    rels = [float(sub_metrics[k]["reliability"]) for k in keys]
    return compose_score(scores, rels, temperature=temperature)["score"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--composites_dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # delta(artefact, base, drop) = recomposed(sev 0.40) - recomposed(sev 0.02)
    per = {}
    for f in Path(args.composites_dir).glob("*/*sev0p02.json"):
        hi = f.with_name(f.name.replace("sev0p02", "sev0p40"))
        if not hi.is_file():
            continue
        lo_d, hi_d = json.load(open(f)), json.load(open(hi))
        art = f.parent.name
        base = f.stem.replace("_sev0p02", "")
        for drop in [None] + _SUBS:
            d = (recompose_without(hi_d["sub_metrics"], drop)
                 - recompose_without(lo_d["sub_metrics"], drop))
            per[(art, base, drop)] = d

    arts = sorted({a for a, _, _ in per})
    bases = sorted({b for _, b, _ in per})
    lines = []
    for base in bases:
        lines.append(f"\n### base: {base}\n")
        lines.append("| artefact | full Δ | " + " | ".join(f"−{s}" for s in _SUBS) + " |")
        lines.append("|---|---:|" + "---:|" * len(_SUBS))
        for art in arts:
            if (art, base, None) not in per:
                continue
            cells = [f"{per[(art, base, None)]:+.3f}"]
            cells += [f"{per.get((art, base, s), float('nan')):+.3f}" for s in _SUBS]
            lines.append(f"| {art} | " + " | ".join(cells) + " |")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests — expect 2 PASS; run on real v4_gate composites.** Read: an artefact's |Δ| collapsing when sub-metric X is dropped ⇒ X uniquely catches it. Expected: background_drift collapses without color_slope; identity_drift (hhsz) collapses without identity.

- [ ] **Step 5: Commit script + tests + generated table.**

### Task 14: Sensitivity sweeps (τ, β, α)

**Files:**
- Create: `scripts/lr_vcc/sensitivity_sweep.py` (no unit tests — pure recompute glue over `recompose_without`; correctness is covered by Task 13's tests)

- [ ] **Step 1: Implement** — same JSON-walking skeleton as `ablation_loo.py`, but instead of dropping sub-metrics: (a) τ ∈ {0.1, 0.2, 0.5} via `recompose_without(subs, drop=None, temperature=tau)`; (b) β ∈ {100, 200, 400} by re-deriving `color_slope.score = exp(-beta * details["max_abs_slope"])` before recomposing (same re-derivation `run_lr_vcc.py:60-71` does); (c) α ∈ {0.3, 0.394, 0.5} by re-scoring the raw color-histogram JSONs in `results/lr_vcc/color_histogram/<artefact>/` through `color_stability_score(raw, alpha=...)`. Output: one markdown table per swept parameter — rows = (artefact, base), cols = parameter values, cells = Δ.

- [ ] **Step 2: Run; confirm verdicts (PASS/WEAK/FLAT/INVERTED per Task 7 thresholds) are stable at the production point's neighbours.** Any verdict that flips between τ=0.1 and τ=0.5 gets a sentence in the thesis limitations.

- [ ] **Step 3: Commit script + tables.**

### Task 15: Freeze + results inventory (July 1)

- [ ] **Step 1: Re-run the full local test suite** — `pytest tests/ -v` — expected: all PASS.
- [ ] **Step 2: Write `reports/figures/INVENTORY.md`** — one line per thesis-bound artefact: verdict matrix, model ranking, ablation table, sweep tables, human-study ρ (if done), with the generating command for each.
- [ ] **Step 3: Pull any remaining server JSONs; confirm `results/` mirrors everything thesis-bound (server disk is not archival).**
- [ ] **Step 4: Commit + push; tag the freeze: `git tag experiments-freeze-2026-07-01 && git push --tags`.**
- [ ] **Step 5: Update the bi-weekly report (June 5–18 successor) and start the writing-track plan.**

---

## Runtime inputs resolved

| Placeholder | Value | Resolved by |
|---|---|---|
| `<NEW_ID_1..3>` | KZ8p6b1zJ9U, BrRLKMbBTYQ, mJog8DlRk_4 (promotions — see note) | Task 4, 2026-06-11 |
| `<LR_DIR>` | _pending_ | Task 8 Step 1 |
| `<HR_DIR>` | _pending_ | Task 8 Step 1 |
| classmate models | _pending_ | Task 8 Step 3 |

**Task 4 outcome (2026-06-11):** the server has NO long-video pool beyond the original 5 —
the "8 bases" assumption was wrong. Decision (Timur): **5 bases total** — promote
KZ8p6b1zJ9U (cooking, intermittent faces, face_rate 0.51), BrRLKMbBTYQ (animated cartoon,
face_rate 0.20), mJog8DlRk_4 (lifestyle TV, scene cuts, face_rate 0.87). All Task 5/6
quantities scale accordingly: 3 new bases × 30 = 90 new clips, Identity ≈ 15 GPU-h total.
Single-face stays n=1 (7WHI) — reported as documented limitation, not category claim.
Disk decision: prune `results/mgld_synthetic` + `results/uav_synthetic` raw frames (~44 GB,
mp4s canonical). Reference cross-assignment: faces KZ←7WHI, BrRLK←hhsz, mJog←hhsz;
backgrounds via `select_reference_scene.py` (τ=0.25) for the 3 new bases + 7WHI re-extract
(Fix 2); hhsz background reference unchanged (it works — Δ −0.276).
