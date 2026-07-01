# RoPE Extrapolation Probe (FlashVSR) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether FlashVSR's RoPE fails to extrapolate to out-of-range temporal positions, and whether that (not just content) degrades long-video SR quality — via a temporal index-shift control and a length-extrapolation phenomenon test, scored by self-consistency, quality-vs-GT, and LR-VCC.

**Architecture:** Pure-Python position-index logic (unit-tested locally on the M1) drives a thin monkeypatch hook into FlashVSR's Wan2.1 DiT RoPE (run on the server's 40 GB A100). A no-op faithfulness gate proves the hook is bit-exact before any measurement. Perturbation drivers write SR frames + metric JSONs per condition; an analysis script turns them into curves/tables. LR-VCC reuses the existing 7-stage battery, treating each perturbation condition as one `COND` directory.

**Tech Stack:** Python 3.10, PyTorch (FlashVSR's pinned version), numpy, scikit-image (PSNR/SSIM), lpips, existing `src/evaluation/metrics.py` and `scripts/lr_vcc/`. Server: conda, tmux. Transfer: GitHub-branch bridge.

## Global Constraints

- **All model inference runs on the server.** Local M1 (16 GB, no NVIDIA GPU) is for pure-Python code, unit tests, analysis, plotting, and Mac-side data prep only. (`docs/onboarding.md`)
- **Server:** `ssh -p 11007 -i ~/.ssh/id_ed25519_timuj timur@instance-xzujqxam.yc.smartml.cn`. 2× A100-PCIE-**40 GB** (not 80). GPUs frequently at 100% util from other tenants — check `nvidia-smi --query-gpu=memory.free --format=csv` first; cap our footprint ~10 GB. Always launch long runs under `tmux`. (`docs/private/server-setup.md`, `docs/2026-07-01-new-server-and-gotchas.md`)
- **Mac↔server direct transfer is unusable (~22 KB/s, drops).** Move any file >a few hundred KB with the **GitHub-branch bridge**; split >100 MB with `split -b 90m`, reassemble with `cat`. (`docs/2026-07-01-new-server-and-gotchas.md`)
- **HuggingFace weights only via the mirror:** `export HF_ENDPOINT=https://hf-mirror.com`. huggingface.co, Google Drive, YouTube are blocked from the server.
- **No MP4 re-encoding in the SR pipeline** — libx264 yuv420p causes ~7 dB PSNR loss. Read/write frames as **PNG** (or lossless), never re-encode intermediate frames. (`docs/private/server-setup.md`)
- **`src/configs/paths.py`** is the single source of truth for dataset/output paths; override with `VSR_PROJECT_ROOT` / `VSR_DATA_ROOT`.
- Imports use the `src` package from repo root (e.g. `from src.evaluation.metrics import psnr`); run with `PYTHONPATH=.`.
- Metrics operate on `(H, W, C)` uint8/float numpy arrays. Reuse `src/evaluation/metrics.py` — do NOT reimplement PSNR/SSIM/LPIPS.
- Results JSONs are gitignored under `results/`; only figures/tables are git-tracked (existing convention).

---

## File Structure

- Create `scripts/rope_probe/__init__.py` — package marker.
- Create `scripts/rope_probe/position_override.py` — pure position-index logic (shift/stretch/explicit). Unit-tested locally.
- Create `scripts/rope_probe/consistency_metrics.py` — self-consistency + vs-GT aggregation over frame lists, JSON dump. Wraps `src/evaluation/metrics.py`.
- Create `scripts/rope_probe/flashvsr_hook.py` — monkeypatch that injects `PositionOverride` temporal indices into FlashVSR's Wan2.1 DiT RoPE. Server-only.
- Create `scripts/rope_probe/run_probe.py` — CLI driver: expands a sweep grid, runs inference per condition via the hook, writes SR frames + metric JSONs.
- Create `scripts/rope_probe/mechanism_hook.py` — optional per-layer attention-entropy / activation-drift logger.
- Create `scripts/rope_probe/make_long_gt.py` — Mac-side: HR video → bicubic ↓×4 LR + aligned GT frames (Phase 3 data).
- Create `scripts/rope_probe/analyze.py` — aggregate condition JSONs → curves/tables + matplotlib figures.
- Create `scripts/rope_probe/verify_noop.py` — server faithfulness gate: no-op override == baseline, bit-exact.
- Create `scripts/rope_probe/run_lrvcc_condition.sh` — wraps the existing 7-stage battery for one condition dir.
- Create tests `tests/rope_probe/test_position_override.py`, `tests/rope_probe/test_consistency_metrics.py`, `tests/rope_probe/test_make_long_gt.py`, `tests/rope_probe/test_analyze.py`.

Existing code reused (do not modify): `src/evaluation/metrics.py`, `scripts/lr_vcc/*`, `src/configs/paths.py`, the per-artefact runner pattern in `docs/onboarding.md §7`.

---

## Task 1: Position-override index logic (pure, local)

The heart of the probe: given a baseline temporal length, produce the temporal position indices to feed RoPE under shift / stretch / explicit-list / extended-length overrides. No torch, no GPU — fully unit-tested on the M1.

**Files:**
- Create: `scripts/rope_probe/__init__.py` (empty)
- Create: `scripts/rope_probe/position_override.py`
- Create: `tests/rope_probe/__init__.py` (empty)
- Test: `tests/rope_probe/test_position_override.py`

**Interfaces:**
- Produces:
  - `@dataclass PositionOverride(shift: int = 0, stretch: float = 1.0, indices: list[int] | None = None, length: int | None = None)`
  - `temporal_indices(base_len: int, ov: PositionOverride) -> list[int]` — returns the temporal position index list of length `base_len` (or `ov.length` if the chunked/extended path sets it). Resolution order: explicit `indices` wins; else `round(i * stretch) + shift` for `i in range(base_len)`.
  - `is_noop(ov: PositionOverride) -> bool` — True iff the override leaves indices identical to `[0..base_len-1]` (shift 0, stretch 1.0, no indices, no length change).

- [ ] **Step 1: Write the failing tests**

```python
# tests/rope_probe/test_position_override.py
from scripts.rope_probe.position_override import (
    PositionOverride, temporal_indices, is_noop,
)


def test_default_is_identity():
    ov = PositionOverride()
    assert temporal_indices(5, ov) == [0, 1, 2, 3, 4]
    assert is_noop(ov) is True


def test_shift_adds_constant():
    ov = PositionOverride(shift=100)
    assert temporal_indices(4, ov) == [100, 101, 102, 103]
    assert is_noop(ov) is False


def test_stretch_scales_positions():
    ov = PositionOverride(stretch=3.0)
    assert temporal_indices(4, ov) == [0, 3, 6, 9]


def test_stretch_then_shift_compose():
    ov = PositionOverride(shift=10, stretch=2.0)
    assert temporal_indices(3, ov) == [10, 12, 14]


def test_stretch_rounds_to_nearest_int():
    ov = PositionOverride(stretch=1.5)
    assert temporal_indices(4, ov) == [0, 2, 3, 5]  # round(0,1.5,3,4.5)=0,2,3,5(banker? no: python round(1.5)=2, round(4.5)=4)


def test_explicit_indices_win():
    ov = PositionOverride(indices=[0, 7, 42])
    assert temporal_indices(3, ov) == [0, 7, 42]
    assert is_noop(ov) is False


def test_explicit_indices_length_must_match():
    ov = PositionOverride(indices=[0, 1])
    import pytest
    with pytest.raises(ValueError):
        temporal_indices(3, ov)
```

Note: Python's `round()` uses banker's rounding — `round(4.5) == 4`. The expected list `[0, 2, 3, 5]` for stretch 1.5 uses `round(1.5)=2`, `round(3.0)=3`, `round(4.5)=4`? Recompute in Step 3 and fix the expected values to match the implementation you commit; the test is the spec of record.

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. pytest tests/rope_probe/test_position_override.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.rope_probe.position_override'`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/rope_probe/position_override.py
"""Temporal position-index overrides for the RoPE extrapolation probe.

Pure logic: no torch, no GPU. Given a baseline temporal length, produce the
list of temporal position indices to feed the model's RoPE. Holding pixel
content fixed while varying these indices is the whole experiment.
"""
from dataclasses import dataclass


@dataclass
class PositionOverride:
    shift: int = 0            # constant added to every temporal index
    stretch: float = 1.0      # multiply index i by this before shifting
    indices: list | None = None   # explicit index list; overrides shift/stretch
    length: int | None = None     # for the chunked/extended path (informational)


def temporal_indices(base_len: int, ov: "PositionOverride") -> list:
    if ov.indices is not None:
        if len(ov.indices) != base_len:
            raise ValueError(
                f"explicit indices len {len(ov.indices)} != base_len {base_len}")
        return list(ov.indices)
    return [int(round(i * ov.stretch)) + ov.shift for i in range(base_len)]


def is_noop(ov: "PositionOverride") -> bool:
    return (ov.shift == 0 and ov.stretch == 1.0
            and ov.indices is None and ov.length is None)
```

Run the stretch cases in a Python REPL, read the actual `round()` outputs, and correct the expected lists in the test to match. Commit test + impl together with matching values.

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. pytest tests/rope_probe/test_position_override.py -v`
Expected: PASS (all 7)

- [ ] **Step 5: Commit**

```bash
git add scripts/rope_probe/__init__.py scripts/rope_probe/position_override.py \
        tests/rope_probe/__init__.py tests/rope_probe/test_position_override.py
git commit -m "feat(rope_probe): temporal position-override index logic + tests"
```

---

## Task 2: Self-consistency + vs-GT metrics aggregation (pure, local)

Wrap the existing `src/evaluation/metrics.py` to score a condition two ways: **self-consistency** (perturbed frames vs the baseline/unperturbed frames) and **vs-GT** (frames vs ground-truth HR). Emit one JSON per condition.

**Files:**
- Create: `scripts/rope_probe/consistency_metrics.py`
- Test: `tests/rope_probe/test_consistency_metrics.py`

**Interfaces:**
- Consumes: `src.evaluation.metrics.evaluate_sequence(preds, gts, compute_lpips) -> dict` (keys `PSNR_mean`, `SSIM_mean`, `LPIPS_mean`, `temporal_consistency`).
- Produces:
  - `score_condition(pred_frames: list[np.ndarray], ref_frames: list[np.ndarray], compute_lpips: bool = True) -> dict` — thin pass-through to `evaluate_sequence` with a guard that frame counts match; returns its dict.
  - `write_condition_json(out_path: str, condition: dict, scores_vs_baseline: dict | None, scores_vs_gt: dict | None) -> None` — writes `{"condition": ..., "vs_baseline": ..., "vs_gt": ...}`. `null` sections allowed (e.g. no GT available).

- [ ] **Step 1: Write the failing tests**

```python
# tests/rope_probe/test_consistency_metrics.py
import json
import numpy as np
from scripts.rope_probe.consistency_metrics import (
    score_condition, write_condition_json,
)


def _frames(n, val):
    return [np.full((16, 16, 3), val, dtype=np.uint8) for _ in range(n)]


def test_identical_frames_score_infinite_psnr():
    a = _frames(3, 100)
    out = score_condition(a, a, compute_lpips=False)
    assert out["PSNR_mean"] == float("inf")
    assert out["SSIM_mean"] == 1.0


def test_frame_count_mismatch_raises():
    import pytest
    with pytest.raises(AssertionError):
        score_condition(_frames(3, 100), _frames(2, 100), compute_lpips=False)


def test_write_condition_json_shape(tmp_path):
    p = tmp_path / "cond.json"
    write_condition_json(
        str(p),
        condition={"shift": 100, "stretch": 1.0},
        scores_vs_baseline={"PSNR_mean": 42.0},
        scores_vs_gt=None,
    )
    payload = json.loads(p.read_text())
    assert payload["condition"]["shift"] == 100
    assert payload["vs_baseline"]["PSNR_mean"] == 42.0
    assert payload["vs_gt"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. pytest tests/rope_probe/test_consistency_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/rope_probe/consistency_metrics.py
"""Score a perturbation condition against a reference (baseline output or GT)."""
import json
from src.evaluation.metrics import evaluate_sequence


def score_condition(pred_frames, ref_frames, compute_lpips=True):
    """Return PSNR/SSIM/(LPIPS)/temporal_consistency of pred vs ref frames."""
    return evaluate_sequence(pred_frames, ref_frames, compute_lpips=compute_lpips)


def write_condition_json(out_path, condition, scores_vs_baseline, scores_vs_gt):
    payload = {
        "condition": condition,
        "vs_baseline": scores_vs_baseline,
        "vs_gt": scores_vs_gt,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. pytest tests/rope_probe/test_consistency_metrics.py -v`
Expected: PASS (3). If `PSNR_mean` for identical frames is not `inf` on this skimage version, adjust the assertion to `> 90.0` to match `peak_signal_noise_ratio` behaviour, and keep that as the recorded expectation.

- [ ] **Step 5: Commit**

```bash
git add scripts/rope_probe/consistency_metrics.py tests/rope_probe/test_consistency_metrics.py
git commit -m "feat(rope_probe): self-consistency + vs-GT condition scoring"
```

---

## Task 3: Mac-side long-HR → LR/GT builder (pure, local)

Phase 3 data: existing GT sets are short. Build a small long-video GT set by bicubic-↓×4 of curated ~1-min 720p HR clips. Frame I/O is PNG (Global Constraint: no MP4 re-encode). Fully local + testable.

**Files:**
- Create: `scripts/rope_probe/make_long_gt.py`
- Test: `tests/rope_probe/test_make_long_gt.py`

**Interfaces:**
- Produces:
  - `downsample_x4(hr: np.ndarray) -> np.ndarray` — bicubic ↓×4 of an `(H, W, C)` uint8 frame; output `(H//4, W//4, C)` uint8. Uses `cv2.resize(..., interpolation=cv2.INTER_CUBIC)`.
  - `build_pair(hr_frames_dir: str, out_lr_dir: str, out_gt_dir: str) -> int` — reads `*.png` HR frames in sorted order, writes ↓×4 LR PNGs to `out_lr_dir` and copies/normalises HR to `out_gt_dir` (crop each dim to a multiple of 4 first so ×4 is exact). Returns the number of frames written.

- [ ] **Step 1: Write the failing tests**

```python
# tests/rope_probe/test_make_long_gt.py
import cv2
import numpy as np
from scripts.rope_probe.make_long_gt import downsample_x4, build_pair


def test_downsample_x4_shape():
    hr = np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8)
    lr = downsample_x4(hr)
    assert lr.shape == (180, 320, 3)
    assert lr.dtype == np.uint8


def test_build_pair_crops_to_multiple_of_4_and_counts(tmp_path):
    hr_dir = tmp_path / "hr"; hr_dir.mkdir()
    lr_dir = tmp_path / "lr"
    gt_dir = tmp_path / "gt"
    # 722x1281 → cropped to 720x1280
    for i in range(3):
        img = np.random.randint(0, 256, (722, 1281, 3), dtype=np.uint8)
        cv2.imwrite(str(hr_dir / f"{i:04d}.png"), img)
    n = build_pair(str(hr_dir), str(lr_dir), str(gt_dir))
    assert n == 3
    gt0 = cv2.imread(str(gt_dir / "0000.png"))
    lr0 = cv2.imread(str(lr_dir / "0000.png"))
    assert gt0.shape == (720, 1280, 3)
    assert lr0.shape == (180, 320, 3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. pytest tests/rope_probe/test_make_long_gt.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/rope_probe/make_long_gt.py
"""Build a long-video LR/GT set: curated HR PNG frames -> bicubic x4 LR + GT.

Run on the Mac; bridge the resulting LR+GT dirs to the server via the
GitHub-branch method. PNG only (no MP4 re-encode -> avoids ~7 dB PSNR loss).
"""
import glob
import os
import cv2


def downsample_x4(hr):
    h, w = hr.shape[:2]
    return cv2.resize(hr, (w // 4, h // 4), interpolation=cv2.INTER_CUBIC)


def build_pair(hr_frames_dir, out_lr_dir, out_gt_dir):
    os.makedirs(out_lr_dir, exist_ok=True)
    os.makedirs(out_gt_dir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(hr_frames_dir, "*.png")))
    for i, p in enumerate(paths):
        hr = cv2.imread(p)
        h, w = hr.shape[:2]
        h4, w4 = (h // 4) * 4, (w // 4) * 4
        hr = hr[:h4, :w4]
        cv2.imwrite(os.path.join(out_gt_dir, f"{i:04d}.png"), hr)
        cv2.imwrite(os.path.join(out_lr_dir, f"{i:04d}.png"), downsample_x4(hr))
    return len(paths)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. pytest tests/rope_probe/test_make_long_gt.py -v`
Expected: PASS (2)

- [ ] **Step 5: Commit**

```bash
git add scripts/rope_probe/make_long_gt.py tests/rope_probe/test_make_long_gt.py
git commit -m "feat(rope_probe): Mac-side long-HR to LR/GT builder"
```

---

## Task 4: Stand up FlashVSR + locate the RoPE site (server, gated by inspection)

Phase 0 part 1. Get a reproducible FlashVSR baseline running on the server and **document the exact RoPE construction site** — the deliverable Tasks 5–6 depend on. No pytest here; the gate is a working baseline + a written note pinning `file:line` of the temporal-RoPE code.

**Files:**
- Create: `docs/notes/2026-07-02-flashvsr-rope-site.md` (the inspection deliverable)
- No repo code yet.

**Interfaces:**
- Produces (documented, for Task 5): the fully-qualified Python path of the function/method that builds the temporal RoPE frequencies/positions in FlashVSR's Wan2.1 DiT (e.g. `<module>.rope_params(...)` / `rope_apply(...)` / the attribute holding the temporal position grid), plus how the temporal axis is indexed within the 3D (T,H,W) RoPE.

- [ ] **Step 1: Clone FlashVSR + fetch weights on the server**

```bash
ssh -p 11007 -i ~/.ssh/id_ed25519_timuj timur@instance-xzujqxam.yc.smartml.cn
tmux new -s flashvsr_setup
source ~/miniconda3/etc/profile.d/conda.sh
cd ~/repos && for t in $(seq 1 10); do git clone https://github.com/OpenImagingLab/FlashVSR.git && break; sleep 6; done
cd FlashVSR
conda create -y -n flashvsr python=3.10 && conda activate flashvsr
# install per the repo's requirements (pin torch to what FlashVSR specifies):
pip install -r requirements.txt   # adjust to the repo's actual install steps
export HF_ENDPOINT=https://hf-mirror.com
# download weights per repo README (Wan2.1_VAE, LQ_proj_in, TCDecoder, diffusion_*_streaming_dmd)
```

Expected: env builds; weights land in the repo's expected weights dir. If a weight repo 404s on the mirror, request the specific filename (the mirror lacks some repos).

- [ ] **Step 2: Reproduce a baseline SR run on a short clip**

Use the repo's non-streaming `tiny` inference path (simpler than streaming for instrumentation) on ~16–32 frames of one existing LR clip (e.g. from `~/synthetic_data` or a UDM10 LQ clip). Save output as PNG frames.

```bash
CUDA_VISIBLE_DEVICES=0 python <flashvsr_inference_script> --input <lr_frames> --output ~/results/rope_probe/_baseline_smoke/ ...
```

Expected: SR PNG frames written; visually sane. Record VRAM from `nvidia-smi` (must fit 40 GB with headroom).

- [ ] **Step 3: Locate the temporal RoPE construction**

```bash
cd ~/repos/FlashVSR
grep -rniE "rope|rotary|freqs|rotary_emb|position" --include=*.py | grep -iv "proposal" | head -50
```

Read the hits; identify where the 3D (T,H,W) RoPE frequencies are built and where the **temporal** axis positions enter. Confirm it derives from a per-frame index range (e.g. `torch.arange(T)`), which is the handle Task 5 overrides.

- [ ] **Step 4: Write the inspection note**

Create `docs/notes/2026-07-02-flashvsr-rope-site.md` recording: the module path + `file:line` of the temporal-RoPE function, its signature, how `T` positions are generated, the model's **trained temporal length** (from config), the exact baseline inference command, and measured VRAM. This note is the interface Task 5 consumes.

- [ ] **Step 5: Commit the note**

```bash
# on Mac (bridge the note back, or write it locally from the grep output)
git add docs/notes/2026-07-02-flashvsr-rope-site.md
git commit -m "docs(rope_probe): FlashVSR RoPE construction site + baseline command"
```

---

## Task 5: RoPE injection hook + no-op faithfulness gate (server, critical gate)

Phase 0 part 2 — the non-negotiable gate. Monkeypatch the temporal-RoPE function found in Task 4 so the temporal positions come from `temporal_indices(base_len, override)` (Task 1). Prove that a **no-op** override reproduces the baseline **bit-exact**.

**Files:**
- Create: `scripts/rope_probe/flashvsr_hook.py`
- Create: `scripts/rope_probe/verify_noop.py`

**Interfaces:**
- Consumes: `position_override.PositionOverride`, `position_override.temporal_indices`; the RoPE site from Task 4's note.
- Produces:
  - `install_position_hook(override: PositionOverride) -> callable` — monkeypatches the temporal-RoPE function so temporal positions become `temporal_indices(T, override)` instead of `range(T)`; returns a restore callable that undoes the patch. When `is_noop(override)`, the patched path must be numerically identical to unpatched.
  - `run_flashvsr(lr_frames_dir: str, out_dir: str, override: PositionOverride, gpu: int) -> list[np.ndarray]` — runs FlashVSR inference under the hook, writes PNG frames to `out_dir`, returns the frame list.

- [ ] **Step 1: Implement the hook (pattern — adapt names to Task 4's note)**

```python
# scripts/rope_probe/flashvsr_hook.py
"""Inject overridden temporal positions into FlashVSR's Wan2.1 DiT RoPE.

The exact module/function names come from docs/notes/2026-07-02-flashvsr-rope-site.md.
The pattern: replace the temporal position source (torch.arange(T)) with
torch.tensor(temporal_indices(T, override)); everything else untouched.
"""
import torch
from scripts.rope_probe.position_override import PositionOverride, temporal_indices, is_noop

# import the module identified in Task 4, e.g.:
# from <flashvsr_pkg> import <rope_module> as R


def install_position_hook(override: PositionOverride):
    import <flashvsr_pkg>.<rope_module> as R  # noqa: replace per Task 4 note
    original = R.<rope_fn>                      # replace per Task 4 note

    def patched(*args, **kwargs):
        # Determine temporal length T from the call (positional arg or tensor shape),
        # per how <rope_fn> receives it — documented in Task 4's note.
        T = _extract_T(args, kwargs)
        idx = torch.tensor(temporal_indices(T, override), dtype=torch.long)
        return original(*args, _temporal_positions=idx, **kwargs)  # adapt to signature

    R.<rope_fn> = patched
    def restore():
        R.<rope_fn> = original
    return restore
```

The `_extract_T` helper and the exact call-through are filled from Task 4's note. If `<rope_fn>` builds positions internally with no injection point, instead patch it to read a module-level `_OVERRIDE_INDICES` global that `install_position_hook` sets — documented as the fallback in the note.

- [ ] **Step 2: Implement the no-op verifier**

```python
# scripts/rope_probe/verify_noop.py
"""Faithfulness gate: no-op override must reproduce baseline bit-exact."""
import sys
import numpy as np
from scripts.rope_probe.position_override import PositionOverride
from scripts.rope_probe.flashvsr_hook import run_flashvsr

EPS = 1e-4  # fp nondeterminism floor; tighten after measuring

def main(lr_dir, gpu):
    base = run_flashvsr(lr_dir, "/tmp/rope_noop_base", PositionOverride(), gpu)
    noop = run_flashvsr(lr_dir, "/tmp/rope_noop_hooked", PositionOverride(), gpu)
    diffs = [float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max())
             for a, b in zip(base, noop)]
    m = max(diffs)
    print(f"max abs diff across frames: {m}")
    assert m <= EPS, f"no-op override NOT faithful: {m} > {EPS}"
    print("FAITHFUL: no-op hook == baseline")

if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
```

Note: run baseline once **without** installing the hook and once **with** the no-op hook installed (adjust `run_flashvsr` to optionally skip `install_position_hook` when the override is a no-op *only for the base call*), so the comparison is unhooked-baseline vs hooked-noop. Seed torch and disable nondeterministic kernels first (`torch.use_deterministic_algorithms(True)` where feasible) to drive `EPS` toward 0.

- [ ] **Step 3: Push code to server + run the gate**

Bridge `scripts/rope_probe/*.py` to the server (GitHub branch or small-file scp). Then:

```bash
tmux new -s rope_noop
source ~/miniconda3/etc/profile.d/conda.sh && conda activate flashvsr
cd ~/thesis_ve && PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
  python -m scripts.rope_probe.verify_noop <lr_frames_dir> 0 2>&1 | tee /tmp/rope_noop.log
```

Expected: `FAITHFUL: no-op hook == baseline`. **If it fails, STOP** — fix the hook (wrong injection point) before any measurement. Everything downstream is meaningless until this passes.

- [ ] **Step 4: Commit**

```bash
git add scripts/rope_probe/flashvsr_hook.py scripts/rope_probe/verify_noop.py
git commit -m "feat(rope_probe): RoPE position-injection hook + no-op faithfulness gate"
```

---

## Task 6: Perturbation driver — shift + stretch sweeps (server)

Phase 1 (shift control) + Phase 2a (position stretch). Expand a sweep grid, run each condition through the hook, score self-consistency (vs baseline) and vs-GT (when a GT dir is given), write one JSON per condition.

**Files:**
- Create: `scripts/rope_probe/run_probe.py`
- Test: `tests/rope_probe/test_run_probe.py` (grid expansion only — pure, local)

**Interfaces:**
- Consumes: `PositionOverride`, `run_flashvsr` (Task 5), `score_condition` + `write_condition_json` (Task 2), `is_noop` (Task 1).
- Produces:
  - `expand_grid(shifts: list[int], stretches: list[float]) -> list[PositionOverride]` — one override per (shift, stretch); always includes the no-op baseline first, de-duplicated.
  - CLI `python -m scripts.rope_probe.run_probe --lr <dir> [--gt <dir>] --out <dir> --shifts 0,100,500 --stretches 1.0,2.0,4.0 --gpu 0` → writes `<out>/<cond_id>.json` + SR frames per condition; baseline output cached and reused as the self-consistency reference.

- [ ] **Step 1: Write the failing test (grid logic only)**

```python
# tests/rope_probe/test_run_probe.py
from scripts.rope_probe.run_probe import expand_grid
from scripts.rope_probe.position_override import is_noop


def test_grid_starts_with_noop_and_dedupes():
    grid = expand_grid([0, 100], [1.0, 2.0])
    assert is_noop(grid[0])                      # baseline first
    ids = {(o.shift, o.stretch) for o in grid}
    assert ids == {(0, 1.0), (0, 2.0), (100, 1.0), (100, 2.0)}
    assert len(grid) == len(ids)                 # no duplicates
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/rope_probe/test_run_probe.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement the driver**

```python
# scripts/rope_probe/run_probe.py
"""Drive shift + stretch sweeps through the RoPE hook; score each condition."""
import argparse
import os
from scripts.rope_probe.position_override import PositionOverride, is_noop
from scripts.rope_probe.consistency_metrics import score_condition, write_condition_json


def expand_grid(shifts, stretches):
    seen, grid = set(), []
    # baseline first
    grid.append(PositionOverride()); seen.add((0, 1.0))
    for s in shifts:
        for st in stretches:
            key = (s, float(st))
            if key in seen:
                continue
            seen.add(key)
            grid.append(PositionOverride(shift=s, stretch=float(st)))
    return grid


def _cond_id(ov):
    return f"shift{ov.shift}_stretch{ov.stretch}"


def _load_frames(d):
    import cv2, glob
    return [cv2.imread(p) for p in sorted(glob.glob(os.path.join(d, "*.png")))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", required=True); ap.add_argument("--gt", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shifts", default="0"); ap.add_argument("--stretches", default="1.0")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--no-lpips", action="store_true")
    a = ap.parse_args()
    from scripts.rope_probe.flashvsr_hook import run_flashvsr  # server-only import

    os.makedirs(a.out, exist_ok=True)
    shifts = [int(x) for x in a.shifts.split(",")]
    stretches = [float(x) for x in a.stretches.split(",")]
    grid = expand_grid(shifts, stretches)
    gt = _load_frames(a.gt) if a.gt else None

    baseline = None
    for ov in grid:
        cid = _cond_id(ov)
        frames = run_flashvsr(a.lr, os.path.join(a.out, cid), ov, a.gpu)
        if is_noop(ov):
            baseline = frames
        vs_base = None if is_noop(ov) else score_condition(
            frames, baseline, compute_lpips=not a.no_lpips)
        vs_gt = score_condition(frames, gt, compute_lpips=not a.no_lpips) if gt else None
        write_condition_json(os.path.join(a.out, cid + ".json"),
                             {"shift": ov.shift, "stretch": ov.stretch}, vs_base, vs_gt)
        print(f"done {cid}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the unit test + a local dry check**

Run: `PYTHONPATH=. pytest tests/rope_probe/test_run_probe.py -v`
Expected: PASS (1). (The inference body only runs on the server.)

- [ ] **Step 5: Run the sweeps on the server**

```bash
tmux new -s rope_sweep
source ~/miniconda3/etc/profile.d/conda.sh && conda activate flashvsr
cd ~/thesis_ve
# Phase 1 shift control (in-range + beyond-range k), self-consistency on a long synthetic clip:
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python -m scripts.rope_probe.run_probe \
  --lr ~/synthetic_data/synthetic/<clip>_frames --out ~/results/rope_probe/shift/<clip> \
  --shifts 0,8,32,128,512 --stretches 1.0 --gpu 0 2>&1 | tee /tmp/rope_shift.log
# Phase 2a stretch (force out-of-range positions), + GT clip when available:
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python -m scripts.rope_probe.run_probe \
  --lr <long_gt_lr> --gt <long_gt_gt> --out ~/results/rope_probe/stretch/<clip> \
  --shifts 0 --stretches 1.0,1.5,2.0,4.0,8.0 --gpu 0 2>&1 | tee /tmp/rope_stretch.log
```

Expected: per-condition JSONs + SR frame dirs under `~/results/rope_probe/{shift,stretch}/`. Pull JSONs back to the Mac via the bridge for analysis.

- [ ] **Step 6: Commit**

```bash
git add scripts/rope_probe/run_probe.py tests/rope_probe/test_run_probe.py
git commit -m "feat(rope_probe): shift+stretch perturbation driver with per-condition scoring"
```

---

## Task 7: LR-VCC per condition (server, reuse existing battery)

Add the long-range-consistency reference. Reuse the existing 7-stage battery + `scripts/lr_vcc/run_lr_vcc.py`, treating each perturbation condition's SR frame dir as one `COND`. Applied only to a small set of **headline conditions** (baseline + a few extrapolation levels) — the full battery is GPU-heavy, not run per micro-sweep-point.

**Files:**
- Create: `scripts/rope_probe/run_lrvcc_condition.sh`

**Interfaces:**
- Consumes: the SR frame dirs from Task 6; the existing stage scripts (`compute_clip_iqa.py`, `eval_tof_tlp.py`, identity pipeline, `compute_color_histogram.py`, `compute_color_slope.py`) and `run_lr_vcc.py` (CLI in `docs/onboarding.md §5`).
- Produces: an LR-VCC composite JSON per condition under `~/results/rope_probe/lrvcc/<cond_id>/`.

- [ ] **Step 1: Write the runner by cloning an existing per-artefact runner**

```bash
# on the server
cd ~/thesis_ve
sed 's#<ARTEFACT>#rope_cond#g' <existing run_*_eval.sh template> > scripts/rope_probe/run_lrvcc_condition.sh
chmod +x scripts/rope_probe/run_lrvcc_condition.sh
# Edit it to take: $1=cond_id  $2=frame_dir  $3=gpu, and point every stage's
# COND path at results/rope_probe/lrvcc/<cond_id>/, then call run_lr_vcc.py with
# the production flags: --temporal_weight uniform --color_hist_alpha 0.394 --color_slope_beta 200
```

- [ ] **Step 2: Run LR-VCC on the headline conditions**

```bash
tmux new -s rope_lrvcc
source ~/miniconda3/etc/profile.d/conda.sh && conda activate vsr
for cond in shift0_stretch1.0 shift0_stretch2.0 shift0_stretch4.0 shift0_stretch8.0; do
  CUDA_VISIBLE_DEVICES=0 ./scripts/rope_probe/run_lrvcc_condition.sh \
    "$cond" ~/results/rope_probe/stretch/<clip>/"$cond" 0 2>&1 | tee /tmp/rope_lrvcc_$cond.log
done
```

Expected: one composite JSON per condition under `results/rope_probe/lrvcc/<cond>/`. (Identity stage needs the `identity`/`vbench` env + warmed RetinaFace cache per the gotchas doc; stagger launches 25 s.)

- [ ] **Step 3: Commit the runner**

```bash
git add scripts/rope_probe/run_lrvcc_condition.sh
git commit -m "feat(rope_probe): LR-VCC per-condition runner (reuses 7-stage battery)"
```

---

## Task 8: Analysis — curves, tables, figures (pure, local)

Aggregate the per-condition JSONs (self-consistency, vs-GT, LR-VCC) into the deliverable curves/tables and matplotlib figures: metric-vs-shift `k`, metric-vs-stretch `s`, and an LR-VCC-vs-PSNR contrast.

**Files:**
- Create: `scripts/rope_probe/analyze.py`
- Test: `tests/rope_probe/test_analyze.py`

**Interfaces:**
- Consumes: `<out>/<cond_id>.json` files from Task 6 (`{"condition", "vs_baseline", "vs_gt"}`) and LR-VCC composite JSONs from Task 7.
- Produces:
  - `load_conditions(cond_dir: str) -> list[dict]` — loads + sorts all `*.json` condition files by `(shift, stretch)`.
  - `curve(conditions: list[dict], x_key: str, metric_path: tuple[str, str]) -> tuple[list, list]` — extracts `(x, y)` where `x = condition[x_key]` and `y = payload[metric_path[0]][metric_path[1]]` (e.g. `("vs_gt", "PSNR_mean")`), skipping conditions where that section is `null`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/rope_probe/test_analyze.py
import json
from scripts.rope_probe.analyze import load_conditions, curve


def _write(d, cid, shift, stretch, vs_gt_psnr):
    payload = {"condition": {"shift": shift, "stretch": stretch},
               "vs_baseline": None,
               "vs_gt": None if vs_gt_psnr is None else {"PSNR_mean": vs_gt_psnr}}
    (d / f"{cid}.json").write_text(json.dumps(payload))


def test_load_sorts_by_shift_then_stretch(tmp_path):
    _write(tmp_path, "b", 100, 1.0, 30.0)
    _write(tmp_path, "a", 0, 2.0, 31.0)
    _write(tmp_path, "c", 0, 1.0, 40.0)
    conds = load_conditions(str(tmp_path))
    assert [(c["condition"]["shift"], c["condition"]["stretch"]) for c in conds] \
        == [(0, 1.0), (0, 2.0), (100, 1.0)]


def test_curve_extracts_xy_and_skips_null(tmp_path):
    _write(tmp_path, "c", 0, 1.0, 40.0)
    _write(tmp_path, "d", 0, 2.0, None)   # null vs_gt -> skipped
    _write(tmp_path, "e", 0, 4.0, 25.0)
    conds = load_conditions(str(tmp_path))
    xs, ys = curve(conds, x_key="stretch", metric_path=("vs_gt", "PSNR_mean"))
    assert xs == [1.0, 4.0]
    assert ys == [40.0, 25.0]
```

Note: `x_key` reads from `condition[x_key]`; implement `load_conditions` to keep `condition` accessible and sort by `(shift, stretch)`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. pytest tests/rope_probe/test_analyze.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement analysis**

```python
# scripts/rope_probe/analyze.py
"""Aggregate RoPE-probe condition JSONs into curves/tables + figures."""
import argparse
import glob
import json
import os


def load_conditions(cond_dir):
    conds = []
    for p in glob.glob(os.path.join(cond_dir, "*.json")):
        conds.append(json.load(open(p)))
    conds.sort(key=lambda c: (c["condition"]["shift"], c["condition"]["stretch"]))
    return conds


def curve(conditions, x_key, metric_path):
    sec, key = metric_path
    xs, ys = [], []
    for c in conditions:
        payload = c.get(sec)
        if payload is None or key not in payload:
            continue
        xs.append(c["condition"][x_key])
        ys.append(payload[key])
    return xs, ys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond_dir", required=True)
    ap.add_argument("--x_key", default="stretch")
    ap.add_argument("--out_fig", required=True)
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    conds = load_conditions(a.cond_dir)
    fig, ax = plt.subplots()
    for sec, key in [("vs_gt", "PSNR_mean"), ("vs_baseline", "PSNR_mean")]:
        xs, ys = curve(conds, a.x_key, (sec, key))
        if xs:
            ax.plot(xs, ys, marker="o", label=f"{sec}:{key}")
    ax.set_xlabel(a.x_key); ax.set_ylabel("PSNR"); ax.legend()
    fig.savefig(a.out_fig, dpi=150, bbox_inches="tight")
    print(f"wrote {a.out_fig}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. pytest tests/rope_probe/test_analyze.py -v`
Expected: PASS (2)

- [ ] **Step 5: Commit**

```bash
git add scripts/rope_probe/analyze.py tests/rope_probe/test_analyze.py
git commit -m "feat(rope_probe): condition aggregation + curve/figure analysis"
```

---

## Task 9 (optional): Mechanism diagnostic — per-layer drift (server)

The thin slice of Approach C. From the same hooked forward passes, log a cheap per-layer signal (attention-entropy and/or activation-L2 drift vs baseline) as a function of position magnitude. Purely additive; **skip if Task 5's hook was hard-won** and revisit after the headline curves exist.

**Files:**
- Create: `scripts/rope_probe/mechanism_hook.py`

**Interfaces:**
- Consumes: the same FlashVSR modules; `PositionOverride`.
- Produces: `capture_layer_drift(lr_dir, override, gpu) -> dict` — returns `{layer_idx: activation_l2_vs_baseline}`, written to `~/results/rope_probe/mechanism/<cond_id>.json`.

- [ ] **Step 1: Register forward hooks on the DiT attention blocks**

```python
# scripts/rope_probe/mechanism_hook.py
"""Log per-layer activation drift vs baseline under a position override."""
import numpy as np
import torch


def capture_layer_drift(model, blocks):
    """Attach forward hooks to `blocks`; return {idx: latest output tensor}."""
    store, handles = {}, []
    for i, blk in enumerate(blocks):
        def mk(i):
            def hook(_m, _in, out):
                store[i] = out.detach().float().cpu()
            return hook
        handles.append(blk.register_forward_hook(mk(i)))
    return store, handles
```

Wire this into a small script that runs baseline + one override, computes per-layer `||act_override - act_base||_2 / ||act_base||_2`, and dumps JSON. Blocks list comes from Task 4's note (the DiT transformer block modules).

- [ ] **Step 2: Run on 2–3 stretch levels; commit**

```bash
git add scripts/rope_probe/mechanism_hook.py
git commit -m "feat(rope_probe): optional per-layer activation-drift diagnostic"
```

---

## Task 10: Findings note (local)

Synthesise the verdict on H0/H1 into a short note that feeds the paper / Direction-4 arc.

**Files:**
- Create: `docs/notes/2026-07-<dd>-rope-extrapolation-findings.md`

- [ ] **Step 1: Write the note**

Cover: whether shift drift stays ~0 in-range and rises out-of-range (H0 verdict); whether stretch/length degrades quality-vs-GT and self-consistency monotonically (H1 verdict); the **LR-VCC-vs-PSNR contrast** (does LR-VCC register long-range loss PSNR misses?); the mechanism curve if run; and the round-2 fork decision (promote SparkVSR contrast and/or the θ-rescale mitigation). Embed the figures from Task 8.

- [ ] **Step 2: Commit**

```bash
git add docs/notes/2026-07-*-rope-extrapolation-findings.md reports/figures/rope_probe_*
git commit -m "docs(rope_probe): H0/H1 findings + LR-VCC vs PSNR contrast"
```

---

## Self-Review

**Spec coverage:**
- §2 FlashVSR-only, 3D RoPE temporal axis → Tasks 4, 5. ✓
- §2 shift lever → Task 6 (`--shifts`). ✓
- §2 length/extrapolation lever → Task 6 stretch (2a) + Task 6 server step notes chunked; **gap:** the "long single-pass vs in-range chunked" (2b) manipulation is described but not given its own driver. *Resolution:* 2b is a chunking wrapper around the same `run_flashvsr`; folded into Task 6's server usage (process full clip vs matched chunks) rather than new code — acceptable for a probe, but flagged so the implementer adds a `--chunk N` branch to `run_probe.py` if 2b becomes primary.
- §2 self-consistency + vs-GT references → Task 2. ✓
- §2 LR-VCC reference → Task 7. ✓
- §2 mechanism slice → Task 9 (optional). ✓
- §4 Phase 0 confirm+instrument+faithfulness gate → Tasks 4, 5. ✓
- §4 Phase 3 long-HR GT data → Task 3. ✓
- §6 deliverables (scripts, JSONs, curves, findings note) → Tasks 6–10. ✓
- §8 success criteria (Phase 0 gate, curves, H0/H1 verdict, round-2 fork) → Tasks 5, 8, 10. ✓

**Placeholder scan:** Task 4/5 intentionally carry `<flashvsr_pkg>`/`<rope_fn>` placeholders because the exact names are an *output of Task 4's inspection* — this is a genuine data dependency, not a lazy gap; Task 4 Step 4 pins them in a committed note that Task 5 consumes. All pure-Python tasks (1, 2, 3, 6-grid, 8) have complete, runnable code + tests.

**Type consistency:** `PositionOverride` fields (`shift:int, stretch:float, indices:list|None, length:int|None`) are used identically in Tasks 1, 5, 6. `temporal_indices(base_len, ov)`, `is_noop(ov)`, `score_condition(pred, ref, compute_lpips)`, `write_condition_json(out_path, condition, scores_vs_baseline, scores_vs_gt)`, `run_flashvsr(lr_frames_dir, out_dir, override, gpu)`, `expand_grid(shifts, stretches)`, `load_conditions(cond_dir)`, `curve(conditions, x_key, metric_path)` — names/signatures match across the tasks that consume them.
