# Plan: Additional DOVE Metrics + VBench Long-Term Consistency Validation

**Date:** 2026-04-28
**Context:** Colleague requested (1) missing DOVE metrics: E\*warp, DOVER, FasterVQA, (2) analysis of VBench effectiveness for long-term consistency in SR, (3) test dataset designs to validate metrics.

---

## Part 1: Missing DOVE Metrics

### What we have vs. what's missing

| Metric | Type | What it measures | Direction | Status |
|--------|------|------------------|-----------|--------|
| PSNR | FR, per-frame | Pixel fidelity (RGB) | ↑ | Done (pyiqa + DOVE eval) |
| SSIM | FR, per-frame | Structural fidelity | ↑ | Done |
| LPIPS | FR, per-frame | Perceptual similarity | ↓ | Done |
| DISTS | FR, per-frame | Structure+texture similarity | ↓ | Done (DOVE eval) |
| CLIP-IQA | NR, per-frame | Perceptual quality (CLIP) | ↑ | Done |
| E\*warp | FR, per-pair | Temporal consistency (warping error) | ↓ | **Missing** |
| DOVER | NR, per-video | Aesthetic+technical quality | ↑ | **Missing** |
| FasterVQA | NR, per-video | Video quality (fragment sampling) | ↑ | **Missing** |

FR = full-reference, NR = no-reference.

**KVQ** is a separate benchmark/dataset, not a DOVE metric.

### Implementation plan

#### 1. E\*warp (temporal consistency)

Measures warping error between consecutive frames using optical flow:
- Estimate flow F_{t→t+1} with RAFT
- Warp frame t using F
- Compute masked L2 error vs frame t+1 (mask occlusions via forward-backward check)
- Average over all frame pairs

**Install:** RAFT is in torchvision. DOVE repo may have reference implementation.
**Long videos:** Pair-by-pair, constant memory — ideal for long videos.
**Priority: HIGH** — this is the only temporal consistency metric in DOVE.

#### 2. DOVER

No-reference video quality, disentangles aesthetic + technical quality.

```bash
pip install dover
```

```python
import dover
model = dover.DOVER(device="cuda")
score = model.evaluate("video.mp4")  # returns dict with aesthetic + technical scores
```

**Long videos:** Fragment-based sampling (~32 temporal fragments). Should handle >1 min. May need chunking for >5 min at 1280x720.
**Priority: MEDIUM**

#### 3. FasterVQA

Included in the `dover` package. Fragment-based video quality.

```python
from dover import DOVER
# FasterVQA is available as a sub-model within dover
```

**Priority: MEDIUM** — similar to DOVER, can run both together.

---

## Part 2: VBench Limitations for SR Long-Term Consistency

### Key findings from code analysis

#### What VBench does well for SR
- `imaging_quality`, `aesthetic_quality` — meaningful per-frame quality scores
- `motion_smoothness`, `dynamic_degree` — well-defined motion metrics
- Slow branch (within 2-second clips) captures short-range consistency

#### Critical limitations for long-range SR evaluation

1. **Fast branch uses only first frames.** For a 2-min video, the fast branch evaluates ~60 frames out of 2880 (2% sampling). Artifacts at non-first positions are invisible.

2. **No long-range temporal flickering.** Unlike subject/background consistency which have slow+fast branches, `temporal_flickering` has NO clip2clip branch — it only averages per-clip MAE scores. Gradual drift is undetectable.

3. **Adjacent-frame MAE only.** Temporal flickering computes MAE between frame[i] and frame[i+1]. Any artifact with period >2 frames is attenuated. Gradual drift produces near-zero per-step MAE.

4. **Scene detection hides chunk boundaries.** PySceneDetect (threshold=27.0) may split the video AT chunk-boundary artifacts, treating brightness jumps as scene changes. The metric actively hides the problem.

5. **50/50 slow/fast weighting is arbitrary.** Not calibrated for SR artifacts. Color drift (long-range) should weight fast branch higher; flickering (short-range) should weight slow branch.

6. **DINOv2 is color-invariant.** `subject_consistency` uses DINOv2 which is trained to be invariant to color — it will MISS gradual color drift.

7. **Quantile mapping miscalibrated for SR.** Score mapping tables are calibrated on text-to-video generation data, not SR. Score ranges may be distorted for SR artifacts.

### Recommendation: report slow/fast scores separately

VBench code produces both `inclip_score` and `clip2clip_score` in detailed results (in `fuse_inclip_clip2clip`). For SR evaluation, report these separately rather than fused — the fused score hides which timescale has the problem.

---

## Part 3: Synthetic Test Datasets for Validation

Design principle: each test video has ONE known, parameterized artifact so we can measure metric sensitivity.

### Test A: Gradual Color Drift

Take a clean 2-min video. Apply linear color temperature shift from 0 to K Kelvin across the duration.
- Severity levels: K = 0, 500, 1000, 2000, 4000
- Tests: Does `background_consistency` detect it? Does `subject_consistency` miss it (DINOv2 color invariance)?
- Expected: DreamSim (background) catches it; DINOv2 (subject) misses it

### Test B: Periodic Flickering at Different Frequencies

Add sinusoidal brightness oscillation at different frequencies.
- Frequencies: 12 Hz, 1 Hz, 0.1 Hz (10s period), 0.02 Hz (50s period)
- Amplitudes: 5, 10, 20, 40 pixel values
- Tests: Does `temporal_flickering` detect low-frequency flicker?
- Expected: Catches high-freq (>1 Hz), misses low-freq (<0.1 Hz) because per-frame MAE is tiny

### Test C: Chunk-Boundary Artifacts (VSR-specific)

Simulate per-chunk brightness offsets (random brightness per chunk of N frames).
- Chunk sizes: 16, 48, 96, 240 frames
- Offset std: 3, 8, 15 pixel values
- Tests: Does VBench detect discontinuities? Does scene detection hide them?

### Test D: Subject Identity Degradation

Gradually blend subject appearance toward a second identity using face morphing.
- Blend at final frame: 0%, 5%, 10%, 20%, 40%
- Tests: Does `subject_consistency` (DINOv2) catch gradual identity drift?

### Test E: Long-Range Background Inconsistency

Static-camera video with gradual background color grading change in second half only.
- Tests: Can the fast branch's first-frame sampling detect background changes?

**All tests can be generated on M1 Mac** — OpenCV/numpy only, no GPU needed.

---

## Part 4: Additional Temporal Metrics to Implement

### 1. Long-range tOF (highest priority)

Extend existing tOF to compute optical flow consistency at multiple temporal distances:
- k = [1, 5, 10, 30, 60, 120] frames apart
- For static camera: flow between distant frames should be near-zero
- Rising error with k indicates temporal drift

### 2. CLIP-IQA temporal trajectory

Already have per-frame CLIP-IQA. Compute variance over sliding windows:
- Window sizes: 24, 120, 600 frames (1s, 5s, 25s)
- High variance = temporal inconsistency
- Plot trajectory to visualize where quality drops

### 3. tLP (temporal LPIPS)

LPIPS between adjacent frames. More perceptually meaningful than pixel-level tOF for texture flicker detection.

---

## Execution Priority

1. **E\*warp** — implement using RAFT, run on existing MGLD-SR + LQ synthetic videos
2. **DOVER + FasterVQA** — `pip install dover`, run on synthetic videos
3. **Synthetic test suite** (Tests A-C) — generate on M1 Mac, run VBench + E\*warp on server
4. **Long-range tOF** — extend `src/evaluation/metrics.py`
5. **VBench separate slow/fast reporting** — modify evaluation scripts
