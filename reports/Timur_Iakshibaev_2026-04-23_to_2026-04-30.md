# Progress Report — April 23–30, 2026

## Summary

This week: completed VBench 2.0 Quality Score evaluation, added DOVER + E\*warp metrics, finished UAV synthetic super-resolution on all 5 videos, and produced full MGLD vs UAV comparison on synthetic long videos.

---

## Baselines: MGLD-VSR vs UAV on Synthetic Long Videos

### Dataset
5 synthetic long videos, 320×180 LQ → 1280×720 SR (4× upscale), 22,412 frames total.

| Video | Frames | Duration |
|-------|--------|----------|
| 7WHI2L_FDNg | 5,000 | 208s |
| BrRLKMbBTYQ | 5,000 | 208s |
| KZ8p6b1zJ9U | 5,000 | 208s |
| hhszUXL1Cu8 | 2,412 | 100s |
| mJog8DlRk_4 | 5,000 | 208s |

### No-Reference Image Quality (pyiqa, mean over all videos)

| Method | CLIP-IQA ↑ | MUSIQ ↑ | NIQE ↓ | BRISQUE ↓ |
|--------|-----------|---------|--------|-----------|
| MGLD-SR | **0.496** | **65.07** | **4.67** | **24.74** |
| UAV | 0.391 | 56.28 | 5.73 | 50.90 |

MGLD-SR wins all 4 NR metrics. UAV's higher BRISQUE (50.9 vs 24.7) and NIQE (5.7 vs 4.7) suggest more visible artifacts.

**Effectiveness notes:**
- **CLIP-IQA** — uses CLIP features, correlates well with human perception of overall quality but can be fooled by stylized content (gives high scores to artistic but blurry images). Per-frame metric.
- **MUSIQ** — multi-scale image quality transformer, strong for natural image quality but trained on photographic content; may give unreliable scores on synthetic/animated content.
- **NIQE** — fully blind statistical metric based on natural scene statistics. Sensitive to noise and blur but not perceptual quality. Can give similar scores to clean blurry frames and clean sharp ones if both follow natural statistics.
- **BRISQUE** — similar to NIQE but trained with human MOS. More aligned with perception but trained primarily on JPEG/Gaussian distortions; less reliable for diffusion artifacts.
- **All four are per-frame** — no temporal awareness. They cannot detect flickering, drift, or chunk-boundary artifacts.

### DOVER Video Quality (no-reference, per-video, mean over all 5)

| Method | Aesthetic ↑ | Technical ↑ | Overall ↑ |
|--------|-------------|-------------|-----------|
| LQ baseline | 49.57 | 2.06 | 10.44 |
| **MGLD-SR** | **99.71** | **9.87** | **73.81** |
| UAV | 99.42 | 8.69 | 65.06 |

MGLD-SR achieves highest overall DOVER quality (~7× over LQ, ~13% over UAV).

**Effectiveness notes:**
- **DOVER** disentangles aesthetic and technical quality with separate predictors. Strong correlation with human MOS (SRCC ~0.85+ on UGC datasets).
- Uses fragment-based sampling — robust to long videos, but only ~32 spatial-temporal fragments evaluated. Local artifacts may be missed.
- Aesthetic score saturates near 100 for any sharp content — not very discriminative for SR comparison; the **technical score** is more informative (UAV 8.69 vs MGLD 9.87 means MGLD is technically slightly better).
- DOVER was trained on user-generated content, not super-resolution outputs. Its "aesthetic" component may overweight stylistic preferences over fidelity.
- Per-video metric, not per-frame, so cannot localize artifacts.

### E\*warp Temporal Consistency (RAFT optical flow, mean)

| Method | E\*warp ↓ |
|--------|-----------|
| LQ | **0.0092** |
| MGLD-SR | 0.0114 |
| UAV | 0.0137 |

LQ has best temporal consistency (no SR processing). MGLD-SR is more temporally consistent than UAV — UAV adds 49% more inconsistency vs LQ, MGLD only 24%.

SR adds slight temporal inconsistency (+24%) versus LQ, expected for diffusion-based super-resolution.

**Effectiveness notes:**
- **E\*warp** estimates optical flow with RAFT, warps frame *t* to *t+1*, computes masked L2 error in non-occluded regions.
- **Strong**: directly measures temporal consistency on a frame-pair basis. Constant memory regardless of video length — ideal for long videos.
- **Weakness 1**: only adjacent frames. Doesn't capture long-range drift (e.g., color shift over 1 minute would not register if each pair-wise change is small).
- **Weakness 2**: depends on RAFT flow quality. Noisy or wrong flow on textureless regions → spurious error.
- **Weakness 3**: forward-backward consistency mask filters occlusions but also may filter genuine SR artifacts that "look like" occlusions.
- **Comparison with VBench's `temporal_flickering`**: VBench uses pixel MAE without flow warping — doesn't separate motion from temporal artifacts. E\*warp is more principled but more expensive.

### VBench 2.0 Quality Score (long-video mode, mean over 5 videos)

All 7 Quality Score dimensions:

| Dimension | LQ | MGLD-SR | UAV |
|-----------|-----|---------|------|
| imaging_quality ↑ | 0.4388 | **0.6810** | 0.6458 |
| motion_smoothness ↑ | 0.9873 | **0.9886** | 0.9882 |
| temporal_flickering ↑ | 0.9811 | **0.9840** | 0.9826 |
| aesthetic_quality ↑ | 0.4128 | **0.5080** | 0.4892 |
| dynamic_degree ↑ | 0.5628 | 0.5942 | 0.5393 |
| subject_consistency ↑ | 0.8936 | 0.8927 | **0.9031** |
| background_consistency ↑ | **0.9333** | 0.9235 | 0.9317 |

MGLD-SR wins 5/7 dimensions (imaging, motion, flickering, aesthetic, dynamic). UAV wins subject_consistency (which uses DINOv2 — SR's diffusion noise less detectable to color-invariant features). LQ wins background_consistency (no SR processing → no frame-to-frame DreamSim variation).

**Effectiveness notes per dimension:**
- **imaging_quality** — uses MUSIQ. Same caveats as MUSIQ above. Largest discriminator between LQ and SR (+55%). Reliable for SR.
- **motion_smoothness** — measures flow consistency between adjacent frames. Saturates near 1.0 for any reasonable video; differences <0.001 are not meaningful.
- **temporal_flickering** — pixel MAE between adjacent frames after static-frame filtering. Saturates near 1.0; gradual drift produces near-zero MAE per step → undetectable.
- **aesthetic_quality** — LAION aesthetic predictor (CLIP-based). Discriminates well (+23% MGLD vs LQ). Reliable.
- **dynamic_degree** — measures motion magnitude using RAFT. Higher = more motion. Not a quality metric per se; useful for content analysis. Slight increase for SR (+6%) likely due to amplified motion artifacts.
- **subject_consistency** — DINOv2 feature cosine similarity across frames. **Limitation**: DINOv2 is trained to be color-invariant, so it would miss color drift artifacts.
- **background_consistency** — DreamSim feature similarity. More sensitive to color/style changes than subject_consistency. Slow-fast branches: slow within 2-second clips, fast across clip first-frames (~2% sample rate).

### VBench Semantic Score — Not Applicable for SR

All 9 Semantic Score dims (`overall_consistency`, `appearance_style`, `temporal_style`, `human_action`, `color`, `object_class`, `multiple_objects`, `spatial_relationship`, `scene`) require text prompts as ground truth. VBench was designed for text-to-video generation benchmarking. For SR evaluation without prompts, these scores are meaningless.

**Why each fails for SR:**
- **overall_consistency** uses ViCLIP for video↔text similarity; with filename as "prompt" → score 0.08 (random).
- **appearance_style, temporal_style** require `auxiliary_info` with style labels in JSON — error if missing.
- **human_action** parses expected action from filename (e.g., `person is running-001.mp4`); our filenames are video IDs → 0.0 detection rate.
- **color, object_class, multiple_objects, spatial_relationship** use GRiT/detectron2 to detect objects, then compare against text prompts.
- **scene** uses Tag2Text to classify scenes, compares to text prompts.

These dimensions could potentially be repurposed for SR if we generated synthetic prompts via captioning models, but VBench's calibration tables are built for text-to-video, not SR.

---

## Infrastructure Improvements

### New metrics added (server)
- **DOVER** v1.0.0 — installed from `VQAssessment/DOVER`, weights SCP'd
- **E\*warp** — custom implementation using torchvision RAFT (DOVE repo had stub but no code)
- **VBench Quality Score** — all 7 dims working with patches

### UAV chunked inference
UAV original implementation OOM'd on 5000-frame videos. Implemented chunked inference (2500 frames per chunk) to handle long videos. All 22,412 frames super-resolved over ~48 hours.

### Server fixes
- DINO cache torch.hub git init for offline use
- detectron2 0.6 installed via `--no-build-isolation`
- timm downgraded 1.0.26 → 1.0.12 (UMT compatibility)
- Tag2Text 4.2 GB SCP'd
- Torchvision RAFT 21 MB SCP'd
- DOVER 229 MB SCP'd

---

## VBench Effectiveness Validation Plan (next step)

### Motivation

We use VBench Quality Score as the main perceptual benchmark for our SR results, and it shows clear improvements (imaging_quality +55%, aesthetic_quality +23%). But before relying on these numbers in the thesis, we need to verify that VBench actually captures the artifacts that matter for SR — particularly **long-range temporal artifacts** that are the hardest part of long-video super-resolution.

### Identified limitations (from code analysis)

After reading VBench 2.0 source code at `vbench/` and `vbench2_beta_long/`:

1. **`temporal_flickering` has no long-range branch.** Source: `vbench2_beta_long/temporal_flickering.py` — only computes MAE between frame *i* and *i+1*. Unlike `subject_consistency` and `background_consistency`, there is NO `clip2clip` branch that compares across clips. **Consequence**: gradual drift over a 2-minute video produces near-zero MAE per step but massive cumulative drift — completely undetectable.

2. **Fast branch uses only first frames.** Source: `utils.py:create_video_from_first_frames()`. For a 2-minute video at 24fps split into 2-second clips, fast branch evaluates 60 frames out of 2880 (2.1% sampling). **Consequence**: artifacts at non-first positions invisible; periodic artifacts have phase-dependent visibility.

3. **DINOv2 (subject_consistency) is color-invariant by design.** Trained with color jitter augmentation; cosine similarity in DINOv2 feature space is largely invariant to white-balance / hue shifts. **Consequence**: color drift in faces or objects produces near-zero subject_consistency change.

4. **PySceneDetect can hide chunk-boundary artifacts.** `ContentDetector(threshold=27.0)` runs before clip splitting. If a VSR model produces a brightness jump at a chunk boundary, scene detection may split the video AT that jump — treating it as a scene change. The artifact then falls between clips and is filtered out.

5. **Score mapping calibrated for text-to-video, not SR.** Configs `subject_mapping_table.yaml`, `background_mapping_table.yaml` quantile-map clip2clip scores onto inclip score distribution. Calibration is from VBench's video generation benchmark — its score distribution differs from SR.

6. **Cosine similarity blind to certain degradations.** `subject_consistency` and `background_consistency` use `F.cosine_similarity`. Two frames with identical angular feature direction but different magnitudes (e.g., different sharpness) get the same score.

7. **Slow/fast 50:50 fusion is arbitrary.** From `slow_fast_params.yaml`. Not validated for SR-specific artifacts. For chunk-boundary flicker (short-range), slow branch should dominate; for color drift (long-range), fast branch should dominate. Fixed fusion conflates them.

### Validation approach

**Methodology:** generate synthetic videos from a clean source video, inject a single known, parameterized artifact at varying severity, and plot VBench score vs severity. A reliable metric should produce a monotonic curve; a blind metric will be flat.

**Source videos:** use one of our 5 synthetic test videos (e.g., hhszUXL1Cu8, ~100s). Apply the artifact post-hoc — no GT or model training needed.

**All test data buildable on M1 Mac** with OpenCV/numpy — no GPU needed for data generation. Only evaluation needs server GPU.

### Test A: Gradual Color Drift

**Construction.** Linear color temperature shift across video duration:
```python
for i, frame in enumerate(frames):
    t = i / (n - 1)  # 0 → 1
    r_gain = 1.0 + t * max_kelvin / 10000
    b_gain = 1.0 - t * max_kelvin / 10000
    frame[:,:,2] *= r_gain  # R (BGR)
    frame[:,:,0] *= b_gain  # B
```

**Severity levels:** max_kelvin ∈ {0, 500, 1000, 2000, 4000} (0 = clean control).

**Hypotheses to test:**
- H1: `background_consistency` (DreamSim) detects drift, score decreases monotonically.
- H2: `subject_consistency` (DINOv2) MISSES drift due to color invariance (flat curve).
- H3: `temporal_flickering` MISSES drift (per-frame MAE near zero).
- H4: `imaging_quality` (MUSIQ) is unaffected (per-frame, not temporal).

**Expected outcome:** identifies which VBench metrics are usable for color drift detection.

### Test B: Periodic Flickering at Different Frequencies

**Construction.** Sinusoidal brightness modulation:
```python
offset = amplitude * sin(2π * frequency_hz * t)
frame = clip(frame + offset, 0, 255)
```

**Severity levels:**
- Frequencies: 12 Hz (within-frame), 1 Hz, 0.1 Hz (10s period), 0.02 Hz (50s period)
- Amplitudes: 5, 10, 20, 40 pixel values
- Cross product: 16 conditions

**Hypotheses to test:**
- H1: `temporal_flickering` detects high-freq (≥1 Hz) flicker — score decreases with amplitude.
- H2: `temporal_flickering` MISSES low-freq (≤0.1 Hz) — per-step MAE near zero.
- H3: Phase-alignment matters: fast-branch first-frame sampling makes detection rate depend on whether sampling lines up with peaks/troughs.

**Expected outcome:** quantifies the "blindness frequency" — below which VBench becomes unreliable.

### Test C: Chunk-Boundary Artifacts (VSR-specific)

**Construction.** Per-chunk random brightness offset (simulates real VSR chunk-boundary discontinuities):
```python
chunk_idx = i // chunk_size
np.random.seed(chunk_idx)
offset = np.random.normal(0, std)
```

**Severity levels:**
- Chunk sizes: 16, 48, 96, 240 frames (matching common VSR window sizes)
- Brightness std: 3, 8, 15 pixel values

**Hypotheses to test:**
- H1: `temporal_flickering` detects boundary jumps if clip splitting is misaligned with chunks.
- H2: `temporal_flickering` MISSES jumps if PySceneDetect splits exactly at boundaries (treating jumps as scene changes).
- H3: A modified eval that disables scene detection should be more reliable.

**Expected outcome:** validates whether VBench can be trusted for diffusion VSR chunk artifacts.

### Test D: Subject Identity Degradation

**Construction.** Gradual face morphing toward a different identity:
- Detect face bounding box per frame
- Apply progressive blend with target identity (alpha ramps 0 → max_alpha)

**Severity levels:** max_alpha ∈ {0, 0.05, 0.10, 0.20, 0.40}.

**Hypotheses to test:**
- H1: `subject_consistency` detects identity morphing (DINOv2 encodes semantic identity).
- H2: DINOv2 fast branch may miss it if first frames of each clip are pre-degradation.
- H3: Slow branch within 2-second clips may miss gradual change too small per clip.

**Expected outcome:** validates VBench for identity preservation in face/character VSR.

### Test E: Long-Range Background Inconsistency

**Construction.** Static-camera video with foreground/background segmentation. First half: original. Second half: gradually apply different color grading to background while keeping foreground unchanged.

**Severity levels:** background drift max_kelvin ∈ {0, 500, 1000, 2000, 4000} applied only in second half.

**Hypotheses to test:**
- H1: `background_consistency` slow branch MISSES it (each 2-second clip looks consistent).
- H2: `background_consistency` fast branch DETECTS it (first-frame across clips shows the shift).
- H3: Reporting fused score vs separate slow/fast tells different stories.

**Expected outcome:** validates the slow-fast architecture and motivates reporting branches separately.

### Reporting recommendations

Based on results, we will recommend that for SR evaluation:
- Report **separate slow and fast branch scores** (they're already computed in detailed JSON; just need to surface them).
- Disable PySceneDetect or use a much higher threshold for SR videos (chunk boundaries shouldn't be treated as scene changes).
- Augment with **non-VBench metrics** that capture long-range drift: long-range tOF, CLIP-IQA temporal variance.

### Additional metrics to implement (complement VBench)

| Metric | Implementation | What it captures |
|--------|----------------|------------------|
| Long-range tOF | Extend `metrics.py:tof()` to k = [1, 5, 10, 30, 60, 120] | Long-range temporal drift |
| tLP | LPIPS between frame *t* and *t+1* | Perceptual flicker (vs pixel MAE) |
| CLIP-IQA temporal variance | Var of per-frame CLIP-IQA over sliding windows (1s, 5s, 25s) | Local quality fluctuation |
| FVD | Frechet Video Distance | Distribution match (compares SR to GT distribution) |
| CLIP feature trajectory | Cosine similarity to first frame over time | Slow drift visualization |

### Estimated effort

- Test data generation (Tests A–E): 1–2 days local
- Run all VBench evaluations on test data: ~12h server compute
- Analysis + plots: 1 day
- Long-range tOF + tLP implementation: 1 day
- Total: ~1 week

### Plan document

Full plan saved at: `docs/plans/2026-04-28-metrics-and-vbench-validation.md`

---

## All Evaluations Complete

All UAV synthetic evaluations finished. Final comparison:

| Metric | LQ | MGLD-SR | UAV | Winner |
|--------|----|---------|------|--------|
| CLIP-IQA ↑ | — | 0.496 | 0.391 | MGLD |
| MUSIQ ↑ | — | 65.07 | 56.28 | MGLD |
| NIQE ↓ | — | 4.67 | 5.73 | MGLD |
| BRISQUE ↓ | — | 24.74 | 50.90 | MGLD |
| DOVER overall ↑ | 10.44 | 73.81 | 65.06 | MGLD |
| E\*warp ↓ | 0.0092 | 0.0114 | 0.0137 | LQ (best), MGLD (best SR) |
| VBench imaging_quality ↑ | 0.4388 | 0.6810 | 0.6458 | MGLD |
| VBench aesthetic ↑ | 0.4128 | 0.5080 | 0.4892 | MGLD |
| VBench subject_consistency ↑ | 0.8936 | 0.8927 | 0.9031 | UAV |

**MGLD-SR wins on 8/9 metrics. UAV only wins on subject_consistency (likely artifact of DINOv2 color invariance).**

---

## Next Steps

1. Complete UAV evaluation (NR, E\*warp, VBench)
2. Implement VBench validation test datasets (A–E above)
3. Add long-range tOF, tLP metrics to evaluation pipeline
4. Generate sample frames/videos for thesis figures
5. Start thesis writing: Introduction + Literature Review chapters
