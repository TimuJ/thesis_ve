# MGLD-VSR Synthetic Long-Video Evaluation Results

**Date:** April 28, 2026 (updated)
**Model:** MGLD-VSR (ECCV 2024)
**Dataset:** 5 synthetic long videos (320x180 LQ → 1280x720 SR, 4x upscale)
**Total frames:** 22,412

## Videos

| Video | Frames | Duration | Resolution (SR) |
|-------|--------|----------|-----------------|
| 7WHI2L_FDNg | 7200 | 300.0s | 1280x720 |
| BrRLKMbBTYQ | 3600 | 150.0s | 1280x720 |
| KZ8p6b1zJ9U | 4200 | 175.0s | 1280x720 |
| hhszUXL1Cu8 | 2412 | 80.4s | 1280x720 |
| mJog8DlRk_4 | 5000 | 208.5s | 1280x720 |

## No-Reference Image Quality Metrics

Evaluated with pyiqa (no ground truth available for synthetic videos).

| Video | CLIP-IQA ↑ | MUSIQ ↑ | NIQE ↓ | BRISQUE ↓ |
|-------|-----------|---------|--------|-----------|
| 7WHI2L_FDNg | 0.457 | 68.68 | 4.29 | 25.75 |
| BrRLKMbBTYQ | 0.529 | 62.31 | 6.20 | 28.74 |
| KZ8p6b1zJ9U | 0.493 | 62.86 | 3.83 | 20.34 |
| hhszUXL1Cu8 | 0.456 | 65.36 | 4.72 | 26.20 |
| mJog8DlRk_4 | 0.543 | 66.15 | 4.32 | 22.69 |
| **Mean** | **0.496** | **65.07** | **4.67** | **24.74** |

## VBench 2.0 Long-Video Evaluation

Evaluated using `vbench2_beta_long` mode (slow-fast temporal consistency).

### Quality Score Dimensions (7/7 — all complete)

| Dimension | MGLD-SR | LQ Baseline | Delta |
|-----------|---------|-------------|-------|
| imaging_quality ↑ | 0.6810 | 0.4388 | +0.2422 (+55%) |
| motion_smoothness ↑ | 0.9886 | 0.9873 | +0.0013 |
| temporal_flickering ↑ | 0.9840 | 0.9811 | +0.0029 |
| aesthetic_quality ↑ | 0.5080 | 0.4128 | +0.0952 (+23%) |
| dynamic_degree ↑ | 0.5942 | 0.5628 | +0.0314 (+6%) |
| subject_consistency ↑ | 0.8927 | 0.8936 | -0.0009 |
| background_consistency ↑ | 0.9235 | 0.9333 | -0.0098 |

Key observations:
- Largest improvement in imaging_quality (+55%) and aesthetic_quality (+23%)
- Temporal metrics (motion_smoothness, temporal_flickering) nearly identical — SR preserves temporal coherence
- subject_consistency and background_consistency slightly lower for SR — diffusion-based SR introduces minor frame-to-frame variation

### Semantic Score Dimensions — Not Applicable for SR

All 9 Semantic Score dimensions require text prompts (video descriptions) as ground truth.
VBench was designed for text-to-video generation benchmarking, where prompts describe expected content.
For super-resolution evaluation (no text prompts), these dimensions produce meaningless scores:

- **overall_consistency** — ViCLIP video-text cosine similarity (uses filename as "prompt")
- **appearance_style, temporal_style** — require `auxiliary_info` with style labels
- **human_action** — extracts expected action from filename, compares to Kinetics-400 predictions
- **color, object_class, multiple_objects, spatial_relationship** — GRiT/detectron2 detection vs text prompt
- **scene** — Tag2Text scene classification vs text prompt

## DOVE Benchmark Results (UDM10, full-reference)

MGLD-VSR matches DOVE paper exactly on UDM10:

| Metric | MGLD-VSR (ours) | DOVE paper |
|--------|----------------|------------|
| PSNR ↑ | 24.2339 | 24.23 |
| SSIM ↑ | 0.6957 | 0.6957 |
| LPIPS ↓ | 0.3272 | 0.3272 |
| DISTS ↓ | 0.1676 | — |
| CLIP-IQA ↑ | 0.4555 | — |

## VBench-2.0 Human_Identity (patched)

Identity consistency score (RetinaFace + ArcFace, ~0–1, higher = more consistent).
Two patches applied to original VBench-2.0:
1. Allow multi-face frames (pick largest face) — original required exactly 1 face
2. Allow late reference frame initialization — original required face in frame 0

| Video | MGLD-SR | UAV |
|-------|---------|-----|
| 7WHI2L_FDNg | 0.035 | **0.116** |
| BrRLKMbBTYQ | **0.401** | 0.339 |
| KZ8p6b1zJ9U | 0.534 | **0.537** |
| hhszUXL1Cu8 | **0.011** | 0.009 |
| mJog8DlRk_4 | **0.018** | 0.012 |
| **Mean** | 0.200 | **0.203** |

**Caveat — VBench-2.0 designed for single-person videos.** Algorithm tracks one reference identity (largest face) and compares each frame's largest face to it. For our crowd scenes (multiple people), the largest face can belong to different people across frames, producing artificially low scores. Even so, UAV slightly edges out MGLD overall (+0.003).

A multi-person identity consistency metric would require an algorithm change (e.g., cluster-based identity tracking, score = fraction of faces matching any tracked cluster). Design spec: `docs/plans/2026-05-06-multiperson-identity-metric.md`.

## VBench-2.0 Human_Identity — Slow-Fast Adapter (long-video extension)

Implemented slow-fast adapter (`scripts/vbench2_long/human_identity_long.py`):
- **Slow branch:** split video into 2-sec clips, run patched VBench-2.0 identity per clip, aggregate
- **Fast branch:** concatenate first frame of each clip into a "fast video", run identity on it (catches long-range identity drift)
- **Fusion:** weighted average (default 50/50)

| Video | MGLD slow | MGLD fast | MGLD fused | UAV slow | UAV fast | UAV fused |
|-------|-----------|-----------|------------|----------|----------|-----------|
| 7WHI2L_FDNg | 0.681 | 0.052 | 0.366 | 0.594 | 0.080 | 0.337 |
| BrRLKMbBTYQ | 0.756 | -1.0¹ | 0.756 | 0.675 | 0.286 | 0.481 |
| KZ8p6b1zJ9U | 0.703 | 0.611 | 0.657 | 0.723 | 0.778 | **0.751** |
| hhszUXL1Cu8 | 0.757 | 0.553 | 0.655 | 0.732 | 0.188 | 0.460 |
| mJog8DlRk_4 | 0.512 | 0.170 | 0.341 | 0.473 | 0.098 | 0.285 |
| **Overall** | **0.682** | 0.346 | **0.555** | 0.639 | 0.286 | 0.463 |

¹ fast=-1 means no faces detected in clip first-frames — falls back to slow only.

**MGLD wins 4/5 videos on fused score and overall (+0.092).** UAV only wins on KZ8p6b1zJ9U.

The slow-fast adapter scores are much higher than the whole-video custom_input run (MGLD 0.555 vs 0.200) because:
1. Per-clip evaluation avoids identity drift accumulating across the whole video
2. Each 2-sec clip typically has consistent identity, even in crowd scenes
3. The fast branch specifically targets long-range drift while slow captures local consistency

## VBench-2.0 Human_Anatomy (whole-video custom_input)

Anomaly-detector score (ViTDetector ensemble: human / face / hand) over all frames. Higher = fewer anatomical anomalies.

| Video | MGLD-SR | UAV | Winner |
|-------|---------|-----|--------|
| 7WHI2L_FDNg | **0.832** | 0.735 | MGLD |
| BrRLKMbBTYQ | **0.522** | 0.437 | MGLD |
| KZ8p6b1zJ9U | 0.144 | **0.435** | UAV (large gap) |
| hhszUXL1Cu8 | **0.925** | 0.878 | MGLD |
| mJog8DlRk_4 | **0.577** | 0.541 | MGLD |
| **Mean** | 0.600 | **0.605** | UAV (+0.005, tie) |

Per-video: **MGLD wins 4/5**. UAV wins only `KZ8p6b1zJ9U` — same outlier where UAV won on Human_Identity. The 0.144 score there drags MGLD's mean to a statistical tie even though it wins everywhere else. `KZ8p6b1zJ9U` is a single-person scene (not crowd), so the multi-person hypothesis from earlier reports does not apply.

**Per-frame diagnostic on `KZ8p6b1zJ9U`:** MGLD is flagged abnormal in 84.8% of frames-with-people (median rate 1.0); UAV in 53.2%. All three detectors fire ~2× more on MGLD (human 2179 vs 858, face 2074 vs 973, hand 2714 vs 1520). Failure is uniform across the video — every 10-second window has MGLD's abnormal-rate above UAV's. The detector regime explains the per-video pattern: on the 4 MGLD-wins videos the detector is in a low-fire regime (both scores high) and MGLD's slightly cleaner output beats UAV by small margins; on `KZ8p6b1zJ9U` the detector is in a high-fire regime (both scores low) where MGLD's diffusion output asymmetrically over-triggers it. Diagnostic doc: `docs/plans/2026-05-07-metric-failure-diagnostic.md`.

Raw eval JSONs: `results/vbench2_anatomy/{mgld,uav}_anatomy_eval_results.json`.
Per-frame trace: `results/vbench2_anatomy/diagnostic_KZ8p6b1zJ9U/{mgld,uav}_KZ8p6b1zJ9U_per_frame.json`.
Patches applied to upstream VBench-2.0 to make the run go through (CLIP-ViT-Base-Patch32 path, `VBENCH2_CACHE_DIR` env, anomaly-detector weight re-download): see `scripts/vbench2_long/README.md`.

## DOVER Video Quality (no-reference, per-video)

Evaluated with [DOVER](https://github.com/VQAssessment/DOVER) — disentangled aesthetic + technical video quality.

| Video | MGLD-SR Aesthetic | MGLD-SR Technical | MGLD-SR Overall | LQ Overall |
|-------|-------------------|-------------------|-----------------|------------|
| 7WHI2L_FDNg | 99.84 | 11.85 | 78.27 | 4.09 |
| BrRLKMbBTYQ | 99.30 | 7.38 | 59.59 | 20.22 |
| KZ8p6b1zJ9U | 99.90 | 11.37 | 80.94 | 3.08 |
| hhszUXL1Cu8 | 99.94 | 8.58 | 81.01 | 16.11 |
| mJog8DlRk_4 | 99.59 | 10.19 | 69.22 | 8.71 |
| **Mean** | **99.71** | **9.87** | **73.81** | **10.44** |

MGLD-SR overall quality ~7x higher than LQ. Aesthetic scores near-perfect; technical scores lower (expected for diffusion-based SR).

## E\*warp Temporal Consistency (RAFT optical flow warping error)

Lower = better (less temporal inconsistency between adjacent frames).

| Video | MGLD-SR ↓ | LQ ↓ | Delta |
|-------|----------|------|-------|
| 7WHI2L_FDNg | 0.0106 | 0.0060 | +0.0046 |
| BrRLKMbBTYQ | 0.0144 | 0.0124 | +0.0020 |
| KZ8p6b1zJ9U | 0.0124 | 0.0116 | +0.0008 |
| hhszUXL1Cu8 | 0.0021 | 0.0013 | +0.0008 |
| mJog8DlRk_4 | 0.0174 | 0.0149 | +0.0025 |
| **Mean** | **0.0114** | **0.0092** | **+0.0022** |

SR introduces slightly more temporal inconsistency (+24%) than LQ — expected for diffusion-based SR where each frame is independently denoised. Absolute values are low for both.

## UAV Synthetic (in progress)

UAV inference running on synthetic videos (n120 g6 s30, chunked at 2500 frames):

- hhszUXL1Cu8 (2412 frames): **Done**
- BrRLKMbBTYQ (5000 frames): **Done**
- KZ8p6b1zJ9U (5000 frames): Running
- 7WHI2L_FDNg, mJog8DlRk_4: Pending

## Environment

- **Inference:** conda `mgldvsr` — PyTorch 2.0.1+cu118, einops 0.3.0, open-clip-torch 2.20.0
- **NR eval:** conda `vsr` — PyTorch 2.5.1+cu121, pyiqa
- **VBench:** conda `vbench` — PyTorch 2.5.1+cu121, VBench v0.1.5, moviepy 1.0.3
- **Server:** Timur@223.109.239.43, results at `/data/disk2/timur/results/`
