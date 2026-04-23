# MGLD-VSR Synthetic Long-Video Evaluation Results

**Date:** April 23, 2026 (updated)
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

## Environment

- **Inference:** conda `mgldvsr` — PyTorch 2.0.1+cu118, einops 0.3.0, open-clip-torch 2.20.0
- **NR eval:** conda `vsr` — PyTorch 2.5.1+cu121, pyiqa
- **VBench:** conda `vbench` — PyTorch 2.5.1+cu121, VBench v0.1.5, moviepy 1.0.3
- **Server:** Timur@223.109.239.43, results at `/data/disk2/timur/results/`
