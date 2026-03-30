# Weekly Progress Report — Timur Iakshibaev

## Period: March 23 – March 30, 2026

## Topic Change & Research Direction

- Changed thesis topic from "Reasoning Video Segmentation" to **Video Super-Resolution for Long Videos (>1 minute)**
- Key reference paper: "Long-Context State-Space Video World Models" (arxiv 2505.20171) — combines state-space models with diffusion for efficient long-range temporal modeling in video
- Research direction: SSM-based temporal modeling + diffusion-based enhancement for long-video SR

## Repository Setup

- Set up project repository with modular Python codebase, LaTeX thesis template, and proposal template
- Configured for local development (M1 Mac) and GPU lab machine portability via environment variable overrides

## Evaluation Infrastructure

- Implemented VSR evaluation metrics: PSNR, SSIM, LPIPS, and temporal consistency
- LPIPS uses lazy model loading with GPU support
- FPS and VRAM tracking utilities for efficiency benchmarking
- Central path configuration for standard VSR datasets (REDS, Vimeo-90K, Vid4, UDM10)
- Unit tests for all metrics

## Baseline Inference Infrastructure

- Designed and implemented a reproducible baseline evaluation pipeline for diffusion-based VSR methods
- Set up two baseline models:
  - **Upscale-A-Video** (CVPR 2024) — diffusion + text prompts for video upscaling
  - **MGLD-VSR** (ECCV 2024) — motion-guided latent diffusion for VSR
- For each model: setup script (clone, conda env, checkpoint download) and inference script with standardized interface
- Shared evaluation script that computes PSNR, SSIM, and LPIPS between model outputs and ground truth
- DOVE dataset download script with support for 6 benchmarks (UDM10, SPMCS, YouHQ40, RealVSR, MVSR4x, VideoLQ)

## Documentation

- Created presentation slides for the reference paper (arxiv 2505.20171)
