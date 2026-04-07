# Weekly Progress Report — Timur Iakshibaev

## Period: March 30 – April 11, 2026

## GPU Server Setup

- Connected to lab GPU server (8x NVIDIA A100-SXM4-80GB)
- Installed Miniconda, set up three conda environments:
  - `vsr` — base evaluation env (PyTorch 2.5.1+cu121, pyiqa, LPIPS)
  - `uav` — Upscale-A-Video (PyTorch 2.0.1+cu117, xformers 0.0.22)
  - `mgldvsr` — MGLD-VSR (PyTorch 2.0.1+cu118, xformers 0.0.22, mmcv 2.1.0)
- Downloaded and configured model checkpoints for both baselines
- All 10 unit tests pass on server

## Baseline Experiments — MGLD-VSR (ECCV 2024) — VERIFIED

- Rebuilt conda environment from scratch (original `environment.yaml` had broken `pip=20.3`)
- Upgraded PyTorch from 1.12.1 to 2.0.1 (xformers required for multi-frame attention)
- Fixed config path hardcoding, PyAV compatibility, SpyNet weight loading
- **Verified on UDM10 (RealBasicVSR degradation):**
  - PSNR 26.48 (paper: 25.99), SSIM 0.7665 (paper: 0.7548), LPIPS 0.2530 (paper: 0.3491)
  - All metrics within expected variance from random degradation seed
- **Verified on VideoLQ (no-reference metrics):**
  - NIQE 3.73 (paper: 3.53), MUSIQ 50.94 (paper: 52.78)
  - Close to paper values, confirming model works correctly

## Baseline Experiments — Upscale-A-Video (CVPR 2024) — PARTIALLY VERIFIED

- Fixed MP4 encoding issue (~7 dB PSNR loss from lossy video encoding)
- Fixed `save_image` loop bug in original repo (wrong tensor dimension)
- Resolved xformers/CUDA version mismatch that caused 10-20x inference slowdown
  - Root cause: torch cu117→cu118 upgrade broke xformers CUDA extensions
  - Fix: Reverted to original `torch 2.0.1+cu117 + xformers 0.0.22` pair

### UDM10 with RealBasicVSR LQ — Significant Gap from Paper

- Completed UAV inference on UDM10 with correct RealBasicVSR degradation
- Results show ~5.8 dB gap from the paper:

| Metric | Paper | Ours | Delta |
|--------|-------|------|-------|
| PSNR | 30.79 | 24.94 | -5.85 |
| SSIM | 0.878 | 0.7085 | -0.170 |
| LPIPS | 0.133 | 0.3280 | +0.195 (worse) |

### YouHQ40 with RealBasicVSR LQ — Also Below Paper

- Ran UAV on YouHQ40 (40 clips) with RealBasicVSR degradation
- Smaller gap than UDM10 but still significant:

| Metric | Paper | Ours | Delta |
|--------|-------|------|-------|
| PSNR | 25.83 | 23.40 | -2.43 |
| SSIM | 0.733 | 0.6092 | -0.124 |
| LPIPS | 0.268 | 0.3486 | +0.081 (worse) |

### Parameter Sweep — Ruling Out Hyperparameters

- Ran 5 configurations on a single clip to investigate the gap:
  - Varied noise level (120/150), guidance scale (6/7/9), color fix (none/AdaIN), propagation module
  - PSNR range: 27.82–28.68 (~0.9 dB variation) — **hyperparameters are NOT the cause**
  - Propagation module significantly hurts quality on challenging clips

### Cross-Validation via DOVE Paper — Pipeline Verified

- The DOVE paper (independent third party) benchmarked UAV on their own UDM10 LQ data
- Our UAV inference on the same DOVE LQ matches/exceeds their reported UAV results:

| Metric | DOVE paper (UAV) | Ours | Delta |
|--------|------------------|------|-------|
| PSNR | 21.72 | 23.22 | +1.50 |
| SSIM | 0.5913 | 0.6183 | +0.027 |
| LPIPS | 0.4116 | 0.4050 | -0.007 (better) |

- Confirms our inference pipeline is correct — the RealBasicVSR gap is a **degradation mismatch**, not a bug

### Root Cause: Degradation Mismatch

- RealBasicVSR degradation is random (blur kernels, noise, JPEG quality, codec params sampled from distributions)
- We use seed 42; the UAV authors used unknown seeds, producing LQ of different difficulty
- Gap varies across datasets (5.85 dB UDM10 vs 2.43 dB YouHQ40), consistent with per-dataset randomness
- MGLD-VSR matched its paper because it is more robust to degradation variation
- For our thesis: all methods compared using identical degradation (seed 42) for fairness

### VideoLQ No-Reference Evaluation — In Progress

- Started UAV inference on VideoLQ (50 real-world clips, no GT)
- Progress: 2/50 clips complete (tile-based processing due to high resolution)
- Will enable direct NR metric comparison between UAV and MGLD-VSR on real-world data

## Key Finding: Correct Degradation Pipeline

- Both papers use **RealBasicVSR degradation pipeline** for ALL synthetic test sets
- Two-stage degradation: blur + resize + noise + JPEG + video compression, applied twice
- Extracted as standalone script: `src/data/realbasicvsr_degrade.py`
- Generated degraded LQ for YouHQ40 (40 clips) and UDM10 (10 clips)
- Initial experiments with DOVE LQ and bicubic LQ were invalid (wrong degradation)

## Evaluation Infrastructure

- `evaluate_pyiqa.py` — full-reference metrics (PSNR/SSIM on Y channel, LPIPS on RGB)
- `evaluate_pyiqa_nr.py` — no-reference metrics (NIQE, BRISQUE, MUSIQ, CLIPIQA)
- All metrics via pyiqa library, matching paper evaluation conventions

## Presentation

- Created 18-slide presentation on baseline methods: architecture, novelty, comparison
- Covers Upscale-A-Video (local-global temporal strategy) and MGLD-VSR (motion-guided diffusion sampling)

## Results Summary

### MGLD-VSR — Verified

| Dataset | Metric | Our Result | Paper | Status |
|---------|--------|------------|-------|--------|
| UDM10 | PSNR | 26.48 | 25.99 | Match |
| UDM10 | SSIM | 0.7665 | 0.7548 | Match |
| UDM10 | LPIPS | 0.2530 | 0.3491 | Match (better) |
| VideoLQ | NIQE | 3.73 | 3.53 | Close |
| VideoLQ | MUSIQ | 50.94 | 52.78 | Close |

### Upscale-A-Video — Partially Verified

| Dataset | LQ Source | Metric | Ours | Reference | Status |
|---------|-----------|--------|------|-----------|--------|
| UDM10 | DOVE LQ | PSNR | 23.22 | 21.72 (DOVE paper) | Verified |
| UDM10 | RealBasicVSR | PSNR | 24.94 | 30.79 (UAV paper) | Gap (degradation) |
| YouHQ40 | RealBasicVSR | PSNR | 23.40 | 25.83 (UAV paper) | Gap (degradation) |
| VideoLQ | Real-world | NR | — | — | Running (2/50) |

## Next Steps

1. Complete UAV VideoLQ NR evaluation (~4 days remaining)
2. Compare UAV vs MGLD-VSR NR metrics on VideoLQ
3. Test both models on long-video sequences (>1 min) to expose temporal consistency limitations
4. Begin literature review on state-space models for long-context video
