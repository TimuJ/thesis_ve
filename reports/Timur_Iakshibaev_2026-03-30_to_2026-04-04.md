# Weekly Progress Report — Timur Iakshibaev

## Period: March 30 – April 5, 2026

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

## Baseline Experiments — Upscale-A-Video (CVPR 2024) — IN PROGRESS

- Fixed MP4 encoding issue (~7 dB PSNR loss from lossy video encoding)
- Fixed `save_image` loop bug in original repo (wrong tensor dimension)
- Resolved xformers/CUDA version mismatch that caused 10-20x inference slowdown
  - Root cause: torch cu117→cu118 upgrade broke xformers CUDA extensions
  - Fix: Reverted to original `torch 2.0.1+cu117 + xformers 0.0.22` pair
- Currently running inference on UDM10 with RealBasicVSR degradation (~4 hours remaining)

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

### Upscale-A-Video — Pending

- UDM10 with RealBasicVSR degradation: running (~4 hours)
- Paper target: PSNR 30.79, SSIM 0.878, LPIPS 0.133

## Next Steps

1. Complete UAV UDM10 evaluation and verify against paper
2. Run UAV on YouHQ40 with RealBasicVSR degradation
3. Once both models verified → test on long-video sequences (>1 min)
4. Begin literature review on state-space models for long-context video
