# Weekly Progress Report — Timur Iakshibaev

## Period: March 30 – April 4, 2026

## GPU Server Setup

- Connected to lab GPU server (8x NVIDIA A100-SXM4-80GB)
- Installed Miniconda, set up three conda environments:
  - `vsr` — base evaluation env (PyTorch 2.5.1+cu121, pyiqa, LPIPS)
  - `uav` — Upscale-A-Video (PyTorch 2.0.1+cu117, xformers 0.0.22)
  - `mgldvsr` — MGLD-VSR (PyTorch 2.0.1+cu118, xformers 0.0.22, mmcv 2.1.0)
- Downloaded and configured model checkpoints for both baselines
- All 10 unit tests pass on server

## Baseline Experiments — Upscale-A-Video (CVPR 2024)

- Ran inference on UDM10 and YouHQ40 datasets
- Discovered that ffmpeg MP4 encoding in the inference pipeline causes ~7 dB PSNR loss — switched to direct frame I/O (model natively supports image folder input + `--save_image` flag)
- Fixed bug in original repo's `save_image` loop (wrong tensor dimension index)
- Added `local_files_only=True` to `from_pretrained()` (server cannot reach HuggingFace)
- Currently running inference on YouHQ40 with proper RealBasicVSR degradation

## Baseline Experiments — MGLD-VSR (ECCV 2024)

- Rebuilt conda environment from scratch (original `environment.yaml` had broken `pip=20.3`)
- Upgraded PyTorch from 1.12.1 to 2.0.1 (xformers required for multi-frame attention)
- Fixed config path hardcoding, PyAV compatibility (`pict_type` int enum), SpyNet weight loading
- Ran inference on DOVE UDM10 and original bicubic UDM10
- Currently running inference on VideoLQ (50 real-world clips, 4 GPUs in parallel)

## Key Finding: Correct Degradation Pipeline

- Discovered that both papers (UAV and MGLD-VSR) use **RealBasicVSR degradation pipeline** for generating test LQ data — not simple bicubic downsampling
- This is a two-stage degradation: blur + resize + noise + JPEG + video compression (applied twice), then final resize to target LQ resolution
- Our initial experiments using DOVE LQ and bicubic LQ produced metrics that didn't match paper-reported values
- Extracted the degradation pipeline from MGLD-VSR repo into a standalone script (`src/data/realbasicvsr_degrade.py`)
- Generated RealBasicVSR-degraded LQ for YouHQ40 and UDM10 datasets

## Evaluation Infrastructure

- Implemented standardized evaluation using pyiqa library:
  - `evaluate_pyiqa.py` — full-reference metrics (PSNR on Y channel, SSIM on Y channel, LPIPS on RGB)
  - `evaluate_pyiqa_nr.py` — no-reference metrics (NIQE, BRISQUE, MUSIQ, CLIPIQA)
- All metrics match paper conventions (Y-channel PSNR/SSIM via pyiqa)

## Results Summary (preliminary)

| Experiment | PSNR | SSIM | LPIPS | Paper Target |
|------------|------|------|-------|-------------|
| UAV YouHQ40 (bicubic LQ) | 24.47 | 0.675 | 0.260 | 25.83 / 0.733 / 0.268 |
| MGLD-VSR DOVE UDM10 | 24.60 | 0.696 | 0.327 | 25.99 / 0.755 / 0.349 |

Note: These used incorrect degradation. Experiments with proper RealBasicVSR degradation are running now.

## Next Steps

1. Evaluate UAV and MGLD-VSR on RealBasicVSR-degraded datasets
2. Verify metrics match paper-reported values (confirms correct setup)
3. Once verified, test both models on long-video sequences (>1 minute) to expose temporal consistency limitations
4. Begin literature review on state-space models for long-context video processing
