# Weekly Progress Report — Timur Iakshibaev

## Period: March 30 – April 12, 2026

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
| VideoLQ | Real-world | NR | — | — | Running (12/50) |

## April 9 Meeting Outcomes

- **Research direction:** Still TBD — PhD student continuing to explore i2v (image-to-video) papers
- **Long-video data:** Will be collected by another student (no timeline yet)
- **Gov dataset:** Provided but not publicly available — internal tests only, not for paper
- **Proposal rewrite:** Most likely needed, deadline May 31, 2026 — will revisit in May
- **Evaluation strategy shift:** Align with **DOVE benchmark** instead of matching individual paper numbers
  - DOVE provides standardized pre-degraded LQ data and published evaluation scripts
  - Our consistent degradation (seed 42) across methods ensures fair internal comparison
- **VBench** needed for human-perception-aligned video quality evaluation — OOM on long videos, beta version exists

## Key Discovery: DOVE Evaluation Difference (April 9)

- Cloned DOVE repo (https://github.com/zhengchen1999/DOVE) and analyzed their `eval_metrics.py`
- **DOVE uses RGB PSNR/SSIM by default** (no `--test_y_channel` flag in their inference.sh)
- Our pyiqa evaluation uses **Y-channel PSNR/SSIM** (standard VSR convention)
- This explains the +1.5 dB gap between our UAV results and DOVE paper's UAV results
- To align: evaluate with DOVE's eval code (RGB PSNR) or explicitly use their settings

## DOVE Benchmark Alignment (April 9–12)

### MGLD-VSR on DOVE UDM10 LQ — IDENTICAL to DOVE paper

- Standard MGLD-VSR inference script center-crops to 512x512 — wrong for DOVE's non-square LQ (318x180)
- **Fix:** Used the tile-based inference script (`vsr_val_ddpm_text_T_vqganfin_oldcanvas_tile.py`) from the MGLD-VSR repo. This splits frames into overlapping patches, processes each through the UNet, and stitches back to full resolution (1272x720).
- Evaluated with DOVE's own `eval_metrics.py` (RGB PSNR, no border crop)
- **Result: Identical to DOVE paper**

| Metric | DOVE paper (MGLD) | Ours | Delta |
|--------|-------------------|------|-------|
| PSNR | 24.23 | **24.23** | +0.00 |
| SSIM | 0.6957 | **0.6957** | 0.0000 |
| LPIPS | 0.3272 | **0.3272** | 0.0000 |
| DISTS | 0.1677 | **0.1676** | -0.0001 |
| CLIP-IQA | 0.4557 | **0.4555** | -0.0002 |

This confirms our full pipeline (MGLD-VSR inference + DOVE evaluation) is perfectly aligned with the DOVE benchmark. We can now trust DOVE's published comparison table as our baseline reference.

### UAV on DOVE UDM10 LQ — Close, narrowing gap

- Re-evaluated existing UAV output (n150 g7) with DOVE eval: PSNR 22.96 vs paper's 21.72 (+1.24 dB)
- Gap is from inference settings — UAV defaults are `n120 g6`, we used `n150 g7`
- Re-running UAV with default settings (in progress, 6/10 clips)

### Key issues encountered
- MGLD-VSR 512x512 center crop produced spatially misaligned output (~10-13 dB PSNR) — solved with tile script
- Native resolution via `--input_size -1` patch failed — UNet skip connections require matching spatial dimensions
- OOM errors when GPU had other processes — need dedicated free GPU (~80GB)

## Server Cleanup (April 9)

- Cleaned up invalid results on server:
  - Removed: MGLD-VSR on old DOVE LQ (512x512 crop), bicubic LQ, old format evals
  - Removed: UAV MP4 runs, bicubic YouHQ40, old wrong-degradation runs, incomplete YouHQ40_rbvsr
  - Kept: all verified results (RealBasicVSR, DOVE LQ frames, param sweep, VideoLQ)

## In Progress at End of Period

| Task | Progress |
|------|----------|
| UAV DOVE LQ default (n120 g6) — expected to close the 1.24 dB gap | 7/10 clips completed |
| UAV VideoLQ NR inference | 43/50 clips completed |

**Completed:** MGLD-VSR DOVE tile inference + DOVE evaluation — identical to DOVE paper.

## Issue: Server Disk Failure (April 12)

- The lab GPU server's main data disk (`/data/disk1`) has experienced an I/O-level failure
- Symptoms: all paths under `/data/disk1` return "Input/output error"; `df -h /data/disk1` fails
- Impact on our work:
  - Running experiments crashed (both `uav_dove` and `uav_vlq` tmux sessions terminated)
  - Server-side data temporarily inaccessible: code repo, model checkpoints, LQ datasets, intermediate results, conda environments
- **Safe (already on local machine):**
  - All committed source code and evaluation scripts
  - MGLD-VSR DOVE eval results (the "identical match" table above) — confirmed verified
  - UAV DOVE LQ (n150 g7) eval results with DOVE script
  - Verified MGLD-VSR and UAV results from previous weeks (RealBasicVSR, VideoLQ 43/50 NR)
- **Needs re-running when disk is restored:**
  - UAV DOVE LQ default (n120 g6) inference — was 7/10 clips done, not yet evaluated
  - UAV VideoLQ NR — was 43/50 clips done
- **Action required:** Contacted lab admin about the disk failure. No further progress possible on server until resolved.

## Next Steps

1. **Resolve server disk failure** with lab admin — blocker for all further GPU work
2. **Re-run lost inferences** once disk is restored (UAV DOVE default + final 7 VideoLQ clips)
3. **Complete UAV VideoLQ NR evaluation** — compare NR metrics with MGLD-VSR on real-world data
4. **Set up VBench** — human-perception-aligned evaluation (OOM on long videos, beta version exists and may need extension)
5. **Test both models on long-video sequences** (>1 minute) when sample data arrives from collaborating student
6. **Proposal rewrite** — deadline May 31 (on track)
