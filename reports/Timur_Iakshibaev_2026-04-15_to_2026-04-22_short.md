# Progress Report — April 15–22, 2026

## Key Results

### 1. MGLD-VSR — Identical Match Confirmed

Re-ran MGLD-VSR on DOVE UDM10 after disk2 migration. Results match DOVE paper exactly:

| Metric | DOVE Paper | Ours |
|--------|-----------|------|
| PSNR | 24.23 | 24.2339 |
| SSIM | 0.6957 | 0.6957 |
| LPIPS | 0.3272 | 0.3272 |

Critical dependency: `einops==0.3.0` (newer versions break MGLD-VSR).

### 2. MGLD-VSR on Synthetic Long Videos — Completed

All 5 synthetic videos (22,412 frames total) super-resolved from 320x180 to 1280x720.

NR evaluation:

| Video | CLIP-IQA ↑ | MUSIQ ↑ | NIQE ↓ | BRISQUE ↓ |
|-------|-----------|---------|--------|-----------|
| 7WHI2L_FDNg | 0.457 | 68.68 | 4.29 | 25.75 |
| BrRLKMbBTYQ | 0.529 | 62.31 | 6.20 | 28.74 |
| KZ8p6b1zJ9U | 0.493 | 62.86 | 3.83 | 20.34 |
| hhszUXL1Cu8 | 0.456 | 65.36 | 4.72 | 26.20 |
| mJog8DlRk_4 | 0.543 | 66.15 | 4.32 | 22.69 |
| **Mean** | **0.496** | **65.07** | **4.67** | **24.74** |

### 3. VBench 2.0 Long-Video Evaluation — Working

Set up VBench 2.0 (`vbench2_beta_long`) for long-video perceptual evaluation. Evaluated both LQ and MGLD-SR synthetic videos:

| Dimension | LQ Baseline | MGLD-SR | Improvement |
|-----------|------------|---------|-------------|
| imaging_quality ↑ | 0.439 | **0.681** | +55% |
| motion_smoothness ↑ | 0.987 | **0.989** | +0.2% |
| temporal_flickering ↑ | 0.981 | **0.984** | +0.3% |

MGLD-SR significantly improves image quality while preserving temporal consistency on long videos. These serve as baseline target metrics.

### 4. UAV DOVE Alignment — Gap Persists

UAV (Upscale-A-Video) results are consistently +1.3–1.7 dB above DOVE paper across two datasets:

| Dataset | DOVE Paper | Ours | Gap |
|---------|-----------|------|-----|
| UDM10 | 21.72 | 23.05 | +1.33 dB |
| SPMCS | 18.81 | 20.49 | +1.68 dB |

Ruled out: input format, random seed, inference settings, frame count, resolution, prompt text, eval script.

Attempted torch 2.5.1 test (to match DOVE's environment), but server's CUDA driver (v570/12.8) is incompatible with torch 2.5.1's cuDNN. This investigation is blocked until a compatible torch version is found or DOVE authors are contacted.

### 5. UAV SPMCS — All 30 Clips Done

Completed UAV inference on DOVE SPMCS dataset (30 clips, n120 g6 s30). Evaluated with DOVE eval script.

## Infrastructure

### Disk2 Migration (April 15)
- Rebuilt all infrastructure after disk1 I/O failure
- 5 conda environments: `uav`, `vsr`, `mgldvsr`, `vbench`, `uav_dove`
- Cloned and patched: Upscale-A-Video, MGLD-VSR, DOVE, VBench repos

### Server Incident (April 16–18)
- Server taken offline for security investigation (suspected compromise via another user's Docker container)
- Disk1 restored after incident
- SSH host keys regenerated (fingerprint update required)
- Server connection still intermittently unstable

### Home Directory Cleanup
- Moved 21 GB of caches from `/home/Timur/` to `/data/disk2/timur/cache/` with symlinks per admin request

## Next Steps

1. Resolve UAV DOVE alignment gap — contact DOVE authors for exact environment details
2. Run UAV on synthetic long videos (after alignment confirmed)
3. Evaluate UAV synthetic with VBench 2.0 for comparison with MGLD-SR
4. Start thesis writing (Introduction + Literature Review chapters) — plan ready
5. Proposal outline when PhD student provides materials
