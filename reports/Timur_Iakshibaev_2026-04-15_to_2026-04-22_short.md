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

## MGLD-VSR Disk2 Environment Verification (April 22)

Ran UDM10 clip 000 on disk2 env to check if env differences affect quality:

| Metric | disk1 env (verified) | disk2 env | Match? |
|--------|---------------------|-----------|--------|
| PSNR | 23.9881 | 23.9881 | IDENTICAL |

Despite different open-clip versions (2.0.2 vs 2.20.0), results are identical. Low quality on synthetic videos is content-related (people/complex scenes), not environmental.

## VBench 2.0 — Full 16-Dimension Evaluation (In Progress)

Running all 16 VBench quality + semantic dimensions on both MGLD-SR and LQ baselines. Required downloading ~4 GB of model weights (DreamSim, DINO, GRiT, UMT, ViCLIP, RAFT, CLIP, Tag2Text) — server has slow connectivity so weights were downloaded locally and SCP'd.

Completed so far (5/16 MGLD, 3/16 LQ):

| Dimension | MGLD-SR | LQ | Category |
|-----------|---------|-----|----------|
| imaging_quality | 0.6810 | 0.4388 | Quality |
| motion_smoothness | 0.9886 | 0.9873 | Quality |
| temporal_flickering | 0.9840 | 0.9811 | Quality |
| aesthetic_quality | 0.5080 | — | Quality |
| dynamic_degree | 0.5942 | — | Quality |

Remaining 11 dimensions running on GPUs 5 and 6 in parallel.

## UAV Torch 2.5.1 Test — Blocked

Attempted to test UAV with torch 2.5.1 (matching DOVE's environment) to close the +1.33 dB PSNR gap. All CUDA variants failed:
- cu124: cuDNN CUDNN_STATUS_NOT_INITIALIZED
- cu121: same cuDNN error
- cu118: missing libcudart.so.11.0

Root cause: server CUDA driver (v570/12.8) incompatible with torch 2.5.1's bundled cuDNN. Need either a different server or to contact DOVE authors for their exact setup.

## Next Steps

1. Complete VBench 2.0 evaluation (all 16 dims for MGLD + LQ)
2. Calculate Quality Score and Semantic Score using VBench normalization
3. Run UAV on synthetic long videos with original env (torch 2.0.1+cu117, n120 g6 s30) — gap documented as +1.33 dB, proceed with consistent setup
4. Evaluate UAV synthetic with DOVE metrics + VBench 2.0 for comparison with MGLD-SR
5. Start thesis writing (Introduction + Literature Review chapters)
6. Proposal outline when PhD student provides materials
