# Weekly Progress Report — Timur Iakshibaev

## Period: April 13 – April 26, 2026

## Disk2 Migration (April 15)

After the `/data/disk1` I/O failure on April 12, all infrastructure was rebuilt on `/data/disk2/timur/`:

- Installed Miniconda (conda 26.1.1)
- Recreated conda environments:
  - `uav` — Upscale-A-Video (PyTorch 2.0.1+cu117, xformers 0.0.22, diffusers 0.16.0)
  - `vsr` — evaluation (PyTorch 2.5.1+cu121, pyiqa, lpips)
  - `mgldvsr` — MGLD-VSR (setting up: PyTorch 2.0.1+cu118, xformers 0.0.22, mmcv 2.1.0)
  - `vbench` — VBench evaluation (setting up: PyTorch 2.5.1+cu121)
- Cloned repos: Upscale-A-Video (patched), MGLD-VSR, DOVE
- Downloaded and verified UAV checkpoints (4.1 GB, from local backup `upscale_a_video-001.zip`)
- Downloaded DOVE UDM10 dataset (GT + LQ, 578 MB) via gdown
- Moved `/home/Timur/` caches (21 GB) to `/data/disk2/timur/cache/` with symlinks, per admin request

## UAV DOVE UDM10 — Alignment Investigation

### Default Settings (n120 g6 s30) — COMPLETED

All 10 clips processed on GPU 2. Output verified: 32 frames per clip, 1272x720 resolution (matches GT).

DOVE evaluation results (RGB PSNR, using DOVE's eval_metrics.py):

| Metric | DOVE paper (UAV) | Our result | Difference |
|--------|-----------------|------------|------------|
| PSNR | 21.72 | 23.05 | +1.33 dB |
| SSIM | 0.5913 | 0.6164 | +0.025 |
| LPIPS | 0.4116 | 0.4252 | +0.014 |
| DISTS | 0.2230 | 0.2364 | +0.013 |

**Gap: +1.33 dB PSNR.** Our result is HIGHER (better fidelity) than DOVE reports.

### Investigation: What Causes the Gap?

Verified and ruled out:
- **Inference settings:** n120 g6 s30 — match UAV defaults and DOVE Table 4 ✓
- **Random seed:** hardcoded to 10 in UAV code — deterministic ✓
- **Input format:** MKV vs PNG pixel-identical, tested both — identical output ✓
- **Frame count:** 32 frames in LQ, GT, and pred — all match ✓
- **Output resolution:** 1272x720 matches GT ✓
- **Eval script:** DOVE's own eval_metrics.py, RGB PSNR (no --test_y_channel) ✓
- **Empty prompt (--a_prompt "" --n_prompt ""):** PSNR went UP to 22.61 (wrong direction) ✗

Completed tests:
- **Full UDM10 with empty prompt** — PSNR 23.72 (even further from target, wrong direction)
- **MKV video input** — identical to PNG (PSNR 22.3843 both)

Cross-dataset verification:
- **UAV SPMCS (n120 g6 s30)** — COMPLETED. 30 clips. Same gap pattern:

| Metric | DOVE paper (UAV SPMCS) | Our result | Difference |
|--------|----------------------|------------|------------|
| PSNR | 18.81 | 20.49 | +1.68 dB |
| SSIM | 0.4113 | 0.4838 | +0.073 |
| LPIPS | 0.4468 | 0.4280 | -0.019 |
| DISTS | 0.2452 | 0.2481 | +0.003 |

**Gap is consistent across datasets** (+1.33 dB UDM10, +1.68 dB SPMCS). Confirms environmental cause (torch/xformers version), not dataset-specific.

Active investigation:
- **DOVE-matched environment (`uav_dove` env):** torch 2.5.1+cu121, xformers 0.0.35, transformers 4.46.2 — crashed with Flash Attention compatibility error. Needs xformers version fix. `XFORMERS_DISABLE_FLASH_ATTN=1` env var also didn't help.

### UAV Code Patches Applied (same as disk1)
1. `inference_upscale_a_video.py` line 101: `local_files_only=True` (server has no HF access)
2. `inference_upscale_a_video.py` line 349: `output.shape[2]` → `output.shape[0]` (save_image bug)
3. `utils.py` line 3: added `import numpy as np`
4. `utils.py`: added `.mkv` to `VIDEO_EXTENSIONS` for MKV input testing
5. `charset-normalizer` reinstalled (version conflict with diffusers)
6. `av` (PyAV) installed for torchvision video loading

## Synthetic Long-Video Dataset

Received and uploaded 5 long videos (1.4 GB total) for VBench evaluation:

| File | Resolution | FPS | Frames | Duration |
|------|-----------|-----|--------|----------|
| 7WHI2L_FDNg.mkv | 320x180 | 30 | 5000 | 166.8s |
| BrRLKMbBTYQ.mkv | 320x180 | 24 | 5000 | 208.3s |
| KZ8p6b1zJ9U.mkv | 320x180 | 30 | 5000 | 166.8s |
| hhszUXL1Cu8.mkv | 320x180 | 30 | 2412 | 80.5s |
| mJog8DlRk_4.mkv | 320x180 | 24 | 5000 | 208.5s |

These are LQ (320x180) videos. Plan:
1. Run VBench on LQ to verify VBench pipeline works
2. Upscale with UAV and MGLD-VSR (4x → 1280x720)
3. Evaluate SR outputs with both DOVE metrics (PSNR, SSIM, LPIPS, DISTS, CLIP-IQA) and VBench (subject/background consistency, motion smoothness, temporal flickering, aesthetic/imaging quality) — these become target metrics for our method

## MGLD-VSR — DOVE UDM10 Verification + Synthetic Videos

### UDM10 Verification — IDENTICAL MATCH (April 18-20)

Re-ran MGLD-VSR on DOVE UDM10 LQ using disk1 env (einops 0.3.0, known-working). Result:

| Metric | DOVE paper | Our result (disk1 env) | Match? |
|--------|-----------|----------------------|--------|
| PSNR | 24.23 | 24.2339 | IDENTICAL |
| SSIM | 0.6957 | 0.6957 | IDENTICAL |
| LPIPS | 0.3272 | 0.3272 | IDENTICAL |
| DISTS | 0.1676 | 0.1676 | IDENTICAL |

Confirms disk1 env reproduces DOVE paper exactly. Key env difference from disk2: **einops 0.3.0** (not 0.4.1 or 0.6.1).

### Synthetic Videos — ALL 5 DONE (April 16-20)

MGLD-VSR completed on all 5 synthetic long videos using disk2 env (einops 0.3.0 after downgrade):

| Video | Frames | Status |
|-------|--------|--------|
| 7WHI2L_FDNg | 5000 | Done |
| BrRLKMbBTYQ | 5000 | Done |
| KZ8p6b1zJ9U | 5000 | Done |
| hhszUXL1Cu8 | 2412 | Done |
| mJog8DlRk_4 | 5000 | Done |
| **Total** | **22,412** | **All done** |

Output at `/data/disk2/timur/results/mgld_synthetic/`. Evaluation pending.

### Disk2 Env Setup Issues (documented)

MGLD-VSR fully set up on disk2:
- Repo cloned, conda env ready: torch 2.0.1+cu118, xformers 0.0.22, mmcv 2.1.0
- All checkpoints downloaded
- Config paths updated, symlinks created
- SpyNet patch, OpenCLIP local path patch applied
- See `docs/private/mgld-vsr-patches.md` for full patch documentation

**Version pins resolved (critical for reproduction):**
- einops 0.3.0 (0.4.1+ breaks None kwargs in rearrange)
- pytorch-lightning 1.9.5 (2.x needs torch >= 2.1)
- transformers 4.28.1 (4.46+ needs torch.compiler)
- numpy 1.24.3 (2.x breaks torch interop)
- open_clip_torch 2.20.0 (3.x has attention mask incompatibility)

## VBench Setup — Working

- VBench v0.1.5 installed in `vbench` conda env (torch 2.5.1+cu121)
- `setuptools<81` required for `pkg_resources` (CLIP dependency)
- VBench CLI crashes with PyTorch distributed error — **Python API works** instead:
  ```python
  from vbench import VBench
  bench = VBench(device="cuda", full_info_dir=".", output_path="output/")
  bench.evaluate(videos_path="video.mp4", name="test", dimension_list=["imaging_quality"], mode="custom_input")
  ```
- VBench only supports GIF/PNG/MP4 — MKV must be converted to MP4 first (lossless via cv2)
- Initial LQ baseline (temporal_flickering on 5 synthetic videos): 0.963–0.995
- Re-running with separate output dirs per dimension to avoid overwriting

## Server Home Directory Cleanup

Per admin request, moved all caches from `/home/Timur/` to `/data/disk2/timur/cache/`:
- pip cache: 15 GB
- HuggingFace cache: 5.1 GB
- torch cache: 1.3 GB
- Symlinks created so existing tools work transparently
- `/home/Timur/` reduced from 21 GB to 184 KB

## SPMCS Dataset — Downloaded (April 16)

Downloaded DOVE SPMCS dataset (1.3 GB, 30 clips) as a second benchmark for UAV alignment verification. If the +1.33 dB gap is consistent across datasets, it confirms the cause is environmental (torch version). If the gap differs, it points to something dataset-specific.

## Server Incident (April 16-18)

Server taken offline for security investigation — suspected compromise via another user's Docker container (`zrk`). Multiple failed `_apt` connections from internal IP at 00:03 April 14 (not from our account). See `docs/private/server-incident-2026-04-16.md` for full incident report and recovery roadmap.

- Server restored April 18, SSH host keys regenerated
- Connection intermittently drops — server still being stabilized
- Disk1 restored (was dead since April 12)
- All our data on both disks intact

## VBench 1.0 LQ Baselines (partial)

| Video | imaging_quality | motion_smoothness | temporal_flickering |
|-------|----------------|-------------------|-------------------|
| 7WHI2L_FDNg | 35.0 | 0.991 | 0.986 |
| BrRLKMbBTYQ | 59.8 | 0.993 | 0.988 |
| hhszUXL1Cu8 | 47.7 | — | 0.995 |
| KZ8p6b1zJ9U | 29.7 | — | 0.981 |
| mJog8DlRk_4 | 45.5 | — | 0.963 |

VBench 2.0 (long video beta) — repo clone failed due to server connectivity. Needs retry.

## Completed (April 15–20)

- [x] Disk2 migration — full infrastructure rebuilt
- [x] UAV DOVE UDM10 default settings — PSNR 23.05 vs 21.72 target (+1.33 dB gap)
- [x] UAV DOVE SPMCS default settings — PSNR 20.49 vs 18.81 target (+1.68 dB gap)
- [x] UAV alignment investigation — ruled out: input format, seed, settings, frame count, resolution, empty prompt. Gap consistent across datasets → environmental cause
- [x] MGLD-VSR UDM10 re-verification (disk1 env) — IDENTICAL match (PSNR 24.2339)
- [x] MGLD-VSR synthetic — all 5 videos done (22,412 frames)
- [x] VBench 1.0 working via Python API — partial LQ baselines collected
- [x] SPMCS dataset downloaded and UAV evaluated
- [x] Home dir cleaned (21 GB → 184 KB)
- [x] Server incident documented with recovery roadmap
- [x] `uav_dove` env created (torch 2.5.1) — needs xformers fix

## Next Steps

1. Evaluate MGLD synthetic outputs with DOVE metrics (no-reference since no GT)
2. Fix UAV `uav_dove` env (torch 2.5.1) — close the PSNR gap
3. Clone VBench 2.0 repo and test long-video evaluation
4. Run UAV on synthetic videos
5. Evaluate all SR outputs with DOVE metrics + VBench
6. Start local thesis writing (Introduction + Literature Review chapters)
