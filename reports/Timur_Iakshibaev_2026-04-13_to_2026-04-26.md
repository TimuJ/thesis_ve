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

Active investigation:
- **DOVE-matched environment (`uav_dove` env):** torch 2.5.1+cu121, xformers 0.0.35, transformers 4.46.2 — but crashed with Flash Attention compatibility error. Needs xformers version fix.
- **SPMCS dataset:** Downloaded (1.3 GB, 30 clips) to test on a second DOVE dataset and check if gap is consistent
- **UAV on SPMCS running** on GPU 7 (n120 g6 s30) — will compare to DOVE paper's SPMCS numbers

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

## MGLD-VSR on Synthetic Videos — Running

MGLD-VSR fully set up on disk2:
- Repo cloned, conda env ready: torch 2.0.1+cu118, xformers 0.0.22, mmcv 2.1.0, pytorch-lightning 1.9.5, transformers 4.28.1, numpy 1.24.3, einops 0.6.1, open-clip-torch 2.20.0
- All checkpoints downloaded: mgldvsr_unet (6.3 GB), video_vae_cfw (766 MB), v2-1_512-ema-pruned (4.9 GB), raft-things.pth (21 MB from UAV propagator)
- Config paths updated from disk1 to disk2
- `configs/video_autoencoder` → `configs/video_vae` symlink created
- SpyNet patch applied (try/except for missing weights — only used in training loss)
- OpenCLIP patch: `FrozenOpenCLIPEmbedder` patched to load weights from local path (server can't reach HuggingFace)
- `from_4d_to_5d` patch: handle `t=None` in einops rearrange (temporal conv receives None from middle_block)
- Synthetic video frames extracted (5 videos, 22,412 frames total)
- Inference relaunched on GPU 3 using tile-based script

**Issues resolved during setup:**
- pytorch-lightning 2.6 → 1.9.5 (torch.utils.flop_counter not in torch 2.0.1)
- transformers 4.57 → 4.28.1 (torch.compiler not in torch 2.0.1)
- numpy 2.0 → 1.24.3 (torch-numpy interop broken with numpy 2.x)
- open_clip_torch 3.3.0 → 2.20.0 (attention mask shape incompatible with torch 2.0.1)
- einops 0.8.2 → 0.6.1 (stricter None handling)
- scikit-image, scikit-learn added
- OpenCLIP weights: patched to use local file instead of downloading (server hangs on HF)
- from_4d_to_5d: patched to infer t from batch size when None

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

## Currently Running (as of April 16 morning)

| tmux session | Task | GPU | Status |
|-------------|------|-----|--------|
| `uav_spmcs` | UAV on SPMCS (30 clips, n120 g6) + DOVE eval | 7 | ~15/30 clips done |
| `mgld_synth` | MGLD-VSR on synthetic (with None-t fix) | 3 | Just relaunched |
| `vbench_redo` | VBench LQ (3 dims × 5 videos, separate outputs) | 5 | Just launched |

## Completed (April 15–16)

- [x] Disk2 migration — full infrastructure rebuilt
- [x] UAV DOVE UDM10 default settings — 10/10 clips done, evaluated (PSNR 23.05 vs 21.72 target)
- [x] UAV alignment investigation — ruled out: input format (MKV=PNG), seed, settings, frame count, resolution, empty prompt
- [x] UAV empty prompt full UDM10 — PSNR 23.72 (worse, wrong direction)
- [x] MGLD-VSR fully set up on disk2 — all checkpoints + deps + patches ready
- [x] Synthetic videos uploaded and frames extracted (5 videos, 22,412 frames)
- [x] VBench working via Python API (MP4 input only, 3 dimensions tested)
- [x] SPMCS dataset downloaded
- [x] Home dir cleaned (21 GB → 184 KB)
- [x] `uav_dove` env created (torch 2.5.1 + DOVE-matched deps) — needs xformers fix for Flash Attention

## Next Steps

1. Check UAV SPMCS results — compare with DOVE paper's SPMCS numbers
2. Fix `uav_dove` env xformers/Flash Attention issue — test UAV with torch 2.5.1
3. Verify MGLD-VSR running on synthetic videos (after None-t patch)
4. Collect VBench LQ baselines (3 dimensions × 5 videos)
5. Run UAV on synthetic videos once DOVE alignment confirmed
6. Evaluate all SR outputs with DOVE metrics + VBench — establish target metrics
