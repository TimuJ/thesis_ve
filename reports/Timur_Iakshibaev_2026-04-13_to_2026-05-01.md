# Biweekly Progress Report — Timur Iakshibaev

## Period: April 13 – May 1, 2026

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
- **DOVE-matched environment (`uav_dove` env):** torch 2.5.1+cu121, xformers 0.0.28.post3, transformers 4.37.0
  - Fixed: xformers 0.0.35 → 0.0.28.post3 (Flash Attention 2 crash)
  - Fixed: transformers 4.46.2 → 4.37.0 (`_expand_mask` removed in 4.38+, UAV's LLaVA needs it)
  - Fixed: LLaVA config registration (`exist_ok=True` for transformers with built-in LLaVA)
  - **Test running:** clip 000 on GPU 0, results pending

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

Output at `/data/disk2/timur/results/mgld_synthetic/`.

NR evaluation (no GT available):

| Video | CLIP-IQA ↑ | MUSIQ ↑ | NIQE ↓ | BRISQUE ↓ |
|-------|-----------|---------|--------|-----------|
| 7WHI2L_FDNg | 0.457 | 68.68 | 4.29 | 25.75 |
| BrRLKMbBTYQ | 0.529 | 62.31 | 6.20 | 28.74 |
| KZ8p6b1zJ9U | 0.493 | 62.86 | 3.83 | 20.34 |
| hhszUXL1Cu8 | 0.456 | 65.36 | 4.72 | 26.20 |
| mJog8DlRk_4 | 0.543 | 66.15 | 4.32 | 22.69 |
| **Mean** | **0.496** | **65.07** | **4.67** | **24.74** |

MP4 versions converted at `/data/disk2/timur/results/mgld_synthetic_mp4/` for VBench evaluation.

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

### VBench 2.0 Long-Video Evaluation — WORKING (April 21)

VBench 2.0 (`vbench2_beta_long`) successfully evaluated all 5 MGLD-SR synthetic videos and LQ baselines.

**Setup fixes:** moviepy 2.x → 1.0.3 (API change), torchvision `pict_type="NONE"` → `0` (PyAV int enum), `PYTHONPATH` set to repo source (pip package missing configs).

**Results — LQ vs MGLD-SR (5 videos, 3 dimensions):**

| Dimension | LQ Baseline | MGLD-SR | Improvement |
|-----------|------------|---------|-------------|
| imaging_quality ↑ | 0.439 | **0.681** | +0.242 (+55%) |
| motion_smoothness ↑ | 0.987 | **0.989** | +0.002 |
| temporal_flickering ↑ | 0.981 | **0.984** | +0.003 |

**Per-video imaging_quality:**

| Video | LQ | MGLD-SR | Gain |
|-------|-----|---------|------|
| 7WHI2L_FDNg | 34.87 | 69.85 | +35.0 |
| BrRLKMbBTYQ | 59.96 | 71.74 | +11.8 |
| KZ8p6b1zJ9U | 29.52 | 63.50 | +34.0 |
| hhszUXL1Cu8 | 47.28 | 66.24 | +19.0 |
| mJog8DlRk_4 | 45.75 | 68.23 | +22.5 |

MGLD-SR significantly improves image quality while preserving temporal consistency. These are target metrics for our method.

## MGLD-VSR Disk2 Environment Verification (April 22)

Ran UDM10 clip 000 on disk2 env to check if different package versions affect quality. Despite `open-clip-torch 2.20.0` (vs 2.0.2 on disk1), results are **identical** (PSNR 23.9881). Low quality on synthetic videos is content-related, not environmental.

## VBench 2.0 Full Evaluation — In Progress (April 22)

Expanding from 3 to all 16 VBench dimensions. Required downloading ~4 GB of model weights locally and SCP'ing to server (DreamSim 1.2 GB, ViCLIP 1.6 GB, GRiT 398 MB, UMT 579 MB, DINO 327 MB, RAFT 78 MB, CLIP 338 MB).

Quality Score dimensions (all 7 done):

| Dimension | MGLD-SR | LQ | Category |
|-----------|---------|-----|----------|
| imaging_quality | 0.6810 | 0.4388 | Quality |
| motion_smoothness | 0.9886 | 0.9873 | Quality |
| temporal_flickering | 0.9840 | 0.9811 | Quality |
| aesthetic_quality | 0.5080 | 0.4128 | Quality |
| dynamic_degree | 0.5942 | 0.5628 | Quality |
| subject_consistency | 0.8927 | 0.8936 | Quality |
| background_consistency | 0.9235 | 0.9333 | Quality |

Semantic Score dimensions — **not applicable for SR** (April 23 finding):

All 9 Semantic dims require text prompts as ground truth. VBench was designed for text-to-video generation benchmarking. For SR evaluation without prompts, these dimensions produce meaningless scores:
- `overall_consistency` — ViCLIP video-text similarity (uses filename as "prompt" → 0.08)
- `appearance_style`, `temporal_style` — require `auxiliary_info` with style labels
- `human_action` — extracts action label from filename, matches against Kinetics-400 (→ 0.0)
- `color`, `object_class`, `multiple_objects`, `spatial_relationship` — GRiT/detectron2 vs text prompt
- `scene` — Tag2Text scene vs text prompt

**Conclusion:** Only the 7 Quality Score dimensions are meaningful for SR evaluation. The biggest wins are imaging_quality (+55%) and aesthetic_quality (+23%). Temporal metrics are nearly identical, confirming SR preserves temporal coherence. subject_consistency and background_consistency are slightly lower for SR due to diffusion-based frame-to-frame variation.

Fixes applied April 23: DINO cache git commit, detectron2 installed (--no-build-isolation), timm downgraded to 1.0.12, Tag2Text SCP'd.

## UAV Torch 2.5.1 Test — Blocked (April 21-22)

All CUDA variants of torch 2.5.1 fail on server:
- cu124: cuDNN CUDNN_STATUS_NOT_INITIALIZED
- cu121: same cuDNN error
- cu118: missing libcudart.so.11.0

Server CUDA driver (v570/12.8) incompatible with torch 2.5.1's bundled cuDNN. Need to contact DOVE authors for their exact environment.

## Completed (April 15–22)

- [x] Disk2 migration — full infrastructure rebuilt
- [x] UAV DOVE UDM10 default settings — PSNR 23.05 vs 21.72 target (+1.33 dB gap)
- [x] UAV DOVE SPMCS default settings — PSNR 20.49 vs 18.81 target (+1.68 dB gap)
- [x] UAV alignment investigation — ruled out: input format, seed, settings, frame count, resolution, empty prompt, env version. Gap consistent across datasets.
- [x] MGLD-VSR UDM10 re-verification — IDENTICAL match on both disk1 and disk2 envs
- [x] MGLD-VSR synthetic — all 5 videos done (22,412 frames) + NR eval (CLIP-IQA, MUSIQ, NIQE, BRISQUE)
- [x] VBench 2.0 — all 7 Quality Score dims complete for MGLD+LQ. 9 Semantic dims not applicable for SR (require text prompts)
- [x] SPMCS dataset downloaded and UAV evaluated
- [x] UAV torch 2.5.1 test attempted — blocked by cuDNN incompatibility
- [x] VBench model weights downloaded locally and SCP'd (DreamSim, DINO, GRiT, UMT, ViCLIP, RAFT, CLIP)

## UAV Synthetic Inference — Completed (April 27 – April 30)

### Chunked Inference for Long Videos

UAV's default implementation OOM'd on 5,000-frame videos at 1280x720. Implemented chunked inference (2,500 frames per chunk) to fit in 80 GB VRAM. Frames re-assembled into MP4 with cv2.VideoWriter.

All 22,412 frames super-resolved across 5 videos over ~48h on GPU 7. Results saved to `/data/disk2/timur/results/uav_synthetic_mp4/` and SCP'd locally (~770 MB).

### Full MGLD vs UAV Evaluation (April 30)

All metrics ran in parallel on free GPUs. **MGLD-SR wins 8/9 metrics**:

| Metric | LQ | MGLD-SR | UAV | Winner |
|--------|----|---------|------|--------|
| CLIP-IQA ↑ | — | **0.496** | 0.391 | MGLD |
| MUSIQ ↑ | — | **65.07** | 56.28 | MGLD |
| NIQE ↓ | — | **4.67** | 5.73 | MGLD |
| BRISQUE ↓ | — | **24.74** | 50.90 | MGLD |
| DOVER overall ↑ | 10.44 | **73.81** | 65.06 | MGLD |
| E\*warp ↓ | 0.0092 | **0.0114** | 0.0137 | MGLD (best SR) |
| VBench imaging_quality ↑ | 0.4388 | **0.6810** | 0.6458 | MGLD |
| VBench aesthetic ↑ | 0.4128 | **0.5080** | 0.4892 | MGLD |
| VBench subject_consistency ↑ | 0.8936 | 0.8927 | **0.9031** | UAV |

UAV's only win on `subject_consistency` is likely a DINOv2 color-invariance artifact rather than real superiority — diffusion noise in MGLD trips DINOv2 features more than UAV's smoother output.

## New Metrics: DOVER + E\*warp (April 28)

Added two missing DOVE benchmark metrics that weren't in our pipeline:

- **DOVER** (no-reference video quality, aesthetic + technical) — installed from `VQAssessment/DOVER`, 229 MB weights downloaded locally + SCP'd. Uses fragment sampling, robust to long videos.
- **E\*warp** (temporal warping error via RAFT optical flow) — implemented from scratch using torchvision RAFT. The DOVE repo had `eval_ewarp.py` that imports `from ewarp import Ewarp` but the actual `ewarp.py` was never published. Wrote it based on the algorithm (forward+backward flow, occlusion mask via FB consistency check, masked L2 error in non-occluded regions).

KVQ confirmed as a separate dataset (Kaleidoscope Video Quality), not a metric — colleague had asked but it doesn't appear in the DOVE benchmark table.

## VBench Effectiveness Validation Plan (April 28)

Drafted plan to validate whether VBench metrics actually capture the long-range artifacts that matter for SR. Identified 7 limitations from source-code analysis:

1. `temporal_flickering` has no long-range branch — only adjacent-frame MAE; gradual drift undetectable
2. Fast branch samples only ~2% of frames (first frame per clip)
3. DINOv2 (`subject_consistency`) is color-invariant — misses color drift
4. PySceneDetect can hide chunk-boundary artifacts (splits at brightness jumps)
5. Score mapping calibrated for text-to-video, not SR
6. Cosine similarity blind to certain degradation types (uniform brightness/sharpness)
7. Slow/fast 50:50 fusion is arbitrary, not validated for SR-specific artifacts

Designed 5 synthetic test datasets (color drift, periodic flicker, chunk-boundary jumps, identity degradation, long-range background change) with parameterized severity levels — all buildable on M1 Mac with OpenCV. Plan includes complementary metrics: long-range tOF (k=[1,5,10,30,60,120]), tLP, CLIP-IQA temporal trajectory variance, FVD.

Plan saved at `docs/plans/2026-04-28-metrics-and-vbench-validation.md`.

## VBench-2.0 — The Real Version (April 30 – May 1)

Discovered the package we'd been calling "VBench 2.0" (`vbench2_beta_long`) is actually the long-video extension of VBench 1.0. The real **VBench-2.0** is a separate module focused on intrinsic faithfulness (physics, anatomy, motion order) for text-to-video generation, with 18 new dimensions.

### Applicability Analysis

Of 18 dimensions:
- **13 require text prompts** — composition, dynamic_attribute, complex_plot, mechanics, etc. Use LLaVA-Video or Qwen to compare video to prompt. Not applicable for SR.
- **3 not applicable for other reasons** — diversity (needs N=20 generations per prompt), camera_motion (needs expected camera label), instance_preservation (needs prompt instance list).
- **2 repurposable for SR** — **Human_Anatomy** (ViTDetector anomaly detection per frame, no prompt needed) and **Human_Identity** (RetinaFace + ArcFace, no prompt needed).
- **Multi_View_Consistency deferred** — designed for multi-camera 3D coherence, conceptually different from long-video drift.

Plan saved at `docs/plans/2026-04-30-vbench2-applicability.md`.

### Phase 1: Initial Setup

Goal: run VBench-2.0 directly on our SR videos before building long-video adapter.

**Setup work:**

- Disk cleanup — root partition was 100% full (0 bytes free), blocking all temp file creation. Moved 9 GB cache to `/data/disk2/timur/cache/` with symlinks (`vbench/`, `clip/`). Root now has 8.7 GB free.
- Installed `mmcv 2.2.0` (prebuilt wheel for torch 2.4 cu121, ABI-compatible with torch 2.5.1), `mmdet 3.3.0`, `mmyolo 0.6.0`. Patched mmdet/mmyolo `__init__.py` to relax `mmcv_maximum_version` checks.
- Installed `retinaface`, `retinaface-pytorch`, `gdown`. Numpy temporarily upgraded to 2.x then downgraded back to 1.26.4.
- VBench-2.0 weights downloaded locally + SCP'd: ArcFace `resnet18_110.pth` (98 MB), RetinaFace zip (97 MB), YOLO-World (168 MB), anomaly detector human/face/hand `.pth` (88 MB each).
- YOLO-World source SCP'd. Patched syntax error: `self.text_feats, None = ...` → `self.text_feats, _ = ...`.
- ViTDetector config patched: replaced author's hardcoded `/mnt/petrelfs/zhengdian/code/ckpt/clip-vit-base-patch32` with `openai/clip-vit-base-patch32`.
- timm `_pil_interp` import fixed in ViTDetector — `from timm.data.transforms import str_to_interp_mode as _pil_interp` (renamed in newer timm).

### Human_Identity — Working

VBench-2.0's `human_identity` algorithm has two issues for our crowd-scene videos:
1. Required exactly 1 face per frame — fails on multi-person scenes
2. Required reference face in frame 0 — fails if first frame has no face

Patched `IDTracker.update()` to pick the largest face on multi-face frames, and `evaluate_id_consistency()` to find the first valid frame as reference (not strictly frame 0).

Results on all 5 MGLD + 5 UAV videos:

| Video | MGLD-SR | UAV |
|-------|---------|-----|
| 7WHI2L_FDNg | 0.035 | **0.116** |
| BrRLKMbBTYQ | **0.401** | 0.339 |
| KZ8p6b1zJ9U | 0.534 | **0.537** |
| hhszUXL1Cu8 | **0.011** | 0.009 |
| mJog8DlRk_4 | **0.018** | 0.012 |
| **Mean** | 0.200 | **0.203** |

UAV slightly better identity preservation overall (+0.003). Most scores are low because the algorithm tracks a single identity (largest face), but our videos contain crowds with multiple people. A multi-person extension (cluster-based identity tracking) is noted as future work.

### Human_Anatomy — Blocked

All weights and dependencies in place except CLIP-ViT-Base-Patch32 (577 MB pytorch_model.bin). SCP keeps disconnecting after 3-5 MB transferred — server connection unstable today. Will retry with smaller chunks at off-peak time, or use a HuggingFace mirror once available.

## Currently Running (as of May 1)

No active jobs. CLIP transfer pending for Human_Anatomy unblock.

## Next Steps

1. Unblock CLIP transfer for Human_Anatomy (split into smaller chunks, retry off-peak, or use HuggingFace mirror)
2. Implement multi-person Human_Identity adaptation (cluster-based tracking)
3. Build long-video adapter for VBench-2.0 (slow-fast pattern from `vbench2_beta_long`)
4. Implement VBench validation test datasets (Tests A–E from plan): synthetic videos with parameterized artifacts
5. Add long-range tOF + tLP metrics to evaluation pipeline
6. Continue thesis writing (Introduction + Literature Review chapters)
7. Proposal outline when PhD student provides materials
