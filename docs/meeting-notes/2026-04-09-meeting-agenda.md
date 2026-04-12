# Meeting Agenda — April 9, 2026

## Attendees
- Timur (student)
- PhD student supervisor
- Supervisor (if present)

## Presentation
- 18-slide baseline methods presentation (MGLD-VSR + Upscale-A-Video)
- Update slides with UAV RealBasicVSR gap findings + DOVE cross-validation before meeting

## Key Questions to Raise

### 1. Research Direction
- What specific architecture/approach should we pursue?
- Current fallback: "SSM replaces attention in diffusion VSR pipeline" — is this aligned?
- Any specific papers or methods the PhD student wants to build on?

### 2. Long-Video Sample Data
- Can the PhD student provide sample long-video data at this meeting?
- We need it to run baselines on long videos and document failure modes
- This is blocking Task 8 in our plan

### 3. Gov Company Dataset Status
- Any update on the full long-video dataset?
- If not coming soon, we need to commit to a fallback:
  - YouTube-VOS long clips
  - MovieNet
  - Stitched REDS sequences
  - Other public long-video sources?

### 4. Proposal Rewrite
- Is a proposal rewrite required for the new VSR topic?
- If yes, estimated ~2-3 days (reformatting method spec + lit review)

### 5. UAV Degradation Gap (share findings)
- UAV shows 5.8 dB PSNR gap on RealBasicVSR LQ (24.94 vs paper 30.79)
- Pipeline verified via DOVE paper cross-validation (23.22 vs DOVE's 21.72 for UAV)
- Param sweep ruled out hyperparameters (~0.9 dB variation)
- Root cause: degradation mismatch (random seeds)
- Question: is this expected? Do they know what seeds/params UAV authors used?
- Our approach: consistent degradation (seed 42) across all methods for fair comparison

### 6. Thesis Title
- Preliminary Chinese title: 基于状态空间模型的长视频超分辨率方法研究
- English: State-Space Model Based Video Super-Resolution for Long Videos
- Should this change based on confirmed research direction?

## Meeting Outcomes (April 9, 2026)

### 1. Research Direction
- **Still unclear.** PhD student will continue exploring i2v (image-to-video) papers and will update later.
- No specific architecture decided yet. SSM direction not confirmed or denied.

### 2. Long-Video Sample Data
- Will be collected by **another student** (not the PhD student directly).
- No timeline given — still a dependency.

### 3. Gov Company Dataset
- Dataset has been provided but is **not publicly available**.
- Will be used for internal tests only, **not for the paper**.
- A separate evaluation metric was mentioned (details TBD) — not urgent now but good to know.

### 4. Proposal Rewrite
- **Most likely needed.** Deadline: **May 31, 2026.**
- Will revisit in May — not urgent now.

### 5. UAV Degradation Gap — New Strategy: Align with DOVE
- **Decision: Don't align with UAV paper's own numbers. Align with DOVE benchmark instead.**
- DOVE provides a standardized benchmark with pre-degraded LQ data and published evaluation scripts.
- Strategy:
  1. **Validate MGLD-VSR on DOVE UDM10 LQ** — we already have the LQ data. If our MGLD-VSR results match DOVE paper's reported MGLD numbers, we know our evaluation pipeline matches theirs.
  2. **Check DOVE's evaluation code** — https://github.com/zhengchen1999/DOVE — their metrics script may differ from our pyiqa evaluation (could explain the UAV gap of +1.5 dB).
  3. **Re-validate UAV on DOVE UDM10 LQ** using DOVE's own evaluation script for exact alignment.
  4. Once aligned, we can **directly use DOVE's published comparison table** as our baseline reference.

DOVE paper comparison table (UDM10, DOVE LQ) — includes MGLD-VSR [50], VEnhancer, STAR, DOVE:

| Metric | RealBasicVSR [38] | Real-ESRGAN [56] | StableSR [5] | UAV [63] | MGLD [50] | DBVSR [9] | RealViformer [48] | DOVE (theirs) |
|--------|-------------------|------------------|--------------|----------|-----------|-----------|-------------------|---------------|
| PSNR | 24.04 | 23.65 | 24.13 | 21.72 | 24.23 | 21.32 | 23.47 | 26.48 |
| SSIM | 0.7107 | 0.6016 | 0.6801 | 0.5913 | 0.6957 | 0.6811 | 0.6804 | 0.7827 |
| LPIPS | 0.3877 | 0.5537 | 0.3908 | 0.4116 | 0.3272 | 0.4344 | 0.4242 | 0.2696 |
| DISTS | 0.2184 | 0.2898 | 0.2067 | 0.2230 | 0.1677 | 0.2310 | 0.2156 | 0.1492 |
| CLIP-IQA | 0.4189 | 0.4344 | 0.3494 | 0.4697 | 0.4557 | 0.2852 | 0.2417 | 0.5107 |
| FasterVQA | 0.7386 | 0.4772 | 0.7744 | 0.6969 | 0.7489 | 0.5493 | 0.7042 | 0.8064 |
| DOVER | 0.7060 | 0.3290 | 0.7564 | 0.7291 | 0.7264 | 0.4576 | 0.4830 | 0.7809 |
| E*warp | 4.83 | 6.12 | 3.10 | 3.97 | 3.59 | 1.03 | 2.08 | 1.77 |

### 6. VBench — Human-Perception-Aligned Evaluation
- Need to set up **VBench** (https://github.com/Vchitect/VBench) for perceptual quality evaluation aligned with human perception.
- **Problem:** VBench gives OOM errors on long videos.
- **Beta version for long videos** exists: https://github.com/Vchitect/VBench/tree/master/vbench2_beta_long
- Need to understand, test, and possibly improve it — this is a **separate and hard task**.

### 7. Thesis Title
- Not discussed — deferred until research direction is confirmed.

---

## Action Items (post-meeting)

### Immediate (this week)
1. **Run MGLD-VSR on DOVE UDM10 LQ** — validate against DOVE paper's MGLD numbers (PSNR 24.23, SSIM 0.6957, LPIPS 0.3272)
   - **Status: DONE — IDENTICAL MATCH.**
   - Used tile-based inference script (`vsr_val_ddpm_text_T_vqganfin_oldcanvas_tile.py`) for full-resolution output
   - Previous attempts: 512x512 center crop (wrong spatial alignment, ~10 dB PSNR), native resolution (UNet skip connection mismatch)
   - Result: PSNR 24.23 / SSIM 0.6957 / LPIPS 0.3272 / DISTS 0.1676 / CLIPIQA 0.4555 — matches DOVE paper exactly
2. **Check DOVE evaluation code** — clone https://github.com/zhengchen1999/DOVE, understand their metrics pipeline, compare with our pyiqa approach
   - **Status: DONE.** Key finding: DOVE uses **RGB PSNR/SSIM** by default (no `--test_y_channel`), while we use Y-channel. DOVE's `inference.sh` calls `eval_metrics.py --metrics psnr,ssim,lpips,dists,clipiqa` without Y-channel flag.
3. **Re-evaluate UAV on DOVE UDM10 LQ** using DOVE's evaluation script (not just pyiqa)
   - **Status: DONE.** DOVE eval gives PSNR 22.96 vs paper's 21.72 (+1.24 dB gap). SSIM/LPIPS/DISTS/CLIPIQA all close.
   - Gap likely from inference settings: we used `n150 g7`, UAV defaults are `n120 g6`
   - **Re-running UAV with default settings** (tmux `uav_dove_default`, GPU 5) to narrow the gap
4. **Cleaned up invalid results on server** — removed MP4 runs, bicubic, old wrong-degradation outputs

### Next priority
5. **Set up VBench** — clone repo, test on short videos, then try long-video beta
6. **Wait for long-video sample data** from other student
7. **Wait for research direction** from PhD student (i2v exploration ongoing)

### May
8. **Proposal rewrite** — deadline May 31

