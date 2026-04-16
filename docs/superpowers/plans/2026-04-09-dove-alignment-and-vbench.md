# Post-Meeting Action Plan — DOVE Alignment + VBench Setup

**Date:** 2026-04-09
**Last updated:** 2026-04-12
**Context:** After April 9 meeting, evaluation strategy shifted to DOVE benchmark alignment. Also need VBench for perceptual evaluation.

**Goal:** Align our evaluation pipeline with DOVE benchmark so we can directly use their published comparison table. Set up VBench for human-perception-aligned metrics.

**Status:** Tasks 1-3 mostly done. MGLD-VSR matches DOVE paper exactly. UAV re-running with default settings. Tasks 4-5 not started.

---

## Task 1: Check DOVE evaluation code — DONE

**Why:** Our pyiqa results for UAV on DOVE LQ gave PSNR 23.22 vs DOVE paper's 21.72 (+1.5 dB gap). The gap may be due to different metric implementations. If we use DOVE's own eval code, we should match their numbers exactly.

**Finding:** DOVE uses RGB PSNR/SSIM by default (no `--test_y_channel`), our pyiqa uses Y-channel. This explains ~0.3 dB of the gap. The rest is inference settings.

- [x] **Step 1: Clone DOVE repo on server**

```bash
ssh -i ~/.ssh/id_ed25519_timuj Timur@223.109.239.43
cd /data/disk1/timur
git clone https://github.com/zhengchen1999/DOVE.git
```

- [x] **Step 2: Read DOVE evaluation scripts**

Identify how they compute PSNR, SSIM, LPIPS, DISTS, CLIP-IQA, FasterVQA, DOVER, E*warp. Compare with our pyiqa approach:
- Do they use Y-channel or RGB for PSNR/SSIM?
- What LPIPS backbone (AlexNet vs VGG)?
- Border crop settings?
- Per-frame averaging vs per-clip averaging?

- [x] **Step 3: Check DOVE environment requirements**

Install any missing deps. DOVE likely needs pyiqa or IQA-PyTorch but may use specific versions or custom code.

---

## Task 2: Run MGLD-VSR on DOVE UDM10 LQ — DONE (IDENTICAL MATCH)

**Why:** We already ran MGLD-VSR on RealBasicVSR LQ (PSNR 26.48, verified). Now we need to run it on DOVE's own LQ data to validate against DOVE paper's MGLD number (PSNR 24.23).

- [x] **Step 1: Verify DOVE UDM10 LQ data exists on server**

```bash
ssh -i ~/.ssh/id_ed25519_timuj Timur@223.109.239.43 "ls /data/disk1/timur/data/UDM10/LQ/"
```

We already have this from earlier experiments (DOVE LQ was used for UAV experiments 1-2).

- [x] **Step 2: Run MGLD-VSR inference on DOVE UDM10 LQ** — Used tile script (`vsr_val_ddpm_text_T_vqganfin_oldcanvas_tile.py`), not standard script. Standard script center-crops to 512x512 which misaligns with GT.

```bash
ssh -i ~/.ssh/id_ed25519_timuj Timur@223.109.239.43 << 'REMOTE'
tmux new-session -d -s mgld_dove "
eval \"\$(/data/disk1/timur/miniconda3/bin/conda shell.bash hook)\"
conda activate mgldvsr
cd /data/disk1/timur/thesis_ve/experiments/baselines/mgld_vsr/repo

python scripts/vsr_val_ddpm_text_T_vqganfin_w_latent.py \
    --config configs/mgldvsr/mgldvsr_512_realbasicvsr_deg.yaml \
    --ckpt checkpoints/mgldvsr_unet.ckpt \
    --vqgan_ckpt checkpoints/vqgan_ckpt.ckpt \
    --seqs-path /data/disk1/timur/data/UDM10/LQ \
    --outdir /data/disk1/timur/thesis_ve/experiments/baselines/results/mgld_vsr/UDM10_dove \
    --latent-dir /tmp/mgld_latents \
    --ddpm_steps 50 --dec_w 1.0 --colorfix_type adain --select_idx 0 --n_gpus 1
"
REMOTE
```

- [x] **Step 3: Evaluate with our pyiqa** — Skipped, DOVE eval is the reference

```bash
/data/disk1/timur/miniconda3/envs/vsr/bin/python experiments/baselines/evaluate_pyiqa.py \
    --results experiments/baselines/results/mgld_vsr/UDM10_dove \
    --gt /data/disk1/timur/data/UDM10/GT \
    --output experiments/baselines/results/mgld_vsr/mgld_UDM10_dove_pyiqa.json
```

- [x] **Step 4: Evaluate with DOVE's eval code** — PSNR 24.23, SSIM 0.6957, LPIPS 0.3272, DISTS 0.1676, CLIPIQA 0.4555

Run DOVE's own evaluation script on the same results. Compare numbers with our pyiqa output. This will tell us if the metric implementations differ.

- [x] **Step 5: Compare with DOVE paper** — IDENTICAL. All metrics match to 4 decimal places.

DOVE paper reports MGLD-VSR [50] on UDM10: PSNR 24.23, SSIM 0.6957, LPIPS 0.3272.

If our numbers match → pipeline aligned. If not → investigate metric implementation differences.

---

## Task 3: Re-evaluate UAV on DOVE UDM10 LQ with DOVE's eval code — IN PROGRESS (+1.33 dB gap)

**Why:** Our pyiqa gave UAV PSNR 23.22 vs DOVE paper's 21.72. Using DOVE's own eval code should resolve whether the gap is metric implementation or inference settings.

- [x] **Step 1: Run with DOVE defaults (n120 g6 s30) + DOVE eval** — PSNR 23.05 (10/10 clips, DOVE eval_metrics.py RGB)
- [x] **Step 2: Test MKV vs PNG input** — Identical results (22.3843 on clip 000 for both)
- [x] **Step 3: Test empty prompt** — PSNR went UP to 22.61 (wrong direction, blurrier output)
- [ ] **Step 4: Full UDM10 with empty prompt** — Running on GPU 7 (10 clips + auto-eval)
- [ ] **Step 5: Test with torch 2.5.1** — Setting up uav25 env (DOVE uses torch >= 2.5.0, most likely cause)

| Source | PSNR | SSIM | LPIPS | DISTS |
|--------|------|------|-------|-------|
| DOVE paper (UAV) | 21.72 | 0.5913 | 0.4116 | 0.2230 |
| Our n150 g7 (DOVE eval) | 22.96 | 0.6183 | 0.4050 | 0.2194 |
| **Our n120 g6 (DOVE eval)** | **23.05** | **0.6164** | **0.4252** | **0.2364** |

**Ruled out:** input format (MKV=PNG), seed (hardcoded 10), frame count, resolution, eval script.
**Most likely cause:** PyTorch version difference (2.0.1+cu117 vs >= 2.5.0). Diffusion models produce numerically different outputs across torch versions even with fixed seed.

---

## Task 4: Set up VBench — INSTALLED, needs debugging

**Why:** VBench provides human-perception-aligned video quality metrics. Needed for our evaluation, but has OOM issues on long videos.

- [x] **Step 1: Install VBench** — `pip install vbench` in `vbench` conda env (v0.1.5, torch 2.5.1+cu121)

- [ ] **Step 2: Test on short video** — Test run crashed with PyTorch distributed error. Needs single-GPU investigation.

```bash
# Old path (disk1 dead):
# cd /data/disk1/timur
# New path:
cd /data/disk2/timur
# VBench installed via pip, not repo clone
```

- [ ] **Step 2: Set up environment**

Check requirements. VBench likely needs a separate conda env. Install dependencies.

- [ ] **Step 3: Test on short video**

Run VBench on one of our UDM10 SR results (32 frames, should fit in memory).

- [ ] **Step 4: Test long-video beta**

```bash
cd VBench/vbench2_beta_long
```

Read the README, understand the approach. Test on a longer sequence. Document OOM threshold.

- [ ] **Step 5: Investigate OOM fix**

Profile memory usage. Possible approaches:
- Chunk-wise evaluation (process N frames at a time)
- Reduce batch size
- Offload features to CPU between chunks
- Use gradient checkpointing if applicable

This is a hard problem — document findings even if not fully solved.

---

## Task 5: Update evaluation infrastructure

**Why:** After Tasks 1-3, we may need to add new evaluation scripts or update existing ones.

- [ ] **Step 1: Add DOVE-aligned evaluation script if needed**

If DOVE uses different metric settings, create `evaluate_dove.py` that matches their exact pipeline.

- [ ] **Step 2: Add DISTS, CLIP-IQA, FasterVQA, DOVER, E*warp metrics**

DOVE's table includes metrics we don't currently compute. Extend our evaluation to cover all of them for completeness.

- [ ] **Step 3: Update target_metrics.md with DOVE-aligned results**

Replace the RealBasicVSR-based comparison with DOVE-aligned comparison.

---

## Priority Order (updated April 12)

1. ~~**Task 1** (check DOVE eval code)~~ — DONE
2. ~~**Task 2** (MGLD-VSR on DOVE LQ)~~ — DONE, IDENTICAL MATCH
3. **Task 3** (re-evaluate UAV with DOVE eval) — IN PROGRESS, re-running with default settings
4. **Task 5** (update eval infrastructure) — partially done (target_metrics.md updated)
5. **Task 4** (VBench) — NOT STARTED

## Currently Running (as of April 15 — disk2 migration)

**Note:** disk1 failed April 12. All infrastructure rebuilt on `/data/disk2/timur/` on April 15.

| tmux session | Task | GPU | Status |
|-------------|------|-----|--------|
| `uav_dove` | UAV DOVE UDM10 default (n120 g6 s30) | 2 | Re-running all 10 clips |
| `mgld_ckpt` | MGLD-VSR checkpoint downloads | — | Downloading (~12 GB) |
| `mgld_env` | MGLD-VSR conda env setup | — | Installing |
| `vbench_setup` | VBench conda env | — | Installing |

Previous disk1 sessions (lost):
- UAV DOVE default: was 7/10 clips — re-running from scratch
- UAV VideoLQ NR: was 43/50 clips — needs re-run after DOVE alignment done
