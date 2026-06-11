# Benchmark Completion Plan — LR-VCC Thesis Experiments (June 11 → July 1 freeze)

**Status:** approved direction, pending spec review
**Date:** 2026-06-11
**Decision owners:** Timur (with PhD-student supervisor informed)

## Context

The thesis contribution is **benchmark-centric**: LR-VCC (Long-Range Video Consistency
Composite) metric + synthetic artefact validation suite + evaluation of real SR models.
Current state: 6 artefact families validated on 2 base videos (7/12 conditions clean);
`background_drift` on hhsz is the strongest signal (Δ −0.276); the 7WHI single-face base
has 3 documented failure modes sharing one root cause (slow-fast pooling on
one-subject-only content).

Binding constraints:
- **July 15, 2026** — thesis blind-review submission (chapters still contain old VOS content)
- **July 1, 2026** — hard experiment freeze; writing runs in parallel from ~June 19
- Server: GPU 0 primary / GPU 7 backup, ~30 GB disk free, Identity slow-fast stage
  ≈ 100 min per 10 videos

The two gaps reviewers will attack:
1. **n=2 base videos** — every verdict table is hhsz vs 7WHI
2. **No real-model discrimination result** — LR-VCC validated only on synthetic artefacts

## Week 1 (Jun 12–18) — Scale synthetic validation + two cheap fixes

### 1a. Base videos: 2 → 8
Selection criteria: >1 min duration; deliberate coverage of failure-mode axes —
≥2 additional single-face (so the 7WHI pathology becomes a category finding),
2 multi-face, 1–2 no-face/scene-dominant. Candidate pool: KZ8p6b1zJ9U, BrRLKMbBTYQ,
mJog8DlRk_4 (already on server) + picks from the VBench long-video set.
Per new base: extract reference face + reference background, precompute Detectron2
human masks, generate 6 artefacts × 5 severities.

Fallback if compute or disk blocks: 4 new bases (6 total).

### 1b. Fix 1 — variance-gated sub-metric I
Per-face embedding variance below threshold → I abstains (reliability → 0) instead of
emitting an inverted/flat score. Local implementation + unit tests first, then included
in all Week 1 eval runs.

### 1c. Fix 2 — curated reference scenes
Script computing CLIP-image distance between candidate reference frames and each base
video; accept a reference only if distance > τ. Re-extract the 7WHI background reference
with it (directly kills the documented inversion mechanism).

### Compute / disk budget
6 new bases × 30 clips = 180 new videos. Identity stage ≈ 30 GPU-h → split GPU 0 + 7
≈ 15 h each ≈ 2 overnight runs. Disk rescue if needed: prune raw-frame directories
`results/mgld_synthetic` (23 GB) and `results/uav_synthetic` (21 GB) — mp4 versions exist.

### Deliverable
6-artefact × 8-base verdict matrix — centerpiece figure of the methodology chapter.

## Week 2 (Jun 19–25) — Real-model discrimination

### 2a. Model roster
Substrate: the existing 5-video long-video set (hhszUXL1Cu8, 7WHI2L_FDNg, KZ8p6b1zJ9U,
BrRLKMbBTYQ, mJog8DlRk_4). MGLD-VSR and Upscale-A-Video outputs already exist for all 5.
Add: **RealESRGAN per-frame** as the zero-temporal-modeling lower anchor (cheapest to run,
cleanest contrast). Plus classmate-sourced outputs (RealBasicVSR, RVRT, diffusion-based —
whatever they have).

New Week 1 bases stay synthetic-validation-only; classmates are asked for 5 videos, not 8.

### 2b. Classmate submission spec
We provide the LR source versions of the 5 videos (download link). Requirements:
outputs on our LR inputs, full duration, native fps preserved, resolution ≥ ours,
CRF ≤ 18, filename `<model>_<base_id>.mp4`. Outputs on *their own* sources are
acceptable only for a metric-agreement side-study, not the ranking table — tell them
which we prefer.

**Precondition to verify first:** the original LR inputs for the 5 videos still exist on
the server. If only HR sources + SR outputs survive, regenerate LR with the same
degradation pipeline before sending anything out.

### 2c. Discrimination experiment
Full metric battery + LR-VCC on all model outputs; alongside, standard per-frame metrics
(PSNR/SSIM/LPIPS) and short-range tOF on the same outputs. Claim under test:
**LR-VCC separates temporally-modeled methods from frame-wise ones where per-frame
metrics rank them incorrectly or cannot distinguish them.**

### 2d. Optional human anchor (do if classmates engaged via 2b)
Timur + 2–3 classmates, pairwise preferences on ~15–20 clip pairs ("which has more
consistent identity/colour over the minute?") → Spearman correlation with LR-VCC.
~1 hour total human time; strongest reviewer-convincing number a metric thesis can have.

### Deliverable
Model ranking table + LR-VCC-vs-standard-metrics disagreement analysis
(+ human correlation if 2d happens).

## Week 3 (Jun 26 – Jul 1) — Ablations + freeze

All recompute from cached per-stage JSONs — no GPU; this week absorbs Week 1/2 overruns.

- Leave-one-out sub-metric ablation (drop each of A/T/I/D/E, recompute verdict matrix;
  show which artefact families each sub-metric uniquely catches)
- Sensitivity sweeps: softmax τ ∈ {0.1, 0.2, 0.5}; color_hist α; slope β —
  show conclusions stable around the production point (uniform tOF, α=0.394, β=200)
- Re-run stragglers; **hard freeze July 1**

## Writing track (parallel from Week 2)

- Switch `zjuthesis.tex` to `Period=paper`; rewrite chapter skeleton
- Methodology chapter first (~70% exists in the proposal)
- Experiments chapter fills as results land
- Jul 2–14: pure writing, figures, `BlindReview=true`, proofread

## Deliberately cut (future work in thesis)

- Deep slow-fast fix (body-embedding anchors) — variance gate is the shipped mitigation
- 7th artefact (subject swap)
- Perceptual (CLIP-distance) trajectory replacing Lab trajectory in sub-metric E
- Large-scale human study

## Risks

| Risk | Mitigation |
|---|---|
| Disk fills during 180-video generation | Prune raw-frame dirs (44 GB reclaimable) before launch |
| Classmate outputs late/absent | Ranking table works with MGLD/UAV/RealESRGAN alone (3 models) |
| LR inputs lost | Regenerate with original degradation pipeline (verify Week 2 day 1) |
| Variance gate threshold needs tuning | Calibrate on existing hhsz/7WHI data where ground truth is known |
| GPU 0/7 occupied | Stages are tmux-resumable; schedule overnight |
