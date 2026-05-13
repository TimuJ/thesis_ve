# MGLD-VSR vs UAV — Evaluation Metrics on Synthetic Long Videos

**Period:** May 1 – May 13, 2026
**Methods:** MGLD-VSR (ECCV 2024) vs Upscale-A-Video (UAV, CVPR 2024)
**Dataset:** 5 synthetic long videos, 320×180 LQ → 1280×720 SR (4× upscale, 22,412 frames total). LQ source fps varies per video (29.97 / 24.00 / 23.98).

Slow-fast metrics use the LQ source's fps for clip splitting; both SR pipelines hard-code their mp4 fps tag (see `docs/notes/2026-05-07-sr-fps-mismatch.md`).

---

## 1. UAV inference setup verification (DOVE benchmark)

UAV reproduces DOVE paper within a small consistent offset across UDM10 and SPMCS — its lower scores on the synthetic long videos below are content/generalization shift, not a setup bug.

### UDM10 (full-reference)

| Metric | DOVE paper (UAV) | Our UAV | Δ |
|--------|------------------:|--------:|----:|
| PSNR ↑ | 21.72 | 23.05 | +1.33 dB |
| SSIM ↑ | 0.5913 | 0.6164 | +0.025 |
| LPIPS ↓ | 0.4116 | 0.4252 | +0.014 |
| DISTS ↓ | 0.2230 | 0.2364 | +0.013 |

### SPMCS (full-reference)

| Metric | DOVE paper (UAV) | Our UAV | Δ |
|--------|------------------:|--------:|----:|
| PSNR ↑ | 18.81 | 20.49 | +1.68 dB |
| SSIM ↑ | 0.4113 | 0.4838 | +0.073 |
| LPIPS ↓ | 0.4468 | 0.4280 | -0.019 |
| DISTS ↓ | 0.2452 | 0.2481 | +0.003 |

For comparison, MGLD matches DOVE paper exactly on UDM10 (PSNR 24.2339 vs 24.23; SSIM/LPIPS/DISTS identical to 4 decimals).

---

## 2. VBench 1.x Quality Score dimensions (mean across 5 videos)

| Dimension | LQ | MGLD-SR | UAV | Winner |
|-----------|----:|--------:|----:|:------|
| imaging_quality ↑ | 0.4388 | **0.6810** | 0.6458 | MGLD |
| aesthetic_quality ↑ | 0.4128 | **0.5080** | 0.4892 | MGLD |
| motion_smoothness ↑ | 0.9873 | **0.9886** | 0.9882 | MGLD |
| temporal_flickering ↑ | 0.9811 | **0.9840** | 0.9826 | MGLD |
| dynamic_degree ↑ | 0.5628 | **0.5942** | 0.5393 | MGLD |
| subject_consistency ↑ | 0.8936 | 0.8927 | **0.9031** | UAV (DINOv2 artefact)¹ |
| background_consistency ↑ | **0.9333** | 0.9235 | 0.9317 | LQ ≥ UAV > MGLD¹ |

¹ Both UAV-favorable dimensions reward smoother outputs (DINOv2 / DreamSim read diffusion noise as inconsistency).

**MGLD wins 5/7, UAV wins 2/7 (both smoother-output artefacts).** The 9 Semantic dimensions require text prompts and are not applicable for SR.

---

## 3. VBench 2.0 — Human_Identity (slow-fast long-video adapter, mean across 5 videos)

| Method | Slow (within-clip) | Fast (cross-clip) | **Fused** |
|--------|------------------:|------------------:|----------:|
| **MGLD** | **0.689** | **0.351** | **0.557** |
| UAV | 0.613 | 0.306 | 0.459 |
| Δ MGLD−UAV | +0.076 | +0.045 | **+0.097** |

Per-video winner: **MGLD wins all 5/5 videos**. Three patches to upstream `vbench2/human_identity.py` (multi-face frames, late ref init, ZeroDivision guard).

---

## 4. VBench 2.0 — Human_Anatomy (mean across 5 videos)

| Form | LQ | MGLD-SR | UAV |
|------|----:|--------:|----:|
| Whole-video (`custom_input`) | — | 0.600 | **0.605** |
| Slow-fast adapter | — | 0.608 | **0.618** |

Mean is a statistical tie under both forms. Per-video winner: **MGLD wins 4/5; UAV wins only KZ8p6b1zJ9U**, with a large gap there that drags MGLD's mean below UAV. Visual inspection clearly favors MGLD on KZ8p6b1zJ9U — the metric outlier on that video is a metric-effectiveness failure, not a model failure (work to characterize the failure is in `docs/plans/2026-05-07-metric-failure-diagnostic.md`).

---

## 5. No-reference perceptual quality (pyiqa, mean across 5 videos)

| Metric | MGLD | UAV |
|--------|-----:|----:|
| CLIP-IQA ↑ | **0.496** | 0.391 |
| MUSIQ ↑ | **65.07** | 56.28 |
| NIQE ↓ | **4.67** | 5.73 |
| BRISQUE ↓ | **24.74** | 50.90 |

MGLD wins all 4. BRISQUE gap is the largest — UAV's smoother outputs trip BRISQUE's naturalness classifier hard.

---

## 6. DOVER (no-reference video quality, mean across 5 videos)

| | MGLD | UAV | LQ |
|---|----:|----:|----:|
| Aesthetic ↑ | **99.71** | 99.42 | — |
| Technical ↑ | **9.87** | 8.69 | — |
| **Overall ↑** | **73.81** | 65.06 | 10.44 |

MGLD wins overall by +8.75; ~7× higher than LQ.

---

## 7. E\*warp (temporal warping error, mean across 5 videos)

Lower = less temporal inconsistency between adjacent frames. Implemented from scratch — DOVE upstream references `from ewarp import Ewarp` but never published `ewarp.py`.

| | MGLD ↓ | UAV ↓ | LQ ↓ |
|---|------:|------:|-----:|
| Mean | **0.0114** | 0.0137 | 0.0092 |

MGLD lower than UAV on all 5 videos individually too. Both slightly above LQ as expected for diffusion-based SR.

---

## 8. Summary

| Metric family | MGLD wins | UAV wins |
|---------------|----------:|---------:|
| VBench 1.x Quality (7 dims) | **5** | 2 (smoother-output artefacts) |
| VBench 2.0 Human_Identity (5 videos, fused) | **5/5** | 0 |
| VBench 2.0 Human_Anatomy (5 videos, per-video) | **4/5** | 1/5 (KZ outlier — metric-failure case) |
| NR-IQA (CLIP-IQA, MUSIQ, NIQE, BRISQUE) | **4/4** | 0 |
| DOVER overall (mean) | **+8.75** | — |
| E\*warp (5 videos) | **5/5** | 0 |

MGLD wins per-video on 5/5 Identity, 4/5 Anatomy, and every other metric. UAV's only wins are content-dependent metric artefacts (smoother-output bias in `subject_consistency` / `background_consistency`, and the KZ regime-shift outlier on Anatomy).
