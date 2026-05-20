# Master's Thesis Proposal — Outline

**Author:** Timur Iakshibaev  
**Topic:** Long-Range Video Consistency Evaluation for Video Super-Resolution  
**Institution:** Zhejiang University, College of Computer Science  
**Date:** May 2026

---

## 1. Background and motivation (1–1.5 pages)

- **Video SR overview**: Advances in diffusion-based and transformer-based SR methods; shift from 4× single-image to long-video upscaling (30–100+ frames).
- **Problem statement**: Long-video SR is under-served by existing evaluation metrics. Most benchmarks focus on:
  - Full-reference metrics (PSNR, SSIM) designed for single-image SR; don't capture temporal consistency on long sequences.
  - Perceptual metrics (LPIPS) that reward smooth outputs over detail-preserving ones.
  - Temporal metrics (tOF k=1) that measure adjacent-frame stability but miss long-range consistency.
- **Existing metric families**:
  - Full-reference: PSNR, SSIM, LPIPS, DISTS (reference-dependent, not applicable to SR in deployment).
  - No-reference single-aspect: CLIP-IQA, MUSIQ (appearance), VBench-1 subject_consistency / background_consistency (temporal perception via DINOv2).
  - Video quality (VBench-2.0): Human_Identity, Human_Anatomy (learned anomaly detection; content-dependent).
  - Temporal pixel-level: E*warp, tOF (optical flow consistency).
  - Temporal perceptual: tLP (LPIPS-based warping distance).
- **Thesis problem**: Long-video SR needs metrics that:
  - Capture multi-time-scale consistency (adjacent frames ≠ long-range structure).
  - Are robust to diffusion-style detail without rewarding mere smoothness (learned representations trained on pristine HR data misinterpret detail as noise).
  - Avoid content-dependent regime shifts (e.g., metric inversions on close-up content).
  - Are no-reference (match real deployment where HR ground truth is unavailable).

---

## 2. Literature review (1–2 pages)

- **VSR methods**:
  - MGLD-VSR (ECCV 2024): Diffusion-based latent-space upscaling; detail-preserving.
  - Upscale-A-Video (UAV, CVPR 2024): T2V-inspired SR; smooth outputs.
  - DOVE (CVPR 2024): Benchmark suite for VSR; includes UDM10, SPMCS test sets.
  - Recent direction: fusion of state-space models (SSMs) with diffusion priors for long-context consistency (e.g., Po et al. 2505.20171).
- **Existing temporal metrics**:
  - tOF / tLP (TecoGAN, Chu et al. 2020): Optical flow consistency and perceptual variant; adjacent-frame focused.
  - E*warp (DOVE): Flow-based warping error; still k=1 dominant.
  - VBench-1 subject_consistency, background_consistency: DINOv2-based perception; known bias against diffusion.
- **Existing perceptual metrics**:
  - CLIP-IQA, MUSIQ, NIQE, BRISQUE: No-reference quality; varying robustness to SR style.
  - DOVER: Multi-modal video quality metric; combines technical and aesthetic axes.
  - VBench-2.0: Semantic and identity metrics via ViT anomaly detection; subject-specific identity tracking.
- **VBench-2.0 and long-video SR alignment**:
  - Human_Identity (slow-fast): Leverages temporal clip splitting for cross-clip face/person consistency. Applicable to SR evaluation.
  - Human_Anatomy (anomaly ViT): Detects out-of-distribution anatomical features; content-dependent calibration failure on close-ups.
  - Limitations: Single aspect per metric; no multi-scale temporal composition; lack of reliability weighting.

---

## 3. Preliminary work (2–3 pages)

### Baseline setup and verification
- **MGLD vs UAV on 5 synthetic long videos**: DOVE benchmark setup (320×180 LQ → 1280×720 SR, 4× upscaling).
- **UAV reproduction verification**: Within +1.33 dB PSNR on DOVE paper's UDM10 / SPMCS results; setup is correct; lower scores on synthetic videos are generalization shift, not setup bugs.

### Comprehensive metric evaluation (reference: `results/uav_mgld_evaluation_metrics.md`)
- **VBench-1.x Quality dimensions (5 videos, per-method means)**:
  - MGLD wins 5/7 dimensions: imaging_quality, aesthetic_quality, motion_smoothness, temporal_flickering, dynamic_degree.
  - UAV wins 2/7: subject_consistency (DINOv2 artefact), background_consistency.
  - **Finding**: Both UAV-favorable dimensions reward smoother outputs; metric misalignment with perceptual preference on diffusion-style SR.
  
- **VBench-2.0 Human_Identity (slow-fast long-video adapter)**:
  - MGLD: 0.689 slow, 0.351 fast, 0.557 fused.
  - UAV: 0.613 slow, 0.306 fast, 0.459 fused.
  - **MGLD wins all 5/5 videos**; identity preservation favors detail-preserving SR.
  
- **VBench-2.0 Human_Anatomy (whole-video + slow-fast aggregation)**:
  - Mean: MGLD 0.608 / UAV 0.618 (tied).
  - Per-video: MGLD wins 4/5; **UAV flips on KZ8p6b1zJ9U** (MGLD 0.137, UAV 0.476). **This is the key failure case**.
  
- **No-reference IQA (CLIP-IQA, MUSIQ, NIQE, BRISQUE)**:
  - MGLD wins all 4 metrics. BRISQUE gap is largest (MGLD 24.74 vs UAV 50.90).
  
- **DOVER (video quality)**:
  - MGLD overall +8.75 above UAV.
  
- **E*warp (temporal warping error)**:
  - MGLD lower on all 5 videos.

### KZ8p6b1zJ9U regime-shift characterization (reference: `docs/notes/2026-05-13-kz-regime-shift-trigger.md`)
- **The problem**: On one video (KZ, a close-up talking-head), Anatomy flips from MGLD-favoring to UAV-favoring, contradicting all other metrics and visual inspection.
- **Finding — close-up body parts are a signal**: 
  - KZ has hand bbox p50 = 18% of frame (vs. 1% on the lowest video).
  - Anatomy anomaly-probability p_abnormal distribution shifts dramatically on close-ups (median jumps from 0.006–0.11 to 0.32–0.42).
  - Detector loses baseline confidence and operates near the decision boundary; small SR-style differences become decisive.
- **Negative result**: Close-up frame filtering does NOT rescue KZ; continuous aggregation (1 - mean p_abnormal) only halves the gap (from -0.339 to -0.148). The failure is not localized to a clean per-frame predicate.
- **Interpretation**: VBench-2.0's anomaly classifier was trained on mid-shot person scales and is miscalibrated on diffusion-SR outputs of close-up content. **Metric-effectiveness finding**: `Human_Anatomy` is unreliable on close-up video SR without additional reliability weighting.

### Long-range temporal consistency findings (reference: `docs/notes/2026-05-14-tof-tlp-long-range-results.md`)
- **tOF crossover at k=5–10**:
  - k=1 (adjacent frames): UAV wins (0.0177 vs MGLD 0.0216).
  - k≥10 (long-range): MGLD wins (k=120: 0.1441 MGLD vs 0.1682 UAV, Δ −0.0241).
  - **Per-video tOF**: MGLD wins 4/5 at k≥10; on KZ (the Anatomy-flip video) tOF flips back to MGLD at k=10+, agreeing with perception.
  
- **tLP (LPIPS-based temporal distance)**:
  - UAV wins systematically across all k (all methods, all videos).
  - Mechanism: LPIPS rewards self-similarity; UAV's smoothness means warped frames look more like the source under LPIPS feature distance.
  - **Same smoother-output bias as VBench subject_consistency (DINOv2)** — three independent learned representations all penalize diffusion-detail.
  
- **Mask coverage flag** (methodological):
  - Coverage drops from 93% (k=1) to 11% (k=120). UAV coverage consistently lower; smooth textures give optical flow estimator fewer features.
  - **Sampling bias caveat**: Long-k comparisons valid pixel subsets differ per method.

### Why this matters — Structural bias across metric families
- **Three independent learned representations** show the same smoother-output bias:
  - VBench-1 subject_consistency (DINOv2): rewards UAV over MGLD.
  - VBench-2.0 Human_Anatomy on close-up (anomaly ViT): rewards UAV on KZ.
  - tLP (LPIPS): rewards UAV across all k.
- **Common root cause**: Representations trained against "pristine real photo / HR" data treat diffusion-detail (per-frame variation, micro-textures) as out-of-distribution / anomalous.
- **Cross-metric agreement on other videos**: NR-IQA (4/4 MGLD), VBench-1 Quality (MGLD 5/7), Identity (MGLD 5/5), DOVER (MGLD), E*warp (MGLD 5/5). **MGLD wins across 5 independent metric families on 4/5 videos; only fails on Anatomy and tLP on specific videos**.
- **Cross-metric agreement on KZ**: On the Anatomy-flip video, tOF k≥10 agrees with perception (MGLD long-range stable); tLP disagreement is consistent with its known bias. 6 of 8 metric families favour MGLD on KZ.

### FPS mismatch discovery and fix (reference: `docs/notes/2026-05-07-sr-fps-mismatch.md`)
- **Methodological contribution**: Slow-fast clip splitting uses LQ source fps, but SR pipelines may hard-code output fps (MGLD / UAV differ). Mismatch causes clip boundary misalignment.
- **Fix**: Force consistent fps in slow-fast aggregate; flag in per-video diagnostics.

---

## 4. Proposed method — LR-VCC (1.5–2 pages)

### Overview
- **LR-VCC** (Long-Range Video Consistency Composite) is a no-reference composite metric for ranking SR methods on long-video temporal, perceptual, and identity consistency.
- **Core insight**: Single-aspect, single-time-scale metrics are inadequate for long-video SR. Content-aware reliability-weighting of multiple sub-metrics resolves the documented failure modes (smoother-output bias and regime-shift sensitivity).
- **Architecture**: Three sub-metrics (Appearance, Temporal, Identity), each outputting `(score, reliability)` pair. Per-video composition via softmax-weighted log-mean; per-method aggregation via exclusion of low-confidence videos and geometric mean.

### Sub-metric A — Appearance stability
- **What it measures**: Perceptual quality stability across the video + mean quality.
- **Build**:
  - Per-frame CLIP-IQA (cheap, ~5 min/video on GPU).
  - `A_score = clamp(mean(quality) − λ·std(quality), 0, 1)` with λ=0.5 (tunable).
  - Mean rewards high quality; std penalizes drift. Symmetric: stably-blurry method has low std but low mean too.
- **Reliability test**:
  - Drop weight if std(quality) < 0.02 (sub-metric not discriminating).
  - Drop if both methods saturate mean(quality) > 0.98 (ceiling regime).
  - `A_reliability ∈ [0, 1]` via sigmoid penalty.
- **Why this avoids smoother-output bias**: CLIP-IQA trained on diverse content, not pristine HR. Drift-variance term is symmetric; penalizes both blurriness and instability.

### Sub-metric T — Temporal stability
- **What it measures**: Multi-time-scale consistency with long-range (k≥10) stability weighted heavier than adjacent frames (k=1).
- **Build**:
  - tOF at k ∈ {1, 5, 10, 30, 60, 120} via existing script.
  - `T_score = 1 − weighted_mean(tOF_k)` where `weight(k) = log(1+k)` (long-k heavier).
  - Diagnostic: report tLP separately, NOT in composite (known bias).
- **Reliability test**:
  - Drop k if mask_coverage < 0.10.
  - Drop T if overall coverage low.
- **Why this avoids smoother-output bias**: Long-k weighting shifts away from the adjacent-frame regime where smoother trivially wins. tOF (pixel L2) avoids LPIPS self-similarity bias entirely.

### Sub-metric I — Identity preservation
- **What it measures**: Stability of identifiable people / objects across the video.
- **Build (v1)**:
  - Slow-fast human_identity_long.py adapter with fps correction.
  - `I_score = fused_slow_fast`.
  - Future: multi-person Identity v2 when available.
- **Reliability test**:
  - `face_detection_rate = n_clips_with_faces / n_clips`. Drop if < 0.20.
  - `bbox_area_p50` close-up flag: if face/hand bbox > 5% of frame, partial downweight (similar content-dependence as Anatomy).
- **Why this avoids smoother-output bias**: ArcFace has known bias against high-frequency noise, but slow-fast averaging within clips reduces it. Reliability test downweights videos where identity is sparse or content is close-up.

### Reliability-weighting composition
- **Inputs**: Three `(s_score, s_reliability)` pairs, all in [0, 1].
- **Reliabilities derived from regime indicators**:
  ```
  A_reliability = 1 − sigmoid_penalty(quality_saturation, drift_too_small)
  T_reliability = 1 − sigmoid_penalty(mean_mask_coverage_low)
  I_reliability = 1 − sigmoid_penalty(low_face_rate, closeup_bbox_ratio)
  ```
- **Log-domain composition** (preserves no-compensation property):
  ```
  weights = softmax([A_reliability, T_reliability, I_reliability] / temperature)
  LR_VCC(V, M) = exp( Σ_s weights[s] · log(s_score + ε) )
  ```
  with temperature=0.2 (sharp softmax; most reliable sub-metric dominates) and ε=1e-6 (numerical stability).
- **Low-confidence exclusion**: If all reliabilities < 0.2, video is `low_confidence` and excluded from method mean by default; reported with flag.

### Why this resolves the documented failures
- **Smoother-output bias**: Reliability weighting downweights LPIPS-based metrics (Anatomy on close-ups, tLP globally) when their regime assumptions are violated. Long-k temporal weighting shifts away from the adjacent-frame regime where smoothness trivially wins.
- **Regime-shift sensitivity**: Per-video reliability tests are keyed to the documented failure modes (close-up detection for Anatomy, mask coverage for tOF). Soft sigmoid transitions prevent sharp metric cliffs.
- **Multi-scale temporal**: Unlike tOF k=1 or E*warp, LR-VCC's T sub-metric captures both adjacent-frame stability and long-range consistency. The k-weighting responds to the documented crossover finding.

---

## 5. Preliminary validation (1 page)

### Layer 1 — Perceptual agreement on 5 synthetic videos
- **Validation goal**: Aggregate LR_VCC(MGLD) > LR_VCC(UAV) and MGLD wins per-video on ≥4/5.
- **Expected result**: MGLD agrees with visual inspection across 5 synthetic videos.
- **Pass criterion**: Per-video table comparing LR-VCC vs existing metrics (VBench Anatomy, tLP) for the two methods. MGLD ≥4/5 per-video.

### Layer 2 — Flip-resistance on KZ (the key test)
- **Validation goal**: LR-VCC(MGLD, KZ) > LR-VCC(UAV, KZ).
- **Why this is the critical test**: Anatomy flips on KZ for documented structural reasons (anomaly classifier miscalibration on close-up content). tLP also flips on KZ (LPIPS self-similarity bias). LR-VCC should NOT flip because:
  - Reliability weighting downweights the Anatomy-style failure (close-up signal lowers A_reliability or I_reliability).
  - Temporal sub-metric (long-k weighted tOF) agrees with perception on KZ (MGLD wins at k≥10).
- **Pass criterion**: `LR_VCC(MGLD, KZ) > LR_VCC(UAV, KZ)` and per-sub-metric diagnostics show why (T_reliability dominates, A_reliability downweighted).

### Per-video results table (preliminary)
- Columns: Video, LR-VCC MGLD / UAV, VBench Anatomy MGLD / UAV, tLP MGLD / UAV, tOF-k=120 MGLD / UAV, Per-video reliability weights.
- Rows: 5 synthetic videos + aggregate mean.
- **Expected pattern**: LR-VCC agrees with perception (MGLD ≥4/5); Anatomy disagrees on KZ; tLP disagrees on KZ; tOF-k=120 agrees with perception.

### Diagnostics output
- Per-video JSON with sub-metric scores / reliabilities, weights used, low-confidence flag, closeup_indicator, mask_coverage, face_detection_rate.
- Transparency: reviewers can see where the composite is driven from (no black box).

---

## 6. Plan and timeline (0.5–1 page)

### Immediate (by May 31 — proposal deadline)
- Implement LR-VCC infrastructure (appearance.py, temporal.py, identity.py, reliability.py, composite.py).
- Run Layer 1 validation on 5 synthetic videos.
- Run Layer 2 validation on KZ; confirm flip-resistance.
- Generate per-video table and diagnostics.
- Document findings in `docs/notes/2026-05-25-lr-vcc-validation.md`.

### June 2026 (thesis phase, before blind review)
- **Layer 3 validation**: Parameterized synthetic test datasets with severity sweeps (color drift, periodic flicker, chunk-boundary jumps, identity degradation, long-range background change).
  - Criterion: LR-VCC monotonically responds to severity on at least one dataset; response smoother than any individual sub-metric.
- **Multi-person Identity v2**: If available, swap in v2 cluster-purity + LQ-reference variants for I sub-metric.
- **Real long-video HR baseline** (if obtainable from collaborators): Evaluate on at least one real long video with reference data (e.g., a 2–3 minute real-world HR video subsampled + interpolated to create synthetic LQ-HR pairs).
- **Cross-method evaluation** (optional): Extend to a third VSR method (e.g., a stable-diffusion baseline, RealESRGAN, or DOVE's own baseline) to confirm LR-VCC generalizes beyond MGLD vs UAV.

### July 2026 (final thesis writeup + blind review)
- Thesis chapters:
  - Literature review: VSR methods, temporal metrics, smoother-output bias in learned representations.
  - Methodology: LR-VCC architecture, sub-metric definitions, reliability-weighting derivation, hyperparameter justification.
  - Experiments: Layer 1+2 results (proposal), Layer 3 results (thesis).
  - Analysis: Failure mode characterization, cross-metric agreement findings, implications for metric design.
- Blind review deadline: **July 15, 2026**.

### August–September 2026 (revisions + final submission)
- Address blind-review feedback.
- Finalize Layer 3 results if June work was incomplete.
- Final thesis submission: **September 30, 2026**.

### Deliverables map
| Milestone | Output | Status |
|-----------|--------|--------|
| May 31 (proposal) | `docs/notes/2026-05-25-lr-vcc-validation.md` + scripts/lr_vcc/ + per-video results JSON | This proposal |
| July 1 (thesis draft) | Full thesis chapters + Layer 3 validation | Future work |
| July 15 (blind review) | Thesis PDF + appendices | Future work |
| Sept 30 (final) | Thesis + revisions | Future work |

---

## 7. Expected contributions

- **Metric-effectiveness characterization**: Systematic documentation of failure modes in existing long-video-SR metrics:
  - Smoother-output bias in three independent learned-representation families (DINOv2, anomaly ViT, LPIPS).
  - Content-dependent regime shifts (Anatomy on close-ups).
  - Time-scale incompleteness (adjacent-frame metrics miss long-range stability; crossover at k=5–10).

- **LR-VCC — a no-reference composite metric** for long-video SR:
  - Resolves smoother-output bias via reliability-weighting of existing sub-metrics.
  - Resolves regime-shift sensitivity via content-aware thresholds derived from documented failures.
  - Captures multi-scale temporal consistency via long-k weighted tOF.
  - Transparent, interpretable design (per-video sub-metric scores + weights reported).

- **Multi-scale temporal evaluation pipeline**:
  - tOF/tLP at multiple k values (1, 5, 10, 30, 60, 120).
  - Methodological contributions (FPS mismatch fix, mask coverage awareness).
  - Reproducible scripts + cached evaluation data for long-video benchmarking.

- **Open-source implementation**:
  - Modular LR-VCC codebase (scripts/lr_vcc/).
  - Evaluation scripts for metric comparison and per-video diagnostics.
  - Cached results on 5 synthetic long videos for reproducibility and future baseline comparison.

---

## Notes for full sections

**Section 3 references**:
- Three figures will be generated in Task 11:
  1. KZ8p6b1zJ9U regime-shift illustration (per-frame p_abnormal distributions, MGLD vs UAV; bbox-size chart).
  2. tOF/tLP crossover curves across k (mean ± std per method; showing crossover at k=5–10).
  3. Metric family agreement heatmap (5 videos × 8 metric families, color: MGLD-win or UAV-win).

**Section 4 references**:
- Architecture diagram (Task 12): Flow chart showing three sub-metrics → reliability computation → softmax composition → per-video LR_VCC → method aggregation.
- Hyperparameter table (above, from design spec): All defaults justified by preliminary findings.

**Section 5 references**:
- Per-video results table will be populated from Task 8 / Task 9 outputs (LR-VCC implementation + Layer 1–2 validation runs).
- Diagnostic JSON structure mirrors the example in design spec; per-method aggregate summary follows.

---

**Proposal status**: This outline is draft 1, awaiting user review and feedback before proceeding to full-prose sections (Tasks 11–13).
