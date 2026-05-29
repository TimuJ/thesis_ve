# Section 2: Statement of the Problem

## 2.1 Problem Overview

The evaluation of long-form video super-resolution is structurally broken in four overlapping ways.

First, **per-frame fidelity metrics** (PSNR, SSIM, LPIPS, DISTS, CLIP-IQA) are temporal-blind by construction: a video that drifts slowly across thousands of frames in colour, that jumps at every tile boundary, or whose face identity degrades from minute one to minute two scores identically to a colour-stable, seam-free, identity-preserving video so long as each individual frame is locally crisp.

Second, **short-range temporal metrics** (E\*warp from TecoGAN, tOF and tLP at adjacent-frame lag k = 1) are scale-blind in the opposite direction: they measure adjacent-frame smoothness but cannot see consistency across the longer time horizons where the most damaging long-video failures actually occur. They also suffer from a systematic **smoother-output bias** — a spatially smoother SR output trivially has lower frame-to-frame difference regardless of whether it is more consistent — which surfaces empirically as the *tOF k-scale crossover* documented in §2.3.

Third, **perceptual / aesthetic aggregates** (DOVER, CLIP-IQA trajectories, VBench-2.0) compute clip-level scores that are then averaged up to a per-video number. The averaging removes the very consistency signal we want to recover. VBench-2.0's "long-video" extension is in practice clip-level reaggregation — the video is partitioned into 2-second clips, each clip evaluated by the same dimension-specific scorer, and per-clip scores averaged with no measurement of cross-clip drift.

Fourth, and most subtly, **even metrics that should catch a failure mode have content-dependent calibration shifts** that are not surfaced to the user. The same metric, computed under identical hyperparameters on two videos of the same overall quality, can flip its method ranking on one of them because some property of the content (close-up faces, dense motion, dim lighting) pushes a learned classifier into a different operating regime. The metric returns a number; it does not return "I am out of distribution on this video."

These four breakdowns are not independent — they reinforce each other. A researcher who wants to know whether method A is more temporally consistent than method B on long-form content has, today, no single trustworthy answer. The remainder of this section documents the breakdowns empirically, working from a case study (§2.2) through the temporal-scale axis (§2.3) into the controlled synthetic artefact battery (§2.4), and finally explaining why each gap is structural rather than a tuning fix (§2.5).

## 2.2 Case Study: A Regime-Shift in VBench-2.0 Metrics

The first empirical finding establishes that two independent VBench-2.0 metrics can fail together, in the same direction, on the same video.

**Setup.** Two representative super-resolution methods — one diffusion-based detail-preserving (sharper, occasional flicker) and one text-conditioned temporal-diffusion smoother (cleaner but with cumulative drift) — were applied to five long-form synthetic SR test videos (320×180 → 1280×720, 22,412 frames total across the five videos). Both methods produced visually distinguishable outputs; on four of the five videos, the detail-preserving method is also visually preferable (held-out informal viewing), and on one video the smoother method's output was visually rated as substantially worse — "really ugly" was the user's characterisation.

Two slow-fast-adapted VBench-2.0 metrics were computed per video: **Human_Anatomy** (anomaly classifier on per-frame body/face/hand ViT detector ensemble) and **Human_Identity** (RetinaFace + ArcFace face embedding stability across the video). Both were adapted to long videos via a 2-second sliding-clip protocol with slow (within-clip) and fast (cross-clip first-frames) components fused.

**Headline result.** On four of the five videos, both metrics rank the detail-preserving method higher, consistent with visual judgement. On the fifth video — a close-up scene in which the visually worse method's output was the "really ugly" one above — **both metrics flip rankings in disagreement with the visual evidence**:

| Metric on regime-shift video | Detail-preserving | Text-conditioned smoother |
|---|---:|---:|
| Human_Identity (fused, slow-fast) | 0.657 | **0.751** |
| Human_Anatomy (whole-video) | 0.144 | **0.435** |

Two independent metrics failing in the same direction on the same video — and on the *one* video where the human-visual judgement is most confident about which method is worse — cannot be dismissed as evaluation noise. There is a systematic property of the visually-better method's output on this particular video that triggers both metrics into a regime where they invert.

**Root cause.** Per-frame Anatomy fire-rate traces over the regime-shift video, broken down by detector channel (body, face, hand) and by anomaly threshold, revealed that the **fraction of frame area occupied by the largest detected hand bbox** is the regime-shift trigger. Across the five-video cohort, the median hand-bbox fraction is ≈ 2–3 % of the frame; on the regime-shift video it is **18 %**. The Anatomy anomaly classifier was trained on a video distribution where close-up hands are rare; on close-up content it fires more aggressively, and it fires *more aggressively on sharper output* because sharper edges produce stronger detector activations. The classifier therefore inverts on close-up content: the visually better method is flagged as "more anomalous" by exactly the trained behaviour the classifier is supposed to provide. The same close-up regime appears to affect Human_Identity through a parallel mechanism (face-detection-confidence cliffs at close range), though the smoking-gun trace there is less clean.

**[Figure 1: per-frame Anatomy fire-rate trace over the regime-shift video with hand-bbox-fraction overlay, showing the threshold crossing at which the classifier enters the high-fire regime.]**

**Generalised lesson.** VBench-style anomaly-detector metrics have **content-dependent calibration shifts** that the metric does not surface. A video whose content happens to push the classifier into the high-fire regime will see *every* method evaluated on it scored under different effective calibration than other videos. There is no per-video reliability signal in the standard pipeline that flags "this video is out of the metric's calibrated range." Documented in full diagnostic depth in `docs/notes/2026-05-13-kz-regime-shift-trigger.md`.

The proposed LR-VCC metric addresses this by introducing per-sub-metric *reliability gates* (Section 4) — for the Identity sub-metric specifically, a `closeup_p50` map built from the per-video Anatomy bbox traces downweights the sub-metric on videos whose close-up fraction exceeds a calibrated threshold.

## 2.3 The tOF k-Scale Crossover

The second empirical finding establishes that the choice of temporal scale at which to evaluate adjacent-frame consistency can flip which method appears better.

**Setup.** tOF (TecoGAN warping-error metric) is conventionally computed at k = 1 (adjacent frames; Chu et al., 2020). We extended the pipeline to compute tOF and tLP at k ∈ {1, 5, 10, 30, 60, 120}, with RAFT-based optical flow + forward-backward consistency masking applied uniformly across all k. Per-video metric values were aggregated over the five-video SR test set using 200 pairs per k per video.

**Finding.** The winner FLIPS based on k:

- At **k = 1**: the smoother method wins on 4 of 5 videos. The smoother method's adjacent-frame transitions are spatially less detailed, so adjacent-frame warping error is trivially lower regardless of cross-frame *consistency*.
- At **k ≥ 30**: the detail-preserving method wins on 4 of 5 videos. The smoother method's per-frame smoothness comes at the cost of cumulative drift across longer time horizons — drift that long-range warping picks up directly.
- The **crossover point is k = 5–10**: between these values neither method dominates. tLP shows the same crossover at the same range.

**[Figure 2: per-video tOF vs k curves for both methods, with crossover annotation at k ≈ 5–10.]**

**Interpretation.** Short-range temporal metrics suffer from the **smoother-output bias** in its most pure form: a smoother output has less to differ from the previous frame, so frame-to-frame difference is mechanically lower. This is the same bias that affects DOVER's aesthetic branch, DINOv2-cosine cross-frame similarity, and the Anatomy fire-rate. tOF at k = 1 has been the literature standard for "temporal consistency" measurement for five years; this standard measures, in practice, *spatial* smoothness.

**Practical consequence.** A research paper that claims a temporal-consistency improvement using tOF at k = 1 may be measuring spatial smoothness, not consistency. A method that wins on the standard benchmark by lowering tOF at k = 1 may lose to its competitor at k = 60. Neither metric is "correct" in isolation; the picture only resolves when k is varied. Documented in full in `docs/notes/2026-05-14-tof-tlp-long-range-results.md`.

**Implication for LR-VCC.** The proposed metric cannot pick a single k. It must aggregate across temporal scales (k ∈ {1, 5, 10, 30, 60, 120}) with weighting that does not drown either end. We expose `--temporal_weight {log, uniform, sqrt}` as a configurable choice; the production setting is `uniform`, which keeps every scale represented.

## 2.4 The Synthetic Artefact Battery

The third empirical finding is the killer demonstration: under a controlled synthetic artefact set where the severity ordering is known ground truth, *the entire set of widely-used baseline metrics is collectively blind to most of the failure modes that matter for long videos.*

**Methodology.** We built a parameterised synthetic-artefact pipeline (`scripts/synthetic_artefacts/`) that operates on existing SR outputs and injects controlled degradations at known severities:

- **`color_drift`** — linear ramp: at frame i out of T total frames, multiply the R channel by (1 + α·i/T) and reduce G/B by the same factor. Severity α ∈ {0.02, 0.05, 0.10, 0.20, 0.40} controls the final-frame colour shift. Tests detection of slow monotonic drift across the whole video.
- **`chunk_boundary`** — at every 60th frame (= 2 seconds at 30 fps), a deterministic per-chunk uniform additive offset ±α·255 to all channels. Severity α controls the step amplitude. Mimics the seams produced by tile-based diffusion VSR when the temporal tile size is 60 frames.
- **`flicker`** — sinusoidal brightness oscillation: per-frame brightness multiplied by `1 + α · sin(2π · i / period)` with period = 15 frames. Severity α controls peak amplitude. Mimics the periodic flicker produced by some diffusion samplers.
- **`identity_degradation`** — per-frame Haar-cascade face detection followed by `GaussianBlur(σ = α · 10)` applied within each detected face bbox with 10 % padding. Severity α controls blur sigma. Mimics the identity-collapse failure mode of diffusion VSR on faces.

Each artefact generator has a unit-test file verifying (a) severity-0 leaves the video unchanged within float tolerance, (b) per-frame statistics match the expected analytic curve, (c) seed reproducibility. The full suite is green. Test videos are generated from two long-form SR base videos (one 80-second, one 167-second at native fps) using the production-quality outputs of one of our detail-preserving SR baselines; the per-base-video count and per-severity grid are deliberately parameterised so the set can be extended (additional base videos, finer severity grid, new artefact families).

**Baseline metrics evaluated.** The same eight widely-used baseline metrics from the SR-evaluation literature were computed per video over the full artefact set:

| Metric | Family | Time scale |
|---|---|---|
| CLIP-IQA | Per-frame NR perceptual | Single frame |
| tOF (k = 1) | Adjacent-frame temporal | 1 frame |
| tOF (k = 60) | Long-range temporal | 60 frames |
| tOF (k = 120) | Long-range temporal | 120 frames |
| tLP (k = 120) | Long-range temporal-perceptual | 120 frames |
| E\*warp | Adjacent-frame temporal (TecoGAN) | 1 frame |
| DOVER | Clip-level NR aesthetic + technical | Per-clip aggregate |
| Identity (fused, slow-fast) | Cross-clip face embedding | 2-sec clip, slow + fast |

For each (metric, artefact) cell, the test is whether the metric responds monotonically across the five severities on both base videos. A cell is **PASS** if the metric's severity-response is monotonic (or near-monotonic with one tolerable mid-severity tie) on both base videos. A cell is **PARTIAL** if monotonic on one base but inverted on the other, or weak on both. A cell is **FAIL** if essentially flat across severities on both base videos.

**The verdict.** Of the 32 cells (8 metrics × 4 artefact families), only 5 are clean PASS:

| Metric | color_drift | chunk_boundary | flicker | identity_degradation |
|---|:---:|:---:|:---:|:---:|
| CLIP-IQA | FAIL | FAIL | FAIL | PARTIAL (drops with blur — global side-effect, not specific signal) |
| tOF k = 1 | FAIL | FAIL | **PASS** | FAIL |
| tOF k = 60 | FAIL | **PASS** | FAIL | FAIL |
| tOF k = 120 | FAIL | **PASS** | FAIL | FAIL |
| tLP k = 120 | FAIL | **PASS** | FAIL | FAIL |
| E\*warp | FAIL | **PASS** | FAIL | FAIL |
| DOVER | FAIL | FAIL | FAIL | FAIL |
| Identity (fused) | FAIL | FAIL | FAIL | PARTIAL (PASS on one base, inverted on the other) |

Of 32 cells: **5 PASS, 2 PARTIAL, 25 FAIL**.

**[Figure 3: 4-artefact severity-response grid (4 rows × 2 columns for the two base videos), each panel showing all 8 baseline metrics' normalised scores vs. severity. The dominant visual pattern is flat lines.]**

The four headline findings from this table:

**Finding 1 — Color drift is a categorical blind spot of the entire baseline literature.** Across both base videos and all five severities, no baseline metric responds monotonically. The mechanism is straightforward: a slow linear colour ramp produces a per-pair colour shift that, between frames k apart, is α · k / T. For our test conditions this per-pair shift sits below the histogram bin width of conventional colour-histogram metrics, below the noise floor of optical-flow warping error (the flow estimator absorbs uniform colour shifts), and below the perceptual threshold of CLIP-IQA and DOVER (the per-frame quality of a slightly red-shifted frame is indistinguishable from the original). The signal exists in the **trajectory of per-channel means across the entire video**, not in any per-pair comparison; this is the gap that the proposed LR-VCC sub-metric E (linear regression on per-frame Lab channel means) is designed to close.

**[Figure 4: zoom on color_drift specifically — all 8 baseline metrics shown flat across the severity range on both base videos, contrasted with LR-VCC sub-metric E's reliability ramp from R² ≈ 0 (no drift detected) to R² ≈ 1 (clear linear trend, full reliability) as severity increases. This is the cleanest single piece of evidence in the proposal that LR-VCC closes a real gap.]**

**Finding 2 — Chunk-boundary detection requires k ≥ chunk size.** All four long-range temporal metrics (tOF k=60, tOF k=120, tLP k=120, E\*warp computed on the same long-range pairs) catch chunk_boundary cleanly. Per-frame metrics (PSNR, SSIM, CLIP-IQA), the adjacent-frame temporal metric tOF k=1, and the clip-level perceptual metrics (DOVER, Identity) are all entirely blind. This proves that temporal metrics MUST be multi-scale: a single-scale temporal metric, no matter how well-tuned, will miss artefacts whose characteristic frequency does not match its measurement scale.

**Finding 3 — Flicker is only caught at k = 1.** All long-k temporal metrics (k ≥ 30) are blind to periodic flicker because the flicker period (15 frames) divides their measurement scale cleanly (60 = 4 periods, 120 = 8 periods), so the per-pair difference cancels by construction. tOF at k = 1 catches it beautifully — the response is monotonic with severity ratio ≈ 4.5× from the lowest to the highest severity. This is the complement of Finding 2: long-k alone is insufficient. A multi-scale aggregation must give meaningful weight to k = 1 as well as to long k. The proposed `--temporal_weight uniform` setting does so; the default `log(1+k)` weighting (used by some prior multi-scale aggregations) under-weights k = 1 enough to lose the flicker signal.

**Finding 4 — Identity_degradation triggers a content-dependent inversion.** On the multi-face base video, the Identity sub-metric correctly drops with blur severity (Δ −0.227 across the severity range): cross-clip embedding similarity decreases as blur erases identity-distinctive features. On the single-face base video, the same sub-metric *rises* with severity (Δ +0.114): heavy face-region blur (σ = 4.0 at maximum severity) erases the identity-distinctive features in *all* frames of that single face, so cross-clip embeddings become more similar to each other in a "generic blurred face" sense. The face detector survives the blur (face detection rate is ≈ 0.96 at every severity), so `face_rate`-based reliability gating does not engage. The pathology is in the slow-fast pooling itself, not in face detection. This is the cleanest characterised failure mode we have found in a vbench-style identity metric; it is documented in §5 as a known LR-VCC limitation, and a fix (gating sub-metric I by face-detection *confidence* rather than face-rate) is in the proposed thesis future work.

The full 8-metric × 4-artefact × 2-base × 5-severity matrix, including all numerical scores and the per-cell severity-response curves, is in `results/lr_vcc/severity_response_table.csv` and the per-condition JSONs under `results/synthetic_artefacts_eval/`. The structural conclusions are summarised in the verdict table above and elaborated in `reports/Timur_Iakshibaev_2026-05-22_to_2026-05-28.md`.

## 2.5 Why the Gaps Are Structural, Not Tuning

The blind spots above are not "use a different threshold" or "retrain on more data." Each has a structural cause:

- **Per-frame metrics**: no temporal awareness *by construction*. No amount of clever per-frame design recovers cross-frame consistency. The architecture forbids it.
- **Adjacent-frame temporal**: the smoother-output bias is *inherent* to small-k temporal differences — a spatially smoother output mechanically has lower frame-to-frame difference. The only fix is to evaluate at multiple temporal scales simultaneously.
- **Clip-level perceptual aggregates**: averaging removes the signal *by construction*. A consistency metric must be defined as *cross-clip* drift, not as the mean of per-clip qualities. VBench-2.0's clip-level reaggregation does the wrong arithmetic on principle.
- **Per-pair temporal at one fixed k**: scale-blind to artefacts whose characteristic frequency does not match k. Findings 2 and 3 are the same structural problem viewed from opposite ends of the spectrum.
- **VBench-style anomaly-detector metrics**: content-dependent calibration shifts (Finding §2.2) are intrinsic to learned classifiers operating on a wider content distribution than they were trained on. The fix is not better training data; the fix is *reliability gating* — flagging at evaluation time when a video is out of the metric's calibrated regime, and downweighting accordingly.

The implication for any proposed long-video consistency metric is that it must combine: (a) multi-scale temporal sampling that gives weight to both high-frequency and long-range failure modes, (b) an explicit colour-trajectory sub-metric that catches drift slower than any per-pair measurement scale, (c) reliability gating per sub-metric so that content-dependent metric flips suppress themselves on the videos where they apply, (d) a composition layer that downweights unreliable sub-metrics on-the-fly rather than reporting a single fixed metric. The proposed LR-VCC metric is specified to satisfy each of these requirements; its design is given in Section 4 and its validation in Section 5.

## 2.6 Summary of Problem Statement

The current long-video VSR evaluation toolbox cannot tell us when a method is consistent across the time horizons that matter for long-form content. We have shown empirically that:

- Two independent VBench-2.0 metrics can flip together on the same video in disagreement with visual judgement, traced to a content-dependent classifier regime shift (§2.2).
- The standard adjacent-frame temporal metric ranks the visually worse method higher because it measures spatial smoothness rather than consistency; this is recovered only by aggregating temporal metrics across multiple scales (§2.3).
- Under a controlled synthetic-artefact set, eight widely-used baseline metrics produce a verdict matrix of 5 PASS / 2 PARTIAL / 25 FAIL on 32 (metric, artefact) cells. Slow colour drift in particular is a categorical blind spot of the entire baseline set (§2.4).
- Each of these failures has a structural root cause; none of them is fixed by retuning a hyperparameter or retraining on more data (§2.5).

The proposal proceeds by stating the research aim and falsifiable hypotheses (Section 3), enumerating the research objectives and preliminary work (Section 4), specifying the proposed LR-VCC metric and its validation (Section 5), listing expected outcomes and innovations (Section 6), and presenting the timeline (Section 7).
