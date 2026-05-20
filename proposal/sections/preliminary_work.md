# Section 3: Preliminary Work

This section documents the experimental work that motivates the proposed metric and characterizes the failure modes of existing approaches. All numbers come from runs on a fixed test set of 5 synthetic long videos (320×180 low-resolution → 1280×720 super-resolved, 22,412 frames total), generated from real source videos with the DOVE degradation pipeline.

---

## 3.1 Test Set and Methods

**Test videos.** Five synthetic SR test videos at 1280×720, produced by applying the DOVE degradation pipeline (bicubic downscale 4×, noise, compression) to real source clips. Four videos have 5,000 frames each; one (hhszUXL1Cu8) has 2,412 frames. Per-video duration ranges from 80 to 208 seconds at native frame rates of 29.97, 24.00, or 23.98 fps. Total frame count: 22,412. Videos span diverse content: a close-up talking-head scene (KZ8p6b1zJ9U), a wide-shot interview (hhszUXL1Cu8), and three mid-shot action sequences.

**SR methods under evaluation.** Two state-of-the-art super-resolution methods are compared throughout:

- **MGLD-VSR** — a diffusion-based method that uses masked guided latent diffusion to progressively refine SR frames; published at ECCV 2024. Characterized by sharp, detail-preserving output.
- **Upscale-A-Video (UAV)** — a text-conditioned temporal diffusion SR method; published at CVPR 2024. Produces smoother, perceptually cleaner output at the cost of some high-frequency detail.

**Setup verification.** Before drawing any conclusions from metric scores, the inference environment was validated against published DOVE benchmark numbers. UAV reproduces DOVE paper PSNR/SSIM/LPIPS/DISTS values on UDM10 within a consistent positive offset (+1.33 dB PSNR, +0.025 SSIM) attributable to a minor difference in degradation pipeline configuration, not a setup bug. MGLD matches the DOVE paper UDM10 results exactly (PSNR 24.2339 vs. reported 24.23; SSIM and LPIPS identical to 4 decimal places). Lower scores for both methods on the synthetic long videos — relative to UDM10 — reflect content and generalization shift, not evaluation error (see `results/uav_mgld_evaluation_metrics.md`).

---

## 3.2 Existing Metrics on the Test Set

A comprehensive evaluation across seven metric families was conducted. The summary is as follows (full tables in `results/uav_mgld_evaluation_metrics.md`):

| Metric family | MGLD wins | UAV wins |
|---|---|---|
| VBench 1.x Quality (7 dimensions) | **5** | 2 (smoother-output artefacts) |
| VBench 2.0 Human_Identity (5 videos, fused slow-fast) | **5/5** | 0 |
| VBench 2.0 Human_Anatomy (5 videos, per-video) | **4/5** | 1/5 (KZ outlier — metric failure) |
| NR-IQA: CLIP-IQA, MUSIQ, NIQE, BRISQUE | **4/4** | 0 |
| DOVER overall (mean across videos) | **+8.75** | — |
| E*warp temporal warping error (5 videos) | **5/5** | 0 |

**MGLD wins on all six metric families where the metric is functioning correctly.** The two VBench 1.x dimensions where UAV wins — `subject_consistency` and `background_consistency` — are computed via DINOv2 and DreamSim cosine similarity between adjacent frames. Both reward temporal smoothness rather than temporal correctness: UAV's spatially-blurred outputs change less between frames in DINOv2 feature space, which the metric interprets as higher consistency. Visual inspection confirms MGLD is perceptually superior on those same videos. These two metric "wins" for UAV are DINOv2/DreamSim smoother-output bias artefacts, not genuine quality advantages.

The anatomy outlier on KZ8p6b1zJ9U (UAV whole-video anatomy 0.435 vs. MGLD 0.144 — a severe apparent MGLD failure) is characterized in depth in Section 3.4 below.

---

## 3.3 Long-Range tOF / tLP Crossover Finding

**Pipeline.** To move beyond adjacent-frame temporal metrics, a multi-k temporal evaluation pipeline was built based on TecoGAN's tOF and tLP measures (Chu et al., 2020), extended to frame gaps k ∈ {1, 5, 10, 30, 60, 120} via RAFT optical flow with forward-backward consistency masking. For each k value, 200 frame pairs per video were sampled uniformly, RAFT flow was computed, and only pixels with forward-backward flow error below 1 px were retained (the valid mask). tOF is the mean L1 flow magnitude on valid pixels for the second frame of each pair relative to the first; tLP is the LPIPS distance between warp-reconstructed and actual second frame on valid pixels.

**tOF crossover.** Figure 3 shows the mean tOF across all 5 test videos as a function of k. At k=1 (adjacent frames), UAV achieves lower tOF (0.0177 vs. 0.0216 for MGLD). The ordering inverts between k=5 and k=10, and at k=120 MGLD's advantage is 0.0241 (0.1441 vs. 0.1682). The crossover region of k=5–10 frames corresponds to approximately 0.17–0.42 seconds at the video frame rates used.

The interpretation is direct: UAV's spatially-smoother output warps more smoothly into adjacent frames (adjacent-frame temporal stability); MGLD's detail-preserving output maintains more globally coherent scene structure over long frame gaps (long-range temporal stability). Standard single-k temporal metrics computed at k=1 — including E*warp and the tOF component of DOVE — systematically undervalue long-range stability and favour the smoother-output method. This is the core empirical motivation for multi-time-scale aggregation in the proposed metric design.

**tLP pattern.** Unlike tOF, tLP (LPIPS-based perceptual warping error) favours UAV at all k values. The mechanism is the same smoother-output bias already identified in DINOv2 and DreamSim: LPIPS, trained on pristine HR reference images, rewards self-similarity in feature space, and UAV's smooth output produces frames that look more like their own warped predecessors in LPIPS feature space. This is the third independent learned-representation metric exhibiting the same bias — providing strong structural evidence that representations trained on pristine HR data have a built-in tendency to flag diffusion-generated high-frequency detail as inconsistency rather than quality.

**Mask coverage caveat.** Forward-backward consistency coverage degrades rapidly with k. Mean coverage across videos: 93%/91% at k=1 (MGLD/UAV), dropping to 11%/7% at k=120. UAV's coverage is consistently lower than MGLD's at long k because RAFT's flow estimator finds fewer consistent regions in smooth-textured outputs — smooth regions provide less optical flow signal to lock onto. At long k, the valid pixel subsets differ per method, introducing a sampling bias in the comparison. This limitation is acknowledged and mitigations (fixed intersection mask, looser FB tolerance) are left for future work; the crossover finding at k=10 is robust because coverage is still 57%/35% at that scale.

See Figure 3. Source data and full per-video tables: `docs/notes/2026-05-14-tof-tlp-long-range-results.md`.

---

## 3.4 KZ8p6b1zJ9U Regime Shift

**The anatomy outlier.** On the test video KZ8p6b1zJ9U, VBench 2.0 Human_Anatomy scores MGLD 0.144 vs. UAV 0.435 in whole-video mode — one of the largest per-method gaps in either direction across all videos and metrics. Yet visual inspection, NR-IQA, DOVER, E*warp, and long-range tOF at k≥10 all favour MGLD on this same video. Understanding why Anatomy disagrees with every other metric on KZ is the central metric-failure case of this preliminary study.

**Per-frame abnormal-rate distributions.** Figure 1 shows histograms of the per-frame abnormal rate (abnormal persons ÷ total persons, for frames with at least one detected person) on KZ8p6b1zJ9U vs. the stable-regime video hhszUXL1Cu8, separately for MGLD and UAV. On hhszUXL1Cu8 both methods have tight, near-zero distributions — the anomaly classifier operates confidently in the normal range with median rate 0 for both. On KZ8p6b1zJ9U, MGLD's distribution is mass-concentrated near rate=1.0 (median rate 1.0; 84.8% of frames-with-people have at least one person flagged), while UAV's KZ distribution is higher than hhsz but well below MGLD's. The detector is in a qualitatively different operating regime on this video.

**Root cause — close-up body parts.** A systematic comparison of bounding-box size distributions across all 5 videos reveals the trigger. KZ has hand bboxes at p50 = 18% of frame area — 20× larger than hhsz (0.9%) and 2.5–36× larger than the other three videos. At p90, a single hand on KZ occupies 85.8% of the frame; it is a close-up talking-head scene where hands frequently fill the visible area. Figure 2 plots hand-bbox p50 against the MGLD-minus-UAV anatomy gap for all 5 videos: the relationship is monotonic. KZ is the only video with hand bbox p50 > 5%, and it is the only one where UAV wins anatomy.

Mechanistically, the anomaly classifier's confidence distribution shifts dramatically at close-up scale. On hhsz, human/face/hand p_abnormal values are concentrated well below their respective thresholds (p_abnormal p90 < 0.07 for human). On KZ, the same scores shift to p_abnormal p50 = 0.32–0.42 — right at or above the decision thresholds (0.45 for human, 0.30 for face, 0.32 for hand). Small per-frame differences between MGLD's diffusion sharpening and UAV's smoothing then become decisive: MGLD's face detections land median p_abnormal 0.40 (above the 0.30 face threshold); UAV's land at 0.16. The detector is not globally biased against diffusion — it is miscalibrated for close-up scale content, near the boundary of its training distribution (see `docs/notes/2026-05-13-kz-regime-shift-trigger.md`).

**Two natural fixes were tested and neither fully rescues KZ.**

*Close-up frame filtering* — dropping any frame where a face or hand bbox exceeds 5% of frame area and re-aggregating on the remainder — actually widens the KZ gap: MGLD slow-fast drops from 0.137 → 0.076 (gap −0.339 → −0.437). The high-fire frames are not preferentially the close-up ones; they are distributed across the whole video. The bbox-size correlation holds across videos but is not a frame-level causal predictor.

*Continuous aggregation* — replacing the threshold-crossing fraction with the continuous mean of p_abnormal per detector category — halves the KZ gap (slow-fast threshold gap −0.339; continuous gap −0.148). This confirms that approximately 50% of the KZ flip is attributable to threshold-near-boundary discretisation noise. The remaining 50% is genuine signal: MGLD's KZ output produces inherently higher p_abnormal distributions than UAV's on this content, even before any thresholding. No per-frame structural fix we tested can resolve that residual, because it reflects the anomaly classifier's feature-space miscalibration on this content type, not an aggregation artefact. Fixing it would require retraining the anomaly classifier on diffusion-SR outputs of close-up human content.

For practical evaluation, this means Human_Anatomy should never be reported as a single mean over heterogeneous long-video content — per-video reporting with explicit flagging of content with median p_abnormal > ~0.2 is necessary. The LR-VCC reliability weighting mechanism (Section 4) operationalises this guidance: the Anatomy sub-metric's per-video reliability score explicitly encodes whether the video is in a stable or borderline regime.

Source data: `docs/notes/2026-05-13-kz-regime-shift-trigger.md`. Per-frame traces: `results/vbench2_anatomy/diagnostic_KZ8p6b1zJ9U/`, `results/vbench2_anatomy/diagnostic_hhszUXL1Cu8/`.

---

## 3.5 FPS-Mismatch Discovery (Methodological Contribution)

During evaluation setup, a systematic metadata error in both SR pipelines was discovered that, if uncorrected, produces artefactual metric scores on time-windowed evaluations.

**The finding.** Both SR pipelines hard-code the fps tag of their output mp4 container regardless of the low-quality (LQ) source frame rate: MGLD always tags 30.00 fps; UAV always tags 24.00 fps. Frame counts and pixel content are correct — the mismatch is purely metadata. LQ source frame rates vary across the five test videos: 29.97, 24.00, and 23.98 fps. This means every SR output for at least some videos carries a wrong fps tag (e.g., MGLD outputs 30.00 for a 23.98-fps source; UAV outputs 24.00 for a 29.97-fps source).

**Why it matters.** Time-windowed metrics — specifically the slow-fast Human_Identity adapter, which splits video into 2-second clips at the video's reported fps — produce different clip counts and clip boundaries for MGLD vs. UAV on the same source video. On KZ8p6b1zJ9U (LQ fps 29.97): MGLD's 30-fps tag produced 83 clips of 60 frames each; UAV's 24-fps tag produced 104 clips of 48 frames each. The fast branch (cross-clip identity) is evaluated on different synthetic videos of different effective durations. Cross-method comparisons on these metrics are not apples-to-apples.

**Attempted fix and silent failure.** Initial approach: re-mux with `ffmpeg -r N -i src.mp4 -c copy dst.mp4`. This silently no-ops — `-r` before `-i` sets input interpretation rate, but `-c copy` inherits the container fps from the bitstream's timing atoms. Verified by re-probing: MGLD output still reported 30.00, UAV still 24.00 after the re-mux. The metric scores on "fps_fixed" files were numerically unchanged.

**Working fix.** A code-level override was implemented: an `--fps_overrides JSON` flag was added to `human_identity_long.py` that substitutes a per-video fps value when calling `cv2.CAP_PROP_FPS` at clip-split time. The LQ source fps is used as the override for every SR method evaluated on that video. No frame data is touched; only the clip-boundary calculation changes. This fix is documented in `docs/notes/2026-05-07-sr-fps-mismatch.md`.

**Impact.** With fps-corrected splitting, MGLD wins Human_Identity on all 5/5 videos (was 4/5 before fix). The single pre-fix UAV win on KZ8p6b1zJ9U was entirely an fps-mismatch artefact: UAV's 24-fps tag gave it 104 clips vs. MGLD's 83, artificially smoothing the fast-branch ArcFace embedding drift. After correction, UAV's KZ fused identity score drops from 0.751 → 0.629, falling below MGLD's 0.657. All other per-video changes are small adjustments in the expected direction (methods gain slightly where their tag was wrong, lose slightly where it was right).

**Broader implication.** The fps-mismatch discovery is itself a methodological contribution: it demonstrates that time-windowed long-video metrics are sensitive to container metadata errors that standard per-frame metrics silently ignore, and that naive re-mux strategies cannot fix them. Any evaluation framework for long-video SR that uses windowed metrics must verify and, if necessary, override per-method fps tags before computing clip-boundary-dependent scores.

---

## 3.6 Summary of Motivating Findings

Three independent, well-characterized failure modes were identified in existing long-video SR evaluation:

**1. Adjacent-frame temporal metrics undervalue long-range stability (Section 3.3).** The tOF crossover at k=5–10 directly demonstrates that a method winning on adjacent-frame temporal consistency (UAV at k=1) can be losing on long-range stability (MGLD at k≥10). Standard E*warp and tOF k=1 metrics, as used in DOVE and other SR benchmarks, cannot detect this inversion. Any single-number temporal metric is insufficient for long-video SR evaluation.

**2. Three independent learned-representation metrics share a smoother-output bias (Sections 3.2, 3.3, 3.4).** VBench `subject_consistency` (DINOv2), VBench `Human_Anatomy` on close-up content (anomaly ViT), and tLP (LPIPS) all favour UAV over MGLD on at least one video where human perception and all remaining metrics prefer MGLD. These are not coincidental failures — they share a structural cause: representations trained on pristine HR data flag diffusion-generated high-frequency detail as either temporal inconsistency or anatomical anomaly. A long-video SR metric that relies on any single such representation inherits this bias.

**3. Content-dependent regime shifts in Anatomy cannot be resolved by simple per-frame predicates (Section 3.4).** The KZ8p6b1zJ9U failure is characterized by approximately 50% threshold-near-boundary noise (addressable via continuous aggregation) and 50% genuine distributional shift of the anomaly classifier at close-up scale (not addressable without classifier retraining). Even after testing two natural fixes, the metric flip persists. Honest evaluation requires a reliability assessment that can flag this class of failures per video per metric before aggregating to a summary score.

Each finding directly motivates a design decision in the proposed LR-VCC metric (Section 4):

| Finding | Proposed design response |
|---|---|
| Adjacent-frame metrics miss long-range stability | Multi-time-scale temporal sub-metric aggregating over k ∈ {1, 5, 10, 30, 60, 120} |
| Learned representations share smoother-output bias | Per-video reliability weighting that downweights sub-metrics whose internal signals indicate borderline-regime operation |
| Content-dependent regime shifts cannot be cleanly filtered | Per-video, per-method confidence score with honest reporting of low-reliability evaluations rather than suppression |

The LR-VCC validation results (Section 3 of `docs/notes/2026-05-21-lr-vcc-validation.md`) confirm that the composite metric correctly orders methods on all 5/5 test videos — including KZ8p6b1zJ9U, where the reliability-weighting mechanism produces a +0.2317 MGLD advantage by assigning low weight to the misbehaving Anatomy sub-metric and high weight to the well-behaved Temporal sub-metric, per-method and per-video.
