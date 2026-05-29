# Section 6: Expected Outcomes and Innovations

## 6.1 Primary Deliverables

This research will deliver three artefacts.

**LR-VCC metric.** A no-reference long-range video consistency composite for super-resolution evaluation, with five reliability-gated sub-metrics (Appearance via CLIP-IQA; Temporal via multi-k tOF; Identity via slow-fast face embedding; Color Stability via Lab histogram L1; Color Slope via per-channel linear regression). The metric is reproducible from a single CLI invocation backed by per-video JSON sub-metric caches; downstream re-tuning of hyperparameters (`--temporal_weight`, `--color_hist_alpha`, `--color_slope_beta`) does not require re-scanning videos. The composition layer (softmax-log-mean of reliabilities at temperature 0.2) is sub-metric-agnostic so additional sub-metrics can be added without disturbing the existing five.

**Long-range consistency synthetic benchmark.** A parameterised synthetic-artefact test set covering four artefact families — slow colour drift, chunk-boundary jumps, periodic flicker, identity degradation — across multiple severity levels per base video. Each generator is implemented as a self-contained Python module under `scripts/synthetic_artefacts/`. The test set is the first long-video VSR consistency benchmark with controlled-severity ground truth; existing benchmarks (REDS, Vimeo-90K, Vid4, UDM10) evaluate per-frame fidelity on short clips. The test set is extensible: additional base videos, finer severity grids, and new artefact families can be added without disturbing the existing infrastructure.

**Reproducibility infrastructure.** The JSON sub-metric cache means a downstream researcher can re-derive composite scores under arbitrary hyperparameter choices in seconds, not hours. Sub-metric independence means new sub-metrics can be developed and tested against the same cache. The cache format and CLI surface are documented in `docs/onboarding.md` so a new collaborator can run the metric end-to-end from a clean checkout.

## 6.2 Scientific Innovations

This research introduces four ideas that we believe are novel in the context of long-video VSR evaluation.

**Reliability-weighted softmax-log-mean composition as a design pattern for no-reference long-video metrics.** Each sub-metric outputs both a score (in [0, 1], higher is better) and a reliability (in [0, 1], higher is more trustworthy). The composition is `weights = softmax(reliabilities / τ)` followed by `score = exp(Σ_s weights[s] · log(score_s + ε))`. Sub-metrics whose reliability gates fire (face_rate too low, close-up content out-of-distribution, histogram entropy too flat, R² of the slope fit too low, optical-flow mask coverage too low) are downweighted automatically *per video*, without any per-sub-metric thresholding decision baked in. To our knowledge this is the first long-video SR consistency metric of this form. The composition is sub-metric-agnostic — adding sub-metric F or removing sub-metric E does not require re-tuning the existing components.

**R²-gated reliability for trajectory-fit sub-metrics.** Sub-metric E closes the colour-drift gap by fitting a linear regression to per-frame Lab-channel means over the video, scoring `exp(−β · max|slope|)` and gating reliability by the maximum R² across the three channels. A drifting video has high R² (the slope is *real*), so the metric speaks confidently; a flicker video has R² near zero (the residuals dominate), so the metric correctly abstains; a clean video has near-zero slope, so the score is near 1 regardless. This pattern — gate reliability by *goodness of fit*, not by score magnitude — is reusable for any "slow signal vs. noisy baseline" detection problem in video evaluation (e.g., slow identity drift, slow geometric warp, slow codec quality degradation, slow luminance drift from sensor effects).

**Multi-scale temporal aggregation with configurable weighting and shared JSON cache.** Existing temporal metrics fix k = 1 (Chu et al., TecoGAN) or report a single k. The proposed sub-metric T aggregates over k ∈ {1, 5, 10, 30, 60, 120}; the per-(video, k) raw tOF and tLP values are cached in JSON; the aggregation weighting (`--temporal_weight {log, uniform, sqrt}`) is applied at composite-time without re-scanning the video. This enables principled hyperparameter sweeps and clean ablation of the temporal-scale axis (the same that produced the k-crossover finding in §2.3).

**Severity-response-driven metric design protocol.** The iterative history of LR-VCC (v1 → v2 → v2_uniform → v3 → v3+slope β=200) is itself a methodological contribution. Each version added exactly one element after a specific failure mode was *characterised* on the synthetic test set: v2 added a colour sub-metric after color_drift was shown to be invisible; v2_uniform changed the temporal weighting after flicker was shown to be down-weighted; v3 recalibrated the colour sub-metric's α after it was shown to dominate the composite by absolute score; v3+slope added the linear-trend sub-metric after the histogram-based colour metric was shown to miss slow drifts whose per-pair distance stayed below the bin width. This protocol — characterise failure mode → add one element → verify on the same test set without regression on the previous failure modes — is reusable beyond VSR for any composite no-reference metric where the failure modes can be parameterised.

## 6.3 Empirical Findings (Characterisations of Existing-Metric Failure Modes)

In addition to the proposed metric, the work produces four empirical characterisations of failure modes in existing widely-used metrics:

**Colour-drift blind spot of the baseline metric suite** (§2.4 Finding 1). No metric in the eight-baseline set responds monotonically to a slow linear colour ramp across the validation set. To our knowledge this is the first systematic documentation of this gap.

**tOF k-scale crossover** (§2.3). The winner ranking flips at temporal scale k = 5–10 because adjacent-frame temporal metrics measure spatial smoothness, not consistency. This is a counter-example to the literature convention of reporting tOF at k = 1 as the temporal-consistency proxy.

**Identity-collapse pathology of slow-fast pooling under heavy degradation** (§2.4 Finding 4). Cross-clip first-frame embedding similarity *rises* with face-region blur severity on single-face content because heavy blur erases identity-distinctive features and all faces look "generically similar." Documented mechanism, characterised behaviour, proposed fix (gate by face-detection confidence + per-face embedding variance) noted as future work.

**Content-dependent regime shifts in VBench-style anomaly-detector metrics** (§2.2). Two independent VBench-2.0 metrics flip ranking together on close-up content (hand-bbox fraction ≈ 18 % vs. cohort median ≈ 2–3 %). Reliability gating by close-up fraction reduces the impact in the proposed LR-VCC; the broader observation — that vbench-style metrics have unsurfaced content-dependent calibration shifts — is the more general finding.

## 6.4 Path to Thesis

The thesis extends the proposal with: the full literature review, extended chapters on the methodology and on each sub-metric's design history, future-work directions (multi-person Identity sub-metric, real-video baseline confirmation, fast-varying brightness sub-metric for the flicker gap, extending validation to a broader set of recent SR methods beyond the proposal's two baselines), and a discussion of how the reliability-weighted composition pattern generalises to other no-reference video quality problems.

A backup graduation path under the ZJU alternate policy (per supervisor instruction) is the **invention patent track**: an invention patent on the reliability-weighted composition mechanism enters substantive examination as a parallel risk-management track. This is contingency, not the primary plan.
