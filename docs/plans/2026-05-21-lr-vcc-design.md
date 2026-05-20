# LR-VCC — Long-Range Video Consistency Composite

**Date:** 2026-05-21
**Status:** Design — awaiting user review, then writing-plans for the implementation
**Goal:** A no-reference composite metric for ranking video super-resolution methods on long-video temporal/perceptual/identity consistency, robust to the smoother-output bias and content-dependent regime-shift failures we've documented in existing metrics.

## Motivation

Every learned-representation metric we've tested rewards smoother SR output in at least one regime where humans prefer detail-preserving (diffusion-style) SR:

- VBench-1 `subject_consistency` (DINOv2) — UAV beats MGLD on the metric despite UAV being visually worse. Color-invariant DINOv2 reads diffusion noise as inconsistency.
- VBench-2 `Human_Anatomy` on close-up content (KZ8p6b1zJ9U) — ViT anomaly classifier flags MGLD's output as more abnormal than UAV's at 2× the rate, even though MGLD looks better. Content-dependent regime shift; survives both close-up filtering and continuous aggregation.
- tLP (LPIPS-based) at every k — UAV scores lower on perceptual temporal-distance simply because UAV's frames look more like themselves under LPIPS feature distance.

But on a different metric family (NR-IQA, DOVER, E\*warp, slow-fast Identity, long-range tOF) MGLD wins on 5/5 videos. The structural problem with the failing metrics is the same: they're built on representations trained against "pristine real photo / HR" data and treat diffusion-detail as out-of-distribution.

There's also a cross-cutting time-scale finding: tOF crosses over between k=1 (adjacent, UAV wins by smoothness) and k≥10 (long-range, MGLD wins on 4/5 videos). Adjacent-frame temporal metrics systematically underweight long-range stability — and long-range stability is exactly what long-video SR is supposed to provide.

**LR-VCC** is built to resolve both problems together: it composes a handful of existing sub-metrics, weights them per-video using *reliability tests* derived from our characterized failure regimes, and aggregates across multiple time scales. The thesis claim is that single-aspect, single-time-scale metrics are inadequate for long-video SR; a content-aware composite isn't.

## Decisions (from the brainstorm)

1. **Use case** — ranking SR methods (single number per method).
2. **Reference** — no-reference (SR mp4 only). Matches our 5 synthetic long videos and matches the deployment use case.
3. **Focus** — composite of three aspects: appearance/content stability, temporal smoothness, identity preservation.
4. **Bias handling** — reliability test + downweight. Use existing learned representations honestly; lean on per-video reliability signals (derived from the regime-shift characterizations we already have) to know when to trust each sub-metric.
5. **Composition** — geometric mean (no compensation across sub-metric failures).

## Architecture overview

Per video V and method M, compute three sub-metric `(score, reliability)` pairs, then combine:

```
A_score, A_reliability = appearance_stability(M(V))
T_score, T_reliability = temporal_stability(M(V))
I_score, I_reliability = identity_preservation(M(V))

weights = softmax([A_reliability, T_reliability, I_reliability] / temperature)
LR_VCC(V, M) = exp( Σ_s weights[s] · log(s_score + ε) )

For method M:
  LR_VCC(M) = mean over V in test set (excluding low_confidence videos) of LR_VCC(V, M)
```

`temperature = 0.2` makes the softmax sharp so the most reliable sub-metric clearly dominates when one is much more trustworthy. Log-mean preserves the "no compensation for failures" property.

A video is marked `low_confidence` if all three reliabilities are below a floor (default 0.2) and excluded from the method mean by default; reported in the per-video table with a flag.

## Sub-metric A — Appearance stability

**What it measures:** does the SR output's perceptual quality stay stable across the video, and is the mean quality high?

**Build:**
- Compute per-frame CLIP-IQA on every frame (cheap, ~5 min/video on GPU).
- Optionally also MUSIQ for cross-validation; start with CLIP-IQA only.
- `A_score = clamp(mean(quality) − λ · std(quality), 0, 1)`
  - `mean` rewards high quality; `std` penalizes drift.
  - `λ` default 0.5; tunable.

**Reliability test for A:**
- Drop sub-metric weight if `std(quality)` is below a small floor (sub-metric not discriminating on this video).
- Drop weight if both methods saturate `mean(quality) > 0.98` (rare; ceiling regime).
- `A_reliability ∈ [0, 1]`, computed as `1 - sigmoid(distance from healthy regime)`.

**Why this avoids smoother-output bias:** CLIP-IQA is trained on diverse content (not just pristine HR), so the perceptual-quality signal is less biased than LPIPS / DINOv2. The drift-variance term is symmetric — a stably-blurry method has low std but low mean too, so `mean − λ·std` penalizes both extremes.

## Sub-metric T — Temporal stability

**What it measures:** how stable is the SR output over multiple time scales, with long-range stability weighted heavier than adjacent-frame stability?

**Build:**
- Compute tOF at k ∈ {1, 5, 10, 30, 60, 120} via existing `scripts/long_range_temporal/eval_tof_tlp.py`.
- `T_score = 1 − weighted_mean(tOF_k)` where weights are heavier on long k. Suggested: `weight(k) = log(1 + k)`.
- Optional: report tLP separately as a diagnostic column, but NOT in the composite (it has known smoother-output bias).

**Reliability test for T:**
- `mean_mask_coverage(k)` — if below a floor (default 0.10), that k's contribution is dropped from the weighted mean.
- If overall coverage across k is below a floor, `T_reliability` drops.

**Why this avoids smoother-output bias:** long-k weighting shifts away from the adjacent-frame regime where smoother outputs trivially win (the crossover we documented at k=5–10). Using tOF (pixel L2) not tLP (LPIPS) avoids LPIPS's self-similarity bias entirely.

## Sub-metric I — Identity preservation

**What it measures:** how stable are identifiable people / objects across the video?

**Build (v1):**
- Run the existing slow-fast `human_identity_long.py` adapter with fps-correction.
- `I_score = fused_slow_fast`.
- Future-proof: when multi-person Identity from `docs/plans/2026-05-06-multiperson-identity-metric.md` lands, swap in the cluster-purity + LQ-reference variants.

**Reliability test for I:**
- `n_clips_with_faces / n_clips`: fraction of clips where any face was detected. If below 0.20, downweight (identity isn't a meaningful signal on this video).
- `bbox_area_p50` close-up flag: if face or hand bbox p50 > 5% of frame area, partially downweight (Identity is known to be vulnerable to close-up cross-clip drift just like Anatomy).

**Why this avoids smoother-output bias:** ArcFace has a known bias against high-frequency noise (we saw it pre-fps-fix on KZ), but the slow-fast averaging within clips reduces it. The reliability test downweights videos where face detection is sparse or content is close-up.

## Reliability-weighting formula

Each sub-metric outputs `(s_score, s_reliability)` both in [0, 1]. Reliabilities are derived from regime indicators:

```
A_reliability = 1 − sigmoid_penalty(quality_saturation, drift_too_small)
T_reliability = 1 − sigmoid_penalty(mean_mask_coverage_low)
I_reliability = 1 − sigmoid_penalty(low_face_rate, closeup_bbox_ratio)
```

Each penalty in [0, 1] with smooth (sigmoid) transitions around documented thresholds (no sharp cliffs).

Composition in log-domain for stability:

```
weights = softmax([A_reliability, T_reliability, I_reliability] / temperature)  # temperature = 0.2
LR_VCC(V, M) = exp( Σ_s weights[s] · log(s_score + ε) )                          # ε = 1e-6
```

**Floor:** if all three reliabilities are below 0.2, the video is `low_confidence` — excluded from the method mean by default; reported in the per-video table with a flag. Optional `--include_low_confidence` flag to still aggregate them.

## Validation strategy

Three layers, in order of compelling-ness for the thesis. For the proposal milestone (May 31) we deliver Layer 1+2; Layer 3 is thesis future work.

**Layer 1 — perceptual agreement on the 5 synthetic videos.** We know MGLD > UAV by visual inspection on all 5. Pass criterion: `LR_VCC(MGLD)` > `LR_VCC(UAV)` on the aggregate mean, and MGLD wins per-video on 5/5 or 4/5. Failing means the metric is broken; passing is a sanity check, not proof.

**Layer 2 — flip-resistance on KZ8p6b1zJ9U.** The key thesis test. Anatomy flips on KZ for known structural reasons; tLP also flips on KZ. LR-VCC should NOT flip because reliability-weighting downweights Anatomy-style failures and the temporal sub-metric (long-k weighted tOF) agrees with perception there. Pass criterion: `LR_VCC(MGLD, KZ) > LR_VCC(UAV, KZ)`.

**Layer 3 — parameterized synthetic test datasets (thesis future work, not proposal).** From `docs/plans/2026-04-28-metrics-and-vbench-validation.md`: 5 datasets with parameterized severity (color drift, periodic flicker, chunk-boundary jumps, identity degradation, long-range BG change). LR-VCC should respond monotonically to severity, and the response curve should be smoother than any individual sub-metric. Pass criterion: LR-VCC beats every individual sub-metric on at least one of these axes.

## Per-video reporting

Output JSON per `(method, video)`:

```json
{
  "video": "KZ8p6b1zJ9U",
  "method": "MGLD",
  "lr_vcc": 0.612,
  "sub_metrics": {
    "appearance": {"score": 0.78, "reliability": 0.90},
    "temporal":   {"score": 0.70, "reliability": 0.50},
    "identity":   {"score": 0.66, "reliability": 0.85}
  },
  "weights_used": [0.40, 0.18, 0.42],
  "low_confidence": false,
  "diagnostics": {
    "closeup_indicator": 0.62,
    "mean_mask_coverage_long_k": 0.07,
    "face_detection_rate": 0.51
  }
}
```

Per-method aggregate table includes both the headline mean and the per-video × per-sub-metric matrix, so reviewers can see WHERE the composite is driven from. No black box.

## Hyperparameters and defaults

| Parameter | Default | Notes |
|-----------|--------:|-------|
| Sub-metric A: `λ` (drift weight) | 0.5 | Tune on Layer 1 |
| Sub-metric A: `drift_floor` for reliability | 0.02 | If `std(quality) < 0.02` → A undiscriminating, drop weight |
| Sub-metric A: `saturation_ceiling` | 0.98 | If `mean(quality) > 0.98` → both methods saturate, drop weight |
| Sub-metric T: `weight(k)` | `log(1+k)` | Other candidates: `k`, `√k` |
| Sub-metric T: `k_values` | `{1,5,10,30,60,120}` | From tOF/tLP script |
| Sub-metric T: mask-coverage floor per k | 0.10 | Drop k's contribution below this |
| Sub-metric I: face-rate floor | 0.20 | Downweight below |
| Sub-metric I: close-up bbox p50 threshold | 0.05 of frame area | Partial downweight above |
| Composition: softmax `temperature` | 0.2 | Sharper → reliable sub-metric dominates |
| Composition: log epsilon | 1e-6 | Numerical stability |
| Low-confidence floor | 0.2 (all reliabilities) | Below → exclude from method mean |
| Sigmoid penalty sharpness | 10 | Smooth transitions, not sharp cliffs |

All thresholds are tunable; defaults chosen from the per-video characterizations we already have.

## File layout

```
scripts/lr_vcc/
├── __init__.py
├── appearance.py          # sub-metric A — CLIP-IQA per frame + mean/std
├── temporal.py            # sub-metric T — long-k weighted tOF (reads our existing tOF JSONs)
├── identity.py            # sub-metric I — wraps human_identity_long.py output
├── reliability.py         # per-sub-metric reliability sigmoids
├── composite.py           # softmax weighting + log-mean composition
├── run_lr_vcc.py          # CLI: takes --videos_path --output_path --identity_results --tof_results
└── README.md              # usage, validation results
results/lr_vcc/
└── <method>/<video>.json  # per-video output
docs/notes/
└── 2026-05-25-lr-vcc-validation.md   # Layer 1+2 validation writeup (target date)
```

## Out of scope (proposal)

- Layer 3 validation (parameterized synthetic datasets) — thesis future work.
- Multi-person Identity v2 — keep using slow-fast v1.
- Cross-method validation beyond MGLD vs UAV.
- Learned-weights composition (gradient-based fit to human preference). Heuristic reliability weighting is enough for the proposal.
- A "metric-effectiveness classifier" (predicting when sub-metrics fail from content alone).

## Risks

- **CLIP-IQA std might not capture appearance drift the way we want.** If `std(CLIP-IQA)` is roughly constant across methods (both stable), sub-metric A becomes uninformative and `A_reliability` correctly drops it — but then the composite is just T + I, which might not differentiate the smoother-output failure mode well. Mitigation: in that case, supplement with a CLIP image embedding drift over time (cosine distance to first frame).
- **Reliability weights may over-correct.** If we systematically downweight Anatomy on close-ups, we're trading bias for variance — the composite might become noisy on close-up videos. Mitigation: keep a per-video diagnostic column showing pre- and post-weighting scores.
- **Hyperparameter overfitting.** With only 5 videos and 2 methods, we can over-tune defaults to fit Layer 1+2. Mitigation: pick all thresholds from independent characterizations (the regime-shift work), document them, and don't tune after seeing the Layer 1+2 result.

## Implementation pointers

- Reuse `scripts/long_range_temporal/eval_tof_tlp.py` for T (don't re-implement).
- Reuse `scripts/vbench2_long/human_identity_long.py` output for I.
- CLIP-IQA via `pyiqa` (in our `vsr` conda env on server).
- Sigmoid penalty: `sigmoid_penalty(x, x_threshold, sharpness=10)` = `sigmoid(sharpness * (x_threshold − x))` for "below-threshold-is-bad" cases.
- Per-video output JSON shape mirrors the example above; per-method aggregate is a separate JSON with `mean_lr_vcc`, the per-video list, and a `low_confidence_videos` array.
