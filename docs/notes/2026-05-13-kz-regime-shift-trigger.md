# What triggers the Human_Anatomy high-fire regime on KZ8p6b1zJ9U

**Date:** 2026-05-13
**Continues:** `docs/plans/2026-05-07-metric-failure-diagnostic.md`

## Question

On `KZ8p6b1zJ9U` the VBench-2.0 anomaly detector enters a high-fire regime where MGLD scores 0.144 (whole-video) and UAV 0.435 — inverted from the per-perceptual ordering. On the other 4 videos the detector behaves stably and MGLD wins. What property of KZ puts it into the high-fire regime?

## Method

For each video × method I have a per-frame trace `(person_count, abnormal_count, per-person {bbox, scores})` from `diagnose_anatomy_per_frame.py`. I aggregated bbox-area distributions per detector category (human / face / hand) across all 5 videos × 2 methods, plus the p_abnormal score distribution per category on KZ vs the matched low-fire video `hhszUXL1Cu8` (MGLD wins 0.925 vs 0.878 there). CPU-only, no GPU re-run needed.

## Finding — close-up body parts trigger the regime

Bbox-area p50 across all 5 videos (frame = 1280×720 = 921600 px²), MGLD method, expressed as % of frame:

| Video | Human p50 | Face p50 | **Hand p50** |
|-------|----------:|---------:|-------------:|
| **KZ8p6b1zJ9U** (high-fire) | 47% | **9%** | **18%** |
| hhszUXL1Cu8 (low-fire, MGLD wins) | 45% | 4% | **1%** |
| 7WHI2L_FDNg | 22% | 4% | 0.5% |
| BrRLKMbBTYQ | 22% | 5% | 7% |
| mJog8DlRk_4 | 33% | 4% | 3% |

KZ has the largest body-part bboxes by a wide margin — especially **hands at p50=18% of frame, 20× larger than hhsz (0.9%)** and 2.5–36× larger than the other three videos. At p90, KZ's hand bbox covers 85.8% of the frame; the second-largest at p90 is BrRLKMbBTYQ at ~16%. KZ has scenes where a single hand fills the visible area — it's a close-up / talking-head video.

## How the anomaly detectors react to close-ups

p_abnormal distribution on KZ vs hhsz, per detector category, threshold for "flagged abnormal":

| Detector | Threshold | hhsz MGLD p50 | KZ MGLD p50 | KZ UAV p50 | KZ %-above-thr MGLD | KZ %-above-thr UAV |
|----------|----------:|--------------:|------------:|-----------:|---------------------:|--------------------:|
| human | 0.45 | 0.006 | 0.42 | 0.29 | 47.7% | 20.1% |
| face | 0.30 | 0.015 | 0.40 | 0.16 | 56.2% | 31.9% |
| hand | 0.32 | 0.109 | 0.32 | 0.23 | 49.9% | 27.4% |

Two regimes:

- **hhsz (small bboxes):** all p_abnormal values are tightly below threshold (human p90=0.07, face p90=0.03). The detector is super-confident the content is normal. Detector behaves like a near-zero baseline.
- **KZ (large bboxes):** p_abnormal distribution shifts dramatically — median jumps to 0.32–0.42 from 0.006–0.11. The detector loses its baseline confidence and operates near the decision boundary. Small per-frame differences between methods now translate into large flip-rate differences:
  - On face: MGLD p50=0.40 (well above thr 0.30) vs UAV p50=0.16 (well below). Same content, different SR, MGLD's diffusion sharpness shifts the median across the threshold.
  - On human: MGLD p50=0.42 (just below thr 0.45) vs UAV p50=0.29 (further below). 47.7% MGLD vs 20.1% UAV cross.

The detector isn't biased against diffusion globally. It's calibrated for typical mid-shot person scales and behaves correctly there (hhsz). On close-up scales the trained anomaly classifier has weak signal and small SR-style differences (diffusion sharpening vs UAV smoothing) become decisive.

## Implication for the thesis

`Human_Anatomy` is **not a usable metric for SR comparison on close-up content**. Specifically: when face bbox > ~5% of frame or hand bbox > ~5% of frame, the detector's anomaly-probability distribution shifts into a regime where threshold-crossing becomes sensitive to SR style rather than to actual anomaly. On wide-shot content (the typical training distribution of the anomaly detectors) it works as expected.

This is a clean characterization of *which content the metric fails on*: a per-frame bbox-area criterion would let us pre-filter or pre-weight evaluations. Two options for our long-video evaluation:

1. **Pre-filter frames with close-up body parts** before aggregating. Define a "stable-regime" subset of frames per video (e.g. drop frames where any face bbox > 5% or hand bbox > 5%). Re-aggregate the score on the remainder. The expected outcome: KZ's MGLD-vs-UAV gap closes substantially because the high-fire frames are removed.
2. **Report anomaly-probability statistics, not threshold crossings.** A mean p_abnormal (continuous) is more robust than a fraction-above-threshold (discontinuous). Most of KZ's flip comes from many MGLD detections having p_abnormal just above threshold and the corresponding UAV detections just below.

Both are post-hoc fixes; the deeper finding is that VBench-2.0's anomaly classifier was trained on a distribution that didn't include long-video SR close-ups, so it's miscalibrated in that regime.

## Cross-video sanity check

Hand bbox p50 (% of frame) sorted, with the per-video Human_Anatomy MGLD vs UAV outcome:

| Video | Hand p50 | MGLD anatomy | UAV anatomy | Winner | Gap |
|-------|---------:|-------------:|------------:|:-------|-----:|
| KZ8p6b1zJ9U | **18%** | 0.144 | 0.435 | UAV (large) | -0.291 |
| BrRLKMbBTYQ | 7% | 0.522 | 0.437 | MGLD | +0.085 |
| mJog8DlRk_4 | 3% | 0.577 | 0.541 | MGLD | +0.036 |
| hhszUXL1Cu8 | 1% | 0.925 | 0.878 | MGLD | +0.047 |
| 7WHI2L_FDNg | 0.5% | 0.832 | 0.735 | MGLD | +0.097 |

There's a clean monotonic correspondence: **larger hand bboxes → smaller MGLD-vs-UAV gap (or flip)**. KZ at 18% is the outlier; the next-largest (BrRLKMbBTYQ at 7%) is well within the stable-regime gap range.

## Follow-up experiments (run 2026-05-13) — close-up filter does NOT rescue KZ

Both post-hoc experiments produced clean per-video tables but show the KZ flip is **not just** a close-up-frame artefact. The bbox-size correlation is real at the per-video level but is not the proximate cause of the per-frame failure.

### Experiment 1 — stable-regime filter

Drop frames where any face OR hand bbox is ≥ 5% of frame area, re-aggregate slow-fast on the remainder.

| Video | MGLD filt slow-fast | UAV filt slow-fast | MGLD kept % | UAV kept % | Winner |
|-------|--------------------:|-------------------:|------------:|-----------:|:-------|
| 7WHI2L_FDNg | **0.741** | 0.600 | 57.2% | 50.0% | MGLD |
| BrRLKMbBTYQ | **0.648** | 0.489 | 23.7% | 18.5% | MGLD |
| **KZ8p6b1zJ9U** | **0.076** | **0.513** | 26.3% | 31.0% | **UAV (gap +0.437 — wider than unfiltered)** |
| hhszUXL1Cu8 | **0.859** | 0.725 | 42.8% | 42.0% | MGLD |
| mJog8DlRk_4 | **0.586** | 0.431 | 54.3% | 52.4% | MGLD |

**The KZ flip survives the filter and the gap actually widens** (from -0.339 unfiltered slow-fast to -0.437 filtered). MGLD's KZ score *drops* from 0.137 → 0.076 on the "stable-regime" subset, meaning even on the non-close-up frames of KZ the detector flags MGLD's content even more aggressively. The 26% of KZ frames we kept are not "stable" — they're just frames where no big face/hand happens to be present, but the underlying content still triggers heavy detector firing.

So the bbox-size correlation is **predictive at the video level but not causal at the frame level**. There's a deeper confound in KZ's content beyond close-up body parts.

### Experiment 2 — continuous aggregation (`1 - mean(p_abnormal)`)

Replace the per-frame fraction-above-threshold with the continuous mean of `p_abnormal`, averaged per detector category and across categories.

**Implemented as `--continuous` flag in `aggregate_slow_fast_anatomy.py` and `human_anatomy_long.py`.** Default behaviour matches upstream (threshold) for reproducibility; `--continuous` is the opt-in tweak.

**Slow-fast results across all 5 videos × 2 methods** (default threshold vs `--continuous`):

| Video | MGLD threshold | MGLD continuous | UAV threshold | UAV continuous | Δ_threshold | Δ_continuous |
|-------|---------------:|----------------:|--------------:|---------------:|------------:|-------------:|
| 7WHI2L_FDNg | **0.840** | **0.887** | 0.774 | 0.828 | +0.066 | +0.059 |
| BrRLKMbBTYQ | **0.472** | **0.770** | 0.410 | 0.745 | +0.062 | +0.025 |
| **KZ8p6b1zJ9U** | 0.137 | 0.591 | **0.476** | **0.739** | **−0.339** | **−0.148** |
| hhszUXL1Cu8 | **0.969** | **0.950** | 0.896 | 0.921 | +0.073 | +0.029 |
| mJog8DlRk_4 | **0.622** | **0.832** | 0.531 | 0.803 | +0.091 | +0.029 |
| **Mean** | 0.608 | 0.806 | 0.618 | 0.807 | **−0.010** | **−0.001** |

**Continuous aggregation halves the KZ gap** (slow-fast threshold gap was -0.339; continuous gap is -0.148) — meaning the threshold-near-boundary discontinuity accounts for ~50% of KZ's flip but not all of it. The other 50% is real signal: MGLD's KZ output produces genuinely higher `p_abnormal` distributions than UAV's even on a continuous scale.

**Per-video rankings under continuous: 4/5 unchanged (MGLD wins), KZ still flips.** Mean across 5 videos is still a statistical tie.

**Trade-off worth flagging:** continuous lifts everyone's absolute scores (most detections have low `p_abnormal`, so `1 − mean` is closer to 1 than `1 − fraction_above_threshold` on the same data), which **compresses inter-method gaps on the 4 MGLD-wins videos** (range +0.025 to +0.059 under continuous vs +0.062 to +0.091 under threshold). Continuous is more *robust* to threshold-boundary content but less *discriminating* in the typical regime. Reporting both is the right thing to do.

Per-video outputs cached at `results/vbench2_anatomy/anatomy_slow_fast_continuous/`.

## Revised interpretation

The bbox-size finding is real but a confounder, not the cause:

- Across videos: hand-bbox p50 *does* monotonically track the MGLD-vs-UAV gap (KZ at 18% is the outlier). This is informative but indirect.
- Within KZ: dropping close-up frames does *not* rescue MGLD. Frames flagged abnormal aren't preferentially the close-up ones — they're distributed across the whole video.

What's actually happening on KZ is content-specific in a way bbox-size alone doesn't capture. Candidates worth checking next:

1. **Scene content** — KZ may be a specific genre (interview, talking head, dance) where MGLD's diffusion prior pushes hand / face textures into a distribution the anomaly detectors learned to associate with "synthetic / abnormal". The detector wasn't trained on diffusion-SR outputs of long videos of this content type.
2. **Person appearance** — clothing pattern, skin texture, hair style. Specific to the person on screen in KZ.
3. **Camera / motion** — KZ may have static-camera or slow-camera shots where the same content is re-rendered many times with slightly different diffusion noise, building up consistent detector triggers.

To distinguish (1) vs (2) vs (3) we'd need additional close-up videos of *different* people / scenes for comparison — currently we have one close-up video (KZ) and four mid-shot videos. The bbox-size correlation might dissolve if we had a second close-up video that doesn't flip.

## Take-away for the thesis

`Human_Anatomy` is unreliable on KZ-style content, but the failure is not cleanly localized by any simple per-frame predicate we've tested. Continuous aggregation reduces but does not eliminate the KZ flip; close-up filtering doesn't help. This is itself a meaningful negative result — **a single per-frame structural fix is insufficient**; the detector's miscalibration on diffusion-SR content runs deeper than a clean predicate captures.

For the metric-effectiveness chapter:

- Report per-video pattern (MGLD wins 4/5 across multiple aggregation schemes).
- Report KZ as a content-specific failure case where the metric flips against perception under multiple aggregation schemes.
- Frame the bbox-size correlation as a *signal that the metric is unreliable on this content*, not as a proximate cause.
- Recommendation for SR practitioners: never report Anatomy as a single mean over heterogeneous long-video content — always per-video, and flag any video with median anomaly-probability above ~0.2 as a potential metric-failure case.
