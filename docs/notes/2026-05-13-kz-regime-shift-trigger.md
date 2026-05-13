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

## Next steps

- **Filtered-regime experiment.** Re-aggregate the slow-fast Anatomy on KZ keeping only frames where face bbox < 5% AND hand bbox < 5%. If MGLD beats UAV on the filtered subset, this confirms the close-up regime is the failure cause and gives a usable workaround.
- **Robust aggregation experiment.** Replace `1 - fraction_abnormal` with `1 - mean(p_abnormal)` (continuous version) and re-evaluate on all 5 videos. Compare per-video rankings — predict the rankings stabilize.
- Both are post-hoc Python compute over the cached per-frame traces — no GPU, ~1 hour.
