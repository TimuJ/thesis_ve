# Evaluating an external video model on the LR-VCC benchmark — readiness note

Written for the collaborating PhD colleague who asked (a) whether the benchmark
is ready to evaluate his video model and (b) what evidence backs its results.

## Short answers

**Ready: yes, for ranking against the four existing methods on the five-video
long-video set.** Adding a method row is a proven, scripted path — the
frame-wise RealESRGAN row was added exactly this way. Turnaround after outputs
arrive is roughly one GPU-day of metric stages plus minutes of composition.

**Evidence: the metric's validity case is layered, and each layer is a
committed, reproducible artefact** (section 3). What the benchmark can honestly
claim about a new model is its rank and per-dimension profile against the
existing rows — with the caveats in section 4 stated rather than hidden.

## 1. What we need from you

Per the standing submission spec:

- SR outputs on **our LR inputs** (we provide the download link for the five
  sources), not on your own sources — outputs on your sources are acceptable
  only for a side-study, not the ranking table.
- Full duration, native fps preserved, resolution ≥ ours, CRF ≤ 18,
  filenames `<model>_<base_id>.mp4`.
- The five bases: hhszUXL1Cu8, 7WHI2L_FDNg, KZ8p6b1zJ9U, BrRLKMbBTYQ,
  mJog8DlRk_4 (all ≥ 1 minute, single-shot, genuinely degraded sources).

## 2. What you get back

- **One composite number per video and the method mean** — position in the
  leaderboard currently reading MGLD 0.622 > FlashVSR 0.610 > RealESRGAN
  0.604 > UAV 0.589 (MGLD winning 5/5 videos).
- **The seven sub-metric scores** (appearance, temporal, identity, three
  colour-stability measures, CLIP-trajectory drift) with per-video
  reliability weights — so the composite is inspectable, not a black box.
  The frame-wise RealESRGAN anchor illustrates the profile's value: it lands
  worst on exactly the sub-metrics a method with no temporal modelling
  should fail (identity, temporal flow, exposure slope).
- Optionally, your model's **severity-response fingerprint** on the synthetic
  corruption battery, if degraded variants of its outputs are of interest.

## 3. Why the numbers are defensible

1. **Perceptual ranking validity.** MGLD > UAV matches human judgement on
   5/5 videos; the composite orders all four methods consistently with
   visual inspection.
2. **A designed lower anchor behaves as designed.** RealESRGAN-per-frame
   (zero temporal modelling) is separated from all genuine video methods,
   and fails on the right dimensions.
3. **Stability.** The method ranking survives a 52-configuration
   hyperparameter sweep (order stable in 45/52, per-video MGLD>UAV in 50/52);
   a leave-one-out ablation shows every sub-metric family carries unique
   signal.
4. **Severity-response validation with pre-registered controls.** Twelve
   synthetic corruption families × five severities; the metric is
   as-designed on 39/55 constrained cells, where control families predicted
   invisible (e.g. horizontal mirror) are correctly flat — the battery is a
   falsifiable instrument check, not a self-graded scorecard.
5. **An adversarial baseline.** The closest published no-reference
   consistency measures (VBench subject/background consistency) rank the
   *degraded input above both of its super-resolutions*, invert the human
   MGLD/UAV ranking, and respond to 0/20 severity cells on the same battery.
6. **Calibration honesty.** Response-parameter fitting was done under
   leave-one-base-out cross-validation with the in-sample/held-out gap
   published; the fitted variant is deliberately NOT adopted at n=5. The
   shipped configuration is frozen and pinned bit-exact by regression tests.

## 4. Caveats we state up front

- **n = 5 videos.** Rankings come with per-video tables, not significance
  claims. The enlarged, degradation-stratified video set (in progress with
  the group) is what turns rankings into statistics.
- **No human-anchoring study yet** — perceptual validity rests on the layered
  evidence above, not on a rank-correlation-with-raters number.
- **Identity-era refresh (we run this, no action for you).** Face detection
  is environment-sensitive on marginal content; two of the four existing
  rows carry identity scores from the previous server. Before your row is
  compared, we recompute the identity stage for those two methods on the
  current stack so every row in the table is same-environment.
- The identity sub-metric's known weakness (self-similarity scoring) is
  documented, and its anchored replacement is prototyped with positive
  results; the shipped table still uses the frozen configuration for
  comparability.

## 5. Mechanics and cost

- Stages per method: CLIP-IQA, tOF, colour histogram/slope/anchored
  histogram, CLIP trajectory, slow-fast identity — one scripted runner,
  roughly a GPU-day on the shared box depending on contention.
- Composition and the report table are one local command from cached
  statistics.
- Your outputs never need to leave the group's server.
