# Research Summary — Timur Iakshibaev (April – August 2026)

**Direction:** Super-resolution of long videos (minute-scale and beyond) —
consistency evaluation and positional-encoding analysis.

## Outcomes at a glance

1. **LR-VCC** — a no-reference benchmark for long-range consistency in video
   super-resolution: a 7-component reliability-gated metric, a
   bit-reproducible synthetic corruption battery for validating it, and a
   4-method leaderboard on real minute-scale videos.
2. **Audit of the field's standard consistency measures** (VBench
   subject/background consistency): shown to rank degraded inputs above
   their super-resolutions and to detect 0 of 20 injected long-range
   corruptions — establishing the need for the new metric.
3. **A causal study of RoPE positional-encoding extrapolation** in a
   streaming video SR model (group deliverable): position offsets are
   quality-free 50× beyond the trained window, real temporal/spatial
   extension is cheap up to ~2×, and the model's long-video drift is
   provably *not* positional — redirecting the group's long-video effort
   toward streaming-cache mechanisms.
4. **Four SR systems reproduced/stood up end-to-end** (MGLD-VSR,
   Upscale-A-Video, FlashVSR, RealESRGAN; SeedVR2 environment ready), with
   reproducible recipes, 174 unit tests, and all results regenerable from
   scripts.

## 1. The problem

Video SR methods are validated on clips of ≤100 frames, but at minute scale
they develop failure modes invisible to standard metrics: identity drift on
faces, slow colour/exposure drift, and visible seams at processing-chunk
boundaries. Real long videos have no ground truth, so evaluation must be
no-reference — and the existing no-reference consistency measures turn out to
reward exactly the wrong thing (smoothness), scoring blur as stability.

## 2. The LR-VCC benchmark

**Metric.** Seven sub-metrics covering appearance quality, multi-scale
temporal flow consistency (frame gaps of 1 to 120), slow-fast face-identity
drift, two colour-stability measures, an anchored colour-drift measure, and a
CLIP-feature trajectory for semantic drift — combined by a reliability-gated
softmax composition, with per-video gates that switch off sub-metrics where
their inputs are unreliable (no faces, too few frames, low flow coverage).

**Validation battery.** 12 synthetic long-range artefact types (drift,
flicker, chunk seams, identity degradation, and 5 sign-flip controls) ×
5 base videos × 3 severities, generated bit-reproducibly. A severity-response
protocol classifies each (artefact, video) cell as PASS/WEAK/FLAT/INVERTED;
the metric is clean on 29/60 cells overall and on 7/10 cells of the
long-range drift families it was designed for. A 52-configuration
hyperparameter sweep shows the method ranking is stable (45/52, with all
exceptions at one extreme temperature), and a leave-one-out ablation shows
each sub-metric family carries unique signal — removing one flips exactly
the verdicts it was responsible for.

**Leaderboard.** On five real minute-scale videos (2,412–5,000 frames),
under a uniform audited protocol:
MGLD-VSR 0.622 > FlashVSR 0.610 > RealESRGAN 0.604 > Upscale-A-Video 0.589,
with MGLD-VSR first on all five videos. The metric also yields per-method
diagnoses: FlashVSR is best on identity but worst on long-range semantic
drift; the frame-wise RealESRGAN anchor is worst precisely on the temporal
and identity components a frame-wise method should fail.

## 3. Audit of existing SOTA consistency measures

VBench's subject- and background-consistency dimensions are the closest
published measures. Run on our real-video set, background consistency ranks
the *degraded input above both of its super-resolutions*, and both dimensions
order the smoother method above the sharper one — the inverse of human
judgement. Under our severity protocol on 50 regenerated corruption clips,
the dimensions respond to **0 of 20** conditions (LR-VCC: 7 of 10), with most
sub-threshold drift in the direction of *rewarding* the corruption. The
mechanism was identified in their long-video configuration (the cross-clip
term is zeroed, making the measures within-clip-only). The official VBench
release does not support long custom input; our adapted protocol is
documented for reproducibility.

## 4. RoPE extrapolation study (group deliverable)

A bit-exact position-injection instrument was built for FlashVSR's rotary
position encoding (verified zero-drift pass-through, 48 unit tests, no
modification of the model repository), enabling causal experiments that
content changes cannot confound. Main verdicts, measured against ground truth
on DOVE-UDM10 and YouHQ40:

- **Position offsets are free**: +0.001 dB even 50× beyond the trained
  window, on all three RoPE axes — translation invariance holds.
- **Geometry distortion is what bites**: distance dilation degrades
  monotonically; mild position-interpolation compression (s=0.75) is free,
  and the study contributes a per-axis empirical fingerprint of the
  interpolation-vs-extrapolation boundary in a video model.
- **Temporal extrapolation is bounded and graceful** (~−1.5 dB plateau at
  100× the trained window); spatial extrapolation is ~2.5× more sensitive.
- **Real extension is cheap**: the model's 4,089-frame single-pass ceiling
  is an implementation artefact — an extended position table sustained a
  5,009-frame single pass; real 2× spatial-extent extension is quality-free
  (an apparent collapse at the top rung was traced to a scoring-geometry
  artefact and corrected transparently).
- **The model's long-video semantic drift is not positional** (three causal
  arms indistinguishable), so long-video improvement effort belongs to the
  streaming cache/generation mechanism — the study's predictions and the
  instrument were handed to the window-extension line of work.

## 5. Systems and reproducibility

MGLD-VSR was reproduced to numerically identical published results;
Upscale-A-Video's +1.33 dB environment gap was documented and resolved by
pinning the original environment. FlashVSR and the SeedVR2 contrast model
were stood up from zero on the lab server with staged, documented setup
recipes (SeedVR2's remaining blocker — a 7.9 GiB resolution-bound rotary
tensor exceeding 40 GB cards — is analysed with a workaround path). The
evaluation stack runs 174 unit tests and every reported table regenerates
from committed scripts and cached measurements.

## Next steps

The benchmark work continues along three lines:

1. **Scale the video set (video sourcing under Teme):** take the long-video
   footage Teme will send, run it through the benchmark's curation checks
   (single-shot, minute-plus, genuinely degraded), and evaluate it with
   LR-VCC; regenerate the corruption battery on the accepted bases,
   extending the validation matrix from 60 toward 180+ cells and giving the
   reliability gates real statistics.
2. **Sensitivity calibration and metric v6:** promote the corruption battery
   from validator to calibration signal — fit each sub-metric's response
   parameters to a target severity-response curve, calibrating and
   validating on disjoint videos, with the sign-flip control families
   guarding against over-calibration; plus failure analysis of every
   non-clean matrix cell against the frozen v5 reference.
3. **Human evaluation study:** pairwise 2AFC comparisons on real SR outputs
   and graded corruptions, reporting LR-VCC-vs-human rank correlation with
   the VBench dimensions as the baseline on identical pairs.
