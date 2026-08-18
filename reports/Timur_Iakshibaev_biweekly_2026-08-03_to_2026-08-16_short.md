# Progress Report — Timur Iakshibaev

**Topic:** Video Super-Resolution for Long Videos — long-range consistency
metric: calibration and structural diagnosis

## Summary

**The synthetic corruption battery became a calibration signal instead of a
scorecard.** Twelve corruption families × five long videos × five severities had
only ever been used to grade the metric; this period the metric's response
parameters were fitted against it under leave-one-out cross-validation, so every
reported number comes from a video the fit never saw.

The verdict is deliberately two-sided. The fit objective improves **35%** (mean
held-out loss 0.0269 → 0.0174, better on **4 of 5 videos** compared like-for-like),
but the reader-facing conformance count is **unchanged at 39/55**. The candidate
metric is therefore **not adopted** — the frozen reference stands, and the
recommendation is to re-fit once the video set grows. What did improve is
qualitative: four cells that previously *rewarded* the corruption are now merely
blind to it, and every control family stays silent on held-out video.

**The more durable outcome is diagnostic.** Every failing cell was attributed to
the stage where the signal was lost: **20 findings are reachable by refitting
constants, 14 are not.** The largest group is the identity sub-metric, whose
mechanism is now pinned — it scores faces against *themselves* rather than
against a reference, so degradation that washes out identifying detail makes
embeddings collapse together and the score **rises** (0.375 → 0.489 as identity
degrades), while carrying roughly five times the weight of the sub-metrics that
detect the corruption correctly. No choice of constants inverts a rising
response; it needs a new measurement, and anchoring it repeats a move that took
the colour measure from 0/5 to 4/5 on this same benchmark.

Alongside this, the video-scaling phase was designed end to end after the lab
footage route closed — native-LQ only, single-shot only, degradation-stratified
quotas — and handed to a colleague for execution. Engineering: 8 test-first
tasks, 253 tests passing, frozen reference pinned bit-exact across all 315
clips, whole fit running in under a minute on a laptop.

Next: anchored identity and the other structural sub-metrics (needs GPU),
preceded by a zero-compute audit of which sub-metrics each family should
physically excite — frozen before any new measurement, so a narrowed
expectation map cannot be mistaken for a metric improvement.
