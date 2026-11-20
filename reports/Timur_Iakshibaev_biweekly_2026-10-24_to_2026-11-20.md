# Biweekly Report — Timur Iakshibaev

## The period in three headlines

1. **The cross-gap consistency benchmark is built end-to-end and producing numbers.** The
   gap threshold is now set by FlashVSR's measured context window, about 3 s, so the
   benchmark tests long-range memory rather than cache carryover. A GT-anchored
   matched-pair evaluation scores each subject before and after its disappearance on the
   three Ring-Forcing axes. FlashVSR drifts on texture and semantics.
2. **A flaw in the combined score was found, verified and fixed, and the scorers were
   upgraded.** Axes with no signal were inflating the composite. Each axis is now
   weighted by its own ground-truth signal. A curated 54-pair evaluation ran on the
   upgraded scorer stack.
3. **An MLLM consistency judge now agrees with human perception where feature metrics do
   not.** An open, reproducible judge tracks a human rater at rank correlation 0.69. The
   best fully blind judge reaches 0.70–0.71. The SIFT/DINO/geometry metrics reach only
   0.09–0.35. The judge measures reconstruction fidelity; it does not yet measure drift
   across the gap.

## Key numbers

- **Gap floor:** FlashVSR context window ≈ 85–89 frames ≈ 3 s at 30 fps. Raising the
  floor from 2 s to 3 s re-filters the pool from 88 to 46 videos; 27 videos have gaps of
  at least 5 s.
- **First three-axis run** (29 pairs; `s_M` is the GT-anchored score, 1 = SR changes
  exactly as much as GT): SIFT 0.834 / LoFTR 0.869 / DINO 0.923. Drift on every axis,
  texture worst.
- **Curated run** (upgraded stack, 54 reviewable pairs, 13 clips): composite mean 0.891;
  23 diverge / 31 preserve; SIFT 0.844 / DINOv2 0.876 / RoMa v1 0.971. The RoMa value is
  suspected saturated and is being checked.
- **Concept check** (one scene, collaborator's memory variants): DINO rewards a
  feature-retrieval variant at 0.829, above a human-preferred variant at 0.761. SIFT
  scores the blurry LQ input highest (0.368 vs ~0.07).
- **Judge vs human** (n=21, single rater; overall Spearman / Pearson):

| Judge | Blind | Overall SRCC | Overall PLCC [95% CI] |
| --- | --- | --- | --- |
| Qwen3-VL-8B (open, reproducible) | ✓ | 0.69 | 0.59 [0.34, 0.80] |
| Gemini-4-argon (closed) | ✓ | 0.70 | 0.71 [0.46, 0.87] |
| Qwen3-VL-235B (open, coarse readout) | ✓ | 0.56 | 0.51 [0.20, 0.73] |
| Feature-metric composite (reference) | — | 0.35 | 0.20 |

## Decisions and framing this period

- **The gap is a function of the enhancer.** With the gap longer than the model's
  context, only genuine identity memory can keep the subject consistent across it. A
  longer-context model would need a longer gap.
- **Differential degradation was chosen for the LQ side.** Before and after the gap, the
  input gets different degradation realizations. This stops a memoryless model from
  looking consistent just because the same input gives the same output.
- **The metric is anchored on ground truth.** It asks "does the SR change as much as the
  GT does", so it penalizes both drift and fabricated over-consistency. The raw
  Ring-Forcing self-similarity is reported alongside it for comparability.
- **The combined score was first reported per-axis, and is now fixed.** Each axis is
  weighted by its own reliability, measured on ground truth rather than the SR, so an
  over-smoothed SR cannot hide its own failure.
- **The scorer upgrade fell back.** The planned stronger models (DINOv3 / RoMa v2) are
  gated and unreachable from the lab server, so the pre-approved fallback (RoMa v1 +
  DINOv2) was adopted.
- **The judge's validated scope is fidelity, not drift.** The judge never sees the
  earlier SR frame. It matches human judgments of reconstruction fidelity, but not the
  earlier drift labels. Reporting it as a fidelity judge is valid now; measuring drift
  needs a re-framed judge. *Open decision.*
- **Pair selection now favours recall.** Selection over-filtered. It now hands back the
  top candidates for manual review instead of dropping them, which gave 93 candidates
  with at least 5 pairs for 16 of 18 videos.
- **Still open:** confirm the GT-anchored matched-pair score as the benchmark's headline
  metric, and decide the open vs closed judge for reporting.

## Outcomes this period

- **Evaluation code:** a matched-pair cross-gap consistency pipeline (selection, crops,
  three scorers, GT-anchored scoring, per-axis reliability, pose gate), with unit tests.
- **Dataset:** the 46-video pool and an 18-video one-per-class starter, ranked by scale
  variation, with differential-degradation infrastructure.
- **Example packs:** a 17-pair reproducible example pack and a 54-pair curated pack
  (auto + hand-curated) with per-axis scores and MLLM scores.
- **Concept-check report:** compares human preference, DINO/SIFT and the MLLM judge on
  the collaborator's memory variants.
- **Judge tooling:** a reproducible Qwen3-VL judge library and batch runner, a blind API
  judge runner, an offline blind rating page, and the human ratings.
- **Findings notes:** the calibration, the selection-signal probe, MLLM vs human,
  stronger-judge control, and the blind judge sweep.

## Next two weeks

1. Run the larger open judge (Qwen3-VL-32B) with the continuous readout, to get a
   reportable open number comparable to the 8B.
2. Check whether the geometry axis is saturated on cross-gap pairs, and gate it if it
   carries no signal.
3. Manually review the 93 recall-mode candidate pairs and rescore the kept set.
4. Recruit 2–3 more raters for inter-rater agreement and tighter confidence intervals.
5. Add the collaborator's model as a second SR row, so the benchmark ranks models rather
   than describing one.
6. Prototype a drift-aware judge that sees both SR frames, and re-validate it against
   the drift labels.

## One-line summary for the meeting

The long-gap consistency benchmark now runs end-to-end on real footage with a fixed
combined score. FlashVSR loses texture and semantic identity across gaps longer than
its memory. An MLLM judge that matches human perception far better than the
feature metrics is validated, with the next step being an open judge that also
measures drift.
