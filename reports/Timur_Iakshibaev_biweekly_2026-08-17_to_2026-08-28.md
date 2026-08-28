# Biweekly Report — Timur Iakshibaev

## The period in three headlines

1. **The structural work moved from a design to a running experiment, and the
   identity defect is now pinned to a specific line of code rather than
   inferred from symptoms.** Reading the shipped implementation showed the
   identity tracker is **re-created for every two-second clip**, so the
   reference face is re-initialised roughly eighty times per video and the
   score is a *within-clip self-similarity*: the fraction of frames matching
   that clip's own first face. Two consequences follow directly, and both
   match what the validation battery showed. Long-range identity drift is
   structurally invisible — every clip re-anchors, so drift between the first
   and last clip is never measured. And degradation *raises* the score,
   because blurring collapses face embeddings toward each other so more frames
   clear the similarity threshold. No choice of constants inverts a rising
   response, which is what makes this a measurement problem rather than a
   tuning problem.
2. **The reproduction gate did its job — it failed, and the failure is
   informative.** The extraction pipeline replays the *existing* scoring from
   the stored embeddings so the dump can be checked before anything is built
   on it. On the first completed video it reproduced 0.6859 against a
   committed 0.6836. Clip count and face-detection count match exactly (83 and
   80), so clip splitting and face detection are identical; only the scores
   differ. The leading explanation is that the battery clips for these families
   had been pruned under disk pressure and had to be regenerated, and video
   re-encoding is not bit-identical. The consequence is concrete: comparisons
   must be recomputed on the regenerated clips rather than read against the
   committed numbers.
3. **Two explanatory deliverables were produced for the group, plus the
   expectation audit that gates the structural work.** The audit is the
   zero-compute first step of the structural phase and is deliberately
   **proposed rather than applied** — narrowing the map of which sub-metrics
   each corruption should excite mechanically improves the numbers without
   improving the metric, so it must be frozen before any replacement
   measurement exists.

## Key numbers

| result | value |
|---|---|
| Identity defect, measured | per-clip scores rise with severity (0.767 → 0.933 on one clip); fused **0.375 → 0.489** as identity degrades |
| Identity weighting | carries roughly **5×** the weight of the sub-metrics that correctly detect the corruption |
| Reproduction gate, first video | replayed **0.6859** vs committed **0.6836** (Δ 0.0023); clip count 83/83 and face count 80/80 identical |
| Expectation audit, proposed impact | findings **34 → 27**; addressable 20 → 16; **structural 14 → 11** |
| Audit, honesty check | structural share **41% before and after** — it shrinks both classes rather than redistributing blame |
| Audit, removals | all seven are appearance; the colour-stability removal changes no number at all |
| Response-curve finding | temporal sub-metric's score span over its entire observed range: **0.068** published → 0.181 fitted |
| Coefficients unmoved by the fit | **4 of 12**, including one full sub-metric and three gates |
| Disk reclaimed | **9.1 GB → 42 GB** free, by removing 51,053 intermediate frames from a closed study while keeping all 872 result files |
| Battery clips regenerated | **50** (two identity families × 5 bases × 5 severities), verified non-degenerate |
| Embedding extraction | running on GPU; 1 of 50 videos complete at time of writing |

## Decisions and framing this period

- **A previously planned structural fix was cancelled on the evidence.** A
  mirror-sensitive sub-metric had been named as needed work. The mirror family
  is a *control* — staying flat is its correct outcome, it conforms on every
  base, and it contributes zero structural findings. Building it would fix
  nothing and put a passing control at risk. It is now a stated non-goal.
- **The expectation audit comes first and gets frozen before any measurement
  changes.** Several recorded "measurement failures" are bookkeeping rather
  than defects: appearance is declared as a target for corruptions that leave
  every frame individually plausible, so per-frame quality genuinely should not
  move. Doing the audit after building a replacement would let the declaration
  be shaped to flatter it.
- **Embeddings are persisted once rather than re-scanned per design variant.**
  The same architecture that took the calibration harness to sub-millisecond
  recomposition: extract the expensive thing once, make every variant cheap
  arithmetic. It also makes the reproduction gate possible, which is what
  caught the regeneration discrepancy.
- **The fitted metric remains unadopted.** Unchanged from last period, and the
  explanatory write-ups state it plainly rather than leading with the 35% loss
  improvement.

## Outcomes this period

- [x] Identity defect traced to its mechanism in the shipped implementation,
      with the per-clip evidence that degradation raises the score.
- [x] Face-embedding extraction pipeline written, deployed and running, with a
      replay of the existing scoring built in as a correctness gate.
- [x] Reproduction gate exercised on the first completed video; discrepancy
      found, characterised, and traced to a probable cause.
- [x] Battery clips for both identity families regenerated after discovering
      the originals had been pruned; output verified non-degenerate.
- [x] Expectation audit computed and written up as a reviewable proposal, with
      its impact measured rather than asserted.
- [x] Calibration method note for the group: what has free parameters, what
      does not, and why the weighting rule was the largest thing the fit
      changed.
- [x] Coefficient-calibration explainer with two new figures — response curves
      showing what a coefficient does, and the in-sample/held-out evidence on
      why the result is not overfitting.
- [x] Shared-disk pressure resolved without losing any derived result.
- [ ] Embedding extraction across both identity families — running.
- [ ] Anchored identity itself — blocked on the above plus the audit sign-off.
- [ ] Video set scale-up — with a colleague.

## Next

1. **Complete the extraction and settle the gate.** The decisive test is to run
   the same dump on a family whose original clips survived. If that reproduces
   exactly, regeneration is confirmed as the cause and the fix is simply to
   recompute the identity baseline on the regenerated clips so both sides of
   every comparison come from the same video files.
2. **Sign off the expectation audit**, including the one genuine judgement call
   it flags — the appearance declaration on identity degradation, which
   accounts for three of the seven proposed removals and where the sub-metric
   is content-dependent rather than blind.
3. **Build anchored identity** against the frozen map: score each clip against
   a reference identity built from the video's own high-confidence opening
   clips instead of against its own first frame. The precedent is the colour
   sub-metric, where the self-referential version was blind to colour drift on
   0 of 5 bases and the anchored variant reached 4 of 5.

## One-line summary for the meeting

The identity sub-metric's defect is now located precisely — it scores faces
against themselves, re-anchoring every two seconds, so it cannot distinguish
consistently-the-right-person from consistently-a-blur and it rewards the
degradation it exists to detect — and the pipeline to replace it is running,
with its own correctness gate already having caught a real discrepancy in the
underlying video files rather than waving it through.
