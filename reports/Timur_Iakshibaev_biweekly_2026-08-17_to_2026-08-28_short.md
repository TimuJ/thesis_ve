# Progress Report — Timur Iakshibaev

**Topic:** Video Super-Resolution for Long Videos — replacing the identity
sub-metric; calibration write-ups for the group

## Summary

**The identity defect is no longer inferred from symptoms — it is located in the
code.** The shipped identity tracker is re-created for every two-second clip, so
the reference face is re-initialised roughly eighty times per video and the score
is a *within-clip self-similarity*: the fraction of frames matching that clip's
own first face. Two consequences follow directly, and both match what the
validation battery had been showing. Long-range identity drift is structurally
invisible, because every clip re-anchors. And degradation *raises* the score,
because blurring collapses face embeddings toward each other so more frames clear
the similarity threshold — measured, the fused score goes 0.375 → 0.489 as
identity degrades, while the sub-metric carries roughly five times the weight of
those that detect the corruption correctly. No choice of constants inverts a
rising response, which is what makes this a measurement problem rather than a
tuning one.

The replacement follows a move that already worked on this benchmark: anchor it.
Score each clip against a reference identity built from the video's own
high-confidence opening clips rather than against its own first frame. The colour
sub-metric made exactly this transition — self-referential was blind to colour
drift on 0 of 5 videos, anchored reached 4 of 5. Rather than re-scan video once
per design variant, the pipeline persists raw face embeddings once so every
anchoring variant becomes cheap offline arithmetic. It is running on GPU now.

**That pipeline's correctness gate has already earned its place.** It replays the
*existing* scoring from the stored embeddings so the dump can be checked before
anything is built on it — and on the first completed video it failed, reproducing
0.6859 against a committed 0.6836. Clip and face-detection counts match exactly,
so splitting and detection are identical; the likely cause is that these battery
clips had been pruned under disk pressure and had to be regenerated, and video
re-encoding is not bit-identical. The consequence is concrete and cheap to
handle: recompute the baseline on the regenerated clips so both sides of every
comparison come from the same files. The point worth making is that the gate
caught it rather than waving it through.

Alongside this, two explanatory write-ups were produced for the group — one on
what the calibration actually fits (notably that the per-sub-metric weights are
*not* fitted; they are derived per video from reliability, and the weighting rule
was the largest thing the fit changed), and one on why changing already-published
coefficients was justified and why the result is not overfitting, with figures
showing that the published temporal coefficient compressed that sub-metric's
entire observed range into a 0.068 score span.

The expectation audit that gates the structural work is written and **proposed
rather than applied**: it would take the structural finding count from 14 to 11,
but narrowing the map improves numbers without improving the metric, so it must
be frozen before any replacement measurement exists. It shrinks both finding
classes roughly equally — the structural share is 41% before and after — which is
what distinguishes it from score-keeping.

Infrastructure: shared-disk pressure resolved, 9.1 GB → 42 GB free, by removing
51,053 intermediate frames from a closed study while keeping all 872 of its
result files.

## Next Period

1. **Finish the extraction and settle the gate** — the decisive test is running
   the same dump on a family whose original clips survived.
2. **Sign off the expectation audit**, including its one genuine judgement call.
3. **Build anchored identity** against the frozen map.
