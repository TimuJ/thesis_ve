# Proposal — strengthening cross-gap pair selection

**Status:** Proposal for discussion. The scoring/consistency scale is calibrated
(fail < 0.90, preserve ≥ 0.90, strong ≥ 0.96; see
`docs/notes/2026-11-05-adr-consistency-calibration-findings.md`). The weak link is
now **pair selection** — which (frame_i, frame_j) the tool picks for review.

## Context

The tool freezes a matched cross-gap pair on the ground truth and reuses it for
every SR model. The current selection cost rewards a large, geometrically similar,
semantically close object. Reviewer triage of all 42 selected pairs showed the
automatic gates (edge / weak-anchor / too-small) agree with human "bad" calls, but
a residual set of unreviewable pairs still gets selected.

## Three weaknesses (with evidence from the 42-pair review)

1. **Wrong objective.** Selection maximises *size × semantic-appearance match*, but
   the operative criterion is **reviewability** — can a human verify the cross-gap
   relationship. These diverge: large-but-unusable pairs are chosen and then cut.
2. **The "same configuration" test is pose-blind.** Pairs are matched with CLIP
   global cosine distance, which is semantic and pose-invariant by design, so
   different-pose pairs pass the gate (giraffe, lion, elephant were all selected and
   rejected as *"different angles, can't check consistency"*).
3. **No signal for the three real reject reasons.** Motion blur, low detail/contrast,
   and partial occlusion are unmeasured. We *proved* Laplacian sharpness cannot
   catch motion blur (the blurry crops had the **highest** variance — it tracks
   clutter, not blur), so monkey (blur), mouse ("small and dark"), and lizard
   (occluded) survive selection.

## Proposed direction: reviewability-driven selection

Replace the size-dominated cost with an explicit **reviewability score** — a
weighted combination of concrete signals — and select the argmax per gap.

| # | Improvement | Fixes | Method | Effort |
|---|---|---|---|---|
| P1 | Reviewability objective | weakness 1 | argmax of the scored signals below, not size | framing |
| P2 | **Geometric** pose-match | weakness 2 | RoMa v2 correspondence-inlier count between candidate GT frames (same pose → dense matches). Already our approved geometry scorer → unifies select + score. Cheap proxy: RANSAC homography inlier ratio | med |
| P3 | Motion signal | motion blur | optical-flow magnitude of the object across t−1…t+1 (fast box ⇒ blurred); directly targets what Laplacian missed | low–med |
| P4 | Detail/contrast signal | "small & dark" vs "small & clear" | crop RMS-contrast / entropy or a no-ref IQA (BRISQUE); keeps gecko & readable plate, cuts mouse | low |
| P5 | Stable-run selection | one-off bad frames | pick the sharpest/medoid frame of a window with stable area/aspect/appearance; optionally score over the run (median) | med |
| P6 | Make the selector **measurable** | hand-tuned thresholds | fit/validate the signal weights (or a small ranker) to the **42 reviewer labels we already have**; report selection precision instead of guessing thresholds | low–med |

## Recommended sequencing

1. **P3 + P4** (cheap, CPU) — fix the two biggest residual eye-cuts; ship this week.
2. **P6** — use existing labels to validate P3/P4 and set weights; turns the gates
   into a measured, reportable criterion.
3. **P2** — the principled pose fix, when the approved trio (RoMa v2 + DINOv3)
   weights are staged; reuses a model we are already committing to.
4. **P5** — robustness upgrade once the signals are in place.

## Open questions for the meeting

- Is using RoMa v2 for **selection** (not only scoring) acceptable on compute?
- Single scalar reviewability vs. hard gates + ranking — which is more defensible?
- How many reviewers should anchor the thresholds before they are fixed (the
  current 0.90 band and gate thresholds rest on one reviewer's judgement)?
