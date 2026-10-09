# Biweekly Report — Timur Iakshibaev

## The period in three headlines

1. **The reduced-reference reframe went from idea to measured result to
   architecture.** The probe set last period returned a clear "go": measuring
   consistency as fidelity to the *low-quality input* — not the output's own
   self-similarity, which a blur maximises — both removes the false positive on
   legitimately-changing content (a camera pan) and recovers whole-video identity
   correspondence (an identity re-linked across a ~2-minute absence). It is now
   written as a reviewable design specification.

2. **The VBench consistency-dimension audit, widened to five corruption
   families, closed with a measured reward inversion.** VBench's temporal and
   motion "consistency" dimensions rate a more-degraded (globally blurred) video
   as *more* consistent — on every base. Since global detail loss is the dominant
   failure mode of a weak super-resolution model, these metrics would rank the
   worse method higher. Measured on the battery, not argued from construct
   validity.

3. **The direction now has a direct external reference point and a concrete data
   plan.** A recent long-video generation paper independently built the
   appear-disappear-reappear benchmark we were designing — validating the
   direction and sharpening our differentiator (we have an input reference, it
   does not) — and the evaluation-dataset search converged on long-term tracking
   footage as the correspondence backbone, with one real trade-off to decide.

## Key numbers

- VBench audit, 5 families × 2 dimensions: blind to gradual colour/background
  drift and face-local blur (score moves ≤ 0.0016 across a 20× severity range);
  correct only on fast global flicker (score falls, −0.0257); **inverted on
  global blur — the score rises on 10 of 10 base×dimension cells** as detail is
  destroyed.
- Reduced-reference applicability: a legitimate camera pan reads as a large
  corruption when self-anchored but collapses **6–23×** under input-differencing,
  while a real corruption is retained.
- Correspondence: from the low-quality input alone, an identity is re-linked
  across a **~2-minute** absence (29 clean identity clusters on one base).
- The anchoring reductio: measured within two-second windows, the degraded input
  itself scores **0.650** identity self-similarity — higher than every
  super-resolved version.

## Decisions and framing this period

- **The core contribution is now stated as reduced-reference.** Consistency =
  fidelity to the structure the input establishes. This answers both earlier
  supervisor questions (when is a consistency measurement applicable; how do we
  know two faces far apart in time are the same person) with one mechanism, and
  explains why self-referential metrics — VBench's included — are not merely
  weaker but *invertible*.
- **An open scope decision is surfaced:** whether the benchmark formally adopts
  the reduced-reference framing. It costs the "uses no side information" line, but
  for method evaluation costs nothing (the input always exists) and removes an
  invertible failure mode. This gates turning the specification into
  implementation.
- **A dataset-backbone decision is teed up:** long-term tracking footage gives
  the disappear-reappear structure with identity ground truth; the fork is
  high-resolution-but-restricted (movie-sourced) vs lower-resolution-but-
  redistributable (Creative Commons).

## Outcomes this period

- Reduced-reference probe complete — two measured figures (applicability,
  correspondence); recommendation "go".
- Reduced-reference reframe written as a design specification (sub-metric
  partition; a frame-index alignment invariant; applicability and correspondence
  layers).
- VBench dimension audit completed across five families and both dimensions,
  with the reward-inversion result and a consolidated VBench-vs-ours comparison.
- A close read of the most relevant competing paper, extracting a concrete
  evaluation recipe (a graded gap axis, general feature matchers, single-shot
  input control, an LLM-judge anchor).
- Evaluation-dataset landscape surveyed and the construction problems mapped.

## Next two weeks

1. **A first quantitative correspondence result on real footage:** select long
   sequences with a genuine disappear-reappear gap from a tracking dataset,
   degrade and super-resolve them, and score whether the output preserves the
   subject across the gap against the identity ground truth — directly comparable
   to the external benchmark's table.
2. **Settle the two open decisions** (reduced-reference adoption; dataset
   backbone).
3. **Add the collaborator's model row** when his outputs arrive (still pending).

## One-line summary for the meeting

Consistency measured against the input, not the output's own past: the reframe
is specified, VBench's dimensions are now provably reward-inverting on the
degradation that matters, and the next step is a real disappear-reappear dataset
on long-term tracking footage.
