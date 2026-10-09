# Progress Report — Timur Iakshibaev

**Topic:** Video Super-Resolution for Long Videos — long-range consistency
benchmark: a reduced-reference reframe, the VBench-dimension audit, and
evaluation-dataset selection.

## Summary

**Consistency is now measured against the input, not the output's own past.**
The probe set last period returned a clear "go". Measuring consistency as
fidelity to the low-quality input — rather than the output's self-similarity,
which a blur maximises — both removes the false positive on legitimately-changing
content (a synthesised camera pan collapses 6–23× under input-differencing while
a real corruption is retained) and recovers whole-video identity correspondence
(from the input alone, an identity is re-linked across a ~2-minute absence). This
answers both of the earlier questions — when a consistency measurement is
applicable, and how two faces far apart in time are known to be the same person —
with one mechanism, and it is now written as a reviewable design specification.

**The VBench consistency dimensions provably reward the degradation that
matters.** Widened to five corruption families, the audit shows VBench's temporal
and motion dimensions are blind to gradual drift and local degradation, respond
correctly only to fast global flicker (which super-resolution does not produce),
and — the key result — rate a globally blurred, detail-destroyed video as *more*
consistent, on every base tested (the score rises on 10 of 10 base×dimension
cells). Since detail loss is exactly what a weak super-resolution model produces,
these metrics would rank the worse method higher. This is the temporal/motion
analogue of the identity reductio, where the degraded input itself scores highest
on within-clip self-similarity.

**The direction has external validation and a concrete data plan.** A recent
long-video generation paper independently built the appear-disappear-reappear
benchmark we were designing, confirming it is a recognised question; the
differentiator is that generation has no reference and must self-compare, while
super-resolution has the input, so the same test runs reduced-reference with
ground truth. The evaluation-dataset search converged on long-term tracking
footage (disappear-reappear structure with identity ground truth) as the
correspondence backbone, with one real trade-off — high-resolution but
redistribution-restricted vs lower-resolution but freely redistributable.

## Next Period

1. **A first quantitative correspondence result on real footage:** select
   tracking sequences with a genuine disappear-reappear gap, degrade and
   super-resolve, and score the output against the identity ground truth.
2. **Settle the two open decisions:** reduced-reference adoption, and the dataset
   backbone.
3. **Add the collaborator's model row** when his outputs arrive.
