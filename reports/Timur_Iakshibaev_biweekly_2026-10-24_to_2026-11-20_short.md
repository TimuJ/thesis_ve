# Progress Report — Timur Iakshibaev

**Topic:** measuring whether super-resolution keeps a subject consistent after it
disappears and reappears, and validating an MLLM judge for it against human perception.

## Summary

**The consistency benchmark now runs end-to-end on real footage and produces numbers.**
The minimum gap is now set by the enhancer's own memory: FlashVSR sees about three
seconds of context, so every gap must be longer than that. Consistency across the gap
then has to come from genuine identity memory rather than short-term carryover. Each
subject is compared before and after its disappearance on texture, geometry and
semantics, and the score is anchored on ground truth. That way the metric penalizes both
drift and fabricated over-consistency. On the current footage, FlashVSR loses fine
texture and semantic identity across the gap.

**A flaw in the combined score was found, verified and fixed, and the scorers were
upgraded.** Axes with no measurable signal on a pair, such as texture on a smooth object,
were voting at full weight and inflating the overall score. Each axis is now weighted by
its own signal, measured on ground truth, so a poor reconstruction cannot hide behind it.
The planned stronger models are not reachable from the lab server, so an agreed fallback
stack was used, and a curated set of 54 reviewed pairs was scored with it. One axis,
geometry, now looks saturated and is being checked before its number is trusted.

**A multimodal-LLM judge agrees with human perception where the feature metrics do
not.** A one-scene check showed the existing metrics can be fooled. DINO rewarded a
method built from the very features it compares, and SIFT rated the blurry input above
every reconstruction. An open, reproducible judge was built and correlated against a
blind human rater. It reached a rank correlation of about 0.7, against 0.35 at best for
the feature metrics. The best fully blind judge, scored in isolated calls, matches that
level. A stronger but non-blind control suggests judge capacity, not task design, is the
current ceiling.

**What the judge measures is the main open framing question.** As built, the judge
compares the reconstruction with the ground truth after the gap. That validates it as a
reconstruction-fidelity judge, not yet as a measure of drift across the gap, which would
require it to see both reconstructed frames. The results so far also come from a single
rater and a single super-resolution model, so additional raters and a second model are
the next steps.

## Next Period

1. Run a larger open judge with the same continuous readout, to get a reportable open
   number.
2. Verify the geometry axis on cross-gap pairs and gate it if it carries no signal.
3. Manually review the enlarged candidate pair set and rescore it.
4. Add two to three raters for inter-rater agreement.
5. Add the collaborator's model as a second comparison row.
6. Prototype a drift-aware judge and validate it against the drift labels.
