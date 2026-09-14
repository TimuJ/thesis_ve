# Progress Report — Timur Iakshibaev

**Topic:** Video Super-Resolution for Long Videos — long-range consistency
benchmark: identity sub-metric repair, evaluation-readiness, and a
reduced-reference reframe

## Summary

**The benchmark's weakest sub-metric was diagnosed in code and fixed in
prototype.** The shipped identity measure re-anchors its reference face every
two seconds, so it scores within-clip self-similarity — it cannot distinguish
*consistently the right person* from *consistently a blur*, and degradation
therefore *raises* its score (fused 0.375 → 0.489, reproduced across server
environments). An anchored replacement — each clip scored against a video-level
reference built from the video's own opening — flips the composite verdict from
INVERTED to WEAK, the correct direction, mirroring the earlier colour-measure
fix that went from 0/5 to 4/5 on colour drift. Full integration is held for the
enlarged video set so the expensive face-embedding pass runs once.

**Two safeguards each caught a real problem a plain score table would have
shipped silently.** The embedding pipeline replays the *existing* identity score
from stored embeddings as a built-in check; at scale it revealed that the
committed identity baselines are not reproducible on the current server for
detection-marginal content (up to 0.12 on identical pixels), traced to a
detector change from the server migration. A pre-registered control caught the
anchored prototype apparently failing — until inspection showed two
corruption-generator reference backgrounds contain real faces; a battery-asset
defect, now a curation rule for the enlarged set. The habit that paid off: build
the check before the result.

**The benchmark is evaluation-ready for an external model, with an era-robust
leaderboard.** MGLD and UAV identity were recomputed on the current detector
stack to close a mixed-era comparability gap; method means moved ≤ 0.0011 and
every per-video verdict held, so every row a collaborator's model would be
compared against is now the same environment. A readiness note and the
six-layer validity case were written for the collaborating PhD colleague who
wants to back his video-SR model with the benchmark.

**A supervisor's question reframed the benchmark's core contribution.** Asked
how we know a consistency measurement is even applicable — when the background
legitimately changes, or how we know two faces far apart in time are the same
person — the answer is that consistency should be measured as *fidelity to the
content structure the low-quality input establishes*, not the output's own
self-similarity (which a blur maximises). The LQ input tells us what is supposed
to be invariant and what legitimately changes; it also establishes which face is
which across the whole video. This reduced-reference reframe answers both
questions and explains mechanically why VBench's two-second chunking cannot — it
never associates content across the video.

Supporting: expectation audit approved and applied (structural ceiling
14/34 → 11/27, conformance untouched); 253 tests passing with the shipped
configuration frozen bit-exact; the fitted recalibration (v6) remains not
adopted at n=5.

## Next Period

1. **Reduced-reference probe (5-day target):** brainstorm plus a small measured
   result showing LQ-anchoring removes the false positive on legitimately-
   changing content and fixes whole-video identity correspondence. Reuses the
   face-embedding dump already built.
2. **Add the collaborator's model row** the moment his outputs arrive.
3. **Widen the VBench dimension audit** beyond the two consistency dimensions.
