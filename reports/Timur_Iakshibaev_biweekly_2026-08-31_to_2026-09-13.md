# Biweekly Report — Timur Iakshibaev

## The period in three headlines

1. **The identity sub-metric's structural flaw was diagnosed at the code level
   and repaired in prototype.** The shipped identity measure re-anchors its
   reference face every two-second clip, so it scores within-clip
   self-similarity — it cannot tell *consistently the right person* from
   *consistently a blur*, and degradation therefore *raises* its score (fused
   0.375 → 0.489). An anchored replacement — scoring each clip against a
   video-level reference built from the video's own opening — flips the
   composite verdict from INVERTED to WEAK, the correct direction. The move
   mirrors the earlier colour-measure fix that went from 0/5 to 4/5 on colour
   drift. Full integration is deferred to the enlarged video set so the
   expensive face-embedding pass runs once, not twice.
2. **A reproduction gate and a pre-registered control each caught a real
   problem that a plain score table would have shipped silently.** The
   embedding pipeline replays the *existing* identity score from stored
   embeddings as a built-in check; run at scale it exposed that the committed
   identity baselines are not reproducible on the current server for
   detection-marginal content (up to 0.12 on identical pixels), traced to a
   detector-environment change from the server migration. Separately, the
   anchored prototype's control family appeared to fail — until inspection
   showed two corruption-generator reference backgrounds contain real faces,
   a battery-asset defect the control correctly surfaced. Both are now
   documented; the second becomes a curation rule for the enlarged set.
3. **The benchmark was made evaluation-ready for an external model, and the
   leaderboard was shown to be era-robust.** The identity stage for MGLD and
   UAV was recomputed on the current detector stack, closing a mixed-era
   comparability gap; method means moved by ≤ 0.0011 and every per-video
   verdict held. A readiness note and the six-layer validity case were written
   for the collaborating PhD colleague who wants to back his video-SR model
   with the benchmark.

## Key numbers

| result | value |
|---|---|
| Identity flaw, measured | fused score **0.375 → 0.489** as identity degrades (rewards the damage) |
| Flaw is era-robust | reproduced on the new environment at **+0.025** (old-server +0.029) |
| Anchored fix, composite verdict | **INVERTED → WEAK** on the degradation family; response-direction correct on 9/10 identity cells |
| Reproduction gate divergence | up to **0.12** on identical pixels (marginal content); ≤ 0.015 on easy content |
| Same-era leaderboard shift | method means moved **≤ 0.0011**; order and 5/5 per-video verdicts unchanged |
| Expectation audit applied | structural-failure ceiling **14/34 → 11/27**; structural share unchanged at 41%; conformance counts untouched |
| Test suite | **253 passing**; shipped configuration frozen, pinned bit-exact |
| VBench comparison (prior, restated) | VBench consistency dims **0/20** severity cells vs LR-VCC **7/10**; 5/5 vs 0/5 on colour drift |

## Decisions and framing this period

- **Reduced-reference reframe adopted as the next research direction.** A
  question from the collaborator's supervisor — *how do we even know a
  consistency measurement is applicable, e.g. when the background legitimately
  changes, or how do we know two faces far apart in time are the same person* —
  reframes the whole benchmark. The answer: consistency should be measured as
  *fidelity to the content structure the low-quality input establishes*, not as
  the output's own self-similarity (which a blur maximises). The LQ input tells
  us what is supposed to be invariant and what legitimately changes. This both
  answers the applicability/correspondence questions and explains mechanically
  why VBench's two-second chunking cannot: it never associates content across
  the video. A five-day brainstorm-plus-small-result probe on this is starting.
- **Expectation audit approved and applied**, frozen before any new measurement
  so a narrowed map cannot flatter one; the pre-registration trail predates
  every anchored-identity result in the commit history.
- **The fitted recalibration (v6) remains not adopted** — held-out conformance
  did not improve at n = 5. The frozen configuration stands.
- **A previously planned mirror-sensitive sub-metric was cancelled on
  evidence** — the mirror family is a control that must stay flat and
  contributes zero structural findings.

## Outcomes this period

- [x] Identity flaw traced to the shipped implementation with per-clip evidence
      that degradation raises the score; era-robustness confirmed.
- [x] Face-embedding extraction pipeline built and run for both identity
      families, with a reproduction gate that caught the detector-era issue.
- [x] Anchored-identity prototype: twelve design variants, adopted settings,
      composite-level swap experiment (INVERTED → WEAK), findings written up.
- [x] Expectation audit computed, approved, applied; attribution and reports
      regenerated; both map versions recorded.
- [x] Full `*_regen` same-era metric-stack recompute for both identity families
      (originals untouched); analysis inputs cached.
- [x] Same-era leaderboard: MGLD/UAV identity refreshed on the current stack;
      ranking shown era-robust.
- [x] External-model onboarding note + six-layer validity case for the
      collaborator; VBench-vs-LR-VCC comparison assembled.
- [ ] Reduced-reference (LQ-anchored) probe — starting, 5-day target.
- [ ] Collaborator's model row — awaiting his outputs on our LR inputs.
- [ ] Enlarged video set — with the senior colleague; blocks the statistics.

## Next two weeks

1. **Reduced-reference probe (5 days):** a brainstorm + small measured result
   showing that anchoring drift and identity to the LQ input (a) removes the
   false positive on legitimately-changing content and (b) fixes the identity
   correspondence problem across the whole video. Reuses the candidate
   face-embedding dump already built — mostly offline, one LQ dump pass.
2. **Fold the collaborator's model into the leaderboard** the moment outputs
   arrive; everything upstream, including the same-era refresh, is done.
3. **Widen the VBench dimension audit** beyond the two consistency dimensions,
   so the "structurally unfit for long-video SR" claim covers the full
   applicable set.

## One-line summary for the meeting

The identity sub-metric's flaw is now understood in code and fixed in
prototype, the benchmark is evaluation-ready with an era-robust leaderboard,
and a supervisor's question has reframed the benchmark's core contribution:
consistency measured as fidelity to the low-quality input's content structure —
which answers the applicability and correspondence questions and explains
mechanically why the field's chunk-based metric cannot.
