# LR-VCC structural sub-metrics — design (Phase C.2)

**Status:** draft — awaiting review
**Date:** 2026-08-14
**Predecessor:** `2026-08-14-metric-v6-calibration-design.md` (Phase C.1, merged)
**Decision owners:** Timur

## Why this exists

Phase C.1 fitted the metric's response parameters and, as a by-product, attributed
every non-conforming validation cell to the stage where the signal was lost. Of 34
attributed findings, **20 are reachable by refitting constants and 14 are not**. The
14 need a different *measurement*, and that is what this design is about.

The targets are derived from those 14 findings, not from intuition:

| sub-metric | structural findings | stages |
|---|---:|---|
| identity | **6** | 3 reward-direction, 3 measurement |
| appearance | **5** | 5 measurement |
| clip trajectory (D″) | 2 | 1 reward-direction, 1 measurement |
| anchor histogram (D′) | 1 | 1 reward-direction |

By family: identity_degradation 4, identity_drift 4, background_drift 3, flicker 2,
flip_channel_shuffle 1.

## A correction to the earlier plan

Previous notes named "a mirror-sensitive sub-metric" as one of two structural fixes.
**The failure data does not support building it.** `flip_horizontal` is a SILENT
control family: FLAT is its *correct* outcome, it conforms on all five bases, and it
contributes zero structural findings. A mirror-sensitive sub-metric would fix nothing
in the matrix and would put a currently-passing control at risk. The horizontal-mirror
blind spot stays documented as a known metric limit — which is what a designed-in null
result is for.

The scene-cut-aware anchor survives, but it is a one-finding item, not a headline.

## Workstream 0 — expectation audit (do this first, costs nothing)

Some of the 14 are not metric failures at all; they are `DESIGNED_FOR` over-claims.
The map declares which sub-metrics each family should excite, and attribution reports a
`measurement` failure whenever a declared sub-metric's raw statistic stays flat. If a
sub-metric was never physically capable of responding to a family, that is a
bookkeeping error masquerading as a structural gap.

Concretely, appearance is declared for `background_drift`, `identity_degradation` and
`flip_channel_shuffle`. Appearance is per-frame perceptual quality: a background that
drifts, a face that degrades locally, or channels that permute all leave each frame
individually plausible, so quality genuinely should not move. Three of the five
appearance findings look like over-claims on that reading; the two `flicker` ones do
not, because flicker changes per-frame quality by construction.

**Deliverable:** re-derive `DESIGNED_FOR` from what each sub-metric physically measures,
with a written justification per entry, and re-run the attribution. The structural count
will fall. Whatever survives is the real work list, and the ceiling figure reported in
the C.1 deliverable should be restated against it.

**Rule:** the audit is done and frozen *before* any new measurement is built, so it
cannot be tuned afterwards to flatter a new sub-metric.

## Workstream 1 — anchored identity (I′). The dominant item

### Mechanism, with evidence

The identity sub-metric scores within-clip embedding self-similarity and pools it across
clips. It has no reference. That is the defect: a no-reference self-consistency measure
cannot distinguish *consistently the right person* from *consistently a blur*.

The battery shows this directly. On `identity_degradation` / `7WHI2L_FDNg`, per-clip
scores rise as severity rises:

| clip | severity 0.02 | severity 0.40 |
|---|---:|---:|
| 0 | 0.043 | 0.065 |
| 1 | 0.867 | 0.950 |
| 2 | 0.767 | 0.933 |
| 3 | 0.050 | 0.800 |
| 4 | 0.750 | 0.783 |

Fused: 0.375 → 0.489. Degradation washes out identifying detail, embeddings collapse
toward each other, self-similarity goes *up*. The composite then rewards the corruption,
and identity carries roughly five times the weight of the sub-metrics that correctly
detect it. Three of the five reward-direction findings in the whole matrix are this.

No choice of constants inverts a monotone-increasing response. This needs a reference.

### Proposed measurement

Anchor identity the way colour was already anchored. The precedent is in this repo and
it worked: the self-referential histogram sub-metric D was FLAT on colour drift on 0/5
bases; the anchored variant D′ reached 4/5.

- Establish a **reference identity set** from the video's own high-confidence early
  clips: take face embeddings from the opening window, keep those whose detection
  confidence and bbox area clear a floor, and cluster them.
- Score each later clip by **distance to the reference set**, not to itself.
- `I'_score` falls monotonically as later faces drift away from the anchored identity —
  and a blurred face is *far* from a sharp reference, so degradation now lowers the
  score instead of raising it.
- Keep the existing face-rate and close-up reliability gates; they are orthogonal and
  they work.

**Open decisions (flagged, not settled here):**
1. Embedding backbone — reuse the existing ArcFace path, or move to a backbone less
   biased against high-frequency detail. ArcFace's noise bias is documented in the
   project notes and is a real confound for diffusion-style super-resolution.
2. Anchor window length, and what to do when the opening window has no usable faces.
3. Multi-person content: per-identity clustering versus a single pooled reference. The
   parked multi-person identity plan is relevant prior art here.

### Validation criterion

I′ must (a) respond monotonically on `identity_degradation` and `identity_drift`,
(b) leave every SILENT control family FLAT, and (c) not degrade the four families it is
not designed for. Reported on held-out bases through the existing calibration harness.

## Workstream 2 — appearance scope

Five findings, all measurement, all "the raw statistic barely moved" (0.6%–3.9%).

Two candidate resolutions, and the audit in W0 decides between them per family:

- **Scope correction** — appearance is the wrong instrument for that family; remove the
  declaration and stop counting a non-response as a failure. Expected for
  `background_drift`, `identity_degradation`, `flip_channel_shuffle`.
- **Reference-anchored appearance (A′)** — for families where appearance *should* respond
  and does not, add a term comparing per-window quality against the opening window rather
  than summarising quality globally. `mean − λ·std` is invariant to a slow monotone drift
  that keeps the distribution's shape; a drift term would not be.

Do not build A′ unless a family survives the audit needing it. On current evidence only
the two `flicker` cells are candidates, and flicker is already caught elsewhere — so the
honest expected outcome of this workstream is "scope correction, no new code."

## Workstream 3 — scene-cut-aware anchor for D′ and D″

One structural finding (`background_drift` / `BrRLKMbBTYQ`, reward-direction), but it is
the visible edge of a documented content-domain limitation: the `BrRLK` base is
cartoon content with frequent hard cuts, and it carries a disproportionate share of the
matrix's inversions.

Both D′ and D″ anchor on a fixed opening window. When the video cuts hard and often, the
opening window does not represent the video, so natural anchor distance dominates any
injected drift.

**Proposal:** detect shot boundaries (a cheap histogram-difference or CLIP-embedding
discontinuity detector is sufficient — this does not need a learned shot detector), then
anchor **per shot** rather than per video, and aggregate drift within shots. A corruption
that drifts across the whole video still registers; a cut does not masquerade as drift.

**Risk to watch:** per-shot anchoring shortens every comparison window, which will reduce
sensitivity to genuinely long-range drift — the exact thing this benchmark exists to
measure. The validation must show long-range drift detection does not regress on
single-shot bases. If it does, the fallback is a scene-cut-aware *reliability gate*
(down-weight D′/D″ on cut-heavy content) rather than a changed measurement.

## Validation protocol

Non-negotiable, and inherited from Phase C.1:

- Every new or changed sub-metric is evaluated through the existing calibration harness,
  on **held-out bases**, against the frozen v5 reference.
- The sign-flip control families must stay silent. A new sub-metric that improves
  designed-for response while breaking a control has not improved the metric.
- The leaderboard guard applies: no structural change may invert the established method
  ranking.
- Report both the loss and the conformance count. Phase C.1's central lesson is that
  these move independently — a 35% loss reduction there produced an unchanged 39/55.
- The C.1 harness recomposes from cached statistics. Any new sub-metric must emit a
  per-clip statistic in the same shape so the response table can absorb it and the fit
  stays a laptop-scale operation.

## Cost and dependencies

Unlike Phase C.1, this work **requires the GPU server**: a new or changed measurement means
re-scanning all 300 battery clips plus the real-model outputs. Rough shape — I′ needs a
face-detection and embedding pass comparable to the existing identity stage; W3 needs a
cheap shot-boundary pass; W0 needs no compute at all.

This does **not** block on the video-scaling line, and that is the argument for doing it
now: the structural failures are mechanism failures visible at n=5, not sample-size
artefacts. When the enlarged base set lands, the recalibration and this work compose.

## Non-goals

- A mirror-sensitive sub-metric (see the correction above).
- Un-parking the identity dispersion gate — I′ supersedes the problem it was meant to
  patch.
- Any change to the composition step, the loss, or the fitted parameters. Those are
  Phase C.1's territory and stay frozen while measurements change underneath, so the two
  effects never confound.
- Human-anchored targets — still unavailable.

## Risks

- **I′ may trade one bias for another.** Anchoring makes the measure reference-relative,
  which imports the reference's own quality. A poor opening window becomes a poor anchor
  for the whole video. Mitigation: confidence-gated reference selection, and report the
  anchor's own quality as a diagnostic.
- **The audit could be self-serving.** Narrowing `DESIGNED_FOR` mechanically improves the
  conformance count without improving the metric. Mitigation: freeze the audit first,
  justify each entry in writing, and report the count under both the old and new maps.
- **Per-shot anchoring may cost long-range sensitivity** — see W3's fallback.
- **n=5 still binds.** Any new sub-metric is validated on the same five bases, so its
  evidence is as thin as v6's. This work should be judged on mechanism, not on a
  conformance delta that five videos cannot resolve.
