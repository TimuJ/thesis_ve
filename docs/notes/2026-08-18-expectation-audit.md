# Expectation audit — proposed revision of the designed-for map

**Status:** proposed, NOT applied. Must be reviewed and frozen before any new
sub-metric is built (Phase C.2, workstream 0).

## What this audits and why it comes first

Failure attribution works by taking, for each corruption family, the list of
sub-metrics that family was *declared* to excite (`DESIGNED_FOR`), and recording
a failure wherever a declared sub-metric fails to respond. That makes the
declaration load-bearing: **if a sub-metric was never physically capable of
responding to a family, a non-response is a bookkeeping artefact, not a metric
defect.**

It comes first, and gets frozen before any new measurement, for a specific
reason: narrowing the map mechanically improves the numbers without improving
the metric. Doing it after building a replacement sub-metric would let the
declaration be shaped — consciously or not — to flatter the new measurement.

## Proposed removals

Four declarations, each justified by what the sub-metric physically measures:

| family | sub-metric | why it cannot respond |
|---|---|---|
| `background_drift` | appearance | Appearance is per-frame perceptual quality. Background drift changes *what is in* the background over time; each frame remains an individually plausible image, so frame quality genuinely should not move. |
| `flip_channel_shuffle` | appearance | Permuting colour channels changes colours, not per-frame plausibility. A channel-swapped frame is still a well-formed image to a no-reference quality model. |
| `identity_degradation` | appearance | The degradation is face-local. Unless faces dominate the frame, global per-frame quality barely moves — and where it does move, it is a property of the content, not a designed capability. |
| `color_drift` | colour stability (D) | D measures *consecutive-frame* histogram stability. A slow progressive drift produces tiny frame-to-frame differences by construction, so D is blind to it. This is already visible in the D-variants comparison, where D is FLAT on colour drift on 0/5 bases while the anchored variant D′ reaches 4/5. |

Every other declaration survives. `chunk_boundary → (T, D)` and
`flicker → (T, A)` are both physically sound: a hard seam produces a flow
discontinuity and a histogram jump; brightness oscillation genuinely changes
per-frame quality. `flip_invert → (D, D′, D″, A)` is sound — inversion destroys
histograms, changes CLIP semantics, and degrades frame quality.

## Impact — measured, not asserted

| | findings | addressable | structural |
|---|---:|---:|---:|
| current declaration | 34 | 20 | 14 |
| after proposed audit | **27** | **16** | **11** |

Seven findings are removed, **all of them appearance**:

| stage | family | base |
|---|---|---|
| measurement | background_drift | 7WHI2L_FDNg |
| measurement | flip_channel_shuffle | hhszUXL1Cu8 |
| measurement | identity_degradation | BrRLKMbBTYQ |
| gate | background_drift | mJog8DlRk_4 |
| gate | identity_degradation | 7WHI2L_FDNg |
| gate | identity_degradation | mJog8DlRk_4 |
| composition | background_drift | BrRLKMbBTYQ |

Two things worth stating plainly, because they are what make this defensible
rather than self-serving:

- **The audit shrinks both classes, not just the inconvenient one.** Four of the
  seven removed findings are calibration-*addressable* (three gate, one
  composition). The structural share is 14/34 = 41% before and 11/27 = 41%
  after — essentially unchanged. The audit removes noise, it does not
  redistribute blame.
- **The `color_drift → D` removal changes no number at all.** It removes zero
  findings, because `color_drift` already conforms almost everywhere. It is
  proposed purely on mechanism. A map edit that costs nothing and gains nothing
  is evidence the edits are being made on physical reasoning rather than on
  score-keeping.

## The one judgement call

`identity_degradation → appearance` is the least clear-cut. Appearance *does*
move on two bases (raw quality shifts of 13% and 20% where faces are prominent)
and barely moves on a third (2.2%). So the sub-metric is not blind — it is
content-dependent.

The proposal removes it, on the grounds that a declaration should describe a
designed capability rather than an accident of framing, and that keeping it
records a failure on exactly the content where faces are small. The counter-
argument is that discarding a real if inconsistent signal loses information.

Flagged rather than settled: this one should be an explicit decision at review,
because it accounts for three of the seven removed findings.

## What this does not change

- No sub-metric, parameter, threshold or score changes. This is a change to what
  we *declare* a family should excite, and therefore to which non-responses get
  counted as failures.
- The conformance count (39/55) is untouched — that is scored per *cell* against
  each family's respond/silent expectation, which this audit does not alter. Only
  the per-sub-metric attribution changes.
- The reported ceiling changes from "14 of 34 findings need new measurements" to
  "11 of 27". Both should appear in any write-up, with the map version named, so
  the effect of the audit stays separable from the effect of any future metric
  change.

## Next

On approval: apply to `expectations.DESIGNED_FOR`, re-run the attribution,
regenerate the failure report, and record both counts. Then — and only then —
build the anchored identity sub-metric against the frozen map.
