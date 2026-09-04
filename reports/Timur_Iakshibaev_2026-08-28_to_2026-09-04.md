# Weekly Progress Report — Timur Iakshibaev

*(covers two weeks — the previous weekly report was skipped)*

## Headline

**The anchored-identity replacement went from design to a tested prototype, and
its first honest result is a diagnosed failure, not a win.** Direction is
correct on both identity corruption families, but the control family responds
harder than the targets — and the probe data pins the mechanism precisely:
face *selection*, not face scoring. Alongside this, the reproduction gate
running at full scale surfaced a finding that reshapes how all identity
comparisons must be made from now on.

## 1. The environment-era finding (reproduction gate at full scale)

The embedding extraction completed for all 50 corrupted clips of the two
identity families, with a built-in gate that replays the *existing* metric
from the stored embeddings. Run at scale, the gate failed in an informative
pattern — and the decisive test on a family whose **original** clips survived
settled the cause:

| | easy-face bases | marginal bases (cartoon, close-up) |
|---|---|---|
| originals, replayed vs committed | Δ ≈ 0.015 | Δ up to **0.122**, detections 16 → 7 |
| clip counts | identical | identical |

Same pixels, same clip boundaries, fewer detections. The committed identity
baselines date from the *old, decommissioned server*; the current host runs a
hand-transplanted face-detection stack under a much newer torch than the
library pins. On detection-marginal content the two stacks flip enough
borderline detections to move scores by 0.05–0.12 **on identical video**.
Clip regeneration (the originals for the identity families had been pruned
and were rebuilt) is a secondary effect at most: all non-detection metrics
agree between original and regenerated clips to four decimal places.

Consequences, now recorded in the project docs:
- Identity comparisons must be **same-era**. The embedding-dump architecture
  turns out to be the fix, not just a convenience: both the legacy score and
  any anchored variant are computed from the *same* stored embeddings — same
  detector, same era, no cross-environment confound possible.
- The frozen published matrix stays valid as an internally consistent
  old-era reference.
- A recompute of the full metric stack for the two identity families is
  running into parallel `_regen` directories (originals untouched), so the
  working baseline and the video files on disk agree again. Six of seven
  stages are through for the first family; the flow-based temporal stage hit
  GPU contention and has an automatic retry armed.
- One reproducibility sentence for the thesis: face detection is
  environment-sensitive on marginal content; persisting detector output once
  removes that sensitivity from everything downstream.

## 2. Anchored identity: prototype, result, diagnosis, fix in flight

The prototype scores every clip against a reference identity built from the
video's own opening clips, entirely offline over the dumped embeddings —
twelve design variants evaluated in seconds, no GPU.

**What worked:** with an area floor and median aggregation, the response
direction is correct on *both* identity families (+0.024 and +0.032, against
the legacy measure's flat/inverted behaviour), with only noise-scale
monotonicity breaks. Raw magnitudes are small, but scaling raw statistics
into scores is exactly what the calibration harness is for. One aggregation
scheme (fraction-above-threshold) inverts and is ruled out.

**What failed, and why it is the valuable part:** the control — background
drift, where the person never changes — responds *stronger* than the identity
families (+0.17–0.21, monotone). The candidate counts explain it: on the
cartoon base, face-bearing frames grow from ~680 to ~3 760 with severity and
survive the area floor. The drifting background *introduces real, large,
detectable faces*, and the inherited largest-face-per-frame selection rule
starts scoring background faces instead of the subject. No offline variant
can fix that from a largest-face-only dump — it is a selection problem, not
a scoring problem.

**The fix is implemented and its data is being generated:** the dump now has
a candidate mode storing every detected face per frame (largest-face arrays
unchanged, so the legacy replay still works), and a candidate re-dump of the
control family is running. The next probe selects, per frame, the face
*closest to the reference* rather than the largest — the direct test of
whether reference-similarity selection rescues the control while preserving
the correct response on the identity families.

## 3. Explainer completed

The coefficient-calibration explainer gained its closing section: what the
calibration actually bought. The fit's most valuable output was not the
coefficients but the attribution — separating the 20 findings coefficients
can reach from the 14 they cannot — and the identity mechanism is now the
document's own cleanest illustration of that boundary: a compressed signal is
fixable by refitting; a signal pointing the wrong way is not fixable by any
constant.

## 4. Asks / pending decisions

- **Expectation-audit sign-off** (unchanged ask): the proposal predates all
  anchored-identity numbers in the commit history, so the pre-registration
  trail is intact — but applying it before the next probe keeps the ordering
  clean. The one judgement call is the appearance declaration on identity
  degradation.
- GPU contention on the shared box is the current pace-setter; everything is
  queued with automatic retries rather than blocked.

## Next week

1. Probe v3: reference-similarity face selection on the candidate dump —
   pass/fail on the control, response preserved on the identity families.
2. Finish the recompute chain + temporal-stage retry; verify the `_regen`
   baseline and recompose the two families' matrix rows on it.
3. If the control passes: candidate re-dump for the identity families, and
   wire the anchored statistic into the response table so the calibration
   harness can absorb it on held-out bases.
