# Reduced-reference probe — findings

**Status:** spike complete for the two questions it was scoped to answer.
Throwaway probe code (`/tmp/probe_A_drift.py`, `/tmp/probe_B*.py`); nothing in
the shipped metric was touched. Recommendation at the end.

## What prompted this

Two questions from a colleague's supervisor, which turn out to be the same
question:

1. *How do we know a consistency measurement is even applicable?* If the
   background legitimately changes — a camera pan — what does "background
   consistency" mean?
2. *If a person appears early and reappears much later, how do we know it is
   the same person*, and therefore that the two should be compared at all?

Both ask: **a consistency metric must know what is supposed to be invariant
before it can measure whether the invariant held.** The shipped metric does not
ask this. It measures the output's similarity to itself, which a blur maximises.

The proposed reframe: measure consistency as **fidelity to the content
structure the low-quality input establishes**, not as the output's own
self-similarity. We are evaluating super-resolution, so the LQ input is always
available; it tells us what should stay fixed, what legitimately changes, and
which face is which.

## Figure A — applicability

`reports/figures/rr_probe_applicability.png`

Anchor drift (Lab-histogram distance, first quarter vs last quarter — the
statistic the `color_hist_anchor` sub-metric uses) computed self-anchored on the
output, and LQ-differenced (output drift minus LQ drift on the same frames). A
legitimate camera pan was synthesised by translating a crop window across the
frame, applied identically to LQ and output.

| base | pan, self-anchored | pan, LQ-differenced | colour drift, self | colour drift, LQ-diff |
|---|---:|---:|---:|---:|
| 7WHI2L_FDNg | 0.231 | **0.010** | 0.619 | 0.419 |
| mJog8DlRk_4 | 0.683 | **0.065** | 0.502 | 0.091 |
| hhszUXL1Cu8 | 0.126 | **0.020** | 0.134 | 0.037 |

**Result: confirmed.** The self-anchored measure reads a legitimate pan as a
large corruption signal. LQ-differencing collapses it by 6–23× while genuine
colour drift still registers. Applicability stops being a heuristic gate and
becomes automatic: where the LQ legitimately drifts, the difference cancels.

## Figure B — correspondence

`reports/figures/rr_probe_correspondence.png`

Every detected face across a whole video (83 clips, ≈2.8 min) was embedded and
clustered into identities (average linkage, cosine cut 0.55). Faces were
detected at the LQ's native 320×180 — **4050 faces over 72 of 83 clips**, so the
low resolution does not prevent correspondence.

**Result: confirmed, and stronger than expected.** The LQ alone recovers 29
identity clusters with clean separation (intra-cluster similarity 0.637 vs
inter-cluster 0.165; separability 0.472). Nine clusters span 10+ clips and six
span more than half the video. The decisive case:

- one identity is present in clips 5–9, **absent for about two minutes**, and
  reappears at clips 70–72 — linked as one person from the LQ alone;
- another appears at clip 1 and returns at clips 80–81, an ≈160 s gap.

This is precisely the "appears at frame 100, reappears at frame 1000" case, and
it is exactly what two-second chunking forbids: clips 1 and 80 are never in the
same window, so no within-clip measure can associate them.

### The hypothesis that did not survive, and what it revealed

The probe also tested whether identity degradation **collapses** whole-video
clustering. It does not: separability across the degradation ladder is flat
(0.434 → 0.449 → 0.461) and detections are unchanged (~4675). Reported as a
negative result.

Measuring the other half explains why, and turns the negative into the probe's
most useful finding. Within-clip self-similarity across the same ladder:

| severity | 0.02 | 0.05 | 0.10 | 0.20 | 0.40 |
|---|---:|---:|---:|---:|---:|
| within-clip self-similarity | 0.5780 | 0.5781 | 0.5778 | 0.5786 | **0.6168** |

**The collapse is local, not global.** Degradation makes faces blur into each
other *within a two-second clip* — exactly where the shipped metric measures —
while across the whole video distinct people remain distinct. The two findings
are complementary, and together they state the design conclusion precisely:
the correspondence layer is robust where the scoring layer is fragile.

The reductio is the sharpest single number here: **the LQ input itself scores
0.650 on within-clip self-similarity — higher than every super-resolved
version.** A self-anchored identity metric therefore ranks the degraded input as
the most identity-consistent video of the set. This is the identity analogue of
the VBench finding that its consistency dimensions rank the degraded input above
both super-resolutions, and it has the same root cause: self-similarity rewards
the absence of detail.

## Honest limitations

- **Figure B is one base** (7WHI2L_FDNg). The second LQ dump was still running
  at write-up; the analysis re-runs unchanged when it lands.
- **The pan is synthetic** — a translating crop, not real camera motion. It
  isolates the mechanism; it does not prove behaviour on real pans.
- **No ground truth on the true number of people.** The LQ yields 29 clusters
  where the output yields 91–109 at the same threshold: the LQ consolidates
  pose variants, the output fragments them. Which is closer to the truth is
  untested, and the clustering threshold is unfitted.
- **Reduced-reference is a framing change.** The metric stops being purely
  no-reference. For method evaluation this costs nothing — the LQ always
  exists — but it is a deliberate scope decision, not a free upgrade.
- Alignment between LQ and output (resolution, fps) must be exact or
  differencing manufactures drift; this probe relied on identical frame counts.

## Recommendation

**Go.** Both questions the probe was built to answer are answered with
measurements, and the mechanism is now specific enough to design against:

1. **Applicability** — anchor drift measures to the LQ rather than to the output
   itself. Cheap, reuses existing statistics, removes the false positive.
2. **Correspondence** — establish who-is-who by clustering faces across the
   whole video from the **LQ**, then score the output's consistency *against
   those groups*. Never score the output against itself: the probe shows
   self-similarity is maximised by the worst video in the set.

The next step is a design document for the reduced-reference reframe, not more
probing. The remaining open decision is the scope call — whether the benchmark
formally adopts reduced-reference framing — which is a decision for the
research owner and the collaborating colleague, not an implementation detail.
