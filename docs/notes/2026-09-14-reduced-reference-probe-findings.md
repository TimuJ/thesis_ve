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

### Second base: the correspondence claim holds, anchor quality does not

`mJog8DlRk_4` (104 clips, faces in only 33 of them) reproduces the
cross-time linking: 118 identity clusters, **max span 102 of 104 clips**, 21
clusters spanning more than half the video, 44 spanning 10+ clips. So
correspondence across the full video is established on both bases tested.

But separability is markedly worse (0.312 vs 0.472) from far fewer faces
(1708 vs 4050), scattered across 118 clusters. **Anchor quality is
content-dependent** — the same conclusion the anchored-identity work reached
independently, and the same reason an anchor-quality gate is already on the
design list. On sparse-face content the LQ still links identities, but less
cleanly.

### The hypothesis that did not survive, and what it revealed

The probe also tested whether identity degradation **collapses** whole-video
clustering. It does not: separability across the degradation ladder is flat
(0.434 -> 0.449 -> 0.461) and detections are unchanged (~4675). Reported as a
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

## Alignment: a concrete design constraint the probe uncovered

The second base initially produced a clip-count mismatch — LQ 104 clips against
the output's 83 — which would silently corrupt any reduced-reference
comparison. The cause is not content but **metadata**:

| base | LQ frames | LQ fps | output frames | output fps |
|---|---:|---:|---:|---:|
| 7WHI2L_FDNg | 5000 | 29.970 | 5000 | 30.000 |
| BrRLKMbBTYQ | 5000 | 24.000 | 5000 | 30.000 |
| KZ8p6b1zJ9U | 5000 | 29.970 | 5000 | 30.000 |
| hhszUXL1Cu8 | 2412 | 29.970 | 2412 | 30.000 |
| mJog8DlRk_4 | 5000 | 23.980 | 5000 | 30.000 |

**Frame counts match exactly on every base; fps tags differ on every base.**
The LQ carries the true frame rate, while the super-resolution pipeline re-tags
its output at a uniform 30.000 fps. At two-second clips this is harmless where
the true rate is ~29.97 (three bases round to the same 83 clips) and breaks on
the 24 fps bases (104 clips against 83) — so it fails on two of five, silently,
and only on some content.

The consequence is a firm rule for the design: **align on frame index, never on
timestamp or clip index.** Frame-index alignment is exact on every base here.
This also retro-explains the per-video fps override list the identity stage
already carries — that list exists for exactly this defect.

Because the mJog output-side clip grid does not match its LQ, the severity-ladder
comparison above remains single-base; the correspondence result, which needs only
the LQ's own clustering, stands on both.

## Honest limitations

- **The severity-ladder half of Figure B is one base** (7WHI2L_FDNg); the
  cross-time correspondence result covers two. The ladder cannot be extended to
  mJog until the clip grids are aligned by frame index (see above).
- **The pan is synthetic** — a translating crop, not real camera motion. It
  isolates the mechanism; it does not prove behaviour on real pans.
- **No ground truth on the true number of people.** The LQ yields 29 clusters
  where the output yields 91–109 at the same threshold: the LQ consolidates
  pose variants, the output fragments them. Which is closer to the truth is
  untested, and the clustering threshold is unfitted.
- **Reduced-reference is a framing change.** The metric stops being purely
  no-reference. For method evaluation this costs nothing — the LQ always
  exists — but it is a deliberate scope decision, not a free upgrade.
- Figure A relied on identical frame counts between LQ and output, which holds
  on every base; had it keyed on timestamps it would have been wrong on two.

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
