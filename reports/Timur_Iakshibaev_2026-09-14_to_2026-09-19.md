# Weekly Progress Report — Timur Iakshibaev

## Headline

**The question raised the previous week — "how do we find the frames on which
consistency should actually be measured?" — was answered this period with
measurements, not argument.** It turns out to be one question with two halves,
and both are answered by the same source of truth: the low-quality input we are
super-resolving. The output must not be judged against its own past (a blur
maximises that); it must be judged against the structure the input establishes.
Two probe figures settle the two halves, a design specification turns the answer
into architecture, and a direct audit of VBench's consistency dimensions shows
what goes wrong when the frames are chosen the other way.

## 1. The question, stated precisely

"Which frames do we measure consistency on" is really two questions:

- **Applicability** — *on which frames is a consistency measurement even valid?*
  If the background legitimately changes (a camera pan), "background
  consistency" is meaningless there; measuring it manufactures a failure.
- **Correspondence** — *which frames should be compared with which?* If a person
  appears early and reappears much later, the two moments should be compared —
  but only once we know it is the same person.

Both need to know what is *supposed* to stay fixed before measuring whether it
did. The current metric never asks; it compares the output to itself. The answer
this period: the input already carries that information, and it is free, because
we only ever measure consistency on super-resolution output, and the input
always exists.

## 2. Answer, half one — applicability (which frames are valid)

Measure drift on the output *relative to the input* over the same frames, rather
than on the output alone. Where the input legitimately drifts, the difference
cancels, so the measurement switches itself off exactly where it should not
apply. A legitimate pan was synthesised (identical crop-translation on input and
output) and compared against a genuine colour corruption:

| base | pan, output-only | pan, input-differenced | real colour drift, input-differenced |
|---|---:|---:|---:|
| 7WHI2L_FDNg | 0.231 | **0.010** | 0.419 |
| mJog8DlRk_4 | 0.683 | **0.065** | 0.091 |
| hhszUXL1Cu8 | 0.126 | **0.020** | 0.037 |

The legitimate-pan false positive collapses 6–23× while a real corruption still
registers. Applicability stops being a hand-set gate and becomes automatic.

## 3. Answer, half two — correspondence (which frames pair with which)

Establish who-is-who once, across the whole video, from the input; then compare
the output only within those groups. Every face across a full ~2.8-minute video
was embedded and clustered from the input alone, at its native low resolution:

- **4050 faces over 72 of 83 two-second windows → 29 clean identity clusters**
  (within-cluster similarity 0.637 vs between-cluster 0.165).
- The decisive case: one identity is present early, **absent for about two
  minutes, and reappears near the end** — linked as one person from the input
  alone. This is exactly the "appears early, reappears much later" case raised the previous week, and it is exactly what a two-second-window measure
  cannot do: the two moments never share a window.
- Reproduced on a second base (correspondence spans 102 of 104 windows there),
  with the honest caveat that cluster quality is content-dependent — sparse-face
  content links identities but less cleanly, which is why an anchor-quality
  weight is on the design list.

## 4. Why the other way of choosing frames fails — two measured reductios

Choosing frames by the output's *self*-similarity is not merely weaker; it is
invertible.

- **Identity:** measured over two-second windows, the low-quality input itself
  scores **0.650** self-similarity — *higher than every super-resolved version*.
  A self-anchored identity measure therefore ranks the degraded input as the most
  consistent video in the set.
- **VBench's own consistency dimensions, audited directly this period.** Over a
  20× corruption-severity ladder on two drift families, both `temporal_flickering`
  and `motion_smoothness` are flat (mean change ≤ 0.0016 on a near-1.0 scale) and
  the residual trend points the wrong way — the score *fails to fall* on 18 of 20
  cases as the corruption worsens. On the same family where these dimensions do
  not move, our input-anchored colour sub-metric responds on 4 of 5 bases.

Both failures share one root: a metric that scores the output against itself
rewards the absence of detail, and measures on frames where nothing is pinned
down.

## 5. The answer, turned into architecture

The reframe is written up as a design specification (a distinct, reviewable
document). The frame-selection logic becomes three concrete components:

1. **Alignment** — pair input and output strictly by *frame index*, never by
   timestamp. The probe uncovered that the pipeline re-tags every output to a
   uniform frame rate while inputs keep their true rate; frame counts match
   exactly, timestamps do not, so timestamp pairing silently mis-selects frames
   on a third of the bases.
2. **Applicability** — the input-differencing of §2, reusing existing statistics.
3. **Correspondence** — the input-side whole-video clustering of §3, with an
   anchor-quality weight for sparse-face content.

Distilled to a rule that can be applied directly: *measure against the
input, not the output's past; measure only where the input says the quantity is
invariant; and compare only frames the input says correspond — aligned by frame
index.*

## 6. Asks / pending decisions

- **The one open decision is scope**: whether the benchmark formally adopts the
  input-referenced (reduced-reference) framing. It costs the "uses no side
  information at all" line, but for method evaluation it costs nothing — the input
  always exists — and it removes a failure mode that is otherwise *invertible*.
  This is the research owner's and collaborators' call; the specification is
  written to be executed the moment it is made.

## Next week

1. Extend the VBench-dimensions audit onto two further corruption families now
   being generated (face-local blur and periodic flicker), to test the
   blur-inversion case in addition to the drift-blindness case already shown.
2. On a scope go-ahead, turn the design specification into an implementation plan
   (alignment invariant first, then applicability, then the correspondence layer).
3. Fold both the identity and the temporal/motion reductios into the
   VBench-versus-ours comparison as measured evidence.
