# VBench vs LR-VCC — the comparison narrative

**Status:** synthesis note. Consolidates the measured evidence for the
benchmark's central claim, drawing on the VBench dimension audit
(`2026-09-14-vbench-dimension-audit-findings.md`) and the reduced-reference probe
(`2026-09-14-reduced-reference-probe-findings.md`). Ring Forcing (arXiv
2608.26794) supplies external corroboration; its deeper integration is a
separate follow-up.

## The claim

Single-aspect, **self-referential** consistency metrics are inadequate for
long-video super-resolution. They reward the absence of detail and cannot say
*which frames* a consistency measurement even applies to. LR-VCC — an
**input-referenced** composite — measures exactly what they miss. The argument
has two measured halves and one shared root cause.

## Half 1 — what VBench misses (measured)

Audit of the two VBench-1.x temporal/motion consistency dimensions over a 20×
severity ladder, four corruption families, five bases. Mean score-spread across
the ladder (a corruption-sensitive metric should move):

| dimension | color_drift | background_drift | identity_degradation | flicker |
|---|---:|---:|---:|---:|
| temporal_flickering | 0.0004 | 0.0016 | 0.0001 | **0.0257** |
| motion_smoothness | 0.0002 | 0.0012 | 0.0001 | **0.0053** |

Both dimensions respond **only to fast, global flicker** — the corruption they
were designed for, where the score correctly falls — and are **blind** to
gradual colour/background drift and face-local blur. The boundary is
high-frequency-global (seen) vs low-frequency/local (missed), and the missed
side **is where SR failures live**: SR models produce gradual drift and
detail/identity loss, not flicker. On the same `color_drift` family where both
VBench dimensions are flat, LR-VCC's anchored-colour sub-metric (D′) responds on
4/5 bases. **And they actively reward detail loss.** On a global-blur family, both dimensions' scores *rise* on all five bases as blur destroys detail (temporal_flickering +0.0026, motion_smoothness +0.0021 mean Δ; 10/10 base×dim cells inverted) — VBench would rank a detail-destroying SR model as *more* temporally consistent and smoother, the temporal/motion analogue of the identity reductio.

## Half 2 — what LR-VCC adds (measured)

The reduced-reference reframe answers the two questions a self-referential metric
cannot:

- **Applicability** — measure drift on the output *relative to the input*. A
  synthesised legitimate pan reads as a large corruption when self-anchored
  (0.231/0.683/0.126) but collapses **6–23×** when input-differenced
  (0.010/0.065/0.020), while a real colour corruption still registers. The metric
  switches itself off exactly where a change is legitimate.
- **Correspondence** — cluster faces across the whole video from the **input**:
  29 clean identity clusters on one base (separability 0.472), one identity
  **linked across a ~2-minute absence**; max span 102/104 windows on a second
  base. Two-second-window measures structurally cannot do this.
- **The reductio** — measured over two-second windows, the low-quality input
  itself scores **0.650** identity self-similarity, *higher than every
  super-resolved version*. A self-anchored identity metric therefore ranks the
  degraded input as the most consistent video in the set.

## The shared root cause

Both halves are the same fact. **Self-similarity is maximised by removing
detail.** VBench's dimensions, and any output-vs-itself metric, inherit it: a
blur has no flicker, static frames interpolate perfectly, a low-detail face
matches itself. The input is the reference that breaks the degeneracy — it says
what should stay fixed, what may legitimately change, and which subject is which.

## External corroboration — Ring Forcing's A-D-R benchmark

Ring Forcing (long-video *generation*) independently built an
**Appear–Disappear–Reappear** benchmark: 64 controlled cases, a subject keyword,
and disappearance gaps swept over 0/1/5/15/30/60 s, scoring reappearance with
SIFT/LoFTR/DINOv3 feature matching. That is the generation-side twin of our
correspondence problem, and it validates that "appears early, reappears late" is
a recognised, benchmarkable question — not a niche worry.

The difference is our edge: **generation has no reference**, so Ring Forcing must
compare the output's reappearance to the output's first appearance — the same
self-comparison whose weakness Half 2 documents. **Super-resolution has the LQ
input**, so we can run an A-D-R-style test *reduced-reference*: establish
who-is-who and what-it-looked-like from the input, then score whether the SR
output preserved it across the gap. Notably, Ring Forcing also reports **AMT
motion_smoothness** as a headline metric — the exact dimension Half 1 shows is
blind to gradual drift.

## What this positions for the paper

- A **two-sided, measured** argument: VBench misses SR's real failure modes;
  LR-VCC catches them, with a mechanism (input-referencing) not a heuristic.
- The **controlled-input dataset** direction agreed this period (single-shot
  clips, à la Ring Forcing's training filter) gives input-consistency ground
  truth to validate input-differencing.
- The next experiment writes itself: a **graded A-D-R correspondence benchmark
  for the SR setting** (controlled re-entry gaps, scored against the LQ), turning
  the qualitative correspondence result into a quantitative one directly
  comparable to Ring Forcing's table.

## Follow-ups (open)

- Pull Ring Forcing's exact A-D-R table numbers as targets.
- Decide whether the reduced-reference framing is formally adopted (the standing
  scope decision) — it is the precondition for the A-D-R-for-SR experiment.
