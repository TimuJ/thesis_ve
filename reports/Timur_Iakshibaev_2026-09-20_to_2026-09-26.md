# Weekly Progress Report — Timur Iakshibaev

## Headline

**The VBench-dimension audit is complete, and its final result is a measured
reward-inversion.** Across five graded corruption families, VBench's temporal
and motion "consistency" dimensions are blind to gradual drift and local
degradation, respond correctly only to fast global flicker (a corruption
super-resolution does not produce), and — the key finding — **rate a
detail-destroyed (globally blurred) video as *more* consistent on every base
tested.** Since detail loss is exactly what a weak SR model produces, these
dimensions would rank the worse method higher. This is now measured on the
battery, not argued. Alongside it: the VBench-vs-ours comparison is consolidated
into one narrative, an external paper (Ring Forcing) independently validates the
correspondence problem, and the evaluation-dataset direction is scoped.

## 1. The audit, completed — a four-regime result

Two VBench-1.x dimensions (`temporal_flickering`, `motion_smoothness`), five
corruption families, five bases, a 20× severity ladder. Mean score-spread and
direction (a corruption-sensitive metric should *fall*):

| family | temporal_flickering | motion_smoothness | behaviour |
|---|---|---|---|
| color_drift | 0.0004 | 0.0002 | **blind** |
| background_drift | 0.0016 | 0.0012 | **blind** |
| identity_degradation (face blur) | 0.0001 | 0.0001 | **blind** |
| flicker (fast global) | 0.0257 ↓ | 0.0053 ↓ | correct (falls) |
| **global_blur (detail loss)** | **+0.0026 ↑** | **+0.0021 ↑** | **inverted (rises)** |

The inversion is **unanimous — 10 of 10 base×dimension cells rise** as blur
worsens. Magnitude is modest (the metrics sit near a ceiling), but the *sign* is
the finding. The boundary is precise: seen = high-frequency global; missed =
low-frequency and local; rewarded = global detail loss. It is the temporal/motion
analogue of the identity reductio (self-similarity is maximised by removing
detail). Figure `reports/figures/vbench_reward_audit.png`.

## 2. Comparison narrative consolidated

The argument is written as one note: **half 1** — what VBench misses and inverts
(above); **half 2** — what LR-VCC adds, measured (input-differenced applicability
collapses a legitimate-pan false positive 6–23×; input-anchored correspondence
links an identity across a ~2-minute absence; the reductio where the degraded
input scores highest on self-similarity). Both halves reduce to one root cause:
self-referential metrics reward the absence of detail; the input is the reference
that breaks the degeneracy.

## 3. External corroboration — the correspondence problem is benchmarkable

A recent long-video paper (Ring Forcing) independently built an
**Appear–Disappear–Reappear** benchmark — controlled cases with a subject that
leaves and returns across graded gaps (up to a minute) — scoring reappearance by
feature matching. That is the generation-side twin of our correspondence problem,
which confirms it is a recognised, benchmarkable question. The distinction is our
edge: generation has no reference and must self-compare (the weakness half 2
documents); super-resolution has the low-quality input, so the same test can be
run *reduced-reference*, with ground truth. Notably that paper also reports AMT
motion-smoothness — the exact dimension the audit shows is invertible.

## 4. Evaluation-dataset direction (the current strategic goal)

To run a graded, quantitative correspondence benchmark we need footage that is
**naturally consistent, contains genuine disappear→reappear gaps, and is long
enough** — with known correspondence across the gap. The landscape splits, and
the split is the decision:

- **Long-term tracking datasets** — LaSOT, TLP, VOT-LT/LTB-35 — *are built around
  disappearance and reappearance* with per-frame identity ground truth and
  explicit absence labels (LTB-35 averages ~12 disappearances per video; TLP
  sequences run many minutes). This is exactly the controlled correspondence the
  benchmark needs. Weakness: resolution/quality is variable and often below what
  fidelity super-resolution wants.
- **High-resolution long video datasets** — UltraVideo (UHD-4K, minutes-long, the
  set Ring Forcing trained on), UVG (4K, short), Inter4K (4K, ~5 s) — give clean
  ground truth but are single-shot, so the subject is continuously present and
  there is *no* natural disappear-reappear gap.

No single dataset gives all three. **Recommendation:** use a long-term tracking
set (LaSOT or TLP) as the correspondence backbone — its identity/absence
annotations are the "controlled input consistency" agreed at the last meeting —
and, because this is a *consistency* rather than a pixel-fidelity benchmark
(inputs are degraded then super-resolved regardless), the resolution compromise
is acceptable; filter to the higher-quality sequences. Complement it with a small
curated high-resolution set for the fidelity end. Our existing face-clustering
tool (from the reduced-reference probe) already auto-discovers these gaps, so it
doubles as the annotation instrument.

## 5. Problems / infrastructure

- The shared GPU box spent stretches this period refusing SSH connections
  (`kex_exchange_identification: Connection reset` — sshd throttling under a load
  average around 10 with ~60 users). It delayed *retrieving* the last audit
  result by a few days but cost nothing: the jobs run detached and completed on
  schedule; results were pulled once the host recovered. Worth flagging as a
  recurring pace-limiter on the shared machine.

## Next week

1. Pilot the correspondence backbone: pull a handful of LaSOT/TLP sequences with
   disappear-reappear gaps, degrade → super-resolve, and run the reduced-reference
   correspondence score against the identity ground truth — a first quantitative
   A-D-R-for-SR point.
2. Optionally extend the corruption audit's inversion result to a second global
   detail-loss family (downscale-upscale) to show it is blur-general, not
   Gaussian-specific.
3. Resolve the standing scope decision (formally adopt the reduced-reference
   framing), the precondition for the A-D-R-for-SR experiment.
