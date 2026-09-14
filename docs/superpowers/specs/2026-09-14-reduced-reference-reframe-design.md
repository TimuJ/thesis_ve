# LR-VCC reduced-reference reframe — design (Phase C.3 / RR)

**Status:** draft — awaiting user + colleague scope review, then writing-plans for the implementation
**Date:** 2026-09-14
**Predecessors:** `docs/notes/2026-09-14-reduced-reference-probe-findings.md` (5-day spike, recommendation "go"); `2026-08-14-structural-submetrics-design.md` (Phase C.2, anchored identity I′)
**Decision owners:** Timur (research owner); collaborating PhD colleague (scope call)

## Why this exists

A consistency metric must know what is *supposed* to be invariant before it can
measure whether the invariant held. The shipped LR-VCC does not ask this. It
measures the output's similarity to itself — and self-similarity is maximised by
a blur.

The spike measured the sharpest possible symptom of that flaw:

> The LQ input itself scores **0.650** on within-clip identity self-similarity —
> higher than every super-resolved version of the same content. A self-anchored
> identity metric therefore ranks the **degraded input** as the most
> identity-consistent video in the set.

This is the identity analogue of the VBench-2 result that its consistency
dimensions rank the degraded input above both super-resolutions, and it has the
same root cause: self-similarity rewards the absence of detail. A metric that
can be maximised by not super-resolving is not measuring super-resolution.

Two questions raised by the colleague's supervisor turn out to be the same
question with the same fix:

1. **Applicability** — if the background legitimately changes (a camera pan),
   what does "background consistency" mean at all?
2. **Correspondence** — if a person appears at clip 5 and reappears at clip 70,
   how do we know it is the same person, and therefore that the two frames
   should be compared in the first place?

Both need a reference for *what is supposed to stay fixed*. Because we evaluate
super-resolution, that reference already exists and is free: **the low-quality
input.** The reframe is to measure consistency as fidelity to the content
structure the LQ establishes, not as the output's self-similarity. The LQ tells
us what should stay fixed, what legitimately changes, and which face is which.

The spike answered both questions with measurements (Figures A and B) and
returned "go". This document turns that result into an architecture.

## The one decision this document asks for

The reframe changes LR-VCC from **purely no-reference** to **reduced-reference**:
the *invariance* sub-metrics gain a dependency on the LQ input. That is a scope
decision, not an implementation detail, so it is surfaced first.

- For **method evaluation** it costs nothing. The LQ always exists — SR cannot be
  run without it — so a reduced-reference metric is exactly as deployable as the
  current one for ranking SR methods.
- It does cost the clean "purely no-reference" line. Mitigation (§7): v5
  no-reference stays frozen and available; RR is a distinct, recommended
  variant, not a replacement that erases the no-reference claim.
- **Appearance stays no-reference** by design. The LQ is low quality by
  construction; rewarding the output for matching the LQ's appearance would be
  backwards (see Non-goals).

**Recommendation: adopt.** The spike shows the no-reference framing is not merely
weaker but *invertible* — it ranks the worst input best on identity — and
reduced-reference removes exactly that inversion while keeping the one property
that matters for deployment (the LQ is always present). The call belongs to the
research owner and the collaborating colleague; everything below is buildable
the moment it is made.

## What the spike established (the evidence the design rests on)

Full detail in `docs/notes/2026-09-14-reduced-reference-probe-findings.md`.
Condensed:

**Applicability — Figure A** (`reports/figures/rr_probe_applicability.png`).
Anchor drift (Lab-histogram distance, first quarter vs last quarter — the exact
statistic the `color_hist_anchor` sub-metric uses), measured self-anchored on the
output vs LQ-differenced (output drift minus LQ drift on the same frames). A
legitimate pan was synthesised as a translating crop applied identically to LQ
and output.

| base | pan, self-anchored | pan, LQ-differenced | colour drift, self | colour drift, LQ-diff |
|---|---:|---:|---:|---:|
| 7WHI2L_FDNg | 0.231 | **0.010** | 0.619 | 0.419 |
| mJog8DlRk_4 | 0.683 | **0.065** | 0.502 | 0.091 |
| hhszUXL1Cu8 | 0.126 | **0.020** | 0.134 | 0.037 |

LQ-differencing collapses the legitimate-pan false positive 6–23× while genuine
colour drift still registers. Applicability stops being a heuristic gate and
becomes automatic: where the LQ legitimately drifts, the difference cancels.

**Correspondence — Figure B** (`reports/figures/rr_probe_correspondence.png`).
Faces across a whole video, embedded and clustered into identities from the LQ
alone (native 320×180). On 7WHI2L_FDNg: **4050 faces over 72 of 83 clips → 29
clean clusters** (intra 0.637 vs inter 0.165; separability 0.472). One identity
is present in clips 5–9, absent for ≈2 minutes, and reappears at clips 70–72 —
linked as one person from the LQ alone; another spans clip 1 → clips 80–81
(≈160 s). Second base mJog8DlRk_4 reproduces the cross-time linking (max span
102 of 104 clips) at lower separability (0.312, from 1708 faces) — correspondence
holds on both bases; **anchor quality is content-dependent.**

**The negative that sharpened the design.** Identity degradation does *not*
collapse whole-video clustering (separability flat 0.434→0.461). It collapses
**within-clip** self-similarity — exactly where the shipped metric measures — as
severity rises (0.578 flat until the 0.40 rung, then 0.617). The correspondence
layer is robust precisely where the scoring layer is fragile. This is what makes
anchoring correspondence to the LQ and scoring the output *against those groups*
the right split of labour.

**The alignment defect.** Frame counts match exactly on every base; fps *tags*
differ on every base (LQ carries the true rate; the SR pipeline re-tags to a
uniform 30.000). At two-second clips this silently mis-splits the two 24 fps
bases (104 clips vs 83) and only on some content. It fed the design a firm
invariant (Workstream 0).

## Sub-metric partition — which measurements actually change

The reframe is not "make everything reference the LQ". It applies only to
sub-metrics that measure an **invariance** (does something that should stay fixed
stay fixed). Quality measures stay no-reference.

| sub-metric | what it measures | reframe | rationale |
|---|---|---|---|
| **I** identity (ArcFace) | who is who, over time | **reduced-ref** (correspondence layer) | Proven: Figure B + reductio. Self-anchoring is invertible. |
| **D′** color_hist_anchor | Lab-hist drift, quarters | **reduced-ref** (LQ-difference) | Proven: Figure A. Directly reuses the existing statistic. |
| **D″** clip_trajectory (CLIP) | semantic drift over clips | candidate reduced-ref | Same drift family as D′; LQ-differencing is the natural extension, unproven in spike → test in WS1. |
| **T** temporal (RAFT tOF) | flow smoothness | candidate reduced-ref | Legitimate camera motion shows in the LQ flow too; differencing could separate legitimate motion from output jitter. Test in WS1. |
| **D** color_stability | consecutive-frame hist | candidate (low priority) | Mostly output-internal jitter; LQ-difference only where the LQ has legitimate frame-to-frame colour change. |
| **E** color_slope | drift slope | tied to D | Follows D's decision. |
| **A** appearance (CLIP-IQA) | per-frame quality | **stays no-reference** | The LQ is low quality; matching it is wrong. Quality is genuinely reference-free here. |

Two proven reframes (I, D′), three candidates to be settled empirically
(D″, T, and the D/E pair), one deliberate no-reference hold-out (A). The design
below builds the two proven ones and specifies the test that decides the
candidates.

## Workstream 0 — the alignment invariant (do first; it is free and prevents silent corruption)

Every reduced-reference comparison pairs an LQ frame with an output frame. If the
pairing is ever keyed on timestamp or clip index it is silently wrong on the two
24 fps bases, and only on some content — the worst kind of bug.

**Deliverable.** A single shared alignment utility that maps LQ↔output **by frame
index**, used by every reduced-reference sub-metric, plus a test that constructs
an fps-mismatched pair (true 24 vs re-tagged 30) and asserts the frame pairing is
unchanged. Frame counts are identical on all five bases, so frame-index
alignment is exact everywhere; nothing else is.

**Why first.** It costs almost nothing, it is a precondition for WS1 and WS2 to
be correct rather than accidentally correct, and it retro-explains the per-video
fps override list the identity stage already carries — that list exists for
exactly this defect and can be retired once alignment is index-based.

**Validation criterion.** The alignment test passes on a synthetic 24-vs-30 pair;
re-running any existing sub-metric through the new loader is bit-identical on the
three ~29.97 bases (where clip grids already matched) and now *also* correct on
the two 24 fps bases.

## Workstream 1 — applicability by LQ-differencing (Figure A → production)

**Mechanism.** For each drift-type sub-metric, compute its raw statistic on the
output and on the LQ over the frame-index-aligned window, and score the
**difference**, not the output value. Where the LQ legitimately drifts, the
difference cancels; where the output drifts *beyond* what the content does, the
difference registers. Applicability becomes automatic — no heuristic gate.

**Proposed measurement.**
- **D′ (proven):** replace the self-anchored quarter-to-quarter Lab-histogram
  distance with `d(output) − d(LQ)`, clamped at zero (the output cannot be
  *more* stable than the content and earn credit for it). Response curve refit
  against the corruption battery under the existing LOBO protocol.
- **D″, T (candidates):** apply the same difference and measure, on the battery,
  whether (a) the legitimate-change families lose their false-positive response
  and (b) the true-corruption families keep their real response. Adopt each only
  if both hold; otherwise leave it no-reference and record why.

**Validation criterion.** On the synthetic-pan family, LQ-differenced D′ collapses
by the 6–23× the spike showed and the genuine colour-drift family retains
majority response. The pre-registered flip control families still read FLAT.
No previously-conforming battery cell regresses.

## Workstream 2 — the correspondence layer (Figure B → production)

This is the headline and the most work. It is a **new pipeline stage**, not a
tweak to an existing sub-metric.

**Mechanism.** Establish who-is-who once, from the LQ, across the whole video;
then score the output's identity consistency *against those groups*. Never score
the output against itself — the reductio proves self-similarity is maximised by
the worst video in the set.

**Proposed measurement.**
1. **Correspondence pre-pass (LQ side).** Detect and embed every face across the
   whole LQ video (reuses the prototyped `dump_identity_embeddings.py`
   infrastructure — the expensive dump runs once per base). Cluster into identity
   groups by average-linkage cosine cut. Output: a per-base correspondence
   structure mapping each detected face, at each frame index, to an identity id,
   valid across arbitrary time gaps.
2. **Anchor-quality gate.** Correspondence is content-dependent (7WHI
   separability 0.472 vs mJog 0.312). Compute a per-base anchor-quality score
   (intra/inter separability, face density, cluster count) and gate the identity
   sub-metric's *reliability* — not a hard on/off — so sparse-face content
   contributes to the composite with reduced weight rather than a spurious
   confident score. This is the reliability-weighting LR-VCC already does, driven
   by a measured quantity.
3. **Output scoring against groups.** For each identity group, gather the output
   frames at the group's frame indices and measure whether the output preserves
   that identity across the group's whole time span (including the long gaps that
   two-second chunking structurally cannot bridge). The score is the output's
   fidelity to the LQ-established correspondence, aggregated per identity then per
   video.

**Validation criterion — the reductio must break.** Under RR identity, the LQ
input must **not** rank as the most identity-consistent video (the self-anchored
metric ranks it first). The cross-time linking demonstrated in Figure B
(identity relinked across a ~2 min absence) must survive on both spike bases.
Anchor-quality must correlate with the spike's separability ordering
(7WHI > mJog).

## Workstream 3 — composition, reliability, versioning

**Composition is unchanged in form.** `weights = softmax(reliabilities/tau)`,
`LR_VCC = exp(Σ wᵢ·log(scoreᵢ+ε))`. The reframe changes what several `scoreᵢ`
mean and adds one measured input to `reliabilityᵢ` (anchor quality for I). No new
free composition parameter beyond the existing `tau`.

**Reliability gains a real signal.** Anchor quality (WS2) and, for the
differenced sub-metrics, the magnitude of LQ drift (large LQ drift ⇒ the
difference is a small residual of two large numbers ⇒ lower reliability) feed the
per-video weights. This is the reliability-derived weighting the metric was
designed around, now grounded in measured quantities rather than fixed heuristics.

**Versioning (mitigates the scope cost).** v5 no-reference stays the frozen
reference; nothing changes retroactively. The RR variant is tracked as a distinct
line (working name **LR-VCC-RR**) so the paper can report *both* — the
no-reference number for the "deployable with zero side information" claim and the
RR number for the "correct under legitimate change and cross-time identity"
claim. The v6 sensitivity calibration is orthogonal and composes: RR sub-metrics
are calibrated with the same battery/LOBO machinery.

## Validation protocol

1. **Battery re-run** with WS1 sub-metrics: applicability improves (legitimate-
   change families lose false positives) with no loss of true-corruption
   response; report per-family, per-base, map version named.
2. **Controls hold:** the pre-registered flip families still read FLAT. A reframe
   that breaks a control is rejected regardless of headline gains.
3. **Reductio anchor:** RR identity does not rank the LQ top (WS2).
4. **Cross-time correspondence** preserved on both spike bases.
5. **Alignment test** (WS0) green; frame-index pairing exact on all five bases.
6. **Honest split verdict**, as with v6: report where RR helps, where it is flat,
   and any cell it costs — no in-sample flattering.

## Cost and dependencies

- **Compute.** The one expensive item is the whole-video LQ face-embedding dump
  (WS2), already prototyped; it runs **once per base** and is cached. WS0 and WS1
  are cheap and reuse cached statistics. No GPU-days beyond the dump.
- **Data scale.** The reframe is *buildable and demonstrable now on the 5 bases*.
  Full validation of correspondence quality (true number of people, threshold
  sensitivity) wants the enlarged, diversified set — Phase A, colleague-owned —
  so the clustering threshold and anchor-quality gate are fitted on more than two
  face-bearing bases. Build now; final numbers scale with Phase A.
- **Reuses:** `dump_identity_embeddings.py`, the corruption battery + LOBO
  calibration harness, the reliability-weighting composition, the pre-registered
  controls.

## Non-goals

- **No mirror-sensitive sub-metric.** Ruled out in Phase C.2 on the data
  (`flip_horizontal` is a SILENT control; FLAT is correct); the reframe does not
  revive it.
- **Appearance is not made reduced-reference.** Matching the LQ's *quality* is
  wrong; A stays no-reference.
- **The clustering threshold is not fitted in this phase.** The spike's cosine
  cut (0.55) is carried as a provisional constant with sensitivity flagged;
  fitting waits for Phase A face-bearing bases.
- **No human validation here.** That is Phase B; RR gives it a better metric to
  correlate against, nothing more.
- **v5 is not retired or altered.** RR is additive.

## Risks

- **Framing cost is real.** "No longer purely no-reference" is a genuine giveback;
  mitigated by keeping both lines (§7) and by the reductio, which shows the
  no-reference line is invertible on identity and therefore not a pure loss to
  qualify.
- **Synthetic pan.** Figure A's pan is a translating crop, not real camera motion.
  It isolates the mechanism; it does not prove behaviour on real pans. First
  Phase-A validation item.
- **Anchor-quality content dependence.** Sparse-face content links identities but
  less cleanly (mJog 0.312). The gate turns this into reduced weight rather than a
  wrong confident score, but content with essentially no faces gives the identity
  channel little to say — expected and reliability-handled, not a defect.
- **Threshold sensitivity.** Correspondence depends on the unfitted cosine cut;
  the LQ yields 29 clusters where the output yields 91–109 at the same threshold
  (LQ consolidates pose variants, output fragments them). Which is closer to
  ground truth is untested. Report the sensitivity; do not present a single
  cluster count as truth.
- **Alignment regression.** Any future code path that reintroduces timestamp/clip
  alignment silently corrupts two of five bases. WS0's test is the guard;
  it must be kept.

## Open decisions (surfaced for review)

1. **Scope (the decision above):** does the benchmark formally adopt
   reduced-reference framing? Owner: research owner + colleague.
2. **Candidate sub-metrics:** which of D″, T, (D/E) become reduced-reference —
   settled by the WS1 battery test, but the *default if a candidate is
   ambiguous* (keep no-reference vs adopt) is a judgement call to confirm.
3. **Anchor-quality gate:** reliability-weight (proposed) vs hard gate — confirm
   the soft-weight choice.
4. **Versioning/naming:** LR-VCC-RR as a parallel line vs a v-number bump —
   confirm before writing-plans, since it sets the report/table layout.
