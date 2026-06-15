# Long-Term Plan — July 15, 2026 → September 30, 2026 (and beyond)

> **Window:** ~11 weeks of bounded thesis work, then open-ended research.
>
> **Reading order:** read `2026-06-15-short-term-plan.md` first for the
> July 15 blind-review handoff that this plan picks up from.

## Phase 0 — context

By July 15 the blind-review thesis should be submitted (per the short-term
plan).  This document covers what happens after — three discrete phases plus
an open-ended research direction section.

| date | event | who decides |
|---|---|---|
| **Jul 15, 2026** | Blind-review thesis submitted | Timur ships |
| Jul 15 – Aug 15 | Review period | Reviewers |
| Aug ~15 | Reviewer feedback received | Reviewers + advisor |
| Aug 15 – Sep 25 | Revisions and final-version preparation | Timur + advisor |
| **Sep 30, 2026** | **Final thesis submission deadline** | Hard university deadline |
| Sep – Dec (rolling) | Paper co-write with PhD-student supervisor | PhD student + Timur |
| 2027+ | Optional follow-ups (open research directions) | Timur, post-graduation |

The thesis quality bar (per the supervisor) is **CCF-B / ECCV class**.  Path A
of the graduation strategy is "all reviewer evaluations A / B → thesis
passes."  Path B is the invention patent route via substantive examination.

## Phase 1 — review period (July 15 → ~Aug 15)

Mostly waiting on external reviewers, but the time is *not* free.

### Goals

1. **Make zero substantive changes to the submitted thesis.**  The blind-review
   PDF is the document under evaluation; touching it now causes version-drift
   problems later.
2. **Start the paper co-write** with the PhD-student supervisor.  Paper structure
   overlaps heavily with thesis chapters 3–5 (methodology + experiments); paper
   adds related-work density and tighter writing.
3. **Address known limitations** in parallel — these become future-work bullets
   if untouched, or new contributions if solved.
4. **Recover.**  Take a deliberate break.  Thesis writing is exhausting.

### Action items

- [ ] **By July 18:** sync with PhD-student supervisor on paper outline.
      Target venue (CCF-B → AAAI / IJCAI / ACM MM 2027 January deadline; or
      ECCV 2026 March deadline).  Confirm authorship order.
- [ ] **July 18 – Aug 15:** draft the paper, leveraging thesis content.  ~20 %
      new content vs thesis: tighter intro, denser related work, ablation
      depth.
- [ ] **July 18 – Aug 15:** in parallel, work on one of the documented
      limitations as a paper extension (see "open research directions" below).

### Risks

- **Reviewer feedback before Aug 15** might be substantive.  Be ready to pivot.
- **Paper deadline conflict** with thesis revisions (mid-Aug to mid-Sep).
  If both compress at once, prioritise thesis revisions — the paper has more
  flexible deadlines.

## Phase 2 — revisions (Aug ~15 → Sep 25)

Reviewer feedback arrives.  Three possible regimes:

### Regime A — light feedback (A / B grades, minor corrections)

Most likely outcome based on the work already done.  Action: typo fixes,
clarifications, one-paragraph responses.  ~2 weeks of work.

### Regime B — substantive feedback (one reviewer asks for new experiments)

Likely if a reviewer is skeptical about the methodology or wants more bases /
more SR models / a human study.

- [ ] **Resurrect server access** for any compute work demanded.  Restore
      procedure documented at `docs/server_restore_guide.md`.
- [ ] **Triage feedback into "must address" vs "footnote vs disagree."**
      The thesis already documents most known limitations honestly — many
      reviewer asks may already be acknowledged.

### Regime C — heavy feedback (revision-required level)

Less likely.  If it happens: prioritise the highest-impact reviewer asks,
defer paper work entirely, focus all remaining time on revisions.

### Sep 25 — final submission preparation

- [ ] All revisions integrated, BlindReview flag re-toggled to `false`,
      author identifying information restored.
- [ ] Final PDF built, internal proofread.
- [ ] Backup copy of final-final PDF to `~/Downloads/`, git remote, and
      Google Drive.

## Phase 3 — paper co-write (parallel and post-thesis, ~Sep – Dec 2026)

Already started in Phase 1.  After Sept 30, paper work becomes the primary
focus.

### Goals

- Submit to one of: ECCV 2027, ACM MM 2027, ACCV 2026 late-track.
- Target experiment depth: more ablation, more bases, broader SR model
  comparison (RealBasicVSR, RVRT, modern diffusion-based methods if they
  emerge), correlation against human judgement.
- Open-source the LR-VCC benchmark on the project's git remote so it can
  cite back to the published paper.

### Authorship and division

- Timur leads metric design, validation, writing.
- PhD-student supervisor contributes SR-model expertise, paper polish,
  reviewer-perspective sanity checks.
- Author order: discuss in Phase 1 sync.

## Open research directions (post-September, optional)

Each documented limitation in the thesis is also a potential research
extension.  Listed here so future-Timur (or a follow-up student) has a
roadmap.

### Direction 1 — solving the BrRLK cartoon-content limitation

**Problem:** Both D' (Lab-histogram anchor) and D'' (CLIP-trajectory) fail
on the cartoon base because natural scene-cut variation in the source
dominates the anchor-distance signal.

**Possible fixes:**
- Adaptive anchor: re-establish the anchor whenever a scene cut is detected
  (PySceneDetect or similar).  Score is then "drift away from the *current
  scene's* anchor."
- Content-aware reliability: down-weight D' / D'' on content with high
  natural scene-change frequency (cartoons, music videos, sports).
- Replace anchor with a learned video-embedding (e.g. ViCLIP or VideoMAE)
  that already accounts for content-domain variation.

**Effort:** 1–2 months of focused research.

### Direction 2 — beating flip_horizontal blindness

**Problem:** Pure horizontal mirror is invisible to all 7 sub-metrics in v5.
CLIP has documented horizontal-flip robustness; histogram methods preserve
by construction.

**Possible fixes:**
- Pixel-correlation sub-metric: directly measure left-right pixel asymmetry
  in a temporally-pooled way.  Sensitive to mirror by construction.
- Use a CLIP variant trained *without* horizontal-flip augmentation
  (would need to identify or train such a model — significant work).
- Pose-or-action-based sub-metric: track human pose / hand orientations
  with MediaPipe; abrupt orientation changes catch mirrors.

**Effort:** 2–4 months, may make a follow-up paper on its own.

### Direction 3 — slow-fast pooling pathology fix

**Problem:** The parked dispersion gate works in principle but couldn't be
calibrated on 2-base data (only 0.003 separation).  More bases or a better
calibration objective could unparked it.

**Possible fixes:**
- Re-calibrate on a larger base set (≥ 8 single-face videos).
- Replace the slow-fast pooling with a learned aggregator (train on
  human consistency judgements).

**Effort:** 1 month if data exists; longer if data needs collection.

### Direction 4 — connect to long-context SSM video models

**Problem:** The original research direction (arxiv 2505.20171, Long-Context
State-Space Video World Models) was tabled in favour of the benchmark
contribution.  Long-video SR with SSM temporal backbones remains an
underdeveloped space.

**Possible fixes:** train or fine-tune an SSM-based VSR model; evaluate with
LR-VCC; demonstrate that consistency-loss training (using LR-VCC as a
loss signal) improves perceptual long-range consistency.

**Effort:** PhD-thesis territory.  Plausibly Timur's next research arc if
continuing into a PhD; otherwise a hand-off to a future student.

### Direction 5 — broaden the benchmark

**Problem:** LR-VCC's claims are based on 5 base videos.  A community
benchmark needs more — 50+ videos, multiple domains, human-judgement
calibration.

**Possible fixes:** crowd-source / curate a public long-video consistency
benchmark; release LR-VCC code, weights, and the dataset; maintain a public
leaderboard.

**Effort:** infrastructure-heavy, paper-grade contribution, ~6 months.

## Decision check-ins on the long-term timeline

| date | check | escalation |
|---|---|---|
| **Aug 1** | Paper outline finalised with PhD student? | If not, Phase 1 paper work slips → revisit |
| **Aug 20** | Reviewer feedback received? regime A / B / C? | Triage and replan revisions |
| **Sep 15** | All revisions done? | If not, request advisor extension if available |
| **Sep 30** | Final thesis submitted | Hard university deadline |
| **Dec 31** | Paper draft to PhD student for review | Soft; venue dependent |

## What success looks like

- **Path A primary:** all reviewer evaluations A / B; thesis defended on
  schedule; Path B (patent) becomes a nice-to-have rather than fallback.
- **Paper accepted** at a CCF-B-or-higher venue within 12 months of
  graduation.
- **LR-VCC benchmark code + dataset** publicly available, cited by at least
  one downstream project within 24 months.
- **Knowledge handed off** to one continuing student or collaborator who
  picks up Direction 1, 2, or 3.

## What survives this plan being wrong

If the timeline compresses (e.g. heavy reviewer feedback) or expands (e.g.
paper accepted-with-revisions and submissions push out to 2027), the core
deliverables ranked by priority:

1. Pass the thesis.
2. Don't damage existing experimental records — `docs/server_restore_guide.md`
   stays the canonical recovery procedure.
3. Submit *something* to a paper venue, even if not the perfect venue.

If only #1 happens, this was still a successful master's program.
