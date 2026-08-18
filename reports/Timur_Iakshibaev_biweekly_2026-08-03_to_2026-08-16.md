# Biweekly Report — Timur Iakshibaev

## The period in three headlines

1. **The video-scaling phase was designed and handed over, after the lab
   footage route closed.** With HR footage from the lab no longer available,
   the set is now built entirely from downloadable sources under fixed
   design rules: **native-LQ only** (genuinely degraded real footage, no
   synthetic downscaling of pristine 4K), **single-shot only**, and
   **degradation-stratified quotas** rather than scraping one source —
   bitrate-starved modern uploads, analog-era digitisations, low-light/sensor
   noise, low-res broadcast archive, with about half the set carrying
   persistent faces so the identity sub-metric stays exercised. Diversity
   across degradation *families* becomes a designed, defensible property of
   the benchmark rather than an accident of sourcing. Staged scale-up to 15
   bases takes the verdict matrix from 60 toward 180 cells. Execution now
   sits with a senior colleague.
2. **The metric-calibration phase was unblocked ahead of schedule and
   delivered — with a deliberately unflattering headline.** The roadmap had
   this phase blocked on the enlarged video set, on the grounds that five
   videos cannot be split into calibration and validation subsets.
   Leave-one-base-out cross-validation *is* a disjoint split, so the phase
   ran now: the severity battery was promoted from pass/fail validator to
   **fit objective**, and every sub-metric's response parameters were fitted
   under five-fold rotation. Mean held-out loss falls **35%** and improves on
   **4 of 5 videos** on a paired same-base comparison — while the
   reader-facing conformance count is **unchanged at 39/55**. Both numbers
   are reported; the candidate metric is **not adopted**.
3. **The 14 failures that calibration provably cannot reach were isolated,
   and the dominant mechanism was pinned.** Every failing cell was attributed
   to the stage where the signal was lost. **20 findings are reachable by
   refitting constants; 14 are not.** The largest single group is the
   identity sub-metric, which scores faces against *themselves* rather than
   against a reference — so degradation that washes out identifying detail
   makes embeddings collapse toward each other and the score **rises**. A
   design for the structural replacements is drafted.

## Key numbers

| result | value |
|---|---|
| Severity-response loss, frozen reference | **0.026884** (all bases) |
| Candidate, in-sample vs held-out | 0.011588 vs **0.017376** — the gap is the overfitting, reported as such |
| Paired per-video comparison | **4/5 videos improve**; the exception is KZ (0.0158 vs 0.0048); largest win BrRLK (0.0421 vs 0.0646) |
| Conformance, as-designed | frozen **39/55** → candidate held-out **39/55** — unchanged |
| Control families (must stay silent) | frozen 14/15 → held-out **15/15**; **in-sample 12/15** — the fit tried to trade control silence for response and it did not survive to unseen video |
| Wrong-direction cells | **4 cells** that actively rewarded the corruption become inert under the candidate |
| Failure attribution | **20 calibration-addressable** (composition 13, gate 4, normalisation 3) vs **14 structural** (measurement 9, reward-direction 5) |
| Structural findings by sub-metric | identity **6**, appearance **5**, CLIP trajectory 2, anchored histogram 1 |
| Identity pathology, measured | fused score **0.375 → 0.489** as identity degrades (per-clip e.g. 0.767 → 0.933) |
| Parameter identifiability at n=5 | roughly half the parameters unconstrained; two of five folds optimise to the grid edge on 4–5 of 7 parameters |
| Frozen-reference integrity | **0.0** worst-case difference across all 315 clips, pinned as a regression test |
| Engineering | 8 TDD tasks, 19 commits, **253 tests passing**; full recomposition 2.7 s → sub-millisecond, whole fit under a minute on a laptop |

## Decisions and framing this period

- **The calibration phase does not need the enlarged video set to begin.**
  Leave-one-out gives a genuinely disjoint calibrate/validate split at the
  current size — weak statistics, sound protocol, and the identical code path
  re-runs on the larger set. This unblocked roughly a month of dependent work.
- **The candidate metric is not adopted; the frozen reference stands.** A 35%
  loss improvement with an unchanged conformance count does not justify
  replacing the published configuration. The recommendation is to treat this
  as infrastructure and re-fit once the video set grows. Put to the group as
  an open decision rather than settled.
- **Controls carry a heavier penalty than response, by design.** Fitting a
  metric to react to corruption has a trivial cheat — react to everything.
  Roughly a quarter of the objective is silence constraints on families that
  are *predicted invisible*, weighted 3× the response term. The in-sample
  drop to 12/15 shows the guard is load-bearing, not decorative.
- **A previously planned structural fix was dropped on the evidence.** A
  mirror-sensitive sub-metric had been named as needed work. The mirror
  family is a *control* — staying flat is its correct outcome, it conforms
  everywhere, and it contributes zero structural findings. Building it would
  fix nothing and put a passing control at risk. Now a stated non-goal.
- **The expectation audit is sequenced first and frozen before any new
  measurement.** Several "measurement failures" are likely bookkeeping:
  appearance is declared as a target for corruptions that leave every frame
  individually plausible, so per-frame quality genuinely should not move.
  Narrowing that map improves the count without improving the metric, so it
  is done first, justified in writing, and reported under both the old and
  new maps.

## Outcomes this period

- [x] Video-scaling phase designed end to end after the lab-footage route
      closed: sourcing rules, degradation quotas, source pool, staged
      scale-up, and release model all fixed; execution handed to a colleague.
- [x] Calibration design + implementation plan written, reviewed, and
      executed as 8 test-first tasks; merged and pushed.
- [x] Battery promoted from validator to fit objective — all five severity
      points now read, where the previous protocol used only the two
      endpoints.
- [x] Leave-one-base-out protocol, with a test that intercepts the arguments
      of every call made during fitting to prove no held-out video leaks in
      through the loss *or* the leaderboard guard.
- [x] Per-cell failure attribution across five stages, separating what
      calibration can and cannot reach.
- [x] Four research reports generated and committed; weekly and biweekly
      reporting package delivered.
- [x] Structural sub-metric design drafted, with targets derived from the
      attribution rather than intuition.
- [ ] Video set scale-up — with a colleague; blocks the statistical claims.
- [ ] SeedVR2 leaderboard row — still needs an 80 GB slot or the rotary patch.
- [ ] GPU time for the structural work — a changed measurement means
      re-scanning the battery.

## One-line summary for the meeting

The severity battery now calibrates the metric instead of merely grading it,
under a cross-validation protocol that reports on videos the fit never saw —
and it returns an honest split verdict: the fit objective improves 35% while
the conformance count does not move, which localised the remaining failures
precisely enough to show that 14 of them need new measurements rather than
new constants, the largest being an identity measure that rewards the
degradation it is supposed to detect.
