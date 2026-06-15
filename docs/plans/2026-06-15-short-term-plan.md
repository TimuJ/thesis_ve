# Short-Term Plan — June 15 → July 15, 2026

> **Window:** 4 weeks. **Hard endpoint:** July 15 blind-review submission.
>
> **Reading order:** this file first, then `2026-06-15-long-term-plan.md`
> for July 15 → September 30 (and beyond).

## Where we are on June 15

- LR-VCC **v5 composite** is the production metric.  D + D' + D'' + A + T + I + E,
  softmax-reliability-weighted log-mean composition.
- **Synthetic validation set:** 5 base videos × 12 artefact families × 5 severities
  = 300 clips.  All 30 phase-1 composites done; 25/30 phase-2 (flip family)
  composites done; `flip_invert` still on the dying server.
- **Headline result:** `background_drift` inversions 4/5 → 1/5 (BrRLK cartoon
  remains, halved magnitude).  Diagnosis confirmed + fixed.
- **Lab server is being decommissioned.**  Local + Google Drive + git remote
  all hold sufficient backups (`docs/server_restore_guide.md`).  Restore on
  a new GPU host is ~2 h for read-only, more for full pipeline.

## Constraints and risks

| risk | impact | mitigation |
|---|---|---|
| New GPU server not available in time | blocks real-SR-model evaluation (Priority 1) and any new artefact runs | already have MGLD + UAV outputs locally → can still compute LR-VCC v5 on them; classmate-supplied outputs evaluated on whichever server we find |
| Classmate models late or absent | thinner ranking table | run the table with 3 in-house methods (MGLD, UAV, RealESRGAN per-frame) — still a strong result |
| `flip_invert` row never completes | one cell missing in 12×5 matrix | predicted PASS row (control) — thesis story unchanged, document as ⏳ in the appendix table |
| Methodology / experiments chapters take longer than estimated | misses July 15 blind-review deadline | freeze experiments earlier (June 28 vs July 1) and route all spare time to writing |

## Goals by July 15

In priority order — drop from the bottom if compute or time slips.

### P1. Real-SR-model evaluation with LR-VCC v5 (the headline thesis result)

Apply v5 composite to existing SR-model outputs on the 5-video set + a frame-wise
lower anchor.  This is the "synthetic artefacts taught us what the metric does;
real models show it ranks them differently than PSNR / SSIM" demonstration that
makes the thesis publishable.

- [ ] On a GPU host, run the 7-stage metric battery for:
  - MGLD-VSR (outputs already in `results/mgld_synthetic_mp4/`)
  - Upscale-A-Video (outputs already in `results/uav_synthetic_mp4/`)
  - RealESRGAN per-frame (need to run — frame-wise SR as the
    "no temporal modelling" anchor)
- [ ] Pull JSONs, compose v5 LR-VCC + PSNR + SSIM + LPIPS
- [ ] Build a model-ranking table: 3 models × 5 videos × {LR-VCC, PSNR, SSIM, LPIPS}
- [ ] **Decision criterion:** if LR-VCC orders models the same way as
      PSNR (any frame-wise metric), the contribution is weaker — write the
      "complementary, not orthogonal" framing.  If it orders them differently
      in a way perceptual intuition supports, write the "captures long-range
      consistency PSNR can't" framing.
- [ ] Headline figure for §5.3 (or wherever the experiments chapter lands).

**Effort:** 2 days battery + 1 day analysis = 3 days.  GPU-bound.

### P2. β / α sensitivity sweep (rigour — addresses obvious reviewer question)

`scripts/lr_vcc/compare_d_variants.py` already supports varying `dprime_beta`
and `dprime2_beta` from cached trajectory JSONs.  Plus the softmax temperature
τ in composition.

- [ ] Sweep `dprime_beta` ∈ {0.25, 0.5, 1.0, 2.0}; `dprime2_beta` ∈ {1.0, 2.0,
      3.0, 5.0}; τ ∈ {0.1, 0.2, 0.5}.  Recomputable from cached data — no GPU.
- [ ] Verdict matrix at each parameter point.  Confirm headline results stable
      within a wide neighbourhood of the production values.
- [ ] One paragraph + sensitivity table in the methodology chapter.

**Effort:** half a day, pure recompute.  Local only.

### P3. Leave-one-out sub-metric ablation

With 7 sub-metrics (A, T, I, D, D', D'', E) the question "which sub-metric
catches what" is the most natural follow-up question reviewers will have.

- [ ] Drop each of the 7 in turn; recompose v5; build the verdict matrix.
      Each artefact family attributes to the sub-metric whose removal moves
      it most.
- [ ] Methodology chapter paragraph: "D' uniquely catches background_drift;
      D'' uniquely catches flip_transpose; D uniquely catches chunk_boundary; ..."

**Effort:** half a day, pure recompute.  Local only.

### P4. Classmate-supplied SR-model outputs (if they engage)

The original Week-2 plan included sourcing classmate model outputs to widen
the ranking-table.  Still a bonus, not a requirement.

- [ ] Send LR-input package to classmates by June 18.  Spec: 5 LR mp4s + a
      one-page submission spec (filename convention, fps, CRF, etc.).
- [ ] Soft deadline for receipts: June 25.  Evaluate any received outputs by
      June 28.
- [ ] Whatever lands gets a row in the ranking table.

**Effort:** 2 hours setup, then async waiting.

### P5. Writing — methodology + experiments chapters

The thesis structure is already in place at `zjuthesis/body/graduate-eng/`.

- [ ] **June 22:** Switch `zjuthesis/zjuthesis.tex` from `Period=proposal` to
      `Period=paper`.
- [ ] **June 22–25:** Rewrite methodology chapter.  ~70 % is liftable from
      the proposal (preliminary work section).  Major new content: D' / D'',
      flip ablation, v5 composite.
- [ ] **June 26–30:** Rewrite experiments chapter.  Major new content: v5
      verdict matrix, sub-metric breakdown tables, model-ranking table (P1
      result).
- [ ] **July 1:** **Hard experiment freeze.**  Nothing new computed after this.
- [ ] **July 2–10:** Polish + figures + introduction + literature review +
      conclusion.
- [ ] **July 10–12:** Internal proofread (Task 15 from the proposal sprint
      that's still pending).
- [ ] **July 13:** Set `BlindReview=true` in `zjuthesis.tex`.
- [ ] **July 14:** Sanity build (`cd zjuthesis && latexmk`), check no
      identifying information, final read-through.
- [ ] **July 15:** Submit blind review.

**Effort:** ~3 weeks of writing.  Local only.

## What's deliberately NOT in scope before July 15

- BrRLK cartoon-content limitation fix.  Documented as content-domain limit.
- `flip_horizontal` invisible-to-CLIP fix.  Documented as metric limit.
- Slow-fast pooling pathology beyond the parked dispersion gate.
- Human-judgement correlation study.  Nice-to-have, not blocking.
- Long-range temporal SSM angle from arxiv 2505.20171.  Long-term plan
  territory (moves to after-blind-review or paper period).
- 8-base scale-up.  5 bases sufficient; more would only confirm.

## Pipeline of dependencies

```
[server restored]──→ P1 (real-SR-model eval) ─────────────────────┐
                                                                  │
P2 (sensitivity sweep)  ──────────────────────────────────────────┤
P3 (LOO ablation)       ──────────────────────────────────────────┤
                                                                  ▼
P4 (classmate models)   ──────────────────────────────────────→ writing
                                                                  ▼
                                                              July 15 submit
```

P2 / P3 are GPU-free and can start immediately even without a server.
P1 is the critical path through whatever GPU access we secure.  P5 (writing)
runs from June 22 onward, gated by P1 / P2 / P3 results.

## Bi-weekly check-ins

- **June 26:** mid-window check.  By this date P1 should be ≥ 50 % done,
  P2 + P3 done, methodology chapter draft started.
- **July 5:** post-freeze check.  By this date all experiments done, writing
  in full swing.
- **July 13:** pre-submit dry-run.  Full LaTeX build, blind-review flag set,
  PDF reviewed end-to-end.

If June 26 check fails P1 progress, decide whether to cut P4 (classmate
models — easiest to drop) and reallocate.

## Open decisions blocking this plan

1. **Which GPU host replaces the lab server?**  Need to confirm before
   June 18 or push P1 timeline.
2. **Classmate outreach copy.**  Draft a one-paragraph ask + the submission
   spec; have the supervisor approve before sending.
3. **Will the supervisor or PhD-student collaborator participate in writing?**
   If yes, set up a shared LaTeX-track (overleaf / git branch) by June 22.
