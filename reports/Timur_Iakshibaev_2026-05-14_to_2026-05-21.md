# Weekly Progress Report — Timur Iakshibaev

## Period: May 14 – May 21, 2026

## Headline

Three substantial wins this week, all building toward the May 31 proposal:

1. **Long-range tOF/tLP eval finished — clean crossover finding.** At adjacent frames (k=1) UAV wins on 4/5 videos (smoothness). At k≥10 MGLD wins on 4/5 videos. The standard adjacent-frame temporal metric (`E*warp`, `tOF k=1`) systematically favours smoother SR — long-range stability flips the ordering. This is the strongest single piece of evidence so far that long-video SR needs multi-time-scale metrics.

2. **Continuous-aggregation Anatomy implemented** as opt-in `--continuous` flag (default stays threshold for paper reproducibility). Halves the KZ8p6b1zJ9U flip-gap but doesn't eliminate it — confirming that the failure has both a threshold-boundary half and a deeper representation half.

3. **LR-VCC (Long-Range Video Consistency Composite) designed and partially implemented.** A no-reference composite that wraps appearance + temporal + identity sub-metrics, each with its own reliability test derived from our characterized failure regimes. Spec + 10-day implementation plan committed; sub-metric wrappers + composition + CLI runner all implemented and tested (20/20 unit tests passing). Server-side CLIP-IQA dump running now; end-to-end Layer 1+2 validation tomorrow.

## tOF/tLP long-range eval — crossover and smoother-output bias

Built `scripts/long_range_temporal/eval_tof_tlp.py` (RAFT forward+backward flow, FB-consistency occlusion mask, masked L2 for tOF and masked LPIPS for tLP). Ran on all 5 synthetic SR videos × 2 methods × 6 `k`-values × 200 pairs per `k`. ~37 min on GPU 4.

### tOF (pixel-level)

| k | Mean MGLD | Mean UAV | Δ (M − U) |
|---|----------:|---------:|----------:|
| 1 | 0.0216 | 0.0177 | **+0.0039 (UAV wins)** |
| 5 | 0.0406 | 0.0424 | −0.0018 |
| 10 | 0.0500 | 0.0618 | −0.0118 |
| 30 | 0.0804 | 0.0922 | −0.0119 |
| 60 | 0.1110 | 0.1314 | −0.0204 |
| 120 | 0.1441 | 0.1682 | **−0.0241 (MGLD wins)** |

Clean crossover at k=5–10. UAV wins adjacent-frame stability (smooth frames warp into themselves better); MGLD wins long-range stability decisively. Per-video at k=120: MGLD wins 4/5.

### tLP (LPIPS-based perceptual)

tLP systematically favours UAV across all k. LPIPS rewards self-similarity → UAV's smoother frames look more like themselves under LPIPS feature distance, even though humans see UAV's output as worse. **Same smoother-output bias as VBench's `subject_consistency` (DINOv2) and `Human_Anatomy` (anomaly ViT) on close-ups** — that's three completely independent learned-representation metrics all exhibiting the same bias. Strong structural evidence for the thesis's central claim.

### Methodological caveat — mask coverage collapses at long k

| k | MGLD coverage | UAV coverage |
|---|--------------:|-------------:|
| 1 | 0.93 | 0.91 |
| 10 | 0.57 | 0.35 |
| 60 | 0.22 | 0.12 |
| 120 | 0.11 | 0.07 |

At k=120 only ~10% of the frame is valid. UAV's coverage is consistently lower than MGLD's at long k — RAFT struggles to find FB-consistent flow on UAV's smoother textures. Long-range numbers compare differently-sized pixel subsets per method; flagged in the writeup.

Full doc: `docs/notes/2026-05-14-tof-tlp-long-range-results.md`.

## Continuous-aggregation Anatomy

Added `--continuous` opt-in flag to `aggregate_slow_fast_anatomy.py` and `human_anatomy_long.py`. Default stays upstream's `1 − fraction_above_threshold` for paper reproducibility; `--continuous` switches to `1 − mean(p_abnormal)` averaged per detector category. Implemented in 5 minutes, validated on all 10 cached per-frame traces (no GPU re-run).

Slow-fast comparison across all 5 videos:

| Video | MGLD threshold | MGLD continuous | UAV threshold | UAV continuous | Δ threshold | Δ continuous |
|-------|---------------:|----------------:|--------------:|---------------:|------------:|-------------:|
| 7WHI2L_FDNg | **0.840** | **0.887** | 0.774 | 0.828 | +0.066 | +0.059 |
| BrRLKMbBTYQ | **0.472** | **0.770** | 0.410 | 0.745 | +0.062 | +0.025 |
| **KZ8p6b1zJ9U** | 0.137 | 0.591 | **0.476** | **0.739** | −0.339 | **−0.148 (halved)** |
| hhszUXL1Cu8 | **0.969** | **0.950** | 0.896 | 0.921 | +0.073 | +0.029 |
| mJog8DlRk_4 | **0.622** | **0.832** | 0.531 | 0.803 | +0.091 | +0.029 |
| **Mean** | 0.608 | 0.806 | 0.618 | 0.807 | −0.010 | −0.001 |

**Per-video winners unchanged (MGLD 4/5)** under both schemes. **KZ gap halves** from −0.339 to −0.148 but doesn't disappear. Trade-off worth flagging: continuous lifts everyone's absolute scores, which compresses inter-method gaps on the 4 MGLD-wins videos (+0.062 to +0.091 → +0.025 to +0.059). Continuous is more *robust* to threshold-boundary content but less *discriminating* in the typical regime. Recommendation: report both side-by-side in the thesis.

Full numbers in `docs/notes/2026-05-13-kz-regime-shift-trigger.md`.

## KZ8p6b1zJ9U regime-shift — content trigger characterized

Bbox-size analysis on the cached per-frame anatomy traces (CPU-only, no re-run):

| Video | Hand bbox p50 (% of frame) | MGLD−UAV anatomy gap |
|-------|---------------------------:|---------------------:|
| **KZ8p6b1zJ9U** | **18%** | **−0.291** |
| BrRLKMbBTYQ | 7% | +0.085 |
| mJog8DlRk_4 | 3% | +0.036 |
| hhszUXL1Cu8 | 1% | +0.047 |
| 7WHI2L_FDNg | 0.5% | +0.097 |

Monotonic across videos: larger body-part bboxes → smaller MGLD-vs-UAV gap (or flip). KZ is a close-up scene; the other 4 are wide/mid-shots.

But two natural follow-up fixes did NOT rescue KZ:

1. **Stable-regime filter** (drop frames where face/hand bbox ≥ 5% of frame area). KZ gap *widens* from −0.339 to −0.437. Even on the 26% of KZ frames without close-up body parts, MGLD's content is flagged abnormal aggressively. Bbox size is predictive *at the per-video level* but not the proximate per-frame cause.
2. **Continuous aggregation** (above). Halves the KZ gap but doesn't eliminate it.

So the failure has two halves: ~50% threshold-near-boundary noise (continuous aggregation fixes this), ~50% deeper representation bias (anomaly classifier genuinely "thinks" MGLD's KZ content looks more abnormal even continuously). The deeper half needs classifier retraining or replacement — too expensive for the proposal, but flagged as thesis-future-work.

Full doc: `docs/notes/2026-05-13-kz-regime-shift-trigger.md`.

## LR-VCC — composite metric design + implementation sprint

This is the main deliverable of the week and the centrepiece of the May 31 proposal.

### Design

A no-reference composite metric for ranking long-video SR methods. Three sub-metrics, each with its own per-video reliability test, combined via softmax-weighted log-mean (geometric mean with reliability weights):

- **Sub-metric A — Appearance stability.** Per-frame CLIP-IQA, `score = mean(quality) − λ·std(quality)`. Reliability drops if `std` too small (sub-metric undiscriminating) or `mean` saturates.
- **Sub-metric T — Temporal stability.** `log(1+k)`-weighted mean of tOF over k ∈ {1, 5, 10, 30, 60, 120} (using the existing tOF JSONs). Reliability drops if mask coverage too low at long k.
- **Sub-metric I — Identity preservation.** Wraps `human_identity_long.py` fused slow-fast. Reliability drops if face-detection rate too low or close-up bbox p50 too high.

Composition: `LR_VCC = exp(Σ_s softmax(reliability)_s · log(score_s + ε))` with `temperature = 0.2`. Sharp softmax lets the most reliable sub-metric dominate when one is much more trustworthy. Log-mean preserves "no compensation for failures" — a sub-metric scoring 0.1 drags the composite down.

Full spec: `docs/plans/2026-05-21-lr-vcc-design.md` (211 lines, hyperparameters table, validation plan).

### Why this design

Built directly on the failure modes we documented:

- Three independent learned representations (DINOv2, Anatomy ViT, LPIPS) reward smoother SR over visually-better diffusion-style SR — Use existing learned metrics but downweight them per-video using regime indicators we already know about (close-up bbox, mask coverage, low face detection).
- Single-time-scale temporal metrics flip with regime (tOF k=1 vs k=120 disagree) — Use multi-k tOF with long-k weighting.
- Single-aspect metrics miss long-video failures — Compose appearance + temporal + identity.

### Implementation sprint (10-day plan, May 21–31)

15-task plan committed: `docs/plans/2026-05-21-lr-vcc-implementation.md`. Subagent-driven execution started today.

| Phase | Tasks | Status |
|-------|-------|--------|
| Phase 1 — sub-metric wrappers + composite | 1–6 | **DONE** (commits `2ff68fa` through `a6811e6`, 20/20 unit tests passing) |
| Phase 2 — Layer 1+2 validation | 7–9 | Task 7 done (close-up map: KZ MGLD 16.2%, others ≤ 7%); CLIP-IQA dump running on server (MGLD finished, UAV in progress); Task 8 (end-to-end run) tomorrow |
| Phase 3 — proposal writing | 10–15 | Task 10 (outline) done; Tasks 11–15 next week |

All Phase 1 code under `scripts/lr_vcc/` (`reliability.py`, `composite.py`, `temporal.py`, `identity.py`, `appearance.py`, `compute_clip_iqa.py`, `run_lr_vcc.py`) + `tests/lr_vcc/` (20 tests).

### What we'll have at the meeting

- Phase 1 fully implemented (sub-metrics + composition).
- Per-video LR-VCC numbers for 5 videos × 2 methods (CLIP-IQA dump in progress; Task 8 to run when it lands ~tomorrow morning).
- Layer 1 + Layer 2 validation results: does LR-VCC give MGLD > UAV on aggregate AND on KZ8p6b1zJ9U (the metric-failure case)?

The Layer 2 result is the key thesis test. If LR-VCC keeps MGLD ahead on KZ when individual metrics flip, the reliability-weighting approach works. If it doesn't, the design needs revisiting.

## Next steps (May 22 – May 31)

1. **Today/tomorrow:** finish CLIP-IQA dump, run LR-VCC end-to-end (Task 8), document validation (Task 9).
2. **Days 6–10 of sprint:** proposal sections (preliminary work, proposed method, validation + timeline, contributions), figures, LaTeX assembly, internal review.
3. **May 31:** proposal submission.

Items punted to post-proposal (thesis future work):
- Multi-person Identity v2 implementation (designed only, in `docs/plans/2026-05-06-multiperson-identity-metric.md`).
- LR-VCC Validation Layer 3 (parameterized synthetic test datasets at controlled artefact severity).
- Long-range temporal metric retraining / classifier replacement to fix the deeper smoother-output bias.

## Talking points for the May 22 meeting

1. **The crossover finding.** tOF flips between k=1 (UAV wins) and k≥10 (MGLD wins) — direct evidence that adjacent-frame temporal metrics undervalue long-range stability. This is the cleanest single result of the week.
2. **Three independent learned representations show the same bias.** DINOv2 (`subject_consistency`), Anatomy ViT (close-up regime), LPIPS (tLP). Structural, not coincidental — suggests "trained on pristine HR" representations all share the bias.
3. **LR-VCC design is committed and partly built.** Composite of A/T/I sub-metrics with reliability weighting. Phase 1 (code + tests) done; Phase 2 (validation on 5 videos) tomorrow. Phase 3 (proposal document) the rest of the sprint.
4. **Continuous-aggregation Anatomy halves the KZ flip-gap.** Implemented as opt-in; default still matches upstream. Useful diagnostic but not a full fix on its own.
5. **Proposal on track for May 31** assuming Layer 1+2 validation results land cleanly tomorrow.

## Blocking questions for the group

1. **LR-VCC reliability hyperparameters.** Defaults are derived from independent characterizations (drift floor 0.02, mask-coverage floor 0.10, face-rate floor 0.20, close-up bbox 0.05 of frame area, softmax temperature 0.2). Any reservations on these before we publish them in the proposal?
2. **Validation Layer 3 scope for the thesis.** 5 parameterized synthetic test datasets (color drift, periodic flicker, chunk-boundary jumps, identity degradation, long-range BG change). Which 2–3 are highest priority for thesis future work?
3. **Real HR long-video baseline.** Currently no-reference only on 5 synthetic videos. Should we invest in sourcing real HR long videos for additional validation, or stay no-reference?
4. **Smoother-output bias — replacement vs reliability-weighting.** LR-VCC takes the reliability-weighting path (cheap, principled, fits proposal scope). Long-term, retraining the offending classifiers (LPIPS, Anatomy ViT) on diffusion-SR data would be the principled fix. Worth pursuing for the full thesis or out of scope?
5. **Time-scale granularity.** Currently `k ∈ {1, 5, 10, 30, 60, 120}` for tOF. Is the multi-k spectrum the right shape, or should we report a single summary number?
