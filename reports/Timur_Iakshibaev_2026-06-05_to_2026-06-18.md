# Bi-weekly Progress Report — Timur Iakshibaev

## Period: June 5 – June 18, 2026

## Headline

This period delivered the long-term-consistency artefacts (`identity_drift`,
`background_drift`) the supervisor asked for at the May group meeting, scaled
the validation set from 2 → 5 base videos, and produced the first complete
6-artefact × 5-base verdict matrix on the LR-VCC composite metric. The matrix
also surfaced a clear mechanistic problem with the metric that we did not
expect to find: **stability-based sub-metrics systematically reward
convergence-type corruptions** — the supervisor independently arrived at
the same diagnosis ("degradation makes videos more stable"). This finding
re-frames the next research period from "add more artefacts" to "redesign
the broken sub-metrics."

1. **Two long-term-consistency artefacts shipped end-to-end.** `identity_drift`
   (progressive face-region morph toward a per-base reference identity, Haar
   detector + alpha blend) and `background_drift` (progressive scene
   replacement with Detectron2 Mask R-CNN human silhouettes preserving the
   subject). 13 new unit tests, all passing.

2. **Validation set scaled 2 → 5 bases.** KZ8p6b1zJ9U, BrRLKMbBTYQ, mJog8DlRk_4
   promoted from reference-source-only to full bases (no new SR runs needed —
   the existing MGLD/UAV outputs already cover all 5). Refs cross-assigned,
   Detectron2 masks precomputed, 90 new artefact clips generated, full
   7-metric battery + LR-VCC v4 composite computed.

3. **The "reference-scene similarity" hypothesis from the last report is dead.**
   We curated CLIP-image-distant reference scenes (d > 0.25) for the
   7WHI background_drift case — including a deliberately-distant
   d=0.84 cartoon scene as the new reference — and re-evaluated.
   Result: composite Δ moved from +0.042 to +0.046. Indistinguishable.
   The inversion mechanism is not about the reference at all.

4. **Sub-metric breakdown revealed the real mechanism.** Per-sub-metric Δ
   analysis on every base shows sub-metric D (colour-histogram stability,
   reliability ≈ 1.0 everywhere) systematically *rises* under background_drift
   regardless of base or reference choice. Replacing a dynamic background
   with a progressively static reference image genuinely makes per-frame
   colour distributions more stable over time. D is measuring exactly what it
   was built to measure — but what it measures rewards rather than penalises
   convergence-type corruptions. The composite's softmax-reliability
   re-weighting then amplifies D's confident-but-perverse signal, manufacturing
   inversions even when 3 of 5 sub-metrics individually catch the artefact.

5. **Variance gate built then parked.** Implemented a clip-score-dispersion
   reliability gate for sub-metric I to abstain on flappy single-face content
   (the 7WHI pathology), but calibration on 2-base data showed only a
   0.003-wide separation between healthy and pathological dispersion
   populations. Per supervisor's spirit (honesty over force-fitting), the
   gate ships off-by-default (`dispersion_threshold=None`) until the
   8-base recalibration has data to support a defensible threshold.

---

## v4 verdict matrix — 6 artefacts × 5 bases

LR-VCC ΔLR-VCC from severity 0.02 → 0.40. Verdicts:
PASS ≤ −0.05, WEAK ≤ −0.02, FLAT < +0.02, INVERTED ≥ +0.02.

| artefact            | hhsz       | 7WHI       | KZ         | BrRLK      | mJog       |
|---------------------|:----------:|:----------:|:----------:|:----------:|:----------:|
| chunk_boundary       | −0.236 PASS | −0.162 PASS | −0.102 PASS | −0.008 flat¹ | −0.069 PASS |
| color_drift          | −0.039 WEAK | −0.111 PASS | −0.086 PASS | −0.063 PASS | −0.034 WEAK |
| flicker              | +0.005 flat | −0.011 flat | −0.020 WEAK | +0.067 INV  | −0.005 flat |
| identity_degradation | −0.070 PASS | +0.043 INV  | −0.032 WEAK | +0.001 flat² | +0.020 INV  |
| identity_drift       | −0.034 WEAK | −0.002 flat | −0.017 flat | +0.001 flat² | −0.002 flat |
| background_drift     | −0.276 PASS | +0.046 INV  | −0.002 flat | +0.127 INV  | +0.064 INV  |

¹ Cartoon scene cuts dominate the source signal — chunk_boundary noise floor matches the artefact magnitude.
² Haar can't detect cartoon faces — verified on the source frames; the artefact was effectively a no-op (2–3 frames in 50 with > codec noise). These two cells should be re-classified N/A in the final thesis presentation.

**Clean (PASS or WEAK): 14/30 cells** (≈ 16/30 after N/A re-classification).

Two families are solid across bases:
- **color_drift: 5/5 clean** — the colour-stability + colour-slope sub-metrics catch it on every type of content
- **chunk_boundary: 4/5 clean** — tOF catches the abrupt offset jumps; the cartoon cell is an honest content-source noise floor, not a metric failure

The four other families have the convergence-rewards-stability problem to varying degrees, plus content-domain effects (Haar / Detectron2 on animation).

---

## Sub-metric breakdown — why background_drift inverts

Score Δ (severity 0.02 → 0.40), and reliability shift, per sub-metric on background_drift:

| base  | A (CLIP-IQA) | T (tOF)       | I (Identity)   | D (colour hist) | E (slope)        | Composite |
|-------|:------------:|:-------------:|:--------------:|:---------------:|:----------------:|:---------:|
| hhsz  | −0.025      | +0.004        | +0.012         | **+0.046** (r 1.0) | **−0.145** (r 0.85→1.00) | **−0.276 PASS** |
| KZ    | **−0.179** ✓ | −0.043 ✓     | **−0.147** ✓  | **+0.045** (r 1.0) | −0.222 (r only 0.25→0.31) | −0.002 flat |
| BrRLK | −0.095 ✓    | −0.031 ✓     | **−0.264** ✓  | +0.021 (r 1.0)  | +0.036 (E saturated at floor on the clean video) | **+0.127 INV** |
| mJog  | −0.073 ✓    | −0.045 ✓     | +0.055        | **+0.066** (r 1.0) | −0.017 flat       | +0.064 INV |
| 7WHI  | −0.026      | −0.034        | +0.022         | **+0.061** (r 1.0) | +0.030 (r 0.7→0.7) | +0.046 INV |

Reading: on every base, sub-metric D's score moves the wrong direction
with full reliability ≈ 1.0, so it gets a large softmax weight. On hhsz the
slope sub-metric E happens to fire with high R²-reliability and dominates,
producing the strongest single result we've collected (Δ −0.276). On every
other base E is either gated down by its R² check or saturated at the noise
floor, leaving D's perverse signal in control of the composite. This is
structural, not data-dependent.

---

## What was done — week-by-week

### Week of June 5–11

- `identity_drift` and `background_drift` generators implemented, 13 unit tests
- Detectron2 mask precomputation, Haar-cascade reference-face extraction
- Three parallel server tmux pipelines (generation + evaluation) successfully orchestrated
- First 2-base composites computed, the "single-face base has three failure modes" pattern emerged
- Disk pressure event resolved (37 GB cleared from pip/vbench/dreamsim caches)
- vbench env setuptools-81 incompatibility worked around (`pip install 'setuptools<81'`)
- Onboarding docs updated with server workflow patterns

### Week of June 12–18

- **Implementation plan written** for the benchmark-completion path to the
  July 1 freeze, broken into 15 tasks (`docs/superpowers/plans/2026-06-11-benchmark-completion.md`)
- **Three code tasks shipped, each TDD with two-stage subagent review:**
  - Clip-score-dispersion reliability gate for sub-metric I (off by default)
  - Calibration script for the gate, parameterised for future base sets
  - CLIP-distance reference-scene selector for background_drift
- **Disk pruned** 33 GB → 76 GB free (raw frame dirs, mp4s are canonical)
- **3 new bases promoted**, ref-faces extracted (largest face per source up to 336 px), curated backgrounds picked (CLIP-distance 0.75–0.88, well above the τ = 0.25 floor)
- **90 new artefact clips generated** in tmux (8-hour generation, 30 GB output)
- **Full 7-metric battery** on 90 new clips: split GPU 0 / GPU 7, two days of overnight compute, all five evaluation stages (CLIP-IQA, tOF/tLP, colour-hist, colour-slope, Identity slow-fast)
- **Verdict-matrix builder** implemented (`scripts/lr_vcc/build_verdict_matrix.py`) — TDD, reviewed, approved
- **v4 composites computed** for all 6 × 5 = 30 conditions, the matrix above is the result
- **Mechanism diagnosis** (D rewards convergence) — independently confirmed by supervisor
- **Adversarial check** of the BrRLK face-artefact "flat" cells: only 2–3 of 50 sampled frames had pixel differences above codec noise — confirms those should be N/A, not FAIL
- **Reference-similarity hypothesis falsified** via the 7WHI background_drift re-run with curated d = 0.84 reference (Δ moved 0.004, indistinguishable)

---

## Server-infrastructure additions

- New runner scripts: `run_b4_prep.sh`, `run_b6_eval.sh`, `run_b6_7whibg.sh`
- Tmux chaining pattern: `until [ -f /tmp/<prev>.done ]; do sleep 60; done` —
  reliable for chaining stages without baby-sitting; survives ssh disconnects
- Identity-stage timing on long videos updated: ~30 min per video on these
  promoted bases (5000 frames each) vs the earlier hhsz reference (2412
  frames). Plan now uses this revised estimate for Week 2 model evaluation.

---

## Code delivered this period

| File | Purpose | Commit |
|------|---------|--------|
| `scripts/lr_vcc/identity.py` (modified) | Three-factor reliability with off-by-default dispersion gate | d09a4d7, bf8f44d |
| `scripts/lr_vcc/calibrate_identity_gate.py` (new) | Per-artefact dispersion stats + suggested threshold; argparse for future base sets | b4fb57a, 6361d98 |
| `scripts/synthetic_artefacts/select_reference_scene.py` (new) | CLIP-image-distance reference picker for background_drift | 23896df, 348d1f8 |
| `scripts/synthetic_artefacts/generate_all.py` (modified) | 5-base roster + per-base ref/mask dicts | bcab369 |
| `scripts/lr_vcc/build_verdict_matrix.py` (new) | Artefact × base verdict tabulator with PASS/WEAK/FLAT/INVERTED thresholds | 75894f7 |
| `docs/superpowers/specs/2026-06-11-benchmark-completion-design.md` (new) | Design for the 3-week experiment plan | 55178d8 |
| `docs/superpowers/plans/2026-06-11-benchmark-completion.md` (new) | 15-task TDD-based implementation plan | 422403a |
| `reports/figures/verdict_matrix_v4.md` (new) | Generated 6 × 5 verdict matrix | (this commit) |

Test suite: **76 passing**, 0 failing. All new code is TDD with explicit subagent code review at two stages (spec compliance + code quality) before merge.

---

## Next period (June 19 – July 2)

The plan needs amendment. The "scale to 8 bases / real-model discrimination" Week-2/3 path stands, but the headline finding makes a partial sub-metric redesign urgent before the model-comparison study would be meaningful — using the current composite to rank SR models would replicate the convergence-rewards-stability bias against any model that produces over-smooth output. Concrete amendments:

1. **Pause Week 2 model-discrimination work by 3–5 days** to brainstorm + experiment with sub-metric D / E redesigns (anchor-window stability, perceptual-trajectory replacement). Decision criterion: a redesign that flips at least 3 of the 5 currently-inverted background_drift cells without regressing the 4 PASS cells.
2. **Continue Week 1 finishes in parallel** (classmate LR-input package, RealESRGAN frame-wise anchor) since they have long external lead time and don't depend on the metric choice.
3. **Drop the "8 bases" stretch goal** — 5 bases is enough to surface the mechanism we just found; more bases would only confirm it. Re-scope the saved compute into the metric-redesign experiments.

Open technical questions:
1. Anchor-window vs. perceptual-trajectory for D: which is more robust to legitimate scene cuts vs scene replacement?
2. Is the slow-fast pooling pathology a separable issue (fixable with the parked dispersion gate at scale) or fundamentally entangled with the convergence story (single-face content lets fewer sub-metrics participate)?
3. What's the minimum number of confirmed-broken cells that justifies a metric version-bump in the thesis?
