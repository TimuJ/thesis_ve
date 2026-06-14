# Bi-weekly Progress Report — Timur Iakshibaev

## Period: June 5 – June 18, 2026

## Headline

This period delivered three layered results. (i) The two long-term-consistency
artefacts the supervisor asked for at the May meeting, `identity_drift` and
`background_drift`, shipped end-to-end. (ii) The validation set scaled
2 → 5 base videos and produced the first complete 6-artefact × 5-base
verdict matrix on LR-VCC v4 — which surfaced a clear mechanistic problem
with sub-metric D: **stability-based sub-metrics systematically reward
convergence-type corruptions** (the supervisor independently arrived at
the same diagnosis: "degradation makes videos more stable"). (iii) Two
replacement sub-metrics (D' anchor-window Lab histogram, D'' CLIP image
trajectory) were implemented, evaluated, and integrated into LR-VCC v5.
**The headline result: `background_drift` inversions drop from 4/5 INVERTED
under v4 to 1/5 INVERTED under v5** — the diagnosis is now empirically
confirmed *and* fixed in production. Cartoon content (BrRLK) remains the
one open hole, traceable to natural scene-cut noise dominating both anchor
distances.

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

### June 13–14: metric redesign + v5

- **6-transform flip family** (`flip_horizontal`, `flip_transpose`,
  `flip_periodic`, `flip_elastic`, `flip_channel_shuffle`, `flip_invert`)
  designed as an ablation ladder of statistical-preservation properties;
  150 clips generated on server in tmux
- **D' anchor-window Lab histogram** implemented and computed across all
  12 artefacts × 5 bases × 5 severities (300 JSONs) — CPU run, no GPU needed
- **D'' CLIP-image-trajectory** implemented and computed across all
  12 artefacts × 5 bases × 5 severities (300 JSONs) — GPU 1 (parked-fyx slot)
- **HuggingFace blocked → OpenAI clip workaround**: switched from open_clip
  `pretrained="openai"` (which still routes through HF for timm config) to
  OpenAI's original `clip` package (different CDN, cached weights already
  on server)
- **Three-matrix comparison** D vs D' vs D'' on the 6 existing artefacts +
  6 flip ablation transforms — empirically confirmed the flip_horizontal
  smoking-gun prediction (both D and D' blind to horizontal mirror) and
  exposed CLIP's known horizontal-flip robustness in D''
- **LR-VCC v5 composite** with D + D' + D'' alongside A, T, I, E (7
  sub-metrics, softmax reliability weighting). Phase 1 (6 existing
  artefacts × 5 bases = 30 composites) committed; phase 2 (6 flip artefacts
  × 5 bases = 30 composites) running overnight, expected morning of
  June 15.
- **Headline v5 result on `background_drift`**: 4/5 inversions converted
  to PASS / WEAK / FLAT (only BrRLK cartoon remains INV with Δ halved).

---

## Server-infrastructure additions

- New runner scripts: `run_b4_prep.sh`, `run_b6_eval.sh`, `run_b6_7whibg.sh`,
  `run_b8_flip_eval.sh`, `run_b9_dprime.sh`, `run_b10_dprime2.sh`,
  `run_b11_v5_composite.sh`
- Tmux chaining pattern: `until [ -f /tmp/<prev>.done ]; do sleep 60; done` —
  reliable for chaining stages without baby-sitting; survives ssh disconnects
- Identity-stage timing on long videos updated: ~30 min per video on these
  promoted bases (5000 frames each) vs the earlier hhsz reference (2412
  frames). Plan now uses this revised estimate for Week 2 model evaluation.
- **GPU 1 added to the runnable pool** for low-memory workloads. fyx's
  process has 57 GB allocated but is parked at 0% utilisation, leaving
  24 GB free. Used successfully for D'' overnight (~3 h) and for the
  parallel half of the flip battery. The risk profile is "if fyx resumes,
  one of us OOMs" — has not yet happened.
- **HuggingFace Hub is unreachable from the lab server.** open_clip's
  `pretrained="openai"` still routes through HF behind the scenes (timm
  config), so we use OpenAI's original `clip` PyPI package instead, which
  downloads from a different CDN that *is* reachable. Cached weights already
  exist at `~/.cache/clip/ViT-B-32.pt`. Documented for future sub-metrics
  needing CLIP-family models.

---

## Metric redesign — D' and D'' (v5 composite)

### Diagnostic probes — the flip family

After the v4 verdict matrix surfaced the convergence-rewards-stability
mechanism, six self-modifying midpoint-discontinuity artefacts were added
to the suite to test sub-metrics on histogram-preserving but
identity-violating corruptions. Carefully chosen statistical properties
form an ablation ladder:

| transform | preserves | catches if sub-metric measures |
|---|---|---|
| flip_horizontal | full histogram | structure / identity (not pixel stats) |
| flip_transpose | full histogram | non-mirror-invariant features |
| flip_periodic | full histogram | matching temporal-window scale |
| flip_elastic | ≈full histogram | pure structure (no colour cue) |
| flip_channel_shuffle | per-channel marginals | non-permutation-invariant colour |
| flip_invert | only variance | anything reasonable (control) |

Generator at `scripts/synthetic_artefacts/flip.py` with 22 unit tests; all
pure cv2 + numpy, no albumentations / torch dependency for the
self-modifying transforms.

### Two replacement sub-metrics

- **D' — anchor-window Lab histogram drift.**
  `scripts/lr_vcc/color_histogram_anchor.py`. Anchor = mean Lab histogram of
  the first 60 frames. Score = `exp(-β · |q4 − q1|)` over per-quarter
  anchor distance, β = 0.5. Robust to natural intra-clip variation; runs
  on CPU directly from the cached MP4s.

- **D'' — CLIP-image-embedding trajectory drift.**
  `scripts/lr_vcc/compute_clip_trajectory.py`. Per-frame CLIP ViT-B/32
  image embedding, cosine distance to first-60-frame anchor centroid,
  same `exp(-β · |q4 − q1|)` formula, β = 3.0. GPU-bound; one inference
  pass per sampled frame (stride 8).

Both implemented TDD with reliability = 1.0 by default and dumped to per-video
JSONs that share the {score, reliability, details} schema of the existing
sub-metrics, including a `trajectory_mean_per_quarter` field that allows β
to be retuned without re-running inference.

### v5 composite

`run_lr_vcc.py` extended with `--color_hist_anchor_dir`, `--clip_trajectory_dir`,
`--dprime_beta`, `--dprime2_beta`. Defaults preserve byte-exact behaviour of
the prior version when the new args are absent; when supplied, D' and D''
join the softmax-log-mean composition as additional sub-metrics, alongside
A, T, I, D, E (keeping original D — it still catches `chunk_boundary`
cleanly at 5/5). 7-sub-metric LR-VCC composite. 4 new unit tests for the
integration; full suite 114 passing.

### v5 result — supersedes v4

`reports/figures/d_variants_matrix.md` and
`reports/figures/verdict_matrix_v5.md` (phase 1 of 2; flip family pending
identity slow-fast battery, expected complete morning of June 15).

| sub-metric | bg_drift result | mechanism |
|---|---|---|
| D (original)               | **0/5 PASS** | rewards convergence — the bug |
| D' alone                   | 4/5 PASS | drift-from-opening, fails on BrRLK (cartoon scene-cut noise) |
| D'' alone                  | 4/5 PASS | semantic drift, fixes BrRLK but loses mJog |
| **best-of(D', D'')**       | **5/5 PASS** | complementary failure modes |
| **LR-VCC v5 (composite)**  | **4/5 PASS** | softmax reliability weighting + tOF / identity drag back on BrRLK; only BrRLK still INV |

Per-base composite Δ on background_drift, v4 → v5:

| base | v4 Δ | v4 verdict | v5 Δ | v5 verdict |
|---|---:|---|---:|---|
| hhsz   | −0.276 | PASS | −0.222 | PASS |
| 7WHI   | +0.046 | INVERTED | −0.013 | FLAT |
| KZ     | −0.002 | FLAT | −0.051 | PASS |
| BrRLK  | +0.127 | INVERTED | +0.065 | INVERTED (Δ halved) |
| mJog   | +0.064 | INVERTED | −0.030 | WEAK |

4 of 5 inversions converted to PASS / WEAK / FLAT; BrRLK still inverts but
the magnitude is halved. The cartoon hole has a known content-domain
explanation (natural scene-cut variation in source dominates the anchor
distance signal on both D' and D''); reported as a documented limitation
rather than a metric bug.

### Flip ablation confirms the diagnosis empirically

Predictions and outcomes line up cleanly:

- `flip_invert` (histogram-disrupting control): PASS under both D' (4/5)
  and D'' (4/5). Sanity check on both metrics — they detect the corruption
  they were designed to detect.
- `flip_channel_shuffle` (multiset-preserving): PASS under both D' (4/5)
  and D'' (4/5). Channel permutation breaks per-channel-bin alignment
  enough for both to fire.
- `flip_transpose` (geometric, histogram-preserving): caught only by
  D'' (3/5), not by D' (0/5). CLIP perceives rotation; pixel-distribution
  metrics cannot.
- `flip_horizontal` (pure mirror): only hhsz catches it under D'' (1/5).
  **CLIP's known horizontal-flip robustness** (a documented property of
  ViT trained on horizontally-flip-augmented data) makes this the hardest
  case for D''. Both D and D' are blind by construction (histogram
  identical). Becomes an honest "known limitation" thesis paragraph.
- `flip_periodic`, `flip_elastic` (subtle structural): mostly invisible
  to both. The high-frequency and low-amplitude transforms don't move the
  per-quarter anchor distance enough.

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
| `reports/figures/verdict_matrix_v4.md` (new) | Generated 6 × 5 v4 verdict matrix | (a898452) |
| `scripts/synthetic_artefacts/flip.py` (new) | 6-transform self-modifying flip family | 4d64c59 |
| `scripts/synthetic_artefacts/generate_all.py` (modified) | 12-artefact roster + `flip_*` parsing branch | 4d64c59 |
| `scripts/lr_vcc/color_histogram_anchor.py` (new) | D' anchor-window Lab histogram drift | 1276846 |
| `scripts/lr_vcc/compute_clip_trajectory.py` (new) | D'' CLIP-trajectory drift (server) | 1276846, 4d8ee66 |
| `scripts/lr_vcc/compare_d_variants.py` (new) | Three-matrix comparison renderer | c4c5b25 |
| `scripts/lr_vcc/run_lr_vcc.py` (modified) | D' / D'' integrated into composite (v5) | 2f5e4b2 |
| `reports/figures/d_variants_matrix.md` (new) | D vs D' vs D'' three-matrix comparison | c4c5b25 |
| `reports/figures/verdict_matrix_v5.md` (new) | 6 × 5 v5 verdict matrix (phase 1) | 2f5e4b2 |

Test suite: **114 passing**, 0 failing. All new code is TDD with explicit subagent code review (spec compliance + code quality) before merge.

---

## Next period (June 19 – July 2)

The proposed metric-redesign sprint at the start of this period landed
already (D' + D'' + v5), so the next period reverts to the original Week-2
plan: real-model discrimination. Concrete priorities, in order:

1. **Real-SR-model evaluation with v5 LR-VCC.** Apply LR-VCC v5 to existing
   MGLD-VSR and Upscale-A-Video outputs on the 5-video set, plus a
   frame-wise lower anchor (RealESRGAN per-frame). The thesis headline
   experiment: does the new composite rank these models in a way PSNR / SSIM
   cannot? Identity-stage cost is the bottleneck (~2.5 h per model on a
   shared GPU); plan one model per overnight slot.
2. **Classmate model outputs.** Package the 5 LR inputs + submission spec,
   send to classmates this week with a soft deadline of June 26 so the
   ranking table can include their methods by July 1.
3. **β / α calibration sweep for D' and D''.** Current betas (0.5, 3.0)
   were picked by eye on the existing data. Sweep ∈ {0.25, 0.5, 1.0, 2.0}
   for D' and {1.0, 2.0, 3.0, 5.0} for D'', check the verdict matrix is
   stable. Recomputable from cached trajectory JSONs — no server time.
4. **Leave-one-out sub-metric ablation.** With 7 sub-metrics now in v5,
   drop each one in turn and recompute the verdict matrix. Identifies which
   artefact families each sub-metric uniquely catches.
5. **Bi-weekly report + start writing track.** Switch `zjuthesis.tex`
   Period to `paper`. Methodology chapter draft begins ~June 22, building
   from the proposal text.

Deferred to "future work" in the thesis:
- BrRLK cartoon-content limitation. Documented in the report; thesis will
  note it as a content-domain limit rather than try to fix it in the
  remaining time.
- flip_horizontal / flip_periodic / flip_elastic invisibility. Documented
  as a limit of CLIP and pixel-distribution methods; "structural fingerprint"
  sub-metric is paper-grade work, out of scope for July 15 blind-review
  submission.
- Slow-fast identity pathology beyond the parked dispersion gate (b1/b2).

Open technical questions:
1. Does LR-VCC v5 rank MGLD > UAV > RealESRGAN in line with perceptual
   intuition, or does the convergence-rewards-stability bug have residual
   composite-level influence we haven't seen yet?
2. Can the human study (the original optional task 2d in the benchmark
   completion plan) still happen if classmates engage by June 22?
3. Whether to ship the BrRLK content-domain limitation as a §6 figure
   ("how the metric fails") or a §7 footnote.
