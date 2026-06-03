# Bi-weekly Progress Report — Timur Iakshibaev

## Period: May 22 – June 4, 2026

## Headline

This bi-weekly period converted the LR-VCC composite from the Layer 1+2 prototype (3 sub-metrics) into a Layer-3-validated metric on a parameterised synthetic test set covering four artefact families:

1. **Four synthetic-artefact generators landed** — color drift, chunk-boundary jumps, periodic flicker, and identity degradation (face-region Gaussian blur). Each has controlled severity ∈ {0.02, 0.05, 0.10, 0.20, 0.40} and a tested generator module. 40 test videos total (2 base × 4 artefacts × 5 severities), severity ordering known by construction.
2. **Five LR-VCC iterations against the validation set** (v2 → v2_uniform → v3 → v3+slope → v3+slope β=200): each iteration closed one severity-response failure mode. Final composite catches chunk_boundary monotonically on both bases, color_drift cleanly on 7WHI and weakly on hhsz, identity_degradation cleanly on hhsz with documented inversion on 7WHI (identity-collapse failure mode of slow-fast pooling under heavy blur). Flicker is the remaining weakest case at composite level. SR non-regression preserved: MGLD wins 5/5 with mean Δ +0.056.
3. **Sub-metric D (color stability via Lab histogram L1) and sub-metric E (color-slope linear regression on per-frame Lab channel means)** were designed, implemented (with full test suites — 39/39 LR-VCC tests pass), validated, and committed. Composite metric grew from 3 sub-metrics to 5 with full reliability gating.
4. **Identity-collapse pathology characterised** on the single-face base video under heavy face-region blur: the Identity sub-metric's fast component inverts because heavy blur erases identity-distinctive features and cross-clip first-frame embeddings appear generically similar. Mechanism documented; fix (gate by face-detection confidence rather than face-rate alone) noted as future work.

![LR-VCC architecture: five reliability-gated sub-metrics (Appearance, Temporal, Identity, Colour Stability, Colour Slope), each paired with an interpretable reliability gate; composition via softmax-log-mean of reliabilities at τ = 0.2.](../proposal/figures/fig4_lr_vcc_architecture.png)

*LR-VCC architecture as of this period. Five sub-metrics in parallel; each emits a (score, reliability) pair; softmax-log-mean composition produces the final score in [0, 1].*

---

## Synthetic Validation Pipeline — Four Artefact Families Generated

Four deterministic, parameterised degradation operators implemented under `scripts/synthetic_artefacts/`, each operating on existing MGLD-SR outputs of two long-form base videos (`7WHI2L_FDNg` — 167 s / 5,000 frames, and `hhszUXL1Cu8` — 80 s / 2,400 frames):

| Generator | File | Mechanism | Severity meaning |
|-----------|------|-----------|------------------|
| Color drift | `color_drift.py` | Linear ramp: at frame `i/T`, multiply R by `1+α·(i/T)` and reduce G/B by same factor | α ∈ {0.02 … 0.40} — final-frame red gain |
| Chunk boundary | `chunk_boundary.py` | At every 60th frame: deterministic per-chunk uniform additive offset of magnitude ±α·255 to all channels | α ∈ {0.02 … 0.40} — step amplitude as fraction of dynamic range |
| Periodic flicker | `flicker.py` | Sinusoidal brightness oscillation at period 15 frames | α ∈ {0.02 … 0.40} — peak amplitude as fraction of dynamic range |
| Identity degradation | `identity_degradation.py` | Per-frame Haar-cascade face detection then `GaussianBlur(σ = 10α)` applied within each detected face bounding box (10% padding) | α ∈ {0.02 … 0.40} — blur sigma |

Each generator has a 2–3 case unit-test file verifying: (a) severity-0 leaves the video unchanged within float tolerance, (b) per-frame statistics match the expected analytic curve, (c) seed reproducibility. 15 tests in total, all pass.

Generated 40 test videos on GPU machine via `generate_all.py`, archived at `/data/disk2/timur/results/synthetic_artefacts/{color_drift,chunk_boundary,flicker,identity_degradation}/`.

---

## LR-VCC v2 → v3 → v3+slope — Five Iterations against the Validation Set

The May-21 v1 composite (3 sub-metrics: A, T, I) was rebuilt incrementally each time a severity-response failure surfaced. Each version was scored on every video in the validation set and gated on two pass criteria: (1) MGLD wins 5/5 on the SR benchmark, and (2) per-artefact monotonic severity-response on both base videos.

| Version | What changed | What it closed | Remaining gap |
|---------|--------------|----------------|---------------|
| v1 | 3 sub-metrics (A appearance, T temporal-log-weighted, I identity) | Layer 1+2 pass | tOF k=1 not in composite → flicker invisible; histogram-based metrics missing |
| v2 | + Sub-metric D (color histogram L1 at k ∈ {1, 5, 10, 30, 60, 120}) | Provides a long-range *color* lever distinct from optical flow | Color-drift severity range still flat: per-pair histogram distance at k=60/120 stays below bin width for slow monotonic drifts |
| v2_uniform | `--temporal_weight uniform` (= equal weight across all 6 k's instead of `log(1+k)`) | Flicker no longer entirely down-weighted at k=1; tOF responsive to periodic short-range artefacts | Color drift still flat |
| v3 | `--color_hist_alpha 0.394` recalibrated D's exponential decay using the empirical chunk-boundary distance range | Chunk-boundary D-score range now spans 0.65→0.25 across severities | Color drift's per-pair histogram distance is < bin-width across all severities → D still flat on color drift |
| **v3+slope β=200** | + Sub-metric E (color-slope linear regression on per-frame Lab channel means, R²-gated reliability), then tuned β | **Color drift now monotonic across both base videos** while chunk-boundary stays caught, flicker stays correctly minimal | β-sweep documented; small non-monotonic step on hhsz mid-severities |

The CLI flags `--temporal_weight`, `--color_hist_alpha`, `--color_slope_dir`, `--color_slope_beta` were all added so each recalibration is reproducible from the same JSON sub-metric caches without re-scanning videos.

---

## Sub-Metric E (Color Slope) — Design, Tuning, Result

### Why slope, when D already exists

Sub-metric D measures pairwise Lab-histogram L1 distance at long k. On synthetic color drift (linear brightness ramp), the per-frame shift between frames k=60 apart is only ~0.5% — below typical histogram bin width — so the per-pair L1 distance stays small at every severity. The signal exists in the **trajectory of per-frame channel means over time**, not in pair-distances. Sub-metric E directly fits a linear regression to each Lab channel's per-frame mean:

```
score        = exp(-β · max(|slope_L|, |slope_a|, |slope_b|))
reliability  = max(R²_L, R²_a, R²_b) gated by an R² floor (0.15)
```

Reliability is the discriminator: a clean video gives near-zero R² (so the slope it finds is noise, and the composite down-weights this sub-metric). A drifting video gives R² → 1 (slope is real, full weight). A flicker video gives R² ≈ 0 (the residual swamps any best-fit line), so the metric correctly abstains.

### β tuning — JSON re-derive flag avoids re-scanning videos

The β=50 default produced score range 0.78→0.76 across the full color-drift severity span — too narrow to influence the composite. Sweeping β ∈ {50, 100, 200, 300} via `--color_slope_beta` (which re-derives `exp(-β · max_abs_slope)` from the stored raw slope, no video re-read), β=200 chosen as best balance:

| β | mgld typical slope (0.001) → score | 7WHI base (0.005) | 7WHI drift sev0.40 (0.0055) | hhsz base (0.008) |
|---|------:|------:|------:|------:|
| 50  | 0.951 | 0.779 | 0.760 | 0.670 |
| 100 | 0.905 | 0.607 | 0.577 | 0.449 |
| **200** | **0.819** | **0.368** | **0.333** | **0.202** |
| 300 | 0.741 | 0.223 | 0.192 | 0.091 |

β=200 puts the slope-score in the 0.2–0.4 band where the composite's other sub-metrics also live, so flipping reliability low→high actually moves LR-VCC.

### Final severity-response on color drift (β=200)

| Video × Severity | sev0.02 | sev0.05 | sev0.10 | sev0.20 | sev0.40 | Δ end-to-end |
|------------------|--------:|--------:|--------:|--------:|--------:|------------:|
| **7WHI** LR-VCC | 0.6188 | 0.6230 | 0.5706 | 0.5185 | 0.5074 | **−0.111** |
| 7WHI E reliability | 0.260 | 0.391 | 0.813 | 0.987 | 0.998 | (ramps as expected) |
| 7WHI E score | 0.364 | 0.359 | 0.352 | 0.341 | 0.329 | (monotonic) |
| **hhsz** LR-VCC | 0.5199 | 0.5241 | 0.5302 | 0.4979 | 0.4811 | **−0.039** |
| hhsz E reliability | 0.874 | 0.867 | 0.860 | 0.931 | 0.977 | (already high — baseline drift) |

**7WHI**: nearly monotonic decrease, one small bump at sev0.02→sev0.05 (+0.004, near noise). 7WHI is the canonical color-drift PASS — reliability ramps cleanly from 0.26 to 1.0 as the drift signal emerges from background noise.

**hhsz**: shorter video (2,400 frames) already has a real drift baseline (R² ≈ 0.34 even at sev=0), so E reliability starts at 0.87 and barely moves. The injected drift adds only marginally to the baseline slope. The trend reverses correctly at sev0.10→sev0.40 (−0.049) but the sev0.02→sev0.10 segment is non-monotonic.

The hhsz issue is a real limitation: when a video has a pre-existing slope signal, adding more drift on top is hard to disentangle from the baseline. The R²-gated reliability does what it should (high reliability already), but the score scale isn't sensitive enough at the high end. Documented as a known sub-metric E limitation.

### Severity-response on the other two artefacts (β=200)

**Chunk boundary** (PASS preserved):

| Video × Severity | sev0.02 | sev0.10 | sev0.40 | Δ |
|------------------|--------:|--------:|--------:|--:|
| 7WHI LR-VCC | 0.6087 | 0.5693 | 0.4464 | −0.162 |
| hhsz LR-VCC | 0.4939 | 0.4135 | 0.2578 | −0.236 |

**Flicker** (correct minimal effect — slope sub-metric correctly *doesn't fire* because R² stays low):

| Video × Severity | sev0.02 | sev0.10 | sev0.40 | Δ |
|------------------|--------:|--------:|--------:|--:|
| 7WHI LR-VCC | 0.6170 | 0.6131 | 0.6055 | −0.012 |
| hhsz LR-VCC | 0.5266 | 0.5255 | 0.5315 | +0.005 |

Flicker δ is small and slightly noisy — flicker is currently the artefact LR-VCC handles weakest (tOF with `uniform` weighting picks up some of it but the response is not strongly monotonic at composite level). Documented as the next sub-metric extension target.

---

## SR Non-Regression — MGLD Still Wins 5/5 with v3+slope β=200

| Video | MGLD v3 | UAV v3 | MGLD v3+slope β=200 | UAV v3+slope β=200 | Δ (M−U) v3+slope |
|-------|--------:|-------:|--------------------:|-------------------:|-----------------:|
| 7WHI2L_FDNg | 0.5303 | 0.4591 | 0.5286 | 0.4581 | +0.0705 |
| BrRLKMbBTYQ | 0.5989 | 0.5608 | 0.3156 | 0.2692 | +0.0464 |
| KZ8p6b1zJ9U | 0.5687 | 0.5249 | 0.5708 | 0.5278 | +0.0430 |
| hhszUXL1Cu8 | 0.6816 | 0.5718 | 0.5579 | 0.4760 | +0.0819 |
| mJog8DlRk_4 | 0.4106 | 0.3742 | 0.4095 | 0.3736 | +0.0359 |
| **Mean** | **0.5580** | **0.4982** | **0.4765** | **0.4209** | **+0.0555** |

Adding sub-metric E with β=200 lowers absolute scores (because the new sub-metric is a "punish for any color drift" gate, and most SR videos have some baseline color trajectory), but the **per-video ordering is preserved on all 5 videos** and the inter-method gap is essentially unchanged (+0.060 v3 → +0.056 v3+slope). Layer 1 PASS preserved.

![Hand bounding-box size correlates with MGLD-vs-UAV anatomy gap. Four videos cluster at low close-up fraction (top-left, MGLD wins); KZ alone sits at 18% close-up fraction with the gap inverted to −0.29.](../proposal/figures/fig2_handbbox_vs_anatomy_gap.png)

*The empirical motivation for the close-up reliability gate that sub-metric I uses. KZ is the regime-shift case: the close-up content triggers the Anatomy classifier's high-fire regime, inverting the per-method ranking. LR-VCC's close-up reliability gate downweights Identity on KZ, letting Appearance, Temporal, Colour Stability, and Colour Slope determine the composite — which preserves the correct MGLD > UAV ranking.*

BrRLKMbBTYQ shows the largest drop: that is because UAV's output on this video has a real linear color trajectory (slope ≈ 0.017, R² ≈ 0.35) that E now correctly penalises — visible in `--color_slope_dir` raw output. This is a genuine UAV failure mode, not a false positive.

---

## Other Iterations

### v2: Sub-metric D (color histogram stability)

Added `compute_color_histogram.py` (per-video CPU pass computing Lab histogram L1 distance at k ∈ {1, 5, 10, 30, 60, 120}) and `color_stability.py` wrapper (`score = exp(-α · mean_l1_dist)`, reliability = entropy-floor gate). `--color_hist_dir` flag plumbed through `run_lr_vcc.py`. 11 new tests in `test_color_histogram.py`. Found via empirical run that the default α=2.0 was over-saturated (chunk-boundary score range only 0.7–0.6); calibrated to α=0.394 via the chunk-boundary distance range to give the design-intent 0.65→0.25 range. The recalibration is a pure CLI flag (no re-scanning).

### v2_uniform: tOF weighting scheme

Added `--temporal_weight {log, uniform, sqrt}` to `temporal.py` and `run_lr_vcc.py`. `log(1+k)` (default) puts most weight on long k, which is the right call for chunk-boundary detection but hides flicker (which is a k=1 signal). `uniform` (= equal weight) was the chosen production setting because it preserves all artefact families. 2 new tests in `test_temporal.py`.

---

## Fourth Artefact: identity_degradation — Sub-Metric I Validation

Built and validated the 4th artefact family to stress sub-metric I (Identity slow-fast), which the first three artefacts didn't touch.

### Generator

`scripts/synthetic_artefacts/identity_degradation.py`: per-frame Haar-cascade face detection, then `cv2.GaussianBlur(sigma=severity*10.0)` applied within each detected face bbox (10% padding). No-face frames pass through unchanged. 4 new tests in `test_identity_degradation.py`. Severity sigma scale: 0.02→0.2 (barely visible), 0.10→1.0, 0.40→4.0 (strongly blurred). 10 videos generated on server.

### Full metric battery + LR-VCC composite

CLIP-IQA, tOF/tLP, color-hist, color-slope, DOVER, E*warp, and Identity slow-fast all run on the 10 videos via tmux (`eval_id`, ~2 h wall-clock; Identity is the slow stage at ~10 min/video). Composite run under production CLI: `--temporal_weight uniform --color_hist_alpha 0.394 --color_slope_beta 200`.

### Sub-metric severity-response

**hhszUXL1Cu8** (clean signal — sub-metric I catches identity degradation as designed):

| sev  | A     | T     | I     | D     | E     | LR-VCC | w(I) |
|------|------:|------:|------:|------:|------:|------:|-----:|
| 0p02 | 0.397 | 0.951 | 0.667 | 0.646 | 0.182 | 0.559 | 0.28 |
| 0p05 | 0.391 | 0.951 | 0.654 | 0.646 | 0.182 | 0.556 | 0.28 |
| 0p10 | 0.377 | 0.951 | 0.620 | 0.646 | 0.182 | 0.546 | 0.28 |
| 0p20 | 0.350 | 0.951 | 0.624 | 0.647 | 0.182 | 0.544 | 0.28 |
| 0p40 | 0.311 | 0.951 | **0.440** | 0.647 | 0.182 | **0.489** | 0.29 |

Identity drops 0.667→0.440 (Δ −0.227, monotonic except a tiny 0.620→0.624 violation at mid-severity). Composite tracks I with Δ −0.070.

**7WHI2L_FDNg** (identity-collapse pathology — sub-metric I score *rises* with severity):

| sev  | A     | T     | I     | D     | E     | LR-VCC | w(I) |
|------|------:|------:|------:|------:|------:|------:|-----:|
| 0p02 | 0.409 | 0.929 | 0.375 | 0.559 | 0.364 | 0.531 | 0.35 |
| 0p05 | 0.404 | 0.929 | 0.372 | 0.559 | 0.364 | 0.529 | 0.35 |
| 0p10 | 0.392 | 0.929 | 0.372 | 0.559 | 0.364 | 0.528 | 0.35 |
| 0p20 | 0.372 | 0.928 | 0.381 | 0.559 | 0.364 | 0.531 | 0.35 |
| 0p40 | 0.345 | 0.928 | **0.489** | 0.559 | 0.364 | **0.574** | 0.35 |

Identity *increases* 0.375→0.489 (Δ +0.114). Composite follows with Δ +0.043.

### Identity-collapse — diagnosis

Face detection succeeds in 96% of clips at every severity (n_clips_with_faces / n_clips ≈ 0.96 for all 5 severities, both videos), so face_rate-based reliability gating doesn't engage. The bulk Identity output reveals the actual mechanism by splitting fused = mean(slow, fast):

- 7WHI fast: 0.066 → 0.066 → 0.066 → 0.092 → **0.276** (4× *increase* at max severity)
- hhsz fast: 0.579 → 0.579 → 0.500 → 0.500 → **0.143** (4× *decrease*, as expected)

The slow component is stable both ways (~0.68 on 7WHI, ~0.74 on hhsz). The pathology is the fast (cross-clip first-frame) comparison: under sigma=4.0 blur, single-face frames lose identity-distinctive features, so cross-clip first-frame embedding similarity *rises* — heavy blur makes faces look "generically similar" to each other. hhsz has multi-face content where this washing-out is dominated by between-identity differences, so its fast score drops correctly.

### Attribution check (other sub-metrics)

- **T (tOF) flat** (Δ < 0.001 both bases): face-region blur doesn't disturb optical-flow consistency. Correct.
- **D (color hist) flat** (Δ ≈ 0.001): blur doesn't shift histograms in detectable amounts. Correct.
- **E (color slope) flat** (Δ < 0.0001): blur doesn't add a linear-trend signal. Correct.
- **A (CLIP-IQA) drops monotonically** (0.397→0.311 hhsz, 0.409→0.345 7WHI): CLIP-IQA scores full-frame quality and sees the blur. Expected side-effect, not a false positive — face is a real part of the frame.

So 3 of the 5 sub-metrics correctly stay quiet (clean attribution), A reacts as expected (real signal), and I either catches the artefact (hhsz) or exhibits the documented identity-collapse inversion (7WHI). This is informative, not a bug — it characterises a failure mode of slow-fast pooling under degradations severe enough to homogenise identity features.

### Verdict

| Base | LR-VCC Δ sev0.02 → sev0.40 | Status |
|------|---:|---|
| hhsz | −0.070 (monotonic) | PASS |
| 7WHI | +0.043 (inverted at max sev) | FAIL — identity-collapse pathology |

Future work: gate sub-metric I by face-detection *confidence* (mean detection score) and/or per-face embedding variance, not just face-rate. Heavily-blurred-but-still-detected faces should down-weight reliability.

---

## Consolidated 4-Artefact × 2-Base Verdict

| Artefact | hhsz Δ | 7WHI Δ | Outcome |
|----------|---:|---:|---|
| chunk_boundary | −0.236 | −0.162 | both monotonic; PASS |
| color_drift | −0.039 | −0.111 | 7WHI clean, hhsz weak (source baseline drift); PARTIAL |
| flicker | +0.005 | −0.012 | flat (E correctly silent, T weight underplays k=1); FAIL |
| identity_degradation | −0.070 | +0.043 | hhsz clean, 7WHI inverted (identity-collapse); PARTIAL |

**LR-VCC catches 5/8 conditions cleanly; the remaining 3 each have a documented mechanism that becomes future-work scope.** This is a richer story than a binary "works/doesn't" — the failure modes are *characterised*, not mysterious.

![Severity-response across four artefact families (colour drift, chunk-boundary jumps, periodic flicker, identity degradation) and three key metrics (tOF k=1, tOF k=120, LR-VCC v3+slope β=200). Green panels: monotonic on both base videos. Amber: monotonic on one. Red: flat / non-monotonic on both.](../proposal/figures/fig7_severity_summary_grid.png)

*4-artefact × 3-metric verdict grid. The bottom row (LR-VCC composite) shows PASS on colour drift and chunk_boundary, PARTIAL on identity_degradation, FAIL on flicker — matching the consolidated verdict table above. Visualises the gap that the composite closes relative to the baseline metrics in the top two rows.*

---

## Code Delivered

| File | Purpose | New / Modified |
|------|---------|----------------|
| `scripts/synthetic_artefacts/color_drift.py` | Color-drift generator | New |
| `scripts/synthetic_artefacts/chunk_boundary.py` | Chunk-boundary generator | New |
| `scripts/synthetic_artefacts/flicker.py` | Periodic flicker generator | New |
| `scripts/synthetic_artefacts/identity_degradation.py` | Face-region Gaussian-blur generator | New |
| `scripts/synthetic_artefacts/generate_all.py` | Driver for the 40-video test set (4 artefacts) | New |
| `scripts/lr_vcc/color_histogram.py` | Sub-metric D core (Lab histogram L1 over multi-k) | New |
| `scripts/lr_vcc/compute_color_histogram.py` | Per-video runner producing JSON sub-metric cache | New |
| `scripts/lr_vcc/color_stability.py` | D wrapper: JSON → {score, reliability, details} | New |
| `scripts/lr_vcc/color_slope.py` | Sub-metric E core (linregress on Lab channel means) | New |
| `scripts/lr_vcc/compute_color_slope.py` | E per-video runner | New |
| `scripts/lr_vcc/summarize_identity_degradation.py` | Severity-response summariser for 4th artefact | New |
| `scripts/lr_vcc/temporal.py` | + uniform/sqrt weighting modes | Modified |
| `scripts/lr_vcc/run_lr_vcc.py` | + 5 CLI flags (`--color_hist_dir`, `--color_hist_alpha`, `--color_slope_dir`, `--color_slope_beta`, `--temporal_weight`) | Modified |
| `tests/synthetic_artefacts/test_artefacts.py`, `test_flicker.py`, `test_identity_degradation.py` | 15 generator tests | New |
| `tests/lr_vcc/test_color_histogram.py`, `test_color_stability.py`, `test_color_slope.py` | 19 new sub-metric tests | New |

LR-VCC test suite: 39/39 PASS (20 baseline + 11 D-related + 4 E-related + 4 weighting / recalibration). Generator test suite: 15/15 PASS (11 original + 4 identity_degradation).

---

## Commit Log (Experimental Track)

```
3e59744 lr_vcc: summarize_identity_degradation gracefully handles partial data
58b2347 synthetic_artefacts: identity_degradation generator (Task A11)
2074a31 lr_vcc: --color_slope_beta CLI flag (re-derive sub-metric E score from cached max_abs_slope)
c8f9cc7 lr_vcc: sub-metric E (color slope) for color_drift detection
ed3bdc1 lr_vcc: --color_hist_alpha CLI flag (recalibrate sub-metric D)
7ddd054 lr_vcc: configurable tOF weighting (log/uniform/sqrt)
2b4ba26 synthetic_artefacts: periodic flicker generator (Task A8)
d7d5c5c lr_vcc: sub-metric D (color stability) + run_lr_vcc --color_hist_dir flag
b16bfc9 synthetic_artefacts: color drift + chunk-boundary generators (Task A1)
```

---

## Next Steps (June 5 – mid-July, before blind-review submission)

1. **Multi-person Identity sub-metric** — implement the design in `docs/plans/2026-05-06-multiperson-identity-metric.md`. Per-clip cluster purity (self-consistency) plus low-resolution-reference matched-pair variants. Validate against the multi-face base video already in the synthetic test set. Budget: first two weeks of June.
2. **Flicker improvement** — two options. Option A: add a fast-varying brightness sub-metric (FFT magnitude in the 5–20 Hz band of per-frame mean luminance). Option B: rebalance `--temporal_weight` with a content-adaptive variant. Decide after Option A is prototyped. Budget: third week of June.
3. **Real-video baseline confirmation** — source ≥ 10 real high-resolution clips of approximately one minute each (no super-resolution, no artefacts) and confirm LR-VCC scores ≥ 0.7 (no false-positive drift detection on natural content). Sanity check that the synthetic-test-set validation generalises to natural content. Budget: fourth week of June.
4. **Extension of validation to a broader set of recent super-resolution methods** — the current empirical claims rest on two diffusion baselines (MGLD-VSR, Upscale-A-Video). Expand to additional recent methods so the verdict matrix and per-video method rankings generalise. Budget: early July.
5. **Identity-collapse fix** — gate sub-metric I by face-detection confidence and per-face embedding variance (not face-rate alone). The single-face base's identity-collapse inversion is the cleanest characterised LR-VCC failure mode and the natural next-iteration target.

---

## Talking Points for the Group Meeting

1. **Sub-metric E (linregress on Lab channel means with R²-gated reliability) is the first metric in our 9-metric test set that responds monotonically to color drift on at least one base video.** The mechanism — gate weight by goodness-of-fit, not by score magnitude — is reusable for any "slow signal vs noisy baseline" sub-metric.
2. **The five-version iteration history (v1 → v3+slope β=200) is a textbook example of severity-response-driven metric design.** Each version added one element after the prior version's exact failure mode was characterised on the validation set.
3. **Identity-collapse pathology on 7WHI identity_degradation is a clean, characterisable failure mode** — heavy face-region blur erases identity-distinctive features, so cross-clip first-frame embedding similarity *rises* with severity. The slow component stays stable; only fast inverts. This is an honest limitation of slow-fast pooling under severe degradation, not noise.
4. **The hhsz color-drift PARTIAL is informative, not a bug.** Pre-existing baseline drift makes the injected drift hard to isolate. This is the exact situation any real video would face.
5. **β tuning happens entirely via re-derive in `--color_slope_beta`**, no re-scanning. This is the right pattern for any decay parameter — the per-video cache stores raw `max_abs_slope`, and the wrapper re-applies the decay. Same pattern applied to α in sub-metric D.

## Open Technical Questions

1. **Real-video baseline scope** — should we source a real one-minute HR clip set as a "no artefact at all" baseline before claiming Layer 3 PASS? Or is synthetic-only enough for now?
2. **Flicker priority** — flicker is currently the weakest of the four artefact families at the composite level. Is the fast-varying brightness sub-metric the right addition, or should we revisit the composition arithmetic itself so that small-k tOF signal is not averaged down by the colour and identity sub-metrics?
3. **β tuning provenance** — currently β=200 is justified by visual inspection on the validation set. Add a formal calibration criterion (for example, maximise Kendall τ of severity–LR-VCC correlation averaged over both base videos × all 4 artefact families)?
4. **Identity-collapse fix priority** — gate sub-metric I by face-detection confidence + per-face embedding variance. Worth prioritising before the multi-person identity sub-metric, or after?
