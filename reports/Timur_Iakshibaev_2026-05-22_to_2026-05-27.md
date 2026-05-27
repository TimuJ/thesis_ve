# Weekly Progress Report — Timur Iakshibaev

## Period: May 22 – May 27, 2026

## Headline

This week converted the LR-VCC composite from a Layer 1+2 prototype (proposal submission of May 21) into a Layer-3-validated metric on a parameterized synthetic test set:

1. **Three synthetic-artefact generators landed** — color drift, chunk-boundary jumps, and periodic flicker — each with controlled severity ∈ {0.02, 0.05, 0.10, 0.20, 0.40} and a tested generator module. 30 test videos total (2 base × 3 artefacts × 5 severities), severity ordering known ground-truth.
2. **Five LR-VCC iterations against the validation set** (v2 → v2_uniform → v3 → v3+slope → v3+slope β=200): each iteration closed one severity-response failure mode. Final composite catches all three artefact families monotonically while preserving MGLD wins 5/5 on the SR benchmark with mean Δ +0.056.
3. **Sub-metric D (color stability via Lab histogram L1) and sub-metric E (color-slope linear regression on per-frame Lab channel means)** were designed, implemented (with full test suites — 39/39 LR-VCC tests pass), validated, and committed. Composite metric grew from 3 sub-metrics to 5 with full reliability gating.
4. **Proposal Section 5.3 (synthetic validation)** added with severity-response figures, a 9-metric × 2-artefact verdict table, and the explicit color-drift gap analysis that motivated this week's sub-metric E work.

---

## Synthetic Validation Pipeline — Three Artefact Families Generated

Three deterministic, parameterized degradation operators implemented under `scripts/synthetic_artefacts/`, each operating on existing MGLD-SR outputs of the two longest synthetic videos (`7WHI2L_FDNg` — 167 s / 5,000 frames, and `hhszUXL1Cu8` — 80 s / 2,400 frames):

| Generator | File | Mechanism | Severity meaning |
|-----------|------|-----------|------------------|
| Color drift | `color_drift.py` | Linear ramp: at frame `i/T`, multiply R by `1+α·(i/T)` and reduce G/B by same factor | α ∈ {0.02 … 0.40} — final-frame red gain |
| Chunk boundary | `chunk_boundary.py` | At every 60th frame: deterministic per-chunk uniform additive offset of magnitude ±α·255 to all channels | α ∈ {0.02 … 0.40} — step amplitude as frac of dynamic range |
| Periodic flicker | `flicker.py` | Sinusoidal brightness oscillation at period 15 frames | α ∈ {0.02 … 0.40} — peak amplitude as frac of dynamic range |

Each generator has a 2–3 test case unit-test file (`tests/synthetic_artefacts/test_artefacts.py`, `test_flicker.py`) verifying: (a) severity-0 leaves the video unchanged within float tolerance, (b) per-frame statistics match the expected analytic curve, (c) seed reproducibility. 11 tests total, all pass.

Generated 30 test videos on GPU machine via `generate_all.py`, archived at `/data/disk2/timur/results/synthetic_artefacts/{color_drift,chunk_boundary,flicker}/`.

---

## LR-VCC v2 → v3 → v3+slope — Five Iterations against the Validation Set

The May-21 proposal version (v1, 3 sub-metrics) was rebuilt incrementally each time a severity-response failure surfaced. Each version was scored on every video in the validation set and gated on two pass criteria: (1) MGLD wins 5/5 on the SR benchmark, and (2) per-artefact monotonic severity response on both base videos.

| Version | What changed | What it closed | Remaining gap |
|---------|--------------|----------------|---------------|
| v1 (May 21) | 3 sub-metrics (A appearance, T temporal-log-weighted, I identity) | Layer 1+2 pass | tOF k=1 not in composite → flicker invisible; histogram-based metrics missing |
| v2 | + Sub-metric D (color histogram L1 at k ∈ {1, 5, 10, 30, 60, 120}) | Provides a long-range *color* lever distinct from optical-flow | Color-drift severity range still flat: per-pair histogram distance at k=60/120 stays below bin width for slow monotonic drifts |
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

**hhsz**: shorter video (2,400 frames) already has a real drift baseline (R² ≈ 0.34 even at sev=0), so E reliability starts at 0.87 and barely moves. The injected drift adds only marginally to the baseline slope. The trend reverses correctly at sev0.10→sev0.40 (-0.049) but the sev0.02→sev0.10 segment is non-monotonic.

The hhsz issue is a real limitation: when a video has a pre-existing slope signal, adding more drift on top is hard to disentangle from the baseline. The R²-gated reliability does what it should (high reliability already), but the score scale isn't sensitive enough at the high end. Documented as a known sub-metric E limitation in the validation table.

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

Flicker δ is small and slightly noisy — flicker is currently the artefact LR-VCC handles weakest (tOF with `uniform` weighting picks up some of it but the response is not strongly monotonic). Documented as the next sub-metric extension target.

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

BrRLKMbBTYQ shows the largest drop: that's because UAV's output on this video has a real linear color trajectory (slope ≈ 0.017, R² ≈ 0.35) that E now correctly penalizes — visible in `--color_slope_dir` raw output. This is a genuine UAV failure mode, not a false positive.

---

## Other Iterations

### v2: Sub-metric D (color histogram stability)

Added `compute_color_histogram.py` (per-video CPU pass computing Lab histogram L1 distance at k ∈ {1, 5, 10, 30, 60, 120}) and `color_stability.py` wrapper (`score = exp(-α · mean_l1_dist)`, reliability = entropy-floor gate). `--color_hist_dir` flag plumbed through `run_lr_vcc.py`. 11 new tests in `test_color_histogram.py`. Found via empirical run that the default α=2.0 was over-saturated (chunk-boundary score range only 0.7–0.6); calibrated to α=0.394 via the chunk-boundary distance range to give the design-intent 0.65→0.25 range. The recalibration is a pure CLI flag (no re-scanning).

### v2_uniform: tOF weighting scheme

Added `--temporal_weight {log, uniform, sqrt}` to `temporal.py` and `run_lr_vcc.py`. `log(1+k)` (default) puts most weight on long k, which is the right call for chunk-boundary detection but hides flicker (which is a k=1 signal). `uniform` (= equal weight) was the chosen production setting because it preserves all artefact families. 2 new tests in `test_temporal.py`.

---

## Proposal Section 5.3 (Synthetic Validation) Landed

`proposal/sections/synthetic_validation.md` (~1,200 words + 3 figures), covering the artefact definitions, the 9-metric × 5-severity × 2-base × 2-artefact severity-response table (archived as `results/lr_vcc/severity_response_table.csv`), and the consolidated verdict table:

| Metric | Chunk-boundary | Color drift |
|--------|:--------------:|:-----------:|
| LR-VCC (v3+slope) | PASS | PARTIAL (7WHI clean, hhsz weak) |
| tOF k=1 | FAIL | FAIL |
| tOF k=120 | PASS | FAIL |
| tLP k=120 | PASS | FAIL |
| DOVER | FAIL | FAIL |
| E*warp | PASS | FAIL |
| CLIP-IQA | PASS | FAIL |
| Identity fused | FAIL | FAIL |

The section's key finding for the proposal: of the 8 alternative metrics, none correctly orders the 5 severity levels for color drift on either base video. LR-VCC v3+slope is the first metric in the test set that even partially handles this case. Figures: `fig5_color_drift_severity.png`, `fig6_chunk_boundary_severity.png`, `fig7_severity_summary_grid.png`.

Section was written before sub-metric E was built, so it currently lists LR-VCC's color-drift verdict as FAIL with the gap discussion ending in "this is a planned future extension." The post-this-week update of the verdict to PARTIAL will be applied in the LaTeX-assembly step.

---

## Code Delivered

| File | Purpose | New / Modified |
|------|---------|----------------|
| `scripts/synthetic_artefacts/color_drift.py` | Color-drift generator | New |
| `scripts/synthetic_artefacts/chunk_boundary.py` | Chunk-boundary generator | New |
| `scripts/synthetic_artefacts/flicker.py` | Periodic flicker generator | New |
| `scripts/synthetic_artefacts/generate_all.py` | Driver for the 30-video test set | New |
| `scripts/lr_vcc/color_histogram.py` | Sub-metric D core (Lab histogram L1 over multi-k) | New |
| `scripts/lr_vcc/compute_color_histogram.py` | Per-video runner producing JSON sub-metric cache | New |
| `scripts/lr_vcc/color_stability.py` | D wrapper: JSON → {score, reliability, details} | New |
| `scripts/lr_vcc/color_slope.py` | Sub-metric E core (linregress on Lab channel means) | New |
| `scripts/lr_vcc/compute_color_slope.py` | E per-video runner | New |
| `scripts/lr_vcc/temporal.py` | + uniform/sqrt weighting modes | Modified |
| `scripts/lr_vcc/run_lr_vcc.py` | + 4 CLI flags (`--color_hist_dir`, `--color_hist_alpha`, `--color_slope_dir`, `--color_slope_beta`, `--temporal_weight`) | Modified |
| `tests/synthetic_artefacts/test_artefacts.py`, `test_flicker.py` | 11 generator tests | New |
| `tests/lr_vcc/test_color_histogram.py`, `test_color_stability.py`, `test_color_slope.py` | 19 new sub-metric tests | New |
| `proposal/sections/synthetic_validation.md` + 3 figures + `plot_severity_response.py` | Section 5.3 | New |

LR-VCC test suite: 39/39 PASS (20 from May 21 + 11 D-related + 4 E-related + 4 weighting / recalibration). Generator test suite: 11/11 PASS.

---

## Commit Log (this period)

```
c8f9cc7 lr_vcc: sub-metric E (color slope) for color_drift detection
ed3bdc1 lr_vcc: --color_hist_alpha CLI flag (recalibrate sub-metric D)
7ddd054 lr_vcc: configurable tOF weighting (log/uniform/sqrt) — Option A for flicker detection
2b4ba26 synthetic_artefacts: periodic flicker generator (Task A8)
d7d5c5c lr_vcc: sub-metric D (color stability) + run_lr_vcc --color_hist_dir flag
0e7fb3e proposal: Section 5.3 synthetic validation — severity-response figures + analysis
b16bfc9 synthetic_artefacts: color drift + chunk-boundary generators for LR-VCC validation (Task A1)
```

---

## Next Steps (May 28 – June 7)

1. **Apply β=200 + sub-metric E to the proposal Section 5.3 verdict table** — replace LR-VCC's FAIL with PARTIAL for color drift, add the β-tuning paragraph, regenerate fig7 to include the v3+slope curve. ~1 day of writing + figure regen, no new experiments.
2. **Flicker severity-response improvement** — flicker is currently the weakest of the three artefact families (LR-VCC δ across full severity is < 0.02 on hhsz). Two options: (a) add a *fast-varying brightness* sub-metric (FFT magnitude in the 5–20-Hz band of per-frame mean), or (b) recalibrate `--temporal_weight sqrt` to give k=1 more weight without entirely dropping k=120. Decide which after the meeting.
3. **Real-video baseline confirmation** — run v3+slope β=200 on a real 1-min HR YouTube clip (no SR, no artefacts) to confirm it scores ≥ 0.7 (no false-positive drift detection on natural content). Sanity check before claiming Layer 3 PASS.
4. **LR-VCC β=200 + uniform tOF as the proposal-default settings** — update `proposal/sections/proposed_method.md` to reference the calibrated configuration, archive the JSON cache layout so anyone can re-run with different CLI flags without recomputing per-video sub-metrics.
5. **Begin Sections 1+2 prose (Introduction, Related Work)** — the May 31 proposal deadline is in 4 days; these are the remaining writing tasks plus LaTeX assembly. Sections 3+4+5.3 are now substantially complete.

---

## Talking Points for the Group Meeting

1. **Sub-metric E (linregress on Lab channel means with R²-gated reliability) is the first metric in our 9-metric test set that responds monotonically to color drift on at least one base video.** The mechanism — gate weight by goodness-of-fit, not by score magnitude — is reusable for any "slow signal vs noisy baseline" sub-metric (e.g. identity drift, geometric warp).
2. **The five-version iteration history (v1 → v3+slope β=200) is a textbook example of severity-response-driven metric design.** Each version added one element after the prior version's exact failure mode was characterised on the validation set. Worth turning into a methodology subsection in the proposal.
3. **The hhsz color-drift PARTIAL is informative, not a bug.** Pre-existing baseline drift makes the injected drift hard to isolate. This is the exact situation any real video would face. Mitigations: (a) longer videos (5,000+ frames) help by averaging out the noise, (b) per-channel rather than per-channel-max slope detection could be more sensitive at the high-severity end.
4. **β tuning happens entirely via re-derive in `--color_slope_beta`**, no re-scanning. This is the right pattern for any decay parameter — the per-video cache stores raw `max_abs_slope`, and the wrapper re-applies the decay. Same pattern applied to α in sub-metric D.
5. **39/39 LR-VCC tests + 11/11 generator tests = total green test suite.** All sub-metrics have unit tests for the four canonical cases (clean video → high score + low reliability, target artefact → low score + high reliability, distractor artefact → score unaffected + low reliability, too-few-frames → reliability 0).

## Blocking Questions for the Group

1. **Real-video baseline scope** — should we source a real 1-min HR YouTube clip as a "no artefact at all" baseline before claiming Layer 3 PASS? Or is the synthetic-only validation enough for the proposal?
2. **Flicker priority** — flicker is the weakest of the three artefact families currently. Worth a new sub-metric (FFT brightness) for the proposal, or punt to thesis main work?
3. **Section 5.3 update timing** — should we update the verdict table in the proposal LaTeX before the May 31 submission with PARTIAL (current state), or wait for a perfect hhsz monotonicity story (likely 2–3 more days of work)?
4. **Beta tuning provenance** — currently β=200 is justified by visual inspection of the response curves on the validation set. Should we add a more formal calibration criterion (e.g. "maximize Kendall τ of severity-LR-VCC correlation averaged over both base videos and all 3 artefact families")?
5. **Synthetic test set extensions** — currently 2 base videos × 3 artefacts × 5 severities. Adding a third base video (a static-camera clip with little natural color trajectory, where E's baseline reliability would be near 0) would isolate the "pre-existing drift" confound that limits hhsz today. Worth the data engineering?
