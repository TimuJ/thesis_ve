# LR-VCC v6 — Sensitivity Calibration Design (Phase C.1)

**Status:** approved direction, pending spec review
**Date:** 2026-08-14
**Roadmap phase:** C (`docs/plans/2026-08-01-benchmark-roadmap.md`), first tranche
**Decision owners:** Timur

## Context

Phase C of the benchmark roadmap promotes the severity battery from *validator*
to *calibration signal*: fit each sub-metric's response parameters so that
severity 0.02 → 0.40 maps to a target monotone response on its designed artefact
family, calibrating and validating on disjoint bases, with the sign-flip control
families guarding against over-calibration — plus failure analysis of every
non-conforming matrix cell against the frozen v5 reference.

Two roadmap assumptions are relaxed by this design:

1. **Phase A does not block calibration.** The roadmap states that at n=5 "there
   is nothing to split". Leave-one-base-out cross-validation *is* a disjoint
   split — five folds, fit on four bases, report on the fifth. The statistics are
   weak at n=5 and the resulting parameters are explicitly provisional, but the
   protocol is sound and the identical code path re-runs on the enlarged base set
   when the video-sourcing line lands.
2. **No server or GPU is required.** All sub-metric outputs for 12 families ×
   5 bases × 5 severities (300 clips, 7 sub-metrics) are cached in
   `results/synthetic_artefacts_eval/` and `results/lr_vcc/`. A full 300-clip
   recomposition takes 2.7 s locally; the recomposition gate reproduces the
   stored v5 real-model composites bit-exact on this machine.

Scope for this tranche: **re-parameterization of v5 only.** No new sub-metrics,
no video re-scanning. Structural fixes surfaced by the failure analysis are
shortlisted for a later tranche with its own design.

## Empirical findings that shape the design

Probes over the cached ladder (all reproducible from `results/`):

### Finding 1 — reliability is not relevance

`weights = softmax(reliabilities / τ)`. Reliability answers *"is this
sub-metric's input trustworthy"*, never *"is it responsive to what is happening
in this clip"*. The colour sub-metrics carry reliability ≈ 1.0 by construction,
so at τ = 0.2 they dominate every cell — including cells where they are blind or
anti-correlated.

flicker / 7WHI2L_FDNg (v5 verdict FLAT, Δ −0.001):

| | A | T | I | D | E | D′ | D″ |
|---|---|---|---|---|---|---|---|
| Δscore 0.02→0.40 | −0.032 ✓ | −0.052 ✓ | +0.001 | +0.012 ✗ | +0.037 ✗ | +0.057 ✗ | −0.028 ✓ |
| weight | 0.039 | 0.130 | 0.206 | 0.207 | 0.005 | 0.207 | 0.207 |

The correctly-responding sub-metrics hold 0.17 of the weight; three moving the
wrong way hold 0.42. The signal exists and the composition cancels it. τ and the
reliability floors are therefore first-class calibration parameters, not
incidental ones.

### Finding 2 — sub-metric T has no response parameter

`T = 1 − weighted_mean(tOF)`. On flicker the raw tOF moves +49 % … +96 % across
the five bases, which the linear map compresses into ≈ 0.05 of score. This is
the largest single lever available and the reason β_T is in scope (approved):
the change is to T's *normalisation*, using the same cached tOF payloads, not to
its measurement.

### Finding 3 — five distinct failure stages, all observed

| stage | evidence |
|---|---|
| measurement | identity_drift: every raw stat moves < 6 % on 4/5 bases |
| normalisation | flicker: raw tOF +70 %, T score −0.05 |
| reward-direction | identity_degradation / 7WHI2L: I rises 0.375 → 0.489 *as identity degrades* (slow-fast pooling pathology), at 5× A's weight → cell INVERTED |
| gate | E carries weight 0.005 in most cells despite responding |
| composition | background_drift / mJog: D′ responds −0.265, cancelled by D″ (+0.085), D (+0.066), I (+0.055) |

A sixth confound: **weight drift along the ladder**. background_drift / BrRLK
has I's weight moving 0.017 → 0.176 *across severities*, so the composite shifts
through weight reallocation rather than any score change. The Δ statistic
silently mixes this with genuine response.

### Finding 4 — half the battery is scored against the wrong criterion

The six flip families are pre-registered *controls* with predicted outcomes
stated in the thesis: flip_horizontal invisible, flip_invert caught everywhere,
flip_channel_shuffle caught partially, periodic/elastic mostly invisible.
Scoring them with the uniform `PASS+WEAK = clean` rule counts a correct FLAT as
a failure. Expectation-aware scoring of the *unchanged* v5 results converts the
29/60 headline into ≈ 45/60 as-designed. This is a reporting fix available
independently of whether v6 ships.

## Architecture

```
scripts/lr_vcc/calibration/
├── __init__.py
├── expectations.py     # pre-registered per-family expectations + designed-for sub-metric map
├── response_table.py   # cached JSONs -> one flat table of raw statistics
├── recompose.py        # (table, parameter vector) -> composite, pure arithmetic
├── objective.py        # severity-response loss + control-silence penalty + hard guards
├── fit.py              # deterministic coordinate search, LOBO folds
└── report.py           # markdown emitters
scripts/lr_vcc/failure_analysis.py   # per-cell stage attribution
tests/test_lr_vcc_calibration.py
```

### `response_table.py`

Extract **once** a table with one row per (family, base, severity) plus one per
(method, video) for the real-model guard. Columns are the *raw statistics* that
free parameters act on, plus the quantities no parameter touches:

| column | source | consumed by |
|---|---|---|
| `a_mean`, `a_std` | `clip_iqa` list | λ_A, A gates |
| `tof_k`, `cov_k` (per k) | tOF payload | β_T, T gate |
| `identity_fused`, `face_rate`, `n_clips`, `dispersion` | identity JSON | I gates |
| `hist_dist` | `mean_l1_dist` | α |
| `slope_abs` | `max_abs_slope` | β_E |
| `anchor_q14` | D′ `|q4−q1|` | β_D′ |
| `clip_q14` | D″ `|q4−q1|` | β_D″ |
| `n_frames`, `closeup_p50`, stored reliabilities | various | gates |

Written to `results/lr_vcc/calibration/response_table.json`. Regenerating it is
the only step that touches the cached sub-metric JSONs.

### `recompose.py`

`composite(row, params) -> {"lr_vcc", "sub_scores", "weights", "low_confidence"}`
as pure arithmetic. This is what makes 5-fold LOBO over a real grid affordable:
2.7 s/config becomes sub-millisecond. Correctness is pinned by a bit-exactness
test against `run_lr_vcc.evaluate_one_video` at production parameters — the same
discipline as the existing `--mode gate`.

### Parameter vector

| symbol | sub-metric | current | new form | search range |
|---|---|---|---|---|
| λ_A | A | 0.5 | unchanged: `mean − λ·std` | [0, 3] linear |
| β_T | T | — (`1 − x`) | `exp(−β_T · weighted_mean_tof)` | [1, 50] log |
| α | D | 0.394 | `exp(−α · hist_dist)` | [0.05, 3] log |
| β_E | E | 200 | `exp(−β_E · slope_abs)` | [20, 2000] log |
| β_D′ | D′ | 0.5 | `exp(−β_D′ · anchor_q14)` | [0.1, 5] log |
| β_D″ | D″ | 3.0 | `exp(−β_D″ · clip_q14)` | [0.5, 30] log |
| τ | composition | 0.2 | `softmax(rel / τ)` | [0.05, 5] log |

β_T continuity: `exp(−x) ≈ 1 − x` for small x, so β_T = 1 approximately recovers
current behaviour on the observed tOF range (0.04–0.17). v5 is therefore a
near-interior point of the search space, not a boundary case.

Gate thresholds enter a second coordinate pass, held fixed during the response-
parameter pass: A `drift_floor` 0.02, A `saturation_ceiling` 0.98, T
`mask_cov_floor` 0.10, I `face_rate_floor` 0.20, I `closeup_threshold` 0.05. The
parked identity dispersion gate (0.346) stays parked — it could not be calibrated
defensibly on two single-face bases and n=5 does not change that.

### `expectations.py`

Per family: `RESPOND`, `SILENT`, or `UNCONSTRAINED`, with the rationale recorded
in the module.

| family | expectation | source |
|---|---|---|
| color_drift, background_drift, chunk_boundary, flicker, identity_degradation, identity_drift | RESPOND | designed-for families |
| flip_invert | RESPOND | positive control, histogram-destroying |
| flip_channel_shuffle | RESPOND | marginal-preserving, appearance-breaking; "caught partially" |
| flip_horizontal, flip_periodic, flip_elastic | SILENT | predicted invisible |
| flip_transpose | UNCONSTRAINED | pre-registration is ambiguous (histogram-preserving but geometry-destroying); excluded from the objective, reported only |

`flip_transpose` is deliberately left out of the fit rather than assigned a
post-hoc expectation after its results were seen.

This partitions the 60 cells into **40 RESPOND** (8 families × 5 bases),
**15 SILENT** (3 × 5) and **5 UNCONSTRAINED**. The fit therefore sees 55 cells,
of which 27 % are silence constraints — the over-calibration guard is a
substantial fraction of the objective, not a token term.

The module also declares the **designed-for sub-metric map** used by failure
attribution — which sub-metrics each family was built to excite:

| family | expected to fire |
|---|---|
| color_drift | D, E, D′ |
| background_drift | D′, D″, A |
| chunk_boundary | T, D |
| flicker | T, A |
| identity_degradation | I, A |
| identity_drift | I, D″ |
| flip_invert | D, D′, D″, A |
| flip_channel_shuffle | D′, D″, A |

### `objective.py`

Per cell `c = (family, base)` with ladder `S = {0.02, 0.05, 0.10, 0.20, 0.40}`
and composite scores `y_s`:

```
R(c) = y_0.02 − y_0.40                              # response; positive = correct direction
M(c) = Σ_i max(0, y_{s_{i+1}} − y_{s_i})            # upward (wrong-way) violation
```

**Sign convention.** `build_verdict_matrix.collect_deltas` uses
`Δ = score(0.40) − score(0.02)`, so PASS is `Δ ≤ −0.05`. The loss uses the
opposite sign, `R = −Δ`, so that "larger is better" holds throughout the fit.
The verdict-matrix emitters keep the existing Δ convention unchanged; only the
objective works in R.

```
RESPOND:  L(c) = max(0, R_target − R(c))² + w_mono · M(c)
SILENT:   L(c) = w_silence · max(0, |R(c)| − R_silent)²
```

Defaults: `R_target = 0.10` (comfortably above the PASS threshold 0.05, below the
largest observed responses), `R_silent = 0.02` (the FLAT band), `w_mono = 1.0`,
`w_silence = 3.0`. The silence penalty is deliberately asymmetric — over-
calibration is the failure mode being guarded against.

```
Loss = mean_{RESPOND} L(c) + mean_{SILENT} L(c)
```

Using all five ladder points rather than the two endpoints is a change from the
v5 protocol, where `collect_deltas` reads only 0.02 and 0.40 (severities 0.05,
0.10 and 0.20 are cached but never used).

**Hard guards.** A parameter vector is rejected outright if it breaks the
headline real-model results: the canonical order `flashvsr > mgld > uav` must
hold on the aggregate mean, and MGLD > UAV must hold on 5/5 videos. v6 may not
buy matrix cells with the leaderboard.

`R_target` is a placeholder for the perceptual target curve: Phase B's human
severity ratings replace the flat magnitude target with a measured curve shape,
so the target enters the loss as a pluggable function of severity.

### `fit.py`

Deterministic coordinate descent over the log grids above (3 passes, fixed
parameter order, no randomness — reproducibility matters more than optimality at
this scale). Two stages: response parameters first, gate thresholds second.

LOBO: for fold *i*, fit on the four bases ≠ *i*, evaluate on base *i*. The five
held-out columns assemble into one full 60-cell matrix in which **every cell was
produced by a fit that never saw its own base** — directly comparable to the v5
matrix. Reported alongside:

- the in-sample matrix (fit on all five bases), so the in-sample/held-out gap is
  a visible overfitting measurement;
- per-fold parameter vectors, so parameter stability across folds is legible;
- the loss surface per parameter (this is the sensitivity deliverable — it shows
  which parameters the data actually constrains at n=5).

Final shipped v6 parameters are refit on all five bases (standard CV practice);
the held-out matrix remains the honest performance estimate, and reports state
this explicitly.

### `failure_analysis.py`

For each cell and each sub-metric expected to fire:

```
rel_raw   = |raw_0.40 − raw_0.02| / (|raw_0.02| + ε)
Δscore    = score_0.40 − score_0.02
w̄         = mean weight over the ladder
contrib   = w̄ · [log(score_0.40 + ε) − log(score_0.02 + ε)]
```

Attribution, first match wins:

| stage | rule |
|---|---|
| measurement | `rel_raw < 0.05` |
| reward-direction | `rel_raw ≥ 0.05` and `Δscore > +0.01` |
| normalisation | `rel_raw ≥ 0.20` and `|Δscore| < 0.02` |
| gate | correct-direction `|Δscore| ≥ 0.02` and `w̄ < 0.05` |
| composition | responds and is weighted, but Σ positive contributions from other sub-metrics ≥ \|its negative contribution\| |

Plus a `weight_drift` flag when `max_s w − min_s w > 0.05`.

Only *measurement* and *reward-direction* are outside re-parameterization's
reach. The count of cells in those two classes is the honest ceiling on what v6
can deliver, and it is reported as such.

## Testing

TDD throughout; tests precede implementation.

1. `test_recompose_bit_exact` — table recomposition equals `evaluate_one_video`
   at production parameters for all 300 artefact clips and 15 real-model clips,
   `|diff| < 1e-12`.
2. `test_v5_frozen` — stored v5 composites reproduce exactly (promotes the
   existing `--mode gate` check into pytest so v5 cannot drift silently).
3. `test_beta_t_continuity` — `exp(−β_T·x)` is monotone decreasing in x for all
   β_T > 0, and matches `1 − x` within 5e-3 at β_T = 1 over x ∈ [0.04, 0.17].
4. `test_loss_silence_penalty` — a responding SILENT cell scores > 0; a silent
   one scores 0.
5. `test_loss_monotonicity` — a non-monotone ladder is penalised; a monotone one
   with equal endpoints is not.
6. `test_lobo_disjoint` — fold *i*'s training set never contains base *i*. This
   guards the central methodological claim.
7. `test_expectations_cover_all_families` — every family in `ARTEFACTS` has a
   declared expectation and, if RESPOND, a designed-for sub-metric list.
8. `test_attribution_rules` — synthetic fixtures, one per stage.
9. `test_hard_guards` — a parameter vector that inverts the method order is
   rejected.

## Deliverables

| artefact | content |
|---|---|
| `reports/figures/response_curves_v5.md` | five-point severity curves per cell, replacing endpoint deltas |
| `reports/figures/expectation_scored_matrix_v5.md` | v5 rescored against pre-registered expectations, both counts reported |
| `reports/figures/failure_attribution_v5.md` | every non-conforming cell with its stage; calibration-addressable vs structural totals |
| `reports/figures/calibration_v6_lobo.md` | v6 held-out matrix vs frozen v5, in-sample gap, per-fold parameters, loss surfaces |
| `results/lr_vcc/calibration/` | response table, fitted parameter vectors, per-fold outputs |

v5 remains frozen and canonical until v6 is reviewed; nothing is rewritten
retroactively.

## Non-goals

- New or replacement sub-metrics (scene-cut-aware anchor, mirror-sensitive
  sub-metric) — shortlisted by the failure analysis, deferred to a later tranche
  with its own design and server time.
- Re-scanning any video, regenerating any artefact clip, or any GPU work.
- Un-parking the identity dispersion gate.
- Human-anchored calibration targets (Phase B, unavailable).
- Any change to the base-video set (Phase A, in progress elsewhere).

## Risks

- **Overfitting at n=5.** Five bases, seven parameters. Mitigations: LOBO as the
  reported protocol, in-sample gap published, per-fold parameter stability shown,
  loss surfaces exposing which parameters the data does not constrain. If held-out
  gain is within noise, the honest outcome is "harness built, v6 deferred" — and
  the spec treats that as a valid result, not a failure.
- **Silence penalty too weak or too strong.** `w_silence = 3.0` is a judgment
  call. Reported with a sensitivity row at 1.0 and 10.0.
- **Guards may bind.** If no parameter vector both improves the matrix and
  preserves the leaderboard, that tension is itself a finding worth reporting
  rather than a bug to tune around.
- **`R_target = 0.10` is arbitrary until Phase B.** Chosen relative to the
  existing verdict thresholds; the loss is written so the human-derived curve
  drops in without restructuring.
