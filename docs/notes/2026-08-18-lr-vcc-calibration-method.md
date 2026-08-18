# How the consistency metric is calibrated — method note

Written for colleagues who asked how the pipeline and its weights are actually
fitted. It covers what has free parameters and what does not, how the weighting
works (and why it was the main thing the fit changed), what the objective is,
and what stops the fit from cheating.

## 1. The two levels, and which one has the weights

The composite is built in two stages, and it matters which is which because the
word "weights" gets used for both.

**Stage 1 — each sub-metric turns a raw measurement into a score in [0, 1].**
Seven of them: appearance (A), temporal (T), identity (I), colour stability (D),
colour slope (E), anchored colour histogram (D′), CLIP trajectory (D″). Each has
a *response function* mapping its raw statistic to a score, e.g.

```
D_score   = exp(-alpha  * mean_histogram_distance)
D''_score = exp(-beta   * |quarter4_mean - quarter1_mean|)
A_score   = mean(quality) - lambda * std(quality)
```

**Stage 2 — the seven scores are combined into one number.**

```
weights = softmax(reliabilities / tau)
LR-VCC  = exp( sum_i  weights_i * log(score_i + eps) )
```

The geometric (log) mean is deliberate: a sub-metric that collapses drags the
composite down and cannot be compensated for by another scoring well. Averaging
in the log domain means "no compensation across failures."

**The key point about the weights: they are not free parameters.** There is no
per-sub-metric weight that gets fitted. Weights are *derived per video* from
each sub-metric's **reliability**, and reliability is a statement about whether
that sub-metric's input is trustworthy on this particular video:

| sub-metric | reliability drops when |
|---|---|
| A | quality saturates, or its spread is too small to discriminate |
| T | optical-flow mask coverage is too low at a given time scale |
| I | too few clips contain faces, or the content is close-up |
| D | too few frames to evaluate the longest time scale |
| E, D′, D″ | carried from the measurement stage |

So a video with no faces automatically stops being scored on identity, without
anyone hand-picking that. Only **one** free parameter lives in stage 2: `tau`,
the softmax temperature, which controls how sharply the most-reliable
sub-metric dominates.

## 2. The flaw this exposed — reliability is not relevance

This is the most useful thing to say about the weighting, because it is a real
design defect that the calibration surfaced rather than a tuning detail.

Reliability answers *"can I trust this sub-metric's input?"* — never *"is this
sub-metric actually responding to what is happening?"* The colour sub-metrics
have reliability ≈ 1.0 by construction on almost every video, because their
inputs are always well-formed. With a sharp softmax they therefore dominate
every cell, **including cells where they are blind or actively anti-correlated.**

A worked case — the flicker corruption on one base video:

| | A | T | I | D | E | D′ | D″ |
|---|---|---|---|---|---|---|---|
| score change as severity rises | −0.032 ✅ | −0.052 ✅ | +0.001 | +0.012 ❌ | +0.037 ❌ | +0.057 ❌ | −0.028 ✅ |
| weight | 0.039 | 0.130 | 0.206 | 0.207 | 0.005 | 0.207 | 0.207 |

The two sub-metrics that correctly detect flicker hold **0.17** of the weight;
three that move the *wrong* way hold **0.42**. The signal exists and the
composition cancels it. The cell reads FLAT.

This is why `tau` moved so much in the fit (0.2 → 1.26): a much flatter softmax
lets responsive sub-metrics carry weight instead of being drowned by reliable
but unresponsive ones. It is the single largest change the calibration made.

## 3. What is actually calibrated — twelve numbers

| | parameter | v5 | v6 candidate |
|---|---|---|---|
| composition | `tau` softmax temperature | 0.2 | **1.26** |
| T | `beta_t` response steepness | *(linear form)* | **33.81** |
| A | `lambda` drift weight | 0.5 | **2.0** |
| D | `alpha` | 0.394 | **0.583** |
| E | `beta_e` | 200 | 200 (unmoved) |
| D′ | `beta_dp` | 0.5 | **0.707** |
| D″ | `beta_dpp` | 3.0 | **8.78** |
| gates | A drift floor | 0.02 | **0.0** |
| gates | T mask-coverage floor | 0.10 | **0.0** |
| gates | A saturation ceiling, I face-rate floor, I close-up threshold | — | unmoved |

Two notes worth making when presenting this:

- **Sub-metric T previously had no response parameter at all.** Its score was
  `1 − weighted_mean(tOF)`. On flicker the raw optical-flow statistic moves
  50–95% while the score moves 0.05 — the linear map compresses the signal
  almost out of existence. Giving T a response function was the largest single
  lever available. Importantly, the *original linear form was left in the search
  grid as an option* and the fit did not choose it, so the new parameter was
  adopted on evidence rather than imposed.
- **`mask_cov_floor` → 0.0 changes T's input set**, not just its scaling: it
  disables the coverage filter entirely, so time scales previously discarded now
  contribute. Worth flagging because it is a behavioural change hiding in what
  looks like a threshold tweak.

## 4. The objective — what "calibration" is fitted against

The battery is 12 synthetic corruption families × 5 long videos × 5 severities.
Each family carries a prediction registered **in advance**:

- **respond** (6 designed-for families + 2 positive controls): the score must
  fall monotonically as severity rises.
- **stay silent** (3 control families): the score must not move at all. A
  horizontal mirror flip preserves colour statistics exactly and CLIP embeddings
  are documented to be flip-robust, so every sub-metric is *physically blind* to
  it — flat is the correct answer, not a failure.
- **unconstrained** (1 family): the prediction was genuinely ambiguous, so it is
  excluded from the objective rather than given a post-hoc expectation.

The loss, per cell, over all five severity points:

```
R = score(severity 0.02) - score(severity 0.40)          # response, larger is better
M = total upward movement along the ladder                # monotonicity violation

respond cells:  loss = max(0, R_target - R)^2  +  w_mono * M
silent  cells:  loss = w_silence * max(0, max_excursion - R_silent)^2
```

with `R_target = 0.10`, `R_silent = 0.02`, `w_mono = 1.0`, `w_silence = 3.0`.

Two design choices to defend:

- **All five severity points are read.** The previous verdict protocol compared
  only the two endpoints, so a wildly non-monotone response scored the same as a
  clean one.
- **Silence is penalised three times as heavily as under-response.** Fitting a
  metric to react to corruption has a trivial cheat — react to *everything*.
  Roughly a quarter of the objective is silence constraints, deliberately
  over-weighted, because over-calibration is the failure mode the fit is
  actively searching toward.

## 5. What stops the fit from cheating

**Controls, as above.** Evidence they are load-bearing rather than decorative:
fitted on all five videos, control conformance drops to 12/15 — the fit *does*
trade away control silence to buy response elsewhere. On held-out videos it is
15/15, so the trade does not generalise. The guard caught a real attempt.

**Nothing is reported in-sample.** Twelve parameters against five videos: fitting
and grading on the same set measures memorisation. Every number comes from
leave-one-out rotation — fit on four videos, evaluate on the fifth, rotate — so
the reported matrix has every cell produced by parameters blind to its own
video. The in-sample/held-out gap (0.0116 vs 0.0174) *is* the overfitting,
reported as such.

The subtlety worth mentioning: the leaderboard guard also had to be restricted
to each fold's training videos. It compares methods on the same five videos, so
evaluating it globally while fitting on four would leak the held-out video back
into the fit through the guard. There is a test that intercepts the arguments of
every call made during fitting to prove this, rather than trusting labels the
code writes about itself.

**A hard constraint on the leaderboard.** Any parameter vector that inverts the
established method ranking is rejected outright — the calibration cannot buy
validation-matrix cells by degrading the headline result.

**The search itself is deterministic.** Coordinate descent over fixed log grids,
fixed parameter order, no randomness, run to convergence rather than to a fixed
iteration budget. Same inputs always produce the same parameters.

## 6. What the calibration did and did not achieve

Held-out loss improves 35% (0.0269 → 0.0174, better on 4 of 5 videos compared
like-for-like). The conformance count is **unchanged at 39/55**. Both are
reported; the candidate is not adopted.

The honest reading is that the objective and the verdict count measure different
things, and at five videos we cannot resolve which one moved for real reasons.
Roughly half the fitted parameters are not identified by this much data — folds
scatter across their grids or optimise to the edge — which is the clearest
evidence that the limitation is sample size rather than method.

## 7. Why this is cheap to re-run

Every sub-metric's raw statistic is extracted once and cached. The fit then
recomposes the whole 315-clip matrix as pure arithmetic — no video decoding, no
network forward passes. A full matrix recomposition went from 2.7 s to under a
millisecond, which is what makes a five-fold search over a real parameter grid
affordable on a laptop; the entire fit finishes in under a minute.

The correctness gate for that shortcut: the fast path must reproduce the
canonical implementation **bit-exactly** at the published parameters. It does —
0.0 worst-case difference across all 315 clips, pinned as a regression test, so
the published configuration cannot drift silently.
