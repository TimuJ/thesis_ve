# LR-VCC v6 — calibration under leave-one-base-out

Every fold's coordinate search is warm-started from `PROD_PARAMS` — v5's parameter vector, which was itself chosen with all five base videos in view. This is not a data leak: each fold's loss and leaderboard guards are strictly restricted to its four training bases, and the held-out base never enters the search that produces that fold's parameters. But the fold results are honest *conditional on* that starting point, not on a blank slate — a cold start could in principle land somewhere else. Targets: R_target=0.1, R_silent=0.02, w_mono=1.0, w_silence=3.0.

`beta_t=None` — sub-metric T's original linear form — was present in the search grid and was not selected; the fit chose beta_t≈33.81 instead. The new response parameter was therefore adopted on the evidence, not imposed.

## Summary

- v5 loss, all five bases: **0.026884**
- v6 in-sample loss, all five bases (the refit sees every base): **0.011588** — the gap to the held-out numbers below is the overfitting gap.
- mean v6 held-out loss (average of the five paired test losses below): **0.017376**
- mean paired v5 loss (v5 scored on each of the same five held-out bases, then averaged): **0.026884**
- **held-out conformance (as-designed) is unchanged from v5 — see Conformance comparison below. The loss improvement above does not carry over to the verdict-level count.**

## Per-fold results (paired, same-base comparison)

v6's held-out loss and v5's loss in each row are measured on the *same* held-out base — never v6's one-base number against v5's five-base aggregate.

| fold (held out) | train loss | v6 held-out loss | v5 loss (same base) | delta (v5−v6) | converged |
|---|---|---|---|---|---|
| 7WHI2L_FDNg | 0.00840 | 0.00942 | 0.02151 | +0.01209 | True |
| BrRLKMbBTYQ | 0.00809 | 0.04214 | 0.06464 | +0.02250 | True |
| KZ8p6b1zJ9U | 0.01203 | 0.01575 | 0.00480 | -0.01095 | True |
| hhszUXL1Cu8 | 0.01347 | 0.00740 | 0.01642 | +0.00903 | True |
| mJog8DlRk_4 | 0.00826 | 0.01217 | 0.02705 | +0.01488 | True |

- v6 improves on v5 on **4/5 folds** on a paired, same-base basis.
  - the exception is **KZ8p6b1zJ9U**, where v6 is worse (0.015752 vs v5's 0.004800).
  - v6's largest win is **BrRLKMbBTYQ** (0.042139 vs v5's 0.064636) — that base is simply the hardest one in the set, not evidence of a regression elsewhere.
- all five folds and the final refit report **converged=True**.

## Final parameters (refit on all five bases)

All twelve searched parameters: the seven response parameters, then the five gate thresholds.

| parameter | v5 | v6 |
|---|---|---|
| tau | 0.2 | 1.25594 |
| beta_t | linear | 33.8122 |
| lambda_a | 0.5 | 2 |
| alpha | 0.394 | 0.583258 |
| beta_e | 200 | 200 |
| beta_dp | 0.5 | 0.707107 |
| beta_dpp | 3 | 8.78367 |
| *(gate thresholds)* | | |
| mask_cov_floor | 0.1 | 0 |
| a_drift_floor | 0.02 | 0 |
| a_sat_ceiling | 0.98 | 0.98 |
| face_rate_floor | 0.2 | 0.2 |
| closeup_threshold | 0.05 | 0.05 |

- gate threshold(s) that moved: `mask_cov_floor` 0.1 → 0, `a_drift_floor` 0.02 → 0.
- **`mask_cov_floor` 0.1 → 0 materially changes sub-metric T's input set.** Under v5, **183/315** rows had at least one tOF sample whose coverage fell below the floor and was excluded from T's weighted average; under v6 the floor is 0, so **0/315** rows are affected — the coverage filter is effectively disabled everywhere it used to fire.

## Per-fold parameter vectors

The seven response parameters and five gate thresholds each fold actually landed on, fit without ever seeing the column's own base. `*` marks a value sitting at the minimum or maximum of its grid: the fold's optimum is at or beyond the edge of the declared search space, so that parameter is not identified by the data at this sample size.

| parameter | 7WHI2L_FDNg | BrRLKMbBTYQ | KZ8p6b1zJ9U | hhszUXL1Cu8 | mJog8DlRk_4 |
|---|---|---|---|---|---|
| tau | 0.792447 | 1.25594 | 0.315479 | 3.15479 | 5* |
| beta_t | 50* | 15.4625 | 33.8122 | 15.4625 | 50* |
| lambda_a | 3* | 1.5 | 1 | 1.5 | 2 |
| alpha | 3* | 0.257176 | 0.583258 | 0.583258 | 3* |
| beta_e | 2000* | 316.979 | 126.191 | 79.6214 | 200 |
| beta_dp | 2.28653 | 0.478176 | 0.707107 | 0.478176 | 2.28653 |
| beta_dpp | 30* | 3.87298 | 8.78367 | 8.78367 | 30* |
| mask_cov_floor | 0* | 0* | 0* | 0* | 0* |
| a_drift_floor | 0* | 0* | 0* | 0* | 0* |
| a_sat_ceiling | 0.98 | 0.98 | 0.98 | 0.98 | 0.98 |
| face_rate_floor | 0.1 | 0.2 | 0.4* | 0.4* | 0.2 |
| closeup_threshold | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 |

- boundary hits among the seven response parameters, per fold (held-out base): 7WHI2L_FDNg 5/7, BrRLKMbBTYQ 0/7, KZ8p6b1zJ9U 0/7, hhszUXL1Cu8 0/7, mJog8DlRk_4 4/7.

## Loss surface (sensitivity at the chosen point)

For each searched parameter, every other parameter is held at its `final_params` value and this one is swept over its declared grid; `matrix_loss` (all five bases, no leaderboard guard) is recorded at every point — the sensitivity a fold-level result cannot show on its own. `spread` is the gap between the loss at the chosen value and the worst point on the grid: small means this data barely constrains that parameter at n=5, large means the fit actively prefers the chosen value over the alternatives.

| parameter | chosen | loss @ chosen | best on grid | worst on grid | spread |
|---|---|---|---|---|---|
| tau | 1.25594 | 0.011588 | 1.25594 (0.011588) | 0.05 (0.019496) | 0.007908 |
| beta_t | 33.8122 | 0.011588 | 33.8122 (0.011588) | 1 (0.017600) | 0.006011 |
| lambda_a | 2 | 0.011588 | 2 (0.011588) | 3 (0.025204) | 0.013615 |
| alpha | 0.583258 | 0.011588 | 0.583258 (0.011588) | 3 (0.013842) | 0.002254 |
| beta_e | 200 | 0.011588 | 200 (0.011588) | 20 (0.014798) | 0.003210 |
| beta_dp | 0.707107 | 0.011588 | 0.707107 (0.011588) | 5 (0.024350) | 0.012762 |
| beta_dpp | 8.78367 | 0.011588 | 8.78367 (0.011588) | 30 (0.015040) | 0.003452 |
| mask_cov_floor | 0 | 0.011588 | 0 (0.011588) | 0.2 (0.014034) | 0.002445 |
| a_drift_floor | 0 | 0.011588 | 0 (0.011588) | 0.1 (0.011759) | 0.000171 |
| a_sat_ceiling | 0.98 | 0.011588 | 0.9 (0.011588) | 0.9 (0.011588) | 0.000000 |
| face_rate_floor | 0.2 | 0.011588 | 0.2 (0.011588) | 0 (0.011717) | 0.000129 |
| closeup_threshold | 0.05 | 0.011588 | 0.02 (0.011588) | 0.02 (0.011588) | 0.000000 |

- **flat** (worst point on the grid raises the loss by less than 5% of the chosen value's loss — not constrained by this data at n=5): a_drift_floor, a_sat_ceiling, face_rate_floor, closeup_threshold.
- **sharp** (worst point raises the loss by more than 50%): tau, beta_t, lambda_a, beta_dp.

## Held-out verdict matrix

Every cell produced by a fit that never saw its own base.

Sign convention: **delta = −R = y(0.40) − y(0.02)**, same as `expectation_scored_matrix_v5.md`; negative is the correct direction.

| artefact | 7WHI2L_FDNg | BrRLKMbBTYQ | KZ8p6b1zJ9U | hhszUXL1Cu8 | mJog8DlRk_4 |
|---|---|---|---|---|---|
| background_drift | -0.005 FLAT ✗ | -0.005 FLAT ✗ | -0.149 PASS ✓ | -0.056 PASS ✓ | -0.019 FLAT ✗ |
| chunk_boundary | -0.046 WEAK ✓ | +0.016 FLAT ✗ | -0.148 PASS ✓ | -0.155 PASS ✓ | -0.049 WEAK ✓ |
| color_drift | -0.045 WEAK ✓ | -0.002 FLAT ✗ | -0.035 WEAK ✓ | -0.015 FLAT ✗ | -0.028 WEAK ✓ |
| flicker | -0.020 WEAK ✓ | -0.009 FLAT ✗ | -0.114 PASS ✓ | -0.034 WEAK ✓ | -0.033 WEAK ✓ |
| flip_channel_shuffle | -0.020 FLAT ✗ | -0.026 WEAK ✓ | -0.136 PASS ✓ | -0.057 PASS ✓ | -0.023 WEAK ✓ |
| flip_elastic | +0.002 FLAT ✓ | -0.007 FLAT ✓ | -0.000 FLAT ✓ | +0.010 FLAT ✓ | -0.009 FLAT ✓ |
| flip_horizontal | +0.013 FLAT ✓ | +0.000 FLAT ✓ | -0.009 FLAT ✓ | -0.007 FLAT ✓ | -0.001 FLAT ✓ |
| flip_invert | -0.068 PASS ✓ | -0.142 PASS ✓ | -0.180 PASS ✓ | -0.330 PASS ✓ | -0.027 WEAK ✓ |
| flip_periodic | -0.003 FLAT ✓ | -0.005 FLAT ✓ | -0.007 FLAT ✓ | -0.005 FLAT ✓ | -0.017 FLAT ✓ |
| flip_transpose | -0.050 WEAK — | -0.003 FLAT — | -0.057 PASS — | -0.070 PASS — | -0.037 WEAK — |
| identity_degradation | -0.003 FLAT ✗ | +0.000 FLAT ✗ | -0.017 FLAT ✗ | -0.057 PASS ✓ | -0.002 FLAT ✗ |
| identity_drift | +0.002 FLAT ✗ | -0.001 FLAT ✗ | -0.007 FLAT ✗ | -0.055 PASS ✓ | -0.002 FLAT ✗ |

- held-out as-designed: **39/55**

## Conformance comparison: v5 vs v6

The loss numbers above are the fit objective, not the reader-facing verdict count; this table puts both scoring protocols side by side for v5, v6 held-out, and v6 in-sample.

| | RESPOND | SILENT | as-designed | uniform PASS+WEAK |
|---|---|---|---|---|
| v5 | 25/40 | 14/15 | 39/55 | 29/60 |
| v6 held-out | 24/40 | 15/15 | 39/55 | 28/60 |
| v6 in-sample | 30/40 | 12/15 | 42/55 | 35/60 |

- **the as-designed count is unchanged at the verdict level: 39/55 for both v5 and v6 held-out. The loss improved (mean held-out 0.017376 vs mean paired v5 0.026884) but conformance did not.**
- the genuine win is at the verdict-shape level, not the count: **4 cells** that were INVERTED under v5 (background_drift/BrRLKMbBTYQ, chunk_boundary/BrRLKMbBTYQ, flicker/BrRLKMbBTYQ, identity_degradation/7WHI2L_FDNg) become FLAT under v6 held-out — a wrong-direction response replaced by no response, even though neither counts as conforming for a RESPOND family.
- SILENT held-out reaches **15/15** — every control family stays FLAT on its held-out base.
- SILENT in-sample drops to **12/15** (3 cells respond when they should stay flat) — direct evidence of the over-calibration the silence penalty exists to catch: with every base in view, the fit can trade a little unwanted control-family sensitivity for a lower RESPOND loss elsewhere. The held-out folds above show that trade does not survive to an unseen base.
