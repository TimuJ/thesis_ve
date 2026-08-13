# LR-VCC v6 — calibration under leave-one-base-out

Every fold's coordinate search is warm-started from `PROD_PARAMS` — v5's parameter vector, which was itself chosen with all five base videos in view. This is not a data leak: each fold's loss and leaderboard guards are strictly restricted to its four training bases, and the held-out base never enters the search that produces that fold's parameters. But the fold results are honest *conditional on* that starting point, not on a blank slate — a cold start could in principle land somewhere else. Targets: R_target=0.1, R_silent=0.02, w_mono=1.0, w_silence=3.0.

`beta_t=None` — sub-metric T's original linear response — was in the search grid and was **not** chosen; the refit on all five bases lands on beta_t≈33.81 (T's exponential form). The new lever earned its place rather than the fit declining it and falling back to v5's form.

## Summary

- v5 loss, all five bases: **0.026884**
- v6 in-sample loss, all five bases (the refit sees every base): **0.011588** — the gap to the held-out numbers below is the overfitting gap.
- mean v6 held-out loss (average of the five paired test losses below): **0.017376**
- mean paired v5 loss (v5 scored on each of the same five held-out bases, then averaged): **0.026884**

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

| parameter | v5 | v6 |
|---|---|---|
| tau | 0.2 | 1.25594 |
| beta_t | linear | 33.8122 |
| lambda_a | 0.5 | 2 |
| alpha | 0.394 | 0.583258 |
| beta_e | 200 | 200 |
| beta_dp | 0.5 | 0.707107 |
| beta_dpp | 3 | 8.78367 |

## Held-out verdict matrix

Every cell produced by a fit that never saw its own base.

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
