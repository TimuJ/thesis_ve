# LR-VCC v5 — failure attribution

For every non-conforming cell, the sub-metrics the family was built to excite, and the stage at which the signal was lost.

| artefact | base | verdict | sub-metric | stage | raw Δ% | score Δ | mean w | weight drift |
|---|---|---|---|---|---|---|---|---|
| background_drift | 7WHI2L_FDNg | FLAT | color_hist_anchor | composition | +54% | -0.088 | 0.202 |  |
| background_drift | 7WHI2L_FDNg | FLAT | clip_trajectory | composition | +36% | -0.069 | 0.202 |  |
| background_drift | 7WHI2L_FDNg | FLAT | appearance | measurement | +2% | -0.008 | 0.038 |  |
| background_drift | BrRLKMbBTYQ | INVERTED | color_hist_anchor | reward_direction | +18% | +0.071 | 0.194 | yes |
| background_drift | BrRLKMbBTYQ | INVERTED | clip_trajectory | composition | +87% | -0.322 | 0.194 | yes |
| background_drift | BrRLKMbBTYQ | INVERTED | appearance | composition | +18% | -0.095 | 0.056 |  |
| background_drift | mJog8DlRk_4 | FLAT | color_hist_anchor | composition | +80% | -0.265 | 0.214 |  |
| background_drift | mJog8DlRk_4 | FLAT | clip_trajectory | reward_direction | +91% | +0.085 | 0.214 |  |
| background_drift | mJog8DlRk_4 | FLAT | appearance | gate | +13% | -0.073 | 0.047 |  |
| chunk_boundary | BrRLKMbBTYQ | INVERTED | temporal | composition | +44% | -0.038 | 0.111 |  |
| chunk_boundary | BrRLKMbBTYQ | INVERTED | color_stability | composition | +44% | -0.208 | 0.233 |  |
| flicker | 7WHI2L_FDNg | FLAT | temporal | composition | +41% | -0.051 | 0.129 |  |
| flicker | 7WHI2L_FDNg | FLAT | appearance | gate | +6% | -0.032 | 0.040 |  |
| flicker | BrRLKMbBTYQ | INVERTED | temporal | composition | +33% | -0.049 | 0.119 |  |
| flicker | BrRLKMbBTYQ | INVERTED | appearance | measurement | +4% | -0.016 | 0.058 |  |
| flicker | hhszUXL1Cu8 | FLAT | temporal | composition | +36% | -0.028 | 0.129 |  |
| flicker | hhszUXL1Cu8 | FLAT | appearance | composition | +8% | -0.041 | 0.051 |  |
| flicker | mJog8DlRk_4 | FLAT | temporal | composition | +49% | -0.080 | 0.066 |  |
| flicker | mJog8DlRk_4 | FLAT | appearance | measurement | +4% | -0.021 | 0.048 |  |
| flip_channel_shuffle | hhszUXL1Cu8 | FLAT | color_hist_anchor | normalisation | +14% | -0.051 | 0.180 |  |
| flip_channel_shuffle | hhszUXL1Cu8 | FLAT | clip_trajectory | normalisation | +41% | -0.139 | 0.180 |  |
| flip_channel_shuffle | hhszUXL1Cu8 | FLAT | appearance | measurement | +1% | +0.003 | 0.047 |  |
| identity_degradation | 7WHI2L_FDNg | INVERTED | identity | reward_direction | +23% | +0.115 | 0.206 |  |
| identity_degradation | 7WHI2L_FDNg | INVERTED | appearance | gate | +13% | -0.064 | 0.039 |  |
| identity_degradation | BrRLKMbBTYQ | FLAT | identity | measurement | +3% | +0.022 | 0.025 |  |
| identity_degradation | BrRLKMbBTYQ | FLAT | appearance | measurement | +2% | -0.012 | 0.058 |  |
| identity_degradation | mJog8DlRk_4 | FLAT | identity | reward_direction | +13% | +0.047 | 0.218 |  |
| identity_degradation | mJog8DlRk_4 | FLAT | appearance | gate | +6% | -0.031 | 0.047 |  |
| identity_drift | 7WHI2L_FDNg | FLAT | identity | measurement | +2% | -0.007 | 0.206 |  |
| identity_drift | 7WHI2L_FDNg | FLAT | clip_trajectory | measurement | +1% | -0.001 | 0.207 |  |
| identity_drift | BrRLKMbBTYQ | FLAT | identity | reward_direction | +8% | +0.049 | 0.022 |  |
| identity_drift | BrRLKMbBTYQ | FLAT | clip_trajectory | composition | +8% | -0.004 | 0.227 |  |
| identity_drift | mJog8DlRk_4 | FLAT | identity | measurement | +1% | -0.004 | 0.219 |  |
| identity_drift | mJog8DlRk_4 | FLAT | clip_trajectory | normalisation | +9% | -0.008 | 0.220 |  |

## Totals by stage

**16 of 55** constrained cells fail their expectation; the table above attributes **34 findings** across them (a cell names every one of its designed-for sub-metrics, so it can contribute more than one row).

- composition: 13
- gate: 4
- measurement: 9
- normalisation: 3
- reward_direction: 5

- **calibration-addressable** (normalisation / gate / composition): 20
- **structural** (measurement / reward-direction — needs a different measurement, not a different constant): 14

The structural count is the honest ceiling on what a re-parameterised v6 can recover: 14 of the 34 attributed findings cannot be fixed by refitting constants alone, no matter how the fit is run.

## SILENT failures with no designed-for sub-metric

These cells fail a SILENT expectation but have no `DESIGNED_FOR` entry, so they contribute no row above. This is the mechanism that broke the silence instead.

| artefact | base | verdict | sub-metrics that broke silence |
|---|---|---|---|
| flip_elastic | mJog8DlRk_4 | WEAK | appearance, identity, clip_trajectory |

## Weight drift invisible to a per-sub-metric-only view

In these **15 cells**, at least one sub-metric's softmax weight moves by more than the drift threshold across the severity ladder, but the sub-metric is not one of the family's designed-for ones, so no finding in the table above flags it. Scoping the drift scan to designed-for sub-metrics only would have hidden this confound entirely.

| artefact | base | conforms | verdict | drifting-weight sub-metrics |
|---|---|---|---|---|
| background_drift | KZ8p6b1zJ9U | True | PASS | temporal |
| background_drift | hhszUXL1Cu8 | True | PASS | color_slope |
| background_drift | mJog8DlRk_4 | False | FLAT | temporal |
| chunk_boundary | BrRLKMbBTYQ | False | INVERTED | color_slope |
| flip_channel_shuffle | BrRLKMbBTYQ | True | PASS | color_slope |
| flip_channel_shuffle | KZ8p6b1zJ9U | True | PASS | color_slope |
| flip_channel_shuffle | hhszUXL1Cu8 | False | FLAT | color_slope |
| flip_channel_shuffle | mJog8DlRk_4 | True | PASS | color_slope |
| flip_invert | 7WHI2L_FDNg | True | PASS | temporal, identity, color_slope |
| flip_invert | BrRLKMbBTYQ | True | PASS | color_slope |
| flip_invert | KZ8p6b1zJ9U | True | PASS | identity, color_slope |
| flip_invert | hhszUXL1Cu8 | True | PASS | temporal |
| flip_invert | mJog8DlRk_4 | True | PASS | identity, color_slope |
| flip_periodic | 7WHI2L_FDNg | True | FLAT | temporal |
| flip_periodic | hhszUXL1Cu8 | True | FLAT | temporal |
