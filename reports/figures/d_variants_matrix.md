# Sub-metric D variants — three-matrix comparison

Δ = score(sev 0.40) − score(sev 0.02). PASS ≤ −0.05, WEAK ≤ −0.02, FLAT < +0.02 ≤ INV.

D' rescored with β=0.5, D'' rescored with β=3.0, both use `exp(-β · |q4-q1|)` over per-quarter trajectory means.


### D (original sub_metric color_stability)

| artefact | hhszUX | 7WHI2L | KZ8p6b | BrRLKM | mJog8D | clean |
|---|---:|---:|---:|---:|---:|---:|
| color_drift | +0.001 FLAT | +0.011 FLAT | +0.014 FLAT | -0.002 FLAT | +0.025 INV | 0/5 |
| chunk_boundary | -0.337 PASS | -0.217 PASS | -0.166 PASS | -0.208 PASS | -0.079 PASS | 5/5 |
| flicker | +0.005 FLAT | +0.012 FLAT | +0.015 FLAT | +0.008 FLAT | +0.017 FLAT | 0/5 |
| identity_degradation | +0.001 FLAT | -0.000 FLAT | -0.000 FLAT | +0.000 FLAT | +0.000 FLAT | 0/5 |
| identity_drift | -0.001 FLAT | +0.002 FLAT | +0.001 FLAT | -0.001 FLAT | +0.000 FLAT | 0/5 |
| background_drift | +0.046 INV | +0.034 INV | +0.045 INV | +0.021 INV | +0.066 INV | 0/5 |
| flip_horizontal | — | — | — | — | — | 0/5 |
| flip_transpose | — | — | — | — | — | 0/5 |
| flip_periodic | — | — | — | — | — | 0/5 |
| flip_elastic | — | — | — | — | — | 0/5 |
| flip_channel_shuffle | — | — | — | — | — | 0/5 |
| flip_invert | — | — | — | — | — | 0/5 |

**D (original sub_metric color_stability): 5/30 PASS/WEAK**

### D' anchor-window L1

| artefact | hhszUX | 7WHI2L | KZ8p6b | BrRLKM | mJog8D | clean |
|---|---:|---:|---:|---:|---:|---:|
| color_drift | -0.071 PASS | -0.305 PASS | -0.306 PASS | -0.011 FLAT | -0.311 PASS | 4/5 |
| chunk_boundary | +0.065 INV | +0.068 INV | -0.115 PASS | +0.481 INV | +0.034 INV | 1/5 |
| flicker | +0.028 INV | +0.057 INV | -0.061 PASS | +0.016 FLAT | +0.017 FLAT | 1/5 |
| identity_degradation | -0.001 FLAT | +0.000 FLAT | +0.003 FLAT | +0.000 FLAT | -0.000 FLAT | 0/5 |
| identity_drift | -0.002 FLAT | +0.010 FLAT | -0.011 FLAT | -0.000 FLAT | -0.004 FLAT | 0/5 |
| background_drift | -0.070 PASS | -0.088 PASS | -0.077 PASS | +0.071 INV | -0.265 PASS | 4/5 |
| flip_horizontal | -0.000 FLAT | +0.007 FLAT | +0.007 FLAT | +0.001 FLAT | -0.018 FLAT | 0/5 |
| flip_transpose | +0.000 FLAT | +0.012 FLAT | +0.002 FLAT | +0.001 FLAT | -0.014 FLAT | 0/5 |
| flip_periodic | +0.000 FLAT | +0.002 FLAT | +0.013 FLAT | +0.002 FLAT | -0.011 FLAT | 0/5 |
| flip_elastic | +0.001 FLAT | +0.003 FLAT | +0.003 FLAT | +0.000 FLAT | +0.004 FLAT | 0/5 |
| flip_channel_shuffle | -0.051 PASS | -0.060 PASS | -0.486 PASS | +0.084 INV | -0.205 PASS | 4/5 |
| flip_invert | -0.365 PASS | -0.070 PASS | -0.567 PASS | +0.300 INV | -0.165 PASS | 4/5 |

**D' anchor-window L1: 18/60 PASS/WEAK**

### D'' CLIP-trajectory

| artefact | hhszUX | 7WHI2L | KZ8p6b | BrRLKM | mJog8D | clean |
|---|---:|---:|---:|---:|---:|---:|
| color_drift | -0.002 FLAT | +0.020 INV | -0.046 WEAK | +0.015 FLAT | -0.075 PASS | 2/5 |
| chunk_boundary | -0.011 FLAT | +0.019 FLAT | -0.005 FLAT | +0.029 INV | +0.083 INV | 0/5 |
| flicker | +0.001 FLAT | -0.028 WEAK | -0.003 FLAT | +0.026 INV | +0.018 FLAT | 1/5 |
| identity_degradation | -0.045 WEAK | -0.010 FLAT | -0.014 FLAT | -0.002 FLAT | -0.014 FLAT | 1/5 |
| identity_drift | -0.036 WEAK | -0.001 FLAT | -0.042 WEAK | -0.004 FLAT | -0.008 FLAT | 2/5 |
| background_drift | -0.039 WEAK | -0.069 PASS | -0.214 PASS | -0.322 PASS | +0.085 INV | 4/5 |
| flip_horizontal | -0.047 WEAK | +0.079 INV | -0.004 FLAT | +0.040 INV | -0.008 FLAT | 1/5 |
| flip_transpose | -0.198 PASS | +0.138 INV | -0.123 PASS | +0.061 INV | -0.217 PASS | 3/5 |
| flip_periodic | -0.025 WEAK | +0.032 INV | +0.005 FLAT | +0.005 FLAT | +0.026 INV | 1/5 |
| flip_elastic | +0.020 FLAT | +0.031 INV | -0.013 FLAT | +0.027 INV | -0.029 WEAK | 1/5 |
| flip_channel_shuffle | -0.139 PASS | -0.080 PASS | -0.135 PASS | -0.017 FLAT | -0.052 PASS | 4/5 |
| flip_invert | -0.319 PASS | -0.194 PASS | -0.303 PASS | -0.010 FLAT | -0.099 PASS | 4/5 |

**D'' CLIP-trajectory: 24/60 PASS/WEAK**

### Best-of(D', D'')

| artefact | hhszUX | 7WHI2L | KZ8p6b | BrRLKM | mJog8D | clean |
|---|---:|---:|---:|---:|---:|---:|
| color_drift | -0.071 PASS | -0.305 PASS | -0.306 PASS | -0.011 FLAT | -0.311 PASS | 4/5 |
| chunk_boundary | +0.065 INV | +0.019 FLAT | -0.115 PASS | +0.481 INV | +0.034 INV | 1/5 |
| flicker | +0.028 INV | -0.028 WEAK | -0.061 PASS | +0.016 FLAT | +0.017 FLAT | 2/5 |
| identity_degradation | -0.001 FLAT | -0.010 FLAT | +0.003 FLAT | +0.000 FLAT | -0.000 FLAT | 0/5 |
| identity_drift | -0.002 FLAT | -0.001 FLAT | -0.011 FLAT | -0.000 FLAT | -0.004 FLAT | 0/5 |
| background_drift | -0.070 PASS | -0.069 PASS | -0.169 PASS | +0.071 INV | -0.254 PASS | 4/5 |
| flip_horizontal | -0.000 FLAT | +0.037 INV | +0.007 FLAT | +0.001 FLAT | -0.018 FLAT | 0/5 |
| flip_transpose | +0.000 FLAT | +0.044 INV | -0.109 PASS | +0.001 FLAT | -0.198 PASS | 2/5 |
| flip_periodic | +0.000 FLAT | +0.032 INV | +0.013 FLAT | +0.002 FLAT | -0.011 FLAT | 0/5 |
| flip_elastic | +0.001 FLAT | +0.031 INV | +0.003 FLAT | +0.000 FLAT | +0.004 FLAT | 0/5 |
| flip_channel_shuffle | -0.051 PASS | -0.080 PASS | -0.486 PASS | +0.084 INV | -0.205 PASS | 4/5 |
| flip_invert | -0.365 PASS | -0.194 PASS | -0.567 PASS | +0.300 INV | -0.165 PASS | 4/5 |

**Best-of(D', D''): 21/60 PASS/WEAK**
