# LR-VCC v5 — expectation-scored verdict matrix

Each cell is scored against the expectation pre-registered for its family, not against a uniform PASS+WEAK rule. A control family predicted invisible conforms by being FLAT.

| artefact | expectation | 7WHI2L_FDNg | BrRLKMbBTYQ | KZ8p6b1zJ9U | hhszUXL1Cu8 | mJog8DlRk_4 |
|---|---|---|---|---|---|---|
| background_drift | RESPOND | +0.002 FLAT ✗ | +0.054 INVERTED ✗ | -0.072 PASS ✓ | -0.201 PASS ✓ | +0.002 FLAT ✗ |
| chunk_boundary | RESPOND | -0.074 PASS ✓ | +0.105 INVERTED ✗ | -0.098 PASS ✓ | -0.109 PASS ✓ | -0.042 WEAK ✓ |
| color_drift | RESPOND | -0.108 PASS ✓ | -0.055 PASS ✓ | -0.150 PASS ✓ | -0.048 WEAK ✓ | -0.111 PASS ✓ |
| flicker | RESPOND | -0.001 FLAT ✗ | +0.059 INVERTED ✗ | -0.030 WEAK ✓ | +0.010 FLAT ✗ | +0.001 FLAT ✗ |
| flip_channel_shuffle | RESPOND | -0.025 WEAK ✓ | -0.080 PASS ✓ | -0.305 PASS ✓ | +0.003 FLAT ✗ | -0.058 PASS ✓ |
| flip_elastic | SILENT | -0.002 FLAT ✓ | +0.000 FLAT ✓ | -0.002 FLAT ✓ | +0.007 FLAT ✓ | -0.022 WEAK ✗ |
| flip_horizontal | SILENT | +0.016 FLAT ✓ | +0.005 FLAT ✓ | -0.000 FLAT ✓ | -0.009 FLAT ✓ | -0.017 FLAT ✓ |
| flip_invert | RESPOND | -0.147 PASS ✓ | -0.224 PASS ✓ | -0.348 PASS ✓ | -0.391 PASS ✓ | -0.069 PASS ✓ |
| flip_periodic | SILENT | -0.009 FLAT ✓ | -0.015 FLAT ✓ | +0.003 FLAT ✓ | -0.018 FLAT ✓ | -0.007 FLAT ✓ |
| flip_transpose | UNCONSTRAINED | +0.012 FLAT — | +0.000 FLAT — | -0.052 PASS — | -0.055 PASS — | -0.031 WEAK — |
| identity_degradation | RESPOND | +0.029 INVERTED ✗ | +0.000 FLAT ✗ | -0.030 WEAK ✓ | -0.053 PASS ✓ | +0.013 FLAT ✗ |
| identity_drift | RESPOND | -0.001 FLAT ✗ | -0.000 FLAT ✗ | -0.025 WEAK ✓ | -0.069 PASS ✓ | -0.003 FLAT ✗ |

- as-designed (expectation-aware): **39/55** (RESPOND 25/40, SILENT 14/15; 5 unconstrained cells excluded)
- clean under the old uniform PASS+WEAK rule: **29/60**

The metric is unchanged between the two counts; only the scoring criterion differs.
