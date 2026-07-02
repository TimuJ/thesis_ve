# Real-model LR-VCC v5 ranking — three methods (2026-07-02)

FlashVSR v1.1 (tiny-long, stock weights) added as the third method row.
Same 5 LR sources, same 1280×720 full-content outputs, same 7-stage battery
and production composite flags (`--temporal_weight uniform
--color_hist_alpha 0.394 --color_slope_beta 200`, v5 dirs, default betas).

FlashVSR run details: pristine repo @ `b527c6f`; the four 5000-frame videos
processed as 2 segments each (stock RoPE table caps a single streaming pass at
4089 frames), seam at mid-video; hhsz single-pass. See
`results/flashvsr_synthetic_mp4/manifest.json` and
`docs/notes/2026-07-02-flashvsr-rope-site.md`.

## Composite (LR-VCC v5)

| video | MGLD | UAV | FlashVSR | winner |
|-------|-----:|----:|---------:|:------:|
| 7WHI2L_FDNg | 0.652 | 0.609 | **0.737** | FlashVSR |
| BrRLKMbBTYQ | **0.407** | 0.380 | 0.393 | MGLD |
| KZ8p6b1zJ9U | **0.732** | 0.712 | 0.722 | MGLD |
| hhszUXL1Cu8 | **0.579** | 0.528 | 0.550 | MGLD |
| mJog8DlRk_4 | 0.575 | 0.533 | **0.649** | FlashVSR |
| **mean** | 0.589 | 0.552 | **0.610** | **FlashVSR** |

FlashVSR wins the mean and 2/5 videos outright (by +0.085 and +0.074); MGLD
keeps 3/5 by small margins (+0.014, +0.010, +0.029).

Closeup-map sensitivity: FlashVSR composed twice, once with mgld's closeup
map and once with uav's (the maps differ up to 44 % relative on 2 videos) —
composite differs by ≤ 0.001 everywhere (0.6101 vs 0.6104 mean). The gate is
insensitive to the map choice on these videos; no FlashVSR-specific anatomy
trace needed.

## Sub-metric means

| sub-metric | MGLD | UAV | FlashVSR |
|------------|-----:|----:|---------:|
| A (CLIP-IQA appearance) | **0.447** | 0.336 | 0.400 |
| T (tOF temporal) | 0.934 | **0.942** | 0.939 |
| I (slow-fast identity) | 0.555 | 0.463 | **0.598** |
| D (histogram stability) | 0.505 | **0.506** | 0.501 |
| E (colour slope) | 0.339 | **0.341** | 0.336 |
| D' (anchor histogram) | **0.708** | 0.687 | 0.705 |
| D'' (CLIP trajectory) | **0.893** | 0.884 | 0.862 |

## Reading

- **FlashVSR's win is driven by identity** (I: +0.043 over MGLD, +0.135 over
  UAV) with competitive appearance; T/D/E near-tied across all three (as with
  the MGLD-UAV pair, these don't separate well-behaved methods).
- **FlashVSR is worst on D'' (CLIP trajectory)** — the sub-metric most
  sensitive to long-range semantic drift from the anchor. Consistent with a
  streaming model whose temporal positions grow monotonically (and with one
  mid-video seam on 4/5 videos). This is exactly the regime the RoPE
  extrapolation probe (Phases 1–2) tests causally — if position overrides
  move D''-like drift, the benchmark's weakest cell for FlashVSR gets a
  mechanistic explanation.
- Interestingly the two videos FlashVSR wins big (7WHI, mJog) are both
  segmented — the seam did not sink the composite; the narrow losses include
  the single-pass video (hhsz), so segmentation alone doesn't explain the
  per-video pattern.

## Provenance

- Stage JSONs: `clip_iqa`, `tof_tlp`, identity — computed on the SmartML
  server (identity: `results_2026-07-02-18:41:52_eval_results.json`, overall
  fused 0.5983); colour stages + D'/D'' — computed locally on the mp4
  previews, same as the MGLD/UAV realmodels run.
- Composites: `results/lr_vcc/composite_v5_realmodels/flashvsr_map{mgld,uav}/`.
- Known caveat: one CLIP-IQA server failure (setuptools-81 `pkg_resources`,
  now also seen in the `vsr` env) was fixed and rerun before scoring.
