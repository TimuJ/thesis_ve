# Real-Model LR-VCC v5 Ranking — MGLD-VSR vs Upscale-A-Video

Computed 2026-07-01 on the new SmartML server's SR outputs (metrics run locally;
D' and D'' newly computed, other sub-metrics + identity reused from the May Task-8 run).

## Per-video LR-VCC (higher = more consistent)

| video | MGLD v3 | UAV v3 | MGLD v5 | UAV v5 | v5 winner |
|---|---:|---:|---:|---:|:--:|
| hhszUXL1Cu8 | 0.558 | 0.476 | **0.579** | 0.528 | MGLD |
| 7WHI2L_FDNg | 0.529 | 0.458 | **0.652** | 0.609 | MGLD |
| KZ8p6b1zJ9U | 0.571 | 0.528 | **0.732** | 0.712 | MGLD |
| BrRLKMbBTYQ | 0.316 | 0.269 | **0.407** | 0.380 | MGLD |
| mJog8DlRk_4 | 0.409 | 0.374 | **0.575** | 0.533 | MGLD |
| **mean** | 0.476 | 0.421 | **0.589** | 0.552 | MGLD |

**MGLD-UAV gap:** v3 = +0.0555, v5 = +0.0366

MGLD-VSR wins on **all 5 videos** under v5 LR-VCC — a unanimous ranking consistent
with MGLD being the stronger method (it won 8/9 metrics in the April no-reference eval).

## v5 sub-metric means

| sub-metric | MGLD | UAV | MGLD−UAV |
|---|---:|---:|---:|
| A (CLIP-IQA) | 0.447 | 0.336 | +0.111 |
| T (tOF) | 0.934 | 0.942 | -0.008 |
| I (slow-fast) | 0.555 | 0.463 | +0.092 |
| D (hist stability) | 0.505 | 0.506 | -0.002 |
| E (slope) | 0.339 | 0.341 | -0.001 |
| D' (anchor hist) | 0.708 | 0.687 | +0.021 |
| D'' (CLIP traj) | 0.893 | 0.884 | +0.009 |

The two new sub-metrics both favour MGLD (D' +0.021, D'' +0.009), reinforcing the
appearance (+0.111) and identity (+0.092) signals. Temporal/color are near-tied.
