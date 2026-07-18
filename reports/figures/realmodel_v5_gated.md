# Real-model LR-VCC v5 — canonical (uniform closeup gating)

Protocol: production v5 flags; identity inputs fps-corrected (overrides files where available); closeup gate applied to ALL methods (non-mgld/uav methods use mgld's map — insensitive to the choice); dispersion gate off (parked). Recomposed from cached JSONs. Supersedes the mixed-gating July-2 table for thesis use; see lr_vcc_provenance_check.md for the audit.

## Composite (LR-VCC v5, gated)

| video | MGLD | UAV | FLASHVSR | winner |
|---|---|---|---|---|
| 7WHI2L_FDNg | 0.738 | 0.700 | 0.737 | mgld |
| BrRLKMbBTYQ | 0.402 | 0.379 | 0.393 | mgld |
| KZ8p6b1zJ9U | 0.750 | 0.705 | 0.722 | mgld |
| hhszUXL1Cu8 | 0.566 | 0.545 | 0.550 | mgld |
| mJog8DlRk_4 | 0.654 | 0.617 | 0.649 | mgld |
| **mean** | 0.622 | 0.589 | 0.610 | mgld |

## Sub-metric means

| sub-metric | MGLD | UAV | FLASHVSR |
|---|---|---|---|
| A (CLIP-IQA appearance) | 0.447 | 0.336 | 0.400 |
| T (tOF temporal) | 0.934 | 0.942 | 0.939 |
| I (slow-fast identity) | 0.557 | 0.459 | 0.598 |
| D (histogram stability) | 0.505 | 0.506 | 0.501 |
| E (colour slope) | 0.339 | 0.341 | 0.336 |
| D' (anchor histogram) | 0.708 | 0.687 | 0.705 |
| D'' (CLIP trajectory) | 0.893 | 0.884 | 0.862 |
