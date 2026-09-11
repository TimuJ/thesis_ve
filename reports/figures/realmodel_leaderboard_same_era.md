# 4-method leaderboard — same-era identity refresh

The identity stage for MGLD and UAV was recomputed on the current server's
detector stack (fps-corrected, closeup-gated protocol unchanged), closing the
mixed-era comparability gap: FlashVSR/RealESRGAN identity was already
new-server. Composition: v5 production parameters.

| video | MGLD | UAV | FlashVSR | RealESRGAN | winner |
|---|---:|---:|---:|---:|:--:|
| 7WHI2L_FDNg | **0.738** | 0.702 | 0.737 | 0.736 | MGLD |
| BrRLKMbBTYQ | **0.402** | 0.379 | 0.393 | 0.401 | MGLD |
| KZ8p6b1zJ9U | **0.750** | 0.706 | 0.722 | 0.724 | MGLD |
| hhszUXL1Cu8 | **0.558** | 0.538 | 0.550 | 0.529 | MGLD |
| mJog8DlRk_4 | **0.656** | 0.620 | 0.649 | 0.631 | MGLD |
| **mean** | **0.6208** | 0.5892 | 0.6101 | 0.6045 | **MGLD** |

Published (mixed-era) means for comparison: 0.6219 / 0.5894 / 0.6101 / 0.6045.

**The ranking is era-robust:** the refresh moves method means by ≤ 0.0011,
order unchanged (MGLD > FlashVSR > RealESRGAN > UAV), MGLD > UAV on 5/5
videos. The detector-environment sensitivity documented for the artefact
identity baselines does not propagate to the leaderboard at method level —
every row an external model is compared against is now same-environment.

Provenance: identity/{mgld,uav}_refresh (new-server, fps overrides
7WHI 29.97 / BrRLK 24.0 / KZ 29.97 / hhsz 29.97 / mJog 23.98);
composition via sweep_sensitivity gated-canonical path at PROD.
