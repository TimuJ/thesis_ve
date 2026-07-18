# LR-VCC v5 — provenance check (recomposition, current code)

## Synthetic matrix: published composites vs uniform recompose

The published 12/12 matrix mixes composition eras: six families composed with the (since parked) identity dispersion gate ON, six with it OFF. Uniform current-code recompose (gate off everywhere):

- clean (PASS+WEAK): published 28/60 -> uniform 29/60
- cells changing verdict class: 6
  - background_drift/mJog8DlRk_4: WEAK->FLAT
  - flicker/KZ8p6b1zJ9U: FLAT->WEAK
  - identity_degradation/7WHI2L_FDNg: FLAT->INVERTED
  - identity_degradation/hhszUXL1Cu8: WEAK->PASS
  - identity_drift/KZ8p6b1zJ9U: FLAT->WEAK
  - identity_drift/hhszUXL1Cu8: WEAK->PASS

## Real-model table under input variants

| variant | MGLD | UAV | FlashVSR | order | MGLD>UAV/video |
|---|---|---|---|---|---|
| published (replica) | 0.5889 | 0.5523 | 0.6101 | flashvsr>mgld>uav | 5/5 |
| corrected identity (fps overrides) | 0.5888 | 0.5525 | 0.6101 | flashvsr>mgld>uav | 5/5 |
| corrected identity + closeup gate on all | 0.6219 | 0.5894 | 0.6101 | mgld>flashvsr>uav | 5/5 |
