# LR-VCC v5 — leave-one-out sub-metric ablation

Each row recomposes the production configuration with one sub-metric removed (cached JSONs, no re-scanning).

| dropped | 3-method order | MGLD>UAV per-video | matrix clean | cells changed vs prod | changed cells |
|---|---|---|---|---|---|
| (none) | flashvsr>mgld>uav | 5/5 | 29/60 | 0 | — |
| appearance | flashvsr>mgld>uav | 4/5 | 29/60 | 5 | color_drift/hhszUXL1Cu8: WEAK->PASS; flip_periodic/hhszUXL1Cu8: FLAT->WEAK; flip_transpose/7WHI2L_FDNg: FLAT->INVERTED; flip_transpose/KZ8p6b1zJ9U: PASS->WEAK; identity_degradation/KZ8p6b1zJ9U: WEAK->FLAT |
| temporal | flashvsr>mgld>uav | 5/5 | 29/60 | 0 | — |
| identity | mgld>flashvsr>uav | 5/5 | 26/60 | 13 | background_drift/KZ8p6b1zJ9U: PASS->WEAK; background_drift/mJog8DlRk_4: FLAT->WEAK; flicker/KZ8p6b1zJ9U: WEAK->FLAT; flip_elastic/mJog8DlRk_4: WEAK->FLAT; flip_horizontal/7WHI2L_FDNg: FLAT->INVERTED; flip_periodic/hhszUXL1Cu8: FLAT->WEAK (+7 more) |
| color_stability | flashvsr>mgld>uav | 5/5 | 31/60 | 10 | background_drift/mJog8DlRk_4: FLAT->WEAK; chunk_boundary/7WHI2L_FDNg: PASS->FLAT; chunk_boundary/KZ8p6b1zJ9U: PASS->WEAK; chunk_boundary/mJog8DlRk_4: WEAK->FLAT; color_drift/hhszUXL1Cu8: WEAK->PASS; flicker/KZ8p6b1zJ9U: WEAK->PASS (+4 more) |
| color_slope | flashvsr>mgld>uav | 5/5 | 26/60 | 10 | background_drift/BrRLKMbBTYQ: INVERTED->WEAK; background_drift/hhszUXL1Cu8: PASS->FLAT; color_drift/BrRLKMbBTYQ: PASS->FLAT; flicker/BrRLKMbBTYQ: INVERTED->FLAT; flip_channel_shuffle/BrRLKMbBTYQ: PASS->INVERTED; flip_channel_shuffle/hhszUXL1Cu8: FLAT->WEAK (+4 more) |
| color_hist_anchor | flashvsr>mgld>uav | 5/5 | 30/60 | 10 | background_drift/7WHI2L_FDNg: FLAT->INVERTED; background_drift/mJog8DlRk_4: FLAT->INVERTED; chunk_boundary/BrRLKMbBTYQ: INVERTED->FLAT; chunk_boundary/mJog8DlRk_4: WEAK->PASS; flip_channel_shuffle/7WHI2L_FDNg: WEAK->FLAT; flip_channel_shuffle/hhszUXL1Cu8: FLAT->INVERTED (+4 more) |
| clip_trajectory | flashvsr>mgld>uav | 5/5 | 27/60 | 13 | background_drift/KZ8p6b1zJ9U: PASS->WEAK; chunk_boundary/mJog8DlRk_4: WEAK->PASS; color_drift/BrRLKMbBTYQ: PASS->WEAK; color_drift/hhszUXL1Cu8: WEAK->PASS; flip_channel_shuffle/7WHI2L_FDNg: WEAK->FLAT; flip_channel_shuffle/hhszUXL1Cu8: FLAT->INVERTED (+7 more) |
