# LR-VCC v5 — hyperparameter sensitivity sweep

Production: dprime_beta=0.5, dprime2_beta=3.0, tau=0.2, alpha=0.394, slope_beta=200. 52 configs recomposed from cached sub-metric JSONs (no video re-scanning).

| b_D' | b_D'' | tau | alpha | b_E | 3-method order | MGLD>UAV per-video | matrix clean | cells changed vs prod |
|---|---|---|---|---|---|---|---|---|
| 0.25 | 1.0 | 0.1 | 0.394 | 200 | flashvsr>mgld>uav | 4/5 | 25/60 | 13 |
| 0.25 | 1.0 | 0.2 | 0.394 | 200 | flashvsr>mgld>uav | 5/5 | 26/60 | 9 |
| 0.25 | 1.0 | 0.5 | 0.394 | 200 | flashvsr>mgld>uav | 5/5 | 25/60 | 10 |
| 0.25 | 2.0 | 0.1 | 0.394 | 200 | flashvsr>mgld>uav | 4/5 | 27/60 | 11 |
| 0.25 | 2.0 | 0.2 | 0.394 | 200 | flashvsr>mgld>uav | 5/5 | 27/60 | 6 |
| 0.25 | 2.0 | 0.5 | 0.394 | 200 | flashvsr>mgld>uav | 5/5 | 27/60 | 8 |
| 0.25 | 3.0 | 0.1 | 0.394 | 200 | flashvsr>mgld>uav | 5/5 | 30/60 | 6 |
| 0.25 | 3.0 | 0.2 | 0.394 | 200 | flashvsr>mgld>uav | 5/5 | 29/60 | 2 |
| 0.25 | 3.0 | 0.5 | 0.394 | 200 | flashvsr>mgld>uav | 5/5 | 28/60 | 6 |
| 0.25 | 5.0 | 0.1 | 0.394 | 200 | flashvsr>mgld>uav | 5/5 | 30/60 | 7 |
| 0.25 | 5.0 | 0.2 | 0.394 | 200 | flashvsr>mgld>uav | 5/5 | 30/60 | 7 |
| 0.25 | 5.0 | 0.5 | 0.394 | 200 | flashvsr>mgld>uav | 5/5 | 30/60 | 6 |
| 0.5 | 1.0 | 0.1 | 0.394 | 200 | flashvsr>mgld>uav | 5/5 | 27/60 | 9 |
| 0.5 | 1.0 | 0.2 | 0.394 | 200 | flashvsr>mgld>uav | 5/5 | 26/60 | 8 |
| 0.5 | 1.0 | 0.5 | 0.394 | 200 | flashvsr>mgld>uav | 5/5 | 26/60 | 13 |
| 0.5 | 2.0 | 0.1 | 0.394 | 200 | flashvsr>mgld>uav | 5/5 | 28/60 | 6 |
| 0.5 | 2.0 | 0.2 | 0.394 | 200 | flashvsr>mgld>uav | 5/5 | 29/60 | 2 |
| 0.5 | 2.0 | 0.5 | 0.394 | 200 | flashvsr>mgld>uav | 5/5 | 27/60 | 11 |
| 0.5 | 3.0 | 0.1 | 0.394 | 200 | flashvsr>mgld>uav | 5/5 | 30/60 | 4 |
| 0.5 | 3.0 | 0.2 | 0.394 | 200 | flashvsr>mgld>uav | 5/5 | 29/60 | 0 |
| 0.5 | 3.0 | 0.5 | 0.394 | 200 | flashvsr>mgld>uav | 5/5 | 29/60 | 6 |
| 0.5 | 5.0 | 0.1 | 0.394 | 200 | flashvsr>mgld>uav | 5/5 | 30/60 | 5 |
| 0.5 | 5.0 | 0.2 | 0.394 | 200 | flashvsr>mgld>uav | 5/5 | 30/60 | 4 |
| 0.5 | 5.0 | 0.5 | 0.394 | 200 | mgld>flashvsr>uav | 5/5 | 29/60 | 7 |
| 1.0 | 1.0 | 0.1 | 0.394 | 200 | flashvsr>mgld>uav | 5/5 | 29/60 | 11 |
| 1.0 | 1.0 | 0.2 | 0.394 | 200 | flashvsr>mgld>uav | 5/5 | 28/60 | 9 |
| 1.0 | 1.0 | 0.5 | 0.394 | 200 | flashvsr>mgld>uav | 5/5 | 26/60 | 10 |
| 1.0 | 2.0 | 0.1 | 0.394 | 200 | flashvsr>mgld>uav | 5/5 | 30/60 | 8 |
| 1.0 | 2.0 | 0.2 | 0.394 | 200 | flashvsr>mgld>uav | 5/5 | 29/60 | 8 |
| 1.0 | 2.0 | 0.5 | 0.394 | 200 | flashvsr>mgld>uav | 5/5 | 29/60 | 9 |
| 1.0 | 3.0 | 0.1 | 0.394 | 200 | flashvsr>mgld>uav | 5/5 | 31/60 | 7 |
| 1.0 | 3.0 | 0.2 | 0.394 | 200 | flashvsr>mgld>uav | 5/5 | 31/60 | 7 |
| 1.0 | 3.0 | 0.5 | 0.394 | 200 | mgld>flashvsr>uav | 5/5 | 29/60 | 8 |
| 1.0 | 5.0 | 0.1 | 0.394 | 200 | flashvsr>mgld>uav | 5/5 | 32/60 | 6 |
| 1.0 | 5.0 | 0.2 | 0.394 | 200 | flashvsr>mgld>uav | 5/5 | 32/60 | 9 |
| 1.0 | 5.0 | 0.5 | 0.394 | 200 | mgld>flashvsr>uav | 5/5 | 30/60 | 7 |
| 2.0 | 1.0 | 0.1 | 0.394 | 200 | flashvsr>mgld>uav | 5/5 | 30/60 | 20 |
| 2.0 | 1.0 | 0.2 | 0.394 | 200 | flashvsr>mgld>uav | 5/5 | 31/60 | 18 |
| 2.0 | 1.0 | 0.5 | 0.394 | 200 | mgld>flashvsr>uav | 5/5 | 28/60 | 17 |
| 2.0 | 2.0 | 0.1 | 0.394 | 200 | flashvsr>mgld>uav | 5/5 | 31/60 | 20 |
| 2.0 | 2.0 | 0.2 | 0.394 | 200 | flashvsr>mgld>uav | 5/5 | 32/60 | 17 |
| 2.0 | 2.0 | 0.5 | 0.394 | 200 | mgld>flashvsr>uav | 5/5 | 29/60 | 15 |
| 2.0 | 3.0 | 0.1 | 0.394 | 200 | flashvsr>mgld>uav | 5/5 | 31/60 | 21 |
| 2.0 | 3.0 | 0.2 | 0.394 | 200 | flashvsr>mgld>uav | 5/5 | 32/60 | 16 |
| 2.0 | 3.0 | 0.5 | 0.394 | 200 | mgld>flashvsr>uav | 5/5 | 28/60 | 15 |
| 2.0 | 5.0 | 0.1 | 0.394 | 200 | flashvsr>mgld>uav | 5/5 | 31/60 | 21 |
| 2.0 | 5.0 | 0.2 | 0.394 | 200 | flashvsr>mgld>uav | 5/5 | 32/60 | 17 |
| 2.0 | 5.0 | 0.5 | 0.394 | 200 | mgld>flashvsr>uav | 5/5 | 28/60 | 17 |
| 0.5 | 3.0 | 0.2 | 0.2 | 200 | flashvsr>mgld>uav | 5/5 | 29/60 | 2 |
| 0.5 | 3.0 | 0.2 | 0.8 | 200 | flashvsr>mgld>uav | 5/5 | 26/60 | 11 |
| 0.5 | 3.0 | 0.2 | 0.394 | 100 | flashvsr>mgld>uav | 5/5 | 29/60 | 6 |
| 0.5 | 3.0 | 0.2 | 0.394 | 300 | flashvsr>mgld>uav | 5/5 | 29/60 | 3 |

Headline stability: 3-method mean order flashvsr>mgld>uav holds in 45/52 configs; MGLD>UAV on every video holds in 50/52 configs.
