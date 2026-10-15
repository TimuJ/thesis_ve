# LVOS v2 — curated disappear-reappear subset (sizeable target + natural gap)

Criteria: an object with median mask-area **≥1%** of frame AND a natural interior
disappear-reappear **gap ≥2 s** (longest empty-mask run flanked by presence).
Measured from ground-truth masks over all 560 train+valid videos
(method: docs/notes/2026-10-16-lvos-mask-curation.md). **64 videos**, ranked by gap.
Target area / gap are for the single best qualifying object per video.

| split | video | dur (s) | target area % | longest gap (s) | #gaps≥1s | WxH |
|---|---|--:|--:|--:|--:|---|
| train | XfbRshfV | 69 | 10.9 | 34.7 | 1 | 1280x720 |
| valid | m37l7ti1 | 49 | 1.2 | 29.5 | 2 | 1280x720 |
| train | mSYwZwke | 46 | 15.1 | 27.8 | 1 | 1280x720 |
| train | MMJdp0ll | 52 | 6.4 | 23.2 | 1 | 1280x720 |
| train | hPDKXxOm | 39 | 9.4 | 20.2 | 2 | 1280x720 |
| valid | P7Ql6wAB | 50 | 1.2 | 18.3 | 5 | 1920x994 |
| train | hP6JJw8p | 155 | 7.0 | 15.0 | 14 | 1280x720 |
| train | M2cWIjUt | 30 | 16.7 | 14.5 | 2 | 1280x720 |
| valid | bl6VuRYE | 76 | 1.0 | 14.0 | 1 | 1280x720 |
| valid | gAHn8WFw | 81 | 2.0 | 13.8 | 9 | 1280x720 |
| train | mQVrNiUG | 39 | 33.2 | 13.7 | 3 | 1920x1080 |
| train | Nzhv8jOX | 80 | 3.5 | 13.2 | 1 | 1280x720 |
| valid | NWIoVXfO | 30 | 14.9 | 12.0 | 1 | 1280x720 |
| train | ZRKtsHlD | 39 | 1.9 | 11.7 | 1 | 1280x720 |
| valid | bxjTooAn | 50 | 15.6 | 11.0 | 2 | 1652x1080 |
| train | XH5TUk6R | 39 | 2.1 | 9.8 | 1 | 1280x720 |
| valid | cUD1dwuP | 132 | 1.1 | 9.8 | 7 | 1280x720 |
| train | lL7u6APJ | 57 | 1.0 | 9.7 | 1 | 1280x720 |
| valid | uXO4RUO3 | 357 | 2.0 | 9.7 | 15 | 1280x720 |
| valid | nfcT3owb | 187 | 3.3 | 9.2 | 8 | 1280x720 |
| valid | rSFfONgp | 39 | 11.1 | 9.0 | 1 | 1280x720 |
| train | J42kkH2a | 39 | 4.1 | 8.8 | 1 | 1280x720 |
| train | k1t2kMGE | 36 | 3.2 | 7.8 | 1 | 1280x720 |
| valid | TI0RU5dY | 39 | 28.7 | 7.5 | 1 | 1280x720 |
| valid | F52kIGgi | 39 | 2.0 | 6.8 | 2 | 1280x720 |
| train | bLfwnotU | 30 | 7.6 | 6.5 | 1 | 1280x720 |
| train | ZC5jwHA0 | 40 | 4.6 | 6.3 | 2 | 1280x720 |
| train | Dw4ecGPo | 38 | 23.2 | 6.2 | 1 | 1280x720 |
| train | B9gBQyqE | 46 | 13.9 | 6.0 | 1 | 1280x720 |
| valid | cjD5WPSv | 49 | 4.6 | 6.0 | 2 | 1280x720 |
| valid | VjokbOOs | 34 | 5.7 | 5.7 | 2 | 1280x720 |
| valid | iWjeKM6a | 39 | 2.6 | 5.2 | 1 | 1280x720 |
| train | AYYEfEMq | 48 | 14.1 | 5.0 | 1 | 1280x720 |
| train | mEe2jLfp | 139 | 3.2 | 5.0 | 1 | 1920x1440 |
| valid | o6HCvWAH | 270 | 7.7 | 4.8 | 1 | 1280x720 |
| valid | PdatOPeT | 39 | 6.0 | 4.7 | 3 | 1280x720 |
| train | KlhduiNN | 68 | 4.5 | 4.5 | 2 | 1280x720 |
| valid | mSjSh9tj | 39 | 4.3 | 4.3 | 1 | 1280x720 |
| train | wGYCoEz3 | 84 | 1.4 | 4.2 | 4 | 1280x720 |
| valid | XWN7ehmd | 39 | 6.7 | 4.2 | 1 | 1280x720 |
| valid | vr4ZIAbp | 39 | 5.0 | 4.2 | 1 | 1280x720 |
| train | UrKAenln | 277 | 2.0 | 4.0 | 2 | 1280x720 |
| train | PEyHFztC | 33 | 3.5 | 3.8 | 7 | 1280x720 |
| train | zRt6qZHQ | 39 | 25.8 | 3.7 | 2 | 1920x1080 |
| valid | UgZ759gD | 94 | 1.1 | 3.7 | 3 | 1280x720 |
| valid | WESvhF1e | 35 | 5.4 | 3.3 | 1 | 1920x1080 |
| train | XYMyIlvq | 50 | 1.9 | 3.2 | 1 | 1280x720 |
| train | pyjnNgiV | 39 | 7.2 | 3.2 | 1 | 1920x1080 |
| valid | usxBn0LJ | 39 | 1.6 | 3.0 | 1 | 1280x720 |
| train | tHqpGbsR | 39 | 4.1 | 2.8 | 4 | 1280x720 |
| train | l3jxnqrp | 33 | 10.8 | 2.7 | 1 | 1280x720 |
| train | y2Ai61Xm | 33 | 4.9 | 2.7 | 1 | 1280x720 |
| valid | 49TNsJzk | 97 | 2.9 | 2.7 | 1 | 1280x720 |
| valid | v3uNUctx | 138 | 2.0 | 2.5 | 2 | 1280x720 |
| train | MNLxhh62 | 39 | 2.3 | 2.3 | 2 | 1280x720 |
| train | 2pMiy6eH | 70 | 1.4 | 2.2 | 1 | 1280x720 |
| train | bhNU9gxv | 39 | 26.4 | 2.2 | 1 | 1280x720 |
| train | fQVilF4a | 52 | 6.5 | 2.2 | 1 | 1280x720 |
| train | qFfJt7WD | 39 | 10.6 | 2.2 | 1 | 1280x720 |
| train | zvLCVOKl | 39 | 3.5 | 2.2 | 1 | 1280x720 |
| valid | N6CONZUW | 181 | 2.1 | 2.2 | 4 | 1280x720 |
| valid | 1DNYxl0a | 39 | 3.1 | 2.0 | 2 | 1280x720 |
| valid | f4DjwV55 | 51 | 2.4 | 2.0 | 1 | 1280x720 |
| valid | pyq6I5Hh | 33 | 3.4 | 2.0 | 1 | 1280x720 |
